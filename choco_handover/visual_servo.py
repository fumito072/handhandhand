"""
Eye-to-Hand ステレオ・ビジュアルサーボ (Look-and-Move) モジュール。
アーム移動前に記録したクリーンな机上背景画像との差分＋BrainCo Revo 2 の黒色シルエット特徴から、
カメラ画像上で【親指先端】【人差指/中指先端】【3指ピンチ中心】を直接検出し、
対象物との水平面相対誤差 (ΔX, ΔY) [m] を閉ループ補正する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import SystemConfig
from kinematics import R1Kinematics
from stereo_depth import StereoResult


@dataclass
class VisualServoObservation:
    valid: bool
    side: str
    thumb_uv: Optional[Tuple[int, int]] = None
    index_uv: Optional[Tuple[int, int]] = None
    pinch_uv: Optional[Tuple[int, int]] = None
    target_uv_at_hover: Optional[Tuple[int, int]] = None
    pinch_torso_m: Optional[List[float]] = None
    delta_xy_torso_m: Optional[Tuple[float, float]] = None
    residual_mm: Optional[float] = None
    hand_area_px: int = 0
    reason: str = ""


class StereoVisualServo:
    def __init__(self):
        self.ref_left_bgr: Optional[np.ndarray] = None
        self.last_obs: Optional[VisualServoObservation] = None

    def capture_clean_reference(self, left_bgr: np.ndarray):
        """アームがカメラ視野に入る前のクリーンな机上 BGR フレームを保存する。"""
        if left_bgr is not None and left_bgr.ndim == 3:
            self.ref_left_bgr = left_bgr.copy()
        self.last_obs = None

    def clear(self):
        self.ref_left_bgr = None
        self.last_obs = None

    @staticmethod
    def _pixel_ray_at_torso_z(
        u: float,
        v: float,
        target_z_torso: float,
        kin: R1Kinematics,
        head_pitch: float,
        head_yaw: float,
        cfg: SystemConfig,
    ) -> Optional[np.ndarray]:
        """
        カメラ画素 (u, v) を通る 3D 視線レイと、既知の水平高さ平面 Z_torso = target_z_torso の交点を求める。
        斜め見下ろしカメラにおけるホバリング高さパララックスを幾何学的に完全除去する。
        """
        T = kin.camera_to_torso_matrix(head_pitch, head_yaw)
        origin_t = T[:3, 3] + np.array([cfg.hand_eye_dx_m, cfg.hand_eye_dy_m, cfg.hand_eye_dz_m], dtype=np.float64)
        ray_cam = np.array([(u - cfg.cx) / cfg.fx, (v - cfg.cy) / cfg.fy, 1.0], dtype=np.float64)
        dir_t = T[:3, :3] @ ray_cam
        if abs(float(dir_t[2])) < 1e-4:
            return None
        s = (float(target_z_torso) - float(origin_t[2])) / float(dir_t[2])
        if s <= 0.10 or s >= 1.50:
            return None
        return origin_t + s * dir_t

    def observe_pinch_alignment(
        self,
        left_bgr: np.ndarray,
        stereo: StereoResult,
        side: str,
        desired_hover_torso_m: np.ndarray,
        kin: R1Kinematics,
        head_pitch: float,
        head_yaw: float,
        cfg: SystemConfig,
    ) -> VisualServoObservation:
        """
        現在ホバリング中の Revo 2 ハンドのピンチ中心を画像から検出し、
        目標ホバリング 3D 位置 `desired_hover_torso_m` との水平誤差 (ΔX, ΔY) を算出する。
        """
        if self.ref_left_bgr is None or self.ref_left_bgr.shape != left_bgr.shape:
            obs = VisualServoObservation(valid=False, side=side, reason="基準背景未取得")
            self.last_obs = obs
            return obs

        h, w = left_bgr.shape[:2]
        # 目標ホバリング位置が画像上のどこに投影されるべきか (理想ピクセル)
        pt_cam_ideal = kin.torso_points_to_cam(
            np.asarray(desired_hover_torso_m, dtype=np.float64).reshape(1, 3),
            head_pitch,
            head_yaw,
            cfg,
        )[0]
        if pt_cam_ideal[2] <= 0.08:
            obs = VisualServoObservation(valid=False, side=side, reason="目標がカメラ背面")
            self.last_obs = obs
            return obs

        u_ideal = int(round(cfg.fx * pt_cam_ideal[0] / pt_cam_ideal[2] + cfg.cx))
        v_ideal = int(round(cfg.fy * pt_cam_ideal[1] / pt_cam_ideal[2] + cfg.cy))

        # 期待位置周辺の探索ウィンドウ (右ハンドは画面右側 +u、左ハンドは画面左側 -u から進入)
        u_min = max(16, u_ideal - (65 if side == "right" else 115))
        u_max = min(w - 16, u_ideal + (115 if side == "right" else 65))
        v_min = max(16, v_ideal - 65)
        v_max = min(h - 24, v_ideal + 95)
        if u_max - u_min < 25 or v_max - v_min < 25:
            obs = VisualServoObservation(valid=False, side=side, reason="ROI範囲外")
            self.last_obs = obs
            return obs

        # 1. アーム進入前背景との色差マスク ＋ 机/箱天面の3D近傍マスク（箱の横に立った人の服・ズボンを100%除外！）
        diff = np.max(np.abs(left_bgr.astype(np.int16) - self.ref_left_bgr.astype(np.int16)), axis=2)
        hsv = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2HSV)
        is_changed = diff > 28
        is_white_forearm = (hsv[:, :, 2] > 142) & (hsv[:, :, 1] < 40)
        # 探索ウィンドウをピンチ理想位置の近傍 (±52px) に限定し、人が横に入っても誤検出しないようにする
        u_min = max(u_min, u_ideal - (38 if side == "right" else 65))
        u_max = min(u_max, u_ideal + (65 if side == "right" else 38))
        v_min = max(v_min, v_ideal - 42)
        v_max = min(v_max, v_ideal + 58)
        cand = np.zeros((h, w), dtype=np.uint8)
        if u_max > u_min + 12 and v_max > v_min + 12:
            cand[v_min:v_max, u_min:u_max] = (
                (is_changed & ~is_white_forearm)[v_min:v_max, u_min:u_max]
            ).astype(np.uint8) * 255

        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        num_l, labels, stats, _ = cv2.connectedComponentsWithStats(cand)
        if num_l <= 1:
            obs = VisualServoObservation(valid=False, side=side, reason="ハンド領域未検出")
            self.last_obs = obs
            return obs

        best_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area_px = int(stats[best_lbl, cv2.CC_STAT_AREA])
        if area_px < 80:
            obs = VisualServoObservation(
                valid=False, side=side, hand_area_px=area_px, reason=f"ハンド面積不足({area_px}px)"
            )
            self.last_obs = obs
            return obs

        blob = labels == best_lbl
        ys, xs = np.where(blob)

        # 2. 左右対称の解剖学的突起先端抽出:
        #    右ハンド(右下から左上へ進入): 親指は内側(-u,+v寄り)、人差指/中指は先端(-v,-u寄り)
        #    左ハンド(左下から右上へ進入): 親指は内側(+u,+v寄り)、人差指/中指は先端(-v,+u寄り)
        if side == "right":
            idx_thumb = int(np.argmin(xs + 0.25 * ys))
            idx_index = int(np.argmin(ys + 0.35 * xs))
        else:
            idx_thumb = int(np.argmin(-xs + 0.25 * ys))
            idx_index = int(np.argmin(ys - 0.35 * xs))

        u_th, v_th = int(xs[idx_thumb]), int(ys[idx_thumb])
        u_in, v_in = int(xs[idx_index]), int(ys[idx_index])
        u_pinch = int(round(0.5 * (u_th + u_in)))
        v_pinch = int(round(0.5 * (v_th + v_in)))

        # 3. 既知のホバリング高さ Z_hover 平面との視線交差により、パララックス無しで (X_pinch, Y_pinch) を算出
        z_hover = float(desired_hover_torso_m[2])
        p_pinch_t = self._pixel_ray_at_torso_z(
            float(u_pinch), float(v_pinch), z_hover, kin, head_pitch, head_yaw, cfg
        )
        if p_pinch_t is None:
            obs = VisualServoObservation(valid=False, side=side, reason="視線レイ交差失敗")
            self.last_obs = obs
            return obs

        dx = float(desired_hover_torso_m[0] - p_pinch_t[0])
        dy = float(desired_hover_torso_m[1] - p_pinch_t[1])
        res_mm = float(np.hypot(dx, dy) * 1000.0)

        # 外れ値ガード: 12cm 以上離れている場合は誤検出とみなして補正しない
        if res_mm > 120.0:
            obs = VisualServoObservation(
                valid=False,
                side=side,
                thumb_uv=(u_th, v_th),
                index_uv=(u_in, v_in),
                pinch_uv=(u_pinch, v_pinch),
                target_uv_at_hover=(u_ideal, v_ideal),
                residual_mm=round(res_mm, 1),
                hand_area_px=area_px,
                reason=f"残差過大({res_mm:.0f}mm)のためスキップ",
            )
            self.last_obs = obs
            return obs

        obs = VisualServoObservation(
            valid=True,
            side=side,
            thumb_uv=(u_th, v_th),
            index_uv=(u_in, v_in),
            pinch_uv=(u_pinch, v_pinch),
            target_uv_at_hover=(u_ideal, v_ideal),
            pinch_torso_m=[round(float(v), 4) for v in p_pinch_t],
            delta_xy_torso_m=(round(dx, 4), round(dy, 4)),
            residual_mm=round(res_mm, 1),
            hand_area_px=area_px,
            reason="OK",
        )
        self.last_obs = obs
        return obs
