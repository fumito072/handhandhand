"""
MediaPipe HandLandmarker (21ランドマーク × 両手) による
「ください」両手カップポーズ認識 ＆ 両手のひら中央のステレオ 3D 受け渡し座標算出モジュール。
参照ポーズ: 両手を並べて掌を上に向け、小指側を揃えて器を作る形。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import MODELS_DIR, SystemConfig
from kinematics import R1Kinematics
from stereo_depth import StereoResult

PALM_INDICES = [0, 1, 5, 9, 13, 17]
FINGERTIP_INDICES = [8, 12, 16, 20]
MCP_INDICES = [5, 9, 13, 17]


@dataclass
class PalmGestureStatus:
    hands_detected: int = 0
    both_palms_up: bool = False
    cupped_together: bool = False
    pose_valid: bool = False
    hold_progress: float = 0.0         # 0.0 〜 1.0 (1.0 で自動発火！)
    triggered_just_now: bool = False
    gap_px: Optional[float] = None
    palm_center_px: Optional[List[int]] = None
    palm_center_cam_m: Optional[List[float]] = None
    palm_center_torso_m: Optional[List[float]] = None
    landmarks_px: List[List[List[int]]] = field(default_factory=list)
    reason: str = "両手を検出していません"


class KudasaiGestureDetector:
    def __init__(self):
        model_path = MODELS_DIR / "hand_landmarker.task"
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.20,
            min_hand_presence_confidence=0.20,
            min_tracking_confidence=0.20,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.hold_accum_s = 0.0
        self.last_eval_t = time.monotonic()
        self.cooldown_until = 0.0

    def reset_hold(self, cooldown_s: float = 3.0):
        self.hold_accum_s = 0.0
        self.cooldown_until = time.monotonic() + cooldown_s

    def evaluate(
        self,
        left_bgr: np.ndarray,
        stereo: StereoResult,
        kin: R1Kinematics,
        head_pitch: float,
        head_yaw: float,
        cfg: SystemConfig,
    ) -> PalmGestureStatus:
        now = time.monotonic()
        dt = max(0.0, min(0.25, now - self.last_eval_t))
        self.last_eval_t = now

        h, w = left_bgr.shape[:2]
        rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(mp_img)

        n_hands = len(res.hand_landmarks) if res and res.hand_landmarks else 0
        all_lms_px: List[List[List[int]]] = []
        if n_hands == 0:
            self.hold_accum_s = max(0.0, self.hold_accum_s - dt * 2.0)
            return PalmGestureStatus(hands_detected=0, reason="両手を検出していません")

        hands_pts = []
        for lms in res.hand_landmarks:
            pts_px = [[int(round(lm.x * w)), int(round(lm.y * h))] for lm in lms]
            all_lms_px.append(pts_px)
            pts_norm = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float64)
            hands_pts.append((pts_norm, pts_px))

        if n_hands < 2:
            self.hold_accum_s = max(0.0, self.hold_accum_s - dt * 1.8)
            return PalmGestureStatus(
                hands_detected=1,
                hold_progress=round(min(1.0, self.hold_accum_s / max(0.3, cfg.gesture_hold_seconds)), 3),
                landmarks_px=all_lms_px,
                reason="片手のみ検出（両手を揃えて出してください）",
            )

        # 画面上の X 座標で左側の手 (hand_L_img) と右側の手 (hand_R_img) に整列
        hands_pts.sort(key=lambda item: float(item[0][9, 0]))
        (lm_left_img, px_left_img), (lm_right_img, px_right_img) = hands_pts[0], hands_pts[1]

        # 条件1: 握り拳ではなく指が開いて器を作っているか (カップ状に曲げても PIP より外または MCP 以上に伸びている)
        def is_open_cup(lm: np.ndarray) -> bool:
            wrist = lm[0, :2]
            mcp_d = np.mean([np.linalg.norm(lm[idx, :2] - wrist) for idx in MCP_INDICES])
            pip_d = np.mean([np.linalg.norm(lm[idx, :2] - wrist) for idx in (6, 10, 14, 18)])
            tip_d = np.mean([np.linalg.norm(lm[idx, :2] - wrist) for idx in FINGERTIP_INDICES])
            return bool(mcp_d > 0.015 and tip_d > 0.98 * mcp_d and tip_d > 0.88 * pip_d)

        open_ok = is_open_cup(lm_left_img) and is_open_cup(lm_right_img)

        # 条件2: 両手とも「掌が上向き（くださいの形）」であるか
        # 対向カメラから見て人が両掌を上に向けて差し出すと:
        #   画面左側の手 (人の右手) は親指(4)が外側(左: 小さいx)、小指(17/20)が内側(中央寄り: 大きいx) になる！
        #   画面右側の手 (人の左手) は小指(17/20)が内側(中央寄り: 小さいx)、親指(4)が外側(右: 大きいx) になる！
        #   (逆に手の甲を上にすると親指が内側、小指が外側になるため、これで表裏が100%幾何学的に判別できる！)
        left_img_palm_up = float(lm_left_img[4, 0]) < float(lm_left_img[17, 0]) + 0.015
        right_img_palm_up = float(lm_right_img[4, 0]) > float(lm_right_img[17, 0]) - 0.015
        both_palms_up = bool(open_ok and left_img_palm_up and right_img_palm_up)

        # 条件3: 左右の小指付け根(17)・小指先(20)・手首(0)の内側同士が揃って器を作っているか
        pinky_gap_px = math.hypot(
            px_left_img[17][0] - px_right_img[17][0],
            px_left_img[17][1] - px_right_img[17][1],
        )
        wrist_gap_px = math.hypot(
            px_left_img[0][0] - px_right_img[0][0],
            px_left_img[0][1] - px_right_img[0][1],
        )
        inner_gap_px = min(pinky_gap_px, wrist_gap_px)
        y_diff_norm = abs(float(lm_left_img[9, 1]) - float(lm_right_img[9, 1]))

        # 両掌中心の 2D / 3D 座標算出
        palm_uvs = [px_left_img[i] for i in PALM_INDICES] + [px_right_img[i] for i in PALM_INDICES]
        cu = int(round(np.mean([p[0] for p in palm_uvs])))
        cv = int(round(np.mean([p[1] for p in palm_uvs])))

        palm_pts_cam = []
        for u_p, v_p in palm_uvs + [[cu, cv]]:
            u0, u1 = max(0, u_p - 4), min(w, u_p + 5)
            v0, v1 = max(0, v_p - 4), min(h, v_p + 5)
            patch = stereo.points_cam[v0:v1, u0:u1].reshape(-1, 3)
            ok = np.isfinite(patch[:, 2]) & (patch[:, 2] >= 0.18) & (patch[:, 2] <= 1.20)
            if np.any(ok):
                palm_pts_cam.append(np.median(patch[ok], axis=0))

        center_cam_m = None
        center_torso_m = None
        gap_3d_ok = inner_gap_px <= (w * 0.26)
        if palm_pts_cam:
            arr_c = np.array(palm_pts_cam, dtype=np.float64)
            med_c = np.median(arr_c, axis=0)
            center_cam_m = [round(float(v), 4) for v in med_c]
            pt_t = kin.cam_point_to_torso(med_c, head_pitch, head_yaw, cfg)
            center_torso_m = [round(float(v), 4) for v in pt_t]
            # 3D 横幅換算で小指間ギャップを評価
            est_gap_m = (inner_gap_px * float(med_c[2])) / max(100.0, float(cfg.fx))
            gap_3d_ok = est_gap_m <= cfg.palm_pair_max_gap_m

        cupped_together = bool(gap_3d_ok and y_diff_norm <= 0.16)
        pose_valid = bool(both_palms_up and cupped_together and center_torso_m is not None)

        if pose_valid:
            if now >= self.cooldown_until:
                self.hold_accum_s = min(cfg.gesture_hold_seconds, self.hold_accum_s + dt)
            reason = "「ください」両手ポーズ認識中！"
        else:
            self.hold_accum_s = max(0.0, self.hold_accum_s - dt * 1.5)
            if not open_ok:
                reason = "指を開いて手のひらを見せてください"
            elif not both_palms_up:
                reason = "両手のひらを上（親指を外側）に向けてください"
            elif not cupped_together:
                reason = "左右の小指側をもう少し近づけて器を作ってください"
            else:
                reason = "両手のステレオ距離を計測中..."

        progress = min(1.0, self.hold_accum_s / max(0.3, cfg.gesture_hold_seconds))
        triggered = False
        if pose_valid and progress >= 0.999 and now >= self.cooldown_until:
            triggered = True
            reason = "「ください」ジェスチャー発火！"

        return PalmGestureStatus(
            hands_detected=n_hands,
            both_palms_up=both_palms_up,
            cupped_together=cupped_together,
            pose_valid=pose_valid,
            hold_progress=round(progress, 3),
            triggered_just_now=triggered,
            gap_px=round(inner_gap_px, 1),
            palm_center_px=[cu, cv],
            palm_center_cam_m=center_cam_m,
            palm_center_torso_m=center_torso_m,
            landmarks_px=all_lms_px,
            reason=reason,
        )
