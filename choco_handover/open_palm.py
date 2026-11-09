"""
片手の開いた手のひらを HD 画像 (1280x720) から見つけ、ステレオの視差で 3D 位置を求める。

ステレオ左画像 (544x448) では手のひらが 20px 程度しかなく手検出に小さすぎるため、検出は HD の机まわりの領域で行う。
HD の点は hd_to_right_rect_px で整準化右画像へ写し、「左画像の画素 u から視差 d を引いた位置が手のひらの内側に入る画素」の
3D 点の中央値を手のひらの位置とする。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from choco_perception import hd_to_right_rect_px
from config import MODELS_DIR, SystemConfig
from kinematics import R1Kinematics
from stereo_depth import StereoResult

HD_ROI = (300, 20, 980, 700)           # 手を探す HD 画像の範囲 (机とその周辺)。手検出モデルの都合で正方形にする
PALM_POLY = [0, 1, 5, 9, 13, 17]       # 手のひらの外周になる点
FINGERS = [(5, 8), (9, 12), (13, 16), (17, 20)]   # (付け根, 指先)
OPEN_RATIO = 1.25                      # 手首〜指先の距離が 手首〜付け根 の何倍以上なら指が伸びているとみなすか


@dataclass
class OpenPalm:
    hd_px: List[int]
    label: str
    score: float
    open_fingers: int
    is_open: bool
    palm_up: bool
    depth_points: int
    center_cam_m: Optional[List[float]] = None
    center_torso_m: Optional[List[float]] = None
    landmarks_hd: Optional[List[List[int]]] = None


class OpenPalmDetector:
    def __init__(self) -> None:
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODELS_DIR / "hand_landmarker.task")),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect(
        self,
        hd_bgr: np.ndarray,
        stereo: StereoResult,
        kin: R1Kinematics,
        head_pitch: float,
        head_yaw: float,
        cfg: SystemConfig,
    ) -> List[OpenPalm]:
        if hd_bgr is None or hd_bgr.shape[:2] != (720, 1280):
            return []
        x0, y0, x1, y1 = HD_ROI
        rgb = cv2.cvtColor(hd_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        out: List[OpenPalm] = []
        for i, lms in enumerate(res.hand_landmarks or []):
            pts = np.array([[lm.x * (x1 - x0) + x0, lm.y * (y1 - y0) + y0] for lm in lms], dtype=np.float64)
            world = np.array([[lm.x, lm.y, lm.z] for lm in res.hand_world_landmarks[i]], dtype=np.float64)
            cat = res.handedness[i][0]
            n_open = self._count_open_fingers(world)
            center_px = pts[PALM_POLY].mean(axis=0)
            palm = OpenPalm(
                hd_px=[int(round(center_px[0])), int(round(center_px[1]))],
                label=str(cat.category_name),
                score=round(float(cat.score), 2),
                open_fingers=n_open,
                is_open=n_open >= 3,
                palm_up=self._palm_faces_camera(pts, str(cat.category_name)),
                depth_points=0,
                landmarks_hd=[[int(round(p[0])), int(round(p[1]))] for p in pts],
            )
            found = self._palm_3d(pts, stereo)
            if found is not None:
                cam, palm.depth_points = found
                palm.center_cam_m = [round(float(v), 4) for v in cam]
                torso = kin.cam_point_to_torso(cam, head_pitch, head_yaw, cfg)
                palm.center_torso_m = [round(float(v), 4) for v in torso]
            out.append(palm)
        return out

    @staticmethod
    def _count_open_fingers(world: np.ndarray) -> int:
        wrist = world[0]
        return sum(
            int(np.linalg.norm(world[tip] - wrist) >= OPEN_RATIO * np.linalg.norm(world[mcp] - wrist))
            for mcp, tip in FINGERS
        )

    @staticmethod
    def _palm_faces_camera(pts: np.ndarray, label: str) -> bool:
        """手首→人差指付け根 と 手首→小指付け根 の外積の符号と左右の判定から、手のひら側が見えているかを返す。

        右手の手のひらが見えているとき、画像座標 (y 下向き) の外積は負になる。
        実機の HD 画像 (鏡像でない) では、右手に "Right" のラベルが付くことを 2026-11-09 に確認した。
        """
        a, b = pts[5] - pts[0], pts[17] - pts[0]
        cross = float(a[0] * b[1] - a[1] * b[0])
        return cross < 0.0 if label == "Right" else cross > 0.0

    @staticmethod
    def _palm_3d(pts_hd: np.ndarray, stereo: StereoResult) -> Optional[Tuple[np.ndarray, int]]:
        ur, vr = hd_to_right_rect_px(pts_hd[PALM_POLY])
        H, W = stereo.disp_px.shape[:2]
        mask_r = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask_r, np.column_stack([ur, vr]).round().astype(np.int32), 1)
        mask_r = cv2.erode(mask_r, np.ones((3, 3), dtype=np.uint8))
        if not mask_r.any():
            return None
        v0, v1 = max(int(vr.min()), 0), min(int(vr.max()) + 2, H)
        u0, u1 = max(int(ur.min()), 0), min(int(ur.max()) + 50, W)   # 左画像では視差ぶん右にずれる
        if v1 <= v0 or u1 <= u0:
            return None
        vv, uu = np.mgrid[v0:v1, u0:u1]
        d = stereo.disp_px[v0:v1, u0:u1]
        valid = np.isfinite(d) & (d > 0.5)
        u_right = np.clip(np.round(uu - np.nan_to_num(d)).astype(np.int32), 0, W - 1)
        inside = valid & (mask_r[vv, u_right] > 0)
        points = stereo.points_cam[vv[inside], uu[inside]]
        points = points[np.isfinite(points[:, 2]) & (points[:, 2] >= 0.15) & (points[:, 2] <= 1.2)]
        if len(points) < 12:
            return None
        z_med = float(np.median(points[:, 2]))
        keep = np.abs(points[:, 2] - z_med) <= 0.04   # 指のすき間から見える背景などを除く
        if int(keep.sum()) < 12:
            return None
        return np.median(points[keep], axis=0), int(keep.sum())
