"""
両眼ステレオカメラ (`544x448` 整準化済み Left=UDP5002 / Right=UDP5003) からの
高精細サブピクセル視差マップ (StereoSGBM + WLS エッジ保存フィルタ) ＆ 机平面 RANSAC・3D点群生成モジュール。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config import SystemConfig


@dataclass
class StereoResult:
    disp_px: np.ndarray        # (H, W) float32 視差 [px]
    depth_m: np.ndarray        # (H, W) float32 奥行き Zc [m] (無効画素は NaN)
    points_cam: np.ndarray     # (H, W, 3) float32 カメラ光学座標系 [Xc, Yc, Zc] [m]
    disp_vis_bgr: np.ndarray   # (H, W, 3) uint8 可視化ヒートマップ
    table_plane: Optional[np.ndarray] = None  # [nx, ny, nz, d] (n・p + d = 0, 法線はカメラ側向き)


class StereoProcessor:
    def __init__(self):
        # 544x448 (fx*B ≈ 6.60) において Z=0.15m〜1.9m は視差 44px〜3.5px に収まるため
        # numDisparities=48 + blockSize=3 で小物体(10x9px)の輪郭潰れと偽視差スパイクを防止する
        window_size = 3
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=48,
            blockSize=window_size,
            P1=8 * 3 * window_size * window_size,
            P2=32 * 3 * window_size * window_size,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=60,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.has_wls = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createDisparityWLSFilter")
        if self.has_wls:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
            # 小物体のエッジに机の視差が滲み込まないよう平滑化強度を抑えエッジ保存性を高める
            self.wls_filter.setLambda(3200.0)
            self.wls_filter.setSigmaColor(0.9)
        self._disp_ring = []
        self._epipolar_M = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

    def compute(self, left_bgr: np.ndarray, right_bgr: np.ndarray, cfg: SystemConfig) -> StereoResult:
        h, w = left_bgr.shape[:2]
        gray_l = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        gray_r_raw = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        # 実機ステレオ382特徴点で計測された垂直エピポーラ誤差 (median dv = -1.00px) を補正
        gray_r = cv2.warpAffine(
            gray_r_raw, self._epipolar_M, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        # 局所コントラスト強調 (机や白キャップ・小箱の微細テクスチャから視差を安定抽出)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        proc_l = clahe.apply(gray_l)
        proc_r = clahe.apply(gray_r)

        disp_l = self.left_matcher.compute(proc_l, proc_r)
        if self.has_wls:
            disp_r = self.right_matcher.compute(proc_r, proc_l)
            disp_filtered = self.wls_filter.filter(disp_l, left_bgr, None, disp_r)
        else:
            disp_filtered = disp_l

        disp_raw = disp_filtered.astype(np.float32) / 16.0
        # 5フレーム時間方向メディアン統合により単フレーム視差ジッタ (±6.7mm) を ±1.5mm 以下へ抑制
        self._disp_ring.append(disp_raw)
        if len(self._disp_ring) > 5:
            self._disp_ring.pop(0)
        if len(self._disp_ring) >= 3:
            stack = np.stack(self._disp_ring, axis=0)
            med = np.median(stack, axis=0)
            # 動体(手やアーム)では直近フレームを優先し、静止領域(机・対象物)のみ時間統合する
            disp_px = np.where(np.abs(disp_raw - med) < 1.8, med, disp_raw)
        else:
            disp_px = disp_raw

        valid = (disp_px >= 3.5) & (disp_px <= 46.0)

        depth_m = np.full((h, w), np.nan, dtype=np.float32)
        fB = float(cfg.fx * cfg.baseline_m)
        depth_m[valid] = fB / disp_px[valid]
        # 実用作業域 (0.16m 〜 2.2m) のみ有効化
        valid_z = np.isfinite(depth_m) & (depth_m >= 0.16) & (depth_m <= 2.20)
        depth_m[~valid_z] = np.nan

        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        xc = (uu - float(cfg.cx)) * depth_m / float(cfg.fx)
        yc = (vv - float(cfg.cy)) * depth_m / float(cfg.fy)
        points_cam = np.stack([xc, yc, depth_m], axis=-1)

        # ヒートマップ可視化 (numDisparities=48 スケールに最適化)
        disp_norm = np.clip((disp_px - 3.5) / 42.0, 0.0, 1.0)
        disp_u8 = (disp_norm * 255.0).astype(np.uint8)
        disp_vis = cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO)
        disp_vis[~valid_z] = (18, 20, 26)

        plane = self._fit_table_plane_ransac(points_cam)
        return StereoResult(
            disp_px=disp_px,
            depth_m=depth_m,
            points_cam=points_cam,
            disp_vis_bgr=disp_vis,
            table_plane=plane,
        )

    @staticmethod
    def _fit_table_plane_ransac(points_cam: np.ndarray) -> Optional[np.ndarray]:
        """机の作業領域 (0.22m〜0.82m) から机上面平面 n・p + d = 0 を高速 RANSAC 推定する (床面誤検出を排除)。"""
        h, w, _ = points_cam.shape
        roi = points_cam[int(h * 0.20):int(h * 0.70), int(w * 0.14):int(w * 0.86)]
        pts = roi.reshape(-1, 3)
        mask = np.isfinite(pts[:, 2]) & (pts[:, 2] >= 0.22) & (pts[:, 2] <= 0.82)
        pts = pts[mask]
        if pts.shape[0] < 250:
            return None

        rng = np.random.default_rng(42)
        if pts.shape[0] > 2000:
            pts = pts[rng.choice(pts.shape[0], size=2000, replace=False)]

        best_inliers = None
        best_count = 0
        n_pts = pts.shape[0]

        for _ in range(55):
            idx = rng.choice(n_pts, size=3, replace=False)
            p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
            normal = np.cross(p1 - p0, p2 - p0)
            nrm = np.linalg.norm(normal)
            if nrm < 1e-6:
                continue
            normal /= nrm
            # カメラ光学系では +Y が下向きなので、机上面の法線は -Y 成分が優勢であるべき
            if normal[1] > 0:
                normal = -normal
            if abs(normal[1]) < 0.55:
                continue
            d = -float(np.dot(normal, p0))
            dist = np.abs(pts @ normal + d)
            inliers = dist < 0.012  # ±12mm 以内
            cnt = int(np.count_nonzero(inliers))
            if cnt > best_count:
                best_count = cnt
                best_inliers = inliers

        if best_inliers is None or best_count < 150:
            return None

        inlier_pts = pts[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        normal = vt[-1]
        if normal[1] > 0:
            normal = -normal
        d = -float(np.dot(normal, centroid))
        return np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
