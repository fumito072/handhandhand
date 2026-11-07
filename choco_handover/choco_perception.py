"""
高精度インスタンスセグメンテーション (`YOLOE-11l-seg`, `retina_masks=True`) ＋
両眼ステレオ点群からの正確な 3D 形状抽出 (ピクセル輪郭ポリゴン・3D OBB・長辺/短辺/高さmm・主軸角・3D輪郭点群)。
クリック指定 (Click-to-Target) による対象ロックおよび GrabCut+3D平面クラスタ補完も備える。
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from config import MODELS_DIR, SystemConfig
from kinematics import R1Kinematics
from stereo_depth import StereoResult

# R1 オンボード実機校正済み: UDP 5001 (1280x720 HD) <-> UDP 5003 (544x448 Rectified Right) ゼロ視差レンズ写像係数 (median 0.31px)
WX_HD2R = np.array([
    275.6910, 161.0125, -0.2411, -39.1384, 8.9381, -7.0069,
    140.4874, -22.3157, 41.8556, 0.1204, 8.8460, 23.6778, 4.5940,
], dtype=np.float64)
WY_HD2R = np.array([
    212.2503, 0.5930, 91.4300, 16.2041, -9.9087, 8.3148,
    -30.3445, 67.2327, 0.1115, 21.9709, 25.1733, -54.0123, 1.4225,
], dtype=np.float64)
WX_R2HD = np.array([
    631.748, 1033.950, 4.787, 188.463, -113.808, 37.150,
    -1064.238, 72.254, -703.398, -7.770, 122.928, -631.819, -36.707,
], dtype=np.float64)
WY_R2HD = np.array([
    403.137, 6.025, 838.077, -137.547, 91.531, -229.640,
    287.718, -693.316, -144.848, -550.070, -330.944, 1519.717, 309.961,
], dtype=np.float64)


def _poly13(pts: np.ndarray, cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
    x = (pts[:, 0] - cx) / sx
    y = (pts[:, 1] - cy) / sy
    return np.column_stack([
        np.ones_like(x), x, y,
        x * x, x * y, y * y,
        x ** 3, (x ** 2) * y, x * (y ** 2), y ** 3,
        x ** 4, (x ** 2) * (y ** 2), y ** 4,
    ])


def hd_to_right_rect_px(pts_hd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = _poly13(np.asarray(pts_hd, dtype=np.float64), 640.0, 360.0, 640.0, 360.0)
    return A @ WX_HD2R, (A @ WY_HD2R) - 1.0


def right_rect_to_hd_px(pts_r: np.ndarray) -> np.ndarray:
    pts_raw = np.asarray(pts_r, dtype=np.float64).copy()
    pts_raw[:, 1] += 1.0
    A = _poly13(pts_raw, 272.0, 224.0, 272.0, 224.0)
    return np.column_stack([A @ WX_R2HD, A @ WY_R2HD])


@dataclass
class ChocoShape3D:
    label: str
    confidence: float
    bbox_xyxy: List[int]                 # [x1, y1, x2, y2] in 544x448 image
    contour_px: List[List[int]]          # 2D 精密ポリゴン輪郭 [[u, v], ...]
    mask_area_px: int
    centroid_cam_m: List[float]          # [Xc, Yc, Zc] [m]
    centroid_torso_m: List[float]        # [Xt, Yt, Zt] [m] (waist_yaw_link 基準, 体積中心)
    length_mm: float                     # 3D OBB 長辺 [mm]
    width_mm: float                      # 3D OBB 短辺 (把持幅) [mm]
    height_mm: float                     # 机上面からの高さ [mm]
    volume_cm3: float                    # 推定体積 [cm^3]
    yaw_deg: float                       # 水平面内の主軸回転角 [deg]
    recommended_hand: str                # "left" | "right"
    pre_open_amount: int                 # Revo 2 事前開き量 (0..350)
    reachable: bool = True               # 物理アームリーチ圏内かどうか
    ik_error_mm: float = 0.0             # 目標位置での IK 位置残差 [mm]
    move_closer_cm: float = 0.0          # リーチ外の場合、あと何cm手前へ寄せれば届くか
    table_z_torso_m: float = -0.035      # 対象直下の机/天板 Z 座標 [m] (広域RANSAC平面実測)
    pinch_yaw_deg: float = 0.0           # 選択した対向フラット2辺の法線ピンチ角 [deg]
    achieved_pinch_yaw_deg: float = 0.0  # IK が到達する世界基準ピンチ軸角 [deg]
    recommended_waist_yaw: float = 0.0   # 第6自由度として推奨される腰ヨー角 [rad]
    grasp_span_mm: float = 26.0          # 選択された挟み辺の実寸幅 [mm]
    joint_margin_rad: float = 0.35       # 選択候補の最小関節リミット余裕 [rad]
    thumb_aux: int = 495                 # Revo 2 親指対立軸角 (475..498)
    xy_source: str = "plane_ray"         # "hd_plane_ray" | "stereo_plane_ray" | "stereo_cloud"
    table_bbox_torso: dict = field(default_factory=dict)                # 天板 3D バウンディングボックス
    hd_used: bool = False                # 1280x720 HD 輪郭・回転角統合が有効化されたか
    hd_obb_px: List[List[int]] = field(default_factory=list)            # HD 1280x720 上の4頂点回転枠
    hd_pinch_line_px: List[List[int]] = field(default_factory=list)     # HD 上の [親指パッド, 人差指パッド] 座標
    st_pinch_line_px: List[List[int]] = field(default_factory=list)     # Stereo 544x448 上の [親指パッド, 人差指パッド]
    obb_corners_px: List[List[int]] = field(default_factory=list)       # 画像投影済み 8頂点
    contour_3d_torso_m: List[List[float]] = field(default_factory=list) # 後続応用向け 3D 輪郭リング


class ChocoPerception:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = None
        self.active_prompts: List[str] = []
        self.model_error: Optional[str] = None
        self.click_uv: Optional[Tuple[int, int]] = None
        self.last_load_attempt = 0.0

    def set_click_hint(self, u: Optional[int], v: Optional[int]):
        if u is None or v is None:
            self.click_uv = None
        else:
            self.click_uv = (int(u), int(v))

    def _ensure_model(self, prompts: List[str]) -> bool:
        now = time.monotonic()
        prev_cwd = None
        try:
            prev_cwd = os.getcwd()
        except Exception:
            prev_cwd = str(MODELS_DIR)

        try:
            # YOLOE は mobileclip_blt.ts をカレントディレクトリから参照するため MODELS_DIR 内でロード/set_classes する
            os.chdir(str(MODELS_DIR))
            if self.model is None:
                if (now - self.last_load_attempt) < 5.0:
                    return False
                self.last_load_attempt = now
                try:
                    from ultralytics import YOLOE

                    weight_path = MODELS_DIR / "yoloe-11l-seg.pt"
                    target = str(weight_path) if weight_path.exists() else "yoloe-11l-seg.pt"
                    self.model = YOLOE(target)
                    self.model_error = None
                except Exception as exc:
                    self.model_error = str(exc)
                    return False

            clean_prompts = [p.strip() for p in prompts if p.strip()] or ["chocolate", "small box"]
            if clean_prompts != self.active_prompts:
                try:
                    self.model.set_classes(clean_prompts)
                    self.active_prompts = list(clean_prompts)
                except Exception as exc:
                    self.model_error = f"set_classes failed: {exc}"
                    return False
            return True
        finally:
            if prev_cwd and prev_cwd != "/":
                try:
                    os.chdir(prev_cwd)
                except Exception:
                    pass

    def detect_and_extract_shape(
        self,
        left_bgr: np.ndarray,
        stereo: StereoResult,
        kin: R1Kinematics,
        head_pitch: float,
        head_yaw: float,
        cfg: SystemConfig,
        hd_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[ChocoShape3D], List[ChocoShape3D]]:
        h, w = left_bgr.shape[:2]
        candidates_mask_info: List[Tuple[np.ndarray, str, float]] = []

        if self._ensure_model(cfg.choco_prompts):
            try:
                # 544x448 の小物体輪郭を高精細に抜くため imgsz=896 + retina_masks=True で推論
                results = self.model.predict(
                    source=left_bgr,
                    imgsz=896,
                    conf=cfg.seg_conf_threshold,
                    iou=0.45,
                    retina_masks=True,
                    device=self.device,
                    verbose=False,
                )
                if results and results[0].masks is not None and results[0].boxes is not None:
                    masks_data = results[0].masks.data.detach().cpu().numpy()
                    boxes = results[0].boxes
                    confs = boxes.conf.detach().cpu().numpy()
                    clss = boxes.cls.detach().cpu().numpy().astype(int)
                    names = results[0].names

                    for m_raw, conf, cid in zip(masks_data, confs, clss):
                        if m_raw.shape[:2] != (h, w):
                            m_bin = cv2.resize(m_raw, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
                        else:
                            m_bin = m_raw > 0.5
                        area = int(np.count_nonzero(m_bin))
                        # 画面全体の巨大領域(机そのもの等)や極小ノイズを除外 (fx=132 広角では 30mm 物体が約 45〜85px)
                        if area < 24 or area > int(h * w * 0.08):
                            continue
                        lbl = names.get(int(cid), "choco") if isinstance(names, dict) else str(cid)
                        candidates_mask_info.append((m_bin, lbl, float(conf)))
            except Exception as exc:
                self.model_error = f"predict error: {exc}"

        # ユーザーが画面上でクリック指定しているのに YOLO 候補がそこに無い場合、GrabCut で精密マスク抽出
        if self.click_uv is not None:
            cu, cv = self.click_uv
            if 0 <= cu < w and 0 <= cv < h:
                hit_any = any(m[cv, cu] for m, _, _ in candidates_mask_info)
                if not hit_any:
                    gc_mask = self._grabcut_around_click(left_bgr, cu, cv)
                    if gc_mask is not None:
                        candidates_mask_info.append((gc_mask, "clicked_target", 0.95))

        shapes: List[ChocoShape3D] = []
        for m_bin, lbl, conf in candidates_mask_info:
            sh = self._compute_3d_shape_from_mask(
                m_bin, lbl, conf, stereo, kin, head_pitch, head_yaw, cfg, hd_bgr=hd_bgr
            )
            if sh is not None:
                shapes.append(sh)

        # モルフォロジーTopHat + 平面内コア抽出を常に併用し、箱外縁の透明テープ反射に惑わされず上面中央の小物体も確実に候補化する
        if cfg.enable_tabletop_depth_fallback:
            for m_fb, lbl_fb, conf_fb in self._detect_tabletop_salient_masks(left_bgr, stereo):
                sh = self._compute_3d_shape_from_mask(
                    m_fb, lbl_fb, conf_fb, stereo, kin, head_pitch, head_yaw, cfg, hd_bgr=hd_bgr
                )
                if sh is not None and (
                    0.16 <= sh.centroid_torso_m[0] <= 0.65
                    and abs(sh.centroid_torso_m[1]) <= 0.36
                    and -0.24 <= sh.centroid_torso_m[2] <= 0.38
                ):
                    shapes.append(sh)

        if not shapes:
            return None, []

        # クリック位置があればそこに近いものを最優先、なければ「リーチ圏内 → 塊状度(細長いテープ反射を除外) → 正面中央寄り → 信頼度」で選択
        if self.click_uv is not None:
            cu, cv = self.click_uv
            shapes.sort(
                key=lambda s: math.hypot(
                    0.5 * (s.bbox_xyxy[0] + s.bbox_xyxy[2]) - cu,
                    0.5 * (s.bbox_xyxy[1] + s.bbox_xyxy[3]) - cv,
                )
            )
        else:
            def _rank(s: ChocoShape3D):
                aspect_3d = s.length_mm / max(1.0, s.width_mm)
                is_strip = (aspect_3d > 2.4 or s.length_mm > 90.0)
                # 汎用フォールバック(tabletop_object)よりも明確な対象クラス(white_cap / YOLO指定プロンプト)を常に最優先する
                is_generic = (s.label == "tabletop_object")
                return (
                    1 if is_strip else 0,
                    1 if is_generic else 0,
                    0 if s.reachable else 1,
                    round(abs(s.centroid_torso_m[1]), 2),
                    -s.confidence,
                )

            shapes.sort(key=_rank)

        return shapes[0], shapes

    @staticmethod
    def _detect_tabletop_salient_masks(
        bgr: np.ndarray,
        stereo: Optional[StereoResult] = None,
    ) -> List[Tuple[np.ndarray, str, float]]:
        """
        段ボール箱・ガムテープ上・机面などの上にある無地の小物体 (白いボトルキャップや小箱等) を
        モルフォロジー Top-Hat / Black-Hat (25x25 楕円カーネル) ＋ 机平面サポート領域から精密抽出する。
        箱全体の輪郭や長いテープ境界に飲み込まれず、孤立した小物体のみを確実に分離できる。
        """
        h, w = bgr.shape[:2]
        x0, y0 = int(w * 0.28), int(h * 0.26)
        x1, y1 = int(w * 0.72), int(h * 0.62)
        roi = bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return []

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bg_ab = cv2.GaussianBlur(lab[:, :, 1:3], (23, 23), 0)
        chroma_diff = np.linalg.norm(lab[:, :, 1:3] - bg_ab, axis=2)

        # 2D 孤立物体サリエンシー:
        #   - 縦方向のガムテープ帯は縦ブラー bg_vert (1x35) と一致するため d_vert ≈ 0
        #   - 横方向の箱前端フラップ隙間は横ブラー bg_horz (35x1) と一致するため d_horz ≈ 0
        #   - チロルチョコや白キャップ等の真の 2D 孤立小物体のみが 3 方向すべてから突出する！
        bg_iso = cv2.GaussianBlur(lab, (25, 25), 0)
        bg_vert = cv2.GaussianBlur(lab, (1, 35), 0)
        bg_horz = cv2.GaussianBlur(lab, (35, 1), 0)
        d_iso = np.linalg.norm(lab - bg_iso, axis=2)
        d_vert = np.linalg.norm(lab - bg_vert, axis=2)
        d_horz = np.linalg.norm(lab - bg_horz, axis=2)
        sal = np.minimum(d_iso, np.minimum(d_vert, d_horz))

        # 落ち影 (Cast Shadow) の汎用除去: 背景と同じ色相・彩度 (chroma_iso < 6.5) のまま輝度だけ中程度低下した影画素を除外
        chroma_iso = np.hypot(lab[:, :, 1] - bg_iso[:, :, 1], lab[:, :, 2] - bg_iso[:, :, 2])
        is_cast_shadow = (chroma_iso < 6.5) & (lab[:, :, 0] >= 0.46 * bg_iso[:, :, 0]) & (lab[:, :, 0] <= 0.94 * bg_iso[:, :, 0])
        sal = np.where(is_cast_shadow, 0.0, sal)

        # 段ボール・机上面の水平平面コア領域 (-10mm 〜 +85mm) のみをサポートとして残し、
        # 箱の正面側面に貼られた配送伝票ラベルや上面外縁のテープ角反射を幾何学的に完全除外する
        if stereo is not None and stereo.table_plane is not None:
            pts_roi = stereo.points_cam[y0:y1, x0:x1]
            n_pl = stereo.table_plane[:3]
            d_pl = float(stereo.table_plane[3])
            valid_z = np.isfinite(pts_roi[:, :, 2])
            signed_h = np.einsum("ijk,k->ij", np.nan_to_num(pts_roi, nan=0.0), n_pl) + d_pl
            on_plane = valid_z & (signed_h >= -0.012) & (signed_h <= 0.085)
            # 最大平面クラスタの凸包 (Convex Hull) を塗り潰して内部の視差欠損穴を完全に埋め、外周エッジだけ12px収縮させる
            cnts_p, _ = cv2.findContours(on_plane.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_p:
                big_p = max(cnts_p, key=cv2.contourArea)
                if cv2.contourArea(big_p) >= 350:
                    hull_mask = np.zeros_like(on_plane, dtype=np.uint8)
                    cv2.fillConvexPoly(hull_mask, cv2.convexHull(big_p), 255)
                    plane_core = cv2.erode(hull_mask, np.ones((13, 25), np.uint8), iterations=1) > 0
                    if int(np.count_nonzero(plane_core)) >= 120:
                        sal = np.where(plane_core, sal, 0.0)

        out: List[Tuple[np.ndarray, str, float]] = []
        centers_seen: List[Tuple[float, float]] = []

        # マルチ閾値 (36 -> 26 -> 18): 縦テープ・横フラップ影を完全排除した純粋な2D小物体マスクを抽出
        for thr in (36.0, 26.0, 18.0):
            bw = (sal >= thr).astype(np.uint8) * 255
            if thr < 24.0:
                bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                area = cv2.contourArea(c)
                if area < 12 or area > 850:
                    continue
                bx, by, bw_c, bh_c = cv2.boundingRect(c)
                if bx <= 2 or by <= 2 or (bx + bw_c) >= (x1 - x0 - 2) or (by + bh_c) >= (y1 - y0 - 2):
                    continue
                aspect = max(bw_c, bh_c) / max(1.0, min(bw_c, bh_c))
                hull = cv2.convexHull(c)
                solidity = area / max(1.0, cv2.contourArea(hull))
                if aspect > 2.25 or solidity < 0.58:
                    continue

                cx_full = bx + 0.5 * bw_c + x0
                cy_full = by + 0.5 * bh_c + y0
                if any(math.hypot(cx_full - ux, cy_full - uy) < 16.0 for ux, uy in centers_seen):
                    continue

                m_local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                cv2.drawContours(m_local, [c], -1, 255, thickness=-1)
                pix_bool = m_local > 0
                mean_s = float(np.mean(hsv[:, :, 1][pix_bool]))
                mean_v = float(np.mean(hsv[:, :, 2][pix_bool]))
                v_contrast = float(np.percentile(hsv[:, :, 2][pix_bool], 90) - np.percentile(hsv[:, :, 2][pix_bool], 15))
                mean_chroma = float(np.mean(chroma_diff[pix_bool]))
                is_white_cap = (mean_v >= 150.0 and mean_s <= 62.0 and aspect <= 2.0)
                is_tirol_choco = (
                    not is_white_cap
                    and aspect <= 1.85
                    and solidity >= 0.68
                    and (mean_chroma >= 12.5 or (v_contrast >= 45.0 and mean_v <= 155.0) or mean_v <= 95.0)
                )
                if is_white_cap:
                    lbl = "white_cap"
                elif is_tirol_choco:
                    lbl = "tirol_choco"
                else:
                    lbl = "tabletop_object"

                m_u8 = np.zeros((h, w), dtype=np.uint8)
                m_u8[y0:y1, x0:x1] = m_local
                peak_score = float(np.clip(np.percentile(sal[pix_bool], 90) / 48.0, 0.45, 0.94))
                if is_white_cap or is_tirol_choco:
                    peak_score = min(0.96, peak_score + 0.12)
                centers_seen.append((cx_full, cy_full))
                out.append((m_u8 > 0, lbl, peak_score))
        return out

    @staticmethod
    def _grabcut_around_click(bgr: np.ndarray, u: int, v: int) -> Optional[np.ndarray]:
        h, w = bgr.shape[:2]
        r = 46
        x1, y1 = max(2, u - r), max(2, v - r)
        x2, y2 = min(w - 3, u + r), min(h - 3, v + r)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(bgr, mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
            fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
            num_labels, labels = cv2.connectedComponents(fg.astype(np.uint8))
            target_lbl = labels[v, u]
            if target_lbl > 0:
                fg = labels == target_lbl
            if 60 <= int(np.count_nonzero(fg)) <= int(h * w * 0.15):
                return fg
        except Exception:
            pass
        return None

    @staticmethod
    def _refine_with_hd_roi(
        hd_bgr: np.ndarray,
        centroid_cam: np.ndarray,
        h_top_m: float,
        stereo: StereoResult,
        kin: R1Kinematics,
        head_pitch: float,
        head_yaw: float,
        cfg: SystemConfig,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        UDP 5001 (1280x720 HD) の局所ウィンドウから落ち影・テープ帯を除外した極細4頂点 minAreaRect を抽出し、
        (box_hd_px (4,2), corners_top_torso (4,3)) を返す。
        """
        if hd_bgr.ndim != 3 or hd_bgr.shape[:2] != (720, 1280) or stereo.table_plane is None:
            return None
        zc = float(centroid_cam[2])
        if zc <= 0.16:
            return None
        ur_c = float(cfg.fx * (float(centroid_cam[0]) - cfg.baseline_m) / zc + cfg.cx)
        vr_c = float(cfg.fy * float(centroid_cam[1]) / zc + cfg.cy)
        hd_c = right_rect_to_hd_px(np.array([[ur_c, vr_c]], dtype=np.float64))[0]
        uh, vh = int(round(hd_c[0])), int(round(hd_c[1]))
        if not (60 <= uh < 1220 and 50 <= vh < 670):
            return None

        r_win = 125
        x0, y0 = max(0, uh - r_win), max(0, vh - r_win)
        x1, y1 = min(1280, uh + r_win), min(720, vh + r_win)
        roi = hd_bgr[y0:y1, x0:x1]
        if roi.shape[0] < 135 or roi.shape[1] < 135:
            return None

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_iso = cv2.GaussianBlur(lab, (89, 89), 0)
        bg_vert = cv2.GaussianBlur(lab, (1, 121), 0)
        bg_horz = cv2.GaussianBlur(lab, (121, 1), 0)
        sal = np.minimum(
            np.linalg.norm(lab - bg_iso, axis=2),
            np.minimum(np.linalg.norm(lab - bg_vert, axis=2), np.linalg.norm(lab - bg_horz, axis=2)),
        )
        chroma_iso = np.hypot(lab[:, :, 1] - bg_iso[:, :, 1], lab[:, :, 2] - bg_iso[:, :, 2])
        is_shadow = (chroma_iso < 6.5) & (lab[:, :, 0] >= 0.46 * bg_iso[:, :, 0]) & (lab[:, :, 0] <= 0.94 * bg_iso[:, :, 0])
        sal = np.where(is_shadow, 0.0, sal)

        best_cnt = None
        best_dist = 1e9
        for thr in (26.0, 20.0, 15.0):
            bw = (sal >= thr).astype(np.uint8) * 255
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 180 or area > 8500:
                    continue
                bx, by, w_c, h_c = cv2.boundingRect(c)
                if bx <= 2 or by <= 2 or (bx + w_c) >= (x1 - x0 - 2) or (by + h_c) >= (y1 - y0 - 2):
                    continue
                if max(w_c, h_c) / max(1.0, min(w_c, h_c)) > 2.35:
                    continue
                cx_f = bx + 0.5 * w_c + x0
                cy_f = by + 0.5 * h_c + y0
                dist = math.hypot(cx_f - hd_c[0], cy_f - hd_c[1])
                if dist <= 48.0 and dist < best_dist:
                    best_dist = dist
                    best_cnt = c + np.array([[[x0, y0]]], dtype=np.int32)
            if best_cnt is not None:
                break

        if best_cnt is None:
            return None

        rect_hd = cv2.minAreaRect(best_cnt)
        box_hd = cv2.boxPoints(rect_hd)  # (4, 2)
        ur, vr = hd_to_right_rect_px(box_hd)
        n_pl = stereo.table_plane[:3]
        d_pl = float(stereo.table_plane[3])
        o_r = np.array([cfg.baseline_m, 0.0, 0.0], dtype=np.float64)
        T_tc = kin.camera_to_torso_matrix(head_pitch, head_yaw)
        off = np.array([cfg.hand_eye_dx_m, cfg.hand_eye_dy_m, cfg.hand_eye_dz_m], dtype=np.float64)

        pts_top_torso = []
        pts_mid_torso = []
        h_mid_m = 0.5 * float(np.clip(h_top_m, 0.008, 0.030))
        for ui, vi in zip(ur, vr):
            ray = np.array([(float(ui) - cfg.cx) / cfg.fx, (float(vi) - cfg.cy) / cfg.fy, 1.0], dtype=np.float64)
            denom = float(np.dot(n_pl, ray))
            if abs(denom) < 1e-4:
                return None
            s_top = -(d_pl - h_top_m + float(np.dot(n_pl, o_r))) / denom
            s_mid = -(d_pl - h_mid_m + float(np.dot(n_pl, o_r))) / denom
            if s_top <= 0.16 or s_top >= 1.50 or s_mid <= 0.16 or s_mid >= 1.50:
                return None
            p_cam_top = o_r + s_top * ray
            p_cam_mid = o_r + s_mid * ray
            pts_top_torso.append((T_tc @ np.append(p_cam_top, 1.0))[:3] + off)
            pts_mid_torso.append((T_tc @ np.append(p_cam_mid, 1.0))[:3] + off)

        return box_hd, np.array(pts_top_torso, dtype=np.float64), np.array(pts_mid_torso, dtype=np.float64)

    @staticmethod
    def extract_table_bbox_torso(
        stereo: StereoResult,
        kin: R1Kinematics,
        head_pitch: float,
        head_yaw: float,
        cfg: SystemConfig,
    ) -> dict:
        """広域ステレオRANSAC机平面インライア点群から Torso 座標系での 3D 天板外形ボックスを算出する。"""
        default_box = {
            "x_min": 0.21,
            "x_max": 0.64,
            "y_min": -0.26,
            "y_max": 0.26,
            "z_top": -0.045,
        }
        if stereo is None or stereo.table_plane is None:
            return default_box
        h, w, _ = stereo.points_cam.shape
        roi = stereo.points_cam[int(h * 0.18):int(h * 0.78), int(w * 0.08):int(w * 0.92)].reshape(-1, 3)
        valid = np.isfinite(roi[:, 2]) & (roi[:, 2] >= 0.20) & (roi[:, 2] <= 0.88)
        pts = roi[valid]
        if pts.shape[0] < 120:
            return default_box
        n_pl = stereo.table_plane[:3]
        d_pl = float(stereo.table_plane[3])
        inliers = np.abs(pts @ n_pl + d_pl) < 0.015
        pts_in = pts[inliers]
        if pts_in.shape[0] < 80:
            return default_box
        if pts_in.shape[0] > 3000:
            pts_in = pts_in[:: max(1, pts_in.shape[0] // 3000)]
        T_tc = kin.camera_to_torso_matrix(head_pitch, head_yaw)
        off = np.array([cfg.hand_eye_dx_m, cfg.hand_eye_dy_m, cfg.hand_eye_dz_m], dtype=np.float64)
        ones = np.ones((pts_in.shape[0], 1), dtype=np.float64)
        pts_t = (np.hstack([pts_in, ones]) @ T_tc.T)[:, :3] + off
        pts_t = pts_t[pts_t[:, 0] >= 0.19]
        if pts_t.shape[0] < 40:
            return default_box
        return {
            "x_min": round(max(0.20, float(np.percentile(pts_t[:, 0], 5))), 3),
            "x_max": round(float(np.percentile(pts_t[:, 0], 97)), 3),
            "y_min": round(float(np.percentile(pts_t[:, 1], 3)), 3),
            "y_max": round(float(np.percentile(pts_t[:, 1], 97)), 3),
            "z_top": round(float(np.median(pts_t[:, 2])), 4),
        }

    @staticmethod
    def _compute_3d_shape_from_mask(
        mask_bool: np.ndarray,
        label: str,
        confidence: float,
        stereo: StereoResult,
        kin: R1Kinematics,
        head_pitch: float,
        head_yaw: float,
        cfg: SystemConfig,
        hd_bgr: Optional[np.ndarray] = None,
    ) -> Optional[ChocoShape3D]:
        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        area_px = int(cv2.contourArea(cnt))
        if area_px < 20:
            return None

        eps = max(1.0, 0.008 * cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        contour_px = [[int(p[0]), int(p[1])] for p in approx]
        x, y, bw, bh = cv2.boundingRect(cnt)
        bbox_xyxy = [int(x), int(y), int(x + bw), int(y + bh)]

        # 小さい物体 (<140px) ではマスク収縮せずそのまま 3D 点を抽出
        if area_px >= 140:
            eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1) > 0
            if np.count_nonzero(eroded) < 12:
                eroded = mask_bool
        else:
            eroded = mask_bool

        pts_cam = stereo.points_cam[eroded]
        valid = np.isfinite(pts_cam[:, 2]) & (pts_cam[:, 2] >= 0.16) & (pts_cam[:, 2] <= 1.85)
        pts_cam = pts_cam[valid]

        # 机平面 RANSAC が有効な場合、机面より 15mm 以上奥へ抜けた背景リーク点を除外し、
        # 光沢包装・純白無地で視差が欠損した場合も机平面レイ交差から正確な 3D 点を復元する
        if stereo.table_plane is not None:
            n_pl = stereo.table_plane[:3]
            d_pl = float(stereo.table_plane[3])
            if pts_cam.shape[0] >= 5:
                signed_h = pts_cam @ n_pl + d_pl
                above_or_on_desk = (signed_h >= -0.015) & (signed_h <= 0.14)
                if int(np.count_nonzero(above_or_on_desk)) >= 5:
                    pts_cam = pts_cam[above_or_on_desk]
            if pts_cam.shape[0] < 6:
                ys_m, xs_m = np.where(eroded)
                if xs_m.size >= 6:
                    rays = np.stack(
                        [(xs_m - cfg.cx) / cfg.fx, (ys_m - cfg.cy) / cfg.fy, np.ones_like(xs_m, dtype=np.float64)],
                        axis=-1,
                    )
                    denom = rays @ n_pl
                    ok_r = np.abs(denom) > 1e-4
                    if np.any(ok_r):
                        s_desk = -d_pl / denom[ok_r]
                        ok_s = (s_desk >= 0.20) & (s_desk <= 0.90)
                        if int(np.count_nonzero(ok_s)) >= 6:
                            pts_cam = rays[ok_r][ok_s] * s_desk[ok_s, None] + 0.008 * n_pl[None, :]

        if pts_cam.shape[0] < 6:
            return None

        # ロバスト深度フィルタ (中央値 ± 38mm 以内のみ残し背景リークを完全排除)
        med_z = float(np.median(pts_cam[:, 2]))
        inlier_z = np.abs(pts_cam[:, 2] - med_z) <= 0.038
        pts_cam = pts_cam[inlier_z]
        if pts_cam.shape[0] < 5:
            return None

        # Torso (`waist_yaw_link`) 3D 座標系へ一括変換
        T_tc = kin.camera_to_torso_matrix(head_pitch, head_yaw)
        ones = np.ones((pts_cam.shape[0], 1), dtype=np.float64)
        pts_torso = (np.hstack([pts_cam, ones]) @ T_tc.T)[:, :3]
        pts_torso += np.array([cfg.hand_eye_dx_m, cfg.hand_eye_dy_m, cfg.hand_eye_dz_m], dtype=np.float64)

        centroid_cam = np.median(pts_cam, axis=0)
        centroid_torso = np.median(pts_torso, axis=0)
        xy_source = "stereo_cloud"

        # 高さおよび対象直下の机天面 Z_table 算出 (広域RANSAC平面×2D画素重心レイ交差で 10x9px 視差量子化ノイズを完全排除)
        off_he = np.array([cfg.hand_eye_dx_m, cfg.hand_eye_dy_m, cfg.hand_eye_dz_m], dtype=np.float64)
        table_bbox_torso = ChocoPerception.extract_table_bbox_torso(stereo, kin, head_pitch, head_yaw, cfg)
        if stereo.table_plane is not None:
            n_pl = stereo.table_plane[:3]
            d_pl = float(stereo.table_plane[3])
            heights_m = np.abs(pts_cam @ n_pl + d_pl)
            height_mm = float(np.clip(np.percentile(heights_m, 88) * 1000.0, 10.0, 42.0))
            ys_mask, xs_mask = np.where(mask_bool)
            if xs_mask.size >= 6:
                u_mc = float(np.mean(xs_mask))
                v_mc = float(np.mean(ys_mask))
                ray_l = np.array([(u_mc - cfg.cx) / cfg.fx, (v_mc - cfg.cy) / cfg.fy, 1.0], dtype=np.float64)
                denom_l = float(np.dot(n_pl, ray_l))
                if abs(denom_l) > 1e-4:
                    h_mid_est = 0.5 * (height_mm / 1000.0)
                    s_tbl = -d_pl / denom_l
                    s_mid = -(d_pl - h_mid_est) / denom_l
                    if 0.18 <= s_tbl <= 1.25 and 0.18 <= s_mid <= 1.25:
                        p_tbl_cam = s_tbl * ray_l
                        centroid_cam = s_mid * ray_l
                        p_tbl_torso = (T_tc @ np.append(p_tbl_cam, 1.0))[:3] + off_he
                        centroid_torso = (T_tc @ np.append(centroid_cam, 1.0))[:3] + off_he
                        xy_source = "stereo_plane_ray"
                        z_bot = float(p_tbl_torso[2])
                        z_top = z_bot + (height_mm / 1000.0)
            if xy_source != "stereo_plane_ray":
                denom_c = float(np.dot(n_pl, centroid_cam))
                p_tbl_cam = centroid_cam * (-d_pl / denom_c) if abs(denom_c) > 1e-4 else (centroid_cam - (height_mm / 1000.0) * n_pl)
                p_tbl_torso = (T_tc @ np.append(p_tbl_cam, 1.0))[:3] + off_he
                z_bot = float(p_tbl_torso[2])
                z_top = z_bot + (height_mm / 1000.0)
        else:
            z_span_mm = float((np.percentile(pts_torso[:, 2], 95) - np.percentile(pts_torso[:, 2], 8)) * 1000.0)
            height_mm = max(12.0, z_span_mm)
            z_top = float(np.percentile(pts_torso[:, 2], 92))
            z_bot = z_top - (height_mm / 1000.0)

        height_mm = float(np.clip(height_mm, 8.0, 140.0))
        z_top = z_bot + (height_mm / 1000.0)

        # 水平面 (Xt, Yt) 上で最小外接回転矩形を算出:
        # UDP 5001 (1280x720 HD, 1px≈1mm) が有効なら HD 光線 × 広域3D机平面交差点を 100% 採用しステレオ視差量子化誤差を排除！
        hd_used = False
        hd_obb_px: List[List[int]] = []
        hd_res = (
            ChocoPerception._refine_with_hd_roi(
                hd_bgr, centroid_cam, height_mm / 1000.0, stereo, kin, head_pitch, head_yaw, cfg
            )
            if (hd_bgr is not None and getattr(cfg, "enable_hd_vision", True))
            else None
        )

        if hd_res is not None:
            box_hd, pts_top_hd, pts_mid_hd = hd_res
            hd_used = True
            hd_obb_px = [[int(round(p[0])), int(round(p[1]))] for p in box_hd]
            e01 = pts_mid_hd[1, :2] - pts_mid_hd[0, :2]
            e12 = pts_mid_hd[2, :2] - pts_mid_hd[1, :2]
            s01_mm = float(np.linalg.norm(e01) * 1000.0)
            s12_mm = float(np.linalg.norm(e12) * 1000.0)
            if s01_mm >= s12_mm:
                length_mm, width_mm = s01_mm, s12_mm
                yaw_deg = float(math.degrees(math.atan2(e01[1], e01[0])))
            else:
                length_mm, width_mm = s12_mm, s01_mm
                yaw_deg = float(math.degrees(math.atan2(e12[1], e12[0])))
            hd_mid_xy = pts_mid_hd[:, :2].mean(axis=0)
            if float(np.linalg.norm(hd_mid_xy - centroid_torso[:2])) <= 0.045:
                centroid_torso[:2] = hd_mid_xy
                xy_source = "hd_plane_ray"
            box2d_m = pts_mid_hd[:, :2]
        else:
            xy_mm = (pts_torso[:, :2] * 1000.0).astype(np.float32)
            rect = cv2.minAreaRect(xy_mm)
            (cx_mm, cy_mm), (dim1_mm, dim2_mm), angle_deg = rect
            if dim1_mm >= dim2_mm:
                length_mm, width_mm = float(dim1_mm), float(dim2_mm)
                yaw_deg = float(angle_deg)
            else:
                length_mm, width_mm = float(dim2_mm), float(dim1_mm)
                yaw_deg = float(angle_deg + 90.0)
            box2d_m = cv2.boxPoints(((cx_mm, cy_mm), (dim1_mm, dim2_mm), angle_deg)) / 1000.0

        # 体積中心 Z (上面屋根ではなく天面 z_bot と上面 z_top の中点) を centroid_torso[2] に格納
        centroid_torso[2] = 0.5 * (z_bot + z_top)

        # 机面全体や大型マウスパッド等の巨大誤検出を除外 (クリック指定時は許容)
        if label != "clicked_target" and (length_mm > 175.0 or width_mm > 135.0):
            return None
        while yaw_deg > 90.0:
            yaw_deg -= 180.0
        while yaw_deg < -90.0:
            yaw_deg += 180.0

        length_mm = float(np.clip(length_mm, 12.0, 220.0))
        width_mm = float(np.clip(width_mm, 10.0, 180.0))
        volume_cm3 = round((length_mm * width_mm * height_mm) / 1000.0, 2)

        # 3D OBB 8頂点を生成し画像へ再投影 (HD有効時は contour_px も実寸4頂点に引き締めて膨張をゼロにする)
        corners_torso = []
        for bx, by in box2d_m:
            corners_torso.append([float(bx), float(by), z_bot])
        for bx, by in box2d_m:
            corners_torso.append([float(bx), float(by), z_top])
        corners_cam = kin.torso_points_to_cam(np.array(corners_torso), head_pitch, head_yaw, cfg)
        obb_px: List[List[int]] = []
        for pc in corners_cam:
            if pc[2] > 0.05:
                u = int(round(cfg.fx * pc[0] / pc[2] + cfg.cx))
                v = int(round(cfg.fy * pc[1] / pc[2] + cfg.cy))
                obb_px.append([u, v])
        if hd_used and len(obb_px) == 8:
            contour_px = obb_px[4:8]
            xs_t = [p[0] for p in contour_px]
            ys_t = [p[1] for p in contour_px]
            bbox_xyxy = [min(xs_t), min(ys_t), max(xs_t), max(ys_t)]

        # 3D 輪郭ポリゴンリング (代表最大32頂点)
        contour_3d_torso: List[List[float]] = []
        step = max(1, len(contour_px) // 32)
        for u_c, v_c in contour_px[::step]:
            zc = float(centroid_cam[2])
            if 0 <= v_c < stereo.depth_m.shape[0] and 0 <= u_c < stereo.depth_m.shape[1]:
                d_val = float(stereo.depth_m[v_c, u_c])
                if math.isfinite(d_val) and abs(d_val - zc) < 0.05:
                    zc = d_val
            xc = (u_c - cfg.cx) * zc / cfg.fx
            yc = (v_c - cfg.cy) * zc / cfg.fy
            pt_t = kin.cam_point_to_torso(np.array([xc, yc, zc]), head_pitch, head_yaw, cfg)
            contour_3d_torso.append([round(float(val), 4) for val in pt_t])

        # 対象物の実際の向き・サイズに合わせて最適なアーム・4通り候補・第6自由度腰角・thumb_aux を自動計画！
        z_pinch_probe = z_bot + float(np.clip(0.42 * (height_mm / 1000.0), 0.0045, 0.014))
        grasp_probe = np.array([float(centroid_torso[0]), float(centroid_torso[1]), z_pinch_probe], dtype=np.float64)
        rec_waist_yaw = float(getattr(cfg, "look_waist_yaw", 0.0))
        grasp_span_mm = float(width_mm)
        j_margin_rad = 0.35
        if getattr(cfg, "grasp_posture", "surface_pinch") in ("surface_pinch", "c_pinch"):
            plan6 = kin.choose_oriented_pinch_plan_6dof(
                grasp_probe,
                yaw_deg,
                length_mm,
                width_mm,
                cfg,
                current_waist_yaw=rec_waist_yaw,
            )
            rec_hand = str(plan6["side"])
            pinch_yaw_deg = float(plan6["world_pinch_yaw_deg"])
            achieved_yaw_deg = float(plan6["world_achieved_yaw_deg"])
            best_err_m = float(plan6["pos_err_m"])
            thumb_aux = int(plan6["thumb_aux"])
            rec_waist_yaw = float(plan6["waist_yaw_cmd"])
            grasp_span_mm = float(plan6["grasp_span_mm"])
            j_margin_rad = float(plan6["joint_margin_rad"])
        else:
            _, err_l = kin.solve_arm_ik("left", grasp_probe, cfg)
            _, err_r = kin.solve_arm_ik("right", grasp_probe, cfg)
            rec_hand = kin.select_hand_for_point(centroid_torso) if abs(err_l - err_r) <= 0.006 else ("left" if err_l < err_r else "right")
            best_err_m = err_l if rec_hand == "left" else err_r
            pinch_yaw_deg = kin.normalize_half_plane_deg(yaw_deg + 90.0)
            achieved_yaw_deg = pinch_yaw_deg
            thumb_aux = 495

        # 親指パッド・人差指パッドの計画接触位置を Stereo / HD 両画像へ投影して UI に可視化
        prad = math.radians(achieved_yaw_deg)
        half_span_m = 0.5 * max(0.020, min(0.052, grasp_span_mm / 1000.0 + 0.010))
        d_vec = np.array([math.cos(prad), math.sin(prad), 0.0], dtype=np.float64)
        p_thumb_pad = np.array([centroid_torso[0], centroid_torso[1], z_top], dtype=np.float64) - half_span_m * d_vec
        p_index_pad = np.array([centroid_torso[0], centroid_torso[1], z_top], dtype=np.float64) + half_span_m * d_vec
        pads_cam = kin.torso_points_to_cam(np.array([p_thumb_pad, p_index_pad]), head_pitch, head_yaw, cfg)
        st_pinch_line_px: List[List[int]] = []
        hd_pinch_line_px: List[List[int]] = []
        for pc in pads_cam:
            if pc[2] > 0.05:
                ul = int(round(cfg.fx * pc[0] / pc[2] + cfg.cx))
                vl = int(round(cfg.fy * pc[1] / pc[2] + cfg.cy))
                st_pinch_line_px.append([ul, vl])
                ur_p = cfg.fx * (pc[0] - cfg.baseline_m) / pc[2] + cfg.cx
                vr_p = cfg.fy * pc[1] / pc[2] + cfg.cy
                hp = right_rect_to_hd_px(np.array([[ur_p, vr_p]], dtype=np.float64))[0]
                hd_pinch_line_px.append([int(round(hp[0])), int(round(hp[1]))])

        in_box = (
            cfg.reach_x_min <= float(centroid_torso[0]) <= cfg.reach_x_max
            and abs(float(centroid_torso[1])) <= cfg.reach_y_abs_max
            and cfg.reach_z_min <= float(centroid_torso[2]) <= cfg.reach_z_max
        )
        ik_err_mm = round(float(best_err_m) * 1000.0, 1)
        is_reachable = bool(in_box and ik_err_mm <= 18.0)
        closer_cm = 0.0
        if not is_reachable and best_err_m > 0.012:
            closer_cm = round(max(1.0, (float(best_err_m) - 0.008) * 100.0 + 1.5), 1)

        # 幅に応じた事前開き量 (2指ピンチで角に一切触れず余裕を持って跨げる開口幅 0..45)
        pre_open = int(np.clip(round((45.0 - grasp_span_mm) * 1.1), 0, 45))

        return ChocoShape3D(
            label=label,
            confidence=round(confidence, 3),
            bbox_xyxy=bbox_xyxy,
            contour_px=contour_px,
            mask_area_px=area_px,
            centroid_cam_m=[round(float(v), 4) for v in centroid_cam],
            centroid_torso_m=[round(float(v), 4) for v in centroid_torso],
            length_mm=round(length_mm, 1),
            width_mm=round(width_mm, 1),
            height_mm=round(height_mm, 1),
            volume_cm3=volume_cm3,
            yaw_deg=round(yaw_deg, 1),
            recommended_hand=rec_hand,
            pre_open_amount=pre_open,
            reachable=is_reachable,
            ik_error_mm=ik_err_mm,
            move_closer_cm=closer_cm,
            table_z_torso_m=round(z_bot, 4),
            pinch_yaw_deg=round(pinch_yaw_deg, 1),
            achieved_pinch_yaw_deg=round(achieved_yaw_deg, 1),
            recommended_waist_yaw=round(rec_waist_yaw, 4),
            grasp_span_mm=round(grasp_span_mm, 1),
            joint_margin_rad=round(j_margin_rad, 3),
            thumb_aux=thumb_aux,
            xy_source=xy_source,
            table_bbox_torso=table_bbox_torso,
            hd_used=hd_used,
            hd_obb_px=hd_obb_px,
            hd_pinch_line_px=hd_pinch_line_px,
            st_pinch_line_px=st_pinch_line_px,
            obb_corners_px=obb_px,
            contour_3d_torso_m=contour_3d_torso,
        )
