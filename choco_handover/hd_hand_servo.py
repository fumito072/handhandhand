"""
HD 画像 (UDP 5001, 1280x720) 上で「親指の先」と「チョコ」を同時に測り、机面上での相対ずれを求める。

着地して指先が机面にある瞬間 (閉じる直前) に使う。手の位置もチョコの位置も同じカメラ模型で机面へ写すので、
カメラ外部パラメータや Hand-Eye 補正 (hand_eye_dx/dy/dz) の誤差は両方に同じように乗り、差を取ると打ち消し合う。

- 親指の先: 腕が視野に入る前の HD 画像 (事前確認) との差分から手の領域を取り出し、
  順運動学で予測した親指の位置の周りで、手の領域の最下点 (画像の一番手前) を親指の先とする。
  surface_pinch では親指がロボット側 (画像の下) を向くので、最下点が親指になる。
- チョコ: 白い包装 (＋赤い文字) の小さなかたまりを、前回位置の周りで探す (指で押されて動くことがあるため毎回測る)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from choco_perception import hd_to_right_rect_px, right_rect_to_hd_px
from config import SystemConfig
from kinematics import R1Kinematics


@dataclass
class HandChocoObservation:
    valid: bool
    reason: str = ""
    thumb_px: Optional[Tuple[int, int]] = None
    thumb_pred_px: Optional[Tuple[int, int]] = None
    choco_px: Optional[Tuple[int, int]] = None
    thumb_xy_m: Optional[List[float]] = None
    choco_xy_m: Optional[List[float]] = None
    shift_xy_m: Optional[List[float]] = None  # 手をこれだけ動かせば、チョコが親指と人差指の間の中央に来る
    along_mm: Optional[float] = None          # shift の把持軸 (親指→人差指) 方向成分
    across_mm: Optional[float] = None         # shift の把持軸と直交する成分


def _hand_eye_offset(cfg: SystemConfig) -> np.ndarray:
    return np.array([cfg.hand_eye_dx_m, cfg.hand_eye_dy_m, cfg.hand_eye_dz_m], dtype=np.float64)


def torso_to_hd_px(
    p_torso: Sequence[float], kin: R1Kinematics, head_pitch: float, head_yaw: float, cfg: SystemConfig
) -> Optional[np.ndarray]:
    """胴体座標の点を HD 画像の画素へ投影する (右ステレオカメラ経由)。"""
    pc = kin.torso_points_to_cam(np.asarray(p_torso, dtype=np.float64).reshape(1, 3), head_pitch, head_yaw, cfg)[0]
    if pc[2] <= 0.05:
        return None
    ur = cfg.fx * (pc[0] - cfg.baseline_m) / pc[2] + cfg.cx
    vr = cfg.fy * pc[1] / pc[2] + cfg.cy
    return right_rect_to_hd_px(np.array([[ur, vr]], dtype=np.float64))[0]


def hd_px_to_torso_at_z(
    u: float, v: float, z_torso: float, kin: R1Kinematics, head_pitch: float, head_yaw: float, cfg: SystemConfig
) -> Optional[np.ndarray]:
    """HD 画素 (u, v) の視線と、胴体座標の水平面 Z = z_torso との交点を返す。"""
    ur, vr = hd_to_right_rect_px(np.array([[u, v]], dtype=np.float64))
    ray = np.array([(float(ur[0]) - cfg.cx) / cfg.fx, (float(vr[0]) - cfg.cy) / cfg.fy, 1.0], dtype=np.float64)
    T = kin.camera_to_torso_matrix(head_pitch, head_yaw)
    origin = (T @ np.array([cfg.baseline_m, 0.0, 0.0, 1.0], dtype=np.float64))[:3] + _hand_eye_offset(cfg)
    d = T[:3, :3] @ ray
    if abs(float(d[2])) < 1e-4:
        return None
    s = (float(z_torso) - float(origin[2])) / float(d[2])
    if s <= 0.10 or s >= 1.50:
        return None
    return origin + s * d


def detect_thumb_tip_px(
    hd_now: np.ndarray,
    hd_ref: np.ndarray,
    pred_px: Sequence[float],
    exclude_boxes: Sequence[Tuple[int, int, int, int]] = (),
    half_w: int = 95,
    up: int = 80,
    down: int = 110,
    diff_thr: int = 38,
) -> Optional[Tuple[int, int]]:
    """予測位置の周りで、事前確認画像との差分から手の領域を取り出し、その最下点を親指の先として返す。"""
    if hd_now is None or hd_ref is None or hd_now.shape != hd_ref.shape:
        return None
    h, w = hd_now.shape[:2]
    pu, pv = int(round(pred_px[0])), int(round(pred_px[1]))
    u0, u1 = max(0, pu - half_w), min(w, pu + half_w)
    v0, v1 = max(0, pv - up), min(h, pv + down)
    if u1 - u0 < 30 or v1 - v0 < 30:
        return None
    a = cv2.GaussianBlur(hd_now[v0:v1, u0:u1], (5, 5), 0).astype(np.int16)
    b = cv2.GaussianBlur(hd_ref[v0:v1, u0:u1], (5, 5), 0).astype(np.int16)
    mask = (np.max(np.abs(a - b), axis=2) > diff_thr).astype(np.uint8) * 255
    for (x0, y0, x1, y1) in exclude_boxes:
        xa, xb = max(0, x0 - u0), min(u1 - u0, x1 - u0)
        ya, yb = max(0, y0 - v0), min(v1 - v0, y1 - v0)
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if int(st[best, cv2.CC_STAT_AREA]) < 300:
        return None
    ys, xs = np.where(lab == best)
    y_max = int(ys.max())
    if y_max >= (v1 - v0) - 3:
        # 手の領域が探索窓の下端まで続いている = 親指の先ではなく影・ケーブル等の可能性が高い
        return None
    x_low = float(xs[ys >= y_max - 6].mean())
    # 黒い親指は黒い机とほとんど差が出ず、先端の銀色の金具だけが別の小さな領域に分かれることがある。
    # 手の最下点のすぐ下 (45px 以内・左右 30px 以内) にある小さな領域は親指の先として採用する。
    for i in range(1, n):
        if i == best or int(st[i, cv2.CC_STAT_AREA]) < 40:
            continue
        ys_i, xs_i = np.where(lab == i)
        yb = int(ys_i.max())
        if y_max < yb <= y_max + 45 and abs(float(xs_i.mean()) - x_low) <= 30 and yb < (v1 - v0) - 3:
            ys, xs, y_max = ys_i, xs_i, yb
    sel = ys >= y_max - 6
    return int(round(xs[sel].mean())) + u0, int(round(ys[sel].mean())) + v0


def detect_choco_px(
    hd_now: np.ndarray, prior_px: Sequence[float], search: int = 90
) -> Optional[Tuple[int, int]]:
    """前回位置の周りで、チョコの白い包装 (＋赤い文字) の小さなかたまりを探し、その中心画素を返す。"""
    if hd_now is None:
        return None
    h, w = hd_now.shape[:2]
    pu, pv = int(round(prior_px[0])), int(round(prior_px[1]))
    u0, u1 = max(0, pu - search), min(w, pu + search)
    v0, v1 = max(0, pv - search), min(h, pv + search)
    hsv = cv2.cvtColor(hd_now[v0:v1, u0:u1], cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    white = (V > 175) & (S < 70)
    red = ((H < 10) | (H > 170)) & (S > 110) & (V > 90)
    mask = (white | red).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(mask)
    best, best_d = None, 1e9
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        # チョコは 1px≈1mm で 25〜45px 角程度。白い前腕のような大きな領域は除く
        if area < 150 or area > 2600 or bw > 75 or bh > 75:
            continue
        cu, cv_ = float(cent[i][0]) + u0, float(cent[i][1]) + v0
        d = float(np.hypot(cu - pu, cv_ - pv))
        if d < best_d:
            best, best_d = (int(round(cu)), int(round(cv_))), d
    return best


def observe_hand_vs_choco(
    hd_now: np.ndarray,
    hd_ref: np.ndarray,
    grasp_pos: Sequence[float],
    pinch_axis_xy: Sequence[float],
    z_table: float,
    choco_prior_px: Sequence[float],
    choco_height_m: float,
    kin: R1Kinematics,
    head_pitch: float,
    head_yaw: float,
    cfg: SystemConfig,
    thumb_z: Optional[float] = None,
) -> HandChocoObservation:
    """HD 画像から、チョコを親指と人差指の間の中央に置くために手をずらす量を求める。

    z_table: チョコを置いている机面の高さ (胴体座標)。チョコの中心は z_table + 高さ/2 の面で求める。
    thumb_z: 親指の先の高さ。着地後は机面、ホバー中は空中の高さを渡す (省略時は z_table)。
    """
    a = np.asarray(pinch_axis_xy, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-9)
    perp = np.array([-a[1], a[0]], dtype=np.float64)
    reach = float(getattr(cfg, "hd_servo_thumb_reach_m", 0.030))
    tz = float(z_table if thumb_z is None else thumb_z)

    # 順運動学 (現在の目標) から見た親指の先の予測位置 → HD 画素
    thumb_pred_t = np.array([grasp_pos[0] - a[0] * reach, grasp_pos[1] - a[1] * reach, tz], dtype=np.float64)
    pred_px = torso_to_hd_px(thumb_pred_t, kin, head_pitch, head_yaw, cfg)
    if pred_px is None:
        return HandChocoObservation(valid=False, reason="親指の予測位置を画像へ投影できない")
    pred_t = (int(round(pred_px[0])), int(round(pred_px[1])))

    choco_px = detect_choco_px(hd_now, choco_prior_px)
    if choco_px is None:
        return HandChocoObservation(valid=False, reason="チョコを画像で見つけられない", thumb_pred_px=pred_t)

    excl = [(choco_px[0] - 30, choco_px[1] - 30, choco_px[0] + 30, choco_px[1] + 30)]
    pu, pv = int(round(choco_prior_px[0])), int(round(choco_prior_px[1]))
    excl.append((pu - 30, pv - 30, pu + 30, pv + 30))  # 事前確認時のチョコ位置 (動いていれば差分に出るため除く)
    thumb_px = detect_thumb_tip_px(hd_now, hd_ref, pred_px, exclude_boxes=excl)
    if thumb_px is None:
        return HandChocoObservation(valid=False, reason="親指の先を画像で見つけられない", thumb_pred_px=pred_t, choco_px=choco_px)

    thumb_xy = hd_px_to_torso_at_z(thumb_px[0], thumb_px[1], tz, kin, head_pitch, head_yaw, cfg)
    choco_xy = hd_px_to_torso_at_z(
        choco_px[0], choco_px[1], z_table + 0.5 * float(choco_height_m), kin, head_pitch, head_yaw, cfg
    )
    if thumb_xy is None or choco_xy is None:
        return HandChocoObservation(valid=False, reason="視線と机面の交点を求められない")

    desired_thumb = choco_xy[:2] - a * reach
    raw = desired_thumb - thumb_xy[:2]
    along = float(np.dot(raw, a))
    across = float(np.dot(raw, perp))
    # 把持軸方向の推定は親指の届き・高さの仮定に左右されやすいので控えめに動かす (残りは次の測定で詰める)
    gain_along = float(getattr(cfg, "hd_servo_along_gain", 0.6))
    shift = a * (along * gain_along) + perp * across
    shift_mm = float(np.hypot(raw[0], raw[1]) * 1000.0)
    obs = HandChocoObservation(
        valid=True,
        reason="OK",
        thumb_px=thumb_px,
        thumb_pred_px=pred_t,
        choco_px=choco_px,
        thumb_xy_m=[round(float(thumb_xy[0]), 4), round(float(thumb_xy[1]), 4)],
        choco_xy_m=[round(float(choco_xy[0]), 4), round(float(choco_xy[1]), 4)],
        shift_xy_m=[round(float(shift[0]), 4), round(float(shift[1]), 4)],
        along_mm=round(along * 1000.0, 1),
        across_mm=round(across * 1000.0, 1),
    )
    if shift_mm > float(getattr(cfg, "hd_servo_max_shift_mm", 90.0)):
        obs.valid = False
        obs.reason = f"ずれ {shift_mm:.0f}mm が大きすぎるため誤検出とみなす"
    return obs
