"""ステレオ (2 眼, 544x448, 基線 50mm) で奥行きを測る。HD 画像で見つけた点を右ステレオ画像へ写し、左画像との視差から 3D 位置を出す。
チョコと手首の金具の両方を測り、「手首から見たチョコの位置 (前後・左右・上下)」を教示時と比べる。"""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "choco_handover"))
import cv2, numpy as np
from config import ConfigStore
from choco_perception import hd_to_right_rect_px
import kinematics as K
import vision
_st = ConfigStore(); _st.load(); cfg = _st.cfg; kin = K.R1Kinematics()
def disparity_at(L, R, ur, vr, half=7, dmin=4, dmax=40):
    """右画像の (ur,vr) 周りの小片を、左画像の同じ行で探して視差 [px] を返す (サブピクセル)。"""
    gl = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY).astype(np.float32); gr = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY).astype(np.float32)
    u, v = int(round(ur)), int(round(vr)); h, w = gr.shape
    if not (half <= v < h - half and half <= u < w - half - dmax): return None
    tpl = gr[v - half:v + half + 1, u - half:u + half + 1]; strip = gl[v - half:v + half + 1, u - half + dmin:u + half + 1 + dmax]
    res = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)[0]; i = int(np.argmax(res)); sc = float(res[i]); sub = 0.0
    if 0 < i < len(res) - 1:
        a, b0, c = res[i - 1], res[i], res[i + 1]; den = a - 2 * b0 + c
        if abs(den) > 1e-6: sub = 0.5 * (a - c) / den
    return dmin + i + sub + (ur - u), sc
def point3d(L, R, u_hd, v_hd, head_pitch, head_yaw, half=7):
    """HD 画素の点の 3D 位置 (胴体座標 [m]) と視差・一致度。"""
    ur, vr = hd_to_right_rect_px(np.array([[u_hd, v_hd]], dtype=np.float64)); ur, vr = float(ur[0]), float(vr[0])
    out = disparity_at(L, R, ur, vr, half=half)
    if out is None: return None
    d, sc = out
    if d <= 1.0: return None
    z = cfg.fx * cfg.baseline_m / d; pc = np.array([cfg.baseline_m + z * (ur - cfg.cx) / cfg.fx, z * (vr - cfg.cy) / cfg.fy, z, 1.0])
    T = kin.camera_to_torso_matrix(head_pitch, head_yaw); pt = (T @ pc)[:3]
    return {"p": pt, "disp": round(float(d), 2), "score": round(sc, 3), "depth_m": round(float(z), 3)}
def choco_vs_wrist(hd, L, R, head_pitch=0.633, head_yaw=0.0):
    """手首の金具から見たチョコの 3D 相対位置 [m] (胴体座標: 前, 左, 上)。"""
    c = vision.find_choco(hd)
    if c is None or vision._TPL is None: return None
    res = cv2.matchTemplate(cv2.cvtColor(hd, cv2.COLOR_BGR2GRAY), vision._TPL, cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(res)
    wu, wv = loc[0] + vision._TPL.shape[1] / 2.0, loc[1] + vision._TPL.shape[0] / 2.0
    pc = point3d(L, R, c["u"], c["v"], head_pitch, head_yaw, half=5); pw = point3d(L, R, wu, wv, head_pitch, head_yaw, half=8)
    if pc is None or pw is None: return None
    return {"rel": [round(float(v), 4) for v in (pc["p"] - pw["p"])], "choco": {"p": [round(float(v), 3) for v in pc["p"]], "disp": pc["disp"], "score": pc["score"], "depth_m": pc["depth_m"]},
            "wrist": {"p": [round(float(v), 3) for v in pw["p"]], "disp": pw["disp"], "score": pw["score"], "depth_m": pw["depth_m"]}, "tpl_score": round(float(mx), 3)}
if __name__ == "__main__":
    D = str(ROOT / "docs/data/place_test_20261112/")
    hd = cv2.imread(D + "/pre_open_grounded_hd.jpg"); L = cv2.imread(D + "/pre_open_grounded_stereo_left.jpg"); R = cv2.imread(D + "/pre_open_grounded_stereo_right.jpg")
    print("taught reference (hand grounded, holding):", choco_vs_wrist(hd, L, R, 0.638, 0.0))
    A = str(ROOT / "docs/data/auto_grasp_20261112/")
    hd = cv2.imread(A + "/_stereo_probe_hd.jpg"); L = cv2.imread(A + "/_stereo_now_L.png"); R = cv2.imread(A + "/_stereo_now_R.png")
    print("now:", choco_vs_wrist(hd, L, R))
