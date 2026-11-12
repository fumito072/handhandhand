"""人の手のひらを探す (既存の open_palm.OpenPalmDetector を使用)。HD で検出し、ステレオで 3D 位置を出す。
find(bot) -> {"px":[u,v], "torso":[x,y,z] (ステレオ・カメラ模型による胴体座標), "open":bool, "palm_up":bool, "score":..} または None
単体実行: palm.py [seconds]  … 動かさずに検出結果を表示し、注釈つき画像を保存する。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import cv2
from config import ConfigStore
from stereo_depth import StereoProcessor
from open_palm import OpenPalmDetector
_st = ConfigStore(); _st.load(); CFG = _st.cfg; _det = None; _sp = None
def find(b, save=None):
    global _det, _sp
    if _det is None: _det = OpenPalmDetector(); _sp = StereoProcessor()
    jpg = b.grab_hd(); st = b.stereo
    if not jpg or not st: return None
    hd = cv2.imdecode(np.frombuffer(jpg, np.uint8), 1); L = cv2.imdecode(np.frombuffer(st[0], np.uint8), 1); R = cv2.imdecode(np.frombuffer(st[1], np.uint8), 1)
    palms = _det.detect(hd, _sp.compute(L, R, CFG), kin, b.rob["q_aux"][1], 0.0, CFG); best = None
    for p in palms:
        cand = {"px": p.hd_px, "torso": p.center_torso_m, "open": bool(p.is_open), "palm_up": bool(p.palm_up), "score": p.score, "label": p.label, "open_fingers": p.open_fingers, "depth_points": p.depth_points}
        if p.center_torso_m is not None and (best is None or p.score > best["score"]): best = cand
        elif best is None: best = cand
    if save:
        for p in palms:
            for (x, y) in p.landmarks_hd or []: cv2.circle(hd, (x, y), 3, (0, 255, 0), -1)
            cv2.circle(hd, tuple(p.hd_px), 14, (0, 0, 255), 2)
        cv2.imwrite(save, hd)
    return best
if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0; b = Bot("palm_test"); t0 = time.time(); n = 0; hits = 0
    while time.time() - t0 < secs:
        r = find(b, save=f"{OUT}/_palm_test.jpg" if n % 3 == 0 else None); n += 1; hits += bool(r)
        print(f"t={time.time()-t0:4.1f}s", r, flush=True); b.pump(0.3)
    print(f"PALM detected in {hits}/{n} frames; wrist FK {np.round(fk(b.qR())[0],3).tolist()} pinch {np.round(pinch(b.qR()),3).tolist()}", flush=True); b.close()
