"""いまの姿勢のまま、HD とステレオで「手首の金具に対するチョコの位置」を数回測って中央値を出す (動かさない)。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision, stereo, cv2
b = Bot("measure"); rows = []
for k in range(7):
    b.pump(0.4); jpg = b.grab_hd(); st = b.stereo; hd = cv2.imdecode(np.frombuffer(jpg, np.uint8), 1); L = cv2.imdecode(np.frombuffer(st[0], np.uint8), 1); R = cv2.imdecode(np.frombuffer(st[1], np.uint8), 1)
    s3 = stereo.choco_vs_wrist(hd, L, R, b.rob["q_aux"][1], 0.0); rel = vision.choco_rel_hand(hd)
    if s3 and rel: rows.append(s3["rel"] + [s3["choco"]["depth_m"], s3["wrist"]["depth_m"], rel["du"], rel["dv"], rel["score"], s3["choco"]["score"]])
q = b.qR(); p = pinch(q); A = np.array(rows) if rows else np.zeros((0, 9)); med = np.median(A, axis=0) if len(A) else None
out = {"n": len(rows), "fk_pinch": [round(float(v), 4) for v in p], "rpy": b.rob["rpy"], "soc": b.rob["bms"]["soc"], "stage": b.rob["stage"]}
if med is not None:
    out.update(stereo_rel=[round(float(v), 4) for v in med[:3]], spread_mm=[round(float(v) * 1000, 1) for v in (A[:, :3].max(0) - A[:, :3].min(0))], choco_depth_m=round(float(med[3]), 3), wrist_depth_m=round(float(med[4]), 3), du=round(float(med[5]), 1), dv=round(float(med[6]), 1),
               wrist_above_taught_mm=round(1000 * (-float(med[2]) - 0.1008), 1), new_ground_z=round(float(p[2]) - (-float(med[2]) - 0.1008), 4))
print("MEASURE " + json.dumps(out), flush=True); b.close()
