"""カメラでチョコの位置を捕捉する。右手を体側へ引いた観察姿勢 (BACK_UP) に置いて机の上を空け、
頭部 HD 画像のチョコの画素から、教示した接地位置からのずれ (dx, dy) [m] を求める。最後に LOCATE {...} を出す。
前提: calib_head.py で頭の向きを合わせてあること。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision, cv2
name = sys.argv[1] if len(sys.argv) > 1 else "loc"; C_REF = (635.6, 363.4)
BIAS_X, BIAS_Y = -0.017, -0.017   # 観察姿勢と掴む姿勢の差 [m]。試行 t11 の着地画像 (手首の金具と、捕捉したチョコ画素の関係) から学習
b = Bot("locate_" + name)
if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
fresh = not b.rob.get("engaged")                        # 制御が入っていなかった (脱力していた) → 腕は机の手前に垂れているとみなす
if fresh and not b.engage(): sys.exit("engage failed")
wrG, RG = fk(GROUND); q_obs, pe, re = ik(pinch(GROUND) + np.array([-0.06, 0.0, 0.10]), RG, [-0.743, -0.703, 0.982, 0.159, 1.85])
h = b.hand()
if min(h["pos"][3:]) < 780 or abs(h["pos"][1] - 590) > 25:
    b.hand_set([h["pos"][0], h["pos"][1], h["pos"][2], 49, 49, 49], 1000, 1.6); b.hand_set([402, h["pos"][1], 49, 847, 847, 847], 1500, 2.2)
b.hand_set(OPEN, 900, 1.4)
q = np.array(b.qR())
if fresh or pinch(q)[2] < -0.10 or q[3] > 1.0:                         # 家の姿勢付近・脱力後 → 体側へ引いて下ろしてから持ち上げる
    if b.move(BACK_DOWN, "l_back_down", speed=0.15) != "done": sys.exit("stopped on the way up")
else:                                                                   # 机の上にいる → 真上へ逃がしてから体側へ引く
    if pinch(q)[2] < pinch(GROUND)[2] + 0.07:
        qq, _, _ = ik(pinch(q) + np.array([0, 0, 0.09]), RG, q); b.move(qq, "l_up_first", speed=0.10)
    b.move(HIGH, "l_high", speed=0.15)
b.move(BACK_UP, "l_observe_pose", speed=0.18); b.pump(1.5)             # 観察姿勢: 手を体側へ引き、机の上を空ける
obs = []
for k in range(4):
    b.save_hd(f"{name}_obs{k}"); c = vision.find_in_jpg(b.hd) if b.hd else None
    if c: obs.append(c)
    b.pump(0.5)
out = None
if len(obs) >= 2:
    u = float(np.median([c["u"] for c in obs])); v = float(np.median([c["v"] for c in obs]))
    out = {"dx": round(vision.MM_PER_PX_X * (v - C_REF[1]) / 1000.0 + BIAS_X, 4), "dy": round(vision.MM_PER_PX_Y * (u - C_REF[0]) / 1000.0 + BIAS_Y, 4), "u": round(u, 1), "v": round(v, 1), "n": len(obs), "size": [obs[-1]["w"], obs[-1]["h"]]}
b.move(HIGH, "l_ready_high", speed=0.18)
b.mark("locate", result=out, **b.snap()); print("LOCATE " + json.dumps(out), flush=True); b.close()
