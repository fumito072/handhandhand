"""頭の向き合わせ: 右腕を基準姿勢 LIFT_TOP に置き、手首の金具が教示時と同じ画素に写るまで首のヨー・ピッチを調整する。
電源を入れ直すと首ヨーの原点がずれるので、起動後に 1 回行う。最後に HEADCAL {...} を出す。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision, cv2, hd_hand_servo as hs
from config import ConfigStore
LIFT_TOP = [-0.743, -0.703, 0.982, 0.159, 1.85]
st = ConfigStore(); st.load(); cfg = st.cfg
b = Bot("calib_head")
if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
b.hand_set(OPEN, 900, 1.4)
if np.abs(np.array(b.rob["cmd_arm"][5:]) - np.array(LIFT_TOP)).max() > 0.02:
    q = np.array(b.qR())
    if pinch(q)[2] < -0.10 or q[3] > 1.0:
        for lab, w, spd in (("c_back_down", BACK_DOWN, 0.15), ("c_back_up", BACK_UP, 0.20), ("c_high", HIGH, 0.20)): b.move(w, lab, speed=spd)
    b.move(LIFT_TOP, "c_lift_top", speed=0.15)
b.pump(1.2)
wr, _ = fk(LIFT_TOP)                                     # 手首の位置での「首角 → 画素」の感度をカメラ模型から求める
def px(p, y): return hs.torso_to_hd_px(wr, kin, p, y, cfg)
gy = (px(0.638, 0.05) - px(0.638, -0.05))[0] / 0.10; gp = (px(0.688, 0.0) - px(0.588, 0.0))[1] / 0.10
b.say("model gains px/rad: yaw", round(float(gy)), "pitch", round(float(gp)))
def shift():
    b.grab_hd(); img = cv2.imdecode(np.frombuffer(b.hd, np.uint8), cv2.IMREAD_COLOR)
    res = cv2.matchTemplate(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), vision._TPL_H, cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(res)
    return loc[0] - vision._TPL_H_ORIGIN[0], loc[1] - vision._TPL_H_ORIGIN[1], float(mx)
hist = []; ok = False
for it in range(6):
    du, dv, sc = shift(); aux = list(b.rob["q_aux"]); hist.append({"it": it, "du": du, "dv": dv, "score": round(sc, 3), "aux": [round(v, 4) for v in aux]}); b.say(hist[-1])
    if sc < 0.75: b.say("template not found"); break
    if abs(du) <= 3 and abs(dv) <= 3: ok = True; break
    dyaw = float(np.clip(-du / gy, -0.2, 0.2)); dpitch = float(np.clip(-dv / gp, -0.08, 0.08))
    if it > 0 and len(hist) >= 2:                          # 実測の感度で補正 (模型の感度が違っていても収束させる)
        p0, p1 = hist[-2], hist[-1]; dy_done = p1["aux"][2] - p0["aux"][2]
        if abs(dy_done) > 0.01 and abs(p1["du"] - p0["du"]) > 3: gy = (p1["du"] - p0["du"]) / dy_done; dyaw = float(np.clip(-du / gy, -0.2, 0.2))
        dp_done = p1["aux"][1] - p0["aux"][1]
        if abs(dp_done) > 0.005 and abs(p1["dv"] - p0["dv"]) > 2: gp = (p1["dv"] - p0["dv"]) / dp_done; dpitch = float(np.clip(-dv / gp, -0.08, 0.08))
    b.move_head([aux[0], aux[1] + dpitch, aux[2] + dyaw], f"head_cal_{it}", speed=0.10); b.pump(1.0)
b.save_hd("headcal_final"); out = {"ok": ok, "aux": [round(v, 4) for v in b.rob["q_aux"]], "hist": hist, "gain_yaw": round(float(gy)), "gain_pitch": round(float(gp))}
b.mark("head_calibration", **out); json.dump(out, open(OUT + "/head_calibration.json", "w"), indent=1); print("HEADCAL " + json.dumps({k: out[k] for k in ("ok", "aux", "gain_yaw", "gain_pitch")}), flush=True); b.close()
