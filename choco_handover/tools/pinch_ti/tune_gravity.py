"""重力補償の合わせ込み: 空中の基準姿勢で、補償の強さと手の質量の設定を変えながら「指令と実測のずれ」を測り、最も小さい組を選ぶ。
腕は基準姿勢 (HIGH と、机の上空の姿勢) に置くだけで、設定を変えるたびに 3 秒待って測る。最後に GRAV {...} を出し、gravity_params.json に保存する。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *
b = Bot("tune_gravity")
if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
if not b.rob.get("engaged") and not b.engage(): sys.exit("engage failed")
HOVER, _, _ = ik_tilt(pinch(GROUND) + np.array([0.0, 0.0, 0.06]), GROUND)
def settle(maxs=6.0):
    t0 = time.time(); q0 = np.array(b.qR())
    while time.time() - t0 < maxs:
        b.pump(0.7); q1 = np.array(b.qR())
        if np.abs(q1 - q0).max() < 0.0015 and time.time() - t0 > 1.5: break
        q0 = q1
def err(): return np.array(b.rob["cmd_arm"][5:]) - np.array(b.qR())
q = np.array(b.qR())
if pinch(q)[2] < -0.10 or q[3] > 1.0:
    for lab, w in (("t_back_down", BACK_DOWN), ("t_back_up", BACK_UP)): b.move(w, lab, speed=0.18)
b.move(HIGH, "t_high", speed=0.18)
results = []
for scale, mass in ((0.85, 0.16), (1.0, 0.16), (1.0, 0.30), (1.0, 0.45), (1.0, 0.60)):
    b.send("arm_set_params", grav_ff_scale=scale, extra_hand_mass_kg=mass); b.pump(0.5); row = {"scale": scale, "mass": mass}
    for nm, pose in (("high", HIGH), ("hover", HOVER)):
        b.move(pose, f"t_{nm}", speed=0.15); settle(); e = err(); p_cmd = pinch(b.rob["cmd_arm"][5:]); p_meas = pinch(b.qR())
        row[nm] = {"joint_err": np.round(e, 3).tolist(), "pinch_err_mm": round(float(np.linalg.norm(p_cmd - p_meas) * 1000), 1), "drop_mm": round(float((p_cmd - p_meas)[2] * 1000), 1), "tau_ext": [round(v, 2) for v in b.rob["tau_ext"][5:]]}
    results.append(row); b.say(row)
best = min(results, key=lambda r: r["high"]["pinch_err_mm"] + r["hover"]["pinch_err_mm"])
b.send("arm_set_params", grav_ff_scale=best["scale"], extra_hand_mass_kg=best["mass"]); b.pump(0.5)
json.dump({"grav_ff_scale": best["scale"], "extra_hand_mass_kg": best["mass"], "results": results}, open(OUT + "/gravity_params.json", "w"), indent=1)
b.mark("gravity_tuned", best=best); print("GRAV " + json.dumps({"scale": best["scale"], "mass": best["mass"], "pinch_err_mm": [best["high"]["pinch_err_mm"], best["hover"]["pinch_err_mm"]], "soc": b.rob["bms"]["soc"]}), flush=True); b.close()
