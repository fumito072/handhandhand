"""チョコを持ったまま、手を少しずつ下ろして机に触れたところで止める。usage: lower_until_touch.py [step_mm=4] [max_steps=6]"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *
step = float(sys.argv[1]) / 1000.0 if len(sys.argv) > 1 else 0.004; nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 6
b = Bot("lower_touch")
if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
h0 = b.hand(); q0 = np.array(b.qR()); _, Rn = fk(q0); p0 = pinch(q0); tau0 = np.array(b.rob["tau_ext"][5:]); why = None
b.say("start pinch", np.round(p0, 4).tolist(), "hand", h0)
for k in range(nmax):
    q = np.array(b.qR()); off = np.clip(np.array(b.rob["cmd_arm"][5:]) - q, -0.12, 0.12); pz_before = pinch(q)[2]
    qn, pe, re = ik(pinch(q) + np.array([0, 0, -max(step, 0.006)]), Rn, q)            # 摩擦で動かないのを避けるため、指令は 6mm 以上
    b.move((np.array(qn) + off).tolist(), f"lt{k}", speed=0.04); t0 = time.time(); z0 = pinch(b.qR())[2]
    while time.time() - t0 < 5:
        b.pump(0.6); z1 = pinch(b.qR())[2]
        if abs(z1 - z0) < 0.0003 and time.time() - t0 > 1.2: break
        z0 = z1
    h = b.hand(); dz = pinch(b.qR())[2] - pz_before; dtau = np.array(b.rob["tau_ext"][5:]) - tau0
    b.say(f"step {k+1}: went down {-dz*1000:.1f}mm (total {-(pinch(b.qR())[2]-p0[2])*1000:.1f}mm) | idx Fn {h['Fn'][1]:.2f} Ft {h['Ft'][1]:.2f} prox {h['prox_k'][1]} | thumb Fn {h['Fn'][0]:.2f} prox {h['prox_k'][0]} cur {h['cur'][0]} | dtau max {np.abs(dtau).max():.2f}")
    if abs(h["Fn"][1] - h0["Fn"][1]) > 0.4 or abs(h["Ft"][1] - h0["Ft"][1]) > 0.4 or abs(h["Fn"][0] - h0["Fn"][0]) > 0.4: why = "finger force changed (touching the table)"; break
    if -dz < 0.0015: why = "stopped going down"; break
b.send("arm_hold_here"); b.pump(0.5); b.save_hd("lowered"); b.mark("lower_until_touch", why=why, **b.snap())
print("LOWER " + json.dumps({"why": why or "step limit", "total_down_mm": round(float(-(pinch(b.qR())[2] - p0[2]) * 1000), 1), "qR": [round(v, 4) for v in b.qR()], "hand": b.hand()}), flush=True); b.close()
