"""今の手の位置のまま、親指と人差指を同じ量ずつ閉じ、持ち上げて、指に物があるかを確かめる。usage: close_lift.py [name]"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *
name = sys.argv[1] if len(sys.argv) > 1 else "cl"; b = Bot("close_" + name)
if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
b.save_hd(name + "_before"); h0 = b.hand(); b.say("before", h0, "soc", b.rob["bms"]["soc"])
tgt = list(OPEN); tgt[0] = max(OPEN[0], min(h0["pos"][0], OPEN[0])); done = {}; CAP = {0: 690, 2: 250}; log = []
b.hand_set(OPEN, 600, 1.0)
for n in range(48):
    for mtr in (0, 2):
        if mtr not in done:
            if tgt[mtr] + 8 > CAP[mtr]: done[mtr] = "cap"
            else: tgt[mtr] += 8
    if len(done) == 2: break
    h = b.hand_set(tgt, 400, 0.5); log.append({"cmd": [tgt[0], tgt[2]], **h})
    if 2 not in done:
        if h["Fn"][1] >= 1.5 and h["prox_k"][1] > 300: done[2] = f"index force {h['Fn'][1]:.2f}N at {h['pos'][2]}"
        elif abs(h["cur"][2]) >= 330: done[2] = f"index current {h['cur'][2]} at {h['pos'][2]}"
        elif tgt[2] - h["pos"][2] > 25: done[2] = f"index stall at {h['pos'][2]}"
    if 0 not in done:
        if abs(h["cur"][0]) >= 380: done[0] = f"thumb current {h['cur'][0]} at {h['pos'][0]}"
        elif 2 in done and h["prox_k"][0] >= 350 and abs(h["cur"][0]) >= 150: done[0] = f"thumb proximity {h['prox_k'][0]}k + current {h['cur'][0]} at {h['pos'][0]}"
        elif tgt[0] - h["pos"][0] > 25: done[0] = f"thumb stall at {h['pos'][0]}"
    if n % 4 == 0 or len(done) == 1 or h["prox_k"][0] > 250: b.say(f"close {n+1:2d} thumb {tgt[0]}/{h['pos'][0]} cur {h['cur'][0]} prox {h['prox_k'][0]} | index {tgt[2]}/{h['pos'][2]} cur {h['cur'][2]} Fn {h['Fn'][1]:.2f} prox {h['prox_k'][1]} | {done}")
b.pump(0.8); h = b.hand(); b.say("closed", done, h); b.save_hd(name + "_closed")
q = np.array(b.qR()); off = np.clip(np.array(b.rob["cmd_arm"][5:]) - q, -0.12, 0.12); _, Rn = fk(q)
for k in (1, 2):
    qn, pe, re = ik(pinch(b.qR()) + np.array([0, 0, 0.03]), Rn, b.qR()); b.say("lift", k, b.move((np.array(qn) + off).tolist(), f"cl_lift{k}", speed=0.05), "ik err", round(pe, 1), round(re, 1)); b.pump(1.0)
h2 = b.hand(); b.save_hd(name + "_lifted"); ok = (h2["prox_k"][1] > 600 and (h2["Fn"][1] - h0["Fn"][1]) > 0.5) or (h2["prox_k"][1] > 1500 and h2["prox_k"][0] > 300)
b.mark("close_lift_result", ok=bool(ok), done={str(k): v for k, v in done.items()}, hand=h2, log=log); print("RESULT " + json.dumps({"ok": bool(ok), "done": {str(k): v for k, v in done.items()}, "hand": h2, "soc": b.rob["bms"]["soc"]}), flush=True); b.close()
