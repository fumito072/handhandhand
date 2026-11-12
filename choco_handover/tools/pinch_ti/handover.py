"""掴んだチョコを前へ差し出して人に渡す。usage: handover.py [name] [wait_s=25]
1. いまの持ち上げ位置から、前へ 10cm・上へ 12cm の位置へゆっくり差し出す。
2. 受け取られるのを待つ: 人差指の指腹の力・横方向の力・近接値が変わったら (引かれた/触れられた)、指を開いて渡す。
3. 渡したら手を開いたまま 6cm 上の位置へ戻る。時間内に受け取られなければ、教示位置へ戻ってチョコを机に置き直す。最後に HANDOVER {...}。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision
T = json.load(open(ROOT / "docs/data/teach_feet_down_20261112/teach_hold.json")); Q_G = np.array(T["released_open"]["q_right"]); OPEN_T = T["open_shape"]; _, R_G = fk(Q_G); P_G = pinch(Q_G)
name = sys.argv[1] if len(sys.argv) > 1 else "ho"; WAIT = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
b = Bot("handover_" + name); res = {"trial": name}
if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
h0 = b.hand()
if not (h0["prox_k"][1] > 500 or h0["prox_k"][0] > 250): print("HANDOVER " + json.dumps({"ok": False, "why": "nothing is held", "hand": h0}), flush=True); b.close(); sys.exit(1)
def go(d, label, speed=0.06):
    q = np.array(b.qR()); off = np.clip(np.array(b.rob["cmd_arm"][5:]) - q, -0.12, 0.12); qs = Q_G
    for a in np.linspace(0.25, 1, 4): qs, pe, re = ik(P_G + a * np.asarray(d, dtype=float), R_G, qs)
    r = b.move((np.array(qs) + off * 0).tolist(), label, speed=speed); b.pump(1.2); return r
r = go([0.05, 0.0, 0.10], "ho_mid"); r = go([0.10, 0.0, 0.12], "ho_out"); b.pump(1.0); base = b.hand(); b.save_hd(name + "_offered")
b.say("offered. waiting for someone to take it", base, "pinch", np.round(pinch(b.qR()), 3).tolist())
t0 = time.time(); taken = None; hist = []
while time.time() - t0 < WAIT:
    b.pump(0.15); h = b.hand(); hist.append({"t": round(time.time() - t0, 2), "Fn": h["Fn"][1], "Ft": h["Ft"][1], "prox": h["prox_k"][1], "thumb_prox": h["prox_k"][0]})
    if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
    if time.time() - t0 < 1.5: base = h; continue
    if abs(h["Ft"][1] - base["Ft"][1]) > 0.8: taken = f"sideways force changed {base['Ft'][1]:.2f}->{h['Ft'][1]:.2f}N"
    elif abs(h["Fn"][1] - base["Fn"][1]) > 1.0: taken = f"grip force changed {base['Fn'][1]:.2f}->{h['Fn'][1]:.2f}N"
    elif h["prox_k"][1] < 0.5 * base["prox_k"][1] and base["prox_k"][1] > 400: taken = f"index proximity dropped {base['prox_k'][1]}k->{h['prox_k'][1]}k"
    if taken: break
res.update(taken=taken, wait_log=hist[-40:])
if taken:
    b.say("TAKE detected:", taken, "-> opening"); b.hand_set(OPEN_T, 500, 1.2); b.pump(1.0); b.save_hd(name + "_released"); go([0.0, 0.0, 0.06], "ho_back"); res["ok"] = True
else:
    b.say("nobody took it -> putting it back on the table"); go([0.05, 0.0, 0.10], "ho_ret_mid"); go([0.0, 0.0, 0.06], "ho_ret_6"); go([0.0, 0.0, 0.03], "ho_ret_3", 0.05); go([0.0, 0.0, 0.004], "ho_ret_0", 0.04)
    hg = b.hand(); tgt = list(hg["pos"])
    while tgt[0] > OPEN_T[0] or tgt[2] > OPEN_T[2]: tgt[0] = max(OPEN_T[0], tgt[0] - 6); tgt[2] = max(OPEN_T[2], tgt[2] - 6); b.hand_set(tgt, 400, 0.4)
    go([0.0, 0.0, 0.06], "ho_up_after_place"); res["ok"] = False; res["why"] = "not taken (placed back)"
b.mark("handover_result", **res); print("HANDOVER " + json.dumps({"ok": res["ok"], "taken": taken, "why": res.get("why"), "soc": b.rob["bms"]["soc"]}, ensure_ascii=False), flush=True); b.close()
