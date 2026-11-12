"""教え直し (持たせてから置く方式)。1 本の接続で最後まで行うので、指の保持は切れない。
1. 手を開いて 6cm 上げる。 2. 人がチョコを親指と人差指の間に差し入れる → 近接値で検知して閉じる。
3. フリー移動: 人がチョコを持たせたまま手を机へ下ろし、3 秒止めると固定。 4. 記録 → 指を開く → 開いた状態を記録 → 6cm 上でも記録。
結果: docs/data/teach_feet_down_20261112/teach_hold.json。最後に TEACH {...}。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision, stereo, cv2
D = str(ROOT / "docs/data/teach_feet_down_20261112"); b = Bot("teach_hold")
if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
if not b.rob.get("engaged"): sys.exit("not engaged")
def shot(tag):
    b.pump(0.5); jpg = b.grab_hd(); st = b.stereo; out = {}
    if jpg: open(f"{D}/{tag}_hd.jpg", "wb").write(jpg)
    if st: open(f"{D}/{tag}_L.jpg", "wb").write(st[0]); open(f"{D}/{tag}_R.jpg", "wb").write(st[1])
    rows = []
    for k in range(7):
        b.pump(0.35); j2 = b.grab_hd(); s2 = b.stereo
        if not j2 or not s2: continue
        hd = cv2.imdecode(np.frombuffer(j2, np.uint8), 1); s3 = stereo.choco_vs_wrist(hd, cv2.imdecode(np.frombuffer(s2[0], np.uint8), 1), cv2.imdecode(np.frombuffer(s2[1], np.uint8), 1), b.rob["q_aux"][1], 0.0); rel = vision.choco_rel_hand(hd)
        if s3 and rel: rows.append(s3["rel"] + [s3["choco"]["depth_m"], s3["choco"]["score"], s3["tpl_score"], rel["choco"]["u"], rel["choco"]["v"], rel["hand_shift"][0], rel["hand_shift"][1]])
    if len(rows) >= 3:
        A = np.array(rows); med = np.median(A, axis=0); out = {"stereo_rel": [round(float(v), 4) for v in med[:3]], "spread_mm": [round(float(v) * 1000, 1) for v in (A[:, :3].max(0) - A[:, :3].min(0))], "choco_depth_m": round(float(med[3]), 3), "choco_score": round(float(med[4]), 3), "tpl_score": round(float(med[5]), 3), "choco_px": [round(float(med[6]), 1), round(float(med[7]), 1)], "hand_shift_px": [int(med[8]), int(med[9])], "n": len(rows)}
    q = b.qR(); wr, R = fk(q); out.update(tag=tag, q_right=[round(v, 4) for v in q], cmd_right=[round(v, 4) for v in b.rob["cmd_arm"][5:]], wrist_pos=[round(float(v), 4) for v in wr], wrist_R=np.round(R, 4).tolist(), rpy=b.rob["rpy"], q_aux=b.rob["q_aux"], hand=b.hand()); return out
res = {}; rot = b.hand()["pos"][1]; open_shape = [402, rot, 49, 846, 847, 846]
if "held" in sys.argv:                                      # すでにチョコを持っている → 閉じる手順を省く
    hc = b.hand(); res.update(hand_closed=hc, held_in_air=True); b.say("already holding:", hc)
else:
    b.hand_set(open_shape, 900, 1.5); b.say("Hold the chocolate between thumb and index. Closing starts in 5 s."); b.pump(5.0); h = b.hand()
    tgt = list(open_shape); done = {}; CAP = {0: 690, 2: 250}; log = []
    for n in range(60):
        for mtr in (0, 2):
            if mtr not in done:
                if tgt[mtr] + 6 > CAP[mtr]: done[mtr] = "cap"
                else: tgt[mtr] += 6
        if len(done) == 2: break
        h = b.hand_set(tgt, 400, 0.55); log.append({"cmd": [tgt[0], tgt[2]], **h})
        if 2 not in done:
            if h["Fn"][1] >= 1.5: done[2] = f"index force {h['Fn'][1]:.2f}N at {h['pos'][2]}"
            elif abs(h["cur"][2]) >= 350: done[2] = f"index current {h['cur'][2]} at {h['pos'][2]}"
            elif tgt[2] - h["pos"][2] > 30: done[2] = f"index stall at {h['pos'][2]}"
        if 0 not in done:
            if h["Fn"][0] >= 1.5: done[0] = f"thumb force {h['Fn'][0]:.2f}N at {h['pos'][0]}"
            elif abs(h["cur"][0]) >= 400: done[0] = f"thumb current {h['cur'][0]} at {h['pos'][0]}"
            elif tgt[0] - h["pos"][0] > 30: done[0] = f"thumb stall at {h['pos'][0]}"
        if n % 4 == 0 or len(done): b.say(f"close {n+1:2d} thumb {tgt[0]}/{h['pos'][0]} cur {h['cur'][0]} Fn {h['Fn'][0]:.2f} prox {h['prox_k'][0]} | index {tgt[2]}/{h['pos'][2]} cur {h['cur'][2]} Fn {h['Fn'][1]:.2f} prox {h['prox_k'][1]} | {done}")
    b.pump(1.0); hc = b.hand(); held = hc["prox_k"][1] > 600 or hc["prox_k"][0] > 400 or hc["Fn"][1] > 1.0
    res.update(close_done={str(k): v for k, v in done.items()}, close_log=log, held_in_air=bool(held), hand_closed=hc); b.say("closed:", done, hc, "held =", held)
    if not held: print("TEACH " + json.dumps({"ok": False, "why": "fingers closed but nothing is held", "hand": hc}), flush=True); b.close(); sys.exit(1)
# --- 3. フリー移動: 机へ下ろして 3 秒止める ---
b.send("arm_set_params", grav_ff_scale=1.0, extra_hand_mass_kg=0.30); b.pump(0.3); b.say("FREE-DRIVE: lower the hand until the chocolate rests on the table, then hold still 3 s (max 180 s)")
t0 = time.time(); q_start = np.array(b.qR()); hist = []; locked = "grounded" in sys.argv     # すでに机に置いてある → フリー移動を省く
while not locked and time.time() - t0 < 180:
    b.send("arm_hold_here"); b.pump(0.38); q = np.array(b.qR()); hist.append((time.time(), q)); hist = [x for x in hist if time.time() - x[0] <= 3.2]
    if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
    moved = float(np.abs(q - q_start).max()); still = float(np.abs(np.array([x[1] for x in hist]) - q).max()) if len(hist) >= 6 else 9.0
    if moved > 0.02 and still < 0.008 and pinch(q)[2] < 0.06 and time.time() - t0 > 8: locked = True; break
b.send("arm_hold_here"); b.pump(0.6); b.send("arm_set_params", grav_ff_scale=0.85, extra_hand_mass_kg=0.16); b.pump(0.4)
if not locked: print("TEACH " + json.dumps({"ok": False, "why": "hand was not brought down to the table and held still"}), flush=True); b.close(); sys.exit(1)
b.say("LOCKED at", [round(v, 3) for v in b.qR()], "pinch", np.round(pinch(b.qR()), 3).tolist(), "- hands off now"); b.pump(2.0)
res["grounded_closed"] = shot("h1_grounded_closed"); hg = b.hand(); b.say("grounded (closed):", hg)
# --- 4. 開いて記録 ---
steps = []; tgt = list(hg["pos"]); tgt[3], tgt[4], tgt[5] = 846, 847, 846
while tgt[0] > 402 or tgt[2] > 49:
    tgt[0] = max(402, tgt[0] - 6); tgt[2] = max(49, tgt[2] - 6); h = b.hand_set(tgt, 400, 0.45); steps.append({"cmd": [tgt[0], tgt[2]], **h})
b.pump(1.0); res["release_steps"] = steps; res["released_open"] = shot("h2_released_open"); b.say("released (open):", {k: res["released_open"].get(k) for k in ("stereo_rel", "spread_mm", "choco_px", "hand_shift_px", "q_right")})
q_g = np.array(b.qR()); cmd_g = np.array(b.rob["cmd_arm"][5:]); _, Rn = fk(q_g)
for dz, tag in ((0.03, "h3_above_3cm"), (0.06, "h4_above_6cm")):
    qn, pe, re = ik(pinch(q_g) + np.array([0, 0, dz]), Rn, q_g); b.move((np.array(qn) + (cmd_g - q_g)).tolist(), "th_" + tag, speed=0.05); b.pump(1.5); res[tag] = shot(tag); b.say(tag, {k: res[tag].get(k) for k in ("stereo_rel", "spread_mm", "choco_px")})
res.update(grasp_q_right=[round(float(v), 4) for v in q_g], grasp_cmd_right=[round(float(v), 4) for v in cmd_g], open_shape=open_shape, grasp_shape=hg["pos"], soc=b.rob["bms"]["soc"], ok=True)
json.dump(res, open(D + "/teach_hold.json", "w"), ensure_ascii=False, indent=1); b.mark("teach_hold_result", grasp_q=res["grasp_q_right"], grasp_shape=hg["pos"])
print("TEACH " + json.dumps({"ok": True, "grasp_q_right": res["grasp_q_right"], "grasp_shape": hg["pos"], "open_shape": open_shape, "released_rel": res["released_open"].get("stereo_rel"), "released_spread_mm": res["released_open"].get("spread_mm"), "soc": res["soc"]}, ensure_ascii=False), flush=True); b.close()
