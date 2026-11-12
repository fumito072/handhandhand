"""足を着けた高さでの教え直し。
1. フリー移動: 腕が手で動かせる状態になる (その場保持を毎秒 2〜3 回かけ直す)。人が右手を、机の上のチョコが親指と人差指の間に入る位置・向きへ持っていく。
   手を 3 秒間止めると、その姿勢で固定する。
2. 指を閉じ (親指と人差指を同じ量ずつ)、3cm 持ち上げて、触覚で持てているかを確かめる。
3. 下ろして指を開き、開いた直後 (チョコが見える状態) の HD・ステレオ画像と関節角を記録する。さらに 6cm 上でも記録する。
結果は docs/data/teach_feet_down_20261112/teach.json。最後に TEACH {...} を出す。
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision, stereo, cv2
D = str(ROOT / "docs/data/teach_feet_down_20261112"); FREE_MAX = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
KEEP = "keep" in sys.argv        # 持ち上げたあと離さずに保持して終わる
PLACED = "placed" in sys.argv    # 手はすでに置いてある (フリー移動を省く)
b = Bot("teach")
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
# --- 1. フリー移動 ---
b.send("arm_set_params", grav_ff_scale=1.0, extra_hand_mass_kg=0.30); b.pump(0.3)
b.say(f"FREE-DRIVE: move the right hand to the chocolate. Hold still 3 s to lock. (max {FREE_MAX:.0f}s)")
t0 = time.time(); q_start = np.array(b.qR()); hist = []; locked = PLACED
while not PLACED and time.time() - t0 < FREE_MAX:
    b.send("arm_hold_here"); b.pump(0.38); q = np.array(b.qR()); hist.append((time.time(), q)); hist = [h for h in hist if time.time() - h[0] <= 3.2]
    if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
    moved = float(np.abs(q - q_start).max()); still = float(np.abs(np.array([h[1] for h in hist]) - q).max()) if len(hist) >= 6 else 9.0
    if int(time.time() - t0) % 5 == 0 and len(hist) % 3 == 0: b.say(f"  t={time.time()-t0:5.1f}s moved {moved:.2f}rad still {still:.3f} near {max(b.hand()['prox_k'][:2]) >= 150} pinch {np.round(pinch(q),3).tolist()} thumb/idx prox_k {b.hand()['prox_k'][:2]}")
    hh = b.hand(); near = max(hh["prox_k"][0], hh["prox_k"][1]) >= 150 or max(hh["Fn"][0], hh["Fn"][1]) >= 0.3     # 親指か人差指の指腹がチョコを感じている
    if near and still < 0.010 and time.time() - t0 > 8: locked = True; break
b.send("arm_hold_here"); b.pump(0.6); b.send("arm_set_params", grav_ff_scale=0.85, extra_hand_mass_kg=0.16); b.pump(0.4)
res = {"locked": locked, "free_seconds": round(time.time() - t0, 1)}
if not locked: print("TEACH " + json.dumps(res), flush=True); b.close(); sys.exit(1)
b.say("LOCKED at", [round(v, 3) for v in b.qR()], "pinch", np.round(pinch(b.qR()), 3).tolist(), "- hands off now"); b.pump(3.0)
res["placed_open"] = shot("1_placed_open"); b.say("placed (open):", {k: res["placed_open"].get(k) for k in ("stereo_rel", "spread_mm", "choco_px", "hand_shift_px")})
# --- 2. 閉じて持ち上げて確かめる ---
h_now = b.hand(); tgt = list(OPEN); tgt[1] = h_now['pos'][1]; done = {}; CAP = {0: 690, 2: 250}; log = []     # 親指の付け根は人が合わせた値のまま
for n in range(48):
    for mtr in (0, 2):
        if mtr not in done:
            if tgt[mtr] + 6 > CAP[mtr]: done[mtr] = "cap"
            else: tgt[mtr] += 6
    if len(done) == 2: break
    h = b.hand_set(tgt, 400, 0.5); log.append({"cmd": [tgt[0], tgt[2]], **h})
    if 2 not in done:
        if h["Fn"][1] >= 1.5 and h["prox_k"][1] > 300: done[2] = f"index force {h['Fn'][1]:.2f}N at {h['pos'][2]}"
        elif abs(h["cur"][2]) >= 330: done[2] = f"index current {h['cur'][2]} at {h['pos'][2]}"
        elif tgt[2] - h["pos"][2] > 25: done[2] = f"index stall at {h['pos'][2]}"
    if 0 not in done:
        if h["Fn"][0] >= 1.5: done[0] = f"thumb force {h['Fn'][0]:.2f}N at {h['pos'][0]}"
        elif abs(h["cur"][0]) >= 380: done[0] = f"thumb current {h['cur'][0]} at {h['pos'][0]}"
        elif 2 in done and h["prox_k"][0] >= 350 and abs(h["cur"][0]) >= 150: done[0] = f"thumb proximity {h['prox_k'][0]}k + current {h['cur'][0]} at {h['pos'][0]}"
        elif tgt[0] - h["pos"][0] > 25: done[0] = f"thumb stall at {h['pos'][0]}"
    if n % 4 == 0 or len(done): b.say(f"close {n+1:2d} thumb {tgt[0]}/{h['pos'][0]} cur {h['cur'][0]} Fn {h['Fn'][0]:.2f} prox {h['prox_k'][0]} | index {tgt[2]}/{h['pos'][2]} cur {h['cur'][2]} Fn {h['Fn'][1]:.2f} prox {h['prox_k'][1]} | {done}")
b.pump(0.8); hc = b.hand(); res.update(close_done={str(k): v for k, v in done.items()}, close_log=log, closed=shot("2_closed")); b.say("closed", done, hc)
q_g = np.array(b.qR()); cmd_g = np.array(b.rob["cmd_arm"][5:]); _, Rn = fk(q_g)
qn, pe, re = ik(pinch(q_g) + np.array([0, 0, 0.03]), Rn, q_g); b.move((np.array(qn) + (cmd_g - q_g)).tolist(), "t_lift3", speed=0.05); b.pump(1.5); hl = b.hand(); b.save_hd("teach_lifted")
held = (hl["prox_k"][1] > 600 and hl["Fn"][1] > 0.9) or (hl["prox_k"][1] > 1500 and hl["prox_k"][0] > 300) or (hl["prox_k"][0] > 500 and hl["prox_k"][1] > 300)
res.update(lifted_hand=hl, held=bool(held), lift_ik_err=[round(pe, 1), round(re, 1)]); b.say("LIFTED 3cm. held =", held, hl)
if KEEP:
    res.update(grasp_q_right=[round(float(v), 4) for v in q_g], grasp_cmd_right=[round(float(v), 4) for v in cmd_g], hand_closed=hc["pos"], soc=b.rob["bms"]["soc"]); json.dump(res, open(D + "/teach_grasp.json", "w"), ensure_ascii=False, indent=1); b.mark("teach_grasp", held=bool(held), grasp_q=res["grasp_q_right"])
    print("TEACH " + json.dumps({"held": bool(held), "close_done": res["close_done"], "hand_closed": hc["pos"], "lifted_hand": hl, "grasp_q_right": res["grasp_q_right"], "placed_rel": res["placed_open"].get("stereo_rel"), "soc": res["soc"]}, ensure_ascii=False), flush=True); b.close(); sys.exit(0)
# --- 3. 下ろして開き、記録 ---
b.move(cmd_g.tolist(), "t_back_down", speed=0.04); b.pump(1.5)
for tgt2 in ([max(OPEN[0], hc["pos"][0] - 80), OPEN[1], max(OPEN[2], hc["pos"][2] - 60)] + OPEN[3:], OPEN): b.hand_set(tgt2, 900, 1.3)
b.pump(1.0); res["released_open"] = shot("3_released_open"); b.say("released (open):", {k: res["released_open"].get(k) for k in ("stereo_rel", "spread_mm", "choco_px", "hand_shift_px", "q_right")})
qn, pe, re = ik(pinch(q_g) + np.array([0, 0, 0.06]), Rn, q_g); b.move((np.array(qn) + (cmd_g - q_g)).tolist(), "t_up6", speed=0.06); b.pump(1.5); res["above_6cm"] = shot("4_above_6cm"); b.say("above 6cm:", {k: res["above_6cm"].get(k) for k in ("stereo_rel", "spread_mm", "choco_px")})
res.update(grasp_q_right=[round(float(v), 4) for v in q_g], grasp_cmd_right=[round(float(v), 4) for v in cmd_g], hand_closed=hc["pos"], soc=b.rob["bms"]["soc"])
json.dump(res, open(D + "/teach.json", "w"), ensure_ascii=False, indent=1); b.mark("teach_result", held=bool(held), grasp_q=res["grasp_q_right"])
print("TEACH " + json.dumps({"held": bool(held), "close_done": res["close_done"], "hand_closed": hc["pos"], "grasp_q_right": res["grasp_q_right"], "released_rel": res["released_open"].get("stereo_rel"), "soc": res["soc"]}, ensure_ascii=False), flush=True); b.close()
