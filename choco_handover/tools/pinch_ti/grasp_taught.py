"""足を着けた高さで教え直した掴み (docs/data/teach_feet_down_20261112/teach_hold.json) を再生する。
usage: grasp_taught.py [name] [dx dy]     dx,dy = 教示位置からのずれ [m] (省略時 0)
1. 6cm 上 → 3cm 上 → 教示した掴み姿勢へ下ろす (指は開き)。
2. 着いたら、親指の近接値 (チョコが親指の横にあれば約 13 万以上) と、HD 画像での手とチョコの見え方を教示時と比べる。
3. 人差指を先に、続いて親指を閉じる (教示で開いた順の逆)。近接値・電流・指腹の力で止める。
4. 3cm → 6cm 持ち上げ、親指の近接値と「机の上からチョコが消えたか」で成否を判定する。成功したら持ったまま保持。最後に RESULT {...}。
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision, cv2
T = json.load(open(ROOT / "docs/data/teach_feet_down_20261112/teach_hold.json"))
name = sys.argv[1] if len(sys.argv) > 1 else "gt"; dx, dy = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0.0, 0.0)
Q_G = np.array(T["released_open"]["q_right"]); OPEN_T = T["open_shape"]; GRASP_T = T["grasp_shape"]
REF_PX = (T["released_open"]["choco_px"][0] - T["released_open"]["hand_shift_px"][0], T["released_open"]["choco_px"][1] - T["released_open"]["hand_shift_px"][1])   # 教示時の「チョコ画素 − 手のずれ」
_, R_G = fk(Q_G); P_G = pinch(Q_G)
b = Bot("grasp_taught_" + name); res = {"trial": name, "dx": dx, "dy": dy, "ok": False}
def finish(code=0):
    res["soc"] = b.rob["bms"]["soc"]; b.mark("grasp_result", **res); print("RESULT " + json.dumps({k: res.get(k) for k in ("ok", "why", "arrive", "close_done", "closed_hand", "lifted_hand", "choco_on_table_after", "soc")}, ensure_ascii=False), flush=True); b.close(); sys.exit(code)
def target(dz):
    q, pe, re = ik(P_G + np.array([dx, dy, dz]), R_G, Q_G); return q, pe, re
def go(dz, label, speed=0.05, guard=None):
    """目標へ。摩擦で小さな指令が動かないので、実測が目標から 4mm 以上ずれていたら 1.5cm 上へ離れて入り直す (最大 2 回)。"""
    qt, pe, re = target(dz); r = "done"
    for attempt in range(3):
        q = np.array(b.qR()); off = np.clip(np.array(b.rob["cmd_arm"][5:]) - q, -0.12, 0.12) if attempt else np.zeros(5)
        r = b.move((np.array(qt) + off).tolist(), f"{label}_{attempt}", speed=speed, guard=guard); t0 = time.time(); z0 = pinch(b.qR())
        while time.time() - t0 < 6:
            b.pump(0.6); z1 = pinch(b.qR())
            if np.linalg.norm(z1 - z0) < 0.0004 and time.time() - t0 > 1.2: break
            z0 = z1
        err = pinch(b.qR()) - pinch(qt)
        if r != "done" or np.linalg.norm(err) < 0.004 or (dz <= 0.001 and err[2] > -0.002 and np.linalg.norm(err[:2]) < 0.004): break
    return r, [round(float(v) * 1000, 1) for v in err]
def look():
    jpg = b.grab_hd()
    if not jpg: return None
    rel = vision.choco_rel_hand(cv2.imdecode(np.frombuffer(jpg, np.uint8), 1))
    if not rel: return None
    return {"du": round(rel["choco"]["u"] - rel["hand_shift"][0] - REF_PX[0], 1), "dv": round(rel["choco"]["v"] - rel["hand_shift"][1] - REF_PX[1], 1), "score": rel["score"], "choco_px": [rel["choco"]["u"], rel["choco"]["v"]]}
h = b.hand_set(OPEN_T, 900, 1.5)
if abs(h["pos"][2] - OPEN_T[2]) > 12 or abs(h["pos"][0] - OPEN_T[0]) > 12: res["why"] = f"fingers not responding {h['pos']}"; finish(5)
def guard():
    hh = b.hand()
    if hh["pos"][2] > OPEN_T[2] + 5 or abs(hh["cur"][2]) > 250: return f"index pushed pos {hh['pos'][2]} cur {hh['cur'][2]}"
    if abs(hh["pos"][0] - OPEN_T[0]) > 8 or abs(hh["cur"][0]) > 300: return f"thumb pushed pos {hh['pos'][0]} cur {hh['cur'][0]}"
    return None
for dz, lab, spd in ((0.06, "t6", 0.08), (0.03, "t3", 0.05)):
    r, e = go(dz, lab, spd, guard); b.say(f"to +{dz*100:.0f}cm: {r}, error mm {e}")
    if r != "done": res["why"] = f"blocked going to +{dz}: {r}"; finish(6)
r, e = go(0.0, "t0", 0.04, guard); b.pump(0.8); h = b.hand(); lk = look(); b.save_hd(name + "_arrive")
res["arrive"] = {"move": r, "err_mm": e, "thumb_prox_k": h["prox_k"][0], "index_prox_k": h["prox_k"][1], "look": lk, "q": [round(v, 4) for v in b.qR()]}; b.say("arrived:", res["arrive"])
if r != "done":                                             # 指が押された → 少し上げて開き直す
    q = np.array(b.qR()); qn, _, _ = ik(pinch(q) + np.array([0, 0, 0.005]), R_G, q); b.move(qn, "relief", speed=0.04); b.hand_set(OPEN_T, 500, 1.0)
# --- 閉じる: 人差指を先に、人差指が (教示の閉じ位置 − 親指の移動量) を過ぎたら親指も ---
tgt = list(OPEN_T); done = {}; log = []; IDX_END = min(250, GRASP_T[2] + 12); TH_END = min(690, GRASP_T[0] + 60); th_start = GRASP_T[2] - (GRASP_T[0] - OPEN_T[0])
for n in range(70):
    if 2 not in done:
        if tgt[2] + 6 > IDX_END: done[2] = "end"
        else: tgt[2] += 6
    if 0 not in done and (tgt[2] >= th_start or 2 in done):
        if tgt[0] + 6 > TH_END: done[0] = "end"
        else: tgt[0] += 6
    if len(done) == 2: break
    h = b.hand_set(tgt, 400, 0.5); log.append({"cmd": [tgt[0], tgt[2]], **h})
    if 2 not in done:
        if h["Fn"][1] >= 1.5: done[2] = f"index force {h['Fn'][1]:.2f}N at {h['pos'][2]}"
        elif abs(h["cur"][2]) >= 330: done[2] = f"index current {h['cur'][2]} at {h['pos'][2]}"
        elif tgt[2] - h["pos"][2] > 25: done[2] = f"index stall at {h['pos'][2]}"
    if 0 not in done:
        if h["Fn"][0] >= 1.5: done[0] = f"thumb force {h['Fn'][0]:.2f}N at {h['pos'][0]}"
        elif abs(h["cur"][0]) >= 300: done[0] = f"thumb current {h['cur'][0]} at {h['pos'][0]}"
        elif h["pos"][0] >= GRASP_T[0] and h["prox_k"][0] >= 600: done[0] = f"thumb proximity {h['prox_k'][0]}k at {h['pos'][0]}"
        elif tgt[0] - h["pos"][0] > 25: done[0] = f"thumb stall at {h['pos'][0]}"
    if n % 4 == 0 or len(done): b.say(f"close {n+1:2d} thumb {tgt[0]}/{h['pos'][0]} cur {h['cur'][0]} Fn {h['Fn'][0]:.2f} prox {h['prox_k'][0]} | index {tgt[2]}/{h['pos'][2]} cur {h['cur'][2]} Fn {h['Fn'][1]:.2f} prox {h['prox_k'][1]} | {done}")
b.pump(0.8); hc = b.hand(); res.update(close_done={str(k): v for k, v in done.items()}, close_log=log, closed_hand=hc); b.say("closed", done, hc); b.save_hd(name + "_closed")
# --- 持ち上げて確かめる ---
for dz, lab in ((0.03, "l3"), (0.06, "l6")):
    q = np.array(b.qR()); off = np.clip(np.array(b.rob["cmd_arm"][5:]) - q, -0.12, 0.12); qt, _, _ = target(dz); b.move((np.array(qt) + off).tolist(), lab, speed=0.05); b.pump(1.2)
hl = b.hand(); jpg = b.grab_hd(); c = None
if jpg: open(f"{OUT}/{name}_lifted.jpg", "wb").write(jpg); c = vision.find_in_jpg(jpg)
on_table = bool(c) and abs(c["u"] - T["released_open"]["choco_px"][0]) < 25 and abs(c["v"] - T["released_open"]["choco_px"][1]) < 25
ok = hl["prox_k"][0] >= 250 and not on_table
res.update(ok=bool(ok), lifted_hand=hl, choco_on_table_after=bool(on_table), choco_seen=c)
if not ok: res["why"] = "chocolate still on the table" if on_table else "nothing felt at the thumb after lift"
b.say("after lift:", hl, "| chocolate still at the taught spot on the table:", on_table, c); finish(0 if ok else 2)
