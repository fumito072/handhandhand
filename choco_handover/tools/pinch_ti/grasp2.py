"""置いてあるチョコを親指×人差指でつまんで持ち上げる (第 2 版)。
usage: grasp2.py DX DY [name]     DX,DY = locate.py が出した、教示した接地位置からのずれ [m]
- 机に対する手の傾きは教示どおりに保つ (ik_tilt)。
- 机の 2cm 上でカメラを見て、手とチョコの関係が教示時と同じになるよう位置を追い込む。
- 指が押されたら・下がらなくなったら接地とみなして止め、親指と人差指を同じ量ずつ閉じる。
成功なら 8cm 持ち上げた位置で保持して終了。最後に RESULT {...} を出す。
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision
dx, dy = float(sys.argv[1]), float(sys.argv[2]); name = sys.argv[3] if len(sys.argv) > 3 else "g"
b = Bot("grasp2_" + name); res = {"trial": name, "dx_in": dx, "dy_in": dy, "ok": False}
pG = pinch(GROUND); zG = float(pG[2]); HEIGHTS = (0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.0, -0.005, -0.01)
def plan(dx, dy):
    q0, pe, te = ik_tilt(pG + np.array([dx, dy, 0.0]), GROUND)
    if pe > 1.0 or te > 1.0: return None, (pe, te)
    ch = {}; qp = q0
    for hgt in sorted(HEIGHTS):
        q, pe, te = ik_tilt(pG + np.array([dx, dy, hgt]), qp if hgt > -0.01 else q0); ch[hgt] = q; qp = q
    return ch, (pe, te)
def finish(code=0):
    b.mark("grasp_result", **res); print("RESULT " + json.dumps({k: res.get(k) for k in ("ok", "why", "dx", "dy", "servo", "landed", "above_ground_mm", "tilt_err_deg", "ground_rel", "close_done", "idx_Fn", "idx_prox_k", "thumb_prox_k")}, ensure_ascii=False), flush=True); b.close(); sys.exit(code)
def height(): return pinch(b.qR())[2] - zG
def rel_now(tag):
    b.pump(0.8); b.save_hd(f"{name}_{tag}"); return vision.rel_in_jpg(b.hd) if b.hd else None
ch, err = plan(dx, dy)
if ch is None: res["why"] = f"unreachable {err}"; finish(4)
h = b.hand_set(OPEN, 900, 1.5)
if abs(h["pos"][2] - OPEN[2]) > 12 or abs(h["pos"][0] - OPEN[0]) > 12: res["why"] = f"fingers not responding {h['pos']}"; finish(5)
def guard():
    hh = b.hand()
    if hh["pos"][2] > OPEN[2] + 4 or abs(hh["cur"][2]) > 220: return f"index pushed pos {hh['pos'][2]} cur {hh['cur'][2]}"
    if abs(hh["pos"][0] - OPEN[0]) > 7 or abs(hh["cur"][0]) > 260: return f"thumb pushed pos {hh['pos'][0]} cur {hh['cur'][0]}"
    if max(hh["Fn"]) > 1.0: return f"pad force {hh['Fn']}"
    return None
for hgt in (0.10, 0.06, 0.03):
    r = b.move(ch[hgt], f"g_to_{hgt:+.2f}", speed=0.12 if hgt > 0.05 else 0.07, guard=guard)
    if r != "done": res["why"] = f"blocked on the way down at {hgt}: {r}"; b.move(ch[0.10], "g_abort_up", speed=0.08); finish(6)
# --- 机の約 2cm 上で位置を追い込む ---
servo = []
for it in range(4):
    r = b.move(ch[0.02], f"g_servo_{it}", speed=0.06, guard=guard); rel = rel_now(f"servo{it}")
    servo.append({"it": it, "dx": round(dx, 4), "dy": round(dy, 4), "height_mm": round(1000 * height(), 1), "rel": rel and (rel["du"], rel["dv"], rel["score"]), "move": r}); b.say("servo", servo[-1])
    if r != "done" or not rel or rel["score"] < 0.8: break
    if abs(rel["du"]) <= 4 and abs(rel["dv"]) <= 4: break
    cx = float(np.clip(vision.MM_PER_PX_X * rel["dv"] / 1000.0, -0.015, 0.015)); cy = float(np.clip(vision.MM_PER_PX_Y * rel["du"] / 1000.0, -0.015, 0.015))
    ch2, err = plan(dx + cx, dy + cy)
    if ch2 is None: b.say("correction unreachable", err); break
    dx, dy, ch = dx + cx, dy + cy, ch2
res["servo"] = servo; res["dx"], res["dy"] = round(dx, 4), round(dy, 4)
# --- 接地まで下ろす ---
landed = None; prev_h = height()
for hgt in (0.015, 0.01, 0.005, 0.0, -0.005, -0.01):
    r = b.move(ch[hgt], f"g_down_{hgt:+.3f}", speed=0.05, guard=guard); b.pump(0.9); hh = b.hand(); hm = height()
    b.say(f"down {hgt:+.3f} -> {r} | height {hm*1000:+.1f}mm (step {(hm-prev_h)*1000:+.1f}) tilt_err {tilt_err_deg(b.qR()):.1f}deg idx/thumb pos {hh['pos'][2]}/{hh['pos'][0]} cur {hh['cur'][2]}/{hh['cur'][0]} prox_k {hh['prox_k'][:2]}")
    if r != "done": landed = r; break
    if hm - prev_h > -0.0015: landed = "stopped going down"; break
    if hm <= 0.0005: landed = "taught ground height"; break
    prev_h = hm
b.send("arm_hold_here"); b.pump(0.5); s = b.snap(); rel = rel_now("ground")
res.update(landed=landed, above_ground_mm=round(1000 * height(), 1), tilt_err_deg=round(tilt_err_deg(b.qR()), 1), ground=s, ground_rel=rel and (rel["du"], rel["dv"], rel["score"])); b.say("landed:", landed, "height mm", res["above_ground_mm"], "tilt err", res["tilt_err_deg"], "rel", res["ground_rel"])
if landed is None or res["above_ground_mm"] > 12:
    res["why"] = "did not reach the table"; b.move(ch[0.10], "g_abort_up", speed=0.08); finish(3)
# 指が押されていたら、いったん開く向きへ指令して保護状態を解く
h = b.hand()
if h["pos"][2] > OPEN[2] + 3 or abs(h["pos"][0] - OPEN[0]) > 5:
    q_up, _, _ = ik_tilt(pinch(b.qR()) + np.array([0, 0, 0.004]), b.qR()); b.move(q_up, "g_relief_4mm", speed=0.04); h = b.hand_set(OPEN, 500, 0.9); b.say("relieved fingers", h["pos"], h["cur"])
# --- 閉じる: 親指と人差指を同じ量ずつ。各指は自分の接触で止まる ---
tgt = list(OPEN); done = {}; CAP = {0: 690, 2: 250}; log = []; STEP = 8
for n in range(48):
    for m in (0, 2):
        if m not in done:
            if tgt[m] + STEP > CAP[m]: done[m] = "cap"
            else: tgt[m] += STEP
    if len(done) == 2: break
    h = b.hand_set(tgt, 400, 0.5); log.append({"cmd": [tgt[0], tgt[2]], **h})
    if 2 not in done:
        if h["Fn"][1] >= 1.5: done[2] = f"index force {h['Fn'][1]:.2f}N"
        elif abs(h["cur"][2]) >= 330: done[2] = f"index current {h['cur'][2]} at {h['pos'][2]}"
        elif tgt[2] - h["pos"][2] > 25: done[2] = f"index stall at {h['pos'][2]}"
    if 0 not in done:
        if h["Fn"][0] >= 1.5: done[0] = f"thumb force {h['Fn'][0]:.2f}N"
        elif abs(h["cur"][0]) >= 380: done[0] = f"thumb current {h['cur'][0]} at {h['pos'][0]}"
        elif tgt[0] - h["pos"][0] > 25: done[0] = f"thumb stall at {h['pos'][0]}"
    if n % 4 == 0 or len(done): b.say(f"close {n+1:2d} thumb {tgt[0]}/{h['pos'][0]} cur {h['cur'][0]} prox {h['prox_k'][0]} | index {tgt[2]}/{h['pos'][2]} cur {h['cur'][2]} Fn {h['Fn'][1]:.2f} prox {h['prox_k'][1]} | {done}")
    if len(done) == 2: break
b.pump(0.8); h = b.hand(); res.update(close_done={str(k): v for k, v in done.items()}, close_log=log, closed_hand=h); b.say("closed", done, h); b.save_hd(name + "_closed")
# --- 持ち上げて確かめる ---
b.move(ch[0.03], "g_lift_3cm", speed=0.05); b.pump(0.8); b.move(ch[0.08], "g_lift_8cm", speed=0.08); b.pump(1.0); h2 = b.hand(); b.save_hd(name + "_lifted")
ok = (h2["Fn"][1] > 0.6 and h2["prox_k"][1] > 600) or (h2["prox_k"][1] > 1500 and h2["prox_k"][0] > 300)
res.update(ok=bool(ok), idx_Fn=h2["Fn"][1], idx_prox_k=h2["prox_k"][1], thumb_prox_k=h2["prox_k"][0], lifted_hand=h2, final=b.snap())
if not ok: res["why"] = "nothing in the fingers after lift"
b.say("after lift", h2); finish(0 if ok else 2)
