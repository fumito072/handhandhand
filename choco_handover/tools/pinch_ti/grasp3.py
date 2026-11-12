"""置いてあるチョコを親指×人差指でつまんで持ち上げる (第 3 版: 腕の垂れ補正つき)。
usage: grasp3.py DX DY [name]     DX,DY = locate.py が出した、教示した接地位置からのずれ [m]
- 机に対する手の傾きは教示どおりに保つ (ik_tilt)。
- 机の 3cm 上で、実測の関節角が計画どおりになるよう指令を補正する (空中では腕が垂れるため)。
- 指が押されたら・下がらなくなったら接地とみなして止め、親指と人差指を同じ量ずつ閉じる。
成功なら 8cm 持ち上げた位置で保持して終了。最後に RESULT {...} を出す。
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision
dx, dy = float(sys.argv[1]), float(sys.argv[2]); name = sys.argv[3] if len(sys.argv) > 3 else "g"
b = Bot("grasp3_" + name); res = {"trial": name, "dx_in": dx, "dy_in": dy, "ok": False}
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
# --- 垂れ補正つきで下ろす: 実測の関節角が計画どおりになるよう、指令にずれ分を上乗せする ---
delta = np.zeros(5)
def mv(hgt, label, speed):
    r = b.move((np.array(ch[hgt]) + delta).tolist(), label, speed=speed, guard=guard)
    t0 = time.time(); h0 = height()                                     # 柔らかい腕はゆっくり落ち着くので、高さが動かなくなるまで待つ
    while time.time() - t0 < 7.0:
        b.pump(0.6); h1 = height()
        if abs(h1 - h0) < 0.0004 and time.time() - t0 > 1.2: break
        h0 = h1
    return r
def comp(hgt):
    global delta
    e = np.array(ch[hgt]) - np.array(b.qR()); delta = np.clip(delta + 0.8 * e, -0.15, 0.15); return e
def pose_err(hgt):
    q = b.qR(); return round(float(np.linalg.norm(pinch(q) - pinch(ch[hgt])) * 1000), 1), round(tilt_err_deg(q), 1)
for hgt in (0.10, 0.06, 0.03):
    r = mv(hgt, f"g_to_{hgt:+.2f}", 0.12 if hgt > 0.05 else 0.07)
    if r != "done": res["why"] = f"blocked on the way down at {hgt}: {r}"; b.move(ch[0.10], "g_abort_up", speed=0.08); finish(6)
# 垂れ補正は固定値 (試行 t5〜t7 で収束した値の平均)。小刻みな逐次補正は腕の摩擦で安定しなかった。
delta = np.array([-0.060, 0.005, -0.022, -0.008, 0.045])
r = mv(0.03, "g_to_+0.03_ff", 0.05); pe = pose_err(0.03); rel = rel_now("above3cm")
servo = [{"delta": delta.tolist(), "pos_err_mm": pe[0], "tilt_err_deg": pe[1], "rel_at_3cm": rel and (rel["du"], rel["dv"], rel["score"])}]; b.say("at +3cm", servo[-1], "soc", b.rob["bms"]["soc"])
res["servo"] = servo; res["dx"], res["dy"] = round(dx, 4), round(dy, 4)
landed = None; prev_h = height()
for hgt in (0.02, 0.01, 0.0, -0.01):
    r = mv(hgt, f"g_down_{hgt:+.3f}", 0.05); hh = b.hand(); hm = height(); pe = pose_err(hgt)
    b.say(f"down {hgt:+.3f} -> {r} | height {hm*1000:+.1f}mm (step {(hm-prev_h)*1000:+.1f}) pos_err {pe[0]}mm tilt_err {pe[1]}deg idx/thumb pos {hh['pos'][2]}/{hh['pos'][0]} cur {hh['cur'][2]}/{hh['cur'][0]} prox_k {hh['prox_k'][:2]}")
    if r != "done": landed = r; break
    if hm <= 0.0015: landed = "taught ground height"; break
    if hm - prev_h > -0.002: landed = "stopped going down"; break
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
        elif 2 in done and h["prox_k"][0] >= 350 and abs(h["cur"][0]) >= 150: done[0] = f"thumb proximity {h['prox_k'][0]}k + current {h['cur'][0]} at {h['pos'][0]}"
        elif tgt[0] - h["pos"][0] > 25: done[0] = f"thumb stall at {h['pos'][0]}"
    if n % 4 == 0 or len(done) == 1 or h["prox_k"][0] > 250: b.say(f"close {n+1:2d} thumb {tgt[0]}/{h['pos'][0]} cur {h['cur'][0]} prox {h['prox_k'][0]} | index {tgt[2]}/{h['pos'][2]} cur {h['cur'][2]} Fn {h['Fn'][1]:.2f} prox {h['prox_k'][1]} | {done}")
    if len(done) == 2: break
b.pump(0.8); h = b.hand(); res.update(close_done={str(k): v for k, v in done.items()}, close_log=log, closed_hand=h); b.say("closed", done, h); b.save_hd(name + "_closed")
# --- 持ち上げて確かめる ---
b.move(ch[0.03], "g_lift_3cm", speed=0.05); b.pump(0.8); b.move(ch[0.08], "g_lift_8cm", speed=0.08); b.pump(1.0); h2 = b.hand(); b.save_hd(name + "_lifted")
ok = (h2["Fn"][1] > 0.6 and h2["prox_k"][1] > 600) or (h2["prox_k"][1] > 1500 and h2["prox_k"][0] > 300)
res.update(ok=bool(ok), idx_Fn=h2["Fn"][1], idx_prox_k=h2["prox_k"][1], thumb_prox_k=h2["prox_k"][0], lifted_hand=h2, final=b.snap())
if not ok: res["why"] = "nothing in the fingers after lift"
b.say("after lift", h2); finish(0 if ok else 2)
