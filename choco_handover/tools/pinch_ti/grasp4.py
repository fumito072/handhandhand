"""置いてあるチョコを親指×人差指でつまんで持ち上げる (第 4 版: 掴む前にカメラで追い込む)。
usage: grasp4.py DX DY name U V    DX,DY,U,V = locate.py が出した、教示位置からのずれ [m] とチョコの画素
1. 机の 3cm 上まで下ろす (机に対する手の傾きは教示どおり + 腕の垂れ分の固定補正)。
2. HD 画像で「手首の金具に対するチョコの見え方」を教示時と比べ、ずれていれば 6cm 上へ戻って位置を直し、下ろし直す (最大 4 回)。
   同時にステレオ 2 眼で奥行きを測り、手首がチョコより何 cm 上にあるかを確かめる。
3. 接地まで下ろし、親指と人差指を同じ量ずつ閉じる (指腹の力・近接値・モーター電流で各指の接触を判定)。
4. 持ち上げて、指に物があるかを触覚で確かめる。最後に RESULT {...} を出す。
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision, stereo, cv2
dx, dy = float(sys.argv[1]), float(sys.argv[2]); name = sys.argv[3] if len(sys.argv) > 3 else "g"
C_LOC = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else None   # locate.py が捕捉したチョコの画素 (着地すると手に隠れるので、こちらを照合に使う)
b = Bot("grasp4_" + name); res = {"trial": name, "dx_in": dx, "dy_in": dy, "ok": False}
pG = pinch(GROUND); zG = float(pG[2]); DELTA = np.array([-0.060, 0.005, -0.022, -0.008, 0.045]); REF_DROP = 0.1008   # 教示時: チョコは手首の金具より 10.1cm 下
def plan(dx, dy):
    q0, pe, te = ik_tilt(pG + np.array([dx, dy, 0.0]), GROUND)
    if pe > 1.0 or te > 1.0: return None
    ch = {}; qp = q0
    for hgt in (-0.01, 0.0, 0.01, 0.02, 0.03, 0.06, 0.10):
        q, _, _ = ik_tilt(pG + np.array([dx, dy, hgt]), qp if hgt > -0.01 else q0); ch[hgt] = q; qp = q
    return ch
def finish(code=0):
    b.mark("grasp_result", **res); print("RESULT " + json.dumps({k: res.get(k) for k in ("ok", "why", "dx", "dy", "servo", "landed", "close_done", "idx_Fn", "idx_prox_k", "thumb_prox_k")}, ensure_ascii=False), flush=True); b.close(); sys.exit(code)
def height(): return float(pinch(b.qR())[2] - zG)
def guard():
    hh = b.hand()
    if hh["pos"][2] > OPEN[2] + 4 or abs(hh["cur"][2]) > 220: return f"index pushed pos {hh['pos'][2]} cur {hh['cur'][2]}"
    if abs(hh["pos"][0] - OPEN[0]) > 7 or abs(hh["cur"][0]) > 260: return f"thumb pushed pos {hh['pos'][0]} cur {hh['cur'][0]}"
    if max(hh["Fn"]) > 1.0: return f"pad force {hh['Fn']}"
    return None
def mv(ch, hgt, label, speed):
    r = b.move((np.array(ch[hgt]) + DELTA).tolist(), label, speed=speed, guard=guard); t0 = time.time(); h0 = height()
    while time.time() - t0 < 7.0:
        b.pump(0.6); h1 = height()
        if abs(h1 - h0) < 0.0004 and time.time() - t0 > 1.2: break
        h0 = h1
    return r
def look(tag):
    """HD とステレオで、手とチョコの関係を測る。"""
    b.pump(0.6); jpg = b.grab_hd(); st = b.stereo
    if not jpg: return {}
    open(f"{OUT}/{name}_{tag}.jpg", "wb").write(jpg); hd = cv2.imdecode(np.frombuffer(jpg, np.uint8), 1); rel = vision.choco_rel_hand(hd); out = {"tag": tag, "fk_height_mm": round(1000 * height(), 1), "tilt_err_deg": round(tilt_err_deg(b.qR()), 1)}
    depth = None
    if st and rel:
        L = cv2.imdecode(np.frombuffer(st[0], np.uint8), 1); R = cv2.imdecode(np.frombuffer(st[1], np.uint8), 1); s3 = stereo.choco_vs_wrist(hd, L, R, b.rob["q_aux"][1], 0.0)
        if s3: depth = s3["choco"]["depth_m"]; out.update(stereo_rel=s3["rel"], stereo_height_mm=round(1000 * (-s3["rel"][2] - REF_DROP), 1), choco_depth_m=depth)
    # 同じ画像の中で手とチョコを同時に見る。ステレオの奥行きが机の距離 (0.39m 以上) なら本物のチョコ、近ければ手の上の反射とみなして捨てる。
    live = bool(rel) and rel["score"] >= 0.8 and depth is not None and depth >= 0.39
    if live: out.update(du=rel["du"], dv=rel["dv"], score=rel["score"], source="live", choco_px=[rel["choco"]["u"], rel["choco"]["v"]])
    else: out.update(source="hidden" if rel is None or depth is None or depth < 0.39 else "low score")
    return out
ch = plan(dx, dy)
if ch is None: res["why"] = "unreachable"; finish(4)
h = b.hand_set(OPEN, 900, 1.5)
if abs(h["pos"][2] - OPEN[2]) > 12 or abs(h["pos"][0] - OPEN[0]) > 12: res["why"] = f"fingers not responding {h['pos']}"; finish(5)
b.say("soc", b.rob["bms"]["soc"]); servo = []
# 指を開いたまま着地 → 机の上で「手とチョコの見え方」を教示時と照合 → ずれていたら 3cm 持ち上げて直し、着地し直す。
# 見え方 (du,dv)[px] と手の位置 (x,y)[m] の関係 J は、動かした結果から学習する (Broyden 法)。
J = np.array([[0.0, 1163.0], [971.0, 0.0]]); prev = None; landed = None; aligned = False
r = mv(ch, 0.06, "g_to_+0.06", 0.10)
for it in range(6):
    r = mv(ch, 0.03, f"g{it}_to_+0.03", 0.06)
    if r != "done": res["why"] = f"blocked at +3cm: {r}"; res["servo"] = servo; b.move(ch[0.10], "g_abort_up", speed=0.08); finish(6)
    r = mv(ch, 0.0, f"g{it}_land", 0.04)
    if r == "done" and height() > 0.004: r = mv(ch, -0.01, f"g{it}_land_more", 0.04)
    landed = r if r != "done" else f"settled at {1000*height():.1f}mm"
    m = look(f"land{it}"); xy = pinch(b.qR())[:2]; hh = b.hand(); m.update(dx=round(dx, 4), dy=round(dy, 4), pinch_xy=[round(float(v), 4) for v in xy], landed=landed, idx_thumb_pos=[hh["pos"][2], hh["pos"][0]], prox_k=hh["prox_k"][:2]); servo.append(m)
    b.say("landed look", {k: m[k] for k in m if k not in ("stereo_score", "tag")})
    if "du" not in m: break
    e = np.array([m["du"], m["dv"]])
    if prev is not None:
        dxy = xy - prev[0]; de = e - prev[1]
        if np.linalg.norm(dxy) > 0.003: J = J + np.outer(de - J @ dxy, dxy) / float(dxy @ dxy)
    prev = (xy, e)
    if abs(e[0]) <= 6 and abs(e[1]) <= 6: aligned = True; break
    if it == 5: break
    try: step = -np.linalg.solve(J, e)
    except np.linalg.LinAlgError: step = np.zeros(2)
    n_ = float(np.linalg.norm(step))
    if n_ > 0.012: step = step * 0.012 / n_
    ch2 = plan(dx + float(step[0]), dy + float(step[1]))
    if ch2 is None: b.say("correction unreachable"); break
    b.hand_set(OPEN, 400, 0.6); mv(ch, 0.03, f"g{it}_lift_before_shift", 0.06)
    dx, dy, ch = dx + float(step[0]), dy + float(step[1]), ch2
res.update(J_learned=J.tolist(), servo=servo, dx=round(dx, 4), dy=round(dy, 4), landed=landed, aligned=aligned, ground=b.snap()); last = servo[-1]
b.send("arm_hold_here"); b.pump(0.5)
if not aligned and ("du" not in last or max(abs(last["du"]), abs(last["dv"])) > 12):
    res["why"] = f"could not line up with the chocolate (last look du,dv = {last.get('du')},{last.get('dv')})"; b.move(ch[0.10], "g_abort_up", speed=0.08); finish(7)
if last.get("fk_height_mm", 99) > 12: res["why"] = "did not reach the table"; b.move(ch[0.10], "g_abort_up", speed=0.08); finish(3)
h = b.hand()
if h["pos"][2] > OPEN[2] + 3 or abs(h["pos"][0] - OPEN[0]) > 5:           # 押された指は、少し持ち上げて開く向きへ指令し直す (保護状態の解除)
    q_up, _, _ = ik_tilt(pinch(b.qR()) + np.array([0, 0, 0.004]), b.qR()); b.move(q_up, "g_relief_4mm", speed=0.04); h = b.hand_set(OPEN, 500, 0.9); b.say("relieved fingers", h["pos"], h["cur"])
# --- 閉じる ---
tgt = list(OPEN); done = {}; CAP = {0: 690, 2: 250}; log = []; STEP = 8
for n in range(48):
    for mtr in (0, 2):
        if mtr not in done:
            if tgt[mtr] + STEP > CAP[mtr]: done[mtr] = "cap"
            else: tgt[mtr] += STEP
    if len(done) == 2: break
    h = b.hand_set(tgt, 400, 0.5); log.append({"cmd": [tgt[0], tgt[2]], **h})
    if 2 not in done:
        if h["Fn"][1] >= 1.5: done[2] = f"index force {h['Fn'][1]:.2f}N at {h['pos'][2]}"
        elif abs(h["cur"][2]) >= 330: done[2] = f"index current {h['cur'][2]} at {h['pos'][2]}"
        elif tgt[2] - h["pos"][2] > 25: done[2] = f"index stall at {h['pos'][2]}"
    if 0 not in done:
        if h["Fn"][0] >= 1.5: done[0] = f"thumb force {h['Fn'][0]:.2f}N at {h['pos'][0]}"
        elif abs(h["cur"][0]) >= 380: done[0] = f"thumb current {h['cur'][0]} at {h['pos'][0]}"
        elif 2 in done and h["prox_k"][0] >= 350 and abs(h["cur"][0]) >= 150: done[0] = f"thumb proximity {h['prox_k'][0]}k + current {h['cur'][0]} at {h['pos'][0]}"
        elif tgt[0] - h["pos"][0] > 25: done[0] = f"thumb stall at {h['pos'][0]}"
    if n % 4 == 0 or len(done) == 1 or h["prox_k"][0] > 250: b.say(f"close {n+1:2d} thumb {tgt[0]}/{h['pos'][0]} cur {h['cur'][0]} prox {h['prox_k'][0]} | index {tgt[2]}/{h['pos'][2]} cur {h['cur'][2]} Fn {h['Fn'][1]:.2f} prox {h['prox_k'][1]} | {done}")
    if len(done) == 2: break
b.pump(0.8); h = b.hand(); res.update(close_done={str(k): v for k, v in done.items()}, close_log=log, closed_hand=h); b.say("closed", done, h); b.save_hd(name + "_closed")
b.move(ch[0.03], "g_lift_3cm", speed=0.05); b.pump(0.8); b.move(ch[0.06], "g_lift_6cm", speed=0.08); b.pump(1.0); h2 = b.hand(); b.save_hd(name + "_lifted")
ok = (h2["Fn"][1] > 0.6 and h2["prox_k"][1] > 600) or (h2["prox_k"][1] > 1500 and h2["prox_k"][0] > 300)
res.update(ok=bool(ok), idx_Fn=h2["Fn"][1], idx_prox_k=h2["prox_k"][1], thumb_prox_k=h2["prox_k"][0], lifted_hand=h2, final=b.snap())
if not ok: res["why"] = "nothing in the fingers after lift"
b.say("after lift", h2); finish(0 if ok else 2)
