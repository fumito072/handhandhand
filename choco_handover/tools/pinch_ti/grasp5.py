"""置いてあるチョコを親指×人差指でつまんで持ち上げる (第 5 版: ステレオ 3D で手とチョコの相対位置を直接合わせる)。
usage: grasp5.py [name]
- 頭部の HD + ステレオ 2 眼で「手首の金具から見たチョコの位置 (前・左・上) [m]」を測る。
- 教示時 (掴んで机に接地した状態) の関係 REF を、今の手の向きに合わせて回したものが目標。差がそのまま手の動かす量になる。
  机の高さや胴体の高さが変わっても、相対位置で合わせるので影響を受けない。
- まず 3cm 上で水平位置を合わせ、次に高さを合わせて下ろし、親指と人差指を同じ量ずつ閉じる。
前提: 手がチョコの近く (10cm 程度以内) の机の上空にあり、チョコがカメラから見えていること。最後に RESULT {...} を出す。
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision, stereo, cv2
name = sys.argv[1] if len(sys.argv) > 1 else "g5"; b = Bot("grasp5_" + name); res = {"trial": name, "ok": False}
REF = np.array([0.0695, 0.0503, -0.1008]); _wr, RG = fk(GROUND); C_HAND = RG.T @ REF            # 手首の座標で見た「金具→つまみ点」
def finish(code=0):
    b.mark("grasp_result", **res); print("RESULT " + json.dumps({k: res.get(k) for k in ("ok", "why", "steps", "close_done", "idx_Fn", "idx_prox_k", "thumb_prox_k", "soc")}, ensure_ascii=False), flush=True); b.close(); sys.exit(code)
def measure(n=5):
    """誤差 e = (今のチョコの相対位置) − (今の手の向きでの目標相対位置) [m] (前, 左, 上)。チョコが見えなければ None。"""
    rows = []
    for k in range(n + 3):
        b.pump(0.35); jpg = b.grab_hd(); st = b.stereo
        if not jpg or not st: continue
        hd = cv2.imdecode(np.frombuffer(jpg, np.uint8), 1); s3 = stereo.choco_vs_wrist(hd, cv2.imdecode(np.frombuffer(st[0], np.uint8), 1), cv2.imdecode(np.frombuffer(st[1], np.uint8), 1), b.rob["q_aux"][1], 0.0)
        if s3 and s3["choco"]["depth_m"] > 0.33 and s3["tpl_score"] > 0.75: rows.append(s3["rel"])
        if len(rows) >= n: break
    if jpg: open(f"{OUT}/{name}_m{len(steps)}.jpg", "wb").write(jpg)
    if len(rows) < 3: return None
    A = np.array(rows); rel = np.median(A, axis=0); _, R = fk(b.qR()); e = rel - R @ C_HAND
    return {"e": e, "rel": rel, "spread_mm": float((A.max(0) - A.min(0)).max() * 1000), "n": len(rows)}
def settle(maxs=7.0):
    t0 = time.time(); q0 = np.array(b.qR())
    while time.time() - t0 < maxs:
        b.pump(0.6); q1 = np.array(b.qR())
        if np.abs(q1 - q0).max() < 0.0012 and time.time() - t0 > 1.2: break
        q0 = q1
def guard():
    hh = b.hand()
    if hh["pos"][2] > OPEN[2] + 4 or abs(hh["cur"][2]) > 220: return f"index pushed pos {hh['pos'][2]} cur {hh['cur'][2]}"
    if abs(hh["pos"][0] - OPEN[0]) > 7 or abs(hh["cur"][0]) > 260: return f"thumb pushed pos {hh['pos'][0]} cur {hh['cur'][0]}"
    if max(hh["Fn"]) > 1.2: return f"pad force {hh['Fn']}"
    return None
def shift(d, label, speed=0.05):
    """つまみ点を d [m] (前, 左, 上) だけ動かす。手の傾きは教示どおり。腕の垂れ分 (今の指令と実測の差) を上乗せする。"""
    q = np.array(b.qR()); off = np.array(b.rob["cmd_arm"][5:]) - q; qn, pe, te = ik_tilt(pinch(q) + np.asarray(d, dtype=float), q)
    if pe > 2.5 or te > 2.5:                                 # 傾きを厳密に保てない位置 → 今の手の向きをなるべく保つ解で代用 (目標は手の向きに合わせて回すので整合する)
        _, Rn = fk(q); qn, pe, re = ik(pinch(q) + np.asarray(d, dtype=float), Rn, q)
        if pe > 3.0 or re > 8.0: return f"unreachable (pos {pe:.1f}mm rot {re:.1f}deg)"
    r = b.move((np.array(qn) + np.clip(off, -0.12, 0.12)).tolist(), label, speed=speed, guard=guard); settle(); return r
steps = []
h = b.hand_set(OPEN, 900, 1.5)
if abs(h["pos"][2] - OPEN[2]) > 12 or abs(h["pos"][0] - OPEN[0]) > 12: res["why"] = f"fingers not responding {h['pos']}"; finish(5)
BLIND = np.array([float(v) for v in sys.argv[2:5]]) if len(sys.argv) >= 5 else None   # チョコが手に隠れているときの、残り誤差の見込み [m]
m = measure()
if m is None and BLIND is not None: m = {"e": BLIND, "rel": None, "spread_mm": -1.0, "n": 0}
if m is None: res["why"] = "chocolate not visible from here"; finish(8)
b.say("first look: error fwd/left/up [mm]", np.round(m["e"] * 1000, 1).tolist(), "spread", round(m["spread_mm"], 1), "soc", b.rob["bms"]["soc"])
# --- 1. チョコの 3cm 上の高さへ (今より低ければ先に上げる)、水平位置を合わせる ---
HOV = 0.015; ok_xy = False     # ほぼ掴む高さのまま水平に滑らせる (指先がチョコの方を向いているので、間に迎え入れる形になる)
for it in range(8):
    e = m["e"]; d = np.array([e[0], e[1], e[2] + HOV]); nxy = float(np.linalg.norm(d[:2]))
    steps.append({"phase": "hover", "it": it, "err_mm": np.round(e * 1000, 1).tolist(), "spread_mm": round(m["spread_mm"], 1)}); b.say("hover", steps[-1])
    if nxy < 0.004 and abs(d[2]) < 0.008: ok_xy = True; break
    if nxy > 0.02: d[:2] *= 0.02 / nxy
    d[2] = float(np.clip(d[2], -0.006, 0.02))                                  # 水平が大きくずれている間は下げない
    p_before = pinch(b.qR()); r = shift(d, f"h{it}"); steps[-1]["move"] = r; moved = pinch(b.qR()) - p_before
    if r != "done": b.say("move:", r); break
    m2 = measure()
    if m2 is None:                                           # 手がチョコを隠した → 最後に測った誤差から、腕が実際に動いた分を引いて残りを見積もる
        m = {"e": m["e"] - moved, "rel": None, "spread_mm": -1.0, "n": 0}; steps.append({"phase": "hover", "note": "chocolate hidden -> dead reckoning", "moved_mm": np.round(moved * 1000, 1).tolist()})
    else: m = m2
if not ok_xy: res.update(why="could not line up above the chocolate", steps=steps); finish(7)
# --- 2. 高さを合わせて下ろす (1cm ずつ、毎回測り直して水平も直す) ---
landed = None
for it in range(6):
    e = m["e"]; steps.append({"phase": "down", "it": it, "err_mm": np.round(e * 1000, 1).tolist()}); b.say("down", steps[-1])
    if e[2] > -0.004: landed = "at grasp height"; break
    d = np.array([e[0], e[1], max(e[2], -0.012)]); p_before = pinch(b.qR()); r = shift(d, f"d{it}", speed=0.04); steps[-1]["move"] = r; moved = pinch(b.qR()) - p_before
    if r != "done": landed = r; break
    m2 = measure()
    m = m2 if m2 is not None else {"e": m["e"] - moved, "rel": None, "spread_mm": -1.0, "n": 0}
res.update(steps=steps, landed=landed, ground=b.snap()); b.send("arm_hold_here"); b.pump(0.5); b.save_hd(name + "_ground"); b.say("landed:", landed)
h = b.hand()
if h["pos"][2] > OPEN[2] + 3 or abs(h["pos"][0] - OPEN[0]) > 5:
    shift([0, 0, 0.004], "relief", speed=0.04); h = b.hand_set(OPEN, 500, 0.9); b.say("relieved fingers", h["pos"], h["cur"])
# --- 3. 閉じる ---
tgt = list(OPEN); done = {}; CAP = {0: 690, 2: 250}; log = []; STEP = 8
for n in range(48):
    for mtr in (0, 2):
        if mtr not in done:
            if tgt[mtr] + STEP > CAP[mtr]: done[mtr] = "cap"
            else: tgt[mtr] += STEP
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
    if len(done) == 2: break
b.pump(0.8); h = b.hand(); res.update(close_done={str(k): v for k, v in done.items()}, close_log=log, closed_hand=h); b.say("closed", done, h); b.save_hd(name + "_closed")
r = shift([0, 0, 0.03], "lift3", speed=0.04); b.pump(0.8); r = shift([0, 0, 0.03], "lift6", speed=0.05); b.pump(1.0); h2 = b.hand(); b.save_hd(name + "_lifted")
ok = (h2["Fn"][1] > 0.9 and h2["prox_k"][1] > 600) or (h2["prox_k"][1] > 1500 and h2["prox_k"][0] > 300)
res.update(ok=bool(ok), idx_Fn=h2["Fn"][1], idx_prox_k=h2["prox_k"][1], thumb_prox_k=h2["prox_k"][0], lifted_hand=h2, final=b.snap(), soc=b.rob["bms"]["soc"])
if not ok: res["why"] = "nothing in the fingers after lift"
b.say("after lift", h2); finish(0 if ok else 2)
