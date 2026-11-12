"""置いてあるチョコを親指×人差指でつまんで持ち上げる。
usage: grasp.py DX DY [name]   (DX,DY = 教示した接地位置からのずれ [m]。上空でカメラを見て 1 回だけ自動補正する)
終了時、成功なら腕は 8cm 持ち上げた位置で保持。標準出力の最後に RESULT {...} を出す。
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision
SHAPE = OPEN   # 下ろすときの指の形
HOVER_REL = (-45.0, 21.5)        # 上空 10cm で「正しく合っている」ときの、手に対するチョコの見え方 [px] (試行 t1 で測定)
dx, dy = float(sys.argv[1]), float(sys.argv[2]); name = sys.argv[3] if len(sys.argv) > 3 else "g"
b = Bot("grasp_" + name); res = {"trial": name, "dx_in": dx, "dy_in": dy, "ok": False}
wrG, RG = fk(GROUND)
def plan(dx, dy):
    pT = pinch(GROUND) + np.array([dx, dy, 0.0]); qT, pe, re = ik(pT, RG, GROUND)
    if pe > 3 or re > 13: return None, pT, (pe, re)
    ch = []; qp = qT
    for dz in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10):
        q, _, _ = ik(pT + np.array([0, 0, dz]), RG, qp if dz > -0.02 else qT); ch.append((dz, q)); qp = q
    return ch[::-1], pT, (pe, re)
def finish(code=0):
    b.mark("grasp_result", **res); print("RESULT " + json.dumps({k: res.get(k) for k in ("ok", "why", "dx", "dy", "landed", "above_ground_mm", "ground_rel", "idx_Fn", "idx_prox_k", "thumb_prox_k", "close_done")}, ensure_ascii=False), flush=True); b.close(); sys.exit(code)
chain, pT, err = plan(dx, dy)
if chain is None: res["why"] = f"unreachable {err}"; finish(4)
h = b.hand_set(OPEN, 900, 1.4); h = b.hand_set(SHAPE, 900, 1.5)          # 開いてから寄せる (指モーターの保護状態を解除する意味もある)
if abs(h["pos"][2] - SHAPE[2]) > 12: res["why"] = f"index not responding {h['pos']}"; finish(5)
r = b.move(chain[0][1], "g_hover", speed=0.12); b.pump(1.0)
if r != "done": res["why"] = "hover: " + r; finish(6)
b.save_hd(name + "_1_hover"); rel = vision.rel_in_jpg(b.hd) if b.hd else None; res["hover_rel"] = rel; b.say("hover", b.snap()["pinch"], "rel", rel)
if rel and rel["score"] > 0.8:
    ex, ey = rel["dv"] - HOVER_REL[1], rel["du"] - HOVER_REL[0]
    cx, cy = vision.MM_PER_PX_X * ex / 1000.0, vision.MM_PER_PX_Y * ey / 1000.0
    b.say(f"visual offset at hover: forward {cx*1000:+.1f}mm, left {cy*1000:+.1f}mm")
    if 0.003 < max(abs(cx), abs(cy)) < 0.035:
        ch2, pT2, err2 = plan(dx + cx, dy + cy)
        if ch2 is not None:
            dx, dy, chain, pT = dx + cx, dy + cy, ch2, pT2; r = b.move(chain[0][1], "g_hover_adj", speed=0.08); b.pump(1.0)
            b.save_hd(name + "_1b_hover_adj"); rel2 = vision.rel_in_jpg(b.hd) if b.hd else None; res["hover_rel_adj"] = rel2; b.say("after adjust rel", rel2)
        else: b.say("adjusted target unreachable", err2)
res["dx"], res["dy"] = round(dx, 4), round(dy, 4); zG = float(pT[2])
# --- 下降 ---
prev = b.snap(); landed = None
def guard():
    hh = b.hand()
    if hh["pos"][2] > SHAPE[2] + 4 or abs(hh["cur"][2]) > 220: return f"index pushed {hh['pos'][2]}"
    if abs(hh["pos"][0] - SHAPE[0]) > 9: return f"thumb pushed {hh['pos'][0]}"
    if max(hh["Fn"]) > 1.0: return f"pad force {hh['Fn']}"
    return None
for dz, q in chain[1:]:
    r = b.move(q, f"g_down_{dz:+.3f}", speed=0.07, guard=guard); b.pump(0.9); s = b.snap(); dzm = s["pinch"][2] - prev["pinch"][2]
    b.say(f"down {dz:+.3f} -> {r} | z above taught ground {1000*(s['pinch'][2]-zG):+.1f}mm (step {dzm*1000:+.1f}) idx/thumb pos {s['hand']['pos'][2]}/{s['hand']['pos'][0]} prox_k {s['hand']['prox_k'][:2]} tau {s['tau_ext']}")
    if r != "done": landed = r; break
    if dz <= 0.03 and dzm > -0.002: landed = "stopped going down"; break
    if s["pinch"][2] <= zG - 0.004: landed = "below taught ground"; break
    prev = s
s = b.snap(); res["landed"] = landed; res["above_ground_mm"] = round(1000 * (s["pinch"][2] - zG), 1); res["ground"] = s
b.send("arm_hold_here"); b.pump(0.5); b.save_hd(name + "_2_ground"); rel = vision.rel_in_jpg(b.hd) if b.hd else None; res["ground_rel"] = rel and (rel["du"], rel["dv"]); b.say("landed:", landed, "above ground mm", res["above_ground_mm"], "rel at ground", rel)
if landed is None or res["above_ground_mm"] > 15:
    res["why"] = "did not reach the table"; b.hand_set(OPEN, 900, 1.2); b.move(chain[0][1], "g_abort_up", speed=0.10); finish(3)
# 押された指を解放してから寄せ直す
h = b.hand()
if h["pos"][2] > SHAPE[2] + 4 or abs(h["pos"][0] - SHAPE[0]) > 6:
    b.hand_set([SHAPE[0] - 20, SHAPE[1], max(49, SHAPE[2] - 30)] + SHAPE[3:], 600, 1.0); h = b.hand_set(SHAPE, 600, 1.0); b.say("re-seated fingers", h["pos"])
# --- 閉じる (親指と人差指を同じ量ずつ) ---
tgt = list(b.hand()["pos"]); tgt[1], tgt[3], tgt[4], tgt[5] = SHAPE[1], SHAPE[3], SHAPE[4], SHAPE[5]; done = {}; CAP = {0: 690, 2: 250}; log = []
for n in range(48):
    for m in (0, 2):
        if m not in done:
            if tgt[m] + 8 > CAP[m]: done[m] = "cap"
            else: tgt[m] += 8
    if len(done) == 2: break
    h = b.hand_set(tgt, 400, 0.5); log.append({"cmd": [tgt[0], tgt[2]], **h})
    if 2 not in done:
        if h["Fn"][1] >= 1.5: done[2] = f"index force {h['Fn'][1]:.2f}N"
        elif abs(h["cur"][2]) >= 350: done[2] = f"index current {h['cur'][2]}"
        elif tgt[2] - h["pos"][2] > 25: done[2] = "index stall"
    if 0 not in done:
        if h["Fn"][0] >= 1.5: done[0] = f"thumb force {h['Fn'][0]:.2f}N"
        elif abs(h["cur"][0]) >= 380: done[0] = f"thumb current {h['cur'][0]}"
        elif tgt[0] - h["pos"][0] > 25: done[0] = "thumb stall"
    if n % 3 == 0 or len(done): b.say(f"close {n+1:2d} thumb {tgt[0]}/{h['pos'][0]} cur {h['cur'][0]} prox {h['prox_k'][0]} | index {tgt[2]}/{h['pos'][2]} cur {h['cur'][2]} Fn {h['Fn'][1]:.2f} prox {h['prox_k'][1]} | {done}")
b.pump(0.8); h = b.hand(); res["close_done"] = {str(k): v for k, v in done.items()}; res["close_log"] = log; res["closed_hand"] = h; b.say("closed", done, h); b.save_hd(name + "_3_closed")
# --- 持ち上げて確かめる ---
up = {round(dz, 3): q for dz, q in chain}
b.move(up[0.03], "g_lift_3cm", speed=0.06); b.pump(0.8); b.move(up[0.08], "g_lift_8cm", speed=0.10); b.pump(1.0); h2 = b.hand(); b.save_hd(name + "_4_lifted"); c2 = vision.find_in_jpg(b.hd) if b.hd else None
ok = (h2["Fn"][1] > 0.6 and h2["prox_k"][1] > 600) or (h2["prox_k"][1] > 1500 and h2["prox_k"][0] > 300)
res.update(ok=bool(ok), idx_Fn=h2["Fn"][1], idx_prox_k=h2["prox_k"][1], thumb_prox_k=h2["prox_k"][0], lifted_hand=h2, choco_on_table_after_lift=c2, final=b.snap())
if not ok: res["why"] = "nothing in the fingers after lift"
b.say("after lift", h2, "choco still seen on table:", c2); finish(0 if ok else 2)
