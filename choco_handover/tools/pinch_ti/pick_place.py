"""置く・見る・掴むの繰り返しと、カメラによる追跡掴み (足を着けた高さ・teach_hold.json の教示を使う)。

  pick_place.py cycle DX DY [name]   持っているチョコを教示位置から (DX,DY)[m] ずらして置く → 観察姿勢からカメラで画素を記録
                                     → 置いた位置へ戻って掴む。結果と (画素, 置いた位置) を samples.jsonl に追記する。
  pick_place.py fit                  samples.jsonl から「画素 → 位置のずれ」の換算 (アフィン) を作り、px_map.json に保存する。
  pick_place.py track [name]         観察姿勢からチョコを探し、換算で位置を出して掴む (チョコを動かしても追う)。
  pick_place.py place DX DY          置くだけ。     pick_place.py observe   見るだけ (観察姿勢へ移動して画素を出す)。
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision, cv2
TD = ROOT / "docs/data/teach_feet_down_20261112"; T = json.load(open(TD / "teach_hold.json"))
Q_G = np.array(T["released_open"]["q_right"]); OPEN_T = T["open_shape"]; GRASP_T = T["grasp_shape"]; _, R_G = fk(Q_G); P_G = pinch(Q_G)
OBS = [-0.25, 0.079, -0.019, -0.65, 1.426]                   # 観察姿勢: 手を右上へ退けて机の上を空ける
SAMPLES = TD / "samples.jsonl"; MAP = TD / "px_map.json"
REACH = {"dx": (-0.045, 0.025), "dy": (-0.065, 0.025)}

class Cell:
    def __init__(self, tag): self.b = Bot(tag); b = self.b; b.recover(any_fault=True) if b.rob.get("stage") in FAULT else None
    def holding(self):
        h = self.b.hand(); return h["prox_k"][1] > 500 or h["prox_k"][0] > 250
    def target(self, dx, dy, dz):
        q0, _, _ = ik(P_G + np.array([dx, dy, 0.0]), R_G, Q_G)
        return ik(P_G + np.array([dx, dy, dz]), R_G, q0) if dz > 0.001 else (q0, 0.0, 0.0)
    def go(self, dx, dy, dz, label, speed=0.05, guard=None):
        """目標へ。実測が 4mm 以上ずれていたら、腕の垂れ分を上乗せして入り直す (小さな指令は摩擦で動かないため)。"""
        b = self.b; qt, _, _ = self.target(dx, dy, dz); r = "done"; err = np.zeros(3)
        for attempt in range(3):
            q = np.array(b.qR()); off = np.clip(np.array(b.rob["cmd_arm"][5:]) - q, -0.12, 0.12) if attempt else np.zeros(5)
            r = b.move((np.array(qt) + off).tolist(), f"{label}_{attempt}", speed=speed, guard=guard); t0 = time.time(); z0 = pinch(b.qR())
            while time.time() - t0 < 6:
                b.pump(0.6); z1 = pinch(b.qR())
                if np.linalg.norm(z1 - z0) < 0.0004 and time.time() - t0 > 1.2: break
                z0 = z1
            err = pinch(b.qR()) - pinch(qt)
            if r != "done" or np.linalg.norm(err) < 0.004 or (dz <= 0.005 and err[2] > -0.002 and np.linalg.norm(err[:2]) < 0.004): break
        return r, [round(float(v) * 1000, 1) for v in err]
    def look(self):
        """着地姿勢で、手首の金具に対するチョコの見え方を教示時と比べる (du,dv [px])。"""
        jpg = self.b.grab_hd()
        if not jpg: return None
        rel = vision.choco_rel_hand(cv2.imdecode(np.frombuffer(jpg, np.uint8), 1))
        if not rel: return None
        ro = T["released_open"]; ref = (ro["choco_px"][0] - ro["hand_shift_px"][0], ro["choco_px"][1] - ro["hand_shift_px"][1])
        return {"du": round(rel["choco"]["u"] - rel["hand_shift"][0] - ref[0], 1), "dv": round(rel["choco"]["v"] - rel["hand_shift"][1] - ref[1], 1), "score": rel["score"]}
    def placed_offset(self):
        p = pinch(self.b.qR()); return [round(float(p[0] - P_G[0]), 4), round(float(p[1] - P_G[1]), 4)]
    def place(self, dx, dy):
        """持っているチョコを (dx,dy) に置き、6cm 上へ退く。置いた実際の位置 (実測) を返す。"""
        b = self.b
        for dz, lab, spd in ((0.06, "p6", 0.07), (0.03, "p3", 0.05), (0.004, "p0", 0.04)): r, e = self.go(dx, dy, dz, lab, spd)
        b.pump(0.6); at = self.placed_offset(); hg = b.hand(); tgt = list(hg["pos"])
        while tgt[0] > OPEN_T[0] or tgt[2] > OPEN_T[2]:
            tgt[0] = max(OPEN_T[0], tgt[0] - 6); tgt[2] = max(OPEN_T[2], tgt[2] - 6); b.hand_set(tgt, 400, 0.4)
        b.hand_set(OPEN_T, 400, 0.8); self.go(dx, dy, 0.03, "pu3", 0.05); self.go(dx, dy, 0.06, "pu6", 0.07)
        b.mark("placed", at=at, cmd=[dx, dy], err_mm=e); return at
    def observe(self, name="obs"):
        """観察姿勢へ移動して、チョコの画素 (数枚の中央値) を返す。"""
        b = self.b; b.move(OBS, "to_obs", speed=0.15); b.pump(1.5); pts = []
        for k in range(6):
            jpg = b.grab_hd()
            if jpg:
                c = vision.find_in_jpg(jpg)
                if c: pts.append((c["u"], c["v"]))
                if k == 0: open(f"{OUT}/{name}_obs.jpg", "wb").write(jpg)
            b.pump(0.3)
        if len(pts) < 3: return None
        A = np.array(pts); return [round(float(np.median(A[:, 0])), 1), round(float(np.median(A[:, 1])), 1), round(float((A.max(0) - A.min(0)).max()), 1)]
    def grasp(self, dx, dy, name="g"):
        """(dx,dy) にあるチョコを掴んで 6cm 持ち上げる。成否と記録を返す。"""
        b = self.b; res = {"dx": dx, "dy": dy, "ok": False}
        h = b.hand_set(OPEN_T, 900, 1.5)
        if abs(h["pos"][2] - OPEN_T[2]) > 12 or abs(h["pos"][0] - OPEN_T[0]) > 12: res["why"] = f"fingers not responding {h['pos']}"; return res
        def guard():
            hh = b.hand()
            if hh["pos"][2] > OPEN_T[2] + 5 or abs(hh["cur"][2]) > 250: return f"index pushed pos {hh['pos'][2]} cur {hh['cur'][2]}"
            if abs(hh["pos"][0] - OPEN_T[0]) > 8 or abs(hh["cur"][0]) > 300: return f"thumb pushed pos {hh['pos'][0]} cur {hh['cur'][0]}"
            return None
        for dz, lab, spd in ((0.06, "g6", 0.08), (0.03, "g3", 0.05)):
            r, e = self.go(dx, dy, dz, lab, spd, guard)
            if r != "done": res["why"] = f"blocked going to +{dz}: {r}"; return res
        r, e = self.go(dx, dy, 0.0, "g0", 0.04, guard); b.pump(0.8); align = []
        for it in range(4):                                  # 着地した状態で、手とチョコの見え方を教示時と比べて直す
            lk = self.look()
            if lk and (abs(lk["du"]) > 60 or abs(lk["dv"]) > 60): b.pump(0.6); lk = self.look()          # 明らかにおかしい値は撮り直す
            if lk and (abs(lk["du"]) > 60 or abs(lk["dv"]) > 60): lk = None
            hh = b.hand(); align.append({"it": it, "look": lk, "thumb_prox_k": hh["prox_k"][0], "dx": round(dx, 4), "dy": round(dy, 4)})
            if r != "done" or not lk or lk["score"] < 0.75 or (abs(lk["du"]) <= 3 and abs(lk["dv"]) <= 3) or it == 3: break
            cx = float(np.clip(-0.0009 * lk["dv"], -0.008, 0.008)); cy = float(np.clip(-0.00085 * lk["du"], -0.008, 0.008))
            self.go(dx, dy, 0.015, f"a{it}_up", 0.05); dx, dy = dx + cx, dy + cy; self.go(dx, dy, 0.015, f"a{it}_shift", 0.05); r, e = self.go(dx, dy, 0.0, f"a{it}_down", 0.04, guard); b.pump(0.8)
        h = b.hand(); b.save_hd(name + "_arrive")
        res["arrive"] = {"move": r, "err_mm": e, "thumb_prox_k": h["prox_k"][0], "index_prox_k": h["prox_k"][1], "align": align, "dx": round(dx, 4), "dy": round(dy, 4)}
        if r != "done":
            q = np.array(b.qR()); qn, _, _ = ik(pinch(q) + np.array([0, 0, 0.005]), R_G, q); b.move(qn, "relief", speed=0.04); b.hand_set(OPEN_T, 500, 1.0)
        tgt = list(OPEN_T); done = {}; IDX_END = min(250, GRASP_T[2] + 12); TH_END = min(690, GRASP_T[0] + 120); th_start = GRASP_T[2] - (GRASP_T[0] - OPEN_T[0])
        for n in range(70):
            if 2 not in done:
                if tgt[2] + 6 > IDX_END: done[2] = "end"
                else: tgt[2] += 6
            if 0 not in done and (tgt[2] >= th_start or 2 in done):
                if tgt[0] + 6 > TH_END: done[0] = "end"
                else: tgt[0] += 6
            if len(done) == 2: break
            h = b.hand_set(tgt, 400, 0.5)
            if 2 not in done:
                if h["Fn"][1] >= 1.5: done[2] = f"index force {h['Fn'][1]:.2f}N at {h['pos'][2]}"
                elif abs(h["cur"][2]) >= 330: done[2] = f"index current {h['cur'][2]} at {h['pos'][2]}"
                elif tgt[2] - h["pos"][2] > 25: done[2] = f"index stall at {h['pos'][2]}"
            if 0 not in done:
                if h["Fn"][0] >= 1.5: done[0] = f"thumb force {h['Fn'][0]:.2f}N at {h['pos'][0]}"
                elif h["pos"][0] >= GRASP_T[0] + 30 and abs(h["cur"][0]) >= 240: done[0] = f"thumb current {h['cur'][0]} at {h['pos'][0]}"     # しっかり当たるまで閉じ込む
                elif abs(h["cur"][0]) >= 330: done[0] = f"thumb current {h['cur'][0]} at {h['pos'][0]}"
                elif tgt[0] - h["pos"][0] > 25: done[0] = f"thumb stall at {h['pos'][0]}"
        b.pump(0.8); hc = b.hand(); res.update(close_done={str(k): v for k, v in done.items()}, closed_hand=hc)
        grip0 = hc["Fn"][1]; res["lift_log"] = []
        for dz, lab in ((0.01, "l1"), (0.03, "l3"), (0.06, "l6")):
            q = np.array(b.qR()); off = np.clip(np.array(b.rob["cmd_arm"][5:]) - q, -0.12, 0.12); qt, _, _ = self.target(dx, dy, dz); b.move((np.array(qt) + off).tolist(), lab, speed=0.04); b.pump(1.2)
            hh = b.hand(); res["lift_log"].append({"dz": dz, "Fn": hh["Fn"][1], "prox_k": hh["prox_k"][:2]})
            if hh["prox_k"][1] < 300 and hh["prox_k"][0] < 200: break                 # 落とした → それ以上は上げない
        hl = b.hand(); b.save_hd(name + "_lifted"); ok = hl["prox_k"][0] >= 250 and (hl["prox_k"][1] >= 400 or hl["Fn"][1] >= 0.8)
        res.update(ok=bool(ok), lifted_hand=hl)
        if not ok: res["why"] = "nothing held after lift"
        b.mark("grasp_result", **res); return res
def load_map():
    return json.load(open(MAP)) if MAP.exists() else None
def px_to_offset(u, v, m):
    A = np.array(m["A"]); return [float(A[0] @ [u, v, 1.0]), float(A[1] @ [u, v, 1.0])]
def fit():
    rows = [json.loads(l) for l in open(SAMPLES)] if SAMPLES.exists() else []; rows = [r for r in rows if r.get("px") and r.get("at")]
    if len(rows) < 3: print("FIT need at least 3 samples, have", len(rows)); return None
    X = np.array([[r["px"][0], r["px"][1], 1.0] for r in rows]); Y = np.array([r["at"] for r in rows]); A, *_ = np.linalg.lstsq(X, Y, rcond=None); res = (X @ A - Y) * 1000
    m = {"A": A.T.tolist(), "n": len(rows), "rms_mm": round(float(np.sqrt((res ** 2).sum(1).mean())), 1), "max_mm": round(float(np.abs(res).max()), 1), "mm_per_px": [round(float(np.linalg.norm(A[0]) * 1000), 2), round(float(np.linalg.norm(A[1]) * 1000), 2)]}
    json.dump(m, open(MAP, "w"), indent=1); print("FIT " + json.dumps({k: m[k] for k in ("n", "rms_mm", "max_mm", "mm_per_px")})); return m
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fit": fit(); sys.exit(0)
    c = Cell("pp_" + cmd); b = c.b; out = {"cmd": cmd}
    if cmd in ("cycle", "place"):
        dx, dy = float(sys.argv[2]), float(sys.argv[3]); name = sys.argv[4] if len(sys.argv) > 4 else "c"
        if not c.holding(): out.update(ok=False, why="nothing is held")
        else:
            at = c.place(dx, dy); out["placed_at"] = at
            if cmd == "cycle":
                px = c.observe(name); out["px"] = px
                with open(SAMPLES, "a") as f: f.write(json.dumps({"t": round(time.time(), 1), "name": name, "cmd": [dx, dy], "at": at, "px": px}) + "\n")
                g = c.grasp(at[0], at[1], name); out.update(ok=g["ok"], why=g.get("why"), arrive=g.get("arrive"), close_done=g.get("close_done"), lifted=g.get("lifted_hand"))
    elif cmd == "observe": out["px"] = c.observe(sys.argv[2] if len(sys.argv) > 2 else "obs")
    elif cmd == "grasp":
        g = c.grasp(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "g"); out.update(ok=g["ok"], why=g.get("why"), arrive=g.get("arrive"), close_done=g.get("close_done"), closed=g.get("closed_hand"), lift_log=g.get("lift_log"), lifted=g.get("lifted_hand"))
    elif cmd == "track":
        name = sys.argv[2] if len(sys.argv) > 2 else "t"; m = load_map()
        if c.holding(): out.update(ok=False, why="already holding something")
        elif not m: out.update(ok=False, why="no pixel map yet (run cycles, then fit)")
        else:
            px = c.observe(name); out["px"] = px
            if not px: out.update(ok=False, why="chocolate not found by the camera")
            else:
                dx, dy = px_to_offset(px[0], px[1], m); out["offset_mm"] = [round(dx * 1000, 1), round(dy * 1000, 1)]
                if not (REACH["dx"][0] <= dx <= REACH["dx"][1] and REACH["dy"][0] <= dy <= REACH["dy"][1]): out.update(ok=False, why="chocolate is outside the reachable area")
                else:
                    g = c.grasp(dx, dy, name); out.update(ok=g["ok"], why=g.get("why"), arrive=g.get("arrive"), close_done=g.get("close_done"), lifted=g.get("lifted_hand"))
    out["soc"] = b.rob["bms"]["soc"]; b.mark("pick_place", **out); print("PP " + json.dumps(out, ensure_ascii=False), flush=True); b.close()
