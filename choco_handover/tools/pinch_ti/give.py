"""掴んだチョコを、差し出された人の手のひらへ渡す (上から落とす)。
usage: give.py [name] [wait_s=60]
1. チョコを持ったまま観察姿勢で待つ。頭部カメラ (HD + ステレオ) で開いた手のひらを探す。
2. 手のひらが 3 回続けてほぼ同じ場所 (3cm 以内) に見え、腕の届く範囲なら、手のひらの真上 7cm へ腕を伸ばす。
3. 落とす前の最終確認: もう一度手のひらを探し、まだそこ (つまみ点の真下から 6cm 以内) にあることを確かめる。
   無ければ落とさない。手が動いていたら狙い直す (2 回まで)。それでも合わなければ観察姿勢へ戻る。
4. 指を開いて落とし、指の近接値が消えたことを確かめて、観察姿勢へ戻る。最後に GIVE {...} を出す。

位置は「ロボットの手首の金具」と「人の手のひら」を同じステレオ処理で測った差で求める (カメラ校正の誤差が打ち消し合う)。
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision, palm as palm_mod, cv2, hd_hand_servo as hs
from open_palm import OpenPalmDetector
from stereo_depth import StereoProcessor
T = json.load(open(ROOT / "docs/data/teach_feet_down_20261112/teach_hold.json")); OPEN_T = T["open_shape"]; Q_G = np.array(T["released_open"]["q_right"]); _, R_G = fk(Q_G)
_q0, _, _ = ik(pinch(Q_G) + np.array([-0.03, -0.03, 0.0]), R_G, Q_G); OBS, _, _ = ik(pinch(Q_G) + np.array([-0.03, -0.03, 0.09]), R_G, _q0)   # 待機姿勢: 掴んだ場所の 9cm 上・少し手前右 (持っているチョコがカメラに写る)
DROP_H = 0.07; C_HAND = np.array([0.13, 0.0, 0.0])      # 手首の金具から見たつまみ点 (手首座標, 概算)
name = sys.argv[1] if len(sys.argv) > 1 else "give"; WAIT = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
b = Bot("give_" + name); res = {"trial": name, "ok": False}; det = OpenPalmDetector(); sp = StereoProcessor(); CFG = palm_mod.CFG
def finish(code=0):
    res["soc"] = b.rob["bms"]["soc"]; b.mark("give_result", **res); print("GIVE " + json.dumps(res, ensure_ascii=False), flush=True); b.close(); sys.exit(code)
def holding():
    h = b.hand(); return h["prox_k"][1] > 500 or h["prox_k"][0] > 250
def see(save=None):
    """(手のひら, 手首の金具) の 3D 位置 [胴体座標, 同じステレオ処理] を返す。見えなければ None。"""
    jpg = b.grab_hd(); st = b.stereo
    if not jpg or not st: return (None, None), None
    hd = cv2.imdecode(np.frombuffer(jpg, np.uint8), 1); sr = sp.compute(cv2.imdecode(np.frombuffer(st[0], np.uint8), 1), cv2.imdecode(np.frombuffer(st[1], np.uint8), 1), CFG)
    palms = [p for p in det.detect(hd, sr, kin, b.rob["q_aux"][1], 0.0, CFG) if p.center_torso_m is not None and p.is_open]
    p = max(palms, key=lambda x: x.score) if palms else None; wrist3d = None
    # 自分の手の位置: 指に挟んでいるチョコを目印にする。順運動学で予想した画素の近くだけを探す (チラシなどの黄色を拾わないため)
    pred = hs.torso_to_hd_px(pinch(b.qR()), kin, b.rob["q_aux"][1], 0.0, CFG)
    if pred is not None:
        u0, v0 = int(pred[0]), int(pred[1]); roi = (max(0, u0 - 130), max(0, v0 - 130), min(1280, u0 + 130), min(720, v0 + 130)); c = vision.find_choco(hd, roi=roi) if roi[2] - roi[0] > 80 and roi[3] - roi[1] > 80 else None
        if c:
            pts21 = np.zeros((21, 2)); ring = np.array([[c["u"] + 9 * np.cos(a), c["v"] + 9 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 7)[:-1]]); pts21[[0, 1, 5, 9, 13, 17]] = ring
            f = OpenPalmDetector._palm_3d(pts21, sr)
            if f is not None: wrist3d = np.asarray(kin.cam_point_to_torso(f[0], b.rob["q_aux"][1], 0.0, CFG), dtype=float)
            if save: cv2.circle(hd, (int(c["u"]), int(c["v"])), 16, (255, 0, 0), 2)
    if save:
        if p:
            for (x, y) in p.landmarks_hd or []: cv2.circle(hd, (x, y), 3, (0, 255, 0), -1)
            cv2.circle(hd, tuple(p.hd_px), 14, (0, 0, 255), 2)
        cv2.imwrite(f"{OUT}/{name}_{save}.jpg", hd)
    return (np.array(p.center_torso_m, dtype=float), p) if p else (None, None), wrist3d
def aim(palm3d, wrist3d):
    """手のひらの真上 DROP_H にチョコが来るための、つまみ点の目標 (順運動学の座標) と到達可否。"""
    q = np.array(b.qR()); pinch_seen = wrist3d; d = palm3d + np.array([0, 0, DROP_H]) - pinch_seen; tgt = pinch(q) + d
    qs = q; pe = re = 0.0
    for a in np.linspace(0.25, 1.0, 4): qs, pe, re = ik(pinch(q) + a * d, R_G, qs)
    return tgt, qs, d, (pe < 4.0 and re < 20.0 and np.linalg.norm(d) < 0.30)
if not holding(): res["why"] = "nothing is held"; finish(1)
if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
b.move(OBS, "gv_obs", speed=0.12); b.pump(1.0); b.say("holding. watching for an open palm (max", WAIT, "s)")
t0 = time.time(); streak = []; plan = None; last_msg = 0
while time.time() - t0 < WAIT:
    (p3, p), w3 = see(save="watch" if int(time.time() - t0) % 5 == 0 else None)
    if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
    if p3 is None or w3 is None:
        streak = []
        if time.time() - last_msg > 5: last_msg = time.time(); b.say(f"  t={time.time()-t0:4.0f}s palm seen: {p3 is not None}, held chocolate seen: {w3 is not None}")
        b.pump(0.2); continue
    streak = [s for s in streak if np.linalg.norm(s[0] - p3) < 0.03] + [(p3, w3)]
    tgt, qs, d, ok = aim(p3, w3)
    if time.time() - last_msg > 2: last_msg = time.time(); b.say(f"  palm at {np.round(p3,3).tolist()} px {p.hd_px} (stable x{len(streak)}), need to move {np.round(d*100,1).tolist()} cm, reachable {ok}")
    if len(streak) >= 3 and ok: plan = (np.mean([s[0] for s in streak], axis=0), tgt, qs, d); break
    b.pump(0.15)
if plan is None: res["why"] = "no reachable open palm was seen"; finish(2)
res["first_palm"] = [round(float(v), 3) for v in plan[0]]; res["move_cm"] = [round(float(v) * 100, 1) for v in plan[3]]
for attempt in range(3):
    palm0, tgt, qs, d = plan; q = np.array(b.qR()); mid, _, _ = ik(pinch(q) + 0.5 * d + np.array([0, 0, 0.03]), R_G, q)
    b.say("reaching toward the palm, move cm", np.round(d * 100, 1).tolist()); b.move(mid, f"gv_mid{attempt}", speed=0.08); b.move(qs, f"gv_reach{attempt}", speed=0.06); b.pump(1.0)
    if not holding(): res["why"] = "the chocolate was lost on the way"; b.move(OBS, "gv_back", speed=0.10); finish(3)
    # --- 最終確認: まだ手があるか ---
    seen = []
    for k in range(6):
        (p3, p), w3 = see(save=f"final{attempt}" if k == 0 else None)
        if p3 is not None: seen.append((p3, w3))
        b.pump(0.25)
    if len(seen) < 2:
        res.setdefault("final_checks", []).append({"attempt": attempt, "palm_frames": len(seen), "decision": "no palm -> do not drop"}); b.say("FINAL CHECK: palm not seen -> not dropping"); plan = None; break
    pm = np.median([s[0] for s in seen], axis=0); ws = [s[1] for s in seen if s[1] is not None]; _, R = fk(b.qR())
    if ws: off = pm[:2] - np.median(ws, axis=0)[:2]          # つまみ点の真下から手のひら中心までの水平ずれ
    else: off = pm[:2] - (palm0[:2])
    chk = {"attempt": attempt, "palm_frames": len(seen), "palm": [round(float(v), 3) for v in pm], "horizontal_offset_cm": [round(float(v) * 100, 1) for v in off]}
    if np.linalg.norm(off) <= 0.06:
        chk["decision"] = "palm is under the chocolate -> drop"; res.setdefault("final_checks", []).append(chk); b.say("FINAL CHECK ok:", chk)
        b.hand_set(OPEN_T, 500, 1.2); b.pump(0.8); h = b.hand(); b.save_hd(name + "_dropped"); res.update(ok=True, released=bool(h["prox_k"][1] < 300), hand_after=h); break
    chk["decision"] = "palm moved -> re-aim"; res.setdefault("final_checks", []).append(chk); b.say("FINAL CHECK: palm moved", chk)
    if ws and attempt < 2:
        tgt, qs, d, ok = aim(pm, np.median(ws, axis=0))
        if ok: plan = (pm, tgt, qs, d); continue
    plan = None; break
if not res["ok"] and "why" not in res: res["why"] = "final check failed (the palm was not under the chocolate) - kept holding"
b.move(OBS, "gv_back", speed=0.10); finish(0 if res["ok"] else 4)
