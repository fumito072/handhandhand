#!/usr/bin/env python3
"""
掴んだチョコを持ち上げて待ち、カメラに人の手が入ってきたら、その手のひらの上に置く。

    .venv/bin/python choco_handover/tools/place_on_palm.py palmtest    # ロボットは動かさず、手の検出結果だけ表示する
    .venv/bin/python choco_handover/tools/place_on_palm.py hovertest   # 手の上空まで動いて止まる (手のひらへは降りない)
    .venv/bin/python choco_handover/tools/place_on_palm.py give        # 持っている状態から手のひらへ置く
    .venv/bin/python choco_handover/tools/place_on_palm.py auto        # 掴む → 持ち上げる → 手のひらへ置く

手は「開いている・手のひらが上・腕が届く範囲にある・ロボット自身の手から離れている」状態で 1 秒以上止まったときに採用する。
ロボットの手で隠れて見えなくなる前に位置を固定し、見えている間に手が動いたら止めて測り直す。
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import grasp_thumb_middle as gtm  # noqa: E402
from kinematics import R1Kinematics  # noqa: E402

ZONE_MIN = np.array([0.22, -0.38, -0.06])   # 手を受け付ける範囲 (胴体座標 [m])。奥の人や床の物を拾わない
ZONE_MAX = np.array([0.50, 0.16, 0.28])     # 前方 50cm より先は置く高さまで腕が届かない (IK で確認)
ROBOT_HAND_BELOW_M = 0.05    # ロボットの手よりこれ以上低い検出だけを人の手として受け付ける (手の甲は手首より高い位置に映る)
ROBOT_HAND_CLEAR_M = 0.10    # ロボット自身の手首・指先からこれより近い検出は、自分の手の誤検出とみなす
CARRY_ABOVE_TABLE_M = 0.14   # 手を待つときのチョコの高さ (机から)
HOVER_ABOVE_PALM_M = 0.12    # 手のひらの上空で一度止まる高さ
PLACE_ABOVE_PALM_M = 0.03    # 近接センサーが反応しないまま、はさむ点がこの高さまで下りたら止める (奥行きの誤差 ±2cm を見込む)
# 降り始めより近接値がこれだけ増えたら手のひらが近いと判断する [千]。チョコを持つ親指と中指はチョコに反応しているので使わない
# 人差指と小指は持っている間ずっと 0、薬指はチョコの影響で 3〜5万あるため、しきい値を分ける
PROX_RISE_K = {1: 20, 3: 40, 4: 20}   # 指番号 (1=人差指, 3=薬指, 4=小指): 増加量
STILL_RADIUS_M = 0.03        # この範囲で止まっていれば「手が止まった」
STILL_SECONDS = 1.0
MOVED_LIMIT_M = 0.05         # 固定した位置からこれ以上動いたら測り直す
SAME_HAND_RADIUS_M = 0.20    # 固定した位置からこの範囲の検出だけを同じ手として扱う
WAIT_PALM_S = 90.0
APPROACH_SPEED_M_S = 0.07
DESCEND_SPEED_M_S = 0.015
MAX_ATTEMPTS = 3
MAX_ROT_ERR_DEG = 35.0       # 持ったまま運ぶ間は、手首の向きが教示から多少ずれても持ち方に影響しない


def set_palm_mode(on: bool) -> None:
    """手を待つ間はチョコの認識を止めて、手の検出の回転を速くする。"""
    req = urllib.request.Request(f"http://127.0.0.1:8890/api/tm/palm_{'on' if on else 'off'}", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=4) as r:
        r.read()


class Ctx:
    def __init__(self, kin: R1Kinematics) -> None:
        self.kin = kin
        last = json.loads(gtm.LAST.read_text(encoding="utf-8"))
        if "R_star" not in last:
            raise SystemExit("掴んだときの記録がありません。先に「掴む」を実行してください")
        self.R = np.array(last["R_star"], dtype=np.float64)
        self.off = np.array(last["off"], dtype=np.float64)
        self.z_table = float(last["z_table_pinch"])

    def pinch(self, q5) -> np.ndarray:
        p, R = self.kin.arm_fk(gtm.SIDE, list(q5), (0.0, 0.0, 0.0))
        return np.asarray(p) + np.asarray(R) @ self.off

    def line(self, p_from, p_to, q_seed, step: float = 0.025):
        """p_from から p_to まで、はさむ点が直線で動く関節列と最大誤差 (mm, 度) を返す。"""
        p_from, p_to = np.asarray(p_from, dtype=np.float64), np.asarray(p_to, dtype=np.float64)
        n = max(2, math.ceil(float(np.linalg.norm(p_to - p_from)) / step))
        chain, q_prev, worst_p, worst_r = [], q_seed, 0.0, 0.0
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            q, e_p, e_r = gtm.ik_pose(self.kin, p_from + (p_to - p_from) * s, self.R, q_prev, tcp=self.off, iters=120)
            chain.append(q)
            q_prev = q
            worst_p, worst_r = max(worst_p, e_p), max(worst_r, e_r)
        return chain, worst_p, worst_r


def robot_hand_points(kin: R1Kinematics, rob: dict) -> list:
    q = rob["q_arm"]
    out = []
    for side, q5 in (("left", q[:5]), ("right", q[5:10])):
        p, R = kin.arm_fk(side, list(q5), (0.0, 0.0, 0.0))
        p, R = np.asarray(p), np.asarray(R)
        out += [p, p + R @ np.array([0.12, 0.0, 0.0]), p + R @ np.array([0.23, 0.0, 0.0])]
    return out


def judge(kin: R1Kinematics, palm: dict, rob: dict) -> str:
    """手のひらを受け付けるなら ""、受け付けないならその理由を返す。"""
    if palm.get("center_torso_m") is None:
        return "距離が測れない"
    p = np.array(palm["center_torso_m"], dtype=np.float64)
    if p[0] > ZONE_MAX[0]:
        return f"遠すぎます (前 {p[0] * 100:.0f}cm)。ロボットに近づけてください"
    if np.any(p < ZONE_MIN) or np.any(p > ZONE_MAX):
        return "腕の範囲外"
    for h in robot_hand_points(kin, rob):
        if float(np.linalg.norm(p[:2] - h[:2])) < ROBOT_HAND_CLEAR_M and float(p[2]) > float(h[2]) - ROBOT_HAND_BELOW_M:
            return "ロボット自身の手の近く"
    if not palm.get("is_open"):
        return f"手が開いていない (指 {palm.get('open_fingers')} 本)"
    if not palm.get("palm_up"):
        return "手のひらが上を向いていない"
    return ""


def fresh_packet(last_t):
    st = gtm.status()
    t = st.get("t")
    if t == last_t or t is None or time.time() - float(t) > 3.0:
        return None, last_t
    return st, t


def wait_palm(kin: R1Kinematics, timeout: float) -> np.ndarray | None:
    samples, last_t, t0, last_note = [], None, time.time(), ""
    while time.time() - t0 < timeout:
        st, last_t = fresh_packet(last_t)
        if st is None:
            time.sleep(0.05)
            continue
        rob = (st.get("telemetry") or {}).get("robot") or {}
        palms = st.get("open_palms") or []
        now = time.time()
        ok = [np.array(p["center_torso_m"], dtype=np.float64) for p in palms if rob.get("q_arm") and not judge(kin, p, rob)]
        if ok:
            ref = samples[-1][1] if samples else ok[0]
            samples.append((now, min(ok, key=lambda p: float(np.linalg.norm(p - ref)))))
        elif samples and now - samples[-1][0] > 1.0:
            samples.clear()
        note = "手を検出" if ok else (judge(kin, palms[0], rob) if palms and rob.get("q_arm") else "手が見えていません")
        kind = note.split(" (")[0]     # 距離の数字だけが変わっても同じ行を繰り返さない
        if kind != last_note:
            print(f"   ... {note}", flush=True)
            last_note = kind
        recent = [s for s in samples if now - s[0] <= 2.5]
        if len(recent) >= 3 and recent[-1][0] - recent[0][0] >= STILL_SECONDS:
            P = np.array([s[1] for s in recent])
            med = np.median(P, axis=0)
            if float(np.linalg.norm(P - med, axis=1).max()) <= STILL_RADIUS_M:
                return med
        time.sleep(0.05)
    return None


def check_same_hand(kin: R1Kinematics, lock: np.ndarray, seconds: float) -> str:
    """固定した手が今もそこにあるかを確かめる。"same" | "moved" | "unseen" を返す。"""
    last_t, t0, seen = None, time.time(), []
    while time.time() - t0 < seconds:
        st, last_t = fresh_packet(last_t)
        if st is None:
            time.sleep(0.05)
            continue
        rob = (st.get("telemetry") or {}).get("robot") or {}
        for p in st.get("open_palms") or []:
            if p.get("center_torso_m") is None or not rob.get("q_arm") or judge(kin, p, rob):
                continue     # ロボット自身の手の誤検出などは数えない
            dist = float(np.linalg.norm(np.array(p["center_torso_m"], dtype=np.float64)[:2] - lock[:2]))
            if dist <= SAME_HAND_RADIUS_M:
                seen.append(dist)
        if len(seen) >= 2:
            return "moved" if float(np.median(seen)) > MOVED_LIMIT_M else "same"
        time.sleep(0.05)
    if seen:
        return "moved" if float(np.median(seen)) > MOVED_LIMIT_M else "same"
    return "unseen"


def run_path(ctx: Ctx, q_path10, aux, duration: float, label: str, lock=None, sag: float | None = None) -> str:
    """経路を送って完了まで見張る。止めたときは腕はその場で保持。

    "done" (最後まで動いた) | "moved" (手が動いた) | "near" (近接センサーが手のひらに反応) | "contact" (何かに触れた) を返す。
    """
    rep = gtm.send("arm_move_path", q_path=[[float(v) for v in q] for q in q_path10], aux=[float(v) for v in aux],
                   duration=duration, label=label, replace=True)
    if not rep or not rep.get("ok"):
        raise RuntimeError(f"arm_move_path 失敗 ({label}): {rep}")
    base = gtm.hand_state() if sag is not None else None
    t0, seen, last_t, hist, far_frames, peak = time.time(), False, None, [], 0, [0, 0, 0, 0, 0]

    def finish(result: str) -> str:
        if sag is not None:
            print(f"   降下中の近接値の最大 (千) 親指〜小指: {peak} / 開始時: {base['prox_k']} → {result}", flush=True)
        return result

    while time.time() - t0 < duration * 3.0 + 8.0:
        st = gtm.status()
        rob = (st.get("telemetry") or {}).get("robot") or {}
        if not st.get("r1_connected") or not rob:
            raise RuntimeError("R1 との接続が切れました")
        stage = rob.get("stage")
        if stage in gtm.FAULT_STAGES or not rob.get("engaged", True):
            raise RuntimeError(f"安全停止: stage={stage} fault={rob.get('last_fault')!r}")
        if stage == label:
            seen = True
        if seen and stage == "holding":
            return finish("done")
        if lock is not None and st.get("t") != last_t:
            last_t = st.get("t")
            # ロボットの手が重なると奥行きの測定が乱れるので、水平方向だけを 2 フレーム続けて見る
            moved = False
            for p in st.get("open_palms") or []:
                if p.get("center_torso_m") is None:
                    continue
                c = np.array(p["center_torso_m"], dtype=np.float64)
                dist = float(np.linalg.norm(c[:2] - lock[:2]))
                moved = moved or (MOVED_LIMIT_M < dist <= SAME_HAND_RADIUS_M and not judge(ctx.kin, p, rob))
            far_frames = far_frames + 1 if moved else 0
            if far_frames >= 2:
                gtm.send("arm_hold_here")
                return finish("moved")
        if sag is not None and seen and rob.get("cmd_arm") and len(rob["cmd_arm"]) == 10:
            # 空中では実際の高さが指令より sag だけ低い。そこから 8mm 以上持ち上がった = 下から支えられている
            lifted = (ctx.pinch(rob["q_arm"][5:10])[2] - ctx.pinch(rob["cmd_arm"][5:10])[2]) - sag > 0.008
            hs = gtm.hand_state()
            pressed = max(abs(hs["Fn"][i] - base["Fn"][i]) for i in (0, 2)) >= 0.5
            peak = [max(a, b) for a, b in zip(peak, hs["prox_k"])]
            near = any(hs["prox_k"][i] - base["prox_k"][i] >= k for i, k in PROX_RISE_K.items())
            now = time.time()
            hist.append((now, lifted or pressed, near))
            recent = [h for h in hist if now - h[0] <= 0.4]
            if len(recent) >= 2 and all(h[2] for h in recent):
                gtm.send("arm_hold_here")
                return finish("near")
            if len(recent) >= 2 and all(h[1] for h in recent):
                gtm.send("arm_hold_here")
                return finish("contact")
        time.sleep(0.1)
    gtm.send("arm_hold_here")
    raise RuntimeError(f"{label} の完了確認がタイムアウトしました")


def raise_to_carry(ctx: Ctx, aux) -> None:
    r = gtm.robot()
    q5, q10 = r["q_arm"][5:10], list(r["q_arm"])
    p = ctx.pinch(q5)
    z_carry = ctx.z_table + CARRY_ABOVE_TABLE_M
    if p[2] >= z_carry - 0.01:
        return
    print(f"[手渡し] 持ち上げて待つ (机から {CARRY_ABOVE_TABLE_M * 100:.0f}cm)", flush=True)
    chain, _, _ = ctx.line(p, [p[0], p[1], z_carry], q5)
    gtm.move_path([gtm.merge(ctx.kin, q10, q) for q in chain], aux, max(3.0, (z_carry - p[2]) / APPROACH_SPEED_M_S), "tm_carry")


def do_give(ctx: Ctx, hover_only: bool = False) -> bool:
    kin = ctx.kin
    r0 = gtm.robot()
    if not r0.get("engaged"):
        raise SystemExit("腕に制御が入っていません。先に「掴む」を実行してください")
    aux = list(r0["q_aux"])
    raise_to_carry(ctx, aux)
    set_palm_mode(True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[手渡し {attempt}] 手のひらを上にして、腕が届く位置に差し出してください", flush=True)
        palm = wait_palm(kin, WAIT_PALM_S)
        if palm is None:
            print("[手渡し] 手が来なかったので、持ったまま待機します (もう一度「手のひらへ置く」で再開)", flush=True)
            return False
        print(f"   手のひらの位置 (固定): {np.round(palm, 3).tolist()}", flush=True)

        r = gtm.robot()
        q5, q10 = r["q_arm"][5:10], list(r["q_arm"])
        p_now = ctx.pinch(q5)
        p_hover = np.array([palm[0], palm[1], max(palm[2] + HOVER_ABOVE_PALM_M, ctx.z_table + 0.08)])
        chain, e_p, e_r = ctx.line(p_now, p_hover, q5)
        if e_p > 6.0 or e_r > MAX_ROT_ERR_DEG:
            print(f"   この位置には届きません ({e_p:.1f}mm / {e_r:.1f}°)。少し手を動かしてください", flush=True)
            continue

        duration = max(2.5, float(np.linalg.norm(p_hover - p_now)) / APPROACH_SPEED_M_S)
        print(f"   手の上空へ移動 ({duration:.1f} 秒)", flush=True)
        if run_path(ctx, [gtm.merge(kin, q10, q) for q in chain], aux, duration, f"tm_to_palm{attempt}", lock=palm) == "moved":
            print("   手が動いたので止めて測り直します", flush=True)
            continue
        time.sleep(0.6)

        seen = check_same_hand(kin, palm, 3.0)
        print(f"   上空での手の確認: {seen}", flush=True)
        if hover_only:
            print("[上空テスト] ここで止めます (手のひらへは降りません)", flush=True)
            return True
        if seen == "moved":
            continue
        if seen == "unseen":
            print("   手がロボットの手に隠れて見えません。近接センサーを見ながら降ります", flush=True)

        r = gtm.robot()
        q5, q10 = r["q_arm"][5:10], list(r["q_arm"])
        p_act, p_cmd = ctx.pinch(q5), ctx.pinch(r["cmd_arm"][5:10])
        sag = float(np.clip(p_act[2] - p_cmd[2], -0.04, 0.0))
        z_cmd_goal = palm[2] + PLACE_ABOVE_PALM_M - sag
        p_start = np.array([p_cmd[0], p_cmd[1], p_cmd[2]])
        chain, e_p, e_r = ctx.line(p_start, [p_start[0], p_start[1], z_cmd_goal], r["cmd_arm"][5:10])
        if e_p > 6.0 or e_r > MAX_ROT_ERR_DEG:
            print(f"   降りる経路が解けません ({e_p:.1f}mm / {e_r:.1f}°)", flush=True)
            continue
        duration = max(2.0, abs(p_start[2] - z_cmd_goal) / DESCEND_SPEED_M_S)
        print(f"   ゆっくり降ろす ({duration:.1f} 秒、腕のたれ {sag * 1000:.0f}mm を補正)", flush=True)
        how = run_path(ctx, [gtm.merge(kin, q10, q) for q in chain], aux, duration, f"tm_lower{attempt}", sag=sag)
        time.sleep(0.5)
        r = gtm.robot()
        p_end = ctx.pinch(r["q_arm"][5:10])
        if how == "done":
            # 手の接近が数値に出ていないときは放さない (手が無い場所へ落とさない)
            print(f"   手のひらの {(p_end[2] - palm[2]) * 100:.1f}cm 上まで降りましたが近接センサーが反応しません。放さずに上へ戻ります", flush=True)
            up, _, _ = ctx.line(p_end, p_hover, r["q_arm"][5:10])
            gtm.move_path([gtm.merge(kin, q10, q) for q in up], aux, 3.0, f"tm_reup{attempt}")
            continue
        print(f"   {'何かに触れた' if how == 'contact' else '手のひらの接近を検出'}: 測った手のひらから {(p_end[2] - palm[2]) * 100:.1f}cm の高さで止まりました", flush=True)

        print("[手渡し] 指をゆっくり開いて手のひらへ落とす", flush=True)
        gtm.hand(gtm.SHAPE_OPEN, 1500, 2.2)
        print("   開いた後:", gtm.hand_state(), flush=True)

        print("[手渡し] 真上へ離れて初期姿勢へ戻る", flush=True)
        r = gtm.robot()
        p_end = ctx.pinch(r["q_arm"][5:10])
        up, _, _ = ctx.line(p_end, [p_end[0], p_end[1], p_end[2] + 0.10], r["q_arm"][5:10])
        gtm.move_path([gtm.merge(kin, r["q_arm"], q) for q in up], aux, 3.0, "tm_after_place")
        gtm.do_return()
        print("[結果] 手のひらへ置きました", flush=True)
        return True
    print("[手渡し] 規定回数内に置けなかったので、持ったまま待機します", flush=True)
    return False


def do_palmtest(kin: R1Kinematics, seconds: float = 40.0) -> None:
    set_palm_mode(True)
    print(f"[検出テスト] {seconds:.0f} 秒間、手の検出結果を表示します (ロボットは動きません)", flush=True)
    last_t, t0 = None, time.time()
    while time.time() - t0 < seconds:
        st, last_t = fresh_packet(last_t)
        if st is None:
            time.sleep(0.05)
            continue
        rob = (st.get("telemetry") or {}).get("robot") or {}
        palms = st.get("open_palms") or []
        desc = [{"torso": p.get("center_torso_m"), "lr": p.get("label"), "score": p.get("score"), "open": p.get("open_fingers"),
                 "up": p.get("palm_up"), "npts": p.get("depth_points"), "ng": judge(kin, p, rob) if rob.get("q_arm") else "?"} for p in palms]
        print(f"t={time.time() - t0:5.1f}s vision={st.get('vision_ms')}ms {json.dumps(desc, ensure_ascii=False) if desc else '手なし'}", flush=True)
    print("[検出テスト] 終了", flush=True)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    kin = R1Kinematics()
    try:
        if mode == "palmtest":
            do_palmtest(kin)
        elif mode in ("give", "hovertest"):
            do_give(Ctx(kin), hover_only=mode == "hovertest")
        elif mode == "auto":
            rec = gtm.do_grasp(dry=False)
            if not rec or rec.get("result") != "held":
                raise SystemExit(f"チョコを掴めていないので手渡しはしません (結果: {(rec or {}).get('result')})")
            do_give(Ctx(kin))
        else:
            raise SystemExit("usage: place_on_palm.py palmtest|hovertest|give|auto")
    except RuntimeError as exc:
        print("[エラー]", exc, flush=True)
        try:
            gtm.send("arm_hold_here")
        except Exception:
            pass
        raise SystemExit(1)
    finally:
        try:
            set_palm_mode(False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
