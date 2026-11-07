#!/usr/bin/env python3
"""
親指 (付け根 720) × 中指でつまみ、薬指を添える手の形で、カメラで見つけたチョコを掴む (検証用・単体スクリプト)。

教示データ docs/data/grasp_offset_teach.json (teach_grasp_offset.py で作成) の
「チョコを掴むときのチョコ中心の位置 (手首座標)」と「手首の向き」を、新しいチョコの位置・向きへ移して手首の目標を決める。

    .venv/bin/python choco_handover/tools/grasp_thumb_middle.py plan    # 腕は動かさず、計画と IK の誤差だけ表示
    .venv/bin/python choco_handover/tools/grasp_thumb_middle.py grasp   # 実行: 上空 → 降下 → 握る → 持ち上げて保持
    .venv/bin/python choco_handover/tools/grasp_thumb_middle.py return  # 持ち上げた位置から初期姿勢へ戻って脱力 (手は開く)

チョコの位置は腕が動く前にロックする (手の甲で隠れても、ロックした位置を使う)。
"""
from __future__ import annotations

import datetime as dt
import json
import socket
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from config import ConfigStore  # noqa: E402
from kinematics import R1Kinematics  # noqa: E402

SIDE = "right"
BRIDGE = ("192.168.123.164", 8891)
TEACH = PKG.parent / "docs" / "data" / "grasp_offset_teach.json"
RUNS = PKG.parent / "docs" / "data" / "grasp_thumb_middle_runs.jsonl"
LAST = PKG.parent / "docs" / "data" / "grasp_thumb_middle_last.json"

# 右腕の関節範囲 (r1_arm_daemon.cpp の ARM_MIN/ARM_MAX の右腕分)
Q_MIN = np.array([-2.60, -2.35, -1.80, -0.90, -1.85])
Q_MAX = np.array([1.80, 0.18, 1.80, 2.05, 1.85])

SHAPE_ROTATE = [0, 720, 49, 49, 49, 49]
SHAPE_FLEX = [470, 720, 49, 49, 49, 49]
SHAPE_OPEN = [430, 720, 49, 200, 220, 49]   # 親指と中指のすき間をチョコ幅より広げておく (触れて押さない余裕)
SHAPE_CLOSE = [490, 720, 49, 340, 320, 49]
HOVER_DZ = 0.06
GRASP_DZ = 0.003          # 教示した高さより 3mm 上で止める (机に押し付けない)
FAULT_STAGES = ("fault_hold", "damped_estop", "fault_stale_lowstate")


# ---------------------------------------------------------------- 通信
def status() -> dict:
    with urllib.request.urlopen("http://127.0.0.1:8890/api/status", timeout=4) as r:
        return json.load(r)


def send(method: str, **kw) -> dict | None:
    s = socket.create_connection(BRIDGE, timeout=5)
    rid = f"tm_{method}_{int(time.time() * 1000) % 100000}"
    body = json.dumps({"method": method, "id": rid, **kw}).encode()
    s.sendall(struct.pack(">IB", len(body) + 1, 1) + body)
    s.settimeout(1.0)
    buf, reply, t0 = b"", None, time.time()
    while time.time() - t0 < 8 and reply is None:
        try:
            chunk = s.recv(262144)
        except socket.timeout:
            continue
        buf += chunk
        while len(buf) >= 5:
            ln, mt = struct.unpack(">IB", buf[:5])
            if len(buf) < 4 + ln:
                break
            payload, buf = buf[5:4 + ln], buf[4 + ln:]
            if mt == 1:
                m = json.loads(payload)
                if m.get("type") == "reply" and m.get("id") == rid:
                    reply = m
    s.close()
    return reply


def robot() -> dict:
    st = status()
    rob = (st.get("telemetry") or {}).get("robot")
    if not st.get("r1_connected") or not rob:
        raise SystemExit("R1 に接続できていません（電源・LAN ケーブル・サーバーを確認してください）")
    return rob


def hand_state() -> dict:
    h = status()["telemetry"]["hands"][SIDE]
    fs = h["tactile"]["fingers"]
    return {
        "position": h["state"]["position"],
        "Fn": [round(f.get("normal_N", 0) or 0, 2) for f in fs],
        "Ft": [round(f.get("tangent_N", 0) or 0, 2) for f in fs],
        "prox_k": [int((f.get("proximity") or 0) / 1000) for f in fs],
    }


def wait_segment(label: str, timeout: float) -> None:
    t0, seen = time.time(), False
    while time.time() - t0 < timeout:
        r = robot()
        st = r.get("stage")
        if st in FAULT_STAGES or not r.get("engaged", True):
            raise RuntimeError(f"安全停止: stage={st} fault={r.get('last_fault')!r}")
        if st == label:
            seen = True
        if seen and st == "holding":
            return
        time.sleep(0.15)
    print(f"  (注意) {label} の完了確認がタイムアウト", flush=True)


def move(q10, aux, duration: float, label: str, max_speed: float | None = None, replace: bool = False) -> None:
    kw = dict(q=[float(v) for v in q10], aux=[float(v) for v in aux], duration=duration, label=label, replace=replace)
    if max_speed is not None:
        kw["max_speed"] = max_speed
    rep = send("arm_move", **kw)
    if not rep or not rep.get("ok"):
        raise RuntimeError(f"arm_move 失敗 ({label}): {rep}")
    wait_segment(label, max(duration * 4.0, 25.0))


def move_path(q_path, aux, duration: float, label: str) -> None:
    rep = send("arm_move_path", q_path=[[float(v) for v in q] for q in q_path], aux=[float(v) for v in aux],
               duration=duration, label=label, replace=True)
    if not rep or not rep.get("ok"):
        raise RuntimeError(f"arm_move_path 失敗 ({label}): {rep}")
    wait_segment(label, max(duration * 4.0, 25.0))


def hand(targets, duration_ms: int, settle: float) -> None:
    rep = send("hand_pose", side=SIDE, targets=list(targets), duration_ms=duration_ms)
    if not rep or not rep.get("ok"):
        raise RuntimeError(f"hand_pose 失敗: {rep}")
    time.sleep(settle)


# ---------------------------------------------------------------- 幾何
def ik_pose(kin: R1Kinematics, p_target, R_target, q_seed, w_rot_rad: float = 0.035, iters: int = 200, tcp=None):
    """点 (手首 + 手首の向き × tcp) の位置を厳密に、手首の向きを重み付きで合わせる 5 自由度の減衰最小二乗 IK。

    tcp を省略すると手首の位置そのものを合わせる。arm_fk の tcp_offset 引数は掛け方が教示データと違うため使わない。
    """
    q = np.clip(np.array(q_seed, dtype=np.float64), Q_MIN, Q_MAX)
    tcp_v = np.zeros(3) if tcp is None else np.asarray(tcp, dtype=np.float64)

    def point(qv):
        p, R = kin.arm_fk(SIDE, qv.tolist(), (0.0, 0.0, 0.0))
        return np.asarray(p) + np.asarray(R) @ tcp_v, np.asarray(R)

    def err(qv):
        p, R = point(qv)
        e_p = (p - p_target) / 0.001
        e_r = Rotation.from_matrix(R_target @ np.asarray(R).T).as_rotvec() / w_rot_rad
        return np.concatenate([e_p, e_r])

    lam = 1e-2
    for _ in range(iters):
        e = err(q)
        J = np.zeros((6, 5))
        for j in range(5):
            dq = np.zeros(5)
            dq[j] = 1e-5
            J[:, j] = (err(np.clip(q + dq, Q_MIN, Q_MAX)) - e) / 1e-5
        step = np.linalg.solve(J.T @ J + lam * np.eye(5), -J.T @ e)
        q_new = np.clip(q + np.clip(step, -0.1, 0.1), Q_MIN, Q_MAX)
        if np.linalg.norm(err(q_new)) < np.linalg.norm(e):
            q, lam = q_new, max(lam * 0.5, 1e-5)
        else:
            lam *= 4.0
        if np.linalg.norm(step) < 1e-6:
            break
    p, R = point(q)
    pos_err_mm = float(np.linalg.norm(p - p_target) * 1000.0)
    rot_err_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R_target @ R.T).as_rotvec())))
    return q, pos_err_mm, rot_err_deg


def lock_choco() -> dict:
    pts, yaws, last = [], [], None
    t0 = time.time()
    while time.time() - t0 < 4.0:
        c = status().get("choco")
        if c and c.get("centroid_torso_m"):
            pts.append(c["centroid_torso_m"])
            yaws.append(((float(c.get("yaw_deg") or 0.0) + 90.0) % 180.0) - 90.0)
            last = c
        time.sleep(0.15)
    if len(pts) < 3:
        raise SystemExit("チョコを認識できません。腕が机の上にない状態で、チョコをカメラから見える位置に置いてください")
    P = np.array(pts)
    med = np.median(P, axis=0)
    keep = np.linalg.norm(P[:, :2] - med[:2], axis=1) < 0.02   # 手など別物の誤検出を除く
    if int(keep.sum()) < 3 or int(keep.sum()) < 0.6 * len(P):
        raise SystemExit(f"チョコの認識がばらついています ({int(keep.sum())}/{len(P)} 回が一致)。動かさずに終了します")
    P = P[keep]
    Y = np.array(yaws)[keep]
    return {
        "centroid_torso_m": np.median(P, axis=0).round(4).tolist(),
        "yaw_deg": round(float(np.median(Y)), 1),
        "detections": int(keep.sum()),
        "table_bbox_torso": (last or {}).get("table_bbox_torso"),
    }


def plan(kin: R1Kinematics, choco: dict) -> dict:
    t = json.loads(TEACH.read_text(encoding="utf-8"))
    s = t["samples"][-1]
    R_t = np.array(s["wrist_R"], dtype=np.float64)
    off = np.array(s["offset_in_wrist_m"], dtype=np.float64)
    yaw_t = float(t["choco"]["yaw_deg"])
    z_c = float(t["choco"]["centroid_torso_m"][2])     # 同じ机なので高さは教示時のチョコ中心の高さを使う
    p_c = np.array([choco["centroid_torso_m"][0], choco["centroid_torso_m"][1], z_c], dtype=np.float64)
    d_yaw = ((float(choco["yaw_deg"]) - yaw_t + 90.0) % 180.0) - 90.0
    R_star = Rotation.from_euler("z", d_yaw, degrees=True).as_matrix() @ R_t
    # はさむ点 (= チョコ中心) を厳密に合わせる。掴む姿勢から上へ順に解き、つながった関節列にする
    p_grasp = p_c + np.array([0.0, 0.0, GRASP_DZ])
    q_grasp, eg_p, eg_r = ik_pose(kin, p_grasp, R_star, s["q5"], tcp=off)
    heights = [0.0, 0.006, 0.012, 0.02, 0.03, 0.045, 0.06, 0.08, 0.10]
    chain, errs, q_prev = [], [], q_grasp
    for h in heights:
        q_h, e_p, e_r = ik_pose(kin, p_grasp + np.array([0.0, 0.0, h]), R_star, q_prev, tcp=off, iters=120)
        chain.append(q_h)
        errs.append([round(h * 100, 1), round(e_p, 1), round(e_r, 1)])
        q_prev = q_h
    descent = list(reversed(chain))          # 10cm 上 → 掴む高さ
    return {
        "d_yaw_deg": round(d_yaw, 1),
        "choco_point_m": p_grasp.round(4).tolist(),
        "q_grasp": q_grasp.round(4).tolist(), "ik_grasp_err": [round(eg_p, 1), round(eg_r, 1)],
        "q_hover": descent[0].round(4).tolist(), "ik_hover_err": errs[-1][1:],
        "descent_err_cm_mm_deg": errs,
        "descent_q5": [q.round(4).tolist() for q in descent[1:]],
        "taught_q5": s["q5"],
    }


def merge(kin, base_q10, q5):
    return kin.merge_single_arm(list(base_q10), SIDE, list(q5))


# ---------------------------------------------------------------- 実行
def do_grasp(dry: bool) -> None:
    store = ConfigStore()
    store.load()
    cfg = store.cfg
    kin = R1Kinematics()
    r0 = robot()
    if r0.get("engaged") and not dry:
        raise SystemExit("腕にすでに制御が入っています。初期姿勢で脱力した状態から始めてください")
    choco = lock_choco()
    pl = plan(kin, choco)
    print("チョコ (ロック):", choco)
    print("計画:", json.dumps({k: v for k, v in pl.items() if k != "descent_q5"}, ensure_ascii=False))
    if pl["ik_grasp_err"][0] > 4.0 or pl["ik_grasp_err"][1] > 15.0 or pl["ik_hover_err"][0] > 6.0:
        raise SystemExit(f"掴む姿勢に届きません (掴む位置 {pl['ik_grasp_err'][0]}mm・向き {pl['ik_grasp_err'][1]}° / 上空 {pl['ik_hover_err'][0]}mm)")
    if dry:
        return

    q_init = list(r0["q_arm"])
    aux = list(r0["q_aux"])
    LAST.write_text(json.dumps({"initial_q10": q_init, "aux": aux, "plan": pl, "choco": choco}, ensure_ascii=False), encoding="utf-8")
    rec = {"time": dt.datetime.now().isoformat(timespec="seconds"), "choco": choco, "plan": pl}
    try:
        print("[1] 手の形を作る (親指を伸ばして回す → 曲げる → 受けの形)", flush=True)
        if abs(hand_state()["position"][1] - 720) > 25:
            hand(SHAPE_ROTATE, 1500, 1.8)
            hand(SHAPE_FLEX, 1500, 1.8)
        hand(SHAPE_OPEN, 1200, 2.0)

        print("[2] 腕の制御を入れる", flush=True)
        rep = send("arm_engage", suspended=1.0, grav_ff_scale=float(cfg.grav_ff_scale),
                   extra_hand_mass_kg=float(cfg.extra_hand_mass_kg), table_z_floor=-0.16)
        if not rep or not rep.get("ok"):
            raise RuntimeError(f"engage 失敗: {rep}")
        t0 = time.time()
        while time.time() - t0 < 14:
            r = robot()
            if r.get("engaged") and r.get("stage") == "holding":
                break
            time.sleep(0.2)

        print("[3] 机を避ける中継点を通ってチョコの上空へ", flush=True)
        wp0, wp1, wp2 = kin.build_safe_table_waypoints(q_init, SIDE, table_bbox=choco.get("table_bbox_torso"))
        move(wp0, aux, 2.2, "tm_side_clear", max_speed=0.30, replace=True)
        move(wp1, aux, 3.0, "tm_lift_behind_desk", max_speed=0.30)
        move(wp2, aux, 2.6, "tm_over_desk", max_speed=0.30)
        q10_hover = merge(kin, wp2, pl["q_hover"])
        move(q10_hover, aux, 3.0, "tm_hover", max_speed=0.30)
        time.sleep(0.4)

        # 教示データ (はさむ点のオフセット・手首の向き・机に指先が着いたときのはさむ点の高さ)
        t = json.loads(TEACH.read_text(encoding="utf-8"))
        off = np.array(t["samples"][-1]["offset_in_wrist_m"], dtype=np.float64)
        z_table_pinch = float(t["choco"]["centroid_torso_m"][2])
        R_star = Rotation.from_euler("z", pl["d_yaw_deg"], degrees=True).as_matrix() @ np.array(t["samples"][-1]["wrist_R"])
        axis = R_star[:2, 0] / (np.linalg.norm(R_star[:2, 0]) + 1e-9)   # 親指→中指 ≒ 手首から指先へ向かう向き (水平)

        def pinch_point(q5):
            p, R = kin.arm_fk(SIDE, list(q5), (0.0, 0.0, 0.0))
            return np.asarray(p) + np.asarray(R) @ off

        def vertical_chain(p_bottom, heights, q_seed):
            out, q_prev = [], q_seed
            for h in heights:
                q_h, _, _ = ik_pose(kin, p_bottom + np.array([0.0, 0.0, h]), R_star, q_prev, tcp=off, iters=120)
                out.append(q_h)
                q_prev = q_h
            return out

        def descend_until_contact(path_q10, duration, label, min_z=None) -> bool:
            """降下経路を送り、はさむ点が机より上で 0.5 秒下がらなくなったら (何かに乗った) その場で止める。止めたら True。"""
            rep = send("arm_move_path", q_path=[[float(v) for v in q] for q in path_q10], aux=[float(v) for v in aux],
                       duration=duration, label=label, replace=True)
            if not rep or not rep.get("ok"):
                raise RuntimeError(f"arm_move_path 失敗 ({label}): {rep}")
            t0, seen, hist = time.time(), False, []
            while time.time() - t0 < duration * 3.0 + 6.0:
                r = robot()
                st = r.get("stage")
                if st in FAULT_STAGES or not r.get("engaged", True):
                    raise RuntimeError(f"安全停止: stage={st} fault={r.get('last_fault')!r}")
                if st == label:
                    seen = True
                if seen and st == "holding":
                    return False
                if seen and r.get("cmd_arm") and len(r["cmd_arm"]) == 10:
                    now = time.time()
                    z = float(pinch_point(r["q_arm"][5:10])[2])
                    z_cmd = float(pinch_point(r["cmd_arm"][5:10])[2])
                    # 空中では腕は指令より約 17mm 低く垂れる。指令より 5mm 以上高い = 何かに乗って支えられている
                    blocked = (z - z_cmd) > 0.005 and z > (z_table_pinch + 0.005 if min_z is None else min_z)
                    hist.append((now, blocked))
                    recent = [b for tt, b in hist if now - tt <= 0.3]
                    if len(recent) >= 2 and all(recent):
                        send("arm_hold_here")      # 乗ったまま押し付けない (軽いチョコが倒れる・ずれるのを防ぐ)
                        time.sleep(0.4)
                        return True
                time.sleep(0.1)
            return False

        # [3.5] 机の高さを指先で実測する (チョコの横 7cm の何もない机面に触れた時のはさむ点の高さ)
        print("[3.5] チョコの横の机に指先を当てて、机の高さを実測", flush=True)
        p_probe = np.array(pl["choco_point_m"], dtype=np.float64) + np.array([0.0, -0.07, 0.0])
        probe_up = vertical_chain(p_probe, [0.07], robot()["q_arm"][5:10])[0]
        move(merge(kin, q10_hover, probe_up), aux, 2.5, "tm_probe_hover", max_speed=0.30)
        probe_chain = vertical_chain(p_probe, [0.05, 0.03, 0.015, 0.0, -0.015, -0.025], probe_up)
        descend_until_contact([merge(kin, q10_hover, q5) for q5 in probe_chain], 4.5, "tm_probe_down", min_z=-9.0)
        time.sleep(0.6)
        z_meas = float(pinch_point(robot()["q_arm"][5:10])[2])
        probe_dz = z_meas - z_table_pinch
        print(f"   机に触れた時のはさむ点の高さ {z_meas:.4f} (教示との差 {probe_dz * 1000:+.1f}mm)", flush=True)
        rec["table_probe_z"] = round(z_meas, 4)
        up_back = vertical_chain(p_probe, [0.07], robot()["q_arm"][5:10])[0]
        move(merge(kin, q10_hover, up_back), aux, 2.0, "tm_probe_up", max_speed=0.30)
        move(q10_hover, aux, 2.0, "tm_back_to_hover", max_speed=0.30)
        if abs(probe_dz) > 0.03:
            print("   (注意) 実測が教示と 3cm 以上違うため使わない", flush=True)
            probe_dz = 0.0
        z_table_pinch = z_table_pinch + probe_dz
        target = np.array(pl["choco_point_m"], dtype=np.float64) + np.array([0.0, 0.0, probe_dz])
        descent_q5 = [q.tolist() for q in vertical_chain(target, [0.08, 0.06, 0.045, 0.03, 0.02, 0.012, 0.006, 0.0], pl["q_hover"])]
        rec["descents"] = []
        for attempt in range(1, 5):
            print(f"[4.{attempt}] はさむ点を真下へ降ろす (目標 {np.round(target, 4).tolist()})", flush=True)
            path = [merge(kin, q10_hover, q5) for q5 in descent_q5]
            stopped = descend_until_contact(path, 9.0 if attempt == 1 else 7.0, f"tm_descend{attempt}")  # 慎重に下ろす
            time.sleep(0.6)
            q_act = robot()["q_arm"][5:10]
            p_act = pinch_point(q_act)
            hs = hand_state()
            dz_mm = (p_act[2] - z_table_pinch) * 1000.0
            prox_t, prox_m = hs["prox_k"][0], hs["prox_k"][2]
            touching = max(prox_t, prox_m) >= 50
            landed_high = dz_mm > 12.0 and touching   # 高いだけで指が何にも触れていなければ「乗った」ではない (誤停止)
            info = {"attempt": attempt, "target": np.round(target, 4).tolist(), "pinch_actual": np.round(p_act, 4).tolist(),
                    "dz_above_table_mm": round(dz_mm, 1), "contact_stop": bool(stopped), "landed_high": bool(landed_high), "hand": hs}
            rec["descents"].append(info)
            print(f"   はさむ点の高さ: 机より {dz_mm:+.1f}mm (途中停止 {stopped}) / 近接 親指 {prox_t}k 中指 {prox_m}k / 力 {hs['Fn']}", flush=True)
            false_stop = dz_mm > 12.0 and not touching
            near_table = dz_mm <= 12.0
            # 握るのは次の 2 パターンだけ: (1) 親指と中指の両方が反応 (2) 机まで降りて片方の指の腹が反応
            # すき間を広げた受けの形では、握る前の近接値は低め (2026-11-08 実測: 親指 8〜11万 / 中指 5〜42万)
            if near_table and prox_t >= 80 and prox_m >= 80:
                grasp_mode = "both"
            elif near_table and prox_m >= 300:
                grasp_mode = "middle_pad"
            elif near_table and prox_t >= 90:
                grasp_mode = "thumb_pad"
            else:
                grasp_mode = None
            info["grasp_mode"] = grasp_mode
            print(f"   握る条件: {grasp_mode}", flush=True)
            if grasp_mode or attempt == 4:
                break
            if false_stop:
                shift, who = np.zeros(2), "なし (誤停止)"
            elif near_table:      # 机まで降りたが指の間にない: 反応の強い指の側へ 8mm
                shift, who = (axis * 0.008, "中指") if prox_m >= prox_t else (-axis * 0.008, "親指")
            elif prox_t > 1.5 * max(prox_m, 1):
                shift, who = -axis * 0.015, "親指"
            else:
                shift, who = axis * 0.015, "中指"
            print(f"   → {who}がチョコに乗った。7cm 浮かせてから空中で{'指先' if who == '中指' else '手首'}の向きへ 15mm ずらし、降り直す", flush=True)
            # 1) まず真上へ十分浮かせる (指令 +7cm。腕のたれ約 17mm を見込んでも実際に約 5cm 浮く)
            up = vertical_chain(p_act, [0.015, 0.03, 0.05, 0.07], q_act)
            move_path([merge(kin, q10_hover, q5) for q5 in up], aux, 3.0, f"tm_up{attempt}")
            time.sleep(0.3)
            # 2) 空中で横へずらす
            target = target + np.array([shift[0], shift[1], 0.0])
            q_air = vertical_chain(target, [0.07], up[-1])[0]
            q10_hover = merge(kin, q10_hover, q_air)
            move(q10_hover, aux, 2.0, f"tm_shift{attempt}")
            time.sleep(0.3)
            # 3) 新しい目標の真上から降りる
            chain = vertical_chain(target, [0.05, 0.035, 0.025, 0.018, 0.012, 0.006, 0.0], q_air)
            descent_q5 = [q.tolist() for q in chain]
        rec["at_grasp"] = rec["descents"][-1]["hand"]

        mode = rec["descents"][-1].get("grasp_mode")
        if not mode:
            print("[5] 掴む条件 (両方の近接 / 指の腹) を満たさないため、握らずに浮かせて終了", flush=True)
            up = vertical_chain(target, [0.02, 0.04, 0.06], robot()["q_arm"][5:10])
            move_path([merge(kin, q10_hover, q5) for q5 in up], aux, 3.0, "tm_abort_up")
            rec["result"] = "no_grasp_condition"
            return
        print(f"[5] 握る ({mode})", flush=True)
        if mode == "both":
            hand(SHAPE_CLOSE, 800, 3.5)
        elif mode == "middle_pad":     # 中指は壁としてそのまま、親指を曲げて寄せる
            hand([510, 720, 49, SHAPE_OPEN[3], SHAPE_OPEN[4], 49], 900, 3.5)
        else:                          # 親指は壁としてそのまま、中指を曲げて寄せる
            hand([SHAPE_OPEN[0], 720, 49, 360, 330, 49], 900, 3.5)
        rec["closed"] = hand_state()
        print("   閉じた後:", rec["closed"], flush=True)

        print("[6] 6cm 持ち上げて保持", flush=True)
        q_now = robot()["q_arm"][5:10]
        up = vertical_chain(target, [0.01, 0.02, 0.03, 0.045, 0.06], q_now)
        move_path([merge(kin, q10_hover, q5) for q5 in up], aux, 3.0, "tm_lift")
        time.sleep(2.5)
        hs = hand_state()
        rec["lifted"] = hs
        held = hs["Fn"][2] >= 0.15 or hs["Ft"][2] >= 0.30 or (hs["prox_k"][0] >= 300 and hs["prox_k"][2] >= 500)
        rec["result"] = "held" if held else "empty"
        print(f"[結果] {'掴めている (中指に力あり)' if held else '空振りの可能性 (中指に力なし)'}: {hs}", flush=True)
    except Exception as exc:
        rec["result"] = f"error: {exc}"
        print("[エラー]", exc, flush=True)
        try:
            send("arm_hold_here")
        except Exception:
            pass
    finally:
        with RUNS.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def do_return() -> None:
    kin = R1Kinematics()
    last = json.loads(LAST.read_text(encoding="utf-8"))
    r0 = robot()
    if not r0.get("engaged"):
        raise SystemExit("腕に制御が入っていません")
    aux = list(r0["q_aux"])
    q_init = last["initial_q10"]
    wp0, wp1, wp2 = kin.build_safe_table_waypoints(q_init, SIDE, table_bbox=(last.get("choco") or {}).get("table_bbox_torso"))
    print("[1] 手を開く", flush=True)
    hand(SHAPE_OPEN, 1000, 2.0)
    print("[2] 机の上 → 机の後ろ → 体の横 → 初期姿勢", flush=True)
    move(wp2, aux, 2.6, "tm_ret_over_desk", replace=True)
    move(wp1, aux, 3.0, "tm_ret_behind_desk")
    move(wp0, aux, 3.0, "tm_ret_side_clear")
    move(q_init, aux, 3.0, "tm_ret_home")
    print("[3] 脱力", flush=True)
    send("arm_disengage")
    time.sleep(3.0)
    print("完了:", robot().get("stage"), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "plan":
        do_grasp(dry=True)
    elif mode == "grasp":
        do_grasp(dry=False)
    elif mode == "return":
        do_return()
    else:
        raise SystemExit("usage: grasp_thumb_middle.py plan|grasp|return")
