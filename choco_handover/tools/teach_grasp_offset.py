#!/usr/bin/env python3
"""
「この手の形でチョコを掴むとき、チョコは手首から見てどこにあるか」を 1 回の実演で測るツール。

手順 (サーバー :8890 が R1 に接続、腕は脱力):
  1. チョコを机に置き、手を離した状態で:   python choco_handover/tools/teach_grasp_offset.py choco
     → カメラ認識のチョコ位置・向きを約 2 秒平均して記録
  2. 人が腕を動かして、手の形のままチョコをはさむ位置に合わせて止めた状態で:
                                          python choco_handover/tools/teach_grasp_offset.py hand
     → 手首の位置・向き (順運動学) を約 1 秒平均し、チョコ中心を手首座標で表したオフセットを計算して保存

カメラ由来のチョコ位置と、同じ時刻の手首姿勢の差をそのまま使うので、その場所でのカメラ→腕の補正誤差も吸収される。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from config import ConfigStore  # noqa: E402
from kinematics import R1Kinematics  # noqa: E402

OUT = PKG.parent / "docs" / "data" / "grasp_offset_teach.json"
SIDE = "right"


def _status() -> dict:
    with urllib.request.urlopen("http://127.0.0.1:8890/api/status", timeout=4) as r:
        return json.load(r)


def _load() -> dict:
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"samples": []}


def _save(d: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def record_choco() -> None:
    pts, yaws, tz, n = [], [], [], 0
    t0 = time.time()
    while time.time() - t0 < 2.5:
        c = _status().get("choco")
        if c and c.get("centroid_torso_m"):
            pts.append(c["centroid_torso_m"])
            yaws.append(float(c.get("yaw_deg") or 0.0))
            if c.get("table_z_torso_m") is not None:
                tz.append(float(c["table_z_torso_m"]))
            n += 1
        time.sleep(0.15)
    if n < 3:
        raise SystemExit(f"チョコを安定して認識できません (認識 {n} 回)。カメラから見える位置に置いてください")
    p = np.median(np.array(pts), axis=0)
    # 長方形は 180° 対称なので向きは [-90, 90) に正規化して中央値を取る
    y = np.array([((v + 90.0) % 180.0) - 90.0 for v in yaws])
    d = _load()
    d["choco"] = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "centroid_torso_m": [round(float(v), 4) for v in p],
        "yaw_deg": round(float(np.median(y)), 1),
        "table_z_torso_m": (round(float(np.median(tz)), 4) if tz else None),
        "detections": n,
    }
    _save(d)
    print("チョコ記録:", d["choco"])


def record_hand() -> None:
    d = _load()
    if "choco" not in d:
        raise SystemExit("先に `choco` でチョコの位置を記録してください")
    store = ConfigStore()
    store.load()
    kin = R1Kinematics()
    qs, hands = [], []
    t0 = time.time()
    while time.time() - t0 < 1.2:
        st = _status()
        r = st["telemetry"]["robot"]
        qs.append(r["q_arm"][5:10] if SIDE == "right" else r["q_arm"][:5])
        hands.append(st["telemetry"]["hands"][SIDE]["state"]["position"])
        time.sleep(0.1)
    q5 = np.median(np.array(qs, dtype=np.float64), axis=0)
    p_w, R_w = kin.arm_fk(SIDE, q5.tolist(), (0.0, 0.0, 0.0))
    p_c = np.array(d["choco"]["centroid_torso_m"], dtype=np.float64)
    offset_w = R_w.T @ (p_c - p_w)
    wrist_yaw = math.degrees(math.atan2(float(R_w[1, 0]), float(R_w[0, 0])))
    sample = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "side": SIDE,
        "q5": [round(float(v), 4) for v in q5],
        "wrist_pos_torso_m": [round(float(v), 4) for v in p_w],
        "wrist_R": [[round(float(v), 5) for v in row] for row in R_w],
        "offset_in_wrist_m": [round(float(v), 4) for v in offset_w],
        "choco_centroid_torso_m": d["choco"]["centroid_torso_m"],
        "choco_yaw_deg": d["choco"]["yaw_deg"],
        "wrist_yaw_deg": round(wrist_yaw, 1),
        "yaw_rel_deg": round(((wrist_yaw - d["choco"]["yaw_deg"] + 90.0) % 180.0) - 90.0, 1),
        "hand_position": [int(v) for v in np.median(np.array(hands), axis=0)],
    }
    d["samples"].append(sample)
    _save(d)
    print("手首記録:", json.dumps(sample, ensure_ascii=False))
    print(f"→ チョコ中心は手首から見て {np.round(offset_w * 1000, 1).tolist()} mm (手首座標)。保存先 {OUT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "choco":
        record_choco()
    elif mode == "hand":
        record_hand()
    else:
        raise SystemExit("usage: teach_grasp_offset.py choco|hand")
