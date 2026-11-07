#!/usr/bin/env python3
"""
人が手で動かして示した「掴めそうな腕＋ハンドの姿勢」を、その瞬間の数値で記録するツール。

使い方 (サーバー :8890 が R1 に接続している状態で):
    .venv/bin/python choco_handover/tools/pose_snapshot.py --label "親指を立てた形" [--side right]

記録するもの (1 回 = JSONL 1 行 + HD 画像 1 枚):
- 両手 Revo 2 の 6 モーター位置 (特に [0] 親指の曲げ / [1] 親指の向き=対立)、電流、各指の近接値・力
- 両腕 10 関節角・腰/首角
- 順運動学: ピンチ中心 (TCP) 位置、手首の向き (ロール/ピッチ/ヨー)、親指→人差指 軸の水平角と上下成分
- 今のチョコ位置に対して FSM が計画する姿勢との差 (手首の向き・把持軸・関節角)
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from config import ConfigStore  # noqa: E402
from kinematics import R1Kinematics  # noqa: E402

MOTOR_NAMES = ["thumb_flex", "thumb_rot", "index", "middle", "ring", "pinky"]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def _get(url: str, timeout: float = 6.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _rpy_deg(R: np.ndarray) -> list:
    """回転行列 → ZYX オイラー (roll, pitch, yaw) [deg]。"""
    pitch = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return [round(math.degrees(roll), 1), round(math.degrees(pitch), 1), round(math.degrees(yaw), 1)]


def _arm_geometry(kin: R1Kinematics, cfg, side: str, q5: list) -> dict:
    tcp_off = kin.active_tcp_offset(cfg, posture="surface_pinch")
    pos, R = kin.arm_fk(side, list(q5), tcp_off)
    vp = kin.pinch_axis_world(R, side=side)
    yaw_deg = math.degrees(math.atan2(float(vp[1]), float(vp[0])))
    return {
        "tcp_torso_m": [round(float(v), 4) for v in pos],
        "wrist_rpy_deg": _rpy_deg(R),
        "wrist_R": [[round(float(v), 4) for v in row] for row in R],
        "pinch_axis_thumb_to_index": [round(float(v), 3) for v in vp],
        "pinch_axis_yaw_deg": round(yaw_deg, 1),
        "pinch_axis_z": round(float(vp[2]), 3),
        "pinch_axis_tilt_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, float(vp[2]))))), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="この姿勢の説明 (例: 親指を立てた形)")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--server", default="http://127.0.0.1:8890")
    ap.add_argument("--out", default=str(PKG.parent / "docs" / "data" / f"hand_pose_teach_{dt.date.today():%Y%m%d}.jsonl"))
    args = ap.parse_args()

    store = ConfigStore()
    store.load()
    cfg = store.cfg
    kin = R1Kinematics()

    st = _get(f"{args.server}/api/status")
    tel = st.get("telemetry") or {}
    rob = tel.get("robot") or {}
    hands = tel.get("hands") or {}
    q_arm = rob.get("q_arm") or []
    if len(q_arm) != 10:
        raise SystemExit("腕の関節角が取得できません (R1 未接続?)")

    now = dt.datetime.now()
    rec = {
        "time": now.isoformat(timespec="seconds"),
        "label": args.label,
        "side": args.side,
        "robot": {
            "engaged": rob.get("engaged"),
            "stage": rob.get("stage"),
            "q_arm": [round(float(v), 4) for v in q_arm],
            "q_aux_waistyaw_headpitch_headyaw": rob.get("q_aux"),
            "q_waist_roll": rob.get("q_waist_roll"),
            "rpy": rob.get("rpy"),
            "bms_soc": (rob.get("bms") or {}).get("soc"),
        },
        "hands": {},
        "arms": {},
    }
    for side in ("left", "right"):
        h = hands.get(side) or {}
        s = h.get("state") or {}
        pos = s.get("position") or []
        fingers = ((h.get("tactile") or {}).get("fingers")) or []
        rec["hands"][side] = {
            "position": dict(zip(MOTOR_NAMES, pos)) if len(pos) == 6 else pos,
            "position_list": pos,
            "current_mA": s.get("current_mA"),
            "tactile": [
                {
                    "finger": FINGER_NAMES[i] if i < 5 else str(i),
                    "proximity": f.get("proximity"),
                    "normal_N": f.get("normal_N"),
                    "tangent_N": f.get("tangent_N"),
                }
                for i, f in enumerate(fingers)
            ],
        }
        q5 = q_arm[:5] if side == "left" else q_arm[5:10]
        rec["arms"][side] = {"q5": [round(float(v), 4) for v in q5], **_arm_geometry(kin, cfg, side, q5)}

    # 今のチョコ位置に対して FSM が計画する姿勢との比較
    choco = st.get("choco")
    if choco and choco.get("centroid_torso_m"):
        c = choco["centroid_torso_m"]
        z_tbl = float(choco.get("table_z_torso_m") or (c[2] - 0.007))
        try:
            p6 = kin.choose_oriented_pinch_plan_6dof(
                np.array([c[0], c[1], z_tbl + 0.001], dtype=np.float64),
                float(choco.get("yaw_deg") or 0.0),
                float(choco.get("length_mm") or 33.0),
                float(choco.get("width_mm") or 26.0),
                cfg,
                allowed_sides=(args.side,),
            )
            q5p = list(p6["q5_grasp"])
            geo_p = _arm_geometry(kin, cfg, args.side, q5p)
            mine = rec["arms"][args.side]
            rec["vs_planner"] = {
                "choco_torso_m": [round(float(v), 4) for v in c],
                "planned_q5": [round(float(v), 4) for v in q5p],
                "planned": geo_p,
                "delta_q5_rad": [round(float(a) - float(b), 4) for a, b in zip(mine["q5"], q5p)],
                "delta_wrist_rpy_deg": [round(a - b, 1) for a, b in zip(mine["wrist_rpy_deg"], geo_p["wrist_rpy_deg"])],
                "delta_pinch_yaw_deg": round(mine["pinch_axis_yaw_deg"] - geo_p["pinch_axis_yaw_deg"], 1),
                "delta_pinch_tilt_deg": round(mine["pinch_axis_tilt_deg"] - geo_p["pinch_axis_tilt_deg"], 1),
            }
        except Exception as exc:  # 計画との比較は補助情報なので失敗しても記録は残す
            rec["vs_planner"] = {"error": str(exc)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img_dir = out.with_suffix("")
    img_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw = _get(f"{args.server}/api/raw_stereo", timeout=8.0)
        if raw.get("hd_b64"):
            img_path = img_dir / f"{now:%H%M%S}_{args.side}_hd.jpg"
            img_path.write_bytes(base64.b64decode(raw["hd_b64"]))
            rec["hd_image"] = str(img_path.relative_to(PKG.parent))
    except Exception as exc:
        rec["hd_image_error"] = str(exc)

    with out.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    h = rec["hands"][args.side]
    a = rec["arms"][args.side]
    print(f"[{rec['time']}] {args.label} ({args.side}) -> {out}")
    print(f"  hand position {h['position_list']}  (thumb_flex={h['position_list'][0] if h['position_list'] else None}, thumb_rot={h['position_list'][1] if len(h['position_list']) > 1 else None})")
    print(f"  arm q5 {a['q5']}")
    print(f"  TCP {a['tcp_torso_m']}  wrist rpy {a['wrist_rpy_deg']}  pinch axis yaw {a['pinch_axis_yaw_deg']} deg, tilt {a['pinch_axis_tilt_deg']} deg")
    if "vs_planner" in rec and "planned" in rec["vs_planner"]:
        v = rec["vs_planner"]
        print(f"  vs planner: d_wrist_rpy {v['delta_wrist_rpy_deg']}  d_pinch_yaw {v['delta_pinch_yaw_deg']}  d_tilt {v['delta_pinch_tilt_deg']}  d_q5 {v['delta_q5_rad']}")


if __name__ == "__main__":
    main()
