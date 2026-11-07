#!/usr/bin/env python3
"""
実機ダイレクトティーチング記録デーモン (`teach_recorder.py`)
R1 (`192.168.1.13:8891`) からリアルタイムに配信される:
  1. 両眼ステレオカメラ左右生画像 (`left.jpg` / `right.jpg`, 約 3 FPS)
  2. 双腕10関節・腰・首角度 (`q_arm`, `q_aux`, `q_waist_roll`, `tau_arm`, `rpy`, `fsm_id`, `bms`)
  3. URDF 順運動学 (`arm_fk`) による左右手首・TCP (指先中心) の 3D 座標 (`torso` 基準 `[X, Y, Z]` [m])
  4. 左右 BrainCo Revo 2 ハンドの 6軸位置・電流 (`position`, `current_mA`) と
     5指すべての近接・触覚センサー値 (`proximity`, `normal_N`, `tangent_N`, `direction_deg`, `flags`)
を `recordings/teach_<timestamp>/` に欠落なく同期保存する。
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import signal
import struct
import time
from pathlib import Path

from config import SystemConfig
from kinematics import R1Kinematics

MSG_JSON = 1
MSG_STEREO = 2

HERE = Path(__file__).resolve().parent
REC_ROOT = HERE / "recordings"


async def run_recorder(host: str = "192.168.1.13", port: int = 8891):
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REC_ROOT / f"teach_{stamp}"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    latest_link = REC_ROOT / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(out_dir.name)
    except Exception:
        pass

    kin = R1Kinematics()
    cfg = SystemConfig()
    tcp_off = (cfg.tcp_offset_x_m, cfg.tcp_offset_y_m, cfg.tcp_offset_z_m)

    jsonl_path = out_dir / "telemetry.jsonl"
    f_log = open(jsonl_path, "a", encoding="utf-8")

    print(f"[REC] Recording started -> {out_dir}", flush=True)

    running = True

    def _stop(*_):
        nonlocal running
        running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _stop)

    frame_id = 0
    sample_id = 0
    last_frame_save = 0.0
    last_print = 0.0
    latest_saved_frame = None

    while running:
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=3.5)
            print(f"[REC] Connected to R1 bridge {host}:{port}", flush=True)
            while running:
                hdr = await asyncio.wait_for(reader.readexactly(5), timeout=5.0)
                length, mtype = struct.unpack(">IB", hdr)
                payload = await asyncio.wait_for(reader.readexactly(length - 1), timeout=5.0)
                now = time.time()

                if mtype == MSG_STEREO:
                    if (now - last_frame_save) >= 0.32:
                        ts_cam, len_l = struct.unpack(">dI", payload[:12])
                        jpg_l = payload[12:12 + len_l]
                        jpg_r = payload[12 + len_l:]
                        frame_id += 1
                        fn_l = f"f_{frame_id:05d}_left.jpg"
                        fn_r = f"f_{frame_id:05d}_right.jpg"
                        (frames_dir / fn_l).write_bytes(jpg_l)
                        (frames_dir / fn_r).write_bytes(jpg_r)
                        latest_saved_frame = {
                            "frame_id": frame_id,
                            "cam_ts": ts_cam,
                            "saved_at": now,
                            "left": f"frames/{fn_l}",
                            "right": f"frames/{fn_r}",
                        }
                        last_frame_save = now

                elif mtype == MSG_JSON:
                    msg = json.loads(payload.decode("utf-8"))
                    if msg.get("type") != "telemetry":
                        continue
                    robot = msg.get("robot") or {}
                    hands = msg.get("hands") or {}
                    q_arm = robot.get("q_arm")
                    fk_info = {}
                    if q_arm and len(q_arm) == 10:
                        q_l = q_arm[:5]
                        q_r = q_arm[5:]
                        wl, _ = kin.arm_fk("left", q_l, (0.0, 0.0, 0.0))
                        tl, Rl = kin.arm_fk("left", q_l, tcp_off)
                        wr, _ = kin.arm_fk("right", q_r, (0.0, 0.0, 0.0))
                        tr, Rr = kin.arm_fk("right", q_r, tcp_off)
                        fk_info = {
                            "left_wrist_torso_m": [round(float(x), 4) for x in wl],
                            "left_tcp_torso_m": [round(float(x), 4) for x in tl],
                            "left_palm_normal": [round(float(x), 4) for x in kin.palm_normal_world("left", Rl)],
                            "right_wrist_torso_m": [round(float(x), 4) for x in wr],
                            "right_tcp_torso_m": [round(float(x), 4) for x in tr],
                            "right_palm_normal": [round(float(x), 4) for x in kin.palm_normal_world("right", Rr)],
                        }

                    sample_id += 1
                    rec = {
                        "sample_id": sample_id,
                        "t": now,
                        "iso": dt.datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
                        "frame": latest_saved_frame,
                        "robot": robot,
                        "fk": fk_info,
                        "hands": hands,
                    }
                    f_log.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                    f_log.flush()

                    if (now - last_print) >= 2.0:
                        last_print = now
                        lt = fk_info.get("left_tcp_torso_m")
                        rt = fk_info.get("right_tcp_torso_m")
                        lp = ((hands.get("left") or {}).get("tactile") or {}).get("fingers") or []
                        rp = ((hands.get("right") or {}).get("tactile") or {}).get("fingers") or []
                        l_fmax = max((f.get("normal_N", 0.0) for f in lp), default=0.0)
                        r_fmax = max((f.get("normal_N", 0.0) for f in rp), default=0.0)
                        print(
                            f"[REC] samples={sample_id} frames={frame_id} "
                            f"L_tcp={lt} R_tcp={rt} L_force={l_fmax:.2f}N R_force={r_fmax:.2f}N",
                            flush=True,
                        )
        except Exception as exc:
            print(f"[REC] reconnecting ({exc})...", flush=True)
            await asyncio.sleep(1.0)

    f_log.close()
    print(f"[REC] Stopped cleanly. Saved {sample_id} samples and {frame_id} stereo pairs in {out_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(run_recorder())
