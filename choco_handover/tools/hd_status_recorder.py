#!/usr/bin/env python3
"""
人の実演 (手で腕・ハンドを動かして掴む) を記録するため、HD 画像と全テレメトリを一定間隔で保存する。

teach_recorder.py (ステレオ画像＋テレメトリ) と並行して使う。こちらはサーバー (:8890) 経由で
HD 画像 (1280x720) と、両手 5 指の法線力・接線力・近接値、モーター位置・電流、腕の関節角、チョコ認識結果を残す。

使い方:
    .venv/bin/python choco_handover/tools/hd_status_recorder.py <出力フォルダ> [--interval 0.5]
停止は SIGINT / SIGTERM。
"""
from __future__ import annotations

import argparse
import base64
import json
import signal
import time
import urllib.request
from pathlib import Path


def _get(url: str, timeout: float = 6.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--server", default="http://127.0.0.1:8890")
    args = ap.parse_args()

    out = Path(args.out_dir)
    hd_dir = out / "hd"
    hd_dir.mkdir(parents=True, exist_ok=True)
    log = (out / "status.jsonl").open("a", encoding="utf-8")

    running = True

    def _stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    n = 0
    last_hd_ts = None
    print(f"[HDREC] recording -> {out}", flush=True)
    while running:
        t0 = time.time()
        rec = {"t": round(t0, 3)}
        try:
            raw = _get(f"{args.server}/api/raw_stereo", timeout=8.0)
            hd_ts = raw.get("hd_ts")
            if raw.get("hd_b64") and hd_ts != last_hd_ts:
                name = f"{time.strftime('%H%M%S', time.localtime(t0))}_{int((t0 % 1) * 1000):03d}.jpg"
                (hd_dir / name).write_bytes(base64.b64decode(raw["hd_b64"]))
                rec["hd_file"] = f"hd/{name}"
                rec["hd_ts"] = hd_ts
                last_hd_ts = hd_ts
        except Exception as exc:
            rec["hd_error"] = str(exc)
        try:
            st = _get(f"{args.server}/api/status")
            tel = st.get("telemetry") or {}
            rec["robot"] = tel.get("robot")
            rec["hands"] = tel.get("hands")
            rec["choco"] = st.get("choco")
            rec["phase"] = (st.get("task") or {}).get("phase")
        except Exception as exc:
            rec["status_error"] = str(exc)
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        n += 1
        if n % 20 == 0:
            rh = ((rec.get("hands") or {}).get("right") or {})
            fn = [round(f.get("normal_N", 0.0), 2) for f in ((rh.get("tactile") or {}).get("fingers") or [])]
            print(f"[HDREC] samples={n} right_pos={(rh.get('state') or {}).get('position')} right_Fn={fn}", flush=True)
        time.sleep(max(0.0, args.interval - (time.time() - t0)))
    log.close()
    print(f"[HDREC] stopped. {n} samples in {out}", flush=True)


if __name__ == "__main__":
    main()
