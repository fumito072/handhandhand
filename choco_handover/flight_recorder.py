"""
全試行フライトレコーダー (Flight Recorder) ＆ ベンチマーク集計モジュール。
各把持試行ごとに画像 (Stereo / HD)・指令関節角・実測関節角・実測トルク・重力補償トルク・外乱トルク・
Revo 2 5指触覚/近接/エンコーダ・FSMフェーズを `logs/run_YYYYMMDD_HHMMSS/` へ完全自動記録し、
オフライン再生検証および多位置×多角度ベンチマーク成功率の定量集計を可能にする。
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from config import BASE_DIR

LOGS_ROOT = BASE_DIR / "logs"


class FlightRecorderSession:
    def __init__(
        self,
        tag: str,
        meta: Dict[str, Any],
        telemetry_getter: Callable[[], Dict[str, Any]],
        phase_getter: Callable[[], str],
    ):
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_tag = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in (tag or "run"))[:32]
        self.run_id = f"run_{stamp}_{safe_tag}"
        self.run_dir = LOGS_ROOT / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.t_start = time.time()
        self._mono_start = time.monotonic()
        self._telemetry_getter = telemetry_getter
        self._phase_getter = phase_getter
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._jsonl_path = self.run_dir / "telemetry.jsonl"
        self.events: List[Dict[str, Any]] = []
        self.sample_count = 0

        meta_out = {
            "run_id": self.run_id,
            "started_at": self.t_start,
            "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tag": tag,
            **meta,
        }
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._sample_loop())

    def save_keyframe(
        self,
        name: str,
        stereo_pair: Optional[tuple] = None,
        hd_tuple: Optional[tuple] = None,
        fallback_left_bgr: Optional[np.ndarray] = None,
    ):
        clean = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in name)
        try:
            if stereo_pair is not None and len(stereo_pair) >= 2 and stereo_pair[0] is not None:
                cv2.imwrite(str(self.run_dir / f"{clean}_left.jpg"), stereo_pair[0])
            elif fallback_left_bgr is not None:
                cv2.imwrite(str(self.run_dir / f"{clean}_left.jpg"), fallback_left_bgr)
            if hd_tuple is not None and len(hd_tuple) >= 1 and hd_tuple[0] is not None:
                cv2.imwrite(str(self.run_dir / f"{clean}_hd.jpg"), hd_tuple[0])
        except Exception:
            pass

    def log_event(self, event_name: str, **data: Any):
        rec = {
            "t_rel_s": round(time.monotonic() - self._mono_start, 3),
            "event": event_name,
            **data,
        }
        self.events.append(rec)

    async def _sample_loop(self):
        try:
            with self._jsonl_path.open("a", encoding="utf-8") as fp:
                while self._running:
                    tel = self._telemetry_getter() or {}
                    rob = tel.get("robot") or {}
                    hands = tel.get("hands") or {}
                    row = {
                        "t": round(time.time(), 4),
                        "t_rel_s": round(time.monotonic() - self._mono_start, 3),
                        "phase": self._phase_getter(),
                        "stage": rob.get("stage"),
                        "engaged": rob.get("engaged"),
                        "grav_ff_gain": rob.get("grav_ff_gain"),
                        "rpy": rob.get("rpy"),
                        "q_arm": rob.get("q_arm"),
                        "cmd_arm": rob.get("cmd_arm"),
                        "dq_arm": rob.get("dq_arm"),
                        "tau_arm": rob.get("tau_arm"),
                        "tau_grav": rob.get("tau_grav"),
                        "tau_ext": rob.get("tau_ext"),
                        "q_aux": rob.get("q_aux"),
                        "last_fault": rob.get("last_fault"),
                        "hands": hands,
                    }
                    fp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    fp.flush()
                    self.sample_count += 1
                    await asyncio.sleep(0.10)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def finish(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except BaseException:
                pass
        summary = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "duration_s": round(time.monotonic() - self._mono_start, 2),
            "telemetry_samples": self.sample_count,
            "events": self.events,
            **outcome,
        }
        try:
            (self.run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
        return summary


def summarize_benchmark_logs(limit: int = 60) -> Dict[str, Any]:
    """直近の試行ログから成功率・失敗分類・着地検知ソース内訳を集計して返す。"""
    if not LOGS_ROOT.exists():
        return {"total_runs": 0, "grasp_successes": 0, "success_rate_pct": 0.0, "failure_modes": {}, "recent_runs": []}

    dirs = sorted([d for d in LOGS_ROOT.iterdir() if d.is_dir() and d.name.startswith("run_")], reverse=True)[: max(1, int(limit))]
    runs = []
    succ = 0
    failure_modes: Dict[str, int] = {}
    for d in dirs:
        sp = d / "summary.json"
        if not sp.exists():
            continue
        try:
            s = json.loads(sp.read_text(encoding="utf-8"))
            if s.get("dry_run"):
                continue
            ok = bool(s.get("grasp_verified"))
            if ok:
                succ += 1
            else:
                fm = str(s.get("failure_mode") or "unknown")
                failure_modes[fm] = failure_modes.get(fm, 0) + 1
            runs.append({
                "run_id": s.get("run_id", d.name),
                "grasp_verified": ok,
                "posture": s.get("posture"),
                "side": s.get("side"),
                "yaw_deg": s.get("choco_yaw_deg"),
                "achieved_pinch_yaw_deg": s.get("achieved_pinch_yaw_deg"),
                "touchdown_source": s.get("touchdown_source"),
                "failure_mode": s.get("failure_mode"),
                "duration_s": s.get("duration_s"),
            })
        except Exception:
            continue

    total = len(runs)
    rate = round(100.0 * succ / total, 1) if total > 0 else 0.0
    return {
        "total_runs": total,
        "grasp_successes": succ,
        "success_rate_pct": rate,
        "failure_modes": failure_modes,
        "recent_runs": runs[:20],
    }
