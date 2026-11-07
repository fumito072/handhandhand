"""
Mac ⇄ Unitree R1 (`192.168.1.13:8891`) 非同期 TCP クライアント ＆ SSH ワンクリック自動デプロイ・起動機構。
R1 の電源投入・バッテリー交換後の再接続にも自動で追従する。
"""
from __future__ import annotations

import asyncio
import json
import socket
import struct
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

MSG_JSON = 1
MSG_STEREO = 2
MSG_HD = 3

ONBOARD_SRC_DIR = Path(__file__).resolve().parent / "r1_onboard"
REMOTE_DIR = "/home/unitree/choco-handover-bridge"


class R1BridgeClient:
    def __init__(self, host: str = "192.168.1.13", port: int = 8891):
        self.host = host
        self.port = port
        self.connected = False
        self.last_connect_error = "未接続"
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.pending_replies: Dict[str, asyncio.Future] = {}

        self.latest_telemetry: Dict[str, Any] = {}
        self.latest_arm_events: list = []
        self.latest_stereo: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
        self.latest_hd: Optional[Tuple[np.ndarray, float]] = None
        self.stereo_fps: float = 0.0
        self._stereo_times: list = []

        self.running = True
        self.deploy_status: str = "idle"
        self.deploy_log: str = ""
        self._write_lock = asyncio.Lock()
        self._deploy_lock = asyncio.Lock()

    def set_endpoint(self, host: str, port: int = 8891):
        self.host = host
        self.port = port

    async def deploy_and_start_remote(self) -> Dict[str, Any]:
        """
        Mac から R1 (`unitree@host`) へ SSH 経由で `r1_onboard/` を転送し、
        未ビルドなら C++ `r1_arm_daemon` をビルドした上で `r1_bridge_server.py` をバックグラウンド起動する。
        既存ディレクトリ (`/home/unitree/revo2-tools`, `/home/unitree/r1-local-tools`) には一切触れない。
        """
        async with self._deploy_lock:
            return await self._deploy_and_start_remote_locked()

    async def _deploy_and_start_remote_locked(self) -> Dict[str, Any]:
        self.deploy_status = "deploying"
        target = f"unitree@{self.host}"
        ssh_opts = [
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=4",
            "-o", "StrictHostKeyChecking=no",
        ]

        def _sync_job() -> Tuple[bool, str]:
            logs = []
            r1 = subprocess.run(
                [
                    "ssh", *ssh_opts, target,
                    f"test -d /home/unitree/revo2-tools && test -d /home/unitree/unitree_sdk2 && mkdir -p {REMOTE_DIR}/build",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if r1.returncode != 0:
                return False, f"SSH接続または対象機確認失敗 ({self.host}): {r1.stderr.strip() or 'Revo2搭載R1 (192.168.1.13) の電源・バッテリー・Wi-Fiを確認してください'}"
            logs.append("SSH・Revo2搭載R1確認OK")

            src_files = [str(p) for p in ONBOARD_SRC_DIR.glob("*") if p.is_file()]
            r2 = subprocess.run(
                ["scp", *ssh_opts, *src_files, f"{target}:{REMOTE_DIR}/"],
                capture_output=True, text=True, timeout=20,
            )
            if r2.returncode != 0:
                return False, f"SCP転送失敗: {r2.stderr.strip()}"
            logs.append("オンボードコード転送OK")

            build_and_launch = (
                f"set -e\ncd {REMOTE_DIR}/build\n"
                "cmake_bin=/home/unitree/cmake/bin/cmake\n"
                "[ -x \"$cmake_bin\" ] || cmake_bin=cmake\n"
                "\"$cmake_bin\" .. >/tmp/choco_cmake.log 2>&1\n"
                "make -j4 >/tmp/choco_make.log 2>&1\n"
                "pkill -f 'r1_bridge_server.py' 2>/dev/null || true\n"
                "for _ in 1 2 3 4 5 6 7 8 9 10; do pgrep -f 'r1_bridge_server.py' >/dev/null || break; sleep 0.25; done\n"
                "pkill -9 -f 'r1_bridge_server.py' 2>/dev/null || true\n"
                "pkill -f 'r1_arm_daemon' 2>/dev/null || true\n"
                "sleep 0.3\n"
                f"nohup python3 -u {REMOTE_DIR}/r1_bridge_server.py >/tmp/choco_bridge.log 2>&1 < /dev/null &\n"
                "sleep 1.2\n"
                "pgrep -fa 'r1_bridge_server.py'\n"
            )
            r3 = subprocess.run(
                ["ssh", *ssh_opts, target, "bash -s"],
                input=build_and_launch,
                capture_output=True, text=True, timeout=90,
            )
            if r3.returncode != 0:
                return False, f"ビルド/起動失敗: {r3.stderr.strip()} {r3.stdout.strip()}"
            logs.append(f"デーモン起動完了: {r3.stdout.strip()}")
            return True, " / ".join(logs)

        try:
            ok, msg = await asyncio.to_thread(_sync_job)
            self.deploy_status = "ready" if ok else "error"
            self.deploy_log = msg
            return {"ok": ok, "message": msg}
        except Exception as exc:
            self.deploy_status = "error"
            self.deploy_log = str(exc)
            return {"ok": False, "message": str(exc)}

    async def send_command(self, method: str, timeout: float = 9.0, **kwargs) -> Dict[str, Any]:
        if not self.connected or self.writer is None:
            raise RuntimeError("R1 ブリッジに未接続です")
        req_id = uuid.uuid4().hex[:10]
        payload = {"id": req_id, "method": method, **kwargs}
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        packet = struct.pack(">IB", len(raw) + 1, MSG_JSON) + raw

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.pending_replies[req_id] = fut
        try:
            async with self._write_lock:
                self.writer.write(packet)
                await self.writer.drain()
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self.pending_replies.pop(req_id, None)

    async def connection_loop(self):
        while self.running:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port),
                    timeout=2.5,
                )
                sock = writer.get_extra_info("socket")
                if sock is not None:
                    try:
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except Exception:
                        pass
                self.reader = reader
                self.writer = writer
                self.connected = True
                self.last_connect_error = ""

                while self.running:
                    hdr = await asyncio.wait_for(reader.readexactly(5), timeout=15.0)
                    length, mtype = struct.unpack(">IB", hdr)
                    body = await asyncio.wait_for(reader.readexactly(length - 1), timeout=15.0)

                    if mtype == MSG_JSON:
                        msg = json.loads(body.decode("utf-8"))
                        tp = msg.get("type")
                        if tp == "telemetry":
                            self.latest_telemetry = msg
                        elif tp == "reply":
                            rid = msg.get("id")
                            fut = self.pending_replies.get(rid)
                            if fut and not fut.done():
                                fut.set_result(msg)
                        elif tp == "arm_event":
                            self.latest_arm_events.append(msg.get("data", {}))
                            if len(self.latest_arm_events) > 30:
                                self.latest_arm_events = self.latest_arm_events[-30:]

                    elif mtype == MSG_STEREO:
                        ts, left_len = struct.unpack(">dI", body[:12])
                        jpg_l = body[12:12 + left_len]
                        jpg_r = body[12 + left_len:]
                        img_l = cv2.imdecode(np.frombuffer(jpg_l, dtype=np.uint8), cv2.IMREAD_COLOR)
                        img_r = cv2.imdecode(np.frombuffer(jpg_r, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img_l is not None and img_r is not None:
                            self.latest_stereo = (img_l, img_r, ts)
                            now = time.monotonic()
                            self._stereo_times.append(now)
                            self._stereo_times = [t for t in self._stereo_times if now - t <= 2.0]
                            if len(self._stereo_times) >= 2:
                                span = self._stereo_times[-1] - self._stereo_times[0]
                                if span > 0.05:
                                    self.stereo_fps = round((len(self._stereo_times) - 1) / span, 1)

                    elif mtype == MSG_HD:
                        (ts_hd,) = struct.unpack(">d", body[:8])
                        jpg_hd = body[8:]
                        img_hd = cv2.imdecode(np.frombuffer(jpg_hd, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img_hd is not None:
                            self.latest_hd = (img_hd, ts_hd)

            except Exception as exc:
                self.last_connect_error = str(exc) or type(exc).__name__
            finally:
                self.connected = False
                for fut in self.pending_replies.values():
                    if not fut.done():
                        fut.set_exception(ConnectionError("Disconnected from R1"))
                self.pending_replies.clear()
                if self.writer is not None:
                    try:
                        self.writer.close()
                    except Exception:
                        pass
                self.reader = None
                self.writer = None

            if self.running:
                await asyncio.sleep(1.5)
