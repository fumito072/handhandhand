#!/usr/bin/env python3
"""
Unitree R1 PC4 (Jetson Orin Nano) Onboard Handover Bridge Server (TCP :8891)
- 両眼ステレオカメラ (UDP 5002 Left / UDP 5003 Right, 544x448 整準化済みペア) のリアルタイム取得・JPEG配信
- BrainCo Revo 2 左右ハンドの指先近接センサー減速＋優しさ把持 (GentleChocoGraspPolicy) 25Hz ローカル実行
- C++ `r1_arm_daemon` (250Hz `rt/arm_sdk` 立位安全補間デーモン) のプロセス管理・双方向中継
"""
import asyncio
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gentle_grasp import Revo2Manager  # noqa: E402

STEREO_W = 544
STEREO_H = 448
FRAME_BYTES = STEREO_W * STEREO_H * 3

HD_RAW_W = 720
HD_RAW_H = 1280

MSG_JSON = 1
MSG_STEREO = 2
MSG_HD = 3


def ensure_stereo_service():
    """PC1 の stereo_patch_pc1 が未起動の場合のみ video_hub -> stereo_patch_pc1 へ切り替える。"""
    code = r"""
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
ChannelFactoryInitialize(0, "eth10")
c = RobotStateClient()
c.SetTimeout(2.5)
c.Init()
try:
    c.ServiceSwitch("video_hub", False)
except Exception:
    pass
time.sleep(0.3)
try:
    c.ServiceSwitch("stereo_patch_pc1", True)
except Exception:
    pass
print("stereo_switch_attempted")
"""
    try:
        subprocess.run([sys.executable, "-c", code], timeout=8.0, check=False)
    except Exception as exc:
        print(f"[stereo] switch helper warning: {exc}", flush=True)


class GstRawReader(threading.Thread):
    """GStreamer から H.264 RTP ストリームを BGR フレームとして継続読み出しするスレッド。"""

    def __init__(
        self,
        port: int,
        name: str,
        width: int = STEREO_W,
        height: int = STEREO_H,
        rotate_cw: bool = False,
        min_interval: float = 0.052,
        jpeg_quality: int = 88,
    ):
        super().__init__(daemon=True, name=name)
        self.port = port
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self.rotate_cw = rotate_cw
        self.min_interval = min_interval
        self.jpeg_quality = jpeg_quality
        self.running = True
        self.proc = None
        self.lock = threading.Lock()
        self.latest_jpg = None
        self.latest_ts = 0.0
        self.frame_count = 0

    def stop(self):
        self.running = False
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def get_latest(self):
        with self.lock:
            return self.latest_jpg, self.latest_ts, self.frame_count

    def run(self):
        cmd = [
            "gst-launch-1.0", "-q",
            "udpsrc", f"port={self.port}",
            "caps=application/x-rtp,media=video,encoding-name=H264,payload=96",
            "!", "rtph264depay",
            "!", "h264parse",
            "!", "avdec_h264",
            "!", "videoconvert",
            "!", f"video/x-raw,format=BGR,width={self.width},height={self.height}",
            "!", "fdsink", "fd=1",
        ]
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        while self.running:
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=self.frame_bytes * 2,
                )
                buf = bytearray()
                last_enc = 0.0
                while self.running and self.proc.poll() is None:
                    chunk = self.proc.stdout.read(self.frame_bytes - len(buf))
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if len(buf) == self.frame_bytes:
                        now = time.time()
                        if (now - last_enc) >= self.min_interval:
                            arr = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 3))
                            if self.rotate_cw:
                                arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
                            ok, jpg = cv2.imencode(".jpg", arr, encode_params)
                            if ok:
                                with self.lock:
                                    self.latest_jpg = jpg.tobytes()
                                    self.latest_ts = now
                                    self.frame_count += 1
                            last_enc = now
                        buf.clear()
            except Exception as exc:
                print(f"[gst:{self.port}] error: {exc}", flush=True)
            finally:
                if self.proc is not None:
                    try:
                        self.proc.kill()
                        self.proc.wait(timeout=1.0)
                    except Exception:
                        pass
            if self.running:
                time.sleep(0.5)


class BridgeServer:
    def __init__(self, host="0.0.0.0", port=8891):
        self.host = host
        self.port = port
        self.clients = {}
        self.arm_proc = None
        self.robot_state = {}
        self.hands_state = {}
        self.revo = Revo2Manager()
        self.hand_lock = threading.Lock()
        self.left_cam = GstRawReader(5002, "cam_left_5002")
        self.right_cam = GstRawReader(5003, "cam_right_5003")
        self.hd_cam = GstRawReader(
            5001,
            "cam_hd_5001",
            width=HD_RAW_W,
            height=HD_RAW_H,
            rotate_cw=True,
            min_interval=0.22,
            jpeg_quality=85,
        )
        self.running = True
        self.loop = None

    def start_hardware(self):
        print("[bridge] Connecting BrainCo Revo 2 hands...", flush=True)
        meta = self.revo.connect()
        print(f"[bridge] Revo 2 hands: {json.dumps(meta, ensure_ascii=False)}", flush=True)

        self.left_cam.start()
        self.right_cam.start()
        self.hd_cam.start()

    async def ensure_live_stereo_async(self):
        await asyncio.sleep(1.5)
        _, ts_l, _ = self.left_cam.get_latest()
        if ts_l == 0.0:
            print("[bridge] No UDP 5002 packets yet; requesting r1_arm_daemon enable_stereo...", flush=True)
            try:
                await self.send_arm_cmd({"cmd": "enable_stereo", "id": "boot_stereo"})
            except Exception as exc:
                print(f"[bridge] enable_stereo warning: {exc}", flush=True)

    async def start_arm_daemon(self):
        bin_path = HERE / "build" / "r1_arm_daemon"
        if not bin_path.exists():
            print(f"[bridge] WARNING: {bin_path} not built yet!", flush=True)
            return
        self.arm_proc = await asyncio.create_subprocess_exec(
            str(bin_path), "eth10",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        asyncio.create_task(self._read_arm_stdout())
        asyncio.create_task(self._arm_heartbeat_loop())

    async def _arm_heartbeat_loop(self):
        while self.running and self.arm_proc and self.arm_proc.returncode is None:
            try:
                self.arm_proc.stdin.write(b'{"cmd":"ping"}\n')
                await self.arm_proc.stdin.drain()
            except Exception:
                break
            await asyncio.sleep(0.8)

    async def _read_arm_stdout(self):
        assert self.arm_proc is not None
        while self.running:
            line = await self.arm_proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text.startswith("{"):
                print(f"[arm_daemon] {text}", flush=True)
                continue
            try:
                msg = json.loads(text)
                ev = msg.get("event")
                if ev == "robot_state":
                    self.robot_state = msg
                else:
                    await self.broadcast_json({"type": "arm_event", "data": msg})
            except Exception:
                pass

    async def send_arm_cmd(self, payload: dict):
        if not self.arm_proc or self.arm_proc.returncode is not None:
            raise RuntimeError("r1_arm_daemon is not running")
        raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        self.arm_proc.stdin.write(raw)
        await self.arm_proc.stdin.drain()

    def _hand_loop_thread(self):
        tick_idx = 0
        while self.running:
            t0 = time.monotonic()
            # 25 Hz motor control, 12.5 Hz tactile register read to keep RS-485 crisp
            read_tactile = (tick_idx % 2 == 0)
            tick_idx += 1
            with self.hand_lock:
                self.hands_state = self.revo.tick(read_sensors=read_tactile)
            elapsed = time.monotonic() - t0
            time.sleep(max(0.005, 0.040 - elapsed))

    async def broadcast_raw(self, msg_type: int, payload: bytes):
        if not self.clients:
            return
        header = struct.pack(">IB", len(payload) + 1, msg_type)
        packet = header + payload
        dead = []
        for writer, wlock in list(self.clients.items()):
            if msg_type in (MSG_STEREO, MSG_HD) and wlock.locked():
                continue
            try:
                async with wlock:
                    writer.write(packet)
                    await writer.drain()
            except Exception as exc:
                print(f"[bridge] broadcast write error: {exc}", flush=True)
                dead.append(writer)
        for w in dead:
            self.clients.pop(w, None)
            try:
                w.close()
            except Exception:
                pass

    async def broadcast_json(self, obj: dict):
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        await self.broadcast_raw(MSG_JSON, data)

    async def telemetry_broadcast_loop(self):
        while self.running:
            if self.clients:
                await self.broadcast_json({
                    "type": "telemetry",
                    "t": time.time(),
                    "robot": self.robot_state,
                    "hands_meta": self.revo.meta,
                    "hands": self.hands_state,
                })
            await asyncio.sleep(0.10)  # 10 Hz telemetry

    async def stereo_broadcast_loop(self):
        last_sent_cnt = -1
        while self.running:
            if self.clients:
                jpg_l, ts_l, cnt_l = self.left_cam.get_latest()
                jpg_r, ts_r, _ = self.right_cam.get_latest()
                if jpg_l and jpg_r and cnt_l != last_sent_cnt:
                    last_sent_cnt = cnt_l
                    ts = 0.5 * (ts_l + ts_r)
                    hdr = struct.pack(">dI", ts, len(jpg_l))
                    await self.broadcast_raw(MSG_STEREO, hdr + jpg_l + jpg_r)
            await asyncio.sleep(0.11)  # ~9 FPS stereo pairs (zero Wi-Fi queue buildup)

    async def hd_broadcast_loop(self):
        last_hd_cnt = -1
        while self.running:
            if self.clients:
                jpg_hd, ts_hd, cnt_hd = self.hd_cam.get_latest()
                if jpg_hd and cnt_hd != last_hd_cnt:
                    last_hd_cnt = cnt_hd
                    hdr = struct.pack(">d", ts_hd)
                    await self.broadcast_raw(MSG_HD, hdr + jpg_hd)
            await asyncio.sleep(0.24)  # ~4 FPS 1280x720 HD stream for razor-sharp contour & yaw perception

    async def handle_command(self, cmd: dict) -> dict:
        method = cmd.get("method", "")
        if method == "ping":
            return {"ok": True, "t": time.time()}
        if method == "ensure_stereo":
            await asyncio.get_running_loop().run_in_executor(None, ensure_stereo_service)
            return {"ok": True}
        if method == "estop":
            with self.hand_lock:
                self.revo.stop_all()
            try:
                await self.send_arm_cmd({"cmd": "estop", "id": str(cmd.get("id", ""))})
            except Exception:
                pass
            return {"ok": True}
        if method == "arm_engage":
            payload = {"cmd": "engage", "id": str(cmd.get("id", ""))}
            for k in ("suspended", "table_z_floor", "grav_ff_scale", "extra_hand_mass_kg"):
                if k in cmd and cmd[k] is not None:
                    payload[k] = float(cmd[k])
            await self.send_arm_cmd(payload)
            return {"ok": True}
        if method == "arm_set_params":
            payload = {"cmd": "set_params", "id": str(cmd.get("id", ""))}
            for k in ("table_z_floor", "grav_ff_scale", "extra_hand_mass_kg"):
                if k in cmd and cmd[k] is not None:
                    payload[k] = float(cmd[k])
            await self.send_arm_cmd(payload)
            return {"ok": True}
        if method == "capture_hires_5001":
            def _grab_5001():
                import base64
                tmp_jpg = "/tmp/r1_hires_5001.jpg"
                gcmd = [
                    "gst-launch-1.0", "-q",
                    "udpsrc", "port=5001",
                    "caps=application/x-rtp,media=video,encoding-name=H264,payload=96",
                    "!", "rtph264depay", "!", "h264parse", "!", "avdec_h264",
                    "!", "videoconvert", "!", "jpegenc", "quality=96",
                    "!", "multifilesink", f"location={tmp_jpg}", "max-files=1",
                ]
                p = subprocess.Popen(gcmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    for _ in range(18):
                        time.sleep(0.1)
                        if os.path.exists(tmp_jpg) and os.path.getsize(tmp_jpg) > 4000:
                            break
                finally:
                    p.terminate()
                if os.path.exists(tmp_jpg) and os.path.getsize(tmp_jpg) > 4000:
                    with open(tmp_jpg, "rb") as f:
                        return base64.b64encode(f.read()).decode("ascii")
                return None
            b64 = await asyncio.get_running_loop().run_in_executor(None, _grab_5001)
            return {"ok": bool(b64), "jpg_b64": b64}
        if method == "arm_disengage":
            await self.send_arm_cmd({"cmd": "disengage", "id": str(cmd.get("id", ""))})
            return {"ok": True}
        if method == "arm_hold_here":
            await self.send_arm_cmd({"cmd": "hold_here", "id": str(cmd.get("id", ""))})
            return {"ok": True}
        if method == "arm_move":
            payload = {
                "cmd": "move",
                "id": str(cmd.get("id", "")),
                "duration": float(cmd.get("duration", 2.0)),
                "label": str(cmd.get("label", "move")),
                "replace": 1.0 if cmd.get("replace", False) else 0.0,
            }
            if "table_z_floor" in cmd and cmd["table_z_floor"] is not None:
                payload["table_z_floor"] = float(cmd["table_z_floor"])
            if "q" in cmd and cmd["q"] is not None:
                payload["q"] = [float(x) for x in cmd["q"]]
            if "aux" in cmd and cmd["aux"] is not None:
                payload["aux"] = [float(x) for x in cmd["aux"]]
            await self.send_arm_cmd(payload)
            return {"ok": True}
        if method == "arm_move_path":
            q_path_raw = cmd.get("q_path") or []
            flat_q = []
            for item in q_path_raw:
                if isinstance(item, (list, tuple)):
                    flat_q.extend(float(x) for x in item)
                else:
                    flat_q.append(float(item))
            payload = {
                "cmd": "move_path",
                "id": str(cmd.get("id", "")),
                "q_path": flat_q,
                "duration": float(cmd.get("duration", 2.2)),
                "label": str(cmd.get("label", "move_path")),
                "replace": 1.0 if cmd.get("replace", False) else 0.0,
            }
            if "table_z_floor" in cmd and cmd["table_z_floor"] is not None:
                payload["table_z_floor"] = float(cmd["table_z_floor"])
            if "aux" in cmd and cmd["aux"] is not None:
                payload["aux"] = [float(x) for x in cmd["aux"]]
            await self.send_arm_cmd(payload)
            return {"ok": True}
        loop = asyncio.get_running_loop()
        if method == "hand_open":
            side = cmd["side"]
            pre = int(cmd.get("pre_open_amount", 0))
            dur = int(cmd.get("duration_ms", 1000))
            posture = str(cmd.get("posture", "flat"))
            two_f = bool(cmd.get("two_finger_only", True))
            t_aux = int(cmd.get("thumb_aux", 485))
            def _do_open():
                with self.hand_lock:
                    self.revo.open_hand(
                        side,
                        pre_open_amount=pre,
                        duration_ms=dur,
                        posture=posture,
                        two_finger_only=two_f,
                        thumb_aux=t_aux,
                    )
            await loop.run_in_executor(None, _do_open)
            return {"ok": True}
        if method == "hand_pose":
            side = cmd["side"]
            targets = [int(v) for v in cmd["targets"]]
            dur = int(cmd.get("duration_ms", 1100))
            def _do_pose():
                with self.hand_lock:
                    self.revo.move_pose(side, targets, duration_ms=dur)
            await loop.run_in_executor(None, _do_pose)
            return {"ok": True}
        if method == "hand_gentle_grasp":
            side = cmd["side"]
            force_N = float(cmd.get("target_force_N", 0.38))
            pre = int(cmd.get("pre_open_amount", 0))
            tripod = bool(cmd.get("use_tripod_only", False))
            learned_prof = cmd.get("learned_profile")
            posture = str(cmd.get("posture", "c_pinch"))
            two_f = bool(cmd.get("two_finger_only", True))
            t_aux = int(cmd.get("thumb_aux", 485))
            exp_w = cmd.get("expected_width_mm")
            exp_w_f = float(exp_w) if exp_w is not None else None
            def _do_grasp():
                with self.hand_lock:
                    self.revo.start_gentle_grasp(
                        side,
                        target_force_N=force_N,
                        pre_open_amount=pre,
                        use_tripod_only=tripod,
                        learned_profile=learned_prof,
                        posture=posture,
                        two_finger_only=two_f,
                        thumb_aux=t_aux,
                        expected_width_mm=exp_w_f,
                    )
            await loop.run_in_executor(None, _do_grasp)
            return {"ok": True}
        if method == "hand_learn_object":
            side = cmd["side"]
            probe_N = float(cmd.get("probe_force_N", 0.44))
            rel_after = bool(cmd.get("release_after", False))
            def _do_learn():
                with self.hand_lock:
                    self.revo.start_learn_object(
                        side,
                        probe_force_N=probe_N,
                        release_after=rel_after,
                    )
            await loop.run_in_executor(None, _do_learn)
            return {"ok": True}
        if method == "hand_gentle_release":
            side = cmd["side"]
            def _do_rel():
                with self.hand_lock:
                    self.revo.release_gentle_grasp(side)
            await loop.run_in_executor(None, _do_rel)
            return {"ok": True}
        if method == "hand_stop":
            def _do_stop():
                with self.hand_lock:
                    self.revo.stop_all()
            await loop.run_in_executor(None, _do_stop)
            return {"ok": True}
        if method == "hand_reconnect":
            with self.hand_lock:
                self.revo.close()
                self.revo = Revo2Manager()
                meta = self.revo.connect()
            return {"ok": True, "hands_meta": meta}
        raise ValueError(f"Unknown method: {method}")

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        sock = writer.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
        peer = writer.get_extra_info("peername")
        print(f"[bridge] Mac client connected: {peer}", flush=True)
        wlock = asyncio.Lock()
        self.clients[writer] = wlock
        try:
            while self.running:
                hdr = await reader.readexactly(5)
                length, mtype = struct.unpack(">IB", hdr)
                body = await reader.readexactly(length - 1)
                if mtype == MSG_JSON:
                    cmd = json.loads(body.decode("utf-8"))
                    req_id = cmd.get("id")
                    try:
                        res = await self.handle_command(cmd)
                        reply = {"type": "reply", "id": req_id, **res}
                    except Exception as exc:
                        reply = {"type": "reply", "id": req_id, "ok": False, "error": str(exc)}
                    out = json.dumps(reply, ensure_ascii=False).encode("utf-8")
                    async with wlock:
                        writer.write(struct.pack(">IB", len(out) + 1, MSG_JSON) + out)
                        await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception as exc:
            print(f"[bridge] client error: {exc}", flush=True)
        finally:
            self.clients.pop(writer, None)
            try:
                writer.close()
            except Exception:
                pass
            print(f"[bridge] Mac client disconnected: {peer}", flush=True)
            # 全クライアント切断時は安全のためハンド・アームをソフトリリース
            if not self.clients:
                with self.hand_lock:
                    self.revo.stop_all()
                try:
                    await self.send_arm_cmd({"cmd": "disengage", "id": "auto_disconnect"})
                except Exception:
                    pass

    async def run_async(self):
        self.loop = asyncio.get_running_loop()
        await self.loop.run_in_executor(None, self.start_hardware)
        threading.Thread(target=self._hand_loop_thread, daemon=True, name="revo2_loop").start()
        await self.start_arm_daemon()
        asyncio.create_task(self.ensure_live_stereo_async())
        asyncio.create_task(self.telemetry_broadcast_loop())
        asyncio.create_task(self.stereo_broadcast_loop())
        asyncio.create_task(self.hd_broadcast_loop())

        server = await asyncio.start_server(self.handle_client, self.host, self.port, reuse_address=True)
        print(f"[bridge] Listening on {self.host}:{self.port}", flush=True)
        async with server:
            while self.running:
                await asyncio.sleep(0.5)

    def shutdown(self):
        self.running = False
        self.left_cam.stop()
        self.right_cam.stop()
        self.hd_cam.stop()
        with self.hand_lock:
            self.revo.close()
        if self.arm_proc and self.arm_proc.returncode is None:
            try:
                self.arm_proc.terminate()
            except Exception:
                pass


def main():
    srv = BridgeServer()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: srv.shutdown())
    try:
        asyncio.run(srv.run_async())
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
