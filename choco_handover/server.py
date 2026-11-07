#!/usr/bin/env python3
"""
Unitree R1 チョコ認識・3D形状取得 ＆「ください」両手手渡しシステム
Mac 側メインサーバー (FastAPI + WebSocket リアルタイム推論・制御ダッシュボード: http://127.0.0.1:8890)
"""
from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from choco_perception import ChocoPerception, ChocoShape3D
from config import BASE_DIR, ConfigStore
from flight_recorder import summarize_benchmark_logs
from handover_fsm import HandoverOrchestrator
from kinematics import R1Kinematics
from palm_gesture import KudasaiGestureDetector, PalmGestureStatus
from r1_client import R1BridgeClient
from stereo_depth import StereoProcessor, StereoResult
from visual_servo import VisualServoObservation

STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR / "samples"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
OBB_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _draw_jaw_bracket(
    canvas: np.ndarray,
    pad_xy: np.ndarray,
    pinch_dir_xy: np.ndarray,
    color: tuple[int, int, int],
    half_len_px: float = 9.0,
    foot_px: float = 4.0,
):
    """ピンチ法線方向 pinch_dir_xy に直交するコの字型フィンガーパッド指標を 1px でシャープに描画する。"""
    tang = np.array([-pinch_dir_xy[1], pinch_dir_xy[0]], dtype=np.float64)
    p_a = pad_xy - half_len_px * tang
    p_b = pad_xy + half_len_px * tang
    p_a_in = p_a + foot_px * pinch_dir_xy
    p_b_in = p_b + foot_px * pinch_dir_xy
    pts = np.round([p_a_in, p_a, p_b, p_b_in]).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)


def draw_annotated_view(
    left_bgr: np.ndarray,
    choco: Optional[ChocoShape3D],
    all_chocos: list[ChocoShape3D],
    gesture: Optional[PalmGestureStatus],
    click_uv: Optional[tuple[int, int]],
    source_tag: str,
    vs_obs: Optional[VisualServoObservation] = None,
    hd_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    vis = left_bgr.copy()
    h, w = vis.shape[:2]

    # 1. 他候補の極細輪郭描画
    for cand in all_chocos:
        if choco is not None and cand is choco:
            continue
        if len(cand.contour_px) >= 3:
            pts = np.array(cand.contour_px, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (110, 145, 80), 1, cv2.LINE_AA)

    # 2. メイン認識オブジェクト: 膨張塗り潰しを廃止し、実寸エッジぴったりの極細 1px 回転枠 ＋ 2指ピンチ軸を描画
    if choco is not None and len(choco.contour_px) >= 3:
        pts = np.array(choco.contour_px, dtype=np.int32).reshape((-1, 1, 2))
        line_col = (40, 255, 120) if choco.reachable else (0, 200, 255)
        cv2.polylines(vis, [pts], True, line_col, 1, cv2.LINE_AA)

        # Stereo 画像上の親指(水色) × 人差指(黄) 計画パッド目盛 (1px)
        if len(choco.st_pinch_line_px) == 2:
            p_th = np.array(choco.st_pinch_line_px[0], dtype=np.float64)
            p_in = np.array(choco.st_pinch_line_px[1], dtype=np.float64)
            dv = p_in - p_th
            dn = float(np.linalg.norm(dv))
            if dn >= 2.0:
                u_dir = dv / dn
                _draw_jaw_bracket(vis, p_th, u_dir, (255, 225, 0), half_len_px=3.5, foot_px=2.0)
                _draw_jaw_bracket(vis, p_in, -u_dir, (0, 235, 255), half_len_px=3.5, foot_px=2.0)

        x1, y1, _, _ = choco.bbox_xyxy
        hand_tag = "L" if choco.recommended_hand == "left" else "R"
        reach_tag = "OK" if choco.reachable else f"-{choco.move_closer_cm:.0f}cm"
        hd_mark = "HD-RAY" if getattr(choco, "xy_source", "") == "hd_plane_ray" else ("HD" if choco.hd_used else "ST")
        info_txt = (
            f"{choco.label} {choco.length_mm:.0f}x{choco.width_mm:.0f}mm "
            f"yaw:{choco.yaw_deg:+.0f}deg [{hand_tag}:{reach_tag}/{hd_mark}]"
        )
        ty = max(22, y1 - 14)
        tx = max(4, min(w - 275, x1 - 24))
        cv2.rectangle(vis, (tx - 3, ty - 13), (min(w - 2, tx + 268), ty + 4), (14, 20, 28), -1)
        txt_col = (110, 255, 165) if choco.reachable else (90, 215, 255)
        cv2.putText(vis, info_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.37, txt_col, 1, cv2.LINE_AA)

        # 2b. 右上ネイティブ 1280x720 HD ズームイン PIP (ボケ無し等倍〜1.48倍・極細1px枠・親指/人差指パッド表示)
        if hd_bgr is not None and len(choco.hd_obb_px) == 4:
            hh, hw = hd_bgr.shape[:2]
            obb_hd = np.array(choco.hd_obb_px, dtype=np.float64)
            cx_hd, cy_hd = float(obb_hd[:, 0].mean()), float(obb_hd[:, 1].mean())
            half_crop = 58  # 116x116 HD patch -> 172x172 PIP (1.48x crisp view)
            cx_i = int(np.clip(round(cx_hd), half_crop, hw - half_crop))
            cy_i = int(np.clip(round(cy_hd), half_crop, hh - half_crop))
            patch = hd_bgr[cy_i - half_crop : cy_i + half_crop, cx_i - half_crop : cx_i + half_crop]
            if patch.shape[0] == 2 * half_crop and patch.shape[1] == 2 * half_crop:
                pip_sz = 172
                scale = pip_sz / float(2 * half_crop)
                pip = cv2.resize(patch, (pip_sz, pip_sz), interpolation=cv2.INTER_CUBIC)
                obb_pip = np.round((obb_hd - np.array([cx_i - half_crop, cy_i - half_crop])) * scale).astype(np.int32)
                cv2.polylines(pip, [obb_pip.reshape((-1, 1, 2))], True, line_col, 1, cv2.LINE_AA)

                if len(choco.hd_pinch_line_px) == 2:
                    pth_h = (np.array(choco.hd_pinch_line_px[0], dtype=np.float64) - np.array([cx_i - half_crop, cy_i - half_crop])) * scale
                    pin_h = (np.array(choco.hd_pinch_line_px[1], dtype=np.float64) - np.array([cx_i - half_crop, cy_i - half_crop])) * scale
                    dvh = pin_h - pth_h
                    dnh = float(np.linalg.norm(dvh))
                    if dnh >= 4.0:
                        uh = dvh / dnh
                        _draw_jaw_bracket(pip, pth_h, uh, (255, 225, 0), half_len_px=11.0, foot_px=5.0)
                        _draw_jaw_bracket(pip, pin_h, -uh, (0, 235, 255), half_len_px=11.0, foot_px=5.0)
                        cv2.putText(
                            pip, "Th", (int(np.clip(pth_h[0] - 8, 3, pip_sz - 22)), int(np.clip(pth_h[1] + 13, 28, pip_sz - 20))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 230, 80), 1, cv2.LINE_AA,
                        )
                        cv2.putText(
                            pip, "In", (int(np.clip(pin_h[0] - 6, 3, pip_sz - 20)), int(np.clip(pin_h[1] - 6, 28, pip_sz - 20))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 240, 255), 1, cv2.LINE_AA,
                        )

                cv2.rectangle(pip, (0, 0), (pip_sz - 1, 18), (12, 18, 26), -1)
                cv2.putText(pip, "HD 1280x720 | Th-In Pinch", (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (185, 230, 255), 1, cv2.LINE_AA)
                cv2.rectangle(pip, (0, pip_sz - 18), (pip_sz - 1, pip_sz - 1), (12, 18, 26), -1)
                cv2.putText(
                    pip,
                    f"rot:{choco.yaw_deg:+.0f}deg pinch:{choco.achieved_pinch_yaw_deg:+.0f}deg",
                    (5, pip_sz - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    txt_col,
                    1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(pip, (0, 0), (pip_sz - 1, pip_sz - 1), (70, 120, 165), 1)
                px0, py0 = w - pip_sz - 8, 8
                vis[py0 : py0 + pip_sz, px0 : px0 + pip_sz] = pip

    # 3. クリック指定マーカー
    if click_uv is not None:
        cu, cv = click_uv
        cv2.drawMarker(vis, (cu, cv), (80, 120, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)

    # 4. MediaPipe 両手骨格 ＆「ください」カップ判定オーバーレイ
    if gesture is not None:
        skel_col = (80, 255, 180) if gesture.pose_valid else (255, 185, 70)
        for hand_px in gesture.landmarks_px:
            for a, b in HAND_CONNECTIONS:
                if a < len(hand_px) and b < len(hand_px):
                    cv2.line(vis, tuple(hand_px[a]), tuple(hand_px[b]), skel_col, 2, cv2.LINE_AA)
            for p in hand_px:
                cv2.circle(vis, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # 左右の小指付け根を結ぶカップライン & 両掌中心ターゲット描画
        if len(gesture.landmarks_px) == 2:
            pL = tuple(gesture.landmarks_px[0][17])
            pR = tuple(gesture.landmarks_px[1][17])
            cup_col = (50, 255, 120) if gesture.cupped_together else (90, 110, 255)
            cv2.line(vis, pL, pR, cup_col, 2, cv2.LINE_AA)

        if gesture.palm_center_px is not None:
            pc = tuple(gesture.palm_center_px)
            radius = 22
            cv2.circle(vis, pc, radius, (70, 90, 110), 2, cv2.LINE_AA)
            if gesture.hold_progress > 0.01:
                end_ang = int(round(360.0 * gesture.hold_progress))
                cv2.ellipse(vis, pc, (radius, radius), -90, 0, end_ang, (40, 255, 140), 4, cv2.LINE_AA)
            cv2.drawMarker(vis, pc, (255, 240, 90), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)

    # 5. Eye-to-Hand ビジュアルサーボ指先ピンチ ⇄ 目標ホバリング位置オーバーレイ
    if vs_obs is not None and vs_obs.pinch_uv is not None:
        if vs_obs.thumb_uv is not None:
            cv2.circle(vis, tuple(vs_obs.thumb_uv), 4, (0, 235, 255), -1, cv2.LINE_AA)
        if vs_obs.index_uv is not None:
            cv2.circle(vis, tuple(vs_obs.index_uv), 4, (255, 220, 0), -1, cv2.LINE_AA)
        cv2.drawMarker(vis, tuple(vs_obs.pinch_uv), (70, 80, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        if vs_obs.target_uv_at_hover is not None:
            cv2.arrowedLine(
                vis, tuple(vs_obs.pinch_uv), tuple(vs_obs.target_uv_at_hover),
                (70, 255, 110), 2, cv2.LINE_AA, tipLength=0.25,
            )

    # ソースバッジ
    cv2.rectangle(vis, (6, 6), (178, 26), (15, 20, 28), -1)
    cv2.putText(vis, f"SRC: {source_tag}", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 220, 255), 1, cv2.LINE_AA)
    return vis


class AppRuntime:
    def __init__(self):
        self.store = ConfigStore()
        self.kin = R1Kinematics()
        self.client = R1BridgeClient(self.store.cfg.r1_host, self.store.cfg.r1_port)
        self.stereo_proc = StereoProcessor()
        self.choco_perc = ChocoPerception()
        self.gesture_det = KudasaiGestureDetector()
        self.orchestrator = HandoverOrchestrator(self.store, self.kin, self.client)

        self.source_mode: str = "auto"  # "auto" | "r1_live" | "mac_webcam" | "r1_sample"
        self.webcam: Optional[cv2.VideoCapture] = None
        self.ws_clients: set[WebSocket] = set()
        self.running = True
        self.latest_frame_packet: Optional[dict] = None
        self._last_auto_deploy_check = 0.0
        self._head_yaw_zero_q: Optional[float] = None

    def _get_fallback_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        cfg = self.store.cfg
        if self.source_mode == "mac_webcam":
            if self.webcam is None or not self.webcam.isOpened():
                self.webcam = cv2.VideoCapture(0)
            ok, frame = self.webcam.read()
            if ok and frame is not None:
                resized = cv2.resize(frame, (cfg.stereo_width, cfg.stereo_height))
                # 単眼テスト時は仮想視差 (約0.48m相当の18pxシフト) を与えて 3D 形状・ジェスチャーパイプライン全体を検証可能にする
                M = np.float32([[1, 0, -18], [0, 1, 0]])
                pseudo_r = cv2.warpAffine(resized, M, (cfg.stereo_width, cfg.stereo_height), borderMode=cv2.BORDER_REPLICATE)
                return resized, pseudo_r, "MAC_WEBCAM_TEST"

        # R1 実機から取得済みの本物の整準化ステレオペア (samples/udp_5002.jpg, udp_5003.jpg)
        p_l = SAMPLES_DIR / "udp_5002.jpg"
        p_r = SAMPLES_DIR / "udp_5003.jpg"
        if p_l.exists() and p_r.exists():
            img_l = cv2.imread(str(p_l))
            img_r = cv2.imread(str(p_r))
            if img_l is not None and img_r is not None:
                return img_l, img_r, "R1_CAPTURED_STEREO (OFFLINE)"
        return None, None, "NO_VIDEO"

    async def auto_deploy_watcher(self):
        """R1 (192.168.1.13) の電源投入を検知したら自動でオンボードブリッジを起動する。"""
        while self.running:
            if not self.client.connected and self.client.deploy_status != "deploying":
                host = self.store.cfg.r1_host
                ok_ping = await asyncio.to_thread(
                    lambda: subprocess.run(
                        ["ping", "-c", "1", "-t", "1", host],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    ).returncode == 0
                )
                if ok_ping and (time.monotonic() - self._last_auto_deploy_check) > 12.0:
                    self._last_auto_deploy_check = time.monotonic()
                    await self.client.deploy_and_start_remote()
            await asyncio.sleep(3.5)

    async def vision_loop(self):
        while self.running:
            t0 = time.monotonic()
            cfg = self.store.cfg

            left_bgr = right_bgr = hd_bgr = None
            src_tag = "OFFLINE"
            if self.source_mode in ("auto", "r1_live") and self.client.connected and self.client.latest_stereo is not None:
                left_bgr, right_bgr, _ = self.client.latest_stereo
                if self.client.latest_hd is not None and (time.time() - self.client.latest_hd[1]) <= 4.5:
                    hd_bgr = self.client.latest_hd[0]
                hd_lbl = "+HD" if hd_bgr is not None else ""
                src_tag = f"R1_STEREO{hd_lbl} ({self.client.stereo_fps:.0f}fps)"
                if self.webcam is not None:
                    self.webcam.release()
                    self.webcam = None
            elif self.source_mode != "r1_live":
                left_bgr, right_bgr, src_tag = await asyncio.to_thread(self._get_fallback_frames)
                p_hd = SAMPLES_DIR / "udp_5001.jpg"
                if p_hd.exists() and self.source_mode != "mac_webcam":
                    hd_bgr = await asyncio.to_thread(cv2.imread, str(p_hd))

            if left_bgr is not None and right_bgr is not None:
                rob = self.client.latest_telemetry.get("robot", {})
                q_aux = rob.get("q_aux")
                if q_aux and len(q_aux) == 3 and self.client.connected:
                    if self._head_yaw_zero_q is None:
                        self._head_yaw_zero_q = float(q_aux[2])
                    head_pitch = float(q_aux[1])
                    if cfg.use_measured_head_yaw:
                        head_yaw = float(q_aux[2])
                    else:
                        head_yaw = float(cfg.look_head_yaw) + (float(q_aux[2]) - self._head_yaw_zero_q)
                else:
                    if not self.client.connected:
                        self._head_yaw_zero_q = None
                    head_pitch, head_yaw = cfg.look_head_pitch, cfg.look_head_yaw

                def _process_frame():
                    st_res: StereoResult = self.stereo_proc.compute(left_bgr, right_bgr, cfg)
                    choco, all_chocos = self.choco_perc.detect_and_extract_shape(
                        left_bgr, st_res, self.kin, head_pitch, head_yaw, cfg, hd_bgr=hd_bgr
                    )
                    gest = self.gesture_det.evaluate(
                        left_bgr, st_res, self.kin, head_pitch, head_yaw, cfg
                    )
                    vis_bgr = draw_annotated_view(
                        left_bgr, choco, all_chocos, gest, self.choco_perc.click_uv, src_tag,
                        vs_obs=self.orchestrator.visual_servo.last_obs,
                        hd_bgr=hd_bgr,
                    )
                    _, jpg_vis = cv2.imencode(".jpg", vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    _, jpg_disp = cv2.imencode(".jpg", st_res.disp_vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
                    return st_res, choco, gest, jpg_vis.tobytes(), jpg_disp.tobytes(), st_res.table_plane is not None

                st_res, choco, gest, jpg_vis, jpg_disp, has_plane = await asyncio.to_thread(_process_frame)
                if gest.triggered_just_now:
                    self.gesture_det.reset_hold(cooldown_s=5.0)
                self.orchestrator.update_perception(
                    choco, gest, left_bgr=left_bgr, stereo=st_res, head_pitch=head_pitch, head_yaw=head_yaw
                )

                choco_dict = None
                if choco is not None:
                    choco_dict = asdict(choco)
                    # WebSocket 帯域節約のため長いポリゴン配列は要約も保持
                    choco_dict["contour_vertices"] = len(choco.contour_px)

                gest_dict = asdict(gest)
                gest_dict.pop("landmarks_px", None)

                packet = {
                    "t": round(time.time(), 3),
                    "source": src_tag,
                    "source_mode": self.source_mode,
                    "r1_connected": self.client.connected,
                    "r1_error": self.client.last_connect_error,
                    "deploy_status": self.client.deploy_status,
                    "deploy_log": self.client.deploy_log,
                    "table_plane_detected": has_plane,
                    "vision_ms": round((time.monotonic() - t0) * 1000.0, 1),
                    "model_error": self.choco_perc.model_error,
                    "choco": choco_dict,
                    "gesture": gest_dict,
                    "task": asdict(self.orchestrator.status),
                    "tactile_memory": self.orchestrator.profile_store.snapshot(),
                    "telemetry": self.client.latest_telemetry,
                    "rgb_b64": base64.b64encode(jpg_vis).decode("ascii"),
                    "disp_b64": base64.b64encode(jpg_disp).decode("ascii"),
                }
                self.latest_frame_packet = packet

                if self.ws_clients:
                    txt = json.dumps(packet, ensure_ascii=False)
                    dead = []
                    for ws in list(self.ws_clients):
                        try:
                            await ws.send_text(txt)
                        except Exception:
                            dead.append(ws)
                    for ws in dead:
                        self.ws_clients.discard(ws)

            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.02, 0.095 - elapsed))


runtime = AppRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    # 事前に取得した実機ステレオフレームがあれば samples/ にコピーしておく
    scratch_dir = Path("/Users/hoshinafumito/.gemini/antigravity/brain/acca1813-c27d-4ce6-8a31-8e0b3a452165/scratch")
    for fn in ("udp_5002.jpg", "udp_5003.jpg"):
        src = scratch_dir / fn
        dst = SAMPLES_DIR / fn
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)

    t1 = asyncio.create_task(runtime.client.connection_loop())
    t2 = asyncio.create_task(runtime.vision_loop())
    t3 = asyncio.create_task(runtime.auto_deploy_watcher())
    yield
    runtime.running = False
    runtime.client.running = False
    for t in (t1, t2, t3):
        t.cancel()
    if runtime.webcam is not None:
        runtime.webcam.release()


app = FastAPI(title="Unitree R1 Choco Handover Studio", lifespan=lifespan)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status(compact: int = 1):
    if runtime.latest_frame_packet is None:
        return {"ok": False, "r1_connected": runtime.client.connected}
    if compact:
        d = {k: v for k, v in runtime.latest_frame_packet.items() if k not in ("rgb_b64", "disp_b64")}
        d["ok"] = True
        return d
    return runtime.latest_frame_packet


@app.get("/api/raw_stereo")
async def get_raw_stereo():
    if runtime.client.latest_stereo is None:
        return JSONResponse({"ok": False, "error": "No live stereo frame yet"}, status_code=404)
    img_l, img_r, ts = runtime.client.latest_stereo
    _, jl = cv2.imencode(".jpg", img_l, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    _, jr = cv2.imencode(".jpg", img_r, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    out = {
        "ok": True,
        "ts": ts,
        "left_b64": base64.b64encode(jl.tobytes()).decode("ascii"),
        "right_b64": base64.b64encode(jr.tobytes()).decode("ascii"),
    }
    if runtime.client.latest_hd is not None:
        img_hd, ts_hd = runtime.client.latest_hd
        _, jh = cv2.imencode(".jpg", img_hd, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        out["hd_ts"] = ts_hd
        out["hd_b64"] = base64.b64encode(jh.tobytes()).decode("ascii")
    return out


@app.get("/api/config")
async def get_config():
    return runtime.store.snapshot()


@app.post("/api/config")
async def update_config(req: Request):
    patch = await req.json()
    snap = runtime.store.update(patch)
    runtime.client.set_endpoint(snap["r1_host"], snap["r1_port"])
    return {"ok": True, "config": snap}


@app.post("/api/source_mode")
async def set_source_mode(req: Request):
    body = await req.json()
    mode = str(body.get("mode", "auto"))
    if mode in ("auto", "r1_live", "mac_webcam", "r1_sample"):
        runtime.source_mode = mode
        if mode != "mac_webcam" and runtime.webcam is not None:
            runtime.webcam.release()
            runtime.webcam = None
        return {"ok": True, "source_mode": mode}
    return JSONResponse({"ok": False, "error": "Invalid mode"}, status_code=400)


@app.post("/api/click_target")
async def set_click_target(req: Request):
    body = await req.json()
    if body.get("clear"):
        runtime.choco_perc.set_click_hint(None, None)
    else:
        runtime.choco_perc.set_click_hint(int(body["u"]), int(body["v"]))
    return {"ok": True, "click_uv": runtime.choco_perc.click_uv}


@app.post("/api/trigger")
async def manual_trigger():
    res = await runtime.orchestrator.start_handover(trigger_source="manual_button")
    return res


@app.post("/api/trigger_fixed")
async def trigger_fixed_benchmark(req: Request):
    body = await req.json() if req.headers.get("content-length") else {}
    return await runtime.orchestrator.start_fixed_benchmark(
        x_m=float(body.get("x_m", 0.36)),
        y_m=float(body.get("y_m", -0.06)),
        z_table_m=(float(body["z_table_m"]) if body.get("z_table_m") is not None else None),
        yaw_deg=float(body.get("yaw_deg", 0.0)),
        offset_x_mm=float(body.get("offset_x_mm", 0.0)),
        offset_y_mm=float(body.get("offset_y_mm", 0.0)),
        length_mm=float(body.get("length_mm", 33.0)),
        width_mm=float(body.get("width_mm", 26.0)),
        height_mm=float(body.get("height_mm", 14.0)),
        side=body.get("side"),
    )


@app.get("/api/benchmark_summary")
async def get_benchmark_summary(limit: int = 60):
    return {"ok": True, **summarize_benchmark_logs(limit=limit)}


@app.post("/api/estop")
async def estop():
    await runtime.orchestrator.emergency_stop()
    return {"ok": True}


@app.post("/api/hold_here")
async def hold_here():
    return await runtime.orchestrator.hold_arm_in_place()


@app.post("/api/safe_return")
async def safe_return(req: Request):
    body = await req.json() if req.headers.get("content-length") else {}
    open_first = bool(body.get("open_hand_first", True))
    return await runtime.orchestrator.safe_return_and_release(open_hand_first=open_first)


@app.post("/api/deploy_r1")
async def deploy_r1():
    return await runtime.client.deploy_and_start_remote()


@app.post("/api/center_head")
async def center_head():
    return await runtime.orchestrator.center_head_forward()


@app.post("/api/finger_shape_scan")
async def finger_shape_scan(req: Request):
    body = await req.json()
    return await runtime.orchestrator.finger_shape_scan(
        side=body.get("side", "right"),
        poses=body.get("poses"),
        hold_s=float(body.get("hold_s", 6.0)),
    )


@app.post("/api/hand_test")
async def hand_test(req: Request):
    body = await req.json()
    side = body.get("side", "right")
    action = body.get("action", "gentle_grasp")
    if not runtime.client.connected:
        return {"ok": False, "error": "R1未接続"}
    if action == "open":
        cfg = runtime.store.cfg
        return await runtime.client.send_command(
            "hand_open",
            side=side,
            pre_open_amount=0,
            duration_ms=900,
            posture=cfg.grasp_posture,
            two_finger_only=cfg.two_finger_pinch,
        )
    if action == "pose":
        # 指形状キャリブレーション用: アームは動かさず指 6 軸のみ指定姿勢へ (安全範囲にクランプ)
        tg = body.get("targets") or []
        if len(tg) != 6:
            return {"ok": False, "error": "targets must have 6 values"}
        lim_hi = [680, 498, 780, 880, 720, 650]
        safe = [max(0, min(lim_hi[i], int(tg[i]))) for i in range(6)]
        safe[1] = max(435, safe[1])
        return await runtime.client.send_command(
            "hand_pose", side=side, targets=safe, duration_ms=int(body.get("duration_ms", 1100))
        )
    if action == "release":
        return await runtime.client.send_command("hand_gentle_release", side=side)
    if action == "gentle_grasp":
        return await runtime.orchestrator.test_hand_gentle_grasp_only(side)
    return {"ok": False, "error": "Unknown hand action"}


@app.post("/api/learn_object")
async def learn_object(req: Request):
    body = await req.json()
    side = str(body.get("side", "right"))
    name = str(body.get("name", "白キャップ"))
    release_after = bool(body.get("release_after", False))
    return await runtime.orchestrator.learn_in_hand_object(
        side=side, name=name, release_after=release_after
    )


@app.get("/api/object_profiles")
async def get_object_profiles():
    return runtime.orchestrator.profile_store.snapshot()


@app.post("/api/object_profiles/select")
async def select_object_profile(req: Request):
    body = await req.json()
    prof_id = body.get("profile_id")
    enabled = body.get("enabled")
    snap = runtime.orchestrator.profile_store.select(profile_id=prof_id, enabled=enabled)
    return {"ok": True, "tactile_memory": snap}


@app.post("/api/object_profiles/delete")
async def delete_object_profile(req: Request):
    body = await req.json()
    prof_id = str(body.get("profile_id", ""))
    snap = runtime.orchestrator.profile_store.delete(prof_id)
    return {"ok": True, "tactile_memory": snap}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    runtime.ws_clients.add(ws)
    try:
        if runtime.latest_frame_packet is not None:
            await ws.send_text(json.dumps(runtime.latest_frame_packet, ensure_ascii=False))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        runtime.ws_clients.discard(ws)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8890, log_level="info")
