"""
「机上チョコ認識・形状抽出 → 指先近接センサー減速＆優しさ把持 → 人の両掌『ください』への手渡し」
統合ステートマシン (FSM) オーケストレータ。
手動ボタン発火・両手『ください』ジェスチャー自動発火・Dry-Run(軌道/IKプレビュー)・ハンド単体把持テストに対応。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from choco_perception import ChocoShape3D
from config import ConfigStore
from flight_recorder import FlightRecorderSession
from kinematics import CARRY_READY_LEFT, CARRY_READY_RIGHT, REST_ARM_POSE, R1Kinematics
from palm_gesture import PalmGestureStatus
from r1_client import R1BridgeClient
from stereo_depth import StereoResult
from tactile_memory import ObjectProfileStore
from visual_servo import StereoVisualServo

# 指形状撮影後に戻すハンド待機姿勢 (表面拘束またぎ開き)
STRADDLE_SCAN_REST = [60, 473, 150, 145, 10, 10]


@dataclass
class HandoverTaskStatus:
    phase: str = "IDLE"
    message: str = "待機中（机上のチョコと『ください』ポーズを監視しています）"
    selected_hand: Optional[str] = None
    locked_choco: Optional[dict] = None
    target_palms_torso_m: Optional[List[float]] = None
    planned_ik_error_mm: Optional[float] = None
    visual_servo: Optional[dict] = None
    last_run_summary: Optional[dict] = None
    step_history: List[str] = field(default_factory=list)
    updated_at: float = 0.0


class HandoverOrchestrator:
    def __init__(
        self,
        store: ConfigStore,
        kin: R1Kinematics,
        client: R1BridgeClient,
        profile_store: Optional[ObjectProfileStore] = None,
    ):
        self.store = store
        self.kin = kin
        self.client = client
        self.profile_store = profile_store or ObjectProfileStore()
        self.visual_servo = StereoVisualServo()
        self.status = HandoverTaskStatus(updated_at=time.time())
        self._active_task: Optional[asyncio.Task] = None
        self._abort_flag = False
        self.latest_choco: Optional[ChocoShape3D] = None
        self.latest_gesture: Optional[PalmGestureStatus] = None
        self.latest_left_bgr: Optional[np.ndarray] = None
        self.latest_stereo: Optional[StereoResult] = None
        self.latest_head_angles: tuple[float, float] = (0.58, 0.0)
        self._recorded_initial_q10: Optional[List[float]] = None
        self._last_table_bbox: Optional[dict] = None
        self._fixed_override_choco: Optional[ChocoShape3D] = None

    @property
    def is_busy(self) -> bool:
        return self._active_task is not None and not self._active_task.done()

    def _set_phase(self, phase: str, msg: str):
        self.status.phase = phase
        self.status.message = msg
        self.status.updated_at = time.time()
        stamp = time.strftime("%H:%M:%S")
        self.status.step_history.append(f"[{stamp}] {phase}: {msg}")
        if len(self.status.step_history) > 25:
            self.status.step_history = self.status.step_history[-25:]

    def update_perception(
        self,
        choco: Optional[ChocoShape3D],
        gesture: Optional[PalmGestureStatus],
        left_bgr: Optional[np.ndarray] = None,
        stereo: Optional[StereoResult] = None,
        head_pitch: float = 0.58,
        head_yaw: float = 0.0,
    ):
        self.latest_choco = choco
        self.latest_gesture = gesture
        if left_bgr is not None:
            self.latest_left_bgr = left_bgr
        if stereo is not None:
            self.latest_stereo = stereo
        self.latest_head_angles = (float(head_pitch), float(head_yaw))
        cfg = self.store.cfg
        if (
            not self.is_busy
            and cfg.auto_gesture_trigger
            and gesture is not None
            and gesture.triggered_just_now
        ):
            # 「ください」ポーズ1秒維持で自動発火！
            asyncio.create_task(self.start_handover(trigger_source="gesture_kudasai"))

    async def emergency_stop(self):
        self._abort_flag = True
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
        self._set_phase("ESTOP", "非常停止を実行しました（ハンド停止・脚0Nm粘性ソフト減衰）")
        if self.client.connected:
            try:
                await self.client.send_command("estop", timeout=2.0)
            except Exception:
                pass

    async def hold_arm_in_place(self) -> Dict:
        """進行中の軌道があれば即キャンセルし、現在の関節角のまま低剛性で静かにその場停止する（暴れゼロ）。"""
        self._abort_flag = True
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
        self._set_phase("HOLD_IN_PLACE", "その場でアーム・ハンドを静かに停止保持しました")
        if not self.client.connected:
            return {"ok": False, "error": "R1未接続"}
        try:
            await self.client.send_command("hand_stop", timeout=2.0)
        except Exception:
            pass
        try:
            return await self.client.send_command("arm_hold_here", timeout=2.0)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def safe_return_and_release(self, open_hand_first: bool = True) -> Dict:
        """ユーザーの明示操作時のみ、箱・机を避けてゆっくり待機姿勢へ戻しソフト脱力する。"""
        if self.is_busy:
            return {"ok": False, "error": "現在動作中です。先に『その場で静止』を押してください"}
        if not self.client.connected:
            return {"ok": False, "error": "R1未接続"}

        async def _retreat_job():
            cfg = self.store.cfg
            side = self.status.selected_hand or "right"
            rob = self.client.latest_telemetry.get("robot", {})
            cur_q10 = rob.get("q_arm") or list(REST_ARM_POSE)
            home_q10 = list(self._recorded_initial_q10) if self._recorded_initial_q10 else list(cur_q10)
            cur_aux = rob.get("q_aux") or [cfg.look_waist_yaw, cfg.look_head_pitch, cfg.look_head_yaw]
            try:
                self._set_phase("SAFE_RETURN", "安全退避ルートでゆっくり初期姿勢へ戻っています...")
                if open_hand_first:
                    await self.client.send_command("hand_gentle_release", side=side)
                    await self._sleep_checked(1.2)
                if rob.get("engaged"):
                    wp0, wp1, wp2 = self.kin.build_safe_table_waypoints(
                        home_q10, side, table_bbox=self._last_table_bbox
                    )
                    q5_cur = cur_q10[:5] if side == "left" else cur_q10[5:10]
                    if len(q5_cur) == 5 and float(q5_cur[0]) >= 0.10:
                        # すでに天板手前より後方(体側)に腕がある場合は天板上空 wp2 へ振り上げず、そのまま外側退避 wp0 経由で戻す
                        await self._move_and_wait(wp0, cur_aux, 3.0, "retreat_side_clear", replace=True)
                    else:
                        await self._move_and_wait(wp2, cur_aux, 2.6, "retreat_over_box", replace=True)
                        await self._move_and_wait(wp1, cur_aux, 3.0, "retreat_behind_box")
                        await self._move_and_wait(wp0, cur_aux, 3.0, "retreat_side_clear")
                    await self._move_and_wait(home_q10, cur_aux, 3.0, "return_rest")
                    await self.client.send_command("arm_disengage")
                    await self._sleep_checked(2.6)
                self.visual_servo.clear()
                self._set_phase("IDLE", "初期姿勢への帰還とソフト脱力が完了しました（待機中）")
            except asyncio.CancelledError:
                self._set_phase("HOLD_IN_PLACE", "帰還を中断しその場で停止しました")
            except Exception as exc:
                self._set_phase("ERROR", f"安全停止維持: {exc}")

        self._abort_flag = False
        self._active_task = asyncio.create_task(_retreat_job())
        return {"ok": True}

    async def center_head_forward(self) -> Dict:
        """現在横を向いている頭部 (head_yaw=30, head_pitch=29) を正面・机見下ろし角へ滑らかに向ける。"""
        if not self.client.connected:
            return {"ok": False, "error": "R1未接続"}
        cfg = self.store.cfg
        rob = self.client.latest_telemetry.get("robot", {})
        bms_soc = rob.get("bms", {}).get("soc", 100)
        if bms_soc <= 10:
            return {"ok": False, "error": f"バッテリー残量低下 ({bms_soc}%) のため充電後に実行してください"}

        q_arm = rob.get("q_arm") or list(REST_ARM_POSE)
        cur_aux = list(rob.get("q_aux") or [cfg.look_waist_yaw, cfg.look_head_pitch, cfg.look_head_yaw])
        target_waist = float(cur_aux[0]) if cfg.use_suspended_mode else float(cfg.look_waist_yaw)
        # motor 30 (head_yaw) の電源再投入時エンコーダオフセットによる首急回転を防止するため現在角を維持
        target_hy = float(cfg.look_head_yaw) if getattr(cfg, "use_measured_head_yaw", False) else float(cur_aux[2])
        target_aux = [target_waist, float(cfg.look_head_pitch), target_hy]
        r_eng = await self.client.send_command(
            "arm_engage",
            suspended=(1.0 if cfg.use_suspended_mode else 0.0),
            grav_ff_scale=float(cfg.grav_ff_scale),
            extra_hand_mass_kg=float(cfg.extra_hand_mass_kg),
        )
        if not r_eng.get("ok"):
            return r_eng
        await asyncio.sleep(1.6)
        r_mv = await self.client.send_command(
            "arm_move",
            q=q_arm,
            aux=target_aux,
            duration=2.8,
            label="center_head",
            replace=True,
        )
        return r_mv

    async def finger_shape_scan(self, side: str = "right", poses: Optional[List[List[int]]] = None, hold_s: float = 6.0) -> Dict:
        """
        指形状キャリブレーション: 天板干渉ゼロ経路で腕を机上空 (+120mm) の把持姿勢へ上げ、
        指定した複数のハンド形状を順に保持して撮影できるようにし、終わったら同じ経路で初期姿勢へ戻して脱力する。
        """
        if self.is_busy:
            return {"ok": False, "error": "現在別のシーケンスを実行中です"}
        if not self.client.connected:
            return {"ok": False, "error": "R1未接続"}
        cfg = self.store.cfg
        rob0 = self.client.latest_telemetry.get("robot") or {}
        if int((rob0.get("bms") or {}).get("soc", 100)) <= 15:
            return {"ok": False, "error": "バッテリー残量不足"}
        poses = poses or [
            [60, 473, 150, 145, 10, 10],
            [60, 473, 300, 300, 10, 10],
            [60, 473, 450, 450, 10, 10],
            [0, 495, 450, 450, 10, 10],
        ]
        ch = self.latest_choco
        if ch is not None:
            z_tbl = float(ch.table_z_torso_m)
            hx, hy = float(ch.centroid_torso_m[0]), float(ch.centroid_torso_m[1])
            tbl_bbox = dict(ch.table_bbox_torso) if getattr(ch, "table_bbox_torso", None) else self._last_table_bbox
        else:
            z_tbl, hx, hy, tbl_bbox = -0.035, 0.34, (0.03 if side == "left" else -0.02), self._last_table_bbox
        hover_pos = self.kin.clamp_workspace(np.array([hx, hy, z_tbl + 0.12], dtype=np.float64), side, cfg)
        q5_hover, err_h, _, _ = self.kin.solve_oriented_pinch_ik(side, hover_pos, 0.0, cfg)
        if err_h > 0.012:
            return {"ok": False, "error": f"上空姿勢の IK 残差が大きすぎます ({err_h*1000:.1f}mm)"}
        side_ja = "左手" if side == "left" else "右手"

        async def _scan_job():
            initial_q10 = None
            engaged = False
            cur_aux = None
            wps = None
            try:
                rob = self.client.latest_telemetry.get("robot") or {}
                initial_q10 = list(rob.get("q_arm") or list(REST_ARM_POSE))
                cur_aux = list(rob.get("q_aux") or [cfg.look_waist_yaw, cfg.look_head_pitch, cfg.look_head_yaw])
                self._set_phase("FINGER_SCAN", f"{side_ja}の腕を机上空 (+120mm) へ上げる準備中")
                r_eng = await self.client.send_command(
                    "arm_engage",
                    suspended=(1.0 if cfg.use_suspended_mode else 0.0),
                    table_z_floor=float(max(-0.16, min(0.12, z_tbl - 0.036))),
                    grav_ff_scale=float(cfg.grav_ff_scale),
                    extra_hand_mass_kg=float(cfg.extra_hand_mass_kg),
                )
                if not r_eng.get("ok"):
                    self._set_phase("ERROR", f"アーム制御開始失敗: {r_eng.get('error')}")
                    return
                deadline = time.monotonic() + 9.5
                while time.monotonic() < deadline:
                    await self._sleep_checked(0.15)
                    rn = self.client.latest_telemetry.get("robot") or {}
                    if rn.get("engaged") and rn.get("stage") == "holding":
                        engaged = True
                        break
                if not engaged:
                    self._set_phase("ERROR", "エンゲージ待機タイムアウト")
                    return
                wps = self.kin.build_safe_table_waypoints(initial_q10, side, table_bbox=tbl_bbox)
                wp0, wp1, wp2 = wps
                q10_hover = self.kin.merge_single_arm(wp2, side, q5_hover)
                self._set_phase("FINGER_SCAN", f"{side_ja}を天板干渉ゼロ経路で机上空へ移動中")
                await self._move_and_wait(wp0, cur_aux, 2.2, "scan_side_clear", replace=True)
                await self._move_and_wait(wp1, cur_aux, 3.0, "scan_lift_behind_desk")
                await self._move_and_wait(wp2, cur_aux, 2.6, "scan_swing_over_desk")
                await self._move_and_wait(q10_hover, cur_aux, 2.2, "scan_hover")
                labels = "ABCDEFGH"
                for k, tg in enumerate(poses):
                    lbl = labels[k] if k < len(labels) else str(k)
                    self._set_phase("FINGER_SCAN", f"形{lbl} {tg} を {hold_s:.0f} 秒保持中（撮影してください）")
                    await self.client.send_command("hand_pose", side=side, targets=[int(v) for v in tg], duration_ms=1100)
                    await self._sleep_checked(hold_s)
                self._set_phase("FINGER_SCAN", "撮影完了 → 同じ経路で腕を下ろしています")
                await self.client.send_command("hand_pose", side=side, targets=list(STRADDLE_SCAN_REST), duration_ms=1000)
                await self._move_and_wait(wp2, cur_aux, 2.4, "scan_back_over_desk", replace=True)
                await self._move_and_wait(wp1, cur_aux, 3.0, "scan_back_behind_desk")
                await self._move_and_wait(wp0, cur_aux, 3.0, "scan_back_side_clear")
                await self._move_and_wait(initial_q10, cur_aux, 3.0, "scan_return_rest")
                await self.client.send_command("arm_disengage")
                await self._sleep_checked(2.6)
                self._set_phase("IDLE", "指形状撮影が完了し、腕を初期姿勢へ戻して脱力しました")
            except asyncio.CancelledError:
                self._set_phase("HOLD_IN_PLACE", "指形状撮影を中断し、その場で停止しました")
            except Exception as exc:
                self._set_phase("ERROR", f"指形状撮影エラー（その場で安全保持）: {exc}")

        self._abort_flag = False
        self._active_task = asyncio.create_task(_scan_job())
        return {"ok": True, "side": side, "hover_torso_m": [round(float(v), 4) for v in hover_pos], "ik_err_mm": round(err_h * 1000, 2)}

    async def test_hand_gentle_grasp_only(self, side: str) -> Dict:
        """アームを動かさず、指定側の Revo 2 ハンド単体で「近接センサー減速＋優しさ把持」をテストする。"""
        if not self.client.connected:
            return {"ok": False, "error": "R1未接続"}
        cfg = self.store.cfg
        active_prof = self.profile_store.get_active_for_grasp()
        force_n = float(active_prof["recommended_force_N"]) if active_prof else cfg.gentle_force_N
        self._set_phase(
            "HAND_TEST",
            f"{'左手' if side == 'left' else '右手'} Revo 2 の近接減速＋優しさ把持 ({force_n:.2f} N) を開始",
        )
        if active_prof and len(active_prof.get("pre_open_positions", [])) == 6 and not cfg.two_finger_pinch:
            await self.client.send_command("hand_pose", side=side, targets=active_prof["pre_open_positions"], duration_ms=900)
        else:
            await self.client.send_command(
                "hand_open",
                side=side,
                pre_open_amount=25,
                duration_ms=900,
                posture=cfg.grasp_posture,
                two_finger_only=cfg.two_finger_pinch,
            )
        await asyncio.sleep(1.1)
        return await self.client.send_command(
            "hand_gentle_grasp",
            side=side,
            target_force_N=force_n,
            pre_open_amount=25,
            use_tripod_only=cfg.use_tripod_grasp,
            two_finger_only=cfg.two_finger_pinch,
            learned_profile=active_prof,
            posture=cfg.grasp_posture,
        )

    async def learn_in_hand_object(
        self,
        side: str = "right",
        name: str = "白キャップ",
        release_after: bool = False,
    ) -> Dict:
        """
        ユーザーが事前にハンド指先へセットした状態からそのまま触診を開始し、
        オブジェクトの挟み角・幅・接触剛性・法線力・近接変化を認識して記憶保存する（アーム動作ゼロ）。
        """
        if self.is_busy:
            return {"ok": False, "error": "現在別のシーケンスを実行中です"}
        if not self.client.connected:
            return {"ok": False, "error": "R1 が未接続です（R1の電源を入れて接続後に実行してください）"}

        clean_name = (name or "").strip() or "白キャップ"
        side_ja = "左ハンド" if side == "left" else "右ハンド"

        async def _learn_job():
            cfg = self.store.cfg
            try:
                self._set_phase(
                    "LEARN_OBJECT_SCAN",
                    f"{side_ja} の現在位置から『{clean_name}』の指先認識・触診を開始しました（アーム固定）",
                )
                r_cmd = await self.client.send_command(
                    "hand_learn_object",
                    side=side,
                    probe_force_N=max(0.38, cfg.gentle_force_N),
                    release_after=release_after,
                )
                if not r_cmd.get("ok"):
                    self._set_phase("ERROR", f"ハンド触覚記憶コマンド失敗: {r_cmd.get('error')}")
                    return

                deadline = time.monotonic() + 12.5
                last_detail = ""
                while time.monotonic() < deadline:
                    await self._sleep_checked(0.10)
                    hands = self.client.latest_telemetry.get("hands", {})
                    g = ((hands.get(side) or {}).get("grasp")) or {}
                    if g.get("mode") != "learn_object":
                        continue
                    st = g.get("stage", "")
                    det = g.get("detail", "")
                    if det and det != last_detail and st not in ("memorized_holding", "memorized_released", "learn_failed"):
                        last_detail = det
                        self._set_phase("LEARN_OBJECT_SCAN", f"{side_ja}: {det}")

                    if st in ("memorized_holding", "memorized_released") and g.get("learned_result"):
                        prof = self.profile_store.save_from_hand_result(clean_name, side, g["learned_result"])
                        hold_note = "自動リリース済み" if release_after else "落下防止のため保持中（『ハンドを開く』で外せます）"
                        self._set_phase(
                            "OBJECT_MEMORIZED",
                            f"【記憶完了】『{prof.name}』を登録しました！ 推定幅 {prof.estimated_span_mm}mm / {prof.hardness_label} / "
                            f"推奨力 {prof.recommended_force_N}N / 接触指: {'・'.join(prof.active_finger_names)} （{hold_note}）",
                        )
                        return

                    if st in ("learn_failed", "paused") or g.get("empty_close"):
                        self._set_phase(
                            "LEARN_FAILED",
                            f"認識未成立: {det or '指の間にオブジェクトが検出されませんでした'}",
                        )
                        return

                self._set_phase("LEARN_FAILED", "オブジェクト触覚認識がタイムアウトしました。指の間を確認して再実行してください。")
            except asyncio.CancelledError:
                self._set_phase("CANCELLED", "オブジェクト記憶を中断しました")
            except Exception as exc:
                self._set_phase("ERROR", f"触覚記憶エラー: {type(exc).__name__}: {exc}")

        self._abort_flag = False
        self._active_task = asyncio.create_task(_learn_job())
        return {"ok": True, "side": side, "name": clean_name}

    async def start_handover(self, trigger_source: str = "manual_button") -> Dict:
        if self.is_busy:
            return {"ok": False, "error": "すでに動作シーケンスを実行中です"}
        if self.latest_choco is None:
            return {"ok": False, "error": "目の前にターゲット（白キャップ等）が認識されていません"}
        cfg = self.store.cfg
        if cfg.execution_mode != "vision_only" and not self.latest_choco.reachable:
            msg = (
                f"対象（{self.latest_choco.label}）は認識できていますが、腕の最大リーチより少し奥にあります。"
                f"あと約 {self.latest_choco.move_closer_cm:.1f} cm だけ手前（R1側）へ寄せてください（IK残差 {self.latest_choco.ik_error_mm:.1f}mm）。"
            )
            self._set_phase("OUT_OF_REACH", msg)
            return {"ok": False, "error": msg}

        self._abort_flag = False
        self.status.step_history.clear()
        self._active_task = asyncio.create_task(self._run_sequence(trigger_source))
        return {"ok": True, "trigger": trigger_source}

    async def start_fixed_benchmark(
        self,
        x_m: float = 0.36,
        y_m: float = -0.06,
        z_table_m: Optional[float] = None,
        yaw_deg: float = 0.0,
        offset_x_mm: float = 0.0,
        offset_y_mm: float = 0.0,
        length_mm: float = 33.0,
        width_mm: float = 26.0,
        height_mm: float = 14.0,
        side: Optional[str] = None,
    ) -> Dict:
        """
        カメラ認識と切り離し、指定された固定 Torso 座標 [x_m, y_m, z_table_m]・回転角 [yaw_deg]
        （および意図的な外乱オフセット offset_x_mm, offset_y_mm）で表面拘束またぎピンチ単体の成功率を検証する。
        """
        if self.is_busy:
            return {"ok": False, "error": "すでに動作シーケンスを実行中です"}
        cfg = self.store.cfg
        if z_table_m is not None:
            z_tbl = float(z_table_m)
        elif self.latest_choco is not None:
            z_tbl = float(self.latest_choco.table_z_torso_m)
        else:
            z_tbl = -0.035

        tx = float(x_m) + float(offset_x_mm) / 1000.0
        ty = float(y_m) + float(offset_y_mm) / 1000.0
        tz = z_tbl + 0.5 * float(height_mm) / 1000.0

        rob = self.client.latest_telemetry.get("robot") or {}
        q_aux = rob.get("q_aux")
        obs_w = float(q_aux[0]) if (q_aux and len(q_aux) >= 1 and self.client.connected) else float(cfg.look_waist_yaw)
        allowed = (side,) if side in ("left", "right") else None
        plan6 = self.kin.choose_oriented_pinch_plan_6dof(
            np.array([tx, ty, z_tbl + 0.002], dtype=np.float64),
            float(yaw_deg),
            float(length_mm),
            float(width_mm),
            cfg,
            current_waist_yaw=obs_w,
            allowed_sides=allowed,
        )
        rec_hand = str(plan6["side"])
        ik_err_mm = round(float(plan6["pos_err_m"]) * 1000.0, 1)
        in_box = (
            cfg.reach_x_min <= tx <= cfg.reach_x_max
            and abs(ty) <= cfg.reach_y_abs_max
            and cfg.reach_z_min <= tz <= cfg.reach_z_max
        )
        is_reachable = bool(in_box and ik_err_mm <= 18.0)
        if cfg.execution_mode != "vision_only" and not is_reachable:
            msg = f"指定座標 ({tx:.3f}, {ty:.3f}, {z_tbl:.3f})m はリーチ範囲外です (IK残差 {ik_err_mm:.1f}mm)"
            self._set_phase("OUT_OF_REACH", msg)
            return {"ok": False, "error": msg}

        if self.latest_choco is not None and self.latest_choco.table_bbox_torso:
            tbl_bbox = dict(self.latest_choco.table_bbox_torso)
            tbl_bbox["z_top"] = round(z_tbl, 4)
        else:
            tbl_bbox = {
                "x_min": round(max(0.20, tx - 0.14), 4),
                "x_max": round(tx + 0.25, 4),
                "y_min": round(min(-0.28, ty - 0.18), 4),
                "y_max": round(max(0.28, ty + 0.18), 4),
                "z_top": round(z_tbl, 4),
            }

        grasp_span_mm = float(plan6["grasp_span_mm"])
        pre_open = int(np.clip(round((45.0 - grasp_span_mm) * 1.1), 0, 45))
        synth = ChocoShape3D(
            label="fixed_benchmark_choco",
            confidence=1.0,
            bbox_xyxy=[0, 0, 0, 0],
            contour_px=[],
            mask_area_px=0,
            centroid_cam_m=[0.0, 0.0, 0.50],
            centroid_torso_m=[round(tx, 4), round(ty, 4), round(tz, 4)],
            length_mm=round(float(length_mm), 1),
            width_mm=round(float(width_mm), 1),
            height_mm=round(float(height_mm), 1),
            volume_cm3=round(float(length_mm) * float(width_mm) * float(height_mm) / 1000.0, 2),
            yaw_deg=round(float(yaw_deg), 1),
            recommended_hand=rec_hand,
            pre_open_amount=pre_open,
            reachable=is_reachable,
            ik_error_mm=ik_err_mm,
            move_closer_cm=0.0,
            table_z_torso_m=round(z_tbl, 4),
            pinch_yaw_deg=float(plan6["world_pinch_yaw_deg"]),
            achieved_pinch_yaw_deg=float(plan6["world_achieved_yaw_deg"]),
            recommended_waist_yaw=float(plan6["waist_yaw_cmd"]),
            grasp_span_mm=round(grasp_span_mm, 1),
            joint_margin_rad=float(plan6["joint_margin_rad"]),
            thumb_aux=int(plan6["thumb_aux"]),
            xy_source="fixed_benchmark",
            table_bbox_torso=tbl_bbox,
            hd_used=False,
        )
        self._fixed_override_choco = synth
        self._abort_flag = False
        self.status.step_history.clear()
        self._active_task = asyncio.create_task(self._run_sequence("fixed_benchmark"))
        return {
            "ok": True,
            "trigger": "fixed_benchmark",
            "side": rec_hand,
            "target_torso_m": [round(tx, 4), round(ty, 4), round(z_tbl, 4)],
            "achieved_pinch_yaw_deg": synth.achieved_pinch_yaw_deg,
            "grasp_span_mm": synth.grasp_span_mm,
            "ik_error_mm": ik_err_mm,
        }

    async def _sleep_checked(self, seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._abort_flag:
                raise asyncio.CancelledError("Aborted")
            await asyncio.sleep(min(0.1, max(0.01, end - time.monotonic())))

    async def _move_and_wait(
        self,
        q: List[float],
        aux: List[float],
        duration: float,
        label: str,
        replace: bool = False,
        table_z_floor: Optional[float] = None,
    ):
        """r1_arm_daemon へ軌道指令を送り、実機が目標姿勢に到達 (`stage == 'holding'`) するまで安全監視つきで待機する。"""
        await self.client.send_command(
            "arm_move",
            q=q,
            aux=aux,
            duration=duration,
            label=label,
            replace=replace,
            table_z_floor=table_z_floor,
        )
        await self._sleep_checked(0.35)
        deadline = time.monotonic() + max(duration * 3.5, 18.0)
        while time.monotonic() < deadline:
            if self._abort_flag:
                raise asyncio.CancelledError("Aborted")
            rob = self.client.latest_telemetry.get("robot") or {}
            st_lbl = rob.get("stage", "")
            if not rob.get("engaged", True) or st_lbl in ("fault_hold", "damped_estop", "fault_stale_lowstate"):
                fault = rob.get("last_fault") or st_lbl or "安全ガード作動"
                raise RuntimeError(f"安全保護によりアームをソフトホールド停止しました ({fault})")
            if st_lbl == "holding":
                return
            await asyncio.sleep(0.10)

    async def _move_path_and_wait(
        self,
        q_path: List[List[float]],
        aux: List[float],
        duration: float,
        label: str,
        replace: bool = False,
        table_z_floor: Optional[float] = None,
    ):
        """r1_arm_daemon へ 250Hz 直線補間ウェイポイント列 (`move_path`) を送り、到達まで安全監視つきで待機する。"""
        await self.client.send_command(
            "arm_move_path",
            q_path=q_path,
            aux=aux,
            duration=duration,
            label=label,
            replace=replace,
            table_z_floor=table_z_floor,
        )
        await self._sleep_checked(0.35)
        deadline = time.monotonic() + max(duration * 3.5, 18.0)
        while time.monotonic() < deadline:
            if self._abort_flag:
                raise asyncio.CancelledError("Aborted")
            rob = self.client.latest_telemetry.get("robot") or {}
            st_lbl = rob.get("stage", "")
            if not rob.get("engaged", True) or st_lbl in ("fault_hold", "damped_estop", "fault_stale_lowstate"):
                fault = rob.get("last_fault") or st_lbl or "安全ガード作動"
                raise RuntimeError(f"安全保護によりアームをソフトホールド停止しました ({fault})")
            if st_lbl == "holding":
                return
            await asyncio.sleep(0.10)

    async def _wait_hand_grasp_settled(self, side: str, timeout_s: float = 6.5) -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._abort_flag:
                raise asyncio.CancelledError("Aborted")
            hands = self.client.latest_telemetry.get("hands", {})
            g = (hands.get(side) or {}).get("grasp") or {}
            st = g.get("stage", "")
            if st in ("holding", "paused"):
                return st
            await asyncio.sleep(0.12)
        return "timeout"

    def _plan_pinch_waypoints(
        self,
        choco: ChocoShape3D,
        side: str,
        posture: str,
        q_seed_grasp: Optional[List[float]] = None,
        current_waist_yaw: float = 0.0,
        lock_side: bool = False,
    ):
        cfg = self.store.cfg
        is_surface_pinch = (posture == "surface_pinch")
        is_cpinch = (posture == "c_pinch")
        z_table = float(
            getattr(choco, "table_z_torso_m", choco.centroid_torso_m[2] - 0.5 * choco.height_mm / 1000.0)
        )
        h_m = max(0.008, float(choco.height_mm) / 1000.0)
        side_entry = float(cfg.cpinch_side_entry_m) if is_cpinch else 0.0
        waist_yaw_cmd = float(current_waist_yaw)
        grasp_span_mm = float(getattr(choco, "grasp_span_mm", choco.width_mm))

        if is_surface_pinch:
            probe_pt = np.array(
                [choco.centroid_torso_m[0], choco.centroid_torso_m[1], z_table + 0.001], dtype=np.float64
            )
            plan6 = self.kin.choose_oriented_pinch_plan_6dof(
                probe_pt,
                float(choco.yaw_deg),
                float(choco.length_mm),
                float(choco.width_mm),
                cfg,
                current_waist_yaw=current_waist_yaw,
                allowed_sides=((side,) if lock_side else None),
            )
            side = str(plan6["side"])
            waist_yaw_cmd = float(plan6["waist_yaw_cmd"])
            grasp_span_mm = float(plan6["grasp_span_mm"])
            want_yaw = float(plan6["pinch_yaw_deg"])
            target_xy = np.asarray(plan6["target_pos_torso"], dtype=np.float64)
            relief_m = float(getattr(cfg, "surface_relief_lift_m", 0.0008))
            hover_z_m = max(float(getattr(cfg, "straddle_hover_z_m", 0.045)), min(h_m, 0.045) + 0.022)

            grasp_pos = self.kin.clamp_workspace(
                np.array([target_xy[0], target_xy[1], z_table + relief_m], dtype=np.float64),
                side,
                cfg,
            )
            q_grasp, err_grasp, act_yaw, thumb_aux = self.kin.solve_oriented_pinch_ik(
                side, grasp_pos, want_yaw, cfg, q_seed=(q_seed_grasp or list(plan6["q5_grasp"]))
            )
            # side_stage_pos = 対象真上 (+45mm) のまたぎ垂直降下開始ホバー位置
            side_stage_pos = self.kin.clamp_workspace(
                np.array([grasp_pos[0], grasp_pos[1], z_table + hover_z_m], dtype=np.float64),
                side,
                cfg,
            )
            # pre_grasp_pos = HD カメラが対象を遮蔽ゼロで再観測できる斜め手前上空 (+82mm)
            out_y = 1.0 if side == "left" else -1.0
            pre_grasp_pos = self.kin.clamp_workspace(
                np.array([grasp_pos[0] - 0.028, grasp_pos[1] + 0.048 * out_y, z_table + 0.082], dtype=np.float64),
                side,
                cfg,
            )
            lift_pos = self.kin.clamp_workspace(
                grasp_pos + np.array([-0.015, 0.0, 0.115], dtype=np.float64),
                side,
                cfg,
            )
            q_side_stage, _, _, _ = self.kin.solve_oriented_pinch_ik(
                side, side_stage_pos, act_yaw, cfg, q_seed=q_grasp
            )
            q_pre, _, _, _ = self.kin.solve_oriented_pinch_ik(
                side, pre_grasp_pos, act_yaw, cfg, q_seed=q_side_stage
            )
            q_lift, _, _, _ = self.kin.solve_oriented_pinch_ik(
                side, lift_pos, act_yaw, cfg, q_seed=q_grasp
            )
        elif is_cpinch:
            z_pinch = z_table + float(np.clip(0.42 * h_m, 0.0045, 0.014))
            grasp_pos = self.kin.clamp_workspace(
                np.array([choco.centroid_torso_m[0], choco.centroid_torso_m[1], z_pinch], dtype=np.float64),
                side,
                cfg,
            )
            want_yaw = float(getattr(choco, "pinch_yaw_deg", 0.0))
            q_grasp, err_grasp, act_yaw, thumb_aux = self.kin.solve_oriented_pinch_ik(
                side, grasp_pos, want_yaw, cfg, q_seed=q_seed_grasp
            )
            side_vec = self.kin.side_entry_vector_xy(side, act_yaw)
            side_stage_pos = self.kin.clamp_workspace(
                grasp_pos + np.array([side_entry * side_vec[0], side_entry * side_vec[1], 0.0], dtype=np.float64),
                side,
                cfg,
            )
            pre_grasp_pos = self.kin.clamp_workspace(
                side_stage_pos + np.array([-0.012, 0.0, 0.082], dtype=np.float64),
                side,
                cfg,
            )
            lift_pos = self.kin.clamp_workspace(
                grasp_pos + np.array([-0.022, 0.0, 0.120], dtype=np.float64),
                side,
                cfg,
            )
            q_side_stage, _, _, _ = self.kin.solve_oriented_pinch_ik(
                side, side_stage_pos, act_yaw, cfg, q_seed=q_grasp
            )
            q_pre, _, _, _ = self.kin.solve_oriented_pinch_ik(
                side, pre_grasp_pos, act_yaw, cfg, q_seed=q_side_stage
            )
            q_lift, _, _, _ = self.kin.solve_oriented_pinch_ik(
                side, lift_pos, act_yaw, cfg, q_seed=q_grasp
            )
        else:
            out_y = 1.0 if side == "left" else -1.0
            choco_pos = self.kin.clamp_workspace(np.array(choco.centroid_torso_m, dtype=np.float64), side, cfg)
            grasp_pos = self.kin.clamp_workspace(choco_pos + np.array([0.0, 0.0, 0.004]), side, cfg)
            side_stage_pos = self.kin.clamp_workspace(choco_pos + np.array([-0.005, out_y * side_entry, 0.004]), side, cfg)
            pre_grasp_pos = self.kin.clamp_workspace(choco_pos + np.array([-0.020, 0.0, 0.085]), side, cfg)
            lift_pos = self.kin.clamp_workspace(choco_pos + np.array([-0.030, 0.0, 0.125]), side, cfg)
            palm_ori = self.kin.desired_palm_normal_for_posture(side, cfg, posture=posture)
            q_pre, _ = self.kin.solve_arm_ik(side, pre_grasp_pos, cfg, desired_palm_normal=palm_ori, posture=posture)
            q_side_stage, _ = self.kin.solve_arm_ik(side, side_stage_pos, cfg, desired_palm_normal=palm_ori, q_seed=q_pre, posture=posture)
            q_grasp, err_grasp = self.kin.solve_arm_ik(side, grasp_pos, cfg, desired_palm_normal=palm_ori, q_seed=q_side_stage, posture=posture)
            q_lift, _ = self.kin.solve_arm_ik(side, lift_pos, cfg, desired_palm_normal=palm_ori, q_seed=q_grasp, posture=posture)
            act_yaw = float(getattr(choco, "achieved_pinch_yaw_deg", 0.0))
            thumb_aux = 485

        return (
            side,
            waist_yaw_cmd,
            grasp_span_mm,
            z_table,
            side_entry,
            pre_grasp_pos,
            side_stage_pos,
            grasp_pos,
            lift_pos,
            q_pre,
            q_side_stage,
            q_grasp,
            q_lift,
            err_grasp,
            act_yaw,
            thumb_aux,
        )

    async def _run_sequence(self, trigger_source: str):
        cfg = self.store.cfg
        if trigger_source == "fixed_benchmark" and self._fixed_override_choco is not None:
            choco = self._fixed_override_choco
            self._fixed_override_choco = None
        else:
            choco = self.latest_choco
        assert choco is not None

        posture = (cfg.grasp_posture or "surface_pinch").strip()
        is_surface_pinch = (posture == "surface_pinch")
        is_cpinch = (posture == "c_pinch")

        rob0 = self.client.latest_telemetry.get("robot") or {}
        q_aux0 = rob0.get("q_aux")
        obs_waist_yaw = (
            float(q_aux0[0])
            if (q_aux0 and len(q_aux0) >= 1 and self.client.connected)
            else float(cfg.look_waist_yaw)
        )

        (
            side,
            waist_yaw_cmd,
            grasp_span_mm,
            z_table,
            side_entry,
            pre_grasp_pos,
            side_stage_pos,
            grasp_pos,
            lift_pos,
            q_pre,
            q_side_stage,
            q_grasp,
            q_lift,
            err_grasp,
            act_yaw,
            thumb_aux,
        ) = self._plan_pinch_waypoints(
            choco,
            choco.recommended_hand,
            posture,
            current_waist_yaw=obs_waist_yaw,
            lock_side=(trigger_source == "fixed_benchmark"),
        )
        side_ja = "左アーム＋左ハンド" if side == "left" else "右アーム＋右ハンド"
        self.status.selected_hand = side
        self.status.locked_choco = asdict(choco)
        self.status.planned_ik_error_mm = round(err_grasp * 1000.0, 1)
        palm_ori = self.kin.desired_palm_normal_for_posture(side, cfg, posture=posture)

        table_bbox = dict(choco.table_bbox_torso) if getattr(choco, "table_bbox_torso", None) else None
        if table_bbox:
            self._last_table_bbox = table_bbox

        # 受け渡し先: 検出済みの両掌 3D 中心 (無ければ安全前方デフォルト位置 [0.36, ±0.04, 0.14])
        if self.latest_gesture and self.latest_gesture.palm_center_torso_m is not None:
            palm_raw = np.array(self.latest_gesture.palm_center_torso_m, dtype=np.float64)
        else:
            palm_raw = np.array([0.36, 0.04 if side == "left" else -0.04, 0.14], dtype=np.float64)
        palm_target = self.kin.clamp_workspace(palm_raw, side, cfg)
        self.status.target_palms_torso_m = [round(float(v), 4) for v in palm_target]

        recorder = FlightRecorderSession(
            tag=f"{posture}_{trigger_source}",
            meta={
                "trigger_source": trigger_source,
                "posture": posture,
                "execution_mode": cfg.execution_mode,
                "side": side,
                "waist_yaw_cmd": round(waist_yaw_cmd, 4),
                "grav_ff_scale": float(cfg.grav_ff_scale),
                "locked_choco": asdict(choco),
            },
            telemetry_getter=lambda: self.client.latest_telemetry,
            phase_getter=lambda: self.status.phase,
        )
        recorder.save_keyframe(
            "01_precheck",
            stereo_pair=self.client.latest_stereo,
            hd_tuple=self.client.latest_hd,
            fallback_left_bgr=self.latest_left_bgr,
        )
        outcome = {
            "trigger_source": trigger_source,
            "posture": posture,
            "side": side,
            "choco_label": choco.label,
            "choco_yaw_deg": float(choco.yaw_deg),
            "achieved_pinch_yaw_deg": float(act_yaw),
            "grasp_span_mm": round(float(grasp_span_mm), 1),
            "xy_source": getattr(choco, "xy_source", "unknown"),
            "touchdown_source": None,
            "z_table_est_m": round(float(z_table), 4),
            "z_surface_meas_m": None,
            "grasp_verified": False,
            "failure_mode": None,
            "dry_run": False,
        }

        try:
            self.status.visual_servo = None
            if self.latest_left_bgr is not None:
                self.visual_servo.capture_clean_reference(self.latest_left_bgr)

            if is_surface_pinch:
                posture_ja = f"表面拘束またぎ2指ピンチ(把持角{act_yaw:+.0f}°, 対象辺{grasp_span_mm:.0f}mm)"
            elif is_cpinch and cfg.two_finger_pinch:
                posture_ja = f"2指C字ピンチ(把持角{act_yaw:+.0f}°)"
            elif is_cpinch:
                posture_ja = "C字サイドピンチ"
            else:
                posture_ja = "真上アプローチ"

            self._set_phase(
                "LOCK_AND_PLAN",
                f"発火({trigger_source}): 対象 {choco.label} ({choco.length_mm:.0f}×{choco.width_mm:.0f}×{choco.height_mm:.0f}mm, 主軸{choco.yaw_deg:+.0f}°) / {side_ja} [{posture_ja}] (IK誤差 {self.status.planned_ik_error_mm}mm)",
            )

            # Dry-Run (Vision Only) モードなら実機モーターを動かさず軌道計画検証のみ完了する
            if cfg.execution_mode not in ("grasp_and_hold", "pick_only", "full_handover") or not self.client.connected:
                outcome["dry_run"] = True
                await self._sleep_checked(1.0)
                self._set_phase(
                    "DRY_RUN_DONE",
                    f"[Dry-Run完了] {side_ja} ({posture_ja}) で {grasp_pos.round(3).tolist()}m を優しさ把持 ({cfg.gentle_force_N:.2f}N) 計画成功！（実機動作はモード切替で実行）",
                )
                return

            recorder.start()

            rob = self.client.latest_telemetry.get("robot", {})
            soc = rob.get("bms", {}).get("soc", 100)
            if soc <= 10:
                outcome["failure_mode"] = "low_battery"
                self._set_phase("ABORT_LOW_BAT", f"バッテリー残量 {soc}% のため安全停止しました。充電後に再実行してください。")
                return

            cur_q10 = list(rob.get("q_arm") or list(REST_ARM_POSE))
            initial_q10 = list(cur_q10)
            self._recorded_initial_q10 = list(initial_q10)
            cur_aux = list(rob.get("q_aux") or [cfg.look_waist_yaw, cfg.look_head_pitch, cfg.look_head_yaw])
            cmd_aux = [waist_yaw_cmd, float(cur_aux[1]), float(cur_aux[2])]

            # Step 1: ハンドを表面拘束またぎ開き (~66mm) / 記憶プロファイルに合わせて事前準備 & 重力補償つきエンゲージ
            active_prof = self.profile_store.get_active_for_grasp()
            mode_desc = "宙吊り脚0Nm＋重力補償モード" if cfg.use_suspended_mode else "立位重力補償安全制御"
            if (
                active_prof
                and len(active_prof.get("pre_open_positions", [])) == 6
                and not ((is_surface_pinch or is_cpinch) and cfg.two_finger_pinch)
            ):
                self._set_phase(
                    "PREPARE_HAND_AND_ARM",
                    f"{side_ja} を記憶『{active_prof['name']}』用の開き角度へ準備し {mode_desc} を開始",
                )
                await self.client.send_command(
                    "hand_pose", side=side, targets=active_prof["pre_open_positions"], duration_ms=900
                )
            else:
                open_posture = "surface_pinch" if is_surface_pinch else ("c_pinch" if is_cpinch else "flat")
                self._set_phase(
                    "PREPARE_HAND_AND_ARM",
                    f"{side_ja} を{posture_ja}形状 (thumb_aux={thumb_aux}) へ事前オープンし {mode_desc} を開始",
                )
                await self.client.send_command(
                    "hand_open",
                    side=side,
                    pre_open_amount=choco.pre_open_amount,
                    duration_ms=900,
                    posture=open_posture,
                    two_finger_only=(cfg.two_finger_pinch if (is_surface_pinch or is_cpinch) else False),
                    thumb_aux=thumb_aux,
                )

            table_z_floor = float(max(-0.16, min(0.12, z_table - 0.036)))
            r_eng = await self.client.send_command(
                "arm_engage",
                suspended=(1.0 if cfg.use_suspended_mode else 0.0),
                table_z_floor=table_z_floor,
                grav_ff_scale=float(cfg.grav_ff_scale),
                extra_hand_mass_kg=float(cfg.extra_hand_mass_kg),
            )
            if not r_eng.get("ok"):
                outcome["failure_mode"] = "engage_cmd_failed"
                self._set_phase("ERROR", f"アーム制御開始失敗: {r_eng.get('error')}")
                return

            # ReleaseMode完了＋2.5秒kpソフトランプアップ完了 (stage == 'holding') を確実に待つ
            eng_deadline = time.monotonic() + 9.5
            saw_engaged = False
            while time.monotonic() < eng_deadline:
                await self._sleep_checked(0.15)
                rob_now = self.client.latest_telemetry.get("robot") or {}
                if rob_now.get("engaged"):
                    saw_engaged = True
                    if rob_now.get("stage") == "holding":
                        break
            if not saw_engaged:
                err_f = (self.client.latest_telemetry.get("robot") or {}).get("last_fault") or "engage未完了"
                outcome["failure_mode"] = "engage_timeout"
                self._set_phase("ERROR", f"安全制御エンゲージ待機タイムアウト ({err_f})")
                return

            # 既にエンゲージ済みだった場合にも最新の table_z_floor / grav_ff_scale を確実に反映する
            await self.client.send_command(
                "arm_set_params",
                table_z_floor=table_z_floor,
                grav_ff_scale=float(cfg.grav_ff_scale),
                extra_hand_mass_kg=float(cfg.extra_hand_mass_kg),
            )

            # Step 2: 3D天板バウンディングボックス干渉チェック済みの3段階サイドリフト → 視界クリアな斜め上空へ移動
            wp0, wp1, wp2 = self.kin.build_safe_table_waypoints(cur_q10, side, table_bbox=table_bbox)
            q10_pre = self.kin.merge_single_arm(wp2, side, q_pre)
            q10_side_stage = self.kin.merge_single_arm(q10_pre, side, q_side_stage)
            q10_grasp = self.kin.merge_single_arm(q10_side_stage, side, q_grasp)

            self._set_phase("APPROACH_ABOVE_CHOCO", f"天板3D包絡との干渉ゼロ経路で対象斜め上空へ移動中 ({posture_ja})")
            await self._move_and_wait(wp0, cur_aux, 2.2, "side_clear", replace=True)
            await self._move_and_wait(wp1, cur_aux, 3.0, "lift_behind_desk")
            await self._move_and_wait(wp2, cmd_aux, 2.6, "swing_over_desk")
            await self._move_and_wait(q10_pre, cmd_aux, 2.1, "above_choco")

            # Step 2.5: 斜め上空(カメラ死角ゼロ)からの HD 光線×3D平面交差 閉ループ再ロック (固定座標ベンチマーク時はスキップ)
            if cfg.enable_visual_servo and trigger_source != "fixed_benchmark":
                if is_surface_pinch or is_cpinch:
                    self._set_phase("VISUAL_SERVO_ALIGN", "視界クリア位置から最新 HD光線×3D平面交差で位置・向きを再ロック中")
                    await self._sleep_checked(0.45)
                    live_ch = self.latest_choco
                    if (
                        live_ch is not None
                        and live_ch.reachable
                        and live_ch.label == choco.label
                        and (live_ch.hd_used or getattr(live_ch, "xy_source", "") in ("hd_plane_ray", "stereo_plane_ray"))
                        and abs(float(live_ch.length_mm) - float(choco.length_mm)) <= 7.5
                        and abs(float(live_ch.width_mm) - float(choco.width_mm)) <= 7.5
                        and abs(float(live_ch.table_z_torso_m) - float(choco.table_z_torso_m)) <= 0.015
                        and float(
                            np.hypot(
                                live_ch.centroid_torso_m[0] - choco.centroid_torso_m[0],
                                live_ch.centroid_torso_m[1] - choco.centroid_torso_m[1],
                            )
                        )
                        <= 0.032
                    ):
                        # 天面高さ table_z_torso_m・重心Z・高さ・天板BBoxは無遮蔽の初期ロック値を厳守し、XYと回転角のみ反映する
                        locked_z_tbl = float(choco.table_z_torso_m)
                        locked_cz = float(choco.centroid_torso_m[2])
                        locked_h_mm = float(choco.height_mm)
                        dx_rel = float(live_ch.centroid_torso_m[0] - choco.centroid_torso_m[0])
                        dy_rel = float(live_ch.centroid_torso_m[1] - choco.centroid_torso_m[1])
                        live_ch.table_z_torso_m = locked_z_tbl
                        live_ch.centroid_torso_m = [
                            float(live_ch.centroid_torso_m[0]),
                            float(live_ch.centroid_torso_m[1]),
                            locked_cz,
                        ]
                        live_ch.height_mm = locked_h_mm
                        if table_bbox:
                            live_ch.table_bbox_torso = table_bbox
                        prev_aux = thumb_aux
                        (
                            side,
                            _,
                            grasp_span_mm,
                            z_table,
                            side_entry,
                            pre_grasp_pos,
                            side_stage_pos,
                            grasp_pos,
                            lift_pos,
                            q_pre,
                            q_side_stage,
                            q_grasp,
                            q_lift,
                            err_grasp,
                            act_yaw,
                            thumb_aux,
                        ) = self._plan_pinch_waypoints(
                            live_ch,
                            side,
                            posture,
                            q_seed_grasp=q_grasp,
                            current_waist_yaw=waist_yaw_cmd,
                            lock_side=True,
                        )
                        choco = live_ch
                        outcome["choco_yaw_deg"] = float(choco.yaw_deg)
                        outcome["achieved_pinch_yaw_deg"] = float(act_yaw)
                        outcome["grasp_span_mm"] = round(float(grasp_span_mm), 1)
                        self.status.locked_choco = asdict(choco)
                        self.status.planned_ik_error_mm = round(err_grasp * 1000.0, 1)
                        self.status.visual_servo = {
                            "valid": True,
                            "delta_xy_torso_m": [round(dx_rel, 4), round(dy_rel, 4)],
                            "reason": f"HD平面光線再ロック(Z固定) yaw={choco.yaw_deg:+.0f}°->pinch={act_yaw:+.0f}°",
                        }
                        recorder.log_event(
                            "visual_relock",
                            dx_mm=round(dx_rel * 1000.0, 2),
                            dy_mm=round(dy_rel * 1000.0, 2),
                            act_yaw=act_yaw,
                            grasp_span_mm=grasp_span_mm,
                        )
                        q10_pre = self.kin.merge_single_arm(wp2, side, q_pre)
                        q10_side_stage = self.kin.merge_single_arm(q10_pre, side, q_side_stage)
                        q10_grasp = self.kin.merge_single_arm(q10_side_stage, side, q_grasp)
                        if abs(thumb_aux - prev_aux) >= 4:
                            await self.client.send_command(
                                "hand_open",
                                side=side,
                                pre_open_amount=choco.pre_open_amount,
                                duration_ms=450,
                                posture=("surface_pinch" if is_surface_pinch else "c_pinch"),
                                two_finger_only=cfg.two_finger_pinch,
                                thumb_aux=thumb_aux,
                            )
                        if not is_surface_pinch and float(np.hypot(dx_rel, dy_rel) * 1000.0) >= 2.5:
                            await self._move_and_wait(q10_pre, cmd_aux, 1.2, "relock_pre_cpinch")
                else:
                    self._set_phase("VISUAL_SERVO_ALIGN", "上空ホバリングからハンド指先と対象の相対位置をカメラ計測中")
                    await self._sleep_checked(0.55)
                    valid_obs = []
                    for _ in range(3):
                        if self.latest_left_bgr is not None and self.latest_stereo is not None:
                            hp, hy = self.latest_head_angles
                            obs = self.visual_servo.observe_pinch_alignment(
                                self.latest_left_bgr,
                                self.latest_stereo,
                                side,
                                pre_grasp_pos,
                                self.kin,
                                hp,
                                hy,
                                cfg,
                            )
                            if obs.valid and obs.delta_xy_torso_m is not None:
                                valid_obs.append(obs)
                        await self._sleep_checked(0.14)

                    if valid_obs:
                        med_dx = float(np.clip(np.median([o.delta_xy_torso_m[0] for o in valid_obs]), -0.045, 0.045))
                        med_dy = float(np.clip(np.median([o.delta_xy_torso_m[1] for o in valid_obs]), -0.045, 0.045))
                        corr_mm = float(np.hypot(med_dx, med_dy) * 1000.0)
                        self.status.visual_servo = asdict(valid_obs[-1])
                        if corr_mm >= 4.0:
                            shift = np.array([med_dx, med_dy, 0.0], dtype=np.float64)
                            pre_grasp_pos = self.kin.clamp_workspace(pre_grasp_pos + shift, side, cfg)
                            side_stage_pos = self.kin.clamp_workspace(side_stage_pos + shift, side, cfg)
                            grasp_pos = self.kin.clamp_workspace(grasp_pos + shift, side, cfg)
                            lift_pos = self.kin.clamp_workspace(lift_pos + shift, side, cfg)
                            q_pre, _ = self.kin.solve_arm_ik(side, pre_grasp_pos, cfg, desired_palm_normal=palm_ori, q_seed=q_pre, posture=posture)
                            q_side_stage, _ = self.kin.solve_arm_ik(side, side_stage_pos, cfg, desired_palm_normal=palm_ori, q_seed=q_pre, posture=posture)
                            q_grasp, err_grasp = self.kin.solve_arm_ik(side, grasp_pos, cfg, desired_palm_normal=palm_ori, q_seed=q_side_stage, posture=posture)
                            q_lift, _ = self.kin.solve_arm_ik(side, lift_pos, cfg, desired_palm_normal=palm_ori, q_seed=q_grasp, posture=posture)
                            self.status.planned_ik_error_mm = round(err_grasp * 1000.0, 1)
                            q10_pre = self.kin.merge_single_arm(wp2, side, q_pre)
                            q10_side_stage = self.kin.merge_single_arm(q10_pre, side, q_side_stage)
                            q10_grasp = self.kin.merge_single_arm(q10_side_stage, side, q_grasp)
                            self._set_phase(
                                "VISUAL_SERVO_ALIGN",
                                f"指先ピンチ誤差 ΔX={med_dx*1000:+.0f}mm, ΔY={med_dy*1000:+.0f}mm を検出 → 微補正中",
                            )
                            await self._move_and_wait(q10_pre, cmd_aux, 1.6, "visual_servo_align")

            # Step 3: アプローチ方式分岐
            #   [A] surface_pinch (最優先推奨): 対象真上 (+45mm) で ~66mm 開口またぎ → 手先直線補間で真下へ微速下降
            #       → (触覚 / 外乱トルク tau_ext / 着地ストール) で机面確定 → +0.8mm 微小除荷 → 机面ガイド自己調心ピンチ
            if is_surface_pinch:
                tcp_off = self.kin.active_tcp_offset(cfg, posture="surface_pinch")
                hover_mm = (float(side_stage_pos[2]) - z_table) * 1000.0
                self._set_phase(
                    "STRADDLE_HOVER",
                    f"対象真上 +{hover_mm:.0f}mm へ移動し、約66mm開口で対象を跨いで触覚・外乱トルク基準値を取得中 (把持角 {act_yaw:+.0f}°)",
                )
                await self._move_and_wait(q10_side_stage, cmd_aux, 1.8, "straddle_hover")
                await self._sleep_checked(0.35)
                recorder.save_keyframe(
                    "02_straddle_hover",
                    stereo_pair=self.client.latest_stereo,
                    hd_tuple=self.client.latest_hd,
                    fallback_left_bgr=self.latest_left_bgr,
                )

                # 空中静止時の (1) Revo2 5指 [normal_N, tangent_N] 基準値 と (2) アーム外乱トルク tau_ext 基準値を同時計測
                norm_s, tan_s, text_s = [], [], []
                for _ in range(5):
                    hands_t = self.client.latest_telemetry.get("hands") or {}
                    f_list = (((hands_t.get(side) or {}).get("tactile") or {}).get("fingers")) or []
                    if len(f_list) == 5:
                        norm_s.append([float(f.get("normal_N", 0.0)) for f in f_list])
                        tan_s.append([float(f.get("tangent_N", 0.0)) for f in f_list])
                    rob_t = self.client.latest_telemetry.get("robot") or {}
                    te_list = rob_t.get("tau_ext")
                    if te_list and len(te_list) == 10:
                        text_s.append([float(x) for x in te_list])
                    await self._sleep_checked(0.05)

                norm_base = np.median(norm_s, axis=0).tolist() if norm_s else [0.0] * 5
                tan_base = np.median(tan_s, axis=0).tolist() if tan_s else [0.0] * 5
                tau_ext_base = np.median(text_s, axis=0).tolist() if text_s else [0.0] * 10
                arm_base_idx = 0 if side == "left" else 5
                tau_ext_thresh = float(getattr(cfg, "tau_ext_contact_nm", 0.68))

                # 対象真上から z_table - 22mm まで 2.5mm 刻みの完全垂直デカルト直線 IK パスを生成して ~12.4mm/s で降下
                probe_bot_pos = self.kin.clamp_workspace(
                    np.array([grasp_pos[0], grasp_pos[1], z_table - 0.022], dtype=np.float64),
                    side,
                    cfg,
                )
                q10_descend_path, _, max_line_dev = self.kin.plan_cartesian_ik_path(
                    side,
                    side_stage_pos,
                    probe_bot_pos,
                    act_yaw,
                    cfg,
                    q10_side_stage,
                    q_seed=q_side_stage,
                    step_m=0.0025,
                )
                self._set_phase(
                    "TOUCHOFF_STRADDLE",
                    f"対象を跨いだまま真下へ直線微速下降中 (直線偏差 {max_line_dev*1000:.2f}mm / 触覚・外乱トルク・着地監視)...",
                )
                await self.client.send_command(
                    "arm_move_path",
                    q_path=q10_descend_path,
                    aux=cmd_aux,
                    duration=5.4,
                    label="touchoff_straddle",
                    replace=True,
                    table_z_floor=table_z_floor,
                )

                t_probe0 = time.monotonic()
                z_surface: Optional[float] = None
                touchdown_source = "fallback_bottom"
                touch_info = ""
                z_hist: List[tuple[float, float]] = []
                q5_now = list(q_side_stage)
                cur_z = float(side_stage_pos[2])
                seen_touchoff = False
                z_gate_hi = min(float(side_stage_pos[2]) - 0.014, z_table + 0.036)

                while time.monotonic() - t_probe0 < 15.0:
                    await self._sleep_checked(0.04)
                    el_p = time.monotonic() - t_probe0
                    rob_p = self.client.latest_telemetry.get("robot") or {}
                    st_p = rob_p.get("stage", "")
                    if st_p == "touchoff_straddle":
                        seen_touchoff = True
                    if not rob_p.get("engaged", True) or st_p in ("fault_hold", "damped_estop", "fault_stale_lowstate"):
                        fault = rob_p.get("last_fault") or st_p or "安全ガード作動"
                        raise RuntimeError(f"安全保護によりアームをソフトホールド停止しました ({fault})")

                    q_arm_now = rob_p.get("q_arm") or q10_side_stage
                    q5_now = list(q_arm_now[:5] if side == "left" else q_arm_now[5:10])
                    cur_tcp, _ = self.kin.arm_fk(side, q5_now, tcp_off)
                    cur_z = float(cur_tcp[2])
                    z_hist.append((el_p, cur_z))

                    # (A) 指先触覚センサー検知 (親指0・人差指1・中指2は高感度 0.045N、他指も安全のため 0.065N で監視)
                    hands_p = self.client.latest_telemetry.get("hands") or {}
                    fl_p = (((hands_p.get(side) or {}).get("tactile") or {}).get("fingers")) or []
                    dn_pinch = dtan_pinch = dn_all = 0.0
                    pinch_ids = (0, 1) if cfg.two_finger_pinch else (0, 1, 2)
                    if len(fl_p) == 5:
                        dn_pinch = max(abs(float(fl_p[i].get("normal_N", 0.0)) - norm_base[i]) for i in pinch_ids)
                        dtan_pinch = max(abs(float(fl_p[i].get("tangent_N", 0.0)) - tan_base[i]) for i in pinch_ids)
                        dn_all = max(abs(float(fl_p[i].get("normal_N", 0.0)) - norm_base[i]) for i in range(5))
                    tactile_hit = (
                        el_p >= 0.35
                        and cur_z <= z_gate_hi
                        and (dn_pinch >= 0.045 or dtan_pinch >= 0.055 or dn_all >= 0.065)
                    )

                    # (B) 外乱トルクオブザーバ検知 (|tau_ext - tau_ext_base| が肩ピッチまたは肘で閾値到達)
                    te_now = rob_p.get("tau_ext")
                    dtau_active = 0.0
                    if te_now and len(te_now) == 10 and text_s:
                        dtau_sp = abs(float(te_now[arm_base_idx + 0]) - tau_ext_base[arm_base_idx + 0])
                        dtau_el = abs(float(te_now[arm_base_idx + 3]) - tau_ext_base[arm_base_idx + 3])
                        dtau_active = max(dtau_sp, dtau_el)
                    tau_ext_hit = (
                        el_p >= 0.45
                        and cur_z <= z_gate_hi
                        and dtau_active >= tau_ext_thresh
                    )

                    # (C) エンコーダ Z 降下ストール検知
                    stall_hit = False
                    if el_p >= 1.10 and st_p == "touchoff_straddle" and cur_z <= z_gate_hi:
                        older = [z for t, z in z_hist if (el_p - t) >= 0.45]
                        if older and (older[-1] - cur_z) < 0.0008:
                            stall_hit = True

                    if tactile_hit or tau_ext_hit or stall_hit:
                        z_surface = cur_z
                        if tactile_hit:
                            touchdown_source = "tactile"
                            trig_ja = "指先触覚"
                        elif tau_ext_hit:
                            touchdown_source = "tau_ext"
                            trig_ja = "外乱トルク"
                        else:
                            touchdown_source = "encoder_stall"
                            trig_ja = "着地ストール"
                        touch_info = (
                            f"{trig_ja} Z={z_surface*1000:.1f}mm "
                            f"(ΔFn={dn_pinch:.2f}N, Δτ={dtau_active:.2f}Nm)"
                        )
                        break

                    if (seen_touchoff and st_p == "holding") or (el_p >= 3.5 and st_p == "holding"):
                        break

                # +0.8mm 微小除荷を最優先で即時送信し、画像保存中の天板への押し込みをゼロにする
                relief_m = float(getattr(cfg, "surface_relief_lift_m", 0.0008))
                rob_g = self.client.latest_telemetry.get("robot") or {}
                pd_trim = 0.0007 if float(rob_g.get("grav_ff_gain") or 0.0) > 0.3 else 0.0020
                z_ref_surf = z_surface if z_surface is not None else cur_z
                z_goal_relief = z_ref_surf + relief_m
                z_cmd_relief = z_goal_relief + pd_trim

                grasp_pos = self.kin.clamp_workspace(
                    np.array([grasp_pos[0], grasp_pos[1], z_cmd_relief], dtype=np.float64), side, cfg
                )
                q_grasp, err_grasp, _, _ = self.kin.solve_oriented_pinch_ik(
                    side, grasp_pos, act_yaw, cfg, q_seed=q5_now
                )
                self.status.planned_ik_error_mm = round(err_grasp * 1000.0, 1)
                q10_grasp = self.kin.merge_single_arm(q10_side_stage, side, q_grasp)
                touch_hdr = f"着地確定[{touch_info}]" if touch_info else f"下限到達(Z={z_ref_surf*1000:.1f}mm)"
                self._set_phase(
                    "SURFACE_MICRO_RELIEF",
                    f"{touch_hdr} → +{relief_m*1000:.1f}mm 微小除荷し机面ガイド姿勢を保持中",
                )
                await self._move_and_wait(q10_grasp, cmd_aux, 0.75, "surface_micro_relief", replace=True)

                outcome["touchdown_source"] = touchdown_source
                outcome["z_surface_meas_m"] = round(float(z_ref_surf), 4)
                recorder.log_event(
                    "touchdown",
                    source=touchdown_source,
                    z_surface_m=outcome["z_surface_meas_m"],
                    z_table_est_m=round(z_table, 4),
                )
                recorder.save_keyframe(
                    "03_touchdown",
                    stereo_pair=self.client.latest_stereo,
                    hd_tuple=self.client.latest_hd,
                    fallback_left_bgr=self.latest_left_bgr,
                )
                await self._sleep_checked(0.15)

                # 万一押し付け残り (<-0.4mm) または浮きすぎ (>+2.4mm) があれば 1 回だけ閉ループ微調整する
                rob_chk = self.client.latest_telemetry.get("robot") or {}
                q_arm_chk = rob_chk.get("q_arm")
                if q_arm_chk and len(q_arm_chk) == 10:
                    q5_chk = q_arm_chk[:5] if side == "left" else q_arm_chk[5:10]
                    chk_tcp, _ = self.kin.arm_fk(side, q5_chk, tcp_off)
                    rel_h = float(chk_tcp[2]) - z_ref_surf
                    if rel_h < -0.0004 or rel_h > 0.0024:
                        err_z = z_goal_relief - float(chk_tcp[2])
                        z_cmd_relief = float(np.clip(z_cmd_relief + err_z, z_ref_surf - 0.001, z_ref_surf + 0.008))
                        grasp_pos = self.kin.clamp_workspace(
                            np.array([grasp_pos[0], grasp_pos[1], z_cmd_relief], dtype=np.float64), side, cfg
                        )
                        q_grasp, _, _, _ = self.kin.solve_oriented_pinch_ik(
                            side, grasp_pos, act_yaw, cfg, q_seed=q_grasp
                        )
                        q10_grasp = self.kin.merge_single_arm(q10_side_stage, side, q_grasp)
                        await self._move_and_wait(q10_grasp, cmd_aux, 0.65, "trim_micro_relief", replace=True)
                        await self._sleep_checked(0.12)

            #   [B] 旧 c_pinch (互換維持): 対象から44mm横の天面タッチオフ → +2.0mm 浮上 → 水平スライド
            elif is_cpinch and side_entry >= 0.004:
                tcp_off = self.kin.active_tcp_offset(cfg, posture="c_pinch")
                side_air_pos = self.kin.clamp_workspace(
                    np.array([side_stage_pos[0], side_stage_pos[1], z_table + 0.036], dtype=np.float64),
                    side,
                    cfg,
                )
                q_side_air, _, _, _ = self.kin.solve_oriented_pinch_ik(
                    side, side_air_pos, act_yaw, cfg, q_seed=q_pre
                )
                q10_side_air = self.kin.merge_single_arm(q10_pre, side, q_side_air)
                self._set_phase("DESCEND_TO_CHOCO", f"対象脇({side_entry*1000:.0f}mm横) +36mm の空中へ移動し、指先触覚のゼロ点を取得中 (把持角 {act_yaw:+.0f}°)")
                await self._move_and_wait(q10_side_air, cmd_aux, 2.0, "descend_side_air")
                await self._sleep_checked(0.35)

                norm_s, tan_s = [], []
                for _ in range(5):
                    hands_t = self.client.latest_telemetry.get("hands") or {}
                    f_list = (((hands_t.get(side) or {}).get("tactile") or {}).get("fingers")) or []
                    if len(f_list) == 5:
                        norm_s.append([float(f.get("normal_N", 0.0)) for f in f_list])
                        tan_s.append([float(f.get("tangent_N", 0.0)) for f in f_list])
                    await self._sleep_checked(0.05)

                norm_base = np.median(norm_s, axis=0).tolist() if norm_s else [0.0] * 5
                tan_base = np.median(tan_s, axis=0).tolist() if tan_s else [0.0] * 5

                probe_bot_pos = self.kin.clamp_workspace(
                    np.array([side_stage_pos[0], side_stage_pos[1], z_table - 0.020], dtype=np.float64),
                    side,
                    cfg,
                )
                q_probe_bot, _, _, _ = self.kin.solve_oriented_pinch_ik(
                    side, probe_bot_pos, act_yaw, cfg, q_seed=q_side_air
                )
                q10_probe_bot = self.kin.merge_single_arm(q10_side_air, side, q_probe_bot)
                self._set_phase(
                    "TOUCHOFF_SURFACE",
                    "親指・人差指の触覚センサー＋エンコーダ着地検知で天面を微速探索中...",
                )
                await self.client.send_command(
                    "arm_move", q=q10_probe_bot, aux=cmd_aux, duration=5.2, label="touchoff_probe", replace=True
                )

                t_probe0 = time.monotonic()
                z_surface = None
                touch_info = ""
                z_hist = []
                while time.monotonic() - t_probe0 < 6.2:
                    await self._sleep_checked(0.04)
                    el_p = time.monotonic() - t_probe0
                    rob_p = self.client.latest_telemetry.get("robot") or {}
                    st_p = rob_p.get("stage", "")
                    if not rob_p.get("engaged", True) or st_p in ("fault_hold", "damped_estop", "fault_stale_lowstate"):
                        fault = rob_p.get("last_fault") or st_p or "安全ガード作動"
                        raise RuntimeError(f"安全保護によりアームをソフトホールド停止しました ({fault})")

                    q_arm_now = rob_p.get("q_arm") or q10_side_air
                    q5_now = q_arm_now[:5] if side == "left" else q_arm_now[5:10]
                    cur_tcp, _ = self.kin.arm_fk(side, q5_now, tcp_off)
                    cur_z = float(cur_tcp[2])
                    z_hist.append((el_p, cur_z))

                    hands_p = self.client.latest_telemetry.get("hands") or {}
                    fl_p = (((hands_p.get(side) or {}).get("tactile") or {}).get("fingers")) or []
                    dn_max = dtan_max = 0.0
                    if len(fl_p) == 5:
                        dn_max = max(abs(float(fl_p[i].get("normal_N", 0.0)) - norm_base[i]) for i in range(5))
                        dtan_max = max(abs(float(fl_p[i].get("tangent_N", 0.0)) - tan_base[i]) for i in range(5))

                    tactile_hit = (
                        el_p >= 0.35
                        and cur_z <= (z_table + 0.026)
                        and (dn_max >= 0.045 or dtan_max >= 0.055)
                    )
                    stall_hit = False
                    if 0.90 <= el_p <= 4.75 and st_p == "touchoff_probe" and cur_z <= (z_table + 0.024):
                        older = [z for t, z in z_hist if (el_p - t) >= 0.42]
                        if older and (older[-1] - cur_z) < 0.0008:
                            stall_hit = True

                    if tactile_hit or stall_hit:
                        z_surface = cur_z
                        outcome["touchdown_source"] = "tactile" if tactile_hit else "encoder_stall"
                        outcome["z_surface_meas_m"] = round(float(z_surface), 4)
                        trig_src = "触覚" if tactile_hit else "着地停止"
                        touch_info = f"{trig_src} Z={z_surface*1000:.1f}mm (ΔFn={dn_max:.2f}N, ΔFt={dtan_max:.2f}N)"
                        break

                    if el_p >= 1.8 and st_p == "holding":
                        break

                lift_m = float(getattr(cfg, "cpinch_surface_lift_m", 0.0020))
                if z_surface is None:
                    z_goal_actual = z_table + 0.0030
                    touch_hdr = f"推定天面({z_table*1000:.1f}mm)"
                else:
                    z_goal_actual = z_surface + lift_m
                    touch_hdr = f"天面検知[{touch_info}]"

                rob_g = self.client.latest_telemetry.get("robot") or {}
                pd_trim_c = 0.0010 if float(rob_g.get("grav_ff_gain") or 0.0) > 0.3 else 0.0032
                z_cmd_clear = z_goal_actual + pd_trim_c
                side_stage_pos = self.kin.clamp_workspace(
                    np.array([side_stage_pos[0], side_stage_pos[1], z_cmd_clear], dtype=np.float64), side, cfg
                )
                q_side_stage, _, _, _ = self.kin.solve_oriented_pinch_ik(
                    side, side_stage_pos, act_yaw, cfg, q_seed=q_side_air
                )
                q10_side_stage = self.kin.merge_single_arm(q10_side_air, side, q_side_stage)
                self._set_phase("LIFT_FROM_SURFACE", f"{touch_hdr} → +{lift_m*1000:.1f}mm へ浮かせてクリアランス確保中")
                await self._move_and_wait(q10_side_stage, cmd_aux, 1.1, "lift_clear_surface", replace=True)
                await self._sleep_checked(0.20)

                rob_chk = self.client.latest_telemetry.get("robot") or {}
                q_arm_chk = rob_chk.get("q_arm")
                if q_arm_chk and len(q_arm_chk) == 10:
                    q5_chk = q_arm_chk[:5] if side == "left" else q_arm_chk[5:10]
                    chk_tcp, _ = self.kin.arm_fk(side, q5_chk, tcp_off)
                    err_z = z_goal_actual - float(chk_tcp[2])
                    if abs(err_z) > 0.0008:
                        z_cmd_clear = float(np.clip(z_cmd_clear + err_z, z_goal_actual - 0.004, z_goal_actual + 0.016))
                        side_stage_pos = self.kin.clamp_workspace(
                            np.array([side_stage_pos[0], side_stage_pos[1], z_cmd_clear], dtype=np.float64), side, cfg
                        )
                        q_side_stage, _, _, _ = self.kin.solve_oriented_pinch_ik(
                            side, side_stage_pos, act_yaw, cfg, q_seed=q_side_stage
                        )
                        q10_side_stage = self.kin.merge_single_arm(q10_side_air, side, q_side_stage)
                        await self._move_and_wait(q10_side_stage, cmd_aux, 0.85, "trim_surface_lift", replace=True)
                        await self._sleep_checked(0.15)
                        rob_chk2 = self.client.latest_telemetry.get("robot") or {}
                        q_arm_chk2 = rob_chk2.get("q_arm") or q_arm_chk
                        q5_chk2 = q_arm_chk2[:5] if side == "left" else q_arm_chk2[5:10]
                        chk_tcp, _ = self.kin.arm_fk(side, q5_chk2, tcp_off)
                    act_clear_mm = (float(chk_tcp[2]) - (z_surface if z_surface is not None else z_table)) * 1000.0
                else:
                    act_clear_mm = lift_m * 1000.0

                grasp_pos = self.kin.clamp_workspace(
                    np.array([grasp_pos[0], grasp_pos[1], z_cmd_clear], dtype=np.float64), side, cfg
                )
                q_grasp, err_grasp, _, _ = self.kin.solve_oriented_pinch_ik(
                    side, grasp_pos, act_yaw, cfg, q_seed=q_side_stage
                )
                self.status.planned_ik_error_mm = round(err_grasp * 1000.0, 1)
                q10_grasp = self.kin.merge_single_arm(q10_side_stage, side, q_grasp)

                self._set_phase(
                    "SLIDE_INTO_CPINCH",
                    f"{touch_hdr} 実測クリアランス +{act_clear_mm:.1f}mm を維持して親指×人差指の開口部へ水平スライド中",
                )
                q10_slide_path, _, _ = self.kin.plan_cartesian_ik_path(
                    side, side_stage_pos, grasp_pos, act_yaw, cfg, q10_side_stage, q_seed=q_side_stage, step_m=0.0025
                )
                await self._move_path_and_wait(q10_slide_path, cmd_aux, 1.6, "glide_into_cpinch")
                await self._sleep_checked(0.25)
            else:
                self._set_phase("DESCEND_TO_CHOCO", "対象位置へ静かに垂直下降中")
                await self._move_and_wait(q10_grasp, cmd_aux, 2.2, "descend_choco")

            # Step 4: 親指×人差指 机面ガイド対称自己調心ピンチ (または近接センサー減速ピンチ)
            use_tripod = cfg.use_tripod_grasp or is_surface_pinch or is_cpinch or (choco.width_mm <= 42.0)
            use_two_finger = bool(cfg.two_finger_pinch and (is_surface_pinch or is_cpinch))
            expected_w = float(grasp_span_mm) if (is_surface_pinch or is_cpinch) else float(choco.width_mm)
            target_f = float(active_prof["recommended_force_N"]) if active_prof else cfg.gentle_force_N
            prof_tag = f"（記憶『{active_prof['name']}』照合ON）" if active_prof else ""
            finger_tag = (
                ("親指×人差指 机面ガイド自己調心" if use_two_finger else "親指×[人差指+中指] 3指机面ガイド把持")
                if is_surface_pinch
                else ("親指×人差指2本" if use_two_finger else "3指ピンチ")
            )
            self._set_phase(
                "GENTLE_GRASP",
                f"{finger_tag} → {target_f:.2f}N 優しさ把持を実行中 (対象辺 {expected_w:.0f}mm){prof_tag}",
            )
            await self.client.send_command(
                "hand_gentle_grasp",
                side=side,
                target_force_N=target_f,
                pre_open_amount=choco.pre_open_amount,
                use_tripod_only=use_tripod,
                two_finger_only=use_two_finger,
                thumb_aux=thumb_aux,
                expected_width_mm=expected_w,
                learned_profile=active_prof,
                posture=posture,
            )
            grasp_res = await self._wait_hand_grasp_settled(side, timeout_s=14.0)
            g_snap = ((self.client.latest_telemetry.get("hands", {}).get(side) or {}).get("grasp")) or {}
            is_verified = bool(g_snap.get("grasp_verified")) and not bool(g_snap.get("empty_close")) and grasp_res == "holding"
            outcome["grasp_verified"] = bool(is_verified)
            outcome["grasp_detail"] = g_snap.get("detail", "")
            recorder.log_event("grasp_settled", grasp_res=grasp_res, verified=is_verified, detail=g_snap.get("detail", ""))
            recorder.save_keyframe(
                "04_pinch_settled",
                stereo_pair=self.client.latest_stereo,
                hd_tuple=self.client.latest_hd,
                fallback_left_bgr=self.latest_left_bgr,
            )

            # ★ 安全ルール: もし掴めなくても絶対に暴れず、真上直線リフト→天板干渉ゼロ経路→初期姿勢へゆっくり戻して静止する！
            if not is_verified:
                if bool(g_snap.get("empty_close")):
                    outcome["failure_mode"] = "empty_close_missed_object"
                elif grasp_res == "timeout":
                    outcome["failure_mode"] = "hand_settle_timeout"
                else:
                    outcome["failure_mode"] = "unverified_contact"

                self._set_phase(
                    "SAFE_RETURN_EMPTY",
                    "対象を確実に掴めなかったため、真上へ逃がしてから天板を避けて元の位置へ戻っています...",
                )
                try:
                    await self.client.send_command("hand_gentle_release", side=side)
                except Exception:
                    pass
                await self._sleep_checked(1.0)
                if is_surface_pinch:
                    q10_up_empty, _, _ = self.kin.plan_cartesian_ik_path(
                        side, grasp_pos, side_stage_pos, act_yaw, cfg, q10_grasp, q_seed=q_grasp, step_m=0.0035
                    )
                    await self._move_path_and_wait(q10_up_empty, cmd_aux, 1.6, "vertical_escape_empty")
                await self._move_and_wait(q10_pre, cmd_aux, 2.0, "retreat_up_empty")
                await self._move_and_wait(wp2, cur_aux, 2.5, "retreat_over_box")
                await self._move_and_wait(wp1, cur_aux, 3.0, "retreat_behind_box")
                await self._move_and_wait(wp0, cur_aux, 3.0, "retreat_side_clear")
                await self._move_and_wait(initial_q10, cur_aux, 2.8, "return_initial")
                await self.client.send_command("arm_disengage")
                await self._sleep_checked(2.6)
                self.visual_servo.clear()
                self._set_phase(
                    "RETURNED_NOT_GRASPED",
                    "対象を掴めなかったため、天板干渉ゼロ経路で元の位置へ戻して完全静止しました。",
                )
                return

            # ★ grasp_and_hold モード: 6.0cm 真上へデカルト直線リフトして把持を確認後、保持したまま元の位置へゆっくり戻して静止！
            if cfg.execution_mode == "grasp_and_hold":
                if is_surface_pinch or is_cpinch:
                    short_lift_pos = self.kin.clamp_workspace(grasp_pos + np.array([0.0, 0.0, 0.060]), side, cfg)
                    q10_lift_path, _, _ = self.kin.plan_cartesian_ik_path(
                        side, grasp_pos, short_lift_pos, act_yaw, cfg, q10_grasp, q_seed=q_grasp, step_m=0.003
                    )
                    self._set_phase("LIFT_VERIFY", f"把持成立確認OK ({g_snap.get('detail', '')}) → 6.0cm 真上へ直線リフト中")
                    await self._move_path_and_wait(q10_lift_path, cmd_aux, 2.3, "vertical_lift_verify")
                else:
                    short_lift_pos = self.kin.clamp_workspace(grasp_pos + np.array([-0.01, 0.0, 0.055]), side, cfg)
                    q_short_lift, _ = self.kin.solve_arm_ik(
                        side, short_lift_pos, cfg, desired_palm_normal=palm_ori, q_seed=q_grasp, posture=posture
                    )
                    q10_short_lift = self.kin.merge_single_arm(q10_grasp, side, q_short_lift)
                    self._set_phase("LIFT_VERIFY", f"把持成立確認OK ({g_snap.get('detail', '')}) → 5.5cm そっと持ち上げ中")
                    await self._move_and_wait(q10_short_lift, cmd_aux, 2.4, "lift_verify_hold")

                await self._sleep_checked(1.8)
                recorder.save_keyframe(
                    "05_lift_verify",
                    stereo_pair=self.client.latest_stereo,
                    hd_tuple=self.client.latest_hd,
                    fallback_left_bgr=self.latest_left_bgr,
                )
                self._set_phase("RETURN_WITH_OBJECT", f"{choco.label} を保持したまま天板干渉ゼロ経路で元の位置へ戻っています...")
                await self._move_and_wait(q10_pre, cmd_aux, 2.0, "retreat_above_box")
                await self._move_and_wait(wp2, cur_aux, 2.5, "retreat_over_box")
                await self._move_and_wait(wp1, cur_aux, 3.0, "retreat_behind_box")
                await self._move_and_wait(wp0, cur_aux, 3.0, "retreat_side_clear")
                await self._move_and_wait(initial_q10, cur_aux, 2.8, "return_initial")
                await self.client.send_command("arm_disengage")
                await self._sleep_checked(2.6)
                self.visual_servo.clear()
                self._set_phase(
                    "GRASP_SUCCESS_HOLD",
                    f"【把持＆帰還完了】{choco.label} を優しく掴んだまま元の位置へ戻り静止しました！（『ハンドを開く』で開放可能）",
                )
                return

            self._set_phase("GRASP_HOLDING", f"対象保持確認OK ({grasp_res}) → 安全高さへ真上リフト")

            # Step 5: 持ち上げ & モード分岐 (把持・リフト確認のみ vs 両掌への手渡し)
            if is_surface_pinch or is_cpinch:
                q10_lift_path, _, _ = self.kin.plan_cartesian_ik_path(
                    side, grasp_pos, lift_pos, act_yaw, cfg, q10_grasp, q_seed=q_grasp, step_m=0.0035
                )
                q10_lift = q10_lift_path[-1]
                await self._move_path_and_wait(q10_lift_path, cmd_aux, 2.5, "lift_choco_straight")
            else:
                q10_lift = self.kin.merge_single_arm(q10_grasp, side, q_lift)
                await self._move_and_wait(q10_lift, cmd_aux, 2.5, "lift_choco")

            if cfg.execution_mode == "pick_only":
                self._set_phase("PICK_HOLD_CHECK", "把持＆持ち上げ成功！3秒間キープしたあと机へ優しく戻します")
                await self._sleep_checked(3.0)
                await self._move_and_wait(q10_grasp, cmd_aux, 2.2, "place_back_table")
                await self.client.send_command("hand_gentle_release", side=side)
                await self._sleep_checked(1.4)
                await self._move_and_wait(q10_pre, cmd_aux, 2.0, "retreat_above_table")
                await self._move_and_wait(wp2, cur_aux, 2.0, "retreat_over_desk")
                await self._move_and_wait(wp1, cur_aux, 2.8, "retreat_behind_desk")
                await self._move_and_wait(wp0, cur_aux, 2.8, "retreat_side_clear")
                await self._move_and_wait(cur_q10, cur_aux, 2.6, "return_home")
                await self.client.send_command("arm_disengage")
                await self._sleep_checked(1.1)
                self.visual_servo.clear()
                self._set_phase("COMPLETED", "対象物の認識・優しさ把持・持ち上げ確認が完了しました！")
                return

            # もし目の前の人が両手を出していれば最新座標へ追従更新！
            if self.latest_gesture and self.latest_gesture.palm_center_torso_m is not None:
                palm_target = self.kin.clamp_workspace(
                    np.array(self.latest_gesture.palm_center_torso_m, dtype=np.float64), side, cfg
                )
                self.status.target_palms_torso_m = [round(float(v), 4) for v in palm_target]

            above_palm_pos = self.kin.clamp_workspace(palm_target + np.array([0.0, 0.0, 0.065]), side, cfg)
            place_palm_pos = self.kin.clamp_workspace(palm_target + np.array([0.0, 0.0, 0.032]), side, cfg)
            palm_flat_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            q_above_palm, _ = self.kin.solve_arm_ik(
                side, above_palm_pos, cfg, desired_palm_normal=palm_flat_down, q_seed=q_lift, posture=posture
            )
            q_place_palm, _ = self.kin.solve_arm_ik(
                side, place_palm_pos, cfg, desired_palm_normal=palm_flat_down, q_seed=q_above_palm, posture=posture
            )

            q10_above_palm = self.kin.merge_single_arm(q10_lift, side, q_above_palm)
            q10_place_palm = self.kin.merge_single_arm(q10_above_palm, side, q_place_palm)

            self._set_phase("MOVE_TO_PALMS", "人の両手のひら上空へゆっくり移動中")
            await self._move_and_wait(q10_above_palm, cur_aux, 2.8, "above_palms")

            # Step 6: 手のひら直上 3cm へそっと降りて指を静かに開き、対象を手のひらへ置く
            self._set_phase("PLACE_ON_PALMS", "手のひらの上へそっと置いています")
            await self._move_and_wait(q10_place_palm, cur_aux, 1.8, "lower_to_palms")
            await self.client.send_command("hand_gentle_release", side=side)
            await self._sleep_checked(1.4)

            # Step 7: 上へ退避して待機姿勢へ戻り ArmSdk を滑らかに解放
            self._set_phase("RETRACT_HOME", "手渡し完了！待機姿勢へ安全に戻っています")
            await self._move_and_wait(q10_above_palm, cur_aux, 1.8, "retreat_up")
            await self._move_and_wait(wp2, cur_aux, 2.0, "retreat_over_desk")
            await self._move_and_wait(wp1, cur_aux, 2.8, "retreat_behind_desk")
            await self._move_and_wait(wp0, cur_aux, 2.8, "retreat_side_clear")
            await self._move_and_wait(cur_q10, cur_aux, 2.6, "return_home")
            await self.client.send_command("arm_disengage")
            await self._sleep_checked(1.1)
            self.visual_servo.clear()

            self._set_phase("COMPLETED", "チョコレートの手渡しが完了しました！")

        except asyncio.CancelledError:
            outcome["failure_mode"] = "cancelled"
            self._set_phase("CANCELLED", "シーケンスを中断しました")
        except Exception as exc:
            outcome["failure_mode"] = f"exception_{type(exc).__name__}"
            outcome["error"] = str(exc)
            self._set_phase("ERROR", f"安全停止: {type(exc).__name__}: {exc}")
            # 異常時は絶対に RL 立位側へ急解放 (disengage) せず、その場で低剛性ソフトホールド＋ハンド停止して暴れを防止する
            try:
                await self.client.send_command("hand_stop")
            except Exception:
                pass
        finally:
            outcome["final_phase"] = self.status.phase
            recorder.save_keyframe(
                "99_final",
                stereo_pair=self.client.latest_stereo,
                hd_tuple=self.client.latest_hd,
                fallback_left_bgr=self.latest_left_bgr,
            )
            self.status.last_run_summary = await recorder.finish(outcome)
