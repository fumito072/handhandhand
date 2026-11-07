#!/usr/bin/env python3
"""
BrainCo Revo 2 ハンド用「近接センサー減速 ＋ 優しさ把持 ＋ 事前オブジェクト触覚認識・記憶 (Touch-to-Learn)」コントローラ。
既存の /home/unitree/revo2-tools/ を一切書き換えず、安全なシリアル通信クラス Hand / CurrentMonitor のみ利用する。
"""
import contextlib
import math
import statistics
import sys
import time
from typing import Optional

sys.path.insert(0, "/home/unitree/revo2-tools")
try:
    from revo2_slow_check import Hand, CurrentMonitor, CurrentTrip  # noqa: E402
except ImportError:
    # オフライン単体テスト用フォールバック (実機 R1 上では必ず /home/unitree/revo2-tools がロードされる)
    class CurrentTrip(Exception):
        pass

    class CurrentMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    Hand = None

PROTECTION_MA = 1000
CURRENT_SLOW_START_MA = 450.0
CURRENT_SOFT_STOP_MA = 900.0
LIVE_COMMAND_INTERVAL = 0.065

# モーター配列: [0: thumb_flex, 1: thumb_aux(対立), 2: index, 3: middle, 4: ring, 5: pinky]
FINGER_MOTORS = [0, 2, 3, 4, 5]
FINGER_LABELS_JA = ["親指", "人差指", "中指", "薬指", "小指"]

OPEN_POSE = [0, 450, 0, 0, 0, 0]
# C字2指精密ピンチ待機姿勢: 人差指(2)を軽くプレカーブ(215)させ親指対立軸(495)とゴムパッド正面同士で対向した約58mm開口を作る
C_PINCH_OPEN_BASE = [110, 495, 215, 12, 10, 10]
# 表面拘束またぎピンチ (Surface-Constrained Straddle Pinch) 待機姿勢:
# 親指(60)・人差指(150)を大きく開いて約65〜68mmの対向開口を作り、±18mmの位置ズレと±20°の角度ズレを吸収する。中指〜小指(10)は完全伸展退避。
STRADDLE_OPEN_BASE = [60, 495, 150, 10, 10, 10]
# 3指トライポッド表面把持の人差指・中指プリカーブ (IMG_9460 空中撮影の形C: 指先が真下を向き親指先端とほぼ同じ高さ・間隔約45mm)
TRIPOD_DOWN_IDX = 450
# Revo 2 の親指対立(1)の実機上限は 500 のため <= 495 に制限、親指屈曲(0)は 680 まで許容
CHOCO_PINCH_MAX = [680, 495, 780, 880, 720, 650]


def parse_capacitive_tactile(values):
    if len(values) != 30:
        raise ValueError("Expected 30 tactile registers")
    fingers = []
    for idx in range(5):
        normal, tangent, angle = values[idx * 3:idx * 3 + 3]
        status = values[25 + idx]
        prox = (values[15 + idx * 2] << 16) | values[16 + idx * 2]
        fingers.append({
            "name": FINGER_LABELS_JA[idx],
            "normal_N": round(normal / 100.0, 3),
            "tangent_N": round(tangent / 100.0, 3),
            "direction_deg": None if angle == 65535 else angle,
            "proximity": int(prox),
            "sequence": status >> 8,
            "flags": status & 255,
        })
    return {"kind": "capacitive", "unit": "N", "fingers": fingers}


def current_speed_scale(current_mA: float) -> float:
    rem = (CURRENT_SOFT_STOP_MA - abs(current_mA)) / (CURRENT_SOFT_STOP_MA - CURRENT_SLOW_START_MA)
    return max(0.0, min(1.0, rem)) ** 2


def estimate_pinch_span_mm(positions, two_finger_only: bool = False) -> float:
    """親指屈曲 q[0] と対向主指 (2指ピンチ時は q[2]、3指時は min(q[2], q[3])) の実機校正エンコーダ和から物理開口幅 [mm] を推定する。"""
    q_thumb = float(positions[0])
    q_opp = float(positions[2]) if (two_finger_only or float(positions[3]) < 60.0) else float(min(positions[2], positions[3]))
    # 実機計測校正: q_sum=325 -> 58mm, q_sum=1345 (親指・人差指ゴム先端の空閉じ接触点) -> 0mm
    span = 58.0 * (1345.0 - (q_thumb + q_opp)) / (1345.0 - 325.0)
    return round(max(0.0, min(90.0, span)), 1)


class InHandObjectLearnPolicy:
    """
    ユーザーが事前に指の間にセットしたオブジェクトを、現在のハンド位置からそのまま優しく触診・認識して
    固有の触覚・形状プロファイル（把持エンコーダ角・幅・硬さ剛性・対向法線力・近接変化）を抽出するポリシー。
    アームは一切動かさず、開始時に指を開いて対象を落とすことも絶対にしない。
    """

    def __init__(
        self,
        start_positions,
        now: float,
        probe_force_N: float = 0.44,
        release_after: bool = False,
    ):
        self.start_positions = [float(v) for v in start_positions]
        self.q = [float(v) for v in start_positions]
        self.probe_force = max(0.30, min(0.75, float(probe_force_N)))
        self.onset_threshold = 0.13
        self.release_after = bool(release_after)

        self.approach_rate = 105.0    # 現位置からのゆっくりした触診アプローチ速度 (units/s)
        self.near_rate = 44.0         # 近接/初接触後の微速アプローチ (units/s)
        self.squeeze_rate = 28.0      # 硬さ (ΔF/Δq) 計測用の微小押し込み速度 (units/s)
        self.release_rate = 165.0

        self.stage = "init_scan"
        self.stage_since = now
        self.last_time = now
        self.detail = "現在の指位置からセット中のオブジェクトを初期スキャンしています..."
        self.motion_started_at = 0.0

        self.init_prox_samples = []
        self.init_norm_samples = []
        self.prox_baseline = [0.0] * 5
        self.prox_threshold = [4000.0] * 5
        self.tare_force = [0.0] * 5

        self.delta_prox = [0.0] * 5
        self.delta_force = [0.0] * 5
        self.near_active = [False] * 5
        self.contact_locked = [False] * 5
        self.effective_current = [0.0] * 6

        self.onset_recorded = [False] * 5
        self.q_onset = [0.0] * 5
        self.f_onset = [0.0] * 5
        self.needs_pre_relax = False

        self.hold_norm_samples = []
        self.hold_tan_samples = []
        self.hold_prox_samples = []
        self.hold_pos_samples = []

        self.grasp_verified = False
        self.empty_close = False
        self.learned_result: Optional[dict] = None

    def command_release(self, now: float, reason: str = "ハンドを優しく開いています"):
        self.stage = "releasing"
        self.stage_since = now
        self.detail = reason

    def snapshot(self) -> dict:
        return {
            "mode": "learn_object",
            "stage": self.stage,
            "detail": self.detail,
            "target_force_N": round(self.probe_force, 3),
            "delta_force_N": [round(v, 3) for v in self.delta_force],
            "delta_proximity": [round(v, 1) for v in self.delta_prox],
            "prox_threshold": [round(v, 1) for v in self.prox_threshold],
            "near_slowdown": list(self.near_active),
            "contact_locked": list(self.contact_locked),
            "grasp_verified": bool(self.grasp_verified),
            "empty_close": bool(self.empty_close),
            "learned_result": self.learned_result,
        }

    def update(self, now: float, actual_pos, sensor: dict, motor_currents) -> list:
        dt = max(0.0, min(0.1, now - self.last_time))
        self.last_time = now

        if motor_currents and len(motor_currents) == 6:
            self.effective_current = [
                max(abs(float(raw)), old * math.exp(-dt / 0.25))
                for raw, old in zip(motor_currents, self.effective_current)
            ]
        cur_scales = [current_speed_scale(c) for c in self.effective_current]

        fingers = sensor.get("fingers", []) if sensor else []
        has_5 = len(fingers) == 5

        if self.stage == "releasing":
            target_open = self.start_positions if self.empty_close else OPEN_POSE
            for m_idx, target in enumerate(target_open):
                step = self.release_rate * dt
                self.q[m_idx] += max(-step, min(step, float(target) - self.q[m_idx]))
            if all(abs(actual_pos[i] - target_open[i]) <= 22 for i in range(6)):
                if self.empty_close:
                    self.stage = "learn_failed"
                    self.detail = "指の間にオブジェクトを検出できませんでした（元の位置へ戻しました）"
                else:
                    self.stage = "memorized_released"
                    self.detail = "オブジェクト特徴の記憶と開放が完了しました"
            return [round(v) for v in self.q]

        if self.stage in ("memorized_holding", "memorized_released", "learn_failed", "paused"):
            if has_5:
                for i in range(5):
                    self.delta_prox[i] = float(fingers[i]["proximity"]) - self.prox_baseline[i]
                    self.delta_force[i] = max(0.0, float(fingers[i]["normal_N"]) - self.tare_force[i])
            return [round(v) for v in self.q]

        if not has_5:
            self.stage = "paused"
            self.detail = "5指の触覚・近接センサーデータがありません"
            return [round(v) for v in actual_pos]

        # Stage 1: 現在のセット位置のまま 0.30秒 スキャン（絶対に開かない！）
        if self.stage == "init_scan":
            self.init_prox_samples.append([float(f["proximity"]) for f in fingers])
            self.init_norm_samples.append([float(f["normal_N"]) for f in fingers])

            # もし親指対立軸 q[1] が完全に横へ寝ていて (q[1] < 160) かつ未接触なら対向位置へゆっくり向ける
            if self.start_positions[1] < 160.0 and float(fingers[0]["normal_N"]) < 0.14:
                step_aux = 260.0 * dt
                self.q[1] = min(450.0, self.q[1] + step_aux)

            if len(self.init_prox_samples) >= 4 and (now - self.stage_since) >= 0.30:
                cols_p = list(zip(*self.init_prox_samples))
                cols_f = list(zip(*self.init_norm_samples))
                self.prox_baseline = [statistics.median(c) for c in cols_p]
                self.prox_threshold = [
                    max(3200.0, 5.5 * 1.4826 * statistics.median(abs(v - self.prox_baseline[i]) for v in c))
                    for i, c in enumerate(cols_p)
                ]
                med_f = [statistics.median(c) for c in cols_f]
                # すでにユーザーが指をオブジェクトに軽く当ててセットしている場合は接触力を消さないよう tare を上限 0.05N に抑える
                self.tare_force = [min(0.05, f0) if f0 >= 0.15 else f0 for f0 in med_f]
                pre_df = [max(0.0, med_f[i] - self.tare_force[i]) for i in range(5)]

                if pre_df[0] >= 0.32 and any(pre_df[j] >= 0.32 for j in (1, 2, 3)):
                    # すでにしっかり挟まれている場合は 16 units (~1.4mm) だけ微緩めしてから剛性を再計測する
                    self.needs_pre_relax = True
                    self.stage = "pre_relax"
                    self.stage_since = now
                    self.detail = "既に挟まれている対象を検知 → 落とさず微小触診して硬さを計測中..."
                else:
                    self.stage = "palpate_approach"
                    self.stage_since = now
                    self.detail = "セットされた指位置から優しく閉じてオブジェクト形状を認識中..."
            return [round(v) for v in self.q]

        # Stage 1.5: 既に強く挟まれていた場合のみ 16 units 微緩め（対象は落とさない）
        if self.stage == "pre_relax":
            for i in range(5):
                m_idx = FINGER_MOTORS[i]
                target_r = max(0.0, self.start_positions[m_idx] - 16.0)
                step = 65.0 * dt
                self.q[m_idx] += max(-step, min(step, target_r - self.q[m_idx]))
            if (now - self.stage_since) >= 0.28:
                self.stage = "palpate_approach"
                self.stage_since = now
                self.detail = "オブジェクト輪郭・接触開始点を計測中..."
            return [round(v) for v in self.q]

        # 共通センサー差分更新
        for i in range(5):
            dp = float(fingers[i]["proximity"]) - self.prox_baseline[i]
            df = max(0.0, float(fingers[i]["normal_N"]) - self.tare_force[i])
            self.delta_prox[i] = dp
            self.delta_force[i] = df
            self.near_active[i] = abs(dp) >= self.prox_threshold[i]

        # Stage 2: ゆっくり閉じて初接触点 (q_onset, f_onset) を検出
        if self.stage == "palpate_approach":
            opposing_ready = self.onset_recorded[0] and any(self.onset_recorded[j] for j in (1, 2, 3))
            lead_opp_q = min(float(actual_pos[2]), float(actual_pos[3]))

            for i in range(5):
                m_idx = FINGER_MOTORS[i]
                df = self.delta_force[i]

                if not self.onset_recorded[i] and (df >= self.onset_threshold or cur_scales[m_idx] <= 0.08):
                    self.onset_recorded[i] = True
                    self.q_onset[i] = float(actual_pos[m_idx])
                    self.f_onset[i] = df
                    self.q[m_idx] = float(actual_pos[m_idx])
                    continue

                if self.onset_recorded[i]:
                    # 他の対向指が触れるまでその場でソフト待機
                    continue

                # 薬指・小指は人差指/中指より大きく巻き込みすぎないよう制限（小物体を下から弾かない）
                if i in (3, 4) and (self.onset_recorded[1] or self.onset_recorded[2]):
                    if float(actual_pos[m_idx]) >= lead_opp_q + 55.0:
                        self.q[m_idx] = float(actual_pos[m_idx])
                        continue

                rate = self.near_rate if (self.near_active[i] or df >= 0.06 or any(self.onset_recorded)) else self.approach_rate
                rate *= cur_scales[m_idx]
                self.q[m_idx] = min(
                    float(CHOCO_PINCH_MAX[m_idx]),
                    self.q[m_idx] + rate * dt,
                    float(actual_pos[m_idx]) + 24.0,
                )

            opposing_ready = self.onset_recorded[0] and any(self.onset_recorded[j] for j in (1, 2, 3))
            if opposing_ready:
                # 未接触の補助指はその場で止める
                for j in range(5):
                    if not self.onset_recorded[j]:
                        self.q[FINGER_MOTORS[j]] = float(actual_pos[FINGER_MOTORS[j]])
                self.stage = "palpate_squeeze"
                self.stage_since = now
                self.detail = "オブジェクト捕捉OK！微小スクイーズで硬さ・反力特性を計測中..."
                return [round(v) for v in self.q]

            # ストローク終端まで閉じても対向接触が無い場合は空振り → 元の位置へ優しく戻す
            primary_maxed = (
                actual_pos[0] >= CHOCO_PINCH_MAX[0] - 18
                and actual_pos[2] >= CHOCO_PINCH_MAX[2] - 18
                and actual_pos[3] >= CHOCO_PINCH_MAX[3] - 18
            )
            if primary_maxed or (now - self.stage_since) > 7.5:
                self.grasp_verified = False
                self.empty_close = True
                self.stage = "releasing"
                self.stage_since = now
                self.detail = "対象が検出されなかったため元の開き位置へ戻しています..."
            return [round(v) for v in self.q]

        # Stage 3: 接触指のみ微小押し込み (onset -> probe_force) で接触剛性 ΔF/Δq を計測
        if self.stage == "palpate_squeeze":
            all_settled = True
            for i in range(5):
                if not self.onset_recorded[i]:
                    continue
                m_idx = FINGER_MOTORS[i]
                df = self.delta_force[i]
                traveled = float(actual_pos[m_idx]) - self.q_onset[i]
                if self.contact_locked[i] or df >= self.probe_force or traveled >= 42.0 or cur_scales[m_idx] <= 0.06:
                    self.contact_locked[i] = True
                    self.q[m_idx] = float(actual_pos[m_idx])
                else:
                    all_settled = False
                    step = self.squeeze_rate * cur_scales[m_idx] * dt
                    self.q[m_idx] = min(
                        float(CHOCO_PINCH_MAX[m_idx]),
                        self.q[m_idx] + step,
                        float(actual_pos[m_idx]) + 14.0,
                    )

            if all_settled or (now - self.stage_since) >= 0.95:
                for i in range(5):
                    self.q[FINGER_MOTORS[i]] = float(actual_pos[FINGER_MOTORS[i]])
                self.stage = "memorize_hold"
                self.stage_since = now
                self.detail = "5指の触覚・近接・関節角プロファイルを統計記録中..."
            return [round(v) for v in self.q]

        # Stage 4: 安定保持しながら複数フレームの中央値でプロファイル確定
        if self.stage == "memorize_hold":
            self.hold_norm_samples.append([float(self.delta_force[i]) for i in range(5)])
            self.hold_tan_samples.append([float(fingers[i]["tangent_N"]) for i in range(5)])
            self.hold_prox_samples.append([float(fingers[i]["proximity"]) for i in range(5)])
            self.hold_pos_samples.append([float(v) for v in actual_pos])

            if len(self.hold_pos_samples) >= 6 and (now - self.stage_since) >= 0.55:
                med_pos = [int(round(statistics.median(c))) for c in zip(*self.hold_pos_samples)]
                med_norm = [round(statistics.median(c), 3) for c in zip(*self.hold_norm_samples)]
                med_tan = [round(statistics.median(c), 3) for c in zip(*self.hold_tan_samples)]
                med_prox = [int(round(statistics.median(c))) for c in zip(*self.hold_prox_samples)]
                d_prox = [round(med_prox[i] - self.prox_baseline[i], 1) for i in range(5)]

                contact_mask = [
                    bool(self.onset_recorded[i] or med_norm[i] >= 0.11)
                    for i in range(5)
                ]
                active_names = [FINGER_LABELS_JA[i] for i in range(5) if contact_mask[i]]

                # 接触剛性 [N / 100 encoder units] の算出
                stiffs = []
                for i in range(5):
                    if contact_mask[i]:
                        m_idx = FINGER_MOTORS[i]
                        dq = max(3.5, float(med_pos[m_idx]) - self.q_onset[i])
                        df_rise = max(0.04, med_norm[i] - min(self.f_onset[i], med_norm[i] - 0.04))
                        stiffs.append((df_rise / dq) * 100.0)
                stiffness = round(statistics.median(stiffs) if stiffs else 1.0, 3)

                if stiffness >= 1.15:
                    hardness = "硬質 (キャップ・箱等)"
                    rec_force = 0.46
                elif stiffness >= 0.55:
                    hardness = "標準 (紙箱・ボトル等)"
                    rec_force = 0.38
                else:
                    hardness = "軟質 (袋菓子・柔らかめ)"
                    rec_force = 0.31

                span_mm = estimate_pinch_span_mm(med_pos)
                self.learned_result = {
                    "start_positions": [int(round(v)) for v in self.start_positions],
                    "pinch_positions": med_pos,
                    "estimated_span_mm": span_mm,
                    "contact_mask": contact_mask,
                    "active_finger_names": active_names,
                    "hold_normal_N": med_norm,
                    "hold_tangent_N": med_tan,
                    "hold_proximity": med_prox,
                    "delta_proximity": d_prox,
                    "stiffness_N_per_100u": stiffness,
                    "hardness_label": hardness,
                    "recommended_force_N": rec_force,
                    "position_tolerance": 125,
                }
                self.grasp_verified = True
                self.empty_close = False

                if self.release_after:
                    self.stage = "releasing"
                    self.stage_since = now
                    self.detail = f"記憶完了（推定幅 {span_mm}mm / {hardness}）→ 自動リリース中"
                else:
                    self.stage = "memorized_holding"
                    self.stage_since = now
                    self.detail = (
                        f"記憶完了！推定幅 {span_mm}mm・{hardness}（落下防止のため優しく保持中 — 『ハンドを開く』で外せます）"
                    )
            return [round(v) for v in self.q]

        return [round(v) for v in self.q]


class GentleChocoGraspPolicy:
    """
    2段階ジェントル把持 ＋ 記憶オブジェクトプロファイル照合（段ボール誤接触・空振り分離）:
      1. ベースライン記録後、対象物へ向けて指を閉じる（アプローチ速度 `fast_rate` = 220 units/s）
      2. 指先の静電容量式【近接センサー (`proximity`)】がベースライン+閾値を超えた指から即座に
         ソフトランディング速度（`prox_slow_rate` = 85 units/s）へ減速！
      3. 指先の【法線力センサー (`normal_N`)】が目標力 `target_force_N` に達した指から個別に位置固定。
      4. `learned_profile` がある場合、成立時の「対向法線力バランス」と「記憶した指エンコーダ幅」を照合し、
         段ボール表面の擦れ（接線力のみ／幅不一致）や空振りを確実に分離する。
    """

    def __init__(
        self,
        start_positions,
        now: float,
        target_force_N: float = 0.38,
        pre_open_amount: int = 0,
        use_tripod_only: bool = False,
        learned_profile: Optional[dict] = None,
        posture: str = "c_pinch",
        two_finger_only: bool = True,
        thumb_aux: int = 485,
        expected_width_mm: Optional[float] = None,
    ):
        self.q = [float(v) for v in start_positions]
        self.learned_profile = learned_profile
        self.posture = posture or "c_pinch"
        self.two_finger_only = bool(two_finger_only)
        self.thumb_aux = max(435, min(498, int(thumb_aux)))
        self.expected_width_mm = (
            max(8.0, min(80.0, float(expected_width_mm))) if expected_width_mm is not None else None
        )
        if learned_profile and learned_profile.get("recommended_force_N"):
            target_force_N = float(learned_profile["recommended_force_N"])
        self.target_force = max(0.25, min(1.2, float(target_force_N)))
        self.contact_threshold = max(0.09, min(0.17, self.target_force * 0.38))
        self.fast_rate = 152.0 if self.posture == "surface_pinch" else (168.0 if self.posture == "c_pinch" else 220.0)
        self.prox_slow_rate = 74.0 if self.posture in ("surface_pinch", "c_pinch") else 85.0
        self.release_rate = 180.0
        self.prepare_rate = 380.0

        if self.posture == "surface_pinch":
            # 表面拘束またぎピンチ: ~66mm ワイド開口から対称閉鎖でチロルチョコを中央へ引き込み整列させる
            # two_finger_only=False 時は 親指(0) × [人差指(1)+中指(2)] の3指トライポッド壁として捕捉幅を2.5倍化（薬指・小指は退避）
            t_aux_eff = self.thumb_aux if self.two_finger_only else max(452, min(482, self.thumb_aux - 22))
            mid_open = STRADDLE_OPEN_BASE[3] if self.two_finger_only else TRIPOD_DOWN_IDX
            idx_open = (
                max(115, min(190, STRADDLE_OPEN_BASE[2] + pre_open_amount // 4))
                if self.two_finger_only else TRIPOD_DOWN_IDX
            )
            self.pre_open_pose = [
                max(45, min(120, STRADDLE_OPEN_BASE[0] + pre_open_amount // 4)),
                t_aux_eff,
                idx_open,
                mid_open,
                STRADDLE_OPEN_BASE[4],
                STRADDLE_OPEN_BASE[5],
            ]
            self.active_finger_indices = [0, 1] if self.two_finger_only else [0, 1, 2]
            idle_motors = (3, 4, 5) if self.two_finger_only else (4, 5)
            for idle_m in idle_motors:
                self.q[idle_m] = float(STRADDLE_OPEN_BASE[idle_m])
        elif self.posture == "c_pinch" and self.two_finger_only:
            # 親指(0)＋人差指(1->motor 2)の純粋2指精密ピンチ: 中指〜小指は退避位置のまま固定し斜め回転チョコの角への干渉を完全防止
            self.pre_open_pose = [
                max(95, min(165, C_PINCH_OPEN_BASE[0] + pre_open_amount // 3)),
                self.thumb_aux,
                max(120, min(195, C_PINCH_OPEN_BASE[2] + pre_open_amount // 3)),
                C_PINCH_OPEN_BASE[3],
                C_PINCH_OPEN_BASE[4],
                C_PINCH_OPEN_BASE[5],
            ]
            self.active_finger_indices = [0, 1]
            # 中指〜小指がもし曲がっていれば退避位置へ戻しておく
            for idle_m in (3, 4, 5):
                self.q[idle_m] = float(C_PINCH_OPEN_BASE[idle_m])
        elif learned_profile and len(learned_profile.get("pre_open_positions", [])) == 6:
            self.pre_open_pose = [int(v) for v in learned_profile["pre_open_positions"]]
            cmask = learned_profile.get("contact_mask", [True, True, True, False, False])
            act = [i for i in range(5) if (i in (0, 1, 2) or (i < len(cmask) and cmask[i]))]
            self.active_finger_indices = act if len(act) >= 2 else [0, 1, 2]
        elif self.posture == "c_pinch":
            self.pre_open_pose = [
                max(115, min(205, C_PINCH_OPEN_BASE[0] + pre_open_amount // 2)),
                self.thumb_aux,
                max(135, min(230, C_PINCH_OPEN_BASE[2] + pre_open_amount // 2)),
                max(135, min(220, 155 + pre_open_amount // 2)),
                C_PINCH_OPEN_BASE[4],
                C_PINCH_OPEN_BASE[5],
            ]
            self.active_finger_indices = [0, 1, 2]
        else:
            self.pre_open_pose = [
                max(0, min(350, pre_open_amount // 2)),
                460,
                max(0, min(400, pre_open_amount)),
                max(0, min(400, pre_open_amount)),
                0 if use_tripod_only else max(0, min(300, pre_open_amount)),
                0 if use_tripod_only else max(0, min(250, pre_open_amount)),
            ]
            self.active_finger_indices = [0, 1, 2] if use_tripod_only else [0, 1, 2, 3, 4]

        self.stage = "baseline"
        self.stage_since = now
        self.last_time = now
        self.detail = "指先近接・触覚センサーの基準値を計測中"

        self.prox_samples = []
        self.force_samples = []
        self.prox_baseline = [0.0] * 5
        self.prox_threshold = [4500.0] * 5
        self.force_ref = [0.0] * 5

        self.delta_prox = [0.0] * 5
        self.delta_force = [0.0] * 5
        self.near_active = [False] * 5
        self.contact_locked = [False] * 5
        self.effective_current = [0.0] * 6
        self.grasp_verified = False
        self.empty_close = False
        self.profile_match_score: Optional[float] = None
        self.motion_started_at = 0.0

    def command_release(self, now: float, reason: str = "手のひら上で優しく開放中"):
        self.stage = "releasing"
        self.stage_since = now
        self.detail = reason

    def snapshot(self) -> dict:
        return {
            "mode": "gentle_grasp",
            "stage": self.stage,
            "detail": self.detail,
            "target_force_N": round(self.target_force, 3),
            "delta_force_N": [round(v, 3) for v in self.delta_force],
            "delta_proximity": [round(v, 1) for v in self.delta_prox],
            "prox_threshold": [round(v, 1) for v in self.prox_threshold],
            "near_slowdown": list(self.near_active),
            "contact_locked": list(self.contact_locked),
            "grasp_verified": bool(self.grasp_verified),
            "empty_close": bool(self.empty_close),
            "profile_match_score": round(self.profile_match_score, 3) if self.profile_match_score is not None else None,
        }

    def _evaluate_settlement(self, actual_pos, fingers, opposing_locked: bool):
        peak_f = max((self.delta_force[i] for i in self.active_finger_indices), default=0.0)
        peak_tan = max((float(fingers[i].get("tangent_N", 0.0)) for i in self.active_finger_indices), default=0.0)
        opp_motors = [FINGER_MOTORS[j] for j in self.active_finger_indices if j != 0] or [2]
        q_thumb_now = float(actual_pos[0])
        q_opp_now = min(float(actual_pos[m]) for m in opp_motors)
        q_sum_now = q_thumb_now + q_opp_now
        span_now_mm = estimate_pinch_span_mm(actual_pos, two_finger_only=(self.active_finger_indices == [0, 1]))

        # 空閉じ接触ガード: 親指と人差指が対象物を挟まず直接ぶつかる位置 (q_sum >= 1265, 開口 <= 5mm) は必ず空振りと判定！
        if q_sum_now >= 1265.0:
            self.grasp_verified = False
            self.empty_close = True
            self.detail = f"空振り検知（親指と人差指が空閉じ接触: 開口 {span_now_mm:.1f}mm）"
            return

        # 段ボール擦れガード: 対向ロックが無く、法線力が微弱で接線（擦れ）力ばかり出ている場合は机面接触と判定
        if not opposing_locked and peak_f < self.contact_threshold and peak_tan > 0.12:
            self.grasp_verified = False
            self.empty_close = False
            self.detail = f"段ボール表面との擦れ検知（対向法線力不足 {peak_f:.2f}N / 接線 {peak_tan:.2f}N）"
            return

        if self.learned_profile and len(self.learned_profile.get("pinch_positions", [])) == 6:
            prof_q = self.learned_profile["pinch_positions"]
            prof_name = self.learned_profile.get("name", "記憶対象")
            tol = float(self.learned_profile.get("position_tolerance", 125))

            prof_opp = float(prof_q[2]) if self.active_finger_indices == [0, 1] else float(min(prof_q[2], prof_q[3]))
            sum_expected = float(prof_q[0]) + prof_opp
            sum_actual = q_sum_now
            err_thumb = abs(q_thumb_now - float(prof_q[0]))
            err_opp = abs(q_opp_now - prof_opp)
            mean_err = 0.5 * (err_thumb + err_opp)
            score = max(0.0, min(1.0, 1.0 - mean_err / (tol * 1.65)))
            self.profile_match_score = score

            if sum_actual > sum_expected + tol * 1.75 and not opposing_locked:
                self.grasp_verified = False
                self.empty_close = True
                self.detail = f"空振り検知（記憶『{prof_name}』の挟み角より閉じすぎました）"
                return
            if sum_actual < sum_expected - tol * 1.95:
                self.grasp_verified = False
                self.empty_close = False
                self.detail = f"段ボール端/異物接触（記憶『{prof_name}』より大幅に広い位置で停止）"
                return
            if opposing_locked or (peak_f >= self.contact_threshold and mean_err <= tol):
                self.grasp_verified = True
                self.empty_close = False
                self.detail = f"記憶『{prof_name}』一致 {score*100:.0f}% で2指把持完了（幅 {span_now_mm:.0f}mm / {peak_f:.2f}N）"
                return

        # カメラ計測された対象幅 expected_width_mm がある場合、それより 14mm 以上広く開いた位置での停止は段ボール摩擦による早期停止と判定！
        if self.expected_width_mm is not None and span_now_mm > (self.expected_width_mm + 14.0):
            self.grasp_verified = False
            self.empty_close = False
            self.detail = f"対象幅({self.expected_width_mm:.0f}mm)に届く前に停止しました（開口 {span_now_mm:.0f}mm）"
            return

        if opposing_locked or (peak_f >= self.contact_threshold and 400.0 <= q_sum_now <= 1255.0) or sum(self.contact_locked) >= 2:
            self.grasp_verified = True
            self.empty_close = False
            self.detail = f"親指・人差指ピンチ把持完了（推定幅 {span_now_mm:.0f}mm / 最大 {peak_f:.2f} N）"
        else:
            self.grasp_verified = False
            self.empty_close = True
            self.detail = "空振り検知（ストローク上限まで触覚反応なし）"

    def update(self, now: float, actual_pos, sensor: dict, motor_currents) -> list:
        dt = max(0.0, min(0.1, now - self.last_time))
        self.last_time = now

        if motor_currents and len(motor_currents) == 6:
            self.effective_current = [
                max(abs(float(raw)), old * math.exp(-dt / 0.25))
                for raw, old in zip(motor_currents, self.effective_current)
            ]
        cur_scales = [current_speed_scale(c) for c in self.effective_current]

        fingers = sensor.get("fingers", []) if sensor else []
        has_5 = len(fingers) == 5

        if self.stage == "releasing":
            for m_idx, target in enumerate(OPEN_POSE):
                step = self.release_rate * dt
                self.q[m_idx] += max(-step, min(step, target - self.q[m_idx]))
            if all(abs(actual_pos[i] - OPEN_POSE[i]) <= 20 for i in range(6)):
                self.stage = "released"
                self.detail = "開放完了"
            return [round(v) for v in self.q]

        if self.stage in ("holding", "released", "paused"):
            if has_5:
                for i in range(5):
                    self.delta_prox[i] = fingers[i]["proximity"] - self.prox_baseline[i]
                    self.delta_force[i] = max(0.0, fingers[i]["normal_N"] - self.force_ref[i])
            return [round(v) for v in self.q]

        if not has_5:
            self.stage = "paused"
            self.detail = "5指の触覚・近接センサーデータがありません"
            return [round(v) for v in actual_pos]

        # Stage 1: Baseline (0.35s / >= 4 sensor frames)
        if self.stage == "baseline":
            step_aux = self.prepare_rate * dt
            self.q[1] += max(-step_aux, min(step_aux, self.pre_open_pose[1] - self.q[1]))

            self.prox_samples.append([float(f["proximity"]) for f in fingers])
            self.force_samples.append([float(f["normal_N"]) for f in fingers])
            if len(self.prox_samples) >= 4 and (now - self.stage_since) >= 0.35:
                cols_p = list(zip(*self.prox_samples))
                cols_f = list(zip(*self.force_samples))
                self.prox_baseline = [statistics.median(c) for c in cols_p]
                self.prox_threshold = [
                    max(3500.0, 6.0 * 1.4826 * statistics.median(abs(v - self.prox_baseline[i]) for v in c))
                    for i, c in enumerate(cols_p)
                ]
                self.force_ref = [statistics.median(c) for c in cols_f]
                self.stage = "closing"
                self.stage_since = now
                self.detail = "近接センサー監視つきアプローチ中"
            return [round(v) for v in self.q]

        # Stage 2: Proximity-decelerated gentle closing
        if self.stage == "closing":
            for i in range(5):
                dp = float(fingers[i]["proximity"]) - self.prox_baseline[i]
                df = max(0.0, float(fingers[i]["normal_N"]) - self.force_ref[i])
                self.delta_prox[i] = dp
                self.delta_force[i] = df
                self.near_active[i] = abs(dp) >= self.prox_threshold[i]

            span_now_mm = estimate_pinch_span_mm(actual_pos, two_finger_only=(self.active_finger_indices == [0, 1]))
            opp_force = max((self.delta_force[j] for j in self.active_finger_indices if j != 0), default=0.0)
            opp_near = any((self.near_active[j] or self.delta_force[j] >= 0.04) for j in self.active_finger_indices if j != 0)
            both_contacting = (
                len(self.active_finger_indices) >= 2
                and self.delta_force[0] >= self.contact_threshold
                and opp_force >= self.contact_threshold
            )
            is_surface = (self.posture == "surface_pinch")
            guard_margin = 9.5 if is_surface else 13.0
            wide_open_guard = (
                (self.expected_width_mm is not None)
                and (span_now_mm > self.expected_width_mm + guard_margin)
                and (not both_contacting)
            )

            if is_surface:
                # 表面拘束またぎピンチ: 机面の静電近接で片指だけ減速すると非対称になるため、実接触力 (>=0.20N) が片側だけ出た時のみ譲る（縦置き転倒防止）
                thumb_ahead = (self.delta_force[0] >= 0.20) and (opp_force < 0.08)
                index_ahead = (opp_force >= 0.20) and (self.delta_force[0] < 0.08)
            else:
                # 空中C字ピンチ: 7g の軽量チロルチョコを片指だけで押して滑らせないよう近接検知から同期減速
                thumb_ahead = (not wide_open_guard) and (self.near_active[0] or self.delta_force[0] >= 0.04) and (not opp_near)
                index_ahead = (not wide_open_guard) and opp_near and not (self.near_active[0] or self.delta_force[0] >= 0.04)

            near_target_span = (
                self.expected_width_mm is not None
                and span_now_mm <= (self.expected_width_mm + 11.0)
            )

            for i in self.active_finger_indices:
                m_idx = FINGER_MOTORS[i]
                df = self.delta_force[i]

                if self.contact_locked[i]:
                    continue

                lock_force = max(0.95 if is_surface else 1.35, self.target_force * (2.5 if is_surface else 3.2)) if wide_open_guard else self.target_force
                if df >= lock_force or cur_scales[m_idx] <= 0.04:
                    self.contact_locked[i] = True
                    # 包装フィルムの滑り落ちを防ぐ微小保持プリロード (+18 encoder units)
                    self.q[m_idx] = min(float(CHOCO_PINCH_MAX[m_idx]), float(actual_pos[m_idx]) + 18.0)
                    continue

                if is_surface and len(self.active_finger_indices) >= 3:
                    # 3指トライポッド: 真下向きの人差指・中指はほぼ固定壁 (曲げすぎると指先が持ち上がる)、親指が主に閉じてチョコを壁へ押し当てる
                    sym_scale = 1.25 if i == 0 else 0.40
                else:
                    sym_scale = (1.09 if i == 0 else 0.93) if is_surface else 1.0
                if (not wide_open_guard) and df >= self.contact_threshold:
                    self.q[m_idx] = min(self.q[m_idx], float(actual_pos[m_idx]) + 8.0)
                    err = max(0.0, self.target_force - df)
                    rate = max(25.0, min(72.0, 140.0 * err))
                elif (i == 0 and thumb_ahead) or (i != 0 and index_ahead):
                    rate = 24.0 if is_surface else 36.0
                elif is_surface and near_target_span:
                    rate = self.prox_slow_rate * sym_scale
                elif (not is_surface) and (not wide_open_guard) and (self.near_active[i] or any(self.near_active[j] for j in self.active_finger_indices)):
                    rate = self.prox_slow_rate
                elif wide_open_guard:
                    rate = max(95.0, self.prox_slow_rate) * sym_scale
                else:
                    rate = self.fast_rate * sym_scale

                rate *= cur_scales[m_idx]
                self.q[m_idx] = min(
                    float(CHOCO_PINCH_MAX[m_idx]),
                    self.q[m_idx] + rate * dt,
                    float(actual_pos[m_idx]) + 26.0,
                )

            if is_surface:
                if len(self.active_finger_indices) >= 3:
                    self.detail = (
                        f"3指表面ガイド把持中（開口 {span_now_mm:.0f}mm / 親指 {self.delta_force[0]:.2f}N・人差指 {self.delta_force[1]:.2f}N・中指 {self.delta_force[2]:.2f}N）"
                    )
                else:
                    self.detail = (
                        f"表面ガイド対称ピンチ中（開口 {span_now_mm:.0f}mm / 親指 {self.delta_force[0]:.2f}N・人差指 {self.delta_force[1]:.2f}N）"
                    )
            elif any(self.near_active):
                self.detail = "指先近接センサー検知 → ソフト減速して優しく接触中"

            opposing_locked = self.contact_locked[0] and any(
                self.contact_locked[j] for j in self.active_finger_indices if j != 0
            )
            all_active_done = all(
                self.contact_locked[j] or actual_pos[FINGER_MOTORS[j]] >= CHOCO_PINCH_MAX[FINGER_MOTORS[j]] - 15
                for j in self.active_finger_indices
            )
            if (opposing_locked and sum(self.contact_locked) >= 2) or all_active_done or (now - self.stage_since) > 8.5:
                for j in range(5):
                    m_idx = FINGER_MOTORS[j]
                    if not self.contact_locked[j]:
                        self.q[m_idx] = float(actual_pos[m_idx])
                self._evaluate_settlement(actual_pos, fingers, opposing_locked=opposing_locked)
                self.stage = "holding"
                self.stage_since = now

            return [round(v) for v in self.q]

        return [round(v) for v in self.q]


class Revo2Manager:
    """左右 BrainCo Revo 2 ハンドを安全に一元管理するラッパ。"""

    def __init__(self):
        self.stack = contextlib.ExitStack()
        self.hands = {}
        self.meta = {}
        self.tactile = {}
        self.states = {}
        self.policies = {}
        self.last_cmd_sent = {"left": 0.0, "right": 0.0}
        self.last_cmd_target = {"left": None, "right": None}
        self.armed = False

    def connect(self):
        for side in ("left", "right"):
            try:
                hand = self.stack.enter_context(Hand(side))
                vendor = hand.read(3, 970, 1)[0]
                applied = hand.configure_protection(PROTECTION_MA)
                if vendor == 1:
                    en = hand.read(3, 4000, 5)
                    if en != [1, 1, 1, 1, 1]:
                        hand.write(4000, [1, 1, 1, 1, 1])
                self.hands[side] = hand
                self.meta[side] = {
                    "port": hand.port,
                    "slave": hand.slave,
                    "vendor": vendor,
                    "protection_mA": applied,
                }
            except Exception as exc:
                self.meta[side] = {"unavailable": str(exc)}
        self.armed = bool(self.hands)
        return self.meta

    def close(self):
        self.stop_all()
        self.stack.close()

    def stop_all(self):
        self.policies.clear()
        for side, hand in self.hands.items():
            try:
                hand.write(1034, [0] * 6)
                hand.motion_started = False
            except Exception:
                pass

    def move_pose(self, side: str, targets: list, duration_ms: int = 1200):
        if side not in self.hands:
            raise RuntimeError(f"Hand '{side}' is unavailable: {self.meta.get(side)}")
        self.policies.pop(side, None)
        hand = self.hands[side]
        targets_clamped = [max(0, min(1000, int(round(v)))) for v in targets]
        dur = max(400, min(2500, int(duration_ms)))
        hand.current_monitor = CurrentMonitor(hand.guard, hand.max_current)
        hand.motion_started = True
        hand.write(1010, [val for t in targets_clamped for val in (t, dur)])
        self.last_cmd_target[side] = targets_clamped

    def open_hand(
        self,
        side: str,
        pre_open_amount: int = 0,
        duration_ms: int = 1000,
        posture: str = "flat",
        two_finger_only: bool = True,
        thumb_aux: int = 485,
    ):
        t_aux = max(435, min(498, int(thumb_aux)))
        if posture == "surface_pinch":
            t_aux_eff = t_aux if two_finger_only else max(452, min(482, t_aux - 22))
            mid_open = STRADDLE_OPEN_BASE[3] if two_finger_only else TRIPOD_DOWN_IDX
            idx_open = (
                max(115, min(190, STRADDLE_OPEN_BASE[2] + pre_open_amount // 4))
                if two_finger_only else TRIPOD_DOWN_IDX
            )
            pose = [
                max(45, min(120, STRADDLE_OPEN_BASE[0] + pre_open_amount // 4)),
                t_aux_eff,
                idx_open,
                mid_open,
                STRADDLE_OPEN_BASE[4],
                STRADDLE_OPEN_BASE[5],
            ]
        elif posture == "c_pinch" and two_finger_only:
            pose = [
                max(95, min(165, C_PINCH_OPEN_BASE[0] + pre_open_amount // 3)),
                t_aux,
                max(120, min(195, C_PINCH_OPEN_BASE[2] + pre_open_amount // 3)),
                C_PINCH_OPEN_BASE[3],
                C_PINCH_OPEN_BASE[4],
                C_PINCH_OPEN_BASE[5],
            ]
        elif posture == "c_pinch":
            pose = [
                max(115, min(205, C_PINCH_OPEN_BASE[0] + pre_open_amount // 2)),
                t_aux,
                max(135, min(230, C_PINCH_OPEN_BASE[2] + pre_open_amount // 2)),
                max(135, min(220, 155 + pre_open_amount // 2)),
                C_PINCH_OPEN_BASE[4],
                C_PINCH_OPEN_BASE[5],
            ]
        else:
            pose = [
                max(0, min(300, pre_open_amount // 2)),
                450,
                max(0, min(350, pre_open_amount)),
                max(0, min(350, pre_open_amount)),
                0,
                0,
            ]
        self.move_pose(side, pose, duration_ms=duration_ms)

    def start_gentle_grasp(
        self,
        side: str,
        target_force_N: float = 0.38,
        pre_open_amount: int = 0,
        use_tripod_only: bool = False,
        learned_profile: Optional[dict] = None,
        posture: str = "c_pinch",
        two_finger_only: bool = True,
        thumb_aux: int = 485,
        expected_width_mm: Optional[float] = None,
    ):
        if side not in self.hands:
            raise RuntimeError(f"Hand '{side}' is unavailable")
        hand = self.hands[side]
        st = hand.state()
        hand.current_monitor = CurrentMonitor(hand.guard, hand.max_current)
        self.policies[side] = GentleChocoGraspPolicy(
            st["position"],
            time.monotonic(),
            target_force_N=target_force_N,
            pre_open_amount=pre_open_amount,
            use_tripod_only=use_tripod_only,
            learned_profile=learned_profile,
            posture=posture,
            two_finger_only=two_finger_only,
            thumb_aux=thumb_aux,
            expected_width_mm=expected_width_mm,
        )
        self.last_cmd_target[side] = None

    def start_learn_object(
        self,
        side: str,
        probe_force_N: float = 0.44,
        release_after: bool = False,
    ):
        """現在の指位置をそのまま維持した状態から、指先にセットされたオブジェクトの触覚認識・記憶を開始する。"""
        if side not in self.hands:
            raise RuntimeError(f"Hand '{side}' is unavailable")
        hand = self.hands[side]
        st = hand.state()
        hand.current_monitor = CurrentMonitor(hand.guard, hand.max_current)
        self.policies[side] = InHandObjectLearnPolicy(
            st["position"],
            time.monotonic(),
            probe_force_N=probe_force_N,
            release_after=release_after,
        )
        self.last_cmd_target[side] = None

    def release_gentle_grasp(self, side: str):
        if side in self.policies:
            self.policies[side].command_release(time.monotonic())
        elif side in self.hands:
            self.move_pose(side, OPEN_POSE, duration_ms=1100)

    def tick(self, read_sensors: bool = True) -> dict:
        now = time.monotonic()
        out_hands = {}
        for side, hand in self.hands.items():
            try:
                st = hand.state()
                self.states[side] = st
                if read_sensors and self.meta[side].get("vendor") == 1:
                    self.tactile[side] = parse_capacitive_tactile(hand.read(4, 4200, 30))

                policy = self.policies.get(side)
                if policy is not None:
                    try:
                        if policy.motion_started_at > 0.0 and (now - policy.motion_started_at) >= 0.18:
                            hand.current_monitor.observe(st["current_mA"], now)
                        target = policy.update(
                            now,
                            st["position"],
                            self.tactile.get(side, {}),
                            st["current_mA"],
                        )
                        if (
                            target != self.last_cmd_target[side]
                            and (now - self.last_cmd_sent[side]) >= LIVE_COMMAND_INTERVAL
                        ):
                            if policy.motion_started_at == 0.0 and policy.stage in (
                                "closing", "palpate_approach", "pre_relax"
                            ):
                                policy.motion_started_at = now
                            hand.motion_started = True
                            hand.write(1010, [v for pos in target for v in (pos, 420)])
                            self.last_cmd_target[side] = list(target)
                            self.last_cmd_sent[side] = now
                    except CurrentTrip as trip:
                        hand.write(1034, [0] * 6)
                        policy.stage = "holding" if getattr(policy, "learned_result", None) is None else "memorized_holding"
                        policy.detail = f"電流保護ホールド ({trip})"

                out_hands[side] = {
                    "state": st,
                    "tactile": self.tactile.get(side),
                    "grasp": policy.snapshot() if policy else None,
                }
            except Exception as exc:
                out_hands[side] = {"error": str(exc)}
        for side, m in self.meta.items():
            if side not in out_hands:
                out_hands[side] = {"unavailable": m.get(
                    "unavailable", "disconnected"
                )}
        return out_hands
