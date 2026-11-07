"""
Unitree R1 チョコ認識・3D形状抽出 ＆「ください」両手手渡しシステム 設定管理
実行中に Web UI からキャリブレーション値・把持力・プロンプトをリアルタイム更新・永続化できる。
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CALIB_PATH = BASE_DIR / "calibration.json"


@dataclass
class SystemConfig:
    # ネットワーク設定
    r1_host: str = "192.168.1.13"
    r1_port: int = 8891

    # 両眼ステレオカメラ (UDP 5002 Left / 5003 Right, 544x448 整準化済み) 実機ティーチング校正値
    stereo_width: int = 544
    stereo_height: int = 448
    fx: float = 132.0          # 実機ティーチング照合済み焦点距離 (px)
    fy: float = 132.0
    cx: float = 274.3
    cy: float = 213.5
    baseline_m: float = 0.050  # R1 頭部ステレオ基線長 50mm (fx*B = 6.60)

    # Hand-Eye 微調整オフセット (Torso = waist_yaw_link 座標系での実機立位校正値 [m])
    # Test#1/#2 HD 1280x720 実測 (Δu=+28px=+27mm, Δv=+8px=+7mm, 天面実測Z=-30.5mm) 反映済み
    hand_eye_dx_m: float = 0.007
    hand_eye_dy_m: float = 0.079
    hand_eye_dz_m: float = 0.041

    # Revo 2 手先ピンチ中心オフセット (wrist_roll_link 原点からの距離 [m])
    tcp_offset_x_m: float = 0.165
    tcp_offset_y_m: float = 0.000
    tcp_offset_z_m: float = 0.000

    # 表面拘束またぎピンチ (Surface-Constrained Straddle Pinch) / C字横ピンチ 設定
    grasp_posture: str = "surface_pinch"   # "surface_pinch" (表面拘束またぎ2指ピンチ推奨) | "c_pinch" | "top_down"
    cpinch_tcp_x_m: float = 0.175          # 親指・人差指の遠位ゴムパッド閉合中心 X (手首カラー28mm込み実機校正値 [m])
    cpinch_tcp_y_m: float = -0.0070        # 掌内側オフセット [m] (親指・人差指遠位パッド対向高さ)
    cpinch_tcp_z_m: float = 0.015          # 橈側オフセット [m] (親指パッドと人差指パッドの真の中点)
    cpinch_side_entry_m: float = 0.044     # (c_pinch時のみ) 対象横タッチオフ後の水平ストローク [m]
    cpinch_surface_lift_m: float = 0.0020  # (c_pinch時のみ) 天面検知後の浮上クリアランス [m]
    surface_relief_lift_m: float = 0.0008  # (surface_pinch時) 着地検知後の微小除荷リフト [m] (+0.8mm: 机面ガイド維持＋摩擦最小化)
    straddle_hover_z_m: float = 0.045      # (surface_pinch時) 対象真上の垂直降下開始ホバー高さ [m] (+45mm)
    enable_waist_yaw_6dof: bool = True     # True: 腰ヨー軸 (±0.22rad) を第6自由度として併用し全方位で関節余裕を最大化
    grav_ff_scale: float = 0.85            # URDF + Revo2 重力補償フィードフォワード係数 [0.0..1.0]
    extra_hand_mass_kg: float = 0.160      # BrainCo Revo2 + アダプタ追加質量 [kg]
    tau_ext_contact_nm: float = 0.68       # 外乱トルクオブザーバ (|tau_est - tau_grav|) による机面接触検知閾値 [Nm]
    two_finger_pinch: bool = False         # False: 親指＋人差指＋中指の3指トライポッド把持 (薬指・小指は退避、捕捉幅2.5倍・転倒防止) / True: 2指のみ
    # 手首ロール目標 (親指→人差指パッド軸の鉛直成分。正 = 人差指側が高い)。旧 C字ピンチは 0.22 (+13°)。
    # IMG_9457/9460 で人差指・中指が親指より 10〜20mm 高かったため、手首を約 +31° 回して親指と指先をほぼ水平 (+2°) に揃える
    pinch_axis_z_goal: float = 0.04
    pinch_pinky_z_min: float = 0.10        # 小指側の最低持ち上げ量 (旧 0.22)。手首を回す分だけ緩和
    enable_hd_vision: bool = True          # True: UDP 5001 (1280x720 HD) 光線×3D平面交差で位置・回転角を高精度抽出

    # 認識プロンプト (YOLOE-seg オープンボキャブラリ)
    choco_prompts: List[str] = field(
        default_factory=lambda: [
            "chocolate",
            "square chocolate",
            "chocolate candy",
            "chocolate box",
            "small box",
            "candy bar",
            "snack package",
            "white bottle cap",
            "white cap",
            "plastic cap",
        ]
    )
    seg_conf_threshold: float = 0.14
    enable_tabletop_depth_fallback: bool = True

    # 頭部の標準見下ろし姿勢 (机と正面の手元が同時に映る角度)
    look_waist_yaw: float = 0.00
    look_head_pitch: float = 0.58
    look_head_yaw: float = 0.00
    # R1 の head_yaw (motor 30) は減速機入力軸エンコーダのため再起動ごとに原点オフセットが変化する。
    # False (推奨): 接続時の q_aux[2] を正面ゼロ点として自動風袋引き (tare) し、電源再投入後の座標回転ズレを防ぐ。
    use_measured_head_yaw: bool = False

    # Revo 2 指先近接・触覚ジェントル把持設定
    gentle_force_N: float = 0.38       # 1指あたりの優しさ把持力 (チロルチョコを潰さず確実に保持する 0.38 N)
    use_tripod_grasp: bool = True      # True: 精密ピンチ (two_finger_pinch=True 時は親指＋人差指の2指のみ)

    # 「ください」両手ジェスチャー発火設定
    auto_gesture_trigger: bool = False  # 安全のため初期は OFF (UIワンクリックでON)
    gesture_hold_seconds: float = 1.0   # 連続維持秒数
    palm_pair_max_gap_m: float = 0.18   # 左右手の小指・手首間の許容距離

    # 実行モード: "grasp_and_hold" (掴んでその場保持・失敗時即停止) | "vision_only" (Dry-Run) | "pick_only" | "full_handover"
    execution_mode: str = "grasp_and_hold"

    # 安全・閉ループ補正オプション
    use_suspended_mode: bool = True    # True: ハーネス宙吊り開発者モード (脚トルク 0Nm で大暴れ物理ゼロ) / False: 立位 FSM 811
    enable_visual_servo: bool = True   # True: 対象上空ホバリング時にハンド指先と対象のズレをカメラで直接計測・微補正

    # 立位バランス安全リーチ範囲 (waist_yaw_link 基準 [m])
    reach_x_min: float = 0.18
    reach_x_max: float = 0.46
    reach_y_abs_max: float = 0.34
    reach_z_min: float = -0.18
    reach_z_max: float = 0.38


class ConfigStore:
    def __init__(self, path: Path = CALIB_PATH):
        self.path = path
        self.lock = threading.Lock()
        self.cfg = SystemConfig()
        self.load()

    def load(self):
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                with self.lock:
                    for k, v in raw.items():
                        if hasattr(self.cfg, k):
                            setattr(self.cfg, k, v)
            except Exception:
                pass

    def save(self):
        with self.lock:
            data = asdict(self.cfg)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def update(self, patch: dict) -> dict:
        with self.lock:
            for k, v in patch.items():
                if hasattr(self.cfg, k):
                    cur = getattr(self.cfg, k)
                    if isinstance(cur, bool):
                        setattr(self.cfg, k, bool(v))
                    elif isinstance(cur, int) and not isinstance(cur, bool):
                        setattr(self.cfg, k, int(v))
                    elif isinstance(cur, float):
                        setattr(self.cfg, k, float(v))
                    elif isinstance(cur, list) and isinstance(v, list):
                        setattr(self.cfg, k, [str(x).strip() for x in v if str(x).strip()])
                    elif isinstance(cur, str):
                        setattr(self.cfg, k, str(v))
            snap = asdict(self.cfg)
        self.save()
        return snap

    def snapshot(self) -> dict:
        with self.lock:
            return asdict(self.cfg)
