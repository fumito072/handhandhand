"""
Revo 2 ハンドの指先触覚・近接・関節エンコーダによる「事前オブジェクト触覚記憶 (Touch-to-Learn)」ストア。
ユーザーが指の間にセットした対象物を優しく触診 (Palpation) して抽出した固有シグネチャを永続化し、
自律把持時の「事前開き姿勢・最適把持力・段ボール誤接触/空振り棄却」に利用する。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import BASE_DIR

PROFILES_PATH = BASE_DIR / "object_profiles.json"


@dataclass
class ObjectTactileProfile:
    profile_id: str
    name: str
    side: str                                    # "right" | "left"
    created_at: str
    start_positions: List[int]                   # ボタン押下時点の6軸エンコーダ値 [0..1000]
    pinch_positions: List[int]                   # 対象保持成立時の6軸エンコーダ値 [0..1000]
    pre_open_positions: List[int]                # 自律把持アプローチ用の推奨事前オープン6軸値
    estimated_span_mm: float                     # エンコーダ幾何から推定した挟み幅 [mm]
    contact_mask: List[bool]                     # 5指の接触有無 [親, 人, 中, 薬, 小]
    active_finger_names: List[str]               # 接触した指の日本語名リスト
    hold_normal_N: List[float]                   # 保持時の5指法線力 [N]
    hold_tangent_N: List[float]                  # 保持時の5指接線力 [N]
    hold_proximity: List[int]                    # 保持時の5指静電容量近接値
    delta_proximity: List[float]                 # 触診前後の近接変化量
    stiffness_N_per_100u: float                  # 接触剛性 ΔF / (Δq/100) [N / 100u]
    hardness_label: str                          # "硬質" | "標準" | "軟質"
    recommended_force_N: float                   # 硬さから自動決定した最適優しさ把持力 [N]
    position_tolerance: int = 125                # 自律把持照合時の許容エンコーダ幅 (±units)


@dataclass
class TactileMemoryState:
    enabled: bool = True
    active_profile_id: Optional[str] = None
    profiles: List[ObjectTactileProfile] = field(default_factory=list)


class ObjectProfileStore:
    def __init__(self, path: Path = PROFILES_PATH):
        self.path = path
        self._lock = threading.Lock()
        self.state = TactileMemoryState()
        self.load()

    def load(self):
        with self._lock:
            if not self.path.exists():
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                profs = []
                for item in raw.get("profiles", []):
                    valid_keys = {k: v for k, v in item.items() if k in ObjectTactileProfile.__dataclass_fields__}
                    profs.append(ObjectTactileProfile(**valid_keys))
                self.state = TactileMemoryState(
                    enabled=bool(raw.get("enabled", True)),
                    active_profile_id=raw.get("active_profile_id"),
                    profiles=profs,
                )
            except Exception:
                pass

    def _save_unlocked(self):
        payload = {
            "enabled": self.state.enabled,
            "active_profile_id": self.state.active_profile_id,
            "profiles": [asdict(p) for p in self.state.profiles],
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def snapshot(self) -> Dict:
        with self._lock:
            active = self._get_active_unlocked()
            return {
                "enabled": self.state.enabled,
                "active_profile_id": self.state.active_profile_id,
                "active_profile": asdict(active) if active else None,
                "profiles": [asdict(p) for p in self.state.profiles],
            }

    def _get_active_unlocked(self) -> Optional[ObjectTactileProfile]:
        if not self.state.profiles:
            return None
        if self.state.active_profile_id:
            for p in self.state.profiles:
                if p.profile_id == self.state.active_profile_id:
                    return p
        return self.state.profiles[0]

    def get_active_for_grasp(self) -> Optional[Dict]:
        with self._lock:
            if not self.state.enabled:
                return None
            act = self._get_active_unlocked()
            return asdict(act) if act else None

    def save_from_hand_result(self, name: str, side: str, learned_dict: Dict) -> ObjectTactileProfile:
        """オンボード InHandObjectLearnPolicy が抽出した辞書データからプロファイルを生成・保存する。"""
        clean_name = (name or "").strip() or "記憶オブジェクト"
        pinch = [int(round(v)) for v in learned_dict["pinch_positions"]]
        start = [int(round(v)) for v in learned_dict.get("start_positions", pinch)]
        # 自律アプローチ時は記憶した対立角 pinch[1] を維持し、屈曲軸に +240 units (約28mm) の余裕を持たせて開く
        pre_open = [
            max(0, pinch[0] - 230),
            max(220, min(750, pinch[1])),
            max(0, pinch[2] - 250),
            max(0, pinch[3] - 250),
            0 if not learned_dict["contact_mask"][3] else max(0, pinch[4] - 240),
            0 if not learned_dict["contact_mask"][4] else max(0, pinch[5] - 240),
        ]
        prof = ObjectTactileProfile(
            profile_id=uuid.uuid4().hex[:8],
            name=clean_name,
            side=side,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            start_positions=start,
            pinch_positions=pinch,
            pre_open_positions=pre_open,
            estimated_span_mm=round(float(learned_dict.get("estimated_span_mm", 30.0)), 1),
            contact_mask=[bool(x) for x in learned_dict["contact_mask"]],
            active_finger_names=list(learned_dict.get("active_finger_names", [])),
            hold_normal_N=[round(float(x), 3) for x in learned_dict["hold_normal_N"]],
            hold_tangent_N=[round(float(x), 3) for x in learned_dict["hold_tangent_N"]],
            hold_proximity=[int(round(x)) for x in learned_dict["hold_proximity"]],
            delta_proximity=[round(float(x), 1) for x in learned_dict["delta_proximity"]],
            stiffness_N_per_100u=round(float(learned_dict.get("stiffness_N_per_100u", 1.0)), 3),
            hardness_label=str(learned_dict.get("hardness_label", "標準")),
            recommended_force_N=round(float(learned_dict.get("recommended_force_N", 0.42)), 2),
            position_tolerance=int(learned_dict.get("position_tolerance", 125)),
        )
        with self._lock:
            # 同名＋同じ手があれば上書き更新してリスト肥大化を防ぎつつ先頭に配置
            self.state.profiles = [
                p for p in self.state.profiles if not (p.name == clean_name and p.side == side)
            ]
            self.state.profiles.insert(0, prof)
            self.state.active_profile_id = prof.profile_id
            self.state.enabled = True
            self._save_unlocked()
        return prof

    def select(self, profile_id: Optional[str] = None, enabled: Optional[bool] = None) -> Dict:
        with self._lock:
            if profile_id is not None:
                self.state.active_profile_id = profile_id
            if enabled is not None:
                self.state.enabled = bool(enabled)
            self._save_unlocked()
        return self.snapshot()

    def delete(self, profile_id: str) -> Dict:
        with self._lock:
            self.state.profiles = [p for p in self.state.profiles if p.profile_id != profile_id]
            if self.state.active_profile_id == profile_id:
                self.state.active_profile_id = self.state.profiles[0].profile_id if self.state.profiles else None
            self._save_unlocked()
        return self.snapshot()
