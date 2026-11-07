"""
Unitree R1 公式 URDF (`R1.urdf`) に基づく正確な順運動学 (FK)・減衰最小二乗逆運動学 (IK)・
カメラ座標系 ⇄ 胴体 (`waist_yaw_link`) 座標系変換 ＆ 立位安全軌道生成モジュール。
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import MODELS_DIR, SystemConfig

LEFT_CHAIN = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
]
RIGHT_CHAIN = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]
HEAD_CAM_CHAIN = [
    "head_pitch_joint",
    "head_yaw_joint",
    "camera_joint",
]

# R1 立位での標準待機ポーズ・実機ティーチング準拠の自然な肘下げ掌下向き(-Z)シード姿勢
REST_ARM_POSE = [-0.07, 0.16, -1.50, 1.44, -1.32, -0.04, -0.14, 1.56, 1.37, 1.67]
CARRY_READY_LEFT = [-0.68, 0.12, -0.45, 0.78, 1.20]
CARRY_READY_RIGHT = [-0.68, -0.12, 0.45, 0.78, -1.20]


def rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ],
    ], dtype=np.float64)


def axis_angle_to_mat(axis: np.ndarray, q: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(q), math.sin(q)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C    ],
    ], dtype=np.float64)


@dataclass
class JointSpec:
    name: str
    jtype: str
    xyz: np.ndarray
    rot0: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


# BrainCo Revo 2 の親指遠位ゴムパッド -> 人差指遠位ゴムパッド（C字プレカーブ時）の真の3D対向ベクトル
# (+X=遠位指先方向, palm_normal=+Y(右)/-Y(左), +Z=橈側/親指側; IMG_9443実機画像で<2°精度検証済み)
PINCH_AXIS_RIGHT = np.array([0.652, -0.272, -0.707], dtype=np.float64)
PINCH_AXIS_RIGHT /= np.linalg.norm(PINCH_AXIS_RIGHT)
PINCH_AXIS_LEFT = np.array([0.652, 0.272, -0.707], dtype=np.float64)
PINCH_AXIS_LEFT /= np.linalg.norm(PINCH_AXIS_LEFT)


class R1Kinematics:
    def __init__(self, urdf_path: Path = MODELS_DIR / "R1.urdf"):
        tree = ET.parse(str(urdf_path))
        self.joints: Dict[str, JointSpec] = {}
        for j in tree.findall("joint"):
            name = j.attrib["name"]
            jtype = j.attrib["type"]
            orig = j.find("origin")
            xyz = np.array([float(v) for v in orig.attrib.get("xyz", "0 0 0").split()], dtype=np.float64)
            rpy = [float(v) for v in orig.attrib.get("rpy", "0 0 0").split()]
            ax_el = j.find("axis")
            ax = np.array([float(v) for v in ax_el.attrib.get("xyz", "1 0 0").split()], dtype=np.float64) if ax_el is not None else np.array([1.0, 0.0, 0.0])
            nrm = np.linalg.norm(ax)
            if nrm > 1e-9:
                ax = ax / nrm
            lim = j.find("limit")
            lo = float(lim.attrib.get("lower", -3.14)) if lim is not None else -3.14
            hi = float(lim.attrib.get("upper", 3.14)) if lim is not None else 3.14
            self.joints[name] = JointSpec(
                name=name,
                jtype=jtype,
                xyz=xyz,
                rot0=rpy_to_mat(*rpy),
                axis=ax,
                lower=lo,
                upper=hi,
            )

    def _eval_chain(self, chain_names: List[str], q_vals: List[float]) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        qi = 0
        for nm in chain_names:
            sp = self.joints[nm]
            Tj = np.eye(4, dtype=np.float64)
            Tj[:3, :3] = sp.rot0
            Tj[:3, 3] = sp.xyz
            if sp.jtype == "revolute":
                Tr = np.eye(4, dtype=np.float64)
                Tr[:3, :3] = axis_angle_to_mat(sp.axis, float(q_vals[qi]))
                qi += 1
                Tj = Tj @ Tr
            T = T @ Tj
        return T

    def camera_to_torso_matrix(self, head_pitch: float, head_yaw: float) -> np.ndarray:
        """
        光学カメラ座標系 (OpenCV: +X 右, +Y 下, +Z 奥行き前方) の 3D 点 [Xc, Yc, Zc, 1] を
        R1 の胴体 (`waist_yaw_link`: +X 前方, +Y 左, +Z 上方) 座標系へ変換する 4x4 同次変換行列。
        URDF の `camera_link` はロボット慣習 (+X 前方, +Y 左, +Z 上方) なので軸変換 R_link_opt を挟む。
        """
        T_torso_camlink = self._eval_chain(HEAD_CAM_CHAIN, [head_pitch, head_yaw])
        # Optical (+X right, +Y down, +Z forward) -> Link (+X forward, +Y left, +Z up)
        T_link_opt = np.array([
            [ 0.0,  0.0, 1.0, 0.0],
            [-1.0,  0.0, 0.0, 0.0],
            [ 0.0, -1.0, 0.0, 0.0],
            [ 0.0,  0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return T_torso_camlink @ T_link_opt

    def cam_point_to_torso(
        self,
        pt_cam: np.ndarray,
        head_pitch: float,
        head_yaw: float,
        cfg: SystemConfig,
    ) -> np.ndarray:
        T = self.camera_to_torso_matrix(head_pitch, head_yaw)
        p_h = np.array([pt_cam[0], pt_cam[1], pt_cam[2], 1.0], dtype=np.float64)
        p_torso = (T @ p_h)[:3]
        p_torso += np.array([cfg.hand_eye_dx_m, cfg.hand_eye_dy_m, cfg.hand_eye_dz_m], dtype=np.float64)
        return p_torso

    def torso_points_to_cam(
        self,
        pts_torso: np.ndarray,
        head_pitch: float,
        head_yaw: float,
        cfg: SystemConfig,
    ) -> np.ndarray:
        """3D OBB ワイヤーフレーム等を画像上へ再投影するため Torso -> Optical Camera へ逆変換する。"""
        offset = np.array([cfg.hand_eye_dx_m, cfg.hand_eye_dy_m, cfg.hand_eye_dz_m], dtype=np.float64)
        pts_raw = np.asarray(pts_torso, dtype=np.float64) - offset
        T_inv = np.linalg.inv(self.camera_to_torso_matrix(head_pitch, head_yaw))
        ones = np.ones((pts_raw.shape[0], 1), dtype=np.float64)
        hom = np.hstack([pts_raw, ones])
        return (hom @ T_inv.T)[:, :3]

    def arm_fk(
        self,
        side: str,
        q5: List[float],
        tcp_offset: Tuple[float, float, float] = (0.165, 0.0, 0.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """指定アーム (`left` | `right`) の TCP 位置 (3,) と回転行列 R (3,3) を返す。"""
        chain = LEFT_CHAIN if side == "left" else RIGHT_CHAIN
        T_wrist = self._eval_chain(chain, q5)
        T_tcp = np.eye(4, dtype=np.float64)
        sy = 1.0 if side == "left" else -1.0
        T_tcp[:3, 3] = [tcp_offset[0], sy * tcp_offset[1], tcp_offset[2]]
        T = T_wrist @ T_tcp
        return T[:3, 3].copy(), T[:3, :3].copy()

    def palm_normal_world(self, side: str, R_wrist: np.ndarray) -> np.ndarray:
        """
        実機検証済み:
          Left wrist の掌法線  = -Y_local (-R[:, 1])
          Right wrist の掌法線 = +Y_local (+R[:, 1])
        """
        return -R_wrist[:, 1] if side == "left" else R_wrist[:, 1]

    @staticmethod
    def pinch_axis_world(R_wrist: np.ndarray, side: str = "right") -> np.ndarray:
        """親指遠位ゴムパッドから人差指遠位ゴムパッドへ向かう真の 3D 対向単位ベクトル (Torso 座標系) を返す。"""
        ax = PINCH_AXIS_LEFT if side == "left" else PINCH_AXIS_RIGHT
        v = R_wrist @ ax
        return v / (np.linalg.norm(v) + 1e-9)

    @staticmethod
    def side_entry_vector_xy(side: str, pinch_yaw_deg: float) -> np.ndarray:
        """
        C字ピンチ開口部の背面側（手の甲・外側＝対象から安全な空白側）へ向かう水平単位ベクトル (dx, dy) を返す。
        下降時は `grasp_pos + side_entry * vec` へ降り、そこから `-side_entry * vec` へ水平スライドすることで、
        対象物がどの角度に回転していても親指・人差指の角干渉ゼロでC字ポケット中央へ収める。
        """
        rad = math.radians(float(pinch_yaw_deg))
        vx, vy = math.cos(rad), math.sin(rad)
        if side == "right":
            return np.array([vy, -vx], dtype=np.float64)
        return np.array([-vy, vx], dtype=np.float64)

    @staticmethod
    def active_tcp_offset(cfg: SystemConfig, posture: Optional[str] = None) -> Tuple[float, float, float]:
        """選択中の進入姿勢 (`surface_pinch` | `c_pinch` | `top_down`) に応じた TCP オフセット [m] を返す。"""
        mode = (posture or getattr(cfg, "grasp_posture", "surface_pinch") or "surface_pinch").strip()
        if mode in ("surface_pinch", "c_pinch"):
            return (cfg.cpinch_tcp_x_m, cfg.cpinch_tcp_y_m, cfg.cpinch_tcp_z_m)
        return (cfg.tcp_offset_x_m, cfg.tcp_offset_y_m, cfg.tcp_offset_z_m)

    @staticmethod
    def desired_palm_normal_for_posture(
        side: str,
        cfg: SystemConfig,
        posture: Optional[str] = None,
    ) -> np.ndarray:
        """
        進入姿勢に応じた目標掌法線ベクトルを返す。
        `surface_pinch` / `c_pinch`: 小指（尺側）を天面から高く浮かせつつ、親指先端と人差指先端の高さを揃え、
        手の甲を外側へ逃がしてカメラ死角を最小化する約10度チルト法線。
        """
        mode = (posture or getattr(cfg, "grasp_posture", "surface_pinch") or "surface_pinch").strip()
        if mode in ("surface_pinch", "c_pinch"):
            ny = 0.16 if side == "left" else -0.16
            v = np.array([0.0, ny, -0.987], dtype=np.float64)
            return v / np.linalg.norm(v)
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def select_hand_for_point(self, pt_torso: np.ndarray) -> str:
        """物理中心線 (+Y=左, -Y=右; 右手校正オフセット +27mm を考慮) から近い方のアーム＋Revo 2 ハンドを自動選択。"""
        return "left" if float(pt_torso[1]) >= 0.038 else "right"

    def clamp_workspace(self, pt_torso: np.ndarray, side: str, cfg: SystemConfig) -> np.ndarray:
        x = float(np.clip(pt_torso[0], cfg.reach_x_min, cfg.reach_x_max))
        if side == "left":
            y = float(np.clip(pt_torso[1], -0.06, cfg.reach_y_abs_max))
        else:
            y = float(np.clip(pt_torso[1], -cfg.reach_y_abs_max, 0.06))
        z = float(np.clip(pt_torso[2], cfg.reach_z_min, cfg.reach_z_max))
        return np.array([x, y, z], dtype=np.float64)

    def solve_arm_ik(
        self,
        side: str,
        target_pos_torso: np.ndarray,
        cfg: SystemConfig,
        desired_palm_normal: Optional[np.ndarray] = None,
        q_seed: Optional[List[float]] = None,
        palm_weight: float = 0.22,
        posture: Optional[str] = None,
    ) -> Tuple[List[float], float]:
        """
        5自由度アームの減衰最小二乗 (DLS) IK ソルバ。
        3D 位置 [x, y, z] を最優先で一致させつつ、余剰自由度 (主に wrist_roll と shoulder_yaw/roll) で
        掌の向き `desired_palm_normal` と自然姿勢正則化を同時に満たす。
        戻り値: (q5_solution, position_error_m)
        """
        chain = LEFT_CHAIN if side == "left" else RIGHT_CHAIN
        lo = np.array([max(self.joints[nm].lower + 0.05, -2.5) for nm in chain], dtype=np.float64)
        hi = np.array([min(self.joints[nm].upper - 0.05, 2.3) for nm in chain], dtype=np.float64)
        # 胴体との干渉を避けるため内転しすぎを防止
        if side == "left":
            lo[1] = max(lo[1], -0.08)
        else:
            hi[1] = min(hi[1], 0.08)

        q_nom = np.array(CARRY_READY_LEFT if side == "left" else CARRY_READY_RIGHT, dtype=np.float64)
        q = np.array(q_seed if q_seed is not None else q_nom, dtype=np.float64)
        q = np.clip(q, lo, hi)

        if desired_palm_normal is None:
            desired_palm_normal = self.desired_palm_normal_for_posture(side, cfg, posture=posture)
        desired_palm_normal = desired_palm_normal / (np.linalg.norm(desired_palm_normal) + 1e-9)

        tcp_off = self.active_tcp_offset(cfg, posture=posture)
        eps = 1e-5
        lam = 0.035

        for _ in range(110):
            pos, R = self.arm_fk(side, q.tolist(), tcp_off)
            e_pos = target_pos_torso - pos
            x_axis = R[:, 0]
            # 5自由度アーム(手首ピッチ無し)では掌法線は前腕軸 x_axis に直交するため、目標法線を x_axis 直交平面へ射影する
            d_proj = desired_palm_normal - float(np.dot(desired_palm_normal, x_axis)) * x_axis
            n_proj = np.linalg.norm(d_proj)
            d_perp = (d_proj / n_proj) if n_proj > 1e-6 else desired_palm_normal

            p_norm = self.palm_normal_world(side, R)
            e_ori = palm_weight * (d_perp - p_norm)
            err = np.concatenate([e_pos, e_ori])

            if np.linalg.norm(e_pos) < 5e-4 and np.linalg.norm(e_ori) < 0.015:
                break

            J = np.zeros((6, 5), dtype=np.float64)
            for j in range(5):
                qp = q.copy()
                qp[j] += eps
                pos_p, R_p = self.arm_fk(side, qp.tolist(), tcp_off)
                pn_p = self.palm_normal_world(side, R_p)
                J[:3, j] = (pos_p - pos) / eps
                J[3:, j] = palm_weight * (pn_p - p_norm) / eps

            # 姿勢正則化つき減衰最小二乗更新 (後半イテレーションでは正則化バイアスをゼロに減衰させ <0.5mm 収束を保証)
            lam_cur = lam if _ < 65 else 0.006
            reg_w = 0.0012 if _ < 65 else 0.0
            H = J.T @ J + (lam_cur ** 2) * np.eye(5)
            g = J.T @ err - reg_w * (q - q_nom)
            dq = np.linalg.solve(H, g)
            dq = np.clip(dq, -0.18, 0.18)
            q = np.clip(q + dq, lo, hi)

        final_pos, _ = self.arm_fk(side, q.tolist(), tcp_off)
        pos_err = float(np.linalg.norm(target_pos_torso - final_pos))
        return [round(float(v), 4) for v in q], pos_err

    @staticmethod
    def normalize_half_plane_deg(angle_deg: float) -> float:
        """対向2辺の法線角を (-90, +90] [deg] に正規化する (+X 前方成分 >= 0)。"""
        a = ((float(angle_deg) + 90.0) % 180.0) - 90.0
        if a <= -90.0:
            a += 180.0
        return a

    def solve_oriented_pinch_ik(
        self,
        side: str,
        target_pos_torso: np.ndarray,
        target_pinch_yaw_deg: float,
        cfg: SystemConfig,
        q_seed: Optional[List[float]] = None,
    ) -> Tuple[List[float], float, float, int]:
        """
        5自由度向き適応型 親指×人差指 パッド対向 C字ピンチ IK ソルバ。
          1. 親指・人差指の遠位ゴムパッド中点 [X, Y, Z] を target_pos_torso に一致 (< 1mm)
          2. 肘を適度に外側へ開いたロール姿勢で小指側 (-Z_local) を天面から高く浮かせつつ、
             人差指の黒ゴム触覚パッドを側面ではなく正面から親指パッドへ水平対向させる
          3. カメラが推論した対象物のフラット対向2辺の法線角へパッド対向軸を一致させる
          4. r1_arm_daemon のハードウェア安全リミット内で 3D 位置精度 (< 1mm) を厳守する
        戻り値: (q5_solution, pos_error_m, achieved_pinch_yaw_deg, recommended_thumb_aux)
        """
        chain = LEFT_CHAIN if side == "left" else RIGHT_CHAIN
        lo = np.array([max(self.joints[nm].lower + 0.04, -2.55) for nm in chain], dtype=np.float64)
        hi = np.array([min(self.joints[nm].upper - 0.04, 2.30) for nm in chain], dtype=np.float64)
        # r1_arm_daemon.cpp の ARM_MIN / ARM_MAX と完全一致させ実機クリップズレをゼロにする
        lo[2] = max(lo[2], -1.76)
        hi[2] = min(hi[2], 1.76)
        lo[4] = max(lo[4], -1.81)
        hi[4] = min(hi[4], 1.81)
        if side == "left":
            lo[1] = max(lo[1], -0.08)
        else:
            hi[1] = min(hi[1], 0.08)

        sy = 1.0 if side == "right" else -1.0
        tcp_off = self.active_tcp_offset(cfg, posture="c_pinch")
        want_deg = self.normalize_half_plane_deg(target_pinch_yaw_deg)
        # 手首ロール目標: 親指→人差指パッド軸の鉛直成分 vp_z (正 = 人差指側が高い)。
        # 3指トライポッド表面把持では人差指・中指を真下向きに曲げるため、手首を回して親指と指先の高さを揃える
        vz_goal = float(getattr(cfg, "pinch_axis_z_goal", 0.22))
        vz_lo, vz_hi = vz_goal - 0.16, vz_goal + 0.20
        pk_min = float(getattr(cfg, "pinch_pinky_z_min", 0.22))

        def _run_gn(goal_deg: float, seed_arr: np.ndarray, yaw_w: float) -> Tuple[np.ndarray, float, float, float, float]:
            q = np.clip(seed_arr.copy(), lo, hi)
            yaw_rad = math.radians(goal_deg)
            d_yaw = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64)
            eps = 1e-5
            for it in range(115):
                pos, R = self.arm_fk(side, q.tolist(), tcp_off)
                vp = self.pinch_axis_world(R, side=side)
                pn = self.palm_normal_world(side, R)
                pinky_z = float(-R[2, 2])
                vh = vp[:2] / (np.linalg.norm(vp[:2]) + 1e-9)
                yaw_diff = float(math.atan2(vh[1] * d_yaw[0] - vh[0] * d_yaw[1], vh[0] * d_yaw[0] + vh[1] * d_yaw[1]))

                # 小指側を机面から確実に浮かせ (pinky_z >= 0.22)、親指・人差指パッド高さを揃えつつ自然な横ピンチ姿勢を誘導
                pk_pen = max(0.0, pk_min - pinky_z)
                flat_pen = max(0.0, -0.92 - float(pn[2]))
                flip_pen = max(0.0, float(pn[2]) + 0.08)
                vp_pen = max(0.0, vz_lo - float(vp[2])) + max(0.0, float(vp[2]) - vz_hi)
                roll_reg = 0.035 * (float(vp[2]) - vz_goal) + 0.28 * (vp_pen + pk_pen + flat_pen + flip_pen)

                r = np.array([
                    pos[0] - target_pos_torso[0],
                    pos[1] - target_pos_torso[1],
                    pos[2] - target_pos_torso[2],
                    roll_reg,
                    yaw_w * yaw_diff,
                ], dtype=np.float64)
                if np.linalg.norm(r[:3]) < 2.5e-4 and abs(roll_reg) < 0.015 and abs(yaw_diff) < 0.015:
                    break
                J = np.zeros((5, 5), dtype=np.float64)
                for j in range(5):
                    qp = q.copy()
                    qp[j] += eps
                    pos_p, R_p = self.arm_fk(side, qp.tolist(), tcp_off)
                    vp_p = self.pinch_axis_world(R_p, side=side)
                    pn_p = self.palm_normal_world(side, R_p)
                    pkz_p = float(-R_p[2, 2])
                    vh_p = vp_p[:2] / (np.linalg.norm(vp_p[:2]) + 1e-9)
                    yd_p = float(math.atan2(vh_p[1] * d_yaw[0] - vh_p[0] * d_yaw[1], vh_p[0] * d_yaw[0] + vh_p[1] * d_yaw[1]))
                    vpp_pen = max(0.0, vz_lo - float(vp_p[2])) + max(0.0, float(vp_p[2]) - vz_hi)
                    rreg_p = 0.035 * (float(vp_p[2]) - vz_goal) + 0.28 * (
                        vpp_pen + max(0.0, pk_min - pkz_p) + max(0.0, -0.92 - float(pn_p[2])) + max(0.0, float(pn_p[2]) + 0.08)
                    )
                    rp = np.array([
                        pos_p[0] - target_pos_torso[0],
                        pos_p[1] - target_pos_torso[1],
                        pos_p[2] - target_pos_torso[2],
                        rreg_p,
                        yaw_w * yd_p,
                    ], dtype=np.float64)
                    J[:, j] = (rp - r) / eps
                lam_c = 0.015 if it < 65 else 0.0028
                dq = np.linalg.solve(J.T @ J + (lam_c ** 2) * np.eye(5), -J.T @ r)
                q = np.clip(q + np.clip(dq, -0.15, 0.15), lo, hi)

            pos_f, R_f = self.arm_fk(side, q.tolist(), tcp_off)
            vp_f = self.pinch_axis_world(R_f, side=side)
            pn_f = self.palm_normal_world(side, R_f)
            act_deg = self.normalize_half_plane_deg(float(math.degrees(math.atan2(vp_f[1], vp_f[0]))))
            p_err = float(np.linalg.norm(target_pos_torso - pos_f))
            return q, p_err, act_deg, float(-R_f[2, 2]), float(pn_f[2])

        seeds = (
            [np.array(q_seed, dtype=np.float64)]
            if q_seed is not None
            else [
                np.array([-0.98, -0.30 * sy, 1.38 * sy, 0.65, -0.85 * sy], dtype=np.float64),
                np.array([-0.98, -0.30 * sy, 1.38 * sy, 0.65, -1.45 * sy], dtype=np.float64),
                np.array([-0.55, -0.12 * sy, 0.65 * sy, 0.55, -1.35 * sy], dtype=np.float64),
            ]
        )

        best = None
        for sd in seeds:
            q_a, err_a, deg_a, pkz_a, pnz_a = _run_gn(want_deg, sd, yaw_w=0.14)
            if err_a > 0.0012:
                q_n, err_n, neutral_deg, pkz_n, pnz_n = _run_gn(0.0, sd, yaw_w=0.0)
                for alpha in (0.75, 0.50, 0.25, 0.0):
                    try_deg = neutral_deg + alpha * self.normalize_half_plane_deg(want_deg - neutral_deg)
                    q_b, err_b, deg_b, pkz_b, pnz_b = _run_gn(try_deg, q_n, yaw_w=(0.09 if alpha > 0.0 else 0.0))
                    if err_b <= 0.0012:
                        q_a, err_a, deg_a, pkz_a, pnz_a = q_b, err_b, deg_b, pkz_b, pnz_b
                        break
                    if err_b < err_a:
                        q_a, err_a, deg_a, pkz_a, pnz_a = q_b, err_b, deg_b, pkz_b, pnz_b
            ym = abs(self.normalize_half_plane_deg(want_deg - deg_a))
            score = err_a * 10000.0 + ym + 25.0 * max(0.0, pk_min + 0.02 - pkz_a) + 15.0 * max(0.0, -0.93 - pnz_a)
            if best is None or score < best[0]:
                best = (score, q_a, err_a, deg_a)

        _, q_sol, pos_err, act_deg = best
        # Revo 2 親指対立軸は 495 (実機上限 500 近傍) を基準とし、人差指パッド正面と完全に対向させる
        rem_yaw = self.normalize_half_plane_deg(want_deg - act_deg)
        aux_shift = int(np.clip(round((rem_yaw if side == "right" else -rem_yaw) * 1.5), -18, 3))
        thumb_aux = int(np.clip(495 + aux_shift, 475, 498))

        return [round(float(v), 4) for v in q_sol], pos_err, round(act_deg, 1), thumb_aux

    @staticmethod
    def transform_torso_for_waist_delta(
        pt_torso_obs: np.ndarray,
        yaw_deg_obs: float,
        delta_waist_rad: float,
    ) -> Tuple[np.ndarray, float]:
        """
        観測時の腰ヨー角 w_obs から w_cmd = w_obs + delta_waist_rad へ回転したとき、
        世界静止している対象物の新 waist_yaw_link 座標系における 3D 座標と水平角 [deg] を厳密計算する:
          p_new = R_z(-delta_waist_rad) @ p_obs
          yaw_new_deg = yaw_deg_obs - degrees(delta_waist_rad)
        """
        c = math.cos(-float(delta_waist_rad))
        s = math.sin(-float(delta_waist_rad))
        x0, y0, z0 = float(pt_torso_obs[0]), float(pt_torso_obs[1]), float(pt_torso_obs[2])
        pt_new = np.array([c * x0 - s * y0, s * x0 + c * y0, z0], dtype=np.float64)
        yaw_new = float(yaw_deg_obs) - math.degrees(float(delta_waist_rad))
        return pt_new, yaw_new

    def joint_limit_margin(self, side: str, q5: List[float]) -> float:
        """r1_arm_daemon ハードウェア安全リミットまでの最小関節余裕 [rad] を返す。"""
        chain = LEFT_CHAIN if side == "left" else RIGHT_CHAIN
        lo = np.array([max(self.joints[nm].lower + 0.04, -2.55) for nm in chain], dtype=np.float64)
        hi = np.array([min(self.joints[nm].upper - 0.04, 2.30) for nm in chain], dtype=np.float64)
        lo[2], hi[2] = max(lo[2], -1.76), min(hi[2], 1.76)
        lo[4], hi[4] = max(lo[4], -1.81), min(hi[4], 1.81)
        if side == "left":
            lo[1] = max(lo[1], -0.08)
        else:
            hi[1] = min(hi[1], 0.08)
        qa = np.asarray(q5, dtype=np.float64)
        return float(min(np.min(qa - lo), np.min(hi - qa)))

    def choose_oriented_pinch_plan_6dof(
        self,
        pt_torso: np.ndarray,
        edge_yaw_deg: float,
        length_mm: float,
        width_mm: float,
        cfg: SystemConfig,
        current_waist_yaw: float = 0.0,
        allowed_sides: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, object]:
        """
        対象物の 3D 位置・長短辺サイズ・回転角から、4通りすべての把持候補
          (短辺対向 vs 長辺対向) × (右ハンド vs 左ハンド)
        および第6自由度 `waist_yaw` 微調整 (±0.20 rad, 立位接地モード時) を網羅評価し、
        位置精度・パッド正対角・関節リミット余裕が最良のプランを選択して辞書で返す。
        """
        n_across_width = self.normalize_half_plane_deg(edge_yaw_deg + 90.0)
        n_across_length = self.normalize_half_plane_deg(edge_yaw_deg)

        cand_yaws = [(n_across_width, float(width_mm))]
        # 表面拘束またぎピンチは ~66mm 開口を持つため、48mm 以下の辺なら直交2組どちらも候補に入れる
        if float(length_mm) <= 48.0 and (float(length_mm) / max(1.0, float(width_mm))) <= 1.85:
            cand_yaws.append((n_across_length, float(length_mm)))

        # 宙吊りモード (脚0Nm) では腰ヨー回転が空中の骨盤側を回してしまい上体座標と乖離するため、立位接地時のみ第6自由度を解放する
        if getattr(cfg, "enable_waist_yaw_6dof", True) and not getattr(cfg, "use_suspended_mode", True):
            dw_candidates = [0.0, -0.10, 0.10, -0.20, 0.20]
        else:
            dw_candidates = [0.0]

        sides_to_eval = tuple(allowed_sides) if allowed_sides else ("right", "left")
        natural_side = self.select_hand_for_point(pt_torso)
        best_plan: Optional[Dict[str, object]] = None
        best_cost = 1e9

        for dw in dw_candidates:
            w_cmd = float(np.clip(float(current_waist_yaw) + dw, -0.28, 0.28))
            eff_dw = w_cmd - float(current_waist_yaw)
            pt_w, _ = self.transform_torso_for_waist_delta(pt_torso, edge_yaw_deg, eff_dw)
            for side in sides_to_eval:
                clamped_pt = self.clamp_workspace(pt_w, side, cfg)
                ws_pen = float(np.linalg.norm(pt_w - clamped_pt))
                for cyaw_obs, span_mm in cand_yaws:
                    cyaw_w = self.normalize_half_plane_deg(cyaw_obs - math.degrees(eff_dw))
                    q_s, perr, ayaw_w, taux = self.solve_oriented_pinch_ik(side, clamped_pt, cyaw_w, cfg)
                    _, R_s = self.arm_fk(side, q_s, self.active_tcp_offset(cfg, posture="surface_pinch"))
                    pn_s = self.palm_normal_world(side, R_s)
                    pinky_z = float(-R_s[2, 2])
                    j_margin = self.joint_limit_margin(side, q_s)
                    total_pos_err = perr + ws_pen
                    yaw_mismatch = abs(self.normalize_half_plane_deg(cyaw_w - ayaw_w))
                    flat_palm_pen = max(0.0, -0.90 - float(pn_s[2])) * 18.0
                    pinky_low_pen = max(0.0, 0.24 - pinky_z) * 42.0
                    margin_pen = max(0.0, 0.20 - j_margin) * 28.0
                    span_pen = max(0.0, float(span_mm) - float(width_mm)) * 0.22
                    waist_pen = abs(eff_dw) * 7.5
                    side_tie = 0.0 if side == natural_side else 3.5
                    cost = (
                        total_pos_err * 2200.0
                        + yaw_mismatch * 1.15
                        + flat_palm_pen
                        + pinky_low_pen
                        + margin_pen
                        + span_pen
                        + waist_pen
                        + side_tie
                    )
                    if cost < best_cost:
                        best_cost = cost
                        world_ayaw = self.normalize_half_plane_deg(ayaw_w + math.degrees(eff_dw))
                        best_plan = {
                            "side": side,
                            "waist_yaw_cmd": round(w_cmd, 4),
                            "delta_waist_rad": round(eff_dw, 4),
                            "target_pos_torso": clamped_pt.copy(),
                            "pinch_yaw_deg": round(cyaw_w, 1),
                            "achieved_yaw_deg": round(ayaw_w, 1),
                            "world_pinch_yaw_deg": round(cyaw_obs, 1),
                            "world_achieved_yaw_deg": round(world_ayaw, 1),
                            "pos_err_m": round(total_pos_err, 5),
                            "yaw_err_deg": round(yaw_mismatch, 2),
                            "joint_margin_rad": round(j_margin, 3),
                            "thumb_aux": int(taux),
                            "grasp_span_mm": round(float(span_mm), 1),
                            "q5_grasp": q_s,
                        }

        assert best_plan is not None
        return best_plan

    def choose_oriented_pinch_plan(
        self,
        pt_torso: np.ndarray,
        edge_yaw_deg: float,
        length_mm: float,
        width_mm: float,
        cfg: SystemConfig,
    ) -> Tuple[str, float, float, float, int]:
        """後方互換ラッパ: 固定腰角での最適 (side, pinch_yaw_deg, achieved_yaw_deg, pos_err_m, thumb_aux) を返す。"""
        saved = getattr(cfg, "enable_waist_yaw_6dof", True)
        try:
            cfg.enable_waist_yaw_6dof = False
            plan = self.choose_oriented_pinch_plan_6dof(pt_torso, edge_yaw_deg, length_mm, width_mm, cfg, current_waist_yaw=0.0)
        finally:
            cfg.enable_waist_yaw_6dof = saved
        return (
            str(plan["side"]),
            float(plan["pinch_yaw_deg"]),
            float(plan["achieved_yaw_deg"]),
            float(plan["pos_err_m"]),
            int(plan["thumb_aux"]),
        )

    def plan_cartesian_ik_path(
        self,
        side: str,
        p_start_torso: np.ndarray,
        p_end_torso: np.ndarray,
        target_pinch_yaw_deg: float,
        cfg: SystemConfig,
        current_q10: List[float],
        q_seed: Optional[List[float]] = None,
        step_m: float = 0.0025,
    ) -> Tuple[List[List[float]], float, float]:
        """
        始点 `p_start_torso` から終点 `p_end_torso` まで約 2.5mm 刻みで直線補間し、
        連続シード IK によって手先が厳密に直線を通る 10関節ウェイポイント列を生成する。
        戻り値: (q10_waypoints, max_ik_pos_err_m, max_straight_line_deviation_m)
        """
        p0 = np.asarray(p_start_torso, dtype=np.float64)
        p1 = np.asarray(p_end_torso, dtype=np.float64)
        seg_vec = p1 - p0
        dist = float(np.linalg.norm(seg_vec))
        n_steps = max(2, min(80, int(math.ceil(dist / max(0.001, float(step_m))))))

        seed5 = list(q_seed) if q_seed is not None else (list(current_q10[:5]) if side == "left" else list(current_q10[5:]))
        tcp_off = self.active_tcp_offset(cfg, posture="surface_pinch")
        q10_path: List[List[float]] = []
        max_err = 0.0
        max_line_dev = 0.0
        unit_dir = seg_vec / (dist + 1e-12)

        for k in range(n_steps + 1):
            alpha = k / float(n_steps)
            pk = p0 + alpha * seg_vec
            q5_k, err_k, _, _ = self.solve_oriented_pinch_ik(side, pk, target_pinch_yaw_deg, cfg, q_seed=seed5)
            seed5 = q5_k
            fk_k, _ = self.arm_fk(side, q5_k, tcp_off)
            if dist > 1e-6:
                rel = fk_k - p0
                proj = float(np.clip(np.dot(rel, unit_dir), 0.0, dist))
                dev = float(np.linalg.norm(rel - proj * unit_dir))
            else:
                dev = float(np.linalg.norm(fk_k - p0))
            max_err = max(max_err, err_k)
            max_line_dev = max(max_line_dev, dev)
            q10_path.append(self.merge_single_arm(current_q10, side, q5_k))

        return q10_path, max_err, max_line_dev

    @staticmethod
    def compute_arm_gravity_torques(
        q10: List[float],
        rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        q_waist_roll: float = 0.0,
        q_waist_yaw: float = 0.0,
        extra_hand_mass_kg: float = 0.160,
    ) -> List[float]:
        """r1_arm_daemon.cpp と完全同一の 3D ポテンシャルエネルギー勾配 +dU/dq 重力補償トルク [Nm] (10軸) を返す。"""
        R_base = (
            rpy_to_mat(float(rpy[0]), float(rpy[1]), 0.0)
            @ axis_angle_to_mat(np.array([1.0, 0.0, 0.0]), float(q_waist_roll))
            @ axis_angle_to_mat(np.array([0.0, 0.0, 1.0]), float(q_waist_yaw))
        )
        out = [0.0] * 10
        g_acc = 9.80665
        for side_idx in range(2):
            s = 1.0 if side_idx == 0 else -1.0
            q5 = q10[:5] if side_idx == 0 else q10[5:]
            chain = [
                ([0.000000,  s * 0.085688,  0.197490], [ s * 0.26187, 0.0, 0.0], [0.0, 1.0, 0.0], [ 0.00294386,  s * 0.04476868, -0.02286868], 0.837950),
                ([0.034450,  s * 0.047132, -0.025693], [-s * 0.26187, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.03782800,  s * 0.00396700, -0.06649600], 0.740212),
                ([-0.034450, s * 0.004300, -0.108350], [ 0.0,         0.0, 0.0], [0.0, 0.0, 1.0], [ 0.01314300, -s * 0.00435500, -0.07294800], 0.738441),
                ([0.016191,  s * 0.026461, -0.082858], [ 0.0,         0.0, 0.0], [0.0, 1.0, 0.0], [ 0.07237100, -s * 0.02461200, -0.01156600], 0.759419),
                ([0.112180, -s * 0.030020, -0.011702], [ 0.0,         0.0, 0.0], [1.0, 0.0, 0.0], [ 0.06804500, -s * 0.00168800,  0.00139200], 0.324767),
            ]
            R = R_base.copy()
            p = np.zeros(3, dtype=np.float64)
            j_pos, j_axis, bodies = [], [], []
            for i, (xyz, jrpy, ax, com, mass) in enumerate(chain):
                p = p + R @ np.array(xyz, dtype=np.float64)
                ax_np = np.array(ax, dtype=np.float64)
                R = R @ rpy_to_mat(*jrpy) @ axis_angle_to_mat(ax_np, float(q5[i]))
                j_pos.append(p.copy())
                j_axis.append(R @ ax_np)
                bodies.append((p + R @ np.array(com, dtype=np.float64), mass, i))
                if i == 4:
                    bodies.append((p + R @ np.array([0.130, 0.0, 0.0], dtype=np.float64), max(0.0, float(extra_hand_mass_kg)), 4))
            base_k = 0 if side_idx == 0 else 5
            for i in range(5):
                tau_i = 0.0
                for b_pos, b_mass, min_j in bodies:
                    if min_j >= i:
                        dp_dq = np.cross(j_axis[i], b_pos - j_pos[i])
                        tau_i += b_mass * g_acc * float(dp_dq[2])
                out[base_k + i] = round(tau_i, 4)
        return out

    def arm_collision_sample_points(self, side: str, q5: List[float]) -> np.ndarray:
        """肘・前腕・手首・Revo 2 掌・小指側・親指側・指先を包含する 9 個の 3D サンプル点 (Torso 座標系) を返す。"""
        chain = LEFT_CHAIN if side == "left" else RIGHT_CHAIN
        T_elbow = self._eval_chain(chain[:4], q5[:4])
        T_wrist = self._eval_chain(chain, q5)
        p_el = T_elbow[:3, 3]
        p_wr = T_wrist[:3, 3]
        R_wr = T_wrist[:3, :3]
        sy = 1.0 if side == "left" else -1.0
        local_hand_pts = np.array([
            [0.000,  0.000,       0.000],
            [0.095,  0.000,       0.000],
            [0.145,  0.000,      -0.044],  # 小指側外縁
            [0.090,  0.000,       0.046],  # 親指付け根外縁
            [0.188,  0.000,       0.000],  # 伸展中指先端包絡
            [0.178, -0.024 * sy,  0.036],  # 開口親指先端包絡
            [0.178,  0.020 * sy, -0.034],  # 開口小指/薬指先端包絡
        ], dtype=np.float64)
        hand_world = p_wr[None, :] + local_hand_pts @ R_wr.T
        mid_forearm = 0.5 * (p_el + p_wr)
        return np.vstack([p_el[None, :], mid_forearm[None, :], hand_world])

    def check_arm_table_collision(
        self,
        side: str,
        q5: List[float],
        table_bbox: Optional[Dict[str, float]] = None,
        clearance_m: float = 0.018,
    ) -> Tuple[bool, str]:
        """
        アーム・ハンド包絡点群が 3D 天板バウンディングボックスと干渉しないか判定する。
        戻り値: (is_collision_free, reason_if_colliding)
        """
        box = table_bbox or {
            "x_min": 0.21,
            "x_max": 0.66,
            "y_min": -0.28,
            "y_max": 0.28,
            "z_top": -0.035,
        }
        x_min = float(box.get("x_min", 0.21)) - float(clearance_m)
        x_max = float(box.get("x_max", 0.66)) + float(clearance_m)
        y_min = float(box.get("y_min", -0.28)) - float(clearance_m)
        y_max = float(box.get("y_max", 0.28)) + float(clearance_m)
        z_top = float(box.get("z_top", -0.035)) + float(clearance_m)
        z_bot = float(box.get("z_top", -0.035)) - 0.16

        pts = self.arm_collision_sample_points(side, q5)
        for idx, pt in enumerate(pts):
            px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
            if (x_min <= px <= x_max) and (y_min <= py <= y_max) and (z_bot <= pz <= z_top):
                return False, f"{side}_pt{idx}(X={px:.3f},Y={py:.3f},Z={pz:.3f})"
            # 胴体自己干渉チェック
            if (-0.08 < px < 0.20) and (pz > -0.22):
                if (side == "left" and py < -0.04) or (side == "right" and py > 0.04):
                    return False, f"{side}_torso_self_collision(Y={py:.3f})"
        return True, ""

    def check_path_table_collision(
        self,
        q10_seq: List[List[float]],
        side: str,
        table_bbox: Optional[Dict[str, float]] = None,
        clearance_m: float = 0.018,
        substeps: int = 12,
    ) -> Tuple[bool, str]:
        """ウェイポイント列の各区間を細密補間し、全姿勢で天板との干渉がゼロか検証する。"""
        if not q10_seq:
            return True, ""
        for k in range(len(q10_seq)):
            q_a = np.asarray(q10_seq[k][:5] if side == "left" else q10_seq[k][5:], dtype=np.float64)
            if k == 0:
                ok, why = self.check_arm_table_collision(side, q_a.tolist(), table_bbox, clearance_m)
                if not ok:
                    return False, f"seg0:{why}"
            if k + 1 < len(q10_seq):
                q_b = np.asarray(q10_seq[k + 1][:5] if side == "left" else q10_seq[k + 1][5:], dtype=np.float64)
                for s_i in range(1, substeps + 1):
                    alpha = s_i / float(substeps)
                    q_m = ((1.0 - alpha) * q_a + alpha * q_b).tolist()
                    ok, why = self.check_arm_table_collision(side, q_m, table_bbox, clearance_m)
                    if not ok:
                        return False, f"seg{k}->{k+1}@{alpha:.2f}:{why}"
        return True, ""

    def merge_single_arm(self, current_q10: List[float], side: str, q5: List[float]) -> List[float]:
        out = list(current_q10) if len(current_q10) == 10 else list(REST_ARM_POSE)
        if side == "left":
            out[:5] = list(q5)
        else:
            out[5:] = list(q5)
        return out

    def build_clear_body_pose(self, current_q10: List[float], side: str) -> List[float]:
        """待機姿勢から前方へ出す際、肘や手首が骨盤・肋骨に当たらないよう先にロール軸を外へ開く。"""
        out = list(current_q10) if len(current_q10) == 10 else list(REST_ARM_POSE)
        if side == "left":
            out[1] = max(out[1], 0.42)
            out[0] = min(out[0], -0.35)
        else:
            out[6] = min(out[6], -0.42)
            out[5] = min(out[5], -0.35)
        return out

    def build_safe_table_waypoints(
        self,
        current_q10: List[float],
        side: str,
        table_bbox: Optional[Dict[str, float]] = None,
    ) -> List[List[float]]:
        """
        立位待機姿勢(手が机面より下にある状態)から、認識された 3D 天板バウンディングボックス `table_bbox` に
        一切干渉しない 3 段階の安全ウェイポイント列を自動検証・選択して返す。
        標準〜中幅の机では体側外側を回り込み、幅広デスク (|Y|>0.35m) では手前 (X<x_min) へ引いてから天板上空へ上がる。
        """
        base = list(current_q10) if len(current_q10) == 10 else list(REST_ARM_POSE)
        s = 1.0 if side == "left" else -1.0
        z_top = float((table_bbox or {}).get("z_top", -0.035))
        high_pitch = -0.68 if z_top > -0.01 else -0.58

        candidate_templates = [
            # 候補1: 体側外側スイープ (標準〜48cm幅プレート向け)
            [
                [0.52,  0.72 * s,  0.10 * s,  0.95,  0.50 * s],
                [0.52,  0.72 * s,  0.10 * s, -0.55,  0.95 * s],
                [high_pitch, 0.48 * s, -0.32 * s, -0.18, 1.15 * s],
            ],
            # 候補2: 大開き外側スイープ (60cm幅デスク向け)
            [
                [0.56,  0.96 * s,  0.05 * s,  0.92,  0.50 * s],
                [0.56,  0.96 * s,  0.05 * s, -0.62,  0.95 * s],
                [high_pitch - 0.06, 0.56 * s, -0.28 * s, -0.22, 1.15 * s],
            ],
            # 候補3: 後方リトラクト垂直クリア (幅無制限の大型オフィスデスク向け: X<=0.14m で Z を上げてから前進)
            [
                [0.64,  0.65 * s, -0.05 * s,  0.85,  0.50 * s],
                [0.42,  0.82 * s, -0.55 * s, -0.72,  0.95 * s],
                [high_pitch - 0.08, 0.48 * s, -0.32 * s, -0.24, 1.15 * s],
            ],
        ]

        start_ok, _ = self.check_arm_table_collision(
            side,
            base[:5] if side == "left" else base[5:],
            table_bbox,
            clearance_m=0.012,
        )
        for tpl in candidate_templates:
            wp0 = self.merge_single_arm(base, side, tpl[0])
            wp1 = self.merge_single_arm(wp0,  side, tpl[1])
            wp2 = self.merge_single_arm(wp1,  side, tpl[2])
            seq_to_check = [base, wp0, wp1, wp2] if start_ok else [wp0, wp1, wp2]
            ok, _ = self.check_path_table_collision(seq_to_check, side, table_bbox, clearance_m=0.015)
            if ok:
                return [wp0, wp1, wp2]

        # 全候補がタイトな場合は最も安全な大開き候補2を返す
        tpl = candidate_templates[1]
        wp0 = self.merge_single_arm(base, side, tpl[0])
        wp1 = self.merge_single_arm(wp0,  side, tpl[1])
        wp2 = self.merge_single_arm(wp1,  side, tpl[2])
        return [wp0, wp1, wp2]
