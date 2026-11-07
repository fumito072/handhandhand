#!/usr/bin/env python3
"""
Phase 3: ChArUco 魚眼カメラ内部/外部キャリブレーション ＆ 手首 AprilTag Hand-Eye バンドル最適化ツール。

使い方:
  1. 印刷用 ChArUco ボードと手首 AprilTag 画像を生成:
       python tools/calibrate_cameras_charuco.py --generate-targets --out-dir calib_targets/
  2. 撮影した Stereo Left / Right / HD 画像群から魚眼内部パラメータと HD⇄Stereo 3D 外部変換を推定:
       python tools/calibrate_cameras_charuco.py --calibrate-cameras --capture-dir calib_captures/
  3. 20〜30 姿勢で記録した手首 AprilTag 観測 + 実測関節角 JSONL から
     カメラ取付姿勢・首ピッチ/ヨー原点オフセット・Hand-Eye 並進を最小二乗バンドル同定し 9点精度 (<3.0mm) を検証:
       python tools/calibrate_cameras_charuco.py --solve-handeye --samples calib_captures/handeye_samples.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CALIB_PATH, SystemConfig
from kinematics import R1Kinematics, rpy_to_mat


def make_charuco_board(squares_x: int = 7, squares_y: int = 5, square_m: float = 0.030, marker_m: float = 0.022):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_m, marker_m, dictionary)
    return dictionary, board


def generate_printable_targets(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dictionary, board = make_charuco_board()
    board_img = board.generateImage((2100, 1500), marginSize=60, borderBits=1)
    cv2.imwrite(str(out_dir / "charuco_7x5_30mm.png"), board_img)

    tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tag_id in (0, 1):
        tag_img = cv2.aruco.generateImageMarker(tag_dict, tag_id, 600)
        padded = cv2.copyMakeBorder(tag_img, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(str(out_dir / f"wrist_apriltag_36h11_id{tag_id}.png"), padded)
    print(f"[ok] Saved printable ChArUco & AprilTag targets to {out_dir}")


def calibrate_fisheye_single(images: List[np.ndarray], board, dictionary) -> Dict:
    detector = cv2.aruco.CharucoDetector(board)
    obj_pts_list, img_pts_list = [], []
    img_shape = None
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        img_shape = (gray.shape[1], gray.shape[0])
        c_corners, c_ids, _, _ = detector.detectBoard(gray)
        if c_corners is not None and c_ids is not None and len(c_ids) >= 8:
            obj_pts, img_pts = board.matchImagePoints(c_corners, c_ids)
            obj_pts_list.append(obj_pts.reshape(-1, 1, 3).astype(np.float64))
            img_pts_list.append(img_pts.reshape(-1, 1, 2).astype(np.float64))

    if len(obj_pts_list) < 6 or img_shape is None:
        raise RuntimeError(f"有効な ChArUco 検出枚数が不足しています ({len(obj_pts_list)}/6 枚)")

    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        obj_pts_list,
        img_pts_list,
        img_shape,
        K,
        D,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    return {
        "width": img_shape[0],
        "height": img_shape[1],
        "rms_px": round(float(rms), 4),
        "K": K.tolist(),
        "D_fisheye": D.ravel().tolist(),
        "valid_views": len(obj_pts_list),
    }


def solve_hand_eye_bundle(samples_jsonl: Path, kin: R1Kinematics, cfg: SystemConfig) -> Dict:
    """
    手首マーカー 3D 観測 (p_cam) と 5軸関節角 (q5, side, head_pitch, head_yaw) のペア列から、
    未知パラメータ x = [dx, dy, dz, d_head_pitch, d_head_yaw, tag_local_x, tag_local_y, tag_local_z]
    を非線形最小二乗法で一括同定し、全姿勢での残差分布 (RMS / Max / 9点合格判定 <=3.0mm) を出力する。
    """
    rows = [
        json.loads(line)
        for line in samples_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 10:
        raise RuntimeError(f"Hand-Eye 同定には最低 10 姿勢以上が必要です (現在 {len(rows)} 姿勢)")

    x0 = np.array(
        [cfg.hand_eye_dx_m, cfg.hand_eye_dy_m, cfg.hand_eye_dz_m, 0.0, 0.0, 0.045, 0.0, 0.032],
        dtype=np.float64,
    )
    lb = np.array([-0.08, -0.08, -0.08, -0.12, -0.12, -0.05, -0.08, -0.08], dtype=np.float64)
    ub = np.array([0.08, 0.08, 0.08, 0.12, 0.12, 0.22, 0.08, 0.08], dtype=np.float64)

    def _residuals(params: np.ndarray) -> np.ndarray:
        dx, dy, dz, dhp, dhy, tx, ty, tz = params
        tag_off = np.array([tx, ty, tz], dtype=np.float64)
        res = []
        for r in rows:
            side = str(r.get("side", "right"))
            q5 = [float(v) for v in r["q5"]]
            hp = float(r["head_pitch"]) + dhp
            hy = float(r["head_yaw"]) + dhy
            p_cam = np.array(r["p_cam_m"], dtype=np.float64)

            T_tc = kin.camera_to_torso_matrix(hp, hy)
            p_obs_torso = (T_tc @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3] + np.array([dx, dy, dz])
            tag_off_side = np.array([tag_off[0], tag_off[1] if side == "right" else -tag_off[1], tag_off[2]])
            p_fk_torso, _ = kin.arm_fk(side, q5, tag_off_side)
            res.extend((p_obs_torso - p_fk_torso).tolist())
        return np.asarray(res, dtype=np.float64)

    sol = least_squares(_residuals, x0, bounds=(lb, ub), loss="huber", f_scale=0.004)
    r_vec = _residuals(sol.x).reshape(-1, 3)
    norms_mm = np.linalg.norm(r_vec, axis=1) * 1000.0
    p90_mm = float(np.percentile(norms_mm, 90))
    rms_mm = float(math.sqrt(np.mean(norms_mm ** 2)))
    passed = bool(p90_mm <= 3.0)

    return {
        "num_poses": len(rows),
        "passed_3mm_criterion": passed,
        "rms_error_mm": round(rms_mm, 2),
        "p90_error_mm": round(p90_mm, 2),
        "max_error_mm": round(float(np.max(norms_mm)), 2),
        "hand_eye_dx_m": round(float(sol.x[0]), 5),
        "hand_eye_dy_m": round(float(sol.x[1]), 5),
        "hand_eye_dz_m": round(float(sol.x[2]), 5),
        "delta_head_pitch_rad": round(float(sol.x[3]), 5),
        "delta_head_yaw_rad": round(float(sol.x[4]), 5),
        "wrist_tag_offset_m": [round(float(v), 5) for v in sol.x[5:8]],
    }


def main():
    ap = argparse.ArgumentParser(description="R1 ChArUco Fisheye & AprilTag Hand-Eye Calibration")
    ap.add_argument("--generate-targets", action="store_true", help="印刷用 ChArUco / AprilTag PNG を生成")
    ap.add_argument("--out-dir", type=Path, default=ROOT_DIR / "calib_targets")
    ap.add_argument("--calibrate-cameras", action="store_true", help="ChArUco 撮影画像から魚眼内部パラメータを推定")
    ap.add_argument("--capture-dir", type=Path, default=ROOT_DIR / "calib_captures")
    ap.add_argument("--solve-handeye", action="store_true", help="AprilTag 姿勢データから Hand-Eye 変換を一括同定")
    ap.add_argument("--samples", type=Path, default=ROOT_DIR / "calib_captures" / "handeye_samples.jsonl")
    ap.add_argument("--apply", action="store_true", help="3mm合格時に calibration.json の hand_eye_d* を自動更新")
    args = ap.parse_args()

    if args.generate_targets:
        generate_printable_targets(args.out_dir)
        return

    if args.calibrate_cameras:
        dictionary, board = make_charuco_board()
        out = {}
        for cam_name in ("left", "right", "hd"):
            paths = sorted((args.capture_dir / cam_name).glob("*.jpg"))
            if paths:
                imgs = [cv2.imread(str(p)) for p in paths]
                imgs = [im for im in imgs if im is not None]
                out[cam_name] = calibrate_fisheye_single(imgs, board, dictionary)
        out_path = args.capture_dir / "camera_intrinsics_fisheye.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if args.solve_handeye:
        kin = R1Kinematics()
        cfg = SystemConfig()
        if CALIB_PATH.exists():
            data = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        res = solve_hand_eye_bundle(args.samples, kin, cfg)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        if args.apply and res["passed_3mm_criterion"]:
            cur = json.loads(CALIB_PATH.read_text(encoding="utf-8")) if CALIB_PATH.exists() else {}
            cur["hand_eye_dx_m"] = res["hand_eye_dx_m"]
            cur["hand_eye_dy_m"] = res["hand_eye_dy_m"]
            cur["hand_eye_dz_m"] = res["hand_eye_dz_m"]
            CALIB_PATH.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[ok] Updated {CALIB_PATH}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
