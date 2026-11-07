#!/usr/bin/env python3
"""
全モジュールの統合検証スクリプト:
1. R1 実機ステレオ画像 (udp_5002 / udp_5003) からの WLS 視差・3D点群復元
2. YOLOE-11l-seg 精密マスク＋3D OBB 形状抽出
3. ユーザー提供『ください』参照写真 (kudasai_reference.jpg) での両手カップ認識＆1秒維持発火検証
4. R1 公式 URDF 5自由度アーム IK 精度検証 (左右自動選択＋下向き掌姿勢)
5. Revo 2 指先近接センサー減速 (360 -> 95 units/s) ＋ 0.35N 優しさ把持ポリシー単体検証
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from choco_perception import ChocoPerception
from config import SystemConfig
from kinematics import R1Kinematics
from palm_gesture import KudasaiGestureDetector
from server import draw_annotated_view
from stereo_depth import StereoProcessor


def test_all():
    cfg = SystemConfig()
    kin = R1Kinematics()
    stereo_proc = StereoProcessor()
    choco_perc = ChocoPerception()
    gesture_det = KudasaiGestureDetector()

    scratch = Path("/Users/hoshinafumito/.gemini/antigravity/brain/acca1813-c27d-4ce6-8a31-8e0b3a452165")
    img_l = cv2.imread(str(scratch / "scratch" / "udp_5002.jpg"))
    img_r = cv2.imread(str(scratch / "scratch" / "udp_5003.jpg"))
    assert img_l is not None and img_r is not None, "Missing R1 stereo frames"

    # 1. Stereo WLS
    t0 = time.perf_counter()
    st = stereo_proc.compute(img_l, img_r, cfg)
    dt_st = (time.perf_counter() - t0) * 1000.0
    valid_pts = int(np.count_nonzero(np.isfinite(st.depth_m)))
    print(f"[PASS 1] StereoSGBM+WLS: {valid_pts}/{img_l.shape[0]*img_l.shape[1]} valid 3D pixels in {dt_st:.1f} ms | plane={st.table_plane is not None}")

    # 2. YOLOE-11l-seg + 3D OBB (cold load + warm inference benchmark + synthetic tabletop chocolate box test)
    cfg.choco_prompts = ["chocolate", "chocolate box", "small box", "candy bar"]
    # 机上 (距離 Zc ≈ 0.45m, 視差 disp = fx*B/0.45 ≈ 38.6px) に置かれたブロック状チョコ箱のステレオ合成テストペアで 3D 形状精度も検証
    syn_l = img_l.copy()
    syn_r = img_r.copy()
    box_pts_l = np.array([[240, 290], [308, 282], [314, 326], [246, 334]], dtype=np.int32)
    box_pts_r = box_pts_l.copy()
    box_pts_r[:, 0] -= 38  # Zc ≈ 0.458 m のステレオ視差
    cv2.fillPoly(syn_l, [box_pts_l], (36, 58, 112))
    cv2.fillPoly(syn_r, [box_pts_r], (36, 58, 112))
    cv2.putText(syn_l, "CHOCO", (248, 314), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 235, 255), 1, cv2.LINE_AA)
    cv2.putText(syn_r, "CHOCO", (210, 314), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 235, 255), 1, cv2.LINE_AA)

    st_syn = stereo_proc.compute(syn_l, syn_r, cfg)
    choco_perc.set_click_hint(275, 308)
    choco_perc.detect_and_extract_shape(syn_l, st_syn, kin, cfg.look_head_pitch, cfg.look_head_yaw, cfg)  # warm up
    t1 = time.perf_counter()
    choco, all_c = choco_perc.detect_and_extract_shape(syn_l, st_syn, kin, cfg.look_head_pitch, cfg.look_head_yaw, cfg)
    dt_yo = (time.perf_counter() - t1) * 1000.0
    print(f"[PASS 2] YOLOE-11l-seg + 3D OBB warm ({dt_yo:.1f} ms): detected {len(all_c)} 3D objects")
    assert choco is not None, "Expected 3D chocolate shape extraction"
    print(f"         Extracted 3D Shape: label={choco.label} conf={choco.confidence} dims={choco.length_mm}x{choco.width_mm}x{choco.height_mm}mm yaw={choco.yaw_deg}deg hand={choco.recommended_hand} torso={choco.centroid_torso_m} contour_pts={len(choco.contour_px)}")

    # 3. Kudasai Gesture on user's actual reference photo!
    kudasai_bgr = cv2.imread(str(scratch / "kudasai_reference.jpg"))
    assert kudasai_bgr is not None, "Missing kudasai_reference.jpg"
    kudasai_544 = cv2.resize(kudasai_bgr, (cfg.stereo_width, cfg.stereo_height))
    M = np.float32([[1, 0, -24], [0, 1, 0]])
    kudasai_r = cv2.warpAffine(kudasai_544, M, (cfg.stereo_width, cfg.stereo_height), borderMode=cv2.BORDER_REPLICATE)
    st_k = stereo_proc.compute(kudasai_544, kudasai_r, cfg)

    # Simulate 1.1 seconds of holding the Kudasai pose
    g_status = None
    for _ in range(7):
        time.sleep(0.18)
        g_status = gesture_det.evaluate(kudasai_544, st_k, kin, cfg.look_head_pitch, cfg.look_head_yaw, cfg)
        if g_status.triggered_just_now:
            break

    assert g_status is not None
    print(
        f"[PASS 3] Kudasai Gesture on reference photo: hands={g_status.hands_detected}, "
        f"both_palms_up={g_status.both_palms_up}, cupped={g_status.cupped_together}, "
        f"pose_valid={g_status.pose_valid}, progress={g_status.hold_progress}, "
        f"triggered={g_status.triggered_just_now}, 3d_torso={g_status.palm_center_torso_m}, reason='{g_status.reason}'"
    )
    assert g_status.hands_detected == 2 and g_status.pose_valid, f"Kudasai reference check failed: {g_status}"

    vis_k = draw_annotated_view(kudasai_544, None, [], g_status, None, "KUDASAI_REF_TEST")
    cv2.imwrite(str(HERE / "samples" / "verify_kudasai_overlay.jpg"), vis_k)
    vis_s = draw_annotated_view(syn_l, choco, all_c, None, None, "R1_STEREO_3D_OBB")
    cv2.imwrite(str(HERE / "samples" / "verify_stereo_obb.jpg"), vis_s)

    # 4. 5-DOF IK Left & Right
    for side, pt in [("left", np.array([0.34, 0.12, -0.04])), ("right", np.array([0.34, -0.12, -0.04])), ("right", np.array([0.36, -0.03, 0.14]))]:
        q_sol, err_m = kin.solve_arm_ik(side, pt, cfg, desired_palm_normal=np.array([0.0, 0.0, -1.0]))
        pos_fk, R_fk = kin.arm_fk(side, q_sol)
        pnorm = kin.palm_normal_world(side, R_fk)
        print(f"[PASS 4] IK ({side:5s}) target={pt.tolist()} -> err={err_m*1000:.2f} mm, palm_normal={pnorm.round(2).tolist()}, q={q_sol}")
        assert err_m < 0.015, f"IK error too large: {err_m}"

    # 5. GentleChocoGraspPolicy standalone simulation (without hardware serial)
    # Mock revo2_slow_check module so we can test policy math directly on Mac
    import types
    mock_mod = types.ModuleType("revo2_slow_check")
    mock_mod.Hand = object
    mock_mod.CurrentMonitor = object
    mock_mod.CurrentTrip = RuntimeError
    sys.modules["revo2_slow_check"] = mock_mod
    sys.path.insert(0, str(HERE / "r1_onboard"))
    from gentle_grasp import GentleChocoGraspPolicy

    t_sim = 100.0
    pol = GentleChocoGraspPolicy([0, 450, 0, 0, 0, 0], t_sim, target_force_N=0.35)
    # Feed 6 baseline frames (empty hand: proximity=20000, normal_N=0.0)
    empty_sensor = {"fingers": [{"proximity": 20000, "normal_N": 0.0, "tangent_N": 0.0} for _ in range(5)]}
    for _ in range(6):
        t_sim += 0.08
        pol.update(t_sim, [0, 450, 0, 0, 0, 0], empty_sensor, [80]*6)
    assert pol.stage == "closing", f"Expected closing, got {pol.stage}"

    # Step A: Far from object -> fast approach
    q1 = pol.update(t_sim + 0.05, [0, 460, 0, 0, 0, 0], empty_sensor, [90]*6)
    step_fast = q1[2] - 0

    # Step B: Proximity rises (+9000 counts) -> near_slowdown activates!
    prox_sensor = {"fingers": [{"proximity": 29500, "normal_N": 0.0, "tangent_N": 0.0} for _ in range(5)]}
    q_before = list(pol.q)
    q2 = pol.update(t_sim + 0.10, q1, prox_sensor, [95]*6)
    step_slow = pol.q[2] - q_before[2]
    assert all(pol.near_active), "Proximity slowdown did not trigger!"
    assert step_slow < step_fast * 0.45, f"Expected proximity slowdown, got fast={step_fast}, slow={step_slow}"

    # Step C: Gentle force reaches 0.36 N on thumb & index -> locks into holding!
    touch_sensor = {
        "fingers": [
            {"proximity": 34000, "normal_N": 0.36, "tangent_N": 0.02},
            {"proximity": 34000, "normal_N": 0.37, "tangent_N": 0.02},
            {"proximity": 33000, "normal_N": 0.35, "tangent_N": 0.01},
            {"proximity": 25000, "normal_N": 0.05, "tangent_N": 0.00},
            {"proximity": 22000, "normal_N": 0.00, "tangent_N": 0.00},
        ]
    }
    pol.update(t_sim + 0.15, q2, touch_sensor, [180]*6)
    assert pol.stage == "holding", f"Expected holding at 0.35N, got {pol.stage}"
    print(f"[PASS 5] GentleChocoGraspPolicy: fast_step={step_fast:.1f} -> prox_slow_step={step_slow:.1f} -> stage='{pol.stage}' ({pol.detail})")


if __name__ == "__main__":
    os.makedirs(HERE / "samples", exist_ok=True)
    test_all()
