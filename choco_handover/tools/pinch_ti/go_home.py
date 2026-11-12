"""右腕を机の縁を避ける経路 (上へ → 体側へ引く → 下ろす) で家の姿勢へ戻し、その場で保持する。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision
b = Bot("go_home")
if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
q = np.array(b.qR()); wrG, RG = fk(GROUND)
if pinch(q)[2] < pinch(GROUND)[2] + 0.07 and q[3] < 1.0:
    qq, _, _ = ik(pinch(q) + np.array([0, 0, 0.09]), RG, q); b.say("up first", b.move(qq, "h_up_first", speed=0.08))
if np.array(b.qR())[3] < 1.0:
    for lab, w, spd in (("h_high", HIGH, 0.12), ("h_back_up", BACK_UP, 0.12), ("h_back_down", BACK_DOWN, 0.12)): b.say(lab, b.move(w, lab, speed=spd))
b.say("h_home", b.move(HOME_R, "h_home", speed=0.10)); b.hand_set(OPEN, 900, 1.2)
p = b.save_hd("home_view"); b.say("stage", b.rob["stage"], "soc", b.rob["bms"]["soc"], "qR", [round(v, 3) for v in b.qR()], "choco", vision.find_choco(__import__("cv2").imread(p), roi=(300, 150, 1100, 560)) if p else None); b.close()
