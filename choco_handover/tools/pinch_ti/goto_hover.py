"""手の形を作り、右腕を机の上空の基準姿勢 (LIFT_TOP) まで運んで HD 画像を保存する。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision
LIFT_TOP = [-0.743, -0.703, 0.982, 0.159, 1.85]
b = Bot("goto_hover"); b.say("start", b.rob["stage"], "soc", b.rob["bms"]["soc"], "qR", [round(v, 3) for v in b.qR()], "hand", b.hand()["pos"])
if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
if not b.rob.get("engaged") and not b.engage(): sys.exit("engage failed")
h = b.hand()
if abs(h["pos"][3] - 846) > 40 or abs(h["pos"][1] - 590) > 25:
    tgt = list(h["pos"])
    for v in range(tgt[3], 850, 40):
        tgt[3] = tgt[4] = tgt[5] = min(847, v + 40); h = b.hand_set(tgt, 400, 0.45)
    b.say("curled", h["pos"], h["cur"])
    if min(h["pos"][3:]) < 780:                                   # 動かなかった指があれば開いて握り直す
        b.hand_set(tgt[:3] + [49, 49, 49], 1200); h = b.hand_set(tgt, 1500, 2.2); b.say("re-curled", h["pos"])
    tgt[0] = 402; tgt[1] = 590; tgt[2] = 49; h = b.hand_set(tgt, 1200, 2.0); b.say("open shape", h["pos"])
    if min(h["pos"][3:]) < 780 or abs(h["pos"][1] - 590) > 25: b.mark("hand_shape_failed", hand=h); sys.exit("hand shape failed")
b.mark("hand_shape", hand=b.hand())
for lab, q, spd in (("go_back_down", BACK_DOWN, 0.15), ("go_back_up", BACK_UP, 0.20), ("go_high", HIGH, 0.20), ("go_lift_top", LIFT_TOP, 0.20)):
    r = b.move(q, lab, speed=spd); b.say(lab, r, "qR", [round(v, 3) for v in b.qR()], "tau_ext", b.snap()["tau_ext"])
    if r != "done": sys.exit("stopped: " + r)
b.pump(1.0); p = b.save_hd(sys.argv[1] if len(sys.argv) > 1 else "hover_view"); b.say("image", p, "choco", vision.find_in_jpg(b.hd), "aux", b.rob["q_aux"], "rpy", b.rob["rpy"]); b.close()
