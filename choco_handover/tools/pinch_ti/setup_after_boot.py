"""電源投入後の準備: 制御を入れる → 首を机へ向ける → 手の形 (3 指握り込み・親指の付け根 590・開き) → 右手を机の上空の高い位置 (HIGH_H) へ。
机が胴体に対して高い配置 (足を着けた状態) 用に、体側で高く引き上げてから前へ出す。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision
BACK_UP_H = [0.35, 0.066, -0.40, -0.90, 0.0]; HIGH_H = [-0.25, 0.079, -0.019, -0.65, 1.426]
b = Bot("setup")
if b.rob.get("stage") in FAULT: b.recover(any_fault=True)
if not b.rob.get("engaged") and not b.engage(): sys.exit("engage failed")
aux = list(b.rob["q_aux"]); b.say("engaged. aux", [round(v, 3) for v in aux], "soc", b.rob["bms"]["soc"])
if abs(aux[2] + 0.12) > 0.05:
    for y in np.linspace(aux[2], -0.12, max(2, int(abs(aux[2] + 0.12) / 0.6) + 1))[1:]: b.move_head([aux[0], 0.60, float(y)], "head_to_table", speed=0.15)
b.say("head", [round(v, 3) for v in b.rob["q_aux"]])
h = b.hand()
if min(h["pos"][3:]) < 780 or abs(h["pos"][1] - 590) > 25:
    tgt = list(h["pos"])
    for v in range(tgt[3], 850, 40): tgt[3] = tgt[4] = tgt[5] = min(847, v + 40); h = b.hand_set(tgt, 400, 0.45)
    if min(h["pos"][3:]) < 780: b.hand_set(tgt[:3] + [49, 49, 49], 1200, 1.8); h = b.hand_set(tgt, 1500, 2.2)
    h = b.hand_set(OPEN, 1200, 2.0)
b.say("hand", h["pos"])
for lab, w, spd in (("s_back_down", BACK_DOWN, 0.15), ("s_back_up_high", BACK_UP_H, 0.15), ("s_high", HIGH_H, 0.15)):
    r = b.move(w, lab, speed=spd); b.say(lab, r, "pinch", np.round(pinch(b.qR()), 3).tolist())
    if r != "done": sys.exit("stopped: " + r)
p = b.save_hd("setup_done"); b.say("image", p, "choco", vision.find_in_jpg(b.hd) if b.hd else None, "rpy", b.rob["rpy"]); b.close()
