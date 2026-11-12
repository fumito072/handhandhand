import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision
b = Bot("status"); r = b.rob
b.say("stage", r["stage"], repr(r["last_fault"]), "engaged", r["engaged"], "susp", r["suspended"], "bms", r["bms"], "temp", r["max_temp"], "fsm", r["fsm_id"])
b.say("qL", [round(v, 3) for v in r["q_arm"][:5]]); b.say("qR", [round(v, 3) for v in b.qR()]); b.say("aux", r["q_aux"], "rpy", r["rpy"])
for sd in ("right", "left"): b.say("hand", sd, b.hand(sd), "status", b.hands[sd]["state"]["status"])
p = b.save_hd(sys.argv[1] if len(sys.argv) > 1 else "status"); b.say("image", p, "choco", vision.find_in_jpg(b.hd) if b.hd else None); b.close()
