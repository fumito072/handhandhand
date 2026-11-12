import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *; import vision
b = Bot("head_scan"); b.say("before", b.rob["stage"], "aux", b.rob["q_aux"], "qL", [round(v, 2) for v in b.rob["q_arm"][:5]], "qR", [round(v, 2) for v in b.qR()])
if not b.rob.get("engaged"):
    ok = b.engage(); b.say("engage", ok, b.rob["stage"], repr(b.rob["last_fault"]), "events", [e for e in b.ev if e.get("ok") is False][-2:])
    if not ok: sys.exit(1)
b.pump(1.0); aux0 = list(b.rob["q_aux"]); b.say("aux after engage", aux0)
yaws = [float(v) for v in sys.argv[1:]] or [1.5, 1.0, 0.5, 0.0]
for i, y in enumerate(yaws):
    r = b.move_head([aux0[0], 0.60, y], f"head_{i}", speed=0.15); b.pump(0.8)
    p = b.save_hd(f"scan_yaw_{y:+.2f}"); b.say("yaw cmd", y, r, "aux", [round(v, 3) for v in b.rob["q_aux"]], "choco", vision.find_in_jpg(b.hd) if b.hd else None, p.split("/")[-1])
b.close()
