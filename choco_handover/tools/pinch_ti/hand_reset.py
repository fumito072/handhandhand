import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from bot import *
b = Bot("hand_reset"); b.say("before", b.hand(), "arm", b.rob["stage"], "qR", [round(v, 3) for v in b.qR()])
h = b.hand_set(OPEN, 1200, 2.2); b.say("open", h)
h = b.hand_set(PRE, 1000, 1.8); b.say("pre", h)
ok = abs(h["pos"][2] - PRE[2]) < 12 and abs(h["pos"][0] - PRE[0]) < 12; b.mark("hand_reset", ok=ok, hand=h); b.say("index responds:", ok); b.close()
