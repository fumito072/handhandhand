"""R1 右腕「親指×人差指つまみ」の自律試行用ヘルパー。

ブリッジ (TCP :8891) へ 1 本の接続を張り、腕・指の操作と記録をまとめる。
- 中断 (SIGINT/SIGTERM) されたら腕をその場で保持する。
- ブリッジのハートビート途絶による安全保持は、その場で制御を入れ直して続行する。
- すべての操作を docs/data/auto_grasp_20261112/events.jsonl に記録する。
リポジトリのルートをカレントディレクトリにして使う。
"""
import socket, struct, json, time, sys, signal, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "choco_handover")); sys.path.insert(0, str(ROOT / "choco_handover" / "tools"))
import numpy as np
import kinematics as K
import grasp_thumb_middle as tm

HOST = ("192.168.123.164", 8891)
FAULT = ("fault_hold", "damped_estop", "fault_stale_lowstate")
OUT = str(ROOT / "docs" / "data" / "auto_grasp_20261112")
TIP = np.array([0.16, 0.0, 0.03])                       # 手首から見たつまみ点のおおよその位置
HOME_L = [-0.0117, 0.0143, 0.6805, 1.4939, -0.1422]
HOME_R = [-0.0788, 0.0662, -0.8067, 1.3471, -1.3485]
BACK_DOWN = [0.35, 0.066, -0.807, 1.347, -1.3485]       # 体側へ引いたまま肘を伸ばした姿勢 (机の縁を避ける)
BACK_UP = [0.35, 0.066, -0.40, -0.45, 0.0]
HIGH = [-0.20, 0.079, -0.019, -0.45, 1.426]
GROUND = [-0.618, -0.403, 0.914, 0.595, 1.67]           # 机に接地した開く直前の実測姿勢 (手で教示)
OPEN = [402, 590, 49, 846, 847, 846]
PRE = [581, 590, 137, 846, 847, 846]
GRASP = [641, 590, 197, 846, 847, 846]
kin = K.R1Kinematics()

def fk(q5):
    wr, R = kin.arm_fk("right", list(q5), (0, 0, 0)); return np.asarray(wr), np.asarray(R)
def pinch(q5):
    wr, R = fk(q5); return wr + R @ TIP
def ik(p, R, seed, iters=200):
    q, pe, re = tm.ik_pose(kin, np.asarray(p, dtype=float), R, np.asarray(seed, dtype=float), tcp=TIP, iters=iters, w_rot_rad=0.05)
    return [float(v) for v in q], float(pe), float(re)

class Bot:
    def __init__(self, tag):
        self.tag = tag; self.s = socket.create_connection(HOST, timeout=5); self.s.settimeout(0.3)
        self.buf = b""; self.rob = {}; self.hands = {}; self.hd = None; self.stereo = None; self.ev = []
        for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sg, self._halt)
        self.pump(1.2); t0 = time.time()
        while "q_arm" not in self.rob and time.time() - t0 < 8: self.pump(0.4)          # 腕の状態が届くまで待つ
        self.left = list(self.rob["cmd_arm"][:5]) if self.rob.get("engaged") else list(self.rob["q_arm"][:5])
        self.rpy0 = list(self.rob["rpy"])
    def _halt(self, sig, frm):
        try: self.send("arm_hold_here"); self.mark("interrupted_hold")
        finally: os._exit(2)
    def pump(self, dur):
        t0 = time.time()
        while time.time() - t0 < dur:
            try: c = self.s.recv(1 << 18)
            except socket.timeout: continue
            if not c: raise EOFError("bridge closed")
            self.buf += c
            while len(self.buf) >= 5:
                ln, mt = struct.unpack(">IB", self.buf[:5])
                if len(self.buf) < 4 + ln: break
                p, self.buf = self.buf[5:4 + ln], self.buf[4 + ln:]
                if mt == 1:
                    m = json.loads(p)
                    if m.get("type") == "telemetry": self.rob = m.get("robot") or self.rob; self.hands = m.get("hands") or self.hands
                    elif m.get("type") == "arm_event": self.ev.append(m["data"])
                elif mt == 3: self.hd = p[8:]
                elif mt == 2:
                    n = struct.unpack(">dI", p[:12])[1]; self.stereo = (p[12:12 + n], p[12 + n:])
    def send(self, method, **kw):
        body = json.dumps({"method": method, "id": f"b_{int(time.time()*1000)%100000}", **kw}).encode()
        self.s.sendall(struct.pack(">IB", len(body) + 1, 1) + body)
    def hand(self, side="right"):
        h = self.hands[side]; f = h["tactile"]["fingers"]
        return {"pos": h["state"]["position"], "cur": [round(v) for v in h["state"]["current_mA"]], "Fn": [f[i]["normal_N"] or 0 for i in range(5)],
                "Ft": [f[i]["tangent_N"] or 0 for i in range(5)], "prox_k": [int((f[i]["proximity"] or 0) / 1000) for i in range(5)]}
    def qR(self): return list(self.rob["q_arm"][5:])
    def snap(self):
        q = self.qR(); p = pinch(q)
        return {"qR": [round(v, 4) for v in q], "cmdR": [round(v, 4) for v in self.rob["cmd_arm"][5:]], "qL": [round(v, 4) for v in self.rob["q_arm"][:5]],
                "tau_ext": [round(v, 2) for v in self.rob["tau_ext"][5:]], "pinch": [round(float(v), 4) for v in p], "rpy": self.rob["rpy"],
                "soc": self.rob["bms"]["soc"], "temp": self.rob["max_temp"], "stage": self.rob.get("stage"), "hand": self.hand()}
    def mark(self, name, **kw):
        rec = {"t": round(time.time(), 3), "tag": self.tag, "event": name, **kw}
        with open(OUT + "/events.jsonl", "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    def say(self, *a): print(*a, flush=True)
    def grab_hd(self):
        self.hd = None; t0 = time.time()
        while self.hd is None and time.time() - t0 < 4: self.pump(0.2)
        return self.hd
    def save_hd(self, name):
        jpg = self.grab_hd()
        if jpg: open(f"{OUT}/{name}.jpg", "wb").write(jpg)
        return f"{OUT}/{name}.jpg" if jpg else None
    def engage(self):
        self.send("arm_engage", suspended=1.0, grav_ff_scale=0.85, extra_hand_mass_kg=0.16, table_z_floor=-0.32)
        t0 = time.time()
        while time.time() - t0 < 14:
            self.pump(0.3)
            if self.rob.get("engaged") and self.rob.get("stage") == "holding" and not self.rob.get("last_fault"):
                self.left = list(self.rob["cmd_arm"][:5]); return True
        return False
    def recover(self, any_fault=False):
        f = self.rob.get("last_fault")
        if f != "Bridge heartbeat timeout" and not any_fault: return False
        self.mark("fault_recover", fault=f); return self.engage()
    def sway(self):
        r = self.rob["rpy"]; return max(abs(r[0] - self.rpy0[0]), abs(r[1] - self.rpy0[1]))
    def move(self, q5, label, speed=0.15, guard=None, left=None, timeout=120):
        """右腕の関節移動。'done' か、guard が返した理由 (その場で保持) を返す。"""
        if self.rob.get("stage") in FAULT and not self.recover(): raise RuntimeError(f"arm fault: {self.rob.get('last_fault')}")
        if not self.rob.get("engaged"): raise RuntimeError("arm not engaged")
        if left is not None: self.left = list(left)
        q10 = self.left + [float(v) for v in q5]; self.mark(label + "_start", target=q10, **self.snap())
        for attempt in range(6):
            n_ev = len(self.ev)
            self.send("arm_move", q=q10, duration=1.5, label=label, replace=True, max_speed=float(speed), table_z_floor=-0.32)
            t0, seen, froze = time.time(), False, False
            while time.time() - t0 < timeout:
                self.pump(0.1); st = self.rob.get("stage")
                if st in FAULT: froze = True; break
                if not self.rob.get("engaged"): raise RuntimeError("disengaged during move")
                if st == label: seen = True
                if any(e.get("event") == "motion_complete" and e.get("label") == label for e in self.ev[n_ev:]): seen = True
                if self.sway() > 0.08:
                    self.send("arm_hold_here"); self.mark(label + "_sway_hold", **self.snap()); return "body sway"
                why = guard() if guard else None
                if why:
                    self.send("arm_hold_here"); self.pump(0.4); self.mark(label + "_guard", why=why, **self.snap()); return why
                if seen and st == "holding":
                    self.pump(0.3); self.mark(label + "_done", **self.snap()); return "done"
                if not seen and time.time() - t0 > 5: raise RuntimeError(f"move {label} not started: {self.ev[-2:]}")
            else: raise RuntimeError(f"move {label} timeout")
            if froze and not self.recover(): raise RuntimeError(f"arm fault during {label}: {self.rob.get('last_fault')}")
        raise RuntimeError("too many freezes")
    def hand_set(self, targets, ms=800, settle=None, side="right"):
        self.send("hand_pose", side=side, targets=[int(v) for v in targets], duration_ms=int(ms))
        self.pump(settle if settle is not None else ms / 1000.0 + 0.5); return self.hand(side)
    def close(self): self.s.close()

def _move_head(self, aux, label, speed=0.15, timeout=60):
    """腰ヨー・首ピッチ・首ヨー (aux) だけを動かす。腕は今の指令のまま。"""
    if self.rob.get("stage") in FAULT and not self.recover(): raise RuntimeError(f"arm fault: {self.rob.get('last_fault')}")
    q10 = list(self.rob["cmd_arm"]); self.mark(label + "_start", aux=aux, q_aux=self.rob["q_aux"])
    for attempt in range(4):
        n_ev = len(self.ev)
        self.send("arm_move", q=q10, aux=[float(v) for v in aux], duration=1.5, label=label, replace=True, max_speed=float(speed), table_z_floor=-0.32)
        t0, seen, froze = time.time(), False, False
        while time.time() - t0 < timeout:
            self.pump(0.1); st = self.rob.get("stage")
            if st in FAULT: froze = True; break
            if st == label or any(e.get("event") == "motion_complete" and e.get("label") == label for e in self.ev[n_ev:]): seen = True
            if seen and st == "holding": self.pump(0.3); self.mark(label + "_done", q_aux=self.rob["q_aux"]); return "done"
            if not seen and time.time() - t0 > 5: raise RuntimeError(f"head move {label} not started: {self.ev[-2:]}")
        else: raise RuntimeError("head move timeout")
        if froze and not self.recover(): raise RuntimeError(f"fault during {label}: {self.rob.get('last_fault')}")
    raise RuntimeError("too many freezes")
Bot.move_head = _move_head

# --- 机に対する手の傾きを教示どおりに保つ逆運動学 (水平方向の回転だけ自由) ---
_wrG, _RG = fk(GROUND); _TILT_G = _RG.T @ np.array([0.0, 0.0, 1.0])       # 教示姿勢で、手から見た「真上」の向き
def ik_tilt(p, seed, iters=200):
    """つまみ点の位置 p を厳密に、手の傾き (下向き角と捻り) を教示どおりに合わせる。戻り値: (q, 位置誤差mm, 傾き誤差deg)。"""
    p = np.asarray(p, dtype=float); q = np.clip(np.asarray(seed, dtype=float), tm.Q_MIN, tm.Q_MAX)
    def err(qv):
        wr, R = fk(qv); return np.concatenate([(wr + R @ TIP - p) / 0.001, (R.T @ np.array([0.0, 0.0, 1.0]) - _TILT_G) / 0.004])
    lam = 1e-2
    for _ in range(iters):
        e = err(q); J = np.zeros((6, 5))
        for j in range(5):
            d = np.zeros(5); d[j] = 1e-5; J[:, j] = (err(np.clip(q + d, tm.Q_MIN, tm.Q_MAX)) - e) / 1e-5
        step = np.linalg.solve(J.T @ J + lam * np.eye(5), -J.T @ e); qn = np.clip(q + np.clip(step, -0.1, 0.1), tm.Q_MIN, tm.Q_MAX)
        if np.linalg.norm(err(qn)) < np.linalg.norm(e): q, lam = qn, max(lam * 0.5, 1e-5)
        else: lam *= 4.0
        if np.linalg.norm(step) < 1e-6: break
    wr, R = fk(q); t = R.T @ np.array([0.0, 0.0, 1.0])
    return [float(v) for v in q], float(np.linalg.norm(wr + R @ TIP - p) * 1000), float(np.degrees(np.arccos(np.clip(t @ _TILT_G, -1, 1))))
def tilt_err_deg(q5):
    wr, R = fk(q5); return float(np.degrees(np.arccos(np.clip((R.T @ np.array([0.0, 0.0, 1.0])) @ _TILT_G, -1, 1))))
