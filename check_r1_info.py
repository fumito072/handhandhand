#!/usr/bin/env python3
"""
Unitree R1 EDU に SSH 接続し (有線LAN 192.168.123.164 または Wi-Fi IP)、
全体ファームウェア番号 (Software / Hardware Version / SN)、本体制御基板 (PC1) の全モジュール、
関節アクチュエータ・BMS・MainBoard 情報、および開発ボード (PC4/Jetson) の構成を一括取得します。
"""

import os
import json
import sys
import paramiko

R1_HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.164"
R1_USER = "unitree"
# SSH パスワードはリポジトリに含めない: 実行前に `export R1_SSH_PASS=...` を設定する
R1_PASS = os.environ.get("R1_SSH_PASS", "")

REMOTE_PY = r"""
import json, os, subprocess, time
from dataclasses import dataclass
from cyclonedds.idl import IdlStruct
import cyclonedds.idl.annotations as annotate
from cyclonedds.topic import Topic
from cyclonedds.sub import Subscriber, DataReader
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.qos import Qos, Policy
from cyclonedds.util import duration

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelFactory, ChannelSubscriber
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_, Response_, RequestIdentity_, RequestLease_, RequestPolicy_, RequestHeader_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, BmsState_, MainBoardState_, IMUState_

@dataclass
@annotate.final
@annotate.autoid("sequential")
class RosString(IdlStruct, typename="std_msgs::msg::dds_::String_"):
    data: str

def sh(c):
    try:
        return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""

ChannelFactoryInitialize(0, "eth10")
dp = ChannelFactory()._ChannelFactory__participant

box = {}
s1 = ChannelSubscriber("rt/lf/lowstate", LowState_)
s1.Init(lambda m: box.setdefault("low", m), 1)
s2 = ChannelSubscriber("rt/lf/bmsstate", BmsState_)
s2.Init(lambda m: box.setdefault("bms", m), 1)
s3 = ChannelSubscriber("rt/lf/mainboardstate", MainBoardState_)
s3.Init(lambda m: box.setdefault("mb", m), 1)
s4 = ChannelSubscriber("rt/lf/secondary_imu", IMUState_)
s4.Init(lambda m: box.setdefault("imu2", m), 1)

r_svc = DataReader(Subscriber(dp), Topic(dp, "rt/servicestate", RosString))

rsc = RobotStateClient()
rsc.SetTimeout(2.0)
rsc.Init()
rsc._SetApiVerson("1.0.0.2")
_, api_ver = rsc.GetServerApiVersion()
rsc._Call(1002, "{\"interval\":1,\"duration\":3}")

wqos = Qos(
    Policy.Reliability.Reliable(duration(seconds=10)),
    Policy.Durability.TransientLocal,
    Policy.History.KeepLast(10),
    Policy.Deadline(duration(milliseconds=1)),
)
rqos = Qos(
    Policy.Reliability.Reliable(duration(seconds=10)),
    Policy.Durability.Volatile,
    Policy.History.KeepLast(10),
)
writer = DataWriter(Publisher(dp), Topic(dp, "rt/api/bashrunner/request", Request_), qos=wqos)
reader = DataReader(Subscriber(dp), Topic(dp, "rt/api/bashrunner/response", Response_), qos=rqos)

rt_w = DataWriter(Publisher(dp), Topic(dp, "rt/api/robot_type_service/request", Request_))
rt_r = DataReader(Subscriber(dp), Topic(dp, "rt/api/robot_type_service/response", Response_), qos=rqos)

time.sleep(0.8)

def make_req(req_id, api_id, param):
    return Request_(
        header=RequestHeader_(
            identity=RequestIdentity_(id=req_id, api_id=api_id),
            lease=RequestLease_(id=0),
            policy=RequestPolicy_(priority=0, noreply=False),
        ),
        parameter=param,
        binary=[],
    )

def qbash(req_id, sc):
    writer.write(make_req(req_id, 1001, json.dumps({"script": sc})))
    for _ in range(15):
        time.sleep(0.08)
        for s in reader.take(10):
            if hasattr(s, "header") and s.header.identity.id == req_id:
                try:
                    return json.loads(s.data).get("info")
                except Exception:
                    return s.data
    return None

sw_ver = qbash(7001, "get_software_version.sh")
hw_raw = qbash(7002, "get_hardware_version.sh")
sn_val = qbash(7003, "get_sn.sh")
rf_pow = qbash(7004, "get_rfpower.sh")

hw_ver = None
if hw_raw is not None:
    try:
        n = int(hw_raw)
        hw_ver = f"{n // 10}.{n % 10}"
    except Exception:
        hw_ver = str(hw_raw)

rt_w.write(make_req(8001, 1001, "{\"version\":\"\"}"))
machine_type = None
for _ in range(10):
    time.sleep(0.08)
    for s in rt_r.take(10):
        if hasattr(s, "data") and s.data:
            try:
                machine_type = json.loads(s.data).get("robot_type")
            except Exception:
                machine_type = s.data
            break
    if machine_type is not None:
        break

svc_list = []
for _ in range(15):
    for s in r_svc.take(10):
        if hasattr(s, "data") and s.data:
            svc_list = json.loads(s.data)
            break
    if svc_list:
        break
    time.sleep(0.1)

low = box.get("low")
bms = box.get("bms")
mb = box.get("mb")
imu2 = box.get("imu2")

active_motors = []
if low:
    for idx, m in enumerate(low.motor_state):
        if m.mode != 0 or m.vol > 0 or any(m.reserve):
            active_motors.append({
                "idx": idx,
                "mode": int(m.mode),
                "vol": round(float(m.vol), 1),
                "motorstate": int(m.motorstate),
                "fw_rev": int(m.reserve[2]),
            })

pc4_pkg = {}
try:
    with open("/unitree/robot/pkg/version/version.json") as f:
        pc4_pkg = json.load(f)
except Exception:
    pass

out = {
    "identity_and_overall": {
        "serial_number": sn_val,
        "machine_type": machine_type,
        "software_version": sw_ver,
        "hardware_version": hw_ver,
        "hardware_version_raw": hw_raw,
        "protocol_generation": "V3",
    },
    "subsystems": {
        "bms_firmware": f"{bms.version_high}.{bms.version_low}" if bms else None,
        "bms_fn": int(bms.fn) if bms else None,
        "bms_cells": len([c for c in bms.cell_vol if c > 0]) if bms else None,
        "bms_soh_percent": int(bms.soh) if bms else None,
        "bms_cycles": int(bms.cycle) if bms else None,
        "battery_soc_percent": int(bms.soc) if bms else None,
        "mainboard_state": list(mb.state) if mb else None,
        "mode_machine": int(low.mode_machine) if low else None,
        "dual_imu_present": bool(low and imu2),
        "rf_clicker_enabled": rf_pow == "1",
        "active_motor_count": len(active_motors),
        "motors": active_motors,
    },
    "pc4_dev_board": {
        "model": sh("tr -d '\\0' < /proc/device-tree/model"),
        "tnspec": sh("awk '/^TNSPEC/{print $2}' /etc/nv_boot_control.conf"),
        "os": sh("lsb_release -ds"),
        "kernel": sh("uname -r"),
        "l4t_release": sh("head -n 1 /etc/nv_tegra_release"),
        "ota_pipe_version": sh("cat /unitree/ota/pipe/version"),
        "pc4_modules": pc4_pkg.get("Module", {}),
    },
    "pc1_control_board": {
        "robot_state_api_version": api_ver,
        "services": svc_list,
    },
}
print(json.dumps(out, ensure_ascii=False, indent=2))
"""


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(R1_HOST, username=R1_USER, password=R1_PASS, timeout=8)

    sftp = ssh.open_sftp()
    with sftp.file("/tmp/.r1_check_info.py", "w") as f:
        f.write(REMOTE_PY)
    sftp.close()

    run_cmd = (
        "LD_LIBRARY_PATH=/home/unitree/cyclonedds_ws/install/cyclonedds/lib:$LD_LIBRARY_PATH "
        "CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds "
        "python3 /tmp/.r1_check_info.py && rm -f /tmp/.r1_check_info.py"
    )
    _, stdout, stderr = ssh.exec_command(run_cmd, timeout=25)
    print(stdout.read().decode("utf-8", errors="replace"))
    err = stderr.read().decode("utf-8", errors="replace").strip()
    if err:
        print("STDERR:", err, file=sys.stderr)
    ssh.close()


if __name__ == "__main__":
    main()
