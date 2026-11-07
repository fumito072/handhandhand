#!/usr/bin/env python3
"""
Unitree R1 EDU (Jetson 192.168.123.164) — SSH Wi-Fi Setup Server
Unitree アカウント不要！有線 LAN (192.168.123.164) 経由で R1 の Wi-Fi を Mac と同じルーターに接続します。
USB Wi-Fi ドングルの装着状態チェック機能付き。
"""

import os
import json
import logging
import shlex
import subprocess
import time
from pathlib import Path

import paramiko
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("r1_ssh_wifi")

R1_ETH_HOST = "192.168.123.164"
R1_USER = "unitree"
# SSH パスワードはリポジトリに含めない: 実行前に `export R1_SSH_PASS=...` を設定する
R1_PASS = os.environ.get("R1_SSH_PASS", "")
RESULT_FILE = Path(__file__).parent / "latest_setup_result_2.json"

progress_steps: list[str] = []


def add_step(msg: str) -> None:
    log.info(msg)
    progress_steps.append(msg)


def ssh_connect() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(R1_ETH_HOST, username=R1_USER, password=R1_PASS, timeout=6)
    return c


def ssh_run(client: paramiko.SSHClient, cmd: str, sudo: bool = False) -> tuple[int, str, str]:
    # Avoid login shell (-l) because ~/.bashrc on some R1 units prompts interactively with `read choose`
    if sudo:
        full_cmd = f"echo {shlex.quote(R1_PASS)} | sudo -S -p '' bash -c {shlex.quote(cmd)}"
    else:
        full_cmd = f"bash -c {shlex.quote(cmd)}"
    _, stdout, stderr = client.exec_command(full_cmd, timeout=30)
    rc = stdout.channel.recv_exit_status()
    return (
        rc,
        stdout.read().decode("utf-8", errors="replace").strip(),
        stderr.read().decode("utf-8", errors="replace").strip(),
    )


class WifiReq(BaseModel):
    ssid: str
    password: str


app = FastAPI(title="Unitree R1 Wi-Fi Setup (SSH)")


@app.get("/api/scan")
def api_scan():
    try:
        c = ssh_connect()
        _, wifi_dev, _ = ssh_run(
            c, "nmcli -t -f DEVICE,TYPE device status | awk -F: '$2==\"wifi\"{print $1; exit}'"
        )
        _, lsusb_out, _ = ssh_run(c, "lsusb")
        if not wifi_dev:
            c.close()
            return {
                "ok": True,
                "has_wifi_dev": False,
                "wifi_dev": "",
                "wlan0": "",
                "lsusb": lsusb_out,
                "networks": [],
            }

        ssh_run(c, "rfkill unblock wifi || true; nmcli radio wifi on; nmcli device wifi rescan || true", sudo=True)
        time.sleep(1.5)
        _, out, _ = ssh_run(c, "nmcli -t -f IN-USE,SSID,CHAN,RATE,SIGNAL,SECURITY dev wifi list")
        _, cur_ip, _ = ssh_run(c, f"ip -4 -br addr show {shlex.quote(wifi_dev)}")
        c.close()

        networks = []
        seen = set()
        for line in out.splitlines():
            parts = line.replace("\\:", "__COLON__").split(":")
            if len(parts) >= 6:
                in_use = parts[0] == "*"
                ssid = parts[1].replace("__COLON__", ":").strip()
                if not ssid or ssid in seen:
                    continue
                seen.add(ssid)
                networks.append({
                    "in_use": in_use,
                    "ssid": ssid,
                    "chan": parts[2],
                    "rate": parts[3],
                    "signal": parts[4],
                    "security": parts[5],
                })
        return {
            "ok": True,
            "has_wifi_dev": True,
            "wifi_dev": wifi_dev,
            "wlan0": cur_ip,
            "lsusb": lsusb_out,
            "networks": networks,
        }
    except Exception as e:
        return {"ok": False, "has_wifi_dev": False, "error": str(e), "networks": []}


@app.get("/api/progress")
def api_progress():
    return {"steps": progress_steps}


@app.post("/api/connect")
def api_connect(req: WifiReq):
    progress_steps.clear()
    ssid = req.ssid.strip()
    psk = req.password

    if not ssid:
        return JSONResponse({"success": False, "error": "SSID を入力してください。"}, status_code=400)

    try:
        add_step(f"1. R1 開発ボード ({R1_ETH_HOST}) に SSH 接続しています...")
        c = ssh_connect()
        add_step("   SSH 接続成功！(Ubuntu / Jetson)")

        add_step("2. 無線インターフェース (USB Wi-Fi ドングル) を確認しています...")
        _, wifi_dev, _ = ssh_run(
            c, "nmcli -t -f DEVICE,TYPE device status | awk -F: '$2==\"wifi\"{print $1; exit}'"
        )
        if not wifi_dev:
            c.close()
            raise RuntimeError(
                "R1 側に Wi-Fi インターフェースが見つかりません。1台目と同様に USB Wi-Fi ドングル（TP-Link 等）を R1 の USB ポートに挿してから再実行してください。"
            )
        add_step(f"   無線デバイス検出: {wifi_dev}")

        ssh_run(c, "rfkill unblock wifi || true; nmcli radio wifi on", sudo=True)

        add_step(f"3. Wi-Fi プロファイル '{ssid}' を作成・最優先自動接続に設定中...")
        ssh_run(c, f"nmcli connection delete {shlex.quote(ssid)} || true", sudo=True)

        add_cmd = (
            f"nmcli connection add type wifi con-name {shlex.quote(ssid)} "
            f"ifname {shlex.quote(wifi_dev)} ssid {shlex.quote(ssid)} -- "
            f"wifi-sec.key-mgmt wpa-psk "
            f"connection.autoconnect yes "
            f"connection.autoconnect-priority 50 "
            f"ipv4.method auto ipv4.route-metric 100 ipv6.method auto"
        )
        rc, out, err = ssh_run(c, add_cmd, sudo=True)
        if rc != 0:
            c.close()
            raise RuntimeError(f"プロファイル作成失敗: {err or out}")

        if psk:
            rc, out, err = ssh_run(
                c,
                f"nmcli connection modify {shlex.quote(ssid)} wifi-sec.psk {shlex.quote(psk)}",
                sudo=True,
            )
            if rc != 0:
                c.close()
                raise RuntimeError(f"パスワード設定失敗: {err or out}")

        add_step(f"4. '{ssid}' へ接続しています...")
        rc, out, err = ssh_run(c, f"nmcli -w 25 connection up {shlex.quote(ssid)}", sudo=True)
        if rc != 0:
            c.close()
            raise RuntimeError(f"Wi-Fi 接続に失敗しました（パスワード等を確認してください）: {err or out}")

        time.sleep(2.0)
        _, ip_line, _ = ssh_run(c, f"ip -4 -br addr show {shlex.quote(wifi_dev)}")
        wifi_ip = ""
        parts = ip_line.split()
        if len(parts) >= 3:
            wifi_ip = parts[2].split("/")[0]

        _, eth_line, _ = ssh_run(c, "ip -4 -br addr show eth10")
        rc_pc1, _, _ = ssh_run(c, "ping -c 1 -W 2 192.168.123.161")
        c.close()

        if not wifi_ip:
            raise RuntimeError(f"接続は完了しましたが IPv4 アドレスが取得できませんでした: {ip_line}")

        add_step(f"   R1 の Wi-Fi IP 取得完了: {wifi_ip} (内部LAN eth10 も正常: PC1疎通={'OK' if rc_pc1 == 0 else 'NG'})")

        add_step(f"5. Mac (Wi-Fi en0) から {wifi_ip} へ ping 疎通確認中...")
        ping_res = subprocess.run(["ping", "-c", "2", "-W", "1000", wifi_ip], capture_output=True, text=True)
        mac_ping_ok = ping_res.returncode == 0
        if mac_ping_ok:
            add_step(f"🎉 Mac から Wi-Fi 経由で {wifi_ip} への疎通に成功しました！もう LAN ケーブルを抜いても大丈夫です！")
        else:
            add_step(f"⚠️ R1 は {wifi_ip} を取得しましたが、Mac から ping が届きませんでした（同じルーターのSSIDか確認してください）。")

        result = {
            "success": True,
            "ssid": ssid,
            "wifi_ip": wifi_ip,
            "eth10": eth_line,
            "pc1_reachable": rc_pc1 == 0,
            "mac_ping_ok": mac_ping_ok,
            "steps": list(progress_steps),
        }
        RESULT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return JSONResponse(result)

    except Exception as e:
        log.exception("Wi-Fi setup error")
        add_step(f"❌ エラー: {e}")
        err_obj = {"success": False, "error": str(e), "steps": list(progress_steps)}
        RESULT_FILE.write_text(json.dumps(err_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return JSONResponse(err_obj, status_code=400)


HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Unitree R1 (2台目: R1_05387) — Wi-Fi セットアップ</title>
<style>
  :root { color-scheme: dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Hiragino Sans", sans-serif;
    background: #0f1117; color: #e6edf3; margin: 0; padding: 32px 16px;
    display: flex; justify-content: center;
  }
  .card {
    width: 100%; max-width: 580px; background: #161b22;
    border: 1px solid #30363d; border-radius: 14px; padding: 24px 28px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.45);
  }
  .badge {
    display: inline-block; border-radius: 999px; padding: 4px 12px;
    font-size: 12px; font-family: ui-monospace, monospace; margin-bottom: 12px;
  }
  .badge.ok { background: #23863633; color: #3fb950; border: 1px solid #23863688; }
  .badge.warn { background: #9e6a0333; color: #d29922; border: 1px solid #9e6a0388; }
  h1 { font-size: 20px; margin: 0 0 6px; }
  .sub { color: #8b949e; font-size: 13px; line-height: 1.55; margin-bottom: 14px; }
  .dongle-box {
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
    padding: 10px 12px; font-size: 12.5px; line-height: 1.5; margin-bottom: 14px;
  }
  label { display: block; font-size: 13px; color: #c9d1d9; margin: 12px 0 5px; font-weight: 500; }
  select, input {
    width: 100%; box-sizing: border-box; padding: 10px 12px;
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
    color: #e6edf3; font-size: 14px;
  }
  select:focus, input:focus { outline: none; border-color: #58a6ff; }
  .btn-row { display: flex; gap: 10px; margin-top: 18px; }
  button {
    flex: 1; padding: 11px 12px; border: none; border-radius: 8px;
    background: #238636; color: #fff; font-size: 14px; font-weight: 600; cursor: pointer;
  }
  button.secondary { background: #21262d; border: 1px solid #30363d; flex: 0 0 auto; }
  button:hover { opacity: 0.92; }
  button:disabled { background: #21262d; color: #8b949e; cursor: not-allowed; }
  .log-box {
    margin-top: 18px; background: #0d1117; border: 1px solid #30363d;
    border-radius: 8px; padding: 12px; font-family: ui-monospace, monospace;
    font-size: 12px; line-height: 1.65; white-space: pre-wrap; min-height: 80px;
  }
  .hint { font-size: 12px; color: #8b949e; margin-top: 6px; line-height: 1.45; }
</style>
</head>
<body>
<div class="card">
  <div class="badge ok" id="sshBadge">✅ 有線LAN直結・SSH接続済み (R1_05387 · 192.168.123.164)</div>
  <h1>2台目 Unitree R1 (R1_05387) を Wi-Fi に繋ぐ</h1>
  <div class="sub">
    1台目と同じルーター (<code>xg100n-c6922d-1</code> / <code>xg100n-c6922d-3</code>) に接続します。
  </div>

  <div class="dongle-box" id="dongleStatus">USB Wi-Fi ドングルの装着状態を確認中...</div>

  <form id="wifiForm" autocomplete="off">
    <label>接続先 Wi-Fi (SSID)</label>
    <select id="ssidSelect">
      <option value="xg100n-c6922d-1">xg100n-c6922d-1 (1台目と同じSSID · 2.4GHz)</option>
      <option value="xg100n-c6922d-3">xg100n-c6922d-3 (同ルーター 5GHz · 高速)</option>
      <option value="xg100n-c6922d-2">xg100n-c6922d-2 (2.4GHz)</option>
      <option value="__custom__">その他の SSID を手入力する...</option>
    </select>

    <div id="customWrap" style="display:none;">
      <label>SSID 手入力</label>
      <input type="text" id="customSsid" placeholder="SSID を入力" />
    </div>

    <label>Wi-Fi パスワード</label>
    <input type="password" id="wifiPwd" required placeholder="Wi-Fi の暗号化キー / パスワード" />
    <div class="hint">※ R1 の開発ボード (Jetson) には内蔵 Wi-Fi がないため、1台目と同様に <strong>USB Wi-Fi ドングル (TP-Link 等)</strong> を R1 の USB ポートに挿した状態で接続ボタンを押してください（ドライバ <code>rtl88x2bu</code> は導入済みです）。</div>

    <div class="btn-row">
      <button type="button" class="secondary" id="rescanBtn">ドングル再確認</button>
      <button type="submit" id="submitBtn">R1 をこの Wi-Fi に接続する</button>
    </div>
  </form>

  <div class="log-box" id="logBox">準備完了！USB Wi-Fi ドングルを挿してパスワードを入力し、ボタンを押してください。</div>
</div>

<script>
const sel = document.getElementById('ssidSelect');
const customWrap = document.getElementById('customWrap');
const customSsid = document.getElementById('customSsid');
const logBox = document.getElementById('logBox');
const btn = document.getElementById('submitBtn');
const rescanBtn = document.getElementById('rescanBtn');
const dongleStatus = document.getElementById('dongleStatus');

sel.addEventListener('change', () => {
  customWrap.style.display = sel.value === '__custom__' ? '' : 'none';
});

async function checkDongle() {
  dongleStatus.textContent = 'R1 側の USB Wi-Fi ドングルを確認しています...';
  try {
    const r = await fetch('/api/scan');
    const d = await r.json();
    if (!d.ok) {
      dongleStatus.innerHTML = '❌ LAN 接続エラー: ' + (d.error || '');
      return;
    }
    if (d.has_wifi_dev) {
      dongleStatus.innerHTML = `✅ <strong>Wi-Fi ドングル認識済み (${d.wifi_dev})</strong> — そのままパスワードを入れて接続できます！` + (d.wlan0 ? `<br/>現在の状態: <code>${d.wlan0}</code>` : '');
    } else {
      dongleStatus.innerHTML = '⚠️ <strong>現在 R1 (PC4) に USB Wi-Fi ドングルが挿さっていません。</strong><br/>1台目に挿していた TP-Link 等の USB Wi-Fi ドングルをこの R1 の USB ポートに挿して「ドングル再確認」を押してください。';
    }
  } catch (e) {
    dongleStatus.textContent = '確認失敗: ' + e;
  }
}
rescanBtn.addEventListener('click', checkDongle);
checkDongle();

let timer = null;
async function poll() {
  try {
    const r = await fetch('/api/progress');
    const d = await r.json();
    if (d.steps && d.steps.length) logBox.textContent = d.steps.join('\\n');
  } catch {}
}

document.getElementById('wifiForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const ssid = sel.value === '__custom__' ? customSsid.value.trim() : sel.value;
  const password = document.getElementById('wifiPwd').value;
  btn.disabled = true;
  btn.textContent = 'R1 の Wi-Fi を設定中...';
  logBox.textContent = '接続処理を開始しています...';
  timer = setInterval(poll, 500);

  try {
    const r = await fetch('/api/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ssid, password }),
    });
    clearInterval(timer);
    const d = await r.json();
    let txt = (d.steps || []).join('\\n');
    if (d.success) {
      txt += `\\n\\n✅ 完了！ R1 の Wi-Fi IP: ${d.wifi_ip}\\nもう LAN ケーブルを抜いても大丈夫です！`;
      checkDongle();
    } else if (d.error) {
      txt += `\\n\\n❌ ${d.error}`;
    }
    logBox.textContent = txt;
  } catch (err) {
    clearInterval(timer);
    logBox.textContent += `\\n❌ エラー: ${err}`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'R1 をこの Wi-Fi に接続する';
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8877, log_level="info")
