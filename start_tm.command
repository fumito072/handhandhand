#!/bin/zsh
# ダブルクリックで起動: サーバーが動いていなければ起動し、把持ページ (/tm) をブラウザで開く。
cd "$(dirname "$0")"
URL="http://127.0.0.1:8890"
if ! curl -s -o /dev/null "$URL/api/tm/status"; then
  echo "サーバーを起動します (ログ: choco_handover/logs/server_latest.log)"
  mkdir -p choco_handover/logs
  nohup .venv/bin/python -u choco_handover/server.py > choco_handover/logs/server_latest.log 2>&1 &
  for i in {1..90}; do
    curl -s -o /dev/null "$URL/api/tm/status" && break
    sleep 1
  done
fi
open "$URL/tm"
