# handhandhand — Unitree R1 × BrainCo Revo 2 チロルチョコ把持

Unitree R1（`R1_05419`）の腕と BrainCo Revo 2 触覚ハンドで、机に置かれたチロルチョコ（約 33×26×14mm、7g）を
カメラで見つけて、親指＋人差指＋中指でつまみ上げるシステムです。

- **作業の引き継ぎは [HANDOFF.md](HANDOFF.md) を最初に読んでください**（現状・経緯・次の一手・注意点をすべてまとめています）。
- 戦略資料: [docs/master_strategy_v2.md](docs/master_strategy_v2.md)、[docs/grasp_analysis_and_roadmap.md](docs/grasp_analysis_and_roadmap.md)

## 構成（概要）

```
Mac (choco_handover/server.py, :8890 Web UI)
  ├─ 認識: ステレオ(544x448) + HD(1280x720) → チョコ 3D 位置・向き
  ├─ 計画: R1 URDF の順運動学/逆運動学 → 天板に当たらない経路
  └─ TCP :8891 ──> R1 (Jetson) r1_onboard/r1_bridge_server.py
                      ├─ r1_arm_daemon (C++, 250Hz, 重力補償・直線補間・安全ガード)
                      └─ gentle_grasp.py (Revo 2 触覚つき把持ポリシー)
```

## クイックスタート

```bash
cd unitree-r1
python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
# モデル重みを choco_handover/models/ に配置（HANDOFF.md §4 参照）
.venv/bin/python -u choco_handover/server.py   # → http://127.0.0.1:8890
```

R1 側のコードは、サーバー起動時に SSH（鍵認証）で `/home/unitree/choco-handover-bridge/` へ自動転送・ビルド・起動されます。
