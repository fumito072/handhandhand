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

## 親指×中指つまみ（2026-11-07 に成功した方式）

人の手で形を作って教えたデータをもとにした把持です。親指の付け根を約 720 まで回し、親指と中指でつまみ、薬指を 3 点目の支えにします（人差指と小指は使いません）。

**一発起動**: `start_tm.command` をダブルクリック → サーバーが起動して http://127.0.0.1:8890/tm が開きます。
「掴む」「戻す」「止める」「計画だけ確認（動かない）」のボタンと、カメラ映像・ログが見られます。
コマンドで直接動かす場合は `.venv/bin/python -u choco_handover/tools/grasp_thumb_middle.py grasp|return|plan`。

流れ（`choco_handover/tools/grasp_thumb_middle.py`）:

1. 初期姿勢のままカメラでチョコの位置と向きを 4 秒間見て固定する（手の甲で隠れても固定した位置を使う）
2. 親指をまっすぐにしてから付け根を回し、その後に曲げる（曲げたまま回すと人差指に当たって約 500 で止まる）
3. 腕を机の上へ（0.30 rad/s）→ チョコの 7cm 手前で手を下ろして机に触れ、机の高さを測る
4. 測った高さを使ってチョコの上からゆっくり下ろし、指が何かに当たったら止める
5. 次のどちらかなら握る（近接値は生値 ÷ 1000）。どちらでもなければ 7cm 持ち上げてから横にずらして下ろし直す（最大 4 回）
   - 親指と中指の近接センサーが両方反応（親指 ≥ 80、中指 ≥ 80）→ 両方曲げる
   - 中指の腹だけ反応（≥ 300）→ 親指を曲げる / 親指の腹だけ反応（≥ 90）→ 中指を曲げる
6. 6cm 持ち上げ、指の力と近接値から持てているかを判定する

- 指の間はチョコの幅より少し広く開けておく（`SHAPE_OPEN`）。ぴったりだと触れた瞬間にチョコがずれる。
- 机の高さは毎回測るので、教えた時の高さから ±2.5cm 以内なら高さが変わっても掴めます（測定範囲外の場合は教えた高さを使う）。
- 手で教えた位置関係: `docs/data/grasp_offset_teach.json`（`tools/teach_grasp_offset.py` で作り直せます）。
- 実行記録: `docs/data/grasp_thumb_middle_runs.jsonl`、手の形の教師データ: `docs/data/hand_pose_teach_20261107*`。
