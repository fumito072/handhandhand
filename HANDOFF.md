# 引き継ぎ資料（HANDOFF）— R1 × Revo 2 チロルチョコ把持

最終更新: 2026-11-07（JST）

この資料だけで作業を再開できるように、目的・構成・起動方法・安全ルール・現在の設定値・これまでに分かったこと・次の一手をまとめています。

---

## 0. 3行まとめ

1. 机の上のチロルチョコを、カメラで位置と向きを推定し、**親指＋人差指＋中指（3指）で机面に沿ってつまむ**方式（`surface_pinch`）まで実装済みです。
2. 直近の失敗原因は「**親指だけ先に机に着き、人差指・中指がチョコの天面に乗る**」ことでした。**手首を約 +31〜48° 回す設定**と**人差指・中指を真下に向ける開き方**で、空中では3本の指先の高さが揃うことを確認済みです（IMG_9461）。
3. **この新設定での本番把持テストはまだ実施していません。** R1 の電源を入れたら、まずそれを試してください（§10）。

---

## 1. ゴール

- R1（`R1_05419`）の前の硬い机（黒い天板）に置いたチロルチョコを、**どの位置・向きに置いても一発でつまんで持ち上げる**。
- 親指と人差指（＋中指）の指腹でつまむ。指の側面や指先の爪側では掴まない。
- 将来的には「カメラ映像から、どこを親指と人差指で掴むかを推論しながら掴む」。

---

## 2. ハードウェアとネットワーク

| 項目 | 内容 |
|---|---|
| ロボット | Unitree R1 EDU（`R1_05419`）、26自由度。Jetson（PC4）上で動作 |
| R1 の IP | `192.168.1.13`（Wi‑Fi。Mac と同じルーター）。有線は `192.168.123.164` |
| SSH | `unitree@192.168.1.13`、**鍵認証**（Mac 側の `~/.ssh` に登録済み）。パスワードはリポジトリに含めない（`R1_SSH_PASS` 環境変数で渡す） |
| ハンド | BrainCo Revo 2（触覚付き）×2。`/dev/ttyUSB0` = 左（slave 126）、`/dev/ttyUSB1` = 右（slave 127） |
| カメラ（R1 頭部） | UDP 5001 = HD 1280×720、UDP 5002/5003 = ステレオ左/右 544×448（整準化済み、基線 50mm、fx≈132px） |
| 通信 | Mac `server.py`（:8890 Web UI / WebSocket）⇔ R1 `r1_bridge_server.py`（TCP :8891） |
| 腕の状態 | ハーネスで吊った状態（**宙吊りモード**、脚 12 軸＋腰ロールは 0 Nm） |

Revo 2 のモーター番号: `[0: 親指の曲げ, 1: 親指の向き(対立), 2: 人差指, 3: 中指, 4: 薬指, 5: 小指]`、範囲 0〜1000（親指の向きは 435〜498）。

---

## 3. ディレクトリ構成

```
unitree-r1/
├─ HANDOFF.md / README.md / requirements.txt / .gitignore
├─ docs/
│   ├─ master_strategy_v2.md          … 承認済みの全体戦略（Phase 1〜4）
│   ├─ grasp_analysis_and_roadmap.md  … それ以前の失敗分析とロードマップ
│   ├─ images/                        … 試行・動画の解析画像
│   └─ data/tactile_press_test_20261107.jsonl … 力センサーの押下テスト記録
├─ choco_handover/                    … 本体
│   ├─ server.py            FastAPI サーバー (:8890)。認識ループ、Web UI、API
│   ├─ handover_fsm.py      把持シーケンス（状態遷移）。HandoverOrchestrator
│   ├─ kinematics.py        R1 URDF の順運動学/逆運動学、カメラ→胴体座標変換、経路生成
│   ├─ choco_perception.py  YOLOE でチョコ検出、ステレオ＋HD で 3D 位置・向き・机面推定
│   ├─ stereo_depth.py      ステレオ視差と机面（RANSAC）
│   ├─ visual_servo.py      上空からの位置再補正
│   ├─ flight_recorder.py   試行ごとのログ保存（logs/run_*）
│   ├─ config.py            全設定値（SystemConfig）。calibration.json で上書き
│   ├─ calibration.json     現在の実機校正値（★これが実際に使われる値）
│   ├─ r1_client.py         R1 への TCP クライアント＋SSH 自動デプロイ
│   ├─ palm_gesture.py      「ください」両手ジェスチャー検出
│   ├─ tactile_memory.py    触覚で覚えたオブジェクトのプロファイル
│   ├─ teach_recorder.py    ティーチング記録
│   ├─ verify_pipeline.py   オフライン検証
│   ├─ static/index.html    Web UI
│   ├─ tools/calibrate_cameras_charuco.py … ChArUco/AprilTag 校正ツール（未実施）
│   ├─ models/              R1.urdf など（重みファイルは Git 管理外）
│   ├─ samples/             オフライン用の実機画像
│   ├─ logs/run_*/          直近 5 試行のフライトログ（画像・テレメトリ・要約）
│   └─ r1_onboard/          ★R1 に転送されるコード
│       ├─ r1_bridge_server.py  TCP :8891 のコマンド受付、カメラ、ハンド管理
│       ├─ r1_arm_daemon.cpp    250Hz 腕制御（重力補償、直線補間、各種安全ガード）
│       ├─ gentle_grasp.py      Revo 2 把持ポリシー（GentleChocoGraspPolicy ほか）
│       └─ CMakeLists.txt
└─ （R1 初期設定用の補助スクリプト: r1_wifi_setup.py, check_r1_info.py, probe_r1.py など）
```

R1 上の配置先: `/home/unitree/choco-handover-bridge/`（`r1_client.py` の `REMOTE_DIR`）。

---

## 4. セットアップと起動

### 4.1 Mac 側
```bash
cd unitree-r1
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```
モデル重み（Git 管理外）を `choco_handover/models/` に置く:
| ファイル | 入手元 |
|---|---|
| `yoloe-11l-seg.pt`（71MB） | Ultralytics YOLOE（`ultralytics` で自動ダウンロード可） |
| `mobileclip_blt.ts`（600MB） | YOLOE のテキストエンコーダ（初回のプロンプト設定時に Ultralytics が自動取得） |
| `hand_landmarker.task`（7.8MB） | MediaPipe Hand Landmarker の公式モデル |

### 4.2 起動
```bash
.venv/bin/python -u choco_handover/server.py
# ブラウザで http://127.0.0.1:8890
```
- 起動すると自動で R1 に SSH し、`r1_onboard/` を転送 → `r1_arm_daemon` をビルド → `r1_bridge_server.py` を起動します。
- `GET /api/status` の `deploy_status` が `ready`、`r1_connected` が `true` になれば準備完了です。
- **サーバーを再起動すると R1 側のブリッジも再起動されます。** 再起動直後の数秒はコマンドが「R1未接続」で失敗することがあるので、`ready` を待ってから操作してください。

### 4.3 R1 側の前提
- `/home/unitree/unitree_sdk2` と `/home/unitree/revo2-tools` が存在すること（デプロイ時に確認）。
- `gentle_grasp.py` は `revo2-tools` の `Hand` / `CurrentMonitor` を**読み込むだけ**です。

---

## 5. 安全ルール（必ず守る）

1. **`/home/unitree/revo2-tools/` と `/home/unitree/r1-local-tools/` は絶対に変更しない。**
2. **宙吊りモードを維持する**（`use_suspended_mode = True`）。脚 12 軸と腰ロールは 0 Nm。R1 が暴れたり振り回したりしないことを最優先する。
3. 腕のゲインは柔らかめ（kp 14〜26）。関節速度の上限は 0.15 rad/s。
4. `r1_arm_daemon` の安全ガード: トルク上限、位置追従誤差、異常速度（宙吊り時は即ダンピング）、モーター温度 60℃、仮想壁（`table_z_floor`）、ハートビート途絶。何かあればその場で柔らかく保持（SAFE_HOLD）する。
5. 非常停止: Web UI の停止ボタン、`POST /api/estop`、リモコンの **L2+B**。
6. ロボットを動かす前は、必ず人に一声かける（ユーザーが撮影や手の退避をするため）。
7. 把持に失敗したら、真上へ逃がしてから天板に当たらない経路で元の位置へ戻し、力を抜く（実装済み）。

---

## 6. 把持シーケンス（`surface_pinch`、[handover_fsm.py](choco_handover/handover_fsm.py)）

| フェーズ | 内容 |
|---|---|
| `LOCK_AND_PLAN` | 認識したチョコの位置・向き・サイズを固定し、左右どちらの手でどの向きにつまむかを計画 |
| `PREPARE_HAND_AND_ARM` | ハンドを開き方（形C）にし、腕を重力補償付きで制御開始 |
| `APPROACH_ABOVE_CHOCO` | 横に逃がす → 机の後ろで持ち上げる → 机の上へ振る → チョコの斜め上空（天板に当たらない3段の経路） |
| `VISUAL_SERVO_ALIGN` | 上空から HD 画像で位置・向きを再推定（XY と向きだけ更新、高さは初回値を固定） |
| `STRADDLE_HOVER` | チョコの真上（高さ +45mm 以上）でチョコをまたぐ |
| `TOUCHOFF_STRADDLE` | 真下へ直線でゆっくり下降。触覚・外力推定・エンコーダの停止のどれかで机への着地を検知 |
| `SURFACE_MICRO_RELIEF` | 着地点から +0.8mm だけ持ち上げて押し付けを抜く |
| `GENTLE_GRASP` | 3指で閉じる。目標の力（0.38N）に達した指から止める |
| 成功時 | 真上へ直線で持ち上げて保持（`grasp_and_hold` モード） |
| 失敗時 `SAFE_RETURN_EMPTY` | 手を開き、真上へ逃がしてから同じ経路で初期姿勢へ戻し、力を抜く |

---

## 7. 現在の設定値（[calibration.json](choco_handover/calibration.json) が実際に使われる値）

| 設定 | 値 | 意味・根拠 |
|---|---|---|
| `grasp_posture` | `surface_pinch` | 机面をガイドにまたいでつまむ方式 |
| `two_finger_pinch` | `false` | 3指（親指＋人差指＋中指）。薬指・小指は伸ばして退避 |
| `pinch_axis_z_goal` | `0.04` | **手首の回し量**。親指→人差指の向きの上下成分の目標（旧 0.22 = 人差指側が 13° 高い）。0.04 で約 +2° ＝ほぼ水平 |
| `pinch_pinky_z_min` | `0.10` | 小指側を持ち上げる最低量（旧 0.22）。手首を回す分だけ緩めた |
| `hand_eye_dx/dy/dz_m` | `0.007 / 0.079 / 0.041` | カメラ→胴体座標の補正。1回目・2回目の試行の HD 画像で、指の閉じた位置がチョコ中心から右 27mm・手前 7mm、机の高さが +33mm ずれていた分を補正 |
| `use_measured_head_yaw` | `false` | 首ヨー（モーター30）は電源を入れるたびに原点がずれるため、**接続時の値を正面＝0 とみなす** |
| `surface_relief_lift_m` | `0.0008` | 着地後に持ち上げる量 |
| `straddle_hover_z_m` | `0.045` | またぎ下降を始める高さ（チョコが高いときは自動でかさ上げ） |
| `grav_ff_scale` / `extra_hand_mass_kg` | `0.85` / `0.160` | 重力補償の強さ、Revo 2 の質量 |
| `tau_ext_contact_nm` | `0.68` | 外力推定で接触とみなすしきい値 |
| `gentle_force_N` | `0.38` | 1指あたりの把持力 |
| `use_suspended_mode` | `true` | 宙吊りモード |
| `execution_mode` | `grasp_and_hold` | 掴んだらその場で保持 |

ハンド側の定数（[gentle_grasp.py](choco_handover/r1_onboard/gentle_grasp.py)）:

| 定数 | 値 | 意味 |
|---|---|---|
| `TRIPOD_DOWN_IDX` | `450` | 3指モードの人差指・中指の開き位置（指先が真下を向く。IMG_9460 の形C） |
| 3指の開き方 | `[60, 473, 450, 450, 10, 10]` | 親指の曲げ 60 / 親指の向き 473 / 人差指 450 / 中指 450 / 薬指・小指 10 |
| 閉じる速さの比率 | 親指 1.25 : 人差指・中指 0.40 | 人差指・中指は壁としてほぼ動かさず、親指で押し当てる（曲げすぎると指先が持ち上がるため） |
| `CHOCO_PINCH_MAX` | `[680, 495, 780, 880, 720, 650]` | 閉じる上限 |

---

## 8. これまでの経緯と分かったこと

### 8.1 大きな流れ
1. 2指（親指×人差指）で「天面から 2mm 浮かせて横から 44mm 滑り込ませる」方式（`c_pinch`）→ mm 単位の精度が必要で、ステレオの奥行き分解能（約 10mm）では足りなかった。
2. Claude の分析を受けて **Master Strategy v2**（[docs/master_strategy_v2.md](docs/master_strategy_v2.md)）を作成・承認:
   - Phase 1: 机面ガイドのまたぎつまみ（`surface_pinch`）
   - Phase 2: C++ 制御に重力補償・外力推定・直線補間・天板の干渉チェック
   - Phase 3: HD 画像の視線と机面の交点でチョコの XY を推定（`xy_source = hd_plane_ray`）
   - Phase 4: フライトレコーダと、認識を使わない固定座標ベンチマーク（`POST /api/trigger_fixed`）
   すべて実装・実機デプロイ済み。
3. 実機テストで以下を順に発見・修正（§8.2）。

### 8.2 重要な発見（再発しやすいので必読）

**(a) 首ヨーのエンコーダ原点が電源投入ごとにずれる**
- `LowState.motor_state()[30].q()`（首ヨー）が再起動ごとに -0.07 → -0.01 → +0.10 → **+0.59 rad（+34°）** と変化。頭は物理的には正面のまま。
- 生の値をカメラ→胴体の変換に使うと、チョコ位置が 27cm も横にずれる。
- 対策: `server.py` で**接続時の首ヨー値を正面ゼロとして差し引く**。`center_head_forward()` も首ヨーを今の値のまま保持し、急に首を振らない。
- 注意: 接続した瞬間に頭が正面を向いている前提です。首を動かした状態で接続しないこと。腰ヨー（モーター13）と首ピッチ（29）は安定しています。

**(b) カメラと手の位置合わせの補正（Hand-Eye）**
- 2回の試行で、指が閉じた位置はチョコ中心から**右へ 27mm・手前へ 7mm**、机の高さは**ステレオ推定より 33mm 高い**位置にずれていた（再現性は 1mm 以内）。
- `hand_eye_dx/dy/dz` に反映済み。ただし**1か所の姿勢で合わせた値**なので、置き場所を大きく変えるとずれる可能性があります。本格的には `tools/calibrate_cameras_charuco.py`（ChArUco ＋手首の AprilTag）で校正するのが筋です（未実施）。

**(c) 力センサーは正常（把持時に 0N だったのは押し返す指がなかったため）**
- 2026-11-07 の5回の試行すべてで、全指の法線力・接線力が 0.00N のまま → 「空振り」判定になっていた。
- 人の指で押すテストでは右手5本とも 1.6〜7.8N を正常に検出（[docs/data/tactile_press_test_20261107.jsonl](docs/data/tactile_press_test_20261107.jsonl)）。
- 原因: 人差指・中指がチョコの天面に乗り、親指の反対側から押し返す指がなかったため、7g のチョコが滑るだけで力が上がらなかった。
- 近接センサーは常に有効: 親指は空中で約 3.5万、机に接触で約 5万、チョコを押すと 50万〜120万。**接触判定の補助として使える**（未実装）。

**(d) 親指だけが下に出ていて、人差指・中指が高すぎる（根本原因）**
- IMG_9457: 親指が先に机に着地（エンコーダの停止で検知）→ その時点で人差指・中指の指先はチョコの天面より上 → 曲げるとチョコの天面に乗る → チョコが斜めに持ち上がるだけ。
- IMG_9460（空中で開き方を比較）: 人差指・中指を 450 まで曲げると指先が真下を向くが、まだ親指より **10〜18mm 高い**。
- 原因: 逆運動学の手首ロールの目標が**わざと人差指側を 13° 高くする設定**（旧2指つまみ用の `0.22`）のままだった。
- 対策: `pinch_axis_z_goal = 0.04`（ほぼ水平）。手首を回す関節が約 +31〜48° 変わる（**肘側から指先を見て時計回り、親指側が上がる向き**）。
- IMG_9461（新しい手首の角度で空中撮影）: **3本の指先がほぼ同じ高さに揃った**（人差指・中指がわずかに低いくらい）。親指と人差指の間隔は約 45mm 以上。
- 計算上の比較（72通りの置き方）: 傾きの中央値 旧 +14〜22° → 新 +2〜10°、最大位置誤差 旧 7.5mm → 新 5.0mm（どちらも手が届くぎりぎりの端の位置で出る誤差）。

**(e) その他**
- 縦置きのチョコは、指腹が当たる面は大きいが倒れやすい。平置きで進める。
- 天板はダンボールより硬い机のほうが良い（着地検知が安定）。
- 机の外形はステレオから推定（`extract_table_bbox_torso`、胸の点を除くため x ≥ 0.20m）。

---

## 9. テスト履歴（2026-11-07）

| 試行 | 置き方 / 指 | 結果 | 原因 |
|---|---|---|---|
| `run_..._010259` | 平置き / 2指 | 失敗（閉じ待ちタイムアウト） | 閉じた位置が右 27mm・手前 7mm ずれ |
| `run_..._011137`（IMG_9455） | 縦置き / 2指 | 空振り | 同上（人差指がチョコの角をかすめた） |
| `run_..._012804` | 縦置き / 3指＋補正後 | 空振り | 着地がチョコの天面付近（z = -1mm）で止まった |
| `run_..._013158` | — | エラー（RuntimeError） | 安全ガード系の例外。詳細はログ未調査 |
| `run_..._013305`（IMG_9457） | 平置き / 3指 | 空振り | §8.2(d)：人差指・中指がチョコの天面に乗った |
| 空中撮影（IMG_9460） | 開き方 A〜D の比較 | — | 形C（人差指・中指 450）が有望。ただし指先が 10〜18mm 高い |
| 空中撮影（IMG_9461） | 新しい手首の角度で形C・D | **指先の高さが揃った** | — |

各試行のログ: `choco_handover/logs/run_*/`（`meta.json` 計画値、`telemetry.jsonl` 10Hz の関節角・トルク・触覚、`summary.json` 結果、`*.jpg` 各段階の画像）。
解析画像: [docs/images/](docs/images/)。

---

## 10. 次にやること（優先順）

1. **新設定での本番把持テスト（最優先）**
   - R1 の電源を入れる（できれば充電。最後に見た残量は 42%）。
   - 頭が正面を向いていることを確認してからサーバーを起動（§8.2(a)）。
   - チョコを平置きで、ロボット正面の机に置く。
   - Web UI の把持ボタン、または `POST /api/trigger`。
   - 撮影しておくと解析が速い（親指側の真横から）。
   - 見るポイント: 3本の指先がほぼ同時に机に着くか / 人差指・中指がチョコの側面に当たるか / 力センサーが 0.38N で止まるか。
2. **接触判定を力センサーだけに頼らない**: 近接値の急上昇、モーター電流、指令位置に指が追いつかない（物に当たって止まった）を組み合わせて、指ごとの接触と「掴んだ」判定を行う（[gentle_grasp.py](choco_handover/r1_onboard/gentle_grasp.py) の `_evaluate_settlement` 付近）。
3. **閉じる間も指先を机に沿わせる**: 閉じると指先が持ち上がる場合は、閉じながら腕を少しずつ下げる（親指の近接値を「机に接触」の範囲に保つ）制御を検討。
4. **肩をひねる関節が上限付近**: 新しい手首の角度では右肩ヨーが 1.74〜1.78 rad（上限 1.76〜1.80）。置き場所によっては届かない可能性。腰ヨーを使う（`enable_waist_yaw_6dof`、宙吊り時は無効）か、左手を使う候補を増やす。
5. **ChArUco/AprilTag による正式な校正**（`tools/calibrate_cameras_charuco.py`）で、1点合わせの Hand-Eye 補正を置き換える。
6. **成功率で評価**: `POST /api/trigger_fixed`（固定座標）で 5位置 × 4向き × 2種類の机などを回し、`GET /api/benchmark_summary` で成功率と失敗の種類を集計する。
7. `run_..._013158` の RuntimeError の原因調査（`telemetry.jsonl` の `last_fault`）。

---

## 11. 未解決の課題・リスク

- 新しい手首の角度と3指の開き方での**実機把持は未検証**。
- 着地検知はほぼエンコーダの停止頼み（触覚は力が出ないと反応しない）。
- 手が届くぎりぎりの位置（x = 0.40m、y = -0.06m など）では逆運動学の位置誤差が 5mm 程度出る。
- Hand-Eye 補正は1か所で合わせた値。
- 首ヨーの補正は「接続時に頭が正面」が前提。
- バッテリー残量が 10% 以下だと動作を拒否する。15% 以下では指形状の撮影を拒否する。

---

## 12. デバッグに使える道具

| 用途 | 方法 |
|---|---|
| 状態の確認 | `GET /api/status?compact=0`（`choco` = 認識結果、`telemetry.robot` = 腕、`telemetry.hands` = ハンドと触覚、`task` = 現在のフェーズ） |
| 指の形だけ動かす（腕は動かさない） | `POST /api/hand_test` `{"side":"right","action":"pose","targets":[60,473,450,450,10,10]}` |
| 腕を机の上空に上げて指の形を撮影 → 自動で下ろす | `POST /api/finger_shape_scan` `{"side":"right","hold_s":6,"poses":[[...],[...]]}` |
| 触覚センサーの確認 | ロボットを止めた状態で `/api/status` を 0.15 秒ごとに記録し、人の指で指腹を押す（`docs/data/` の記録を参照） |
| 固定座標でのベンチマーク | `POST /api/trigger_fixed`（x, y, yaw などを指定）→ `GET /api/benchmark_summary` |
| 非常停止・その場で保持・安全に戻す | `POST /api/estop`、`/api/hold_here`、`/api/safe_return` |
| R1 へ再デプロイ | `POST /api/deploy_r1`（またはサーバー再起動） |

R1 ブリッジのコマンド（TCP :8891、JSON）: `arm_engage`, `arm_set_params`, `arm_move`, `arm_move_path`, `arm_hold_here`, `arm_disengage`, `hand_open`, `hand_pose`, `hand_gentle_grasp`, `hand_learn_object`, `hand_gentle_release`, `hand_stop`, `hand_reconnect`, `estop`, `capture_hires_5001`, `ensure_stereo`, `ping`。

---

## 13. 変更の要点（直近セッションで入れたもの）

- `server.py`: 首ヨーの接続時ゼロ補正、`/api/hand_test` の `pose`、`/api/finger_shape_scan`。
- `handover_fsm.py`: `finger_shape_scan()`、`center_head_forward()` の首ヨー保持、着地検知の高さ条件の見直しと着地後すぐ除荷、ホバー高さの自動かさ上げ、再ロック時の机の高さチェック、3指の表示とタイムアウト延長。
- `kinematics.py`: 手首ロール目標を `pinch_axis_z_goal` / `pinch_pinky_z_min` で設定可能に。
- `config.py` / `calibration.json`: Hand-Eye 補正、3指モード、手首ロール目標。
- `r1_onboard/gentle_grasp.py`: 3指の開き方（人差指・中指 450）、親指主体で閉じる比率、接触判定を「親指 対 人差指または中指」に拡張。
- `r1_wifi_setup.py` / `check_r1_info.py`: SSH パスワードを環境変数 `R1_SSH_PASS` から読むように変更（リポジトリに含めないため）。
