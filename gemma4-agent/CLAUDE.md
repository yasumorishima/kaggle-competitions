# Gemma 4 Developer Agent — 作業の引き継ぎ（Claude Code 向け）

Kaggle `gemma-4-developer-agent`（Featured・メダルあり・**締切 2026-12-02 23:59 UTC**・参加締切 11-25）。2026-10-01 着手。
このファイルが作業の正本。セッションの終わりに「進捗」と「次の一手」を更新して commit する。報告は日本語・です/ます。

## 課題（一次資料＝コンペのページと `HARNESS_README.md`・2026-10-01 に読んだ）

- SWE-bench 型：Python リポジトリ（fastapi・rich・requests・httpx）の issue を直すパッチを agent が作り、隠しテストが通れば 1 点。点＝解けた割合。
- **提出は agent の設定一式（`submission.zip`）**：`agent.yaml`（Google ADK の宣言的 YAML）・prompt（.md）・`eval_config.yaml`（1 課題あたりの予算）・skills・LoRA（PEFT・safetensors・合計 3GiB 未満）。Python コードは出せない。
- モデルは **`gemma-4-31b-it-qat-w4a16-ct` 固定**（4×L4・vLLM・文脈 32,768 トークン）。採点は主催側の機械で動く＝こちらの GPU は使わない。
- 全課題で 12 時間（課題数は非公開）。公開の 0.10 点ノートブックは 1 課題 4.5 分・道具 40 回に絞っている（12 時間に収める配慮と読める）。
- 道具 9 つ：`run_command`・`read_file`（150 行まで）・`edit_file`・`write_file`・`search_similar_code`・`get_code_neighbors`・`get_code_subgraph`・`get_status`（無料）・`submit_patch`（無料・最後に呼ぶ）。
- 注意：/workspace に作った作業用ファイルはパッチに入る（/tmp に置く）。`pytest.ini`・`conftest.py` を触らない。大きな編集は出力上限で途切れる。
- **1 日 1 本**・最終 2 本は自分で選ぶ。
- 練習用：`tasks.jsonl`（129 課題・正解パッチと検証テスト付き）・snapshots・graphs・embeddings（データ全体 22GB）。
- 手元検証：公式の `adk_submission`（`metric/gemma-4-developer-agent-wheelhouse` の wheel）で **YAML の検証とコンパイル**は cloud でできる（python3.12 の venv）。
  実際に解かせる評価は 31B を動かす GPU が要る（Kaggle の L4 機・GPU 枠を使う）。

## 🎯 メダルへの道筋（2026-10-01 策定・毎セッション最初に読む）

**LB（10-01・1,167 チーム）：1 位 0.17・金 0.13・銀 0.12・銅 117 位前後＝0.12・中央値 0.06。** 自分：未提出。締切 12-02（残り 62 日）。
点の刻みが粗い（課題数が少ない）ので、銅は「上位と同じくらい解ける agent」で届く範囲。

### 差の中身
- 公式の見本は 1 課題 1 分・道具 10 回で、これでは足りない。予算を適正にし、手順（場所特定→最小編集→検証→提出）をはっきり指示するだけで 0.06→0.10 前後と見る。
- その上：場所特定を早くする（graph 道具・grep の使い分け）、必ず提出させる（予算切れで 0 点を防ぐ）、検証の仕方。
- 最後に LoRA（TPU で学習・PEFT 形式に変換）。129 課題の正解の手順を教師にする。

### 段取り（合格基準は LB。手元で解かせる評価は GPU 枠を使うので要所だけ）
1. **10-01：v1（自作の prompt・4.5 分・40 回）を出す**。合格：LB 0.08 以上。
2. **〜10-15：prompt と予算の改良**を 1 日 1 本で。必要なら Kaggle の L4 で 129 課題の一部を手元評価。合格：LB 0.12（銅）。
3. **〜11-15：LoRA**（TPU で学習）。合格：手元の部分評価で prompt 版より上。
4. **12 月：最終 2 本**。

### 毎日の動き方
- 1 本／日。提出前に `adk_submission` で検証・コンパイルを通す（壊れた設定で枠を失わない）。memo に仮説を書く。

### 進捗
- 10-01：着手。ルール・harness を読解。自作の v1（`submission/`）を作成し、公式パッケージで検証・コンパイル済み。
  提出の流れ：`kernels/pack`（CPU・設定を zip に固める）＋ `.github/workflows/gemma4-kaggle.yml`（`gemma4-agent/requests/kaggle.json` を push）。
