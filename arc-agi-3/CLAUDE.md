# ARC Prize 2026 — ARC-AGI-3 — 作業の引き継ぎ（Claude Code 向け）

Kaggle `arc-prize-2026-arc-agi-3`（Featured・メダルあり・**締切 2026-11-02 23:59 UTC**・参加締切 10-26）。2026-10-01 着手。
このファイルが作業の正本。セッションの終わりに「進捗」と「次の一手」を更新して commit する。報告は日本語・です/ます。

## 課題（一次資料＝コンペのページ・2026-10-01 に読んだ）

- 対話型のゲーム（環境）を agent が遊ぶ。毎手 frame（最大 64×64・値 0〜15 の格子＋状態）を受け取り、行動を返す。
  行動は `RESET`・`ACTION1`〜`ACTION5`（単純）・`ACTION6`（座標 x,y つき）・`ACTION7`。意味はゲームごとに違い、探って知るしかない。
  各ゲームは複数レベル。状態は `NOT_FINISHED`／`WIN`／`GAME_OVER`。
- 評価：非公開の **110 ゲーム**（半分が Public LB・半分が Private LB）。
  - レベルの点＝`min(人間の手数 / agent の手数, 1)` の **2 乗**。ゲームの点＝レベル番号で重み付けした平均。総合＝ゲームの平均（0〜100%）。
  - ⇒ **解けたレベルの数**と**手数の少なさ**の両方が効く。無駄な手は 2 乗で響く。
- 提出：Notebook のみ・CPU/GPU とも 9 時間以内・インターネット不可・公開の外部データと学習済みモデルは可。**1 日 1 本**・最終 2 本は自分で選ぶ。
  この大会だけ RTX Pro 6000（`g4-standard-48`）が使える。
- 配布物：`ARC-AGI-3-Agents/`（agent の枠組み）・`arc_agi_3_wheels/`（`arc-agi` 本体の wheel・PyPI は 0.0.7 で古い）・`environment_files/`（**公開ゲーム 25 本**＝手元検証用）。
  手元では `arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=...)` で 25 本をオフラインで遊べる（公開ノートブックの読解より）。
  本番は `KAGGLE_IS_COMPETITION_RERUN` のとき `http://gateway:8001/` の COMPETITION モード。
- ダウンロードには**ルールへの同意が要る**（10-01 時点で未同意＝403）。

## 公開ノートブックの現状（2026-10-01 に読んだ・流用はしない＝読解だけ）

- 上位の公開は「duck harness」系（Tufa Labs が 6 月の milestone を 1.21 で取った枠組みの派生）。
  Qwen 27B 級の LLM を vLLM で RTX Pro 6000 に載せ、LLM がゲームを観察・仮説・行動する。解法の本体は dataset（taaf source）に入っている。
  派生の公開で LB 9 前後（「LB-9 duck v12 with Qwen 3.8 27B」）。
- LLM なしの探索（状態の BFS・記憶つき探索）の公開は LB 0.5〜0.9 程度。
- 乱数の agent の公開見本（`inversion/arc3-sample-submission-random-agent`）あり。

## 🎯 メダルへの道筋（2026-10-01 策定・毎セッション最初に読む）

**LB（10-05 取り直し・3,722 チーム）：1 位 55.89・金 17 位＝33.19・銀 186 位＝29.42・銅 372 位＝27.58・中央値 0.44。**（10-03：銅 24.65）
（10-01 は銅 3.99。09-30 に Milestone 2 の受賞解法 `dfranzen/arc-agi-3-milestone-2-solution`（Tufa Labs の Duck harness＋Qwen3.8-Flash-Next・SGLang）が公開され、
400 チームほどがその fork で 25〜33 点に並んだ。500 位で 5.35 点。）
**自分：explore-1/3/4＝0.15（約 2,774 位）。銅まで 27.4 点。締切 11-02。**
**正直な見立て（10-03）：銅は公開の受賞解法（または Duck harness）を使わない限り届かない。** 自前の LLM agent は手元 0.08〜0.15（探索だけの 0.11〜0.26 以下）。
「公開ノートブック・runtime dataset の流用はしない」方針のままなら、ARC は**メダル圏外を受け入れて最小の手間（1 日 1 本の探索の仮説確認）に留め、GPU 枠と時間を enveda・Gemma に回す**のが妥当。**10-03 user 了承：ARC は最小の手間に切り替え**（1 日 1 本は探索の仮説確認、GPU は使わない、時間と GPU 枠は enveda・Gemma へ）。公開解法・Duck harness は使わない（方針どおり）。

### 差の中身（どこで点を取るか）
- LLM なしの探索は 1 点未満＝銅（4 点）には **LLM でゲームの仕組みを読む agent が要る**。公開の duck 系は流用できないので、**自前の harness を組む**。
- 点の式から：レベル 1 を解くだけでも点になる（重み 1）。無駄な手を減らす（2 乗）。⇒ ①まず各ゲームのレベル 1〜2 を確実に解く、②手数を絞る。
- 使ってよいもの：公開の学習済みモデル（Qwen 系の重み・Kaggle Models / HF）、配布の `arc-agi` wheel。自分で書く：agent の観察・仮説・行動の仕組み全部。

### 段取り（判定は手元の公開 25 ゲームの点・同じ式で計算）
1. **〜10-04：手元の土台**。データ取得（ルール同意後）→ `arc-agi` を cloud で動かし、25 ゲームをオフラインで回す採点器を作る。
   乱数 agent と、状態ハッシュの探索 agent（新しい frame を優先して試す）で手元の点を測る。合格：探索 agent が手元で乱数より上、提出 1 本目で LB に載る。
2. **〜10-12：LLM agent の第 1 版**。Kaggle Notebook（RTX Pro 6000）で公開の Qwen 重みを vLLM か transformers で動かす。
   frame を文字の格子＋差分で渡し、「各行動で何が変わったか」の記録を持たせて行動を選ばせる。探索 agent を下敷きにして、LLM は目標の推定と次の一手の選択に使う。合格：手元 25 ゲームで 2 点以上。
3. **〜10-26：手数を絞る**（解けたレベルの道を覚えて同じゲームの次の試行で最短化・無駄な RESET を減らす）と時間配分（9 時間で 110 ゲーム）。合格：手元 4 点以上。
4. **〜11-02：最終 2 本の選択**（手元と LB の両方で上位）。

### 正直な見立て
- 1 か月で自前の LLM agent を公開の上位系統（9 点）並みにするのは難しい。**銅（4 点）は「探索＋LLM の素直な組み合わせ」が手元 25 ゲームで 4 点に届くかどうかが分かれ目**。
- 確実にしたいのは、段取り 1・2 で LB に載せて 1〜2 点帯に入ること。段取り 2 の手元点が 2 点を越えたら、GPU 枠（週 30 時間・enveda と共用）を ARC に寄せる。
- 1 日 1 本なので、提出は手元で上回った版だけにする（最終 2 本は選べるので損はないが、枠が少ない）。

### 毎日の動き方
- 1 本／日を使う。段取りのどの仮説を確かめる提出かを memo に書く。
- 重い計算：手元の 25 ゲーム評価は cloud コンテナ（LLM なし）、LLM は Kaggle Notebook の GPU。有料のものは使わない。

### 進捗
- 10-05：**explore-4（手数上限 10,000）＝LB 0.15**（2,500 と同点＝隠しゲームでも上限の増分では新しいレベルに届かない）。
  クリック候補の数（`ARC3_MAX_CLICKS`・1 状態あたり、小さい物体から）を手元 25 ゲームで比較：8＝0.1165（14 レベル）・24（従来）＝0.1138（16）・**64＝0.1536（18）**。
  ゲームごとの時間は 7.5 時間から割り振るので上限は越えない。**提出 explore-5**（64・仮説：クリック型のゲームでボタンが小さい 24 個の外にある＝LB 0.15 超）。採点待ち。
- 10-04：**explore-3＝LB 0.15**（explore-1 と同点＝LB は再現する・explore-2 の 0.09 は本物の負け）。
  手元 25 ゲームで手数上限 2,500→10,000：レベル 12→16・点 0.1131→0.1138（後から解けたレベルは手数が多く点が小さい・下がる要素は無い・時間は約 4 倍で 7.5 時間に十分収まる）。
  **提出 explore-4**（上限 10,000・仮説：隠しゲームでも届くレベルが増え LB 0.15 以上）。
- 10-03：**explore-2＝LB 0.09**（explore-1 0.15 より下・手元は 0.262 対 0.113 で逆）。手元の点は r11l 1 本（4 点）でほぼ決まる＝25 本では探索の変種を並べられない。
  採点の仕組みを読んだ（`arc_agi/scorecard.py`・`arcengine/base_game.py`）：本番は `ONLY_RESET_LEVELS=true` 相当（RESET はレベルのやり直しだけ・1 手に数える）、レベルの点＝min(115, 100·(基準/手数)²)、重み＝レベル番号。
  手元で同じ規則の採点器（`comp_score`・`LEVEL_LOG`）を作り、従来の scorecard と一致を確認（0.1131・0.2620）＝手元の採点は正しい。
  **LLM v1**（物体の一覧・行動の効果を物体の動きで・モデル自身のメモを持ち越し・12 手までの計画・レベルあたり 12 回）：Instruct＝**0.149**、Thinking-2507（6,000 トークン）＝**0.084**。
  モデルが解いたレベルは 0（全部 explorer の手）。Thinking は hex 格子の文字数え（「この行は 5 が 25 個…」）で予算を使い切り、計画を出せないことが多い。
  ⇒ 30B 級の文字格子読みでは届かない。**提出 explore-3**（explore-1 と同じ手順・仮説：LB は再現する＝0.15 が出れば explore-2 の負けは本物）。
  公開の上位（25〜33 点）は Milestone 2 の受賞解法の fork（上の見立て）。
- 10-02：**explore-2**（探索の手数削減）：「手の種類」（行動＋クリックした色）ごとに、画面が変わらなかった割合を数え、2 回以上試して変わらないことの多い種類を後回し（`ARC3_NOOP_MIN`）。
  手元 25 ゲーム：**0.113→0.262（12→15 レベル）**・NOOP_MIN 1/2/3/6＝.228/.262/.173/.105（値に敏感＝ゲームの数が少ない）。提出（仮説：無駄手が減り LB が explore-1 の 0.15 を上回る）。
  段取り 2 の下準備：公開上位は全部 `NvidiaRtxPro6000`＋自前の vLLM 実行環境 dataset＋Qwen 系（流用しない）。公式の Kaggle Models に `qwen-lm/qwen-3`（30b-a3b-instruct-2507-fp8・32b-fp8 ほか）・`google/gemma-4` がある。
  `kernels/gpuprobe`（RTX Pro 6000・Qwen3-30B-A3B-Instruct-2507-FP8・vLLM が画像にあるか／pip で入るか・読み込み時間・生成速度）を投入（gpuprobe-1）。
  本番はインターネット不可なので、vLLM が画像に無ければ wheel を GHA で取って自前 dataset にする（enveda の offline-wheels と同じ方式）。
  **probe の結果**：GPU＋インターネット有りの kernel は push が `SaveKernel 400`（RTX Pro 6000・L4 とも）。CPU＋インターネット有りは通る＝**GPU kernel はインターネット無しで作る**（enveda の GPU kernel も無し）。
  CPU 画像（gpuprobe-3）：Python 3.12.13・torch 2.10.0+cpu・transformers 5.0.0・**vLLM／sglang／flash_attn は無し**。`pip install vllm` で 0.30.0（torch 2.13 を連れてくる・242 秒）、
  同じプロセスで import すると古い torch が読み込み済みで落ちる（別プロセスなら動く見込み）。モデルは `/kaggle/input/models/qwen-lm/qwen-3/transformers/30b-a3b-instruct-2507-fp8/1`（30GB）にマウント。
  **GPU 画像（gpuprobe-4・RTX Pro 6000・インターネット無し）**：RTX PRO 6000 Blackwell 96GB・ドライバ CUDA 13.0・torch 2.10.0+cu128・CPU 46・RAM 176GB・ディスク 20GB・**vLLM 無し**。
  30GB のモデルのマウントに約 20 分（kernel 起動の待ち時間に含まれる）。
  ⇒ **自前の wheel dataset `yasunorim/arc3-vllm-wheels`**（`datasets/vllm-wheels/prepare.sh`＝GHA で vllm 0.30.0 と依存を全部 wheel で取得）を作成中（vllmwheels-1）。
  kernel では `pip install --no-index --find-links <dataset> vllm` で入れ、vLLM は別プロセス（OpenAI 互換サーバ）で動かす（同じプロセスだと古い torch と衝突）。
- 10-02：**LLM agent v0 が Kaggle で通しで動いた**（`kernels/llm`＝`build.py` で explorer＋`agent_llm.py` を 1 ファイルに・dataset `yasunorim/arc3-vllm-wheels`（3.9GB・vllm 0.30.0 一式）・開発用 kernel `yasunorim/arc3-llm-agent`）。
  動かすまでの詰まり（全部環境変数で解決）：①DeepGEMM の FP8 JIT が NVCC 12.9 以上を要求→`VLLM_USE_DEEP_GEMM=0`、②FlashInfer のサンプラーが sm120 を sm75 未満と誤判定→`VLLM_USE_FLASHINFER_SAMPLER=0`。
  pip で入れると torch 2.13.0+cu130 に上がる（CUDA 使用可・(12,0)）。vLLM の起動は約 6 分（重みの読み込み約 2 分＋CUDA グラフ）。
  **v0 の手元 25 ゲーム：0.084（12 レベル）＝explore-2（0.262）より下**。LLM 1 手あたり約 6 秒（25 ゲーム並列）、250 手のうち 35〜97% は解釈不能か既知の無効手で explorer に回った。
  ⇒ 生の hex 格子を毎手読ませるだけでは仕組みを掴めない（予想どおり）。次の版で直す点：
  (1) 格子をそのまま渡さず「物体の一覧（色・大きさ・位置）と、各行動で何がどう動いたかの表」に要約して渡す、
  (2) LLM は毎手ではなく、数手ごとに「仮説と次の数手の計画」を出させる（呼び出し回数を減らし、explorer の情報を土台にする）、
  (3) LLM が新しくレベルを解いたかを数えて効果を測る（今回は explorer 単独と同じ程度）。
- 10-01（下調べ）：段取り 2 の LLM 候補＝公式の公開重み（Kaggle Models）：`qwen-lm/qwen-3`（各サイズ）・`qwen-lm/qwen3-next-80b`・`google/gemma-4`・`danielhanchen/gpt-oss-20b/120b`。
  RTX Pro 6000（96GB）なら 27〜32B を bf16/FP8 で載る。推論系（vLLM が Kaggle の画像にあるか・無ければ transformers）を最初の GPU kernel で確かめる。
  他人の wheelhouse・解法 dataset（taaf 系）は使わない（自前で組む）。
- 10-01（同意後）：データ取得（44MB・`arc_agi_3_wheels` は cp312＝cloud では `python3.12 -m venv` に入れる）。
  手元の 25 ゲーム（`ARC3_LOCAL=1 ARC3_COMP_DIR=… ARC3_WORK=… v312/bin/python kernels/explore/main.py`・約 2 分）：**explore＝0.11（183 レベル中 12）**。
  解けたのは cd82・sp80・vc33（2）・su15・tu93（4）・lf52・ls20・r11l（1 レベルで 2.40）。2,500 手の総当たりなので効率の点はほぼ 0。
  **提出 explore-1**（仮説：提出の流れが動き、LB は手元の 0.11 前後）→ run 36812669367。**LB 0.15**（手元 0.11 とほぼ同じ＝手元の 25 ゲームで LB の見当がつく）。約 2,650 位相当（中央値 0.33 より下）。
- 10-01：`kernels/explore`（LLM なしの状態グラフ探索・自作）と `.github/workflows/arc3-kaggle.yml`（enveda と同じ依頼ファイル方式・`arc-agi-3/requests/kaggle.json` を push）を用意。
  模擬ゲーム（迷路 2 レベル・端に手数カウンタ）で動作確認：カウンタの行を隠す処理を入れて 726 手で 2 レベル。
  **気づき：レベルの点は (人間の手数 / agent の手数)² なので、総当たりで解いても点はほぼ 0**（15 手のところ 363 手なら 0.002）。公開の BFS が 1 点未満なのはこのため。
  ⇒ 探索 agent は LB に載せる土台。点を取るのは「少ない手で仕組みを掴む」部分＝段取り 2（LLM）が本命。
  データはルール同意待ちで、kernel も同意しないと動かない（competition_sources の取得に同意が要る）。同意後の最初の 1 本＝explore（仮説：自前の土台が LB 0 点より上に載る）。
- 10-01：着手。ページと公開ノートブックを読解、メダル線を取得。データはルール同意待ち（user に依頼済み）。PyPI の `arc-agi` は 0.0.7（古い）＝配布 wheel を使う。
