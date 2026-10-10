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

**LB（10-10 01:01 取り直し・2,231 チーム）：1 位 0.24・金 14 位＝0.18・銀 111 位＝0.17・銅 223 位＝0.15・中央値 0.10。自分：0.08（g4v6-1・g4v7-2）・1,587 位。銅まで +0.07（公開 約 58 課題で約 4 課題）。締切 12-02。**
- 銅の線は 5 日で 0.13→0.15、中央値も 0.08→0.10 に上がった。
- **計画監査（fable・10-10）での一番弱い前提**：手元の解決率が隠しに移ること。手元では v7 が 22 対 14 で v6 に勝ったが、LB は両方 0.08 だった。
  - (a) CPU で 129 課題の健全性の対照（空パッチで落ち、正解パッチで通る）を作り、評価の母集団を固める。期日 10-11、GPU は使わない。
  - (b) 4×L4 の直列で、v7 と v8 を shard 0/3 の 43 課題で比べる。合格は v8 が +4 課題以上、かつ「5 分切れ・パッチなし」が半分以下。期日 10-13、GPU 約 7.2 時間。勝てば g4v8-1 を出す。
  - tpueval-4（6 課題）は、TPU にモデルが載るかと 1 課題の秒数を見るだけで、勝ち負けは決められない。
  - LoRA（G3）は今週は手を付けない。
- 今週の GPU：Gemma に約 8 時間（直列の対比較 1 回）。

**LB（10-05 取り直し・1,741 チーム）：1 位 0.24・金 13 位＝0.15・銀 87 位＝0.13・銅 174 位＝0.13・中央値 0.08。** 自分：最良 **g4v5-1＝0.06**（10-06・g4v3-1 0.05 の次）。締切 12-02。
（10-01 は 1 位 0.17・中央値 0.06。0.10・0.08 に各 270 チーム余り＝公開 walkthrough 系が並ぶ帯。）
点の刻みが粗い（課題数が少ない）ので、銅は「上位と同じくらい解ける agent」で届く範囲。

### 差の中身（10-07 書き直し・公開の計測ノート `dmitriigluzdov/gemma-4-measure-before-you-tune` を読んだ・流用なし）
- **公開 LB は雑音が大きい**：同じ zip（Rozen 版）が 0.12・0.10・0.06 を出した（公開は約 58 課題＝1 課題 0.017、±3〜4 課題ぶれる）。
  ⇒ 当方の 0.03〜0.06 の上下も雑音。**LB を見て prompt を刻むやり方では銅は選べない（10-07 user 指示：現状維持の提出は出さない）**。
- 手元で解かせる評価はできる：
  - 公式の wheelhouse に `swegemma 0.2.7`（`swegemma eval --sandbox subprocess`）と `vllm 0.19.1` がある。
  - モデルは Kaggle Models の `google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct`。
  - 公開側の計測：129 課題のうち健全（空パッチで落ち、正解パッチで通る）なのは 114。公開 baseline の解決率は dev 16/70・Rich 12/44。
- **時間が縛り**：約 120 課題を直列で 12 時間＝準備込みで 1 課題 約 280 秒。公開の coder＋analyzer（8 分）は 12〜13 時間かかり、時間切れの危険がある。
- 上位（0.17〜0.24）の手法は公開されていない。**銅＝隠し約 120 課題で、公開 baseline より確かに多く解く agent**。手元の健全 114 課題で baseline より +8 課題（+7 ポイント）以上を目安にする。

### 段取り
1. **G1（〜10-10）手元の評価台**：
   - Kaggle GPU（RTX Pro 6000 96GB・インターネット無し）に、公式 wheel の vLLM と swegemma を入れる。
   - 31B QAT を立て、`swegemma eval --sandbox subprocess` で 129 課題を回す kernel（`kernels/localeval`）を作る。
   - 空パッチ・正解パッチの対照で健全な課題を自分で決める。
   - 合格：正解パッチの対照が約 115 課題で通り、当方 v5／v6 の解決率と 1 課題あたりの時間が出る。
2. **G2（〜10-20）解決率を上げる設計の比較**：同じ課題・同じ seed で対にして比べる。
   - 比べる設計：再現テストを先に書く→直す→確かめる、場所特定の sub-agent、時間の配分。
   - 合格：健全課題で公開 baseline 相当より +8 課題、かつ 1 課題の平均が 250 秒以内。
3. **G3（〜11-15）LoRA**：手元で解けた軌跡だけを教師にし、TPU で学習する。
   - 合格：手元の対の比較で prompt 版より有意に上。KV の縮みの検査も通すこと。
4. **12 月**：最終 2 本を、手元の解決率で選ぶ（LB は雑音）。

### 正直な見立て（10-07）
- 公開の壁は 0.12。銅（0.13）は、隠しの約 120 課題で壁より数課題多く解けば届く距離。ただし、手元評価なしの当方 v1〜v6 は壁にも届いていない。
- 評価台ができれば、改良を雑音なしで選べる。**G1 が立たない限り銅は運任せ**なので、G1 を最優先にする。

### 毎日の動き方
- 手元評価で、対の比較に勝った版だけを出す。LB の 1 本は「手元で勝った版が隠しでも勝つか」の確認に使う。勝った版が無い日は出さない。

### 進捗
- 10-07：g4v6-1＝LB 0.08（雑音の範囲）。**G1 の手元評価 kernel**（`kernels/localeval`＝公式スターターの手順・公式 wheel・`swegemma` の Evaluator を subprocess の sandbox で）を作成。
  - 構成：v6 と公式 sample を同じ 129 課題で対にして比べる。`build.py NAME=DIR` で設定を埋め込む。
  - **公式 wheel は cp312**：Kaggle の新しい画像（3.13）では入らない。metadata の `docker_image` に、スターターと同じ競技用の画像（`gcr.io/kaggle-private-byod/python@sha256:37c6…`）を指定し、3.12 で動いた。
  - localeval-1：その画像だと RTX Pro 6000 の指定が効かず T4 に載り、bf16 が非対応で落ちた。localeval-2 は `NvidiaL4`（スターターと同じ）で再実行。
  - **localeval-2（L4・6 並列）：v6＝129 課題中 15 解決（11.6%）・平均 265 秒／課題**（fastapi 8・rich 7・requests 0・httpx 0）。公式 sample は LoRA の別名が 404（`main_lora` が vLLM に無い）で全滅＝比較から外す。
  - Kaggle のログ API は末尾の約 1,000 行しか返さない（課題ごとの行が消えた）⇒ 最後に設定ごとの 1 行表（`ROWS 名前 id:解決:秒 …`）を出す形に直した。
  - localeval-3：v6 と公開 0.13 構成（`bases/pub013`・planner→coder・出典 `bases/pub013_SOURCE.txt`）を対にして比べる。
    **結果（4×L4＝本番と同じ機械・6 並列）：v6 11/129（平均 282 秒）・pub013 0/129（全課題が 5 分の上限で切れた）。**
    - v6 は localeval-2 の 15 と比べて ±4 課題ぶれた＝同じ設定でも雑音がある。
    - pub013 は LB 0.13 なのに 0：6 並列では 1 課題あたりの速度が本番（直列）の数分の 1 になり、planner が時間を食い尽くす。⇒ 並列の評価は時間制限のある構成に不公平。
    - **次は直列（workers 1）で、課題の部分集合（shard 0/3＝43 課題、1 構成 約 3.6 時間）を測る。**
    - 失敗の内訳（v6）：
      - read_file の範囲誤り（開始行が終了行より大きい等）と、存在しないパス。
      - 道具の呼び出し回数の上限（80 回）切れ。
      - edit_file の old_string 不一致。
      - /tmp への書き込みの拒否。
      - ⇒ 道具の使い方の誤りで回数と時間を浪費している。直す候補。
  - **v7**（`submission/`）：prompt に道具の呼び出しの決まり（read_file はパスだけ・行番号は別の数、not found なら素のパスで再試行、edit_file は old_string 必須・新規は write_file、失敗した呼び出しを同じ引数で繰り返さない）を足した。v6 は `bases/v6`。
  - **localeval-5（4×L4・6 並列・同じ 129 課題で対）：v7 22/129・v6 14/129。v7 だけ解けた 10・v6 だけ 2・両方 12（符号検定 p=0.019）**。rich 10 対 5・fastapi 12 対 8。
    ⇒ 道具の使い方の誤りが主な損だった、という見立てが手元で確かめられた。**g4v7-1 を提出**（仮説：LB が v6 の 0.08 を上回る）。
  - **Kaggle の GPU 週 30 時間を使い切った（10-07 12:27 UTC。enveda の push が "Maximum weekly GPU quota" で拒否）**。手元評価（4×L4）は枠が戻るまで回せない。その間は失敗の内訳（g5.log）から次の部品を作る。
- 10-08：g4v7-1 は 24 時間たっても採点中。v7 の失敗の内訳（g5.log・v7 の 258 行）：解決 44、5 分の上限で切れた 134（うちパッチなし 92）、パッチを出したが未解決 46、道具の回数上限 18、その他 16。
  **6 並列の評価は 1 課題あたりの速度が本番（直列）より遅く、時間切れが水増しされている**。次の部品の比較は直列で行う必要がある。
  - **GPU 週枠が切れている間の評価台＝TPU（`kernels/tpueval`）**：vLLM の TPU 版（PyPI の `vllm-tpu` 0.31）で Gemma 4 31B を TPU v5e-8 に載せる（TP 8・w4a16 が載らなければ同じモデルの bf16 QAT 版）。
    評価側は公式 wheel（adk・swegemma）を uv の Python 3.12 の venv に入れる（インターネット可）。v7 を shard 0/3（43 課題）で直列に回し、4×L4 の結果と比べて評価台として使えるかを見る。
    tpueval-1 は「batch TPU session は同時 1 本」で拒否（enveda の e2tpu-1 が TPU 待ち）＝e2tpu-1 の後に出し直す。
    **10-09：tpueval-2 を push**（enveda の e2c3ice-2 が 14:15 に終わり TPU が空いた）。TPU の batch は約 2 時間で切られた例があるので、v7 を shard 0/3 の先頭 12 課題・直列に縮めた。見るのは、モデルが載るか、1 課題あたりの秒数が 4×L4 の直列に近いか。
    監査（10-09）：`tool_call_parser="gemma4"` が vllm-tpu 0.31 に無いとサーバーが即死し、原因がログに残らない。⇒ 失敗時に vLLM のログ末尾と parser の選択肢を出す版を **tpueval-3** として出し直した（中身は同じ 12 課題・直列）。速度の基準には、同じ 12 課題を GPU で直列に回す対照が別に要る。
    tpueval-3 は「batch TPU session は同時 1 本」で拒否（tpueval-2 が待ち行列にいる）。**tpueval-2（ログ追加の前の版）をそのまま走らせる**。サーバーの例外には元からログ末尾 2,000 字が入るので、起動失敗の原因は読める。
  - **g4v7-1 は 39 時間後に Kaggle の "A system error. Please try resubmitting" で終わった（点なし）**＝v7 の仮説は未検証のまま。**g4v7-2 として同じ v7 を出し直した**（10-08 の 1 枠・仮説は同じ：LB > v6 0.08）。
- 10-09：**g4v7-2＝LB 0.08**（v6 と同じ）。手元の v7 22 対 v6 14 は、隠しでは差として出なかった（公開 LB は約 58 課題・±3〜4 課題の雑音）。銅 0.13 まで約 3 課題分。
  - 監査（fable）で挙がった最大の損：v7 の未解決のうち「5 分で切れ、パッチなし」が 92/258。時間切れでも編集した差分は採点されるので（HARNESS_README 577 行：未提出なら終了時に git diff を取る）、**編集に入るまでの時間**が損の中心。
  - ⇒ **v8（`submission/`・編集先行）**：locator の sub-agent を外し、自分で git grep と read_file を 4 回までして、7 回目までに必ず編集する。
    編集の後に `get_status`（無料）で残り時間を見て、120 秒より多ければ checker を 1 回だけ呼ぶ。60 秒を切ったら提出する。v7 は `bases/v7` に保存。公式パッケージの compile は通った（道具は stub）。
  - 次：TPU（tpueval-2 で評価台として使えると分かれば）か、GPU の週枠が戻った後の 4×L4 で、**v7 と v8 を同じ課題・直列で対にして比べる**。勝ったら提出する（仮説：時間切れでパッチなしの課題が減り、LB 0.10 以上）。
- 10-10：**tpueval-2 は 10-09 23:53 に開始（約 9 時間待ち）→ 00:02 にエラー**。
  - 分かったこと：TPU 機は 96 コア・RAM 377GB・Python 3.12。vllm-tpu 0.31.0 と jax 0.11.0 は入り、TPU も見えた（入れるのに約 6 分）。
  - 落ちた所：評価用の公式 wheel を入れる段。wheelhouse の adk_submission が 0.2.12 から **0.2.13** に上がっていて、版を決め打ちにしたファイル名が無かった。
  - ⇒ wheel を版ではなくパッケージ名で選ぶように直した。
  - **tpueval-4** として、v7 と v8 を shard 0/3 の先頭 6 課題・直列で対にして出した。モデルが載るか、1 課題の秒数、v8 で「パッチなしの時間切れ」が減るかを見る。
- 10-01：着手。ルール・harness を読解。自作の v1（`submission/`）を作成し、公式パッケージで検証・コンパイル済み。
  提出の流れ：`kernels/pack`（CPU・設定を zip に固める）＋ `.github/workflows/gemma4-kaggle.yml`（`gemma4-agent/requests/kaggle.json` を push）。
- 10-01：**g4v1-1＝エラー**（"Your notebook hit an unhandled error while rerunning your code"・点なし）。CPU の notebook 自体は問題ない（公開の 0.10 walkthrough も CPU）。
  公開 walkthrough の読解（流用なし）から疑わしい点：`max_time_minutes: 4.5`（小数）、`include_thoughts: true`（採点で動いた公開版は全部 thinking 切り）、
  `search_similar_code` が関数本体を上限なしで返し文脈 32k を溢れさせる（未捕捉エラー）。また全課題は**直列**で 12 時間（1 課題あたり準備・テスト込み約 6 分）。
  ⇒ **v2**（`submission/` 更新済み・検証/コンパイル済み・未提出）：整数の予算（4 分・120 秒・50 回・80 ターン）、thinking 切り・出力 4,096、prompt で `search_similar_code` を禁止し `git grep`（`rg` は無い）を指示。10-02 の枠で出す。
- 10-01：TPU。`TpuV6E8` は 5.5 時間 QUEUED のまま（無料枠では割り当てられないと見る。公開の TPU notebook は 9 月に `TpuV5E8` 23 本・`TpuV38` 系 3 本）。
  「batch TPU session は同時 1 本」なので、待ち続けた kernel を workflow の `action: delete` で消し、`yasunorim/gemma4-tpu-probe-v5e`（`TpuV5E8`）で再 probe（tpuprobe-3）。
  **LoRA の重い注意（forum 報告）：adapter を入れると vLLM が LoRA 枠 8・rank 128 で起動し、4×L4 の KV cache が約 46k→約 7.6k トークンに縮む＝長い agent 会話が詰まる。**
  LoRA は「短い会話で済む agent」と組み合わせるか、KV の縮みを上回る効果が手元評価で出たときだけ出す。
- 10-02：TPU の再 probe（tpuprobe-3・`TpuV5E8`）も GHA の待ち上限（5.8 時間）まで QUEUED のまま＝**Kaggle の無料 TPU は今つかまらない**。kernel は Kaggle 側で待ち続けている（動けば後で出力を読む）。
  **g4v2-1 を提出**（v2・仮説：設定の脆い所を直せば再実行エラーが消え、LB 0.08 以上）。
- 10-03：**g4v2-1＝LB 0.03**（エラーは消えた・約 2 課題分）。公開 walkthrough は v1 系 0.06〜0.12。HARNESS_README と walkthrough の読解で分かった落とし穴：
  `write_file`／`read_file` は /workspace の外を拒む（v2 は「/tmp に検証スクリプトを書け」と指示＝失敗の連鎖）、`run_command` は出力の**先頭** 5,000 文字だけ返す（pytest の判定は末尾）、
  会話が 14k トークンを超えると古い道具の出力が消える（モデル自身の文は残る）。`{problem_description}` は instruction に差し込める（README 5.1）。
  ⇒ **v3**：検証スクリプトは `run_command` の heredoc で /tmp に、長い出力は `| tail`、所見を文で書かせる、issue を instruction の末尾に再掲。検証・コンパイル済み。
  **g4v3-1 を提出**（仮説：検証の失敗と文脈の消失が低得点の原因＝LB 0.06 以上）。
- 10-03：**TPU probe（`gemma4-tpu-probe-v5e`・`TpuV5E8`）は待ちの末に動いた**：v5litepod-8・RAM 377GB・jax 0.10.2・flax 0.12.7・keras 3.15・keras_hub 0.29.1・torch 2.8 (cpu)。
  probe 最後の torch_xla の import 付近で Segmentation fault（jax/keras_hub 系で組めば使える見込み）。割り当ては数時間待ち＝**LoRA 学習は夜間に 1 本投げて待つ運用**。KV の縮みの件があるので優先度は prompt 改良の後。
- 10-04：**g4v3-1＝LB 0.05**（v2 の 0.03 から上がった・公開 0.10〜0.12 には届かない）。公開の検証ノートブック（読むだけ）によると、0.12 の構成（coder＋analyzer の sub-agent）は **1 課題 8 分・道具 100 回・thinking 4,096** で動き、
  隠しテストは約 120 課題。当方の v2/v3 は 4 分・50 回・thinking 切り。
  ⇒ **v4**：prompt は v3 のまま、予算を 8 分・100 回・150 ターン・コマンド 180 秒、thinking 4,096（思考は返さない）・出力 8,192。検証・コンパイル済み。
  **g4v4-1 を提出**（仮説：予算と thinking 切りが 0.10 未満の原因＝LB 0.08 以上）。次の候補：analyzer の sub-agent（自作の prompt）・場所特定の手順の強化。
- 10-05：**g4v4-1＝LB 0.03**（v3 の 0.05 より下。公開 LB は隠し約 120 課題の約半分＝1 課題 ≈ 0.017 なので 1 課題差＝雑音の範囲）。
  予算を増やし thinking を入れても上がらない（forum：thinking の中身は道具呼び出しの間で捨てられ、毎回考え直す＝予算の浪費）。
  ⇒ **v5**：自作の**読むだけの locator sub-agent**（`sub_agents/locator.yaml`・`prompts/locator.md`・道具は run_command・read_file・get_code_neighbors。
  issue は state から `{problem_description}` で受け取る＝ADK の AgentTool は親の state を写すことを確認）が FILES／CAUSE／CHANGE／CODE／CHECK の形で返す。
  coder は最初に locator を 1 回呼び、名指しの行を 1 回読んで確かめてから編集。ファイルを読む会話が coder の 14k の窓に入らない。
  thinking 切り・5 分・80 回・コマンド 120 秒・120 ターン。検証・コンパイル済み。**g4v5-1 を提出**（仮説：別の文脈での場所特定が足りないもの＝LB 0.07 以上）。採点待ち。
  次の候補：locator の答えを `output_key` で state に残す、issue の種類（バグ／機能追加）で手順を分ける、LoRA は KV 縮みの件で後回し。
- 10-06：**g4v5-1＝LB 0.06（自己最高・+1 課題）**＝場所特定を別の文脈に出すのは効く方向（差は 1 課題で雑音の範囲だが、4 本中の最高）。
  ⇒ **v6**：自作の**読むだけの checker sub-agent**（`sub_agents/checker.yaml`・`prompts/checker.md`・道具は run_command・read_file）を編集の後に 1 回呼ぶ。
  git diff を見て、issue の例を /tmp の使い捨てスクリプトで再現し、近い既存テストだけを `| tail` 付きで回し、VERDICT／EVIDENCE／FIX で返す（長い pytest 出力が coder の窓に入らない・FAIL なら 1 回だけ直して再確認）。
  予算は v5 のまま。検証済み。**g4v6-1 を提出**（仮説：場所特定の次の損は未検証・壊れた編集＝LB 0.08 以上）。
