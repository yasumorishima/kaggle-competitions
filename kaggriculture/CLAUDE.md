# Kaggriculture — 作業の引き継ぎ（Claude Code 向け）

Kaggle `kaggriculture`（Featured・2 人対戦の農業経営シミュ・**締切 2026-09-30 23:59 UTC**）。
GPU は使わない。このファイルが作業の正本。**セッションの終わりに「現在地」と「次の一手」を必ず更新して commit する**
（ここに書かれていないことは次のセッションに残らない）。

報告は日本語・です/ます。作業中の思考は英語でよい。

## 最終順位の決まり方（一次資料で確認済み）

- 追跡されるのは**最新 2 提出だけ**で、最終評価もその 2 つ。古い提出は落ちる。提出は 1 日 5 本。
- 締切後 10/1〜10/15 頃に対局を回し Bradley-Terry で最終 LB を作る。**今の LB は持ち越されない**。
- **レーティングは勝敗だけで動く。コイン差は効かない**。
- 現在の最新 2 提出＝**v48_sched（2026-09-23 提出）と v45**。次に出すと v45 が押し出される
  ⇒ **v48_sched 以上だと直接対決で示せたものだけを出す**。

## 採否の物差し（ここを間違えると全部無駄になる）

- **表示 LB は版差を分解できない**（1 提出 ≈ 40 対局・1 試合の margin の散らばり ≈ 35,000）。LB で版を判定しない。
- **候補は現提出との直接対決で測る**。`kaggriculture-sweep.yml` で `agent_b=agents/v48_sched.py`（双子）
  を 96 試合（`episodes=48`・両側）、上位への距離は公開 router 相手 48 試合（`episodes=24`）。**`margin` 列（自分−相手）で読む**。
  `variants[0]` が ref。単発の BETTER は別 seed 帯で引き直す。
- **固定の第三者相手（`starter` 等）の順位を採否に使わない**（診断専用）。
- 1〜2 seed の試走は機構の確認にだけ使う。12 seed の平均差でも万級は主張しない（季節の sd 15,000〜21,000）。
- 新しいノブは**既定値で直前の版と 1 円一致**を確かめてから掃く（`sim/reemit.py` で再生成）。

## 現在地（2026-09-24）

- LB 6,269 位 / 9,478（09-19）。銅ライン 約 2,393。**銅には自分の金 ×1.5〜2.0 が要る**。
- **提出候補＝`agents/v51_sched.py`**（router 型の開幕一式・下の節）。未提出。
- 提出中の土台＝`agents/v48_sched.py`（v47 と同一）。双子 v45 相手 96 試合で勝率 0.72・自分の金 77,771。
- **上位との距離**：v48_sched は公開リプレイ boatlee に **0/32・69,558 対 131,411**（`kaggriculture-panel.yml` の band 4 行）。
  公開 router に seed 86000 で 85,673 対 169,076。
- **行動列の登坂チェーン（`kaggriculture-optimize.yml`・6 時間ごとの schedule で自動）は負けている**：
  `mean` 目的で対 v48_sched holdout 0/6、`margin` 目的でも 2 リンク連続 0/24（−22,445／−23,576）、
  訓練 seed が上がる間に holdout は悪化（過学習）。チェーンは止めなくてよいが、**判定材料にしない**。

### 🔑 router が 2 倍稼ぐ仕組み（2026-09-24 に `diag/traj.py` で測定）

router は **0 日目にメロン 12 区画＋牛 2＋羊 2** に現金を使い切り（残金 $37）、**10 日目にメロンを一斉収穫して現金 16,689**
（当方 319）。その資金で 8 日目までに牛 9・羊 4、11 日目に 3 区画目、イチゴ 33 区画へ拡大する。
当方の開幕は 小麦 6・トマト 6・イチゴ 3・メロン 2。

メロンは町が補充しない一発の池（`sq` 曲線・72 個で約 16,400）＝**先に売った側が取る**。当方が取れない理由は 2 つ：
1. `market_cap` が相手の供給を差し引き、メロンを 2 区画しか計画しない（旧ノブ `melon16` が一度も発火しなかった理由）
2. `plant_ready` が一発作物を `max_yield_day`（メロン 12 日）まで収穫しない。齢 6〜10 に毎日水やりすれば 10 日目に既に上限 6 個

`32d90ef` でノブを追加（**既定オフ・v49_sched ≡ v48 を seed 86000/82000 で 1 円一致・テスト 9 本 PASS**）：
- `open_melon`: `((last_day, tiles), ...)`＝その日まで最低 tiles 区画のメロンを計画し、種をメロン優先で買う
- `harvest_at_cap`: 一発作物を上限到達で即収穫

2 seed の試走（対 router）では全変種が base より悪化。メロン 12 区画は植わるが現金が種に吸われて家畜が遅れる。
`harvest_at_cap` 単体でも seed 82000 で 24 日目に牛が全部消える連鎖が出た（1 試合の分岐が大きい系）。

### ✅ router 型の開幕一式（2026-09-24・cloud セッション）＝**v51_sched が提出候補**

`main.py` にノブ 2 本（既定オフ・**v50_sched ≡ v49 ≡ v48 を router 86000/82000・双子 86000 で 1 円一致**・テスト全 PASS）：
- `open_crop_zero`: `((last_day, [crops]), ...)`＝その日まで指定作物の計画を 0（0 日目にトマト・イチゴを植えない）
- `harvest_at_cap` に作物名のリストも渡せるように（`["MELON"]`＝メロンだけ早取り）

変種（すべて v50_sched 上。r1 系の共通部＝`open_melon [[0,12]]`＋`open_crop_zero [[0,["TOMATO","STRAWBERRY"]]]`＋
`sched_herd_floor [[0,{COW:2,SHEEP:2}],[6,{COW:2,SHEEP:3}]]`）。**sweep は cloud コンテナ内で `sim/sweep.py` を直接実行**（GHA dispatch は 403）：

| 変種 | 対 v48 双子 96 試合（seed0 90000） | 対 公開 router 48 試合（seed0 91000） |
|---|---|---|
| **r1n**（共通部のみ） | **勝率 0.84・margin +5,097 ± 1,296 BETTER**・自分の金 81,997（base 81,251） | margin −5,759 ± 8,649 **tie**・自分の金 61,060（base 60,641） |
| r1（＋メロン早取り） | margin −2,968 ± 3,527 tie | margin −15,020 ± 7,422 WORSE |
| r1l（r1＋11 日目 3 区画目） | margin −12,630 ± 5,495 WORSE | margin −14,704 ± 9,169 WORSE |

- **r1n は別 seed 帯（92000〜92031・両側 64 試合）で margin +7,315 ± 2,188＝HELD**。
- 機構（`diag/traj.py` seed 90000 対 双子）：0 日目 メロン 12・小麦 6・牛 2・羊 1（router は羊 2＝現金不足）、9 日目に牛 7、
  **12 日目にメロンを売って現金 12,847**（双子 5,709）→ 即座に牛 9・羊 4。router との違いは収穫が 12 日目（router は 10 日目）。
- 早取り（`harvest_at_cap`）は今回も害（r1 < r1n が両相手で一貫）。**router と同日に売り合うと後手で安く売るだけ**。
  10 日目の現金で 3 区画目を買う（r1l）のも害。
- 固定：`agents/v51_sched.py`＝v50＋r1n（sweep の r1n と seed 90000 で 1 円一致）。

## ▶▶ 次の一手

0. **v51_sched を提出するかの判断**（user の指示があれば下の「依頼ファイル」で cloud が PR 作成→merge まで）：双子 v48 に BETTER＋HELD、router に tie＝「v48_sched 以上だと直接対決で
   示せた」条件は満たす。出すと v45 が押し出され、最新 2 本＝v51_sched と v48_sched になる。
   **GHA で別 seed 帯（180000〜・cloud 未使用）でも再現**：双子 96 試合 勝率 0.85・margin **+6,040 ± 1,657 BETTER**（run `35961918542`）、
   router 48 試合 −5,543 ± 7,224 tie（run `35961921180`）。
   提出前にもう 1 帯（seed0 ≠ 90000/92000）で対 v48 双子を引き直せればなお良い（cloud 内 `sim/sweep.py` で 96 試合 約 15 分）。
1. 次の構造の手（v51 を土台に）：0 日目の羊 2 頭目（現金不足で 1 頭止まり＝day-0 の飼料買い `feed_buy_days` を削るか）、
   router のように 2〜9 日目に糞（肥料）を売って牛を 1 頭ずつ足す流れ。どちらも v51 を base に双子＋router で測る。

### （前回まで）開幕メロン単品と早取り

1. **対 router 48 試合（`35938317910`）は読了＝全変種が base 以下**。margin（base 比）：
   m8 −1,389 tie／hc −2,053 tie／m8h −6,771 WORSE／m12 −13,505 WORSE／m12h −22,437 WORSE／m12hc −28,915 WORSE
   （base の自分の金 64,996・勝率は全変種 0.00）。**メロンを増やすほど単調に悪化**＝単品で開幕メロンを足す形は閉じてよい。
   **対 v48 双子 96 試合（`35938315207`）も全変種 WORSE**（base 自分の金 79,495・勝率 0.29）：margin m8 −2,989／m12hc −5,097／
   m12 −6,510／m8h −6,745／m12h −6,851／**hc −29,186**（`harvest_at_cap` 単体が最悪＝一発作物の早取りは害）。
   ⇒ **開幕メロンと早取りは両相手で閉じた**。ノブは既定オフのまま残す。
2. （→ 上の「router 型の開幕一式」で実施済み）**単品のノブではなく router 型の開幕を一式で**作る：0 日目はメロン＋家畜に集中し、
   トマト・イチゴを 0 日目に植えない・10 日目の現金を即座に家畜と土地へ回す。`diag/traj.py` で router と同じ日程の形になったかを
   先に確かめてから sweep に出す（機構 → 指標の順）。
3. 勝った変種は `sim/reemit.py agents/v49_sched.py --out agents/v50_sched.py --set k=v` で固定し、
   別 seed 帯で引き直して符号が保てば提出候補。

## cloud から GHA を動かす仕方（依頼ファイル・2026-09-24〜）

cloud は push と PR はできるが workflow の dispatch はできない（403）。**依頼ファイルを push すると GHA が走り、結果がファイルで戻る**。

- **sweep**：作業ブランチで `kaggriculture/requests/sweep.json` を書いて push。中身は sweep の入力と同じ
  （`memo`・`variants`・`agent_a`・`agent_b`・`episodes`・`seed0`・`replicate`）。**前回と同じ中身だと走らない**（ファイルが変わった push だけが起点）＝memo を変える。
  結果は GHA が**同じブランチ**へ `kaggriculture/requests/results/sweep-<run id>.txt` として commit する（10〜60 分）＝`git pull` で読む。
  2026-09-24 に動作確認済み（run `35963153234`）。重い掃引はコンテナでなくこちらで（ランナーの並列が使える）。
- **提出**：`kaggriculture/requests/submit.json`（`memo`・`agent`・`message`・`evidence`・`confirm: "submit"`・`dry_run`）を
  **PR に入れて main に merge する（main に入った時だけ走る）**。`dry_run` を省くと true（提出しない）。
  結果は main に `kaggriculture/requests/results/submit-<run id>.txt`。2026-09-24 に v48 の dry run で動作確認済み（run `35963501480`）。
  **本番（`dry_run: false`）はまだ一度も走っていない**。
- 提出 workflow の止め：main のみ／symlink 不可／単独で 720 ステップ完走／提出文に `[run <id> md5 <12桁>]` を付け、
  同じ md5 が一覧にあれば止める／再実行では提出しない／提出後に一覧を読み直して無ければ失敗（CLI は 404 でも終了コード 0）。
- 🔴 **提出は user がそのセッションで明示的に指示した agent だけ**（最新 2 本だけが最終評価に残る＝古い 1 本を押し出す）。
  指示があれば **cloud が PR を作って自分で merge まで行い**、`requests/results/submit-<run id>.txt` を `git pull` で読んで結果を報告する。
  指示の無い提出・dry_run: false の独断は禁止。1 回に 1 本。

## 閉じた線（再提案しない・詳細な数字は過去の記録にあり）

capacity 一式（3 区画目・人手 1.25 倍・遊休地の埋め）／群れ拡大（通算 9 回）／肥料の重み（糞が世話を追い出す）／
`care_repeat`（同日 2 回目の世話は no-op）／開幕現金 `cash_buffer`／day-0 の家畜の顔ぶれ／市場の枠順／売りの繕い層／
`dist_weight` 0.7（0.9 が局所最適）／`lump_span`／トマト・にんじん 0／`crop_order`。
開幕メロン単品（`open_melon` だけ）／一発作物の早取り（`harvest_at_cap`・メロン限定でも害）／10〜11 日目の 3 区画目＋家畜の積み増し（r1l・r2）。
**「取り合う商品の生産量を増やす手」は対戦ではおおよそゼロ**（第三者相手で輝いて実戦で消える型）。

## 禁止事項

- ⛔ **公開 notebook・公開リプレイを fork してそのまま（または少し変えて）提出しない**。公開実装は対戦相手と読解の対象としてだけ使う。
- ⛔ `starter` 相手の録画を開始列にしない。
- ⛔ 中位の刻み（ノブを 1 本ずつ微調整）を成果として扱わない。狙いは上位＝構造を変える手。
- ⛔ 提出は LB を見るためではなく、最新 2 本を入れ替える意思決定としてだけ行う。

## 道具

- エンジン：`main.py`（方策）、`agents/*_sched.py`（カレンダーとノブを固定した版）。
  カレンダー行（`SCHEDULE = {...}`）は巨大なので grep で表示しない。
- `sim/sweep.py`（`kaggriculture-sweep.yml`）＝変種掃引。`agent_b` に `kernel:owner/slug` で公開 kernel を相手にできる
  （公開 router は `kernel:thomastschinkel/kaggriculture-93-8-win-rate-public-state-router`）。GHA 手動実行は `-f memo=` 必須。
- `kaggriculture-panel.yml`（`-f agent_a=`）＝帯別の相手パネル。見るのは band 4（boatlee）の行。
- `kaggriculture-optimize.yml`＝行動列の登坂（自動・締切後に `gate` job で自分で止まる）。
- `diag/traj.py`（日次の状態）・`diag/ledger.py`（商品別の実約定・注文でなく約定を数える）・`diag/probe.py`（数 seed の試走）。
  相手はローカルでは `python sim/fetch_opponent.py --exec <owner/slug>` で `opponents/` に取得（git には入れない）。
  1 試合 10〜15 秒。重い掃引はローカルで回さず GHA へ。
- `sim/test_*.py`＝push/PR で `Kaggriculture Tests` が走る。commit 前にローカルでも全部通す。

## cloud 版で動かすときの注意（公式 docs 2026-09-24 時点・未実地確認）

- push できるのは**セッションの作業ブランチだけ**＝main へは PR を立てて merge する（自分のリポなので即 merge でよい）。
  workflow は main の定義で走るので、`main.py` や agent を変えたら **merge してから** sweep を dispatch する。
- 既定のネットワークは PyPI 等のみ＝**kaggle.com は許可リスト外**。LB 取得・公開 kernel の取得には環境でドメイン追加か Full が要る。
  `KAGGLE_API_TOKEN` は環境変数でなく **API credentials** に置く（環境変数は共有メンバー全員に見える）。
- `gh` で workflow を dispatch できるかは docs に記載なし＝最初のセッションで確かめてここに書く。

### 実地確認（2026-09-24・cloud セッション。1〜4 は既定ネットワーク、5 は設定変更後）

1. **シミュレータは動く**：Python 3.11.15。`pip install kaggle-environments` は debian 管理の `blinker` を消せず失敗するので
   `pip install --ignore-installed blinker kaggle-environments`（約 1 分）。kaggle-environments 1.32.7 で `make("kaggriculture")` OK。
2. **probe も動く**：`OPP=agents/v48_sched.py python diag/probe.py '{"base":{}}' 86000` →
   `base 86000 [81991, 81991] day10money 136`・**実時間 8.5 秒**（2 席分）。数 seed の試走は cloud でできる。
   **同じ命令を RPi5（kaggle-environments 1.32.7）で回して 81,991 / 81,991・day10 136 と 1 円一致**＝cloud の試走は信頼できる。
3. ~~kaggle.com は届かない~~（初回セッション時点。**5 で更新**）：当初は `www.kaggle.com` も proxy が CONNECT を 403。
   pypi.org と api.github.com は 200。`~/.kaggle` も `KAGGLE_*` 環境変数も無し。
4. **`gh` は未インストール**。代わりに GitHub MCP ツールで run 一覧は読める（`actions_list`→ 最新は `Kaggriculture Tests` #52 success）。
   MCP に `actions_run_trigger` があるので dispatch もそちらで試せる見込み（未実行）。
5. **Kaggle 接続（2026-09-24・環境設定の変更後）**：
   - `curl https://www.kaggle.com/api/v1/competitions/list` → **200**（JSON の大会一覧が返る）。
     ただし一覧は公開情報なので、200 だけでは proxy のトークン付与（認証）が効いている証明にはならない。
   - `pip install --ignore-installed blinker 'kaggle==2.0.0' 'kagglesdk==0.1.15'` は約 16 秒で入る。
   - `KAGGLE_API_TOKEN=dummy kaggle competitions leaderboard kaggriculture --show` → **失敗**：
     `ProxyError ... host='api.kaggle.com' ... /v1/security.OAuthService/IntrospectToken ... Tunnel connection failed: 403 Forbidden`。
     kaggle 2.0 CLI は `www.kaggle.com` でなく **`api.kaggle.com`** に行き、そこが許可リスト外（`curl https://api.kaggle.com/` も CONNECT 403）。
   - ⇒ CLI で LB・提出・kernel 取得をするには、環境の許可ドメインに **`api.kaggle.com` を追加**（`*.kaggle.com` 可ならそれ）が要る。
     それまでは `www.kaggle.com/api/v1/...` を curl で直接叩くのが唯一の経路。
   - ⚠️ 訂正：許可ドメインに `api.kaggle.com` を足すだけでは**繋がってもトークンが付かない**。proxy がトークンを付けるのは
     **API credentials の「許可ウェブサイト」に書いたホストだけ**（今は `www.kaggle.com` のみ）で、そこに書いたホストは
     ネットワーク設定に関係なく届く（docs）。CLI を使うなら認証情報を `*.kaggle.com` で登録し直す（編集不可＝削除して再追加・値が要る）。
   - ▶ 先に試すこと：今の登録のまま **認証が要る** `www.kaggle.com` の endpoint が通るか。
     `curl --max-time 20 -sS https://www.kaggle.com/api/v1/competitions/submissions/list/kaggriculture | head -c 400`
     が自分の提出（v48_sched 等）を返せば、認証は効いている＝CLI を使わず curl で LB・提出一覧が取れる。
   - ✅ **2026-09-24 実測：通った**＝自分の提出一覧（最新 v48・publicScore 610.1）が返った。**認証は今の登録（`www.kaggle.com`）で効いている**
     ⇒ 提出一覧・LB は **CLI を使わず `www.kaggle.com/api/v1/...` を直接 GET して読む**。認証情報の登録し直しは不要。
     提出は Kaggle の CLI が `api.kaggle.com` に行くので cloud からは直接出せない＝上の「依頼ファイル」で GHA に出させる。
6. **GHA の dispatch は cloud からできない（2026-09-24 実測）**：GitHub MCP の `actions_run_trigger`（`run_workflow`・
   `kaggriculture-sweep.yml`・ref＝作業ブランチ）→ **`403 Resource not accessible by integration`**（Claude の GitHub App に
   Actions の書き込み権限が無い）。`gh` も無い。⇒ **sweep は cloud コンテナ内で `sim/sweep.py` を直接回す**（4 コア・
   `--workers 4`）か、RPi5 から dispatch する。コンテナ内の sweep は GHA と同じ `sim/sweep.py` なので数字の読み方は同じ。
7. **公開 kernel（router）は CLI 無しで取れる**：`curl "https://www.kaggle.com/api/v1/kernels/pull?userName=<owner>&kernelSlug=<slug>"`
   の JSON の `blob.source` が notebook 本体。これを `.ipynb` に書き、`sim/fetch_opponent.py` の `code_cells`＋`from_writefile`
   で `opponents/` に書き出す（router は `%%writefile` 型で `--exec` 不要）。取れた router で seed 86000 が 85,673 対 169,076 と記録に 1 円一致。


## 環境の確定事項（一次資料＝interpreter）

- 動物：COW $400・初乳 8 日目・隔日／SHEEP $500・6 日目・3 日ごと／GOOSE $300・4 日目・毎日。開幕資金 3,000。
- 肥料は全頭が餌と無関係に毎日 1 個出す。boolean で毎日上書き＝取らなければ消える。町は肥料を消費しない（有限の池）。
- 市場：MELON と WOOL は `sq`（崩れる）、STRAWBERRY と MILK は `linear`、WHEAT と EGG は `log`（崩れない）。
  メロンを買う店は無い（町の中心が 1 日 1 個だけ）。
- 店は 3 日ごとに 1 軒・最大 8 軒。抽選は盤面依存で seed だけでは季節が決まらない。
- CARE は 1 頭 1 日 1 回が全部（同日 2 回目は no-op）。

## 締切後の片付け（10-01）

- `kaggriculture-optimize.yml` の `schedule:` を消す。
- 別機で動いている backstop（optimize が止まっていたら dispatch する cron）も user 側で外す。
