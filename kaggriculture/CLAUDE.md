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
- 土台＝`agents/v48_sched.py`（v47 と同一）。双子 v45 相手 96 試合で勝率 0.72・自分の金 77,771。
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

## ▶▶ 次の一手

1. **対 router 48 試合（`35938317910`）は読了＝全変種が base 以下**。margin（base 比）：
   m8 −1,389 tie／hc −2,053 tie／m8h −6,771 WORSE／m12 −13,505 WORSE／m12h −22,437 WORSE／m12hc −28,915 WORSE
   （base の自分の金 64,996・勝率は全変種 0.00）。**メロンを増やすほど単調に悪化**＝単品で開幕メロンを足す形は閉じてよい。
   **対 v48 双子 96 試合（`35938315207`）も全変種 WORSE**（base 自分の金 79,495・勝率 0.29）：margin m8 −2,989／m12hc −5,097／
   m12 −6,510／m8h −6,745／m12h −6,851／**hc −29,186**（`harvest_at_cap` 単体が最悪＝一発作物の早取りは害）。
   ⇒ **開幕メロンと早取りは両相手で閉じた**。ノブは既定オフのまま残す。
2. 負けていれば、**単品のノブではなく router 型の開幕を一式で**作る：0 日目はメロン＋家畜に集中し、
   トマト・イチゴを 0 日目に植えない・10 日目の現金を即座に家畜と土地へ回す。`diag/traj.py` で router と同じ日程の形になったかを
   先に確かめてから sweep に出す（機構 → 指標の順）。
3. 勝った変種は `sim/reemit.py agents/v49_sched.py --out agents/v50_sched.py --set k=v` で固定し、
   別 seed 帯で引き直して符号が保てば提出候補。

## 閉じた線（再提案しない・詳細な数字は過去の記録にあり）

capacity 一式（3 区画目・人手 1.25 倍・遊休地の埋め）／群れ拡大（通算 9 回）／肥料の重み（糞が世話を追い出す）／
`care_repeat`（同日 2 回目の世話は no-op）／開幕現金 `cash_buffer`／day-0 の家畜の顔ぶれ／市場の枠順／売りの繕い層／
`dist_weight` 0.7（0.9 が局所最適）／`lump_span`／トマト・にんじん 0／`crop_order`。
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

### 実地確認（2026-09-24・cloud セッション・既定ネットワーク）

1. **シミュレータは動く**：Python 3.11.15。`pip install kaggle-environments` は debian 管理の `blinker` を消せず失敗するので
   `pip install --ignore-installed blinker kaggle-environments`（約 1 分）。kaggle-environments 1.32.7 で `make("kaggriculture")` OK。
2. **probe も動く**：`OPP=agents/v48_sched.py python diag/probe.py '{"base":{}}' 86000` →
   `base 86000 [81991, 81991] day10money 136`・**実時間 8.5 秒**（2 席分）。数 seed の試走は cloud でできる。
   **同じ命令を RPi5（kaggle-environments 1.32.7）で回して 81,991 / 81,991・day10 136 と 1 円一致**＝cloud の試走は信頼できる。
3. **kaggle.com は届かない**：`curl https://www.kaggle.com` → 接続拒否（proxy が CONNECT を 403・organization policy）。
   pypi.org と api.github.com は 200。`~/.kaggle` も `KAGGLE_*` 環境変数も無し。
   ⇒ LB 取得・公開 kernel の取得（`sim/fetch_opponent.py`）・提出は cloud からはできない（環境のネットワーク設定の変更が要る）。
4. **`gh` は未インストール**。代わりに GitHub MCP ツールで run 一覧は読める（`actions_list`→ 最新は `Kaggriculture Tests` #52 success）。
   MCP に `actions_run_trigger` があるので dispatch もそちらで試せる見込み（未実行）。


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
