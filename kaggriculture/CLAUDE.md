# Kaggriculture — 作業の引き継ぎ（Claude Code 向け）

Kaggle `kaggriculture`（Featured・2 人対戦の農業経営シミュ・**締切 2026-09-30 23:59 UTC**）。
GPU は使わない。このファイルが作業の正本。**セッションの終わりに「現在地」と「次の一手」を必ず更新して commit する**
（ここに書かれていないことは次のセッションに残らない）。

報告は日本語・です/ます。作業中の思考は英語でよい。

**このプロジェクトは cloud セッションだけで完結させる。** 行き詰まっても、user に別の場所（手元の PC など）へ持っていかせない。
原因を自分で調べて直し、どうしても user にしかできない操作（設定画面の変更など）があるときだけ、その操作を 1 つ具体的に頼む。

## 最終順位の決まり方（一次資料で確認済み）

- 追跡されるのは**最新 2 提出だけ**で、最終評価もその 2 つ。古い提出は落ちる。提出は 1 日 5 本。
- 締切後 10/1〜10/15 頃に対局を回し Bradley-Terry で最終 LB を作る。**今の LB は持ち越されない**。
- **レーティングは勝敗だけで動く。コイン差は効かない**。
- 現在の最新 2 提出＝**v64_sched（2026-09-25 5 本目・下の節）と v63_sched（2026-09-25 05:59 UTC・run `36100768348`）**。v62 以前は押し出された。
  2026-09-24・09-25 とも 5 本（上限）。次に出せるのは UTC 09-26 から。
- LB（09-24）：1 位 3,123・10 位 2,945・20 位 2,871（金圏 ≈ 2,900）。自分 v55 622.4・v56 605.8。
  次に出すと古い方が押し出される ⇒ **現提出以上だと直接対決で示せたものは聞かずに出す**（下の「提出の決まり」）。

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
- **提出中の土台＝`agents/v51_sched.py`**（router 型の開幕一式・下の節）。2026-09-24 06:40 UTC 提出・publicScore 542.5（v48 は 582.6・LB で版は判定しない）。
- **最新の提出＝`agents/v53_sched.py`**（v51＋肥料販売＋家畜先行の開幕・下の「糞を売って牛を足す」節）。双子 v51 に BETTER＋HELD・router に tie。
  2026-09-24 09:17 UTC 提出（run `35980271383`・md5 `914116103e2a`）・提出直後は PENDING。**次の候補は v53 に双子で勝つもの**。
- 1 つ前の提出＝`agents/v48_sched.py`（v47 と同一）。双子 v45 相手 96 試合で勝率 0.72・自分の金 77,771。
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

### ✅ 糞を売って牛を足す開幕（2026-09-24・cloud セッション）＝**v53_sched が提出候補**

`main.py` にノブ 2 本（既定オフ・**v52_sched＝v51 を再生成して router／双子 × seed 90000／82000 で 1 円一致**・テスト 9 本 PASS）：
- `fert_sell_days`: その日まで施肥しない・肥料を手元に残さない（`reserve_by_item {"FERTILIZER":0}` と組で使う）
- `pending_by_species`: 家畜の不足数から**同じ種の**運搬中・shed 分だけを引く（-1＝オフ・値はその日から）

診断（`diag/traj.py`＋計測版、seed 90000 対 router）：
- v51 は 1〜8 日目に糞を毎日 3〜6 個集めるが、`reserve_frac 1.0`（100 円未満で売らない）で売らず、毎日 3 個を小麦・メロンに施肥していた。
  router は同じ量を毎日売って 2〜8 日目に牛を 1 頭ずつ足す。
- 0 日目の羊が 1 頭なのは `need = 目標 − 手元 − shed − pending` の `pending` に**同じターンに買った牛 2 頭**が入るため。
  直す（`pending_by_species 0`）と羊 2 頭を買うが飼料代が消え、**4 日目までに家畜が全部逃げる**。1 日目からなら v51 と 1 円一致（無効）。
  ⇒ **0 日目の羊 2 頭目は閉じた**（現金が足りない。router は小麦の売買で回している）。
- 肥料を売らせても（fs9）増えた現金は土地と作物に回り、牛は増えない。牛の床を上げても日程は同じ：3〜5 日目は 1 区画目が作物で満杯（room 0）、
  6〜7 日目は種と土地が先に現金を使う、7 日目以降は forward の回収見込み（payback 21）で拒否。
  ⇒ `opening_days 9`（家畜が種より先）＋ 5 日目までトマト・イチゴ・ニンジン 0 を足した一式（coz）で牛 5 日目 3 → 6 日目 4 → 9 日目 5。

変種（v52 上。F＝`fert_sell_days 9`＋`reserve_by_item {"FERTILIZER":0.0}`、Z＝`sched_herd_floor [[0,{COW2,SHEEP2}],[3,{COW3,SHEEP2}],[5,{COW4,SHEEP2}],[6,{COW5,SHEEP3}],[8,{COW6,SHEEP3}]]`
＋`opening_days 9`＋`open_crop_zero [[5,["TOMATO","STRAWBERRY","CARROT"]]]`）：

| 変種 | 対 v51 双子 96 試合（GHA run `35975586359`・seed0 190000） | 対 公開 router 48 試合（cloud・seed0 191000） |
|---|---|---|
| **coz**（F＋Z） | **勝率 0.89・margin +8,194 ± 1,702 BETTER**・192000〜 64 試合 **+7,364 ± 1,466 HELD** | margin +3,783 ± 7,157 tie（自分の金 57,421・base 63,182） |
| fs9（F のみ） | 勝率 0.80・margin +3,920 ± 1,408 BETTER | margin +4,024 ± 4,835 tie（自分の金 63,406） |
| z（Z のみ） | margin +450 ± 1,611 tie | margin −917 ± 9,331 tie |

- 肥料販売が効きの本体で、Z は F と組んだときだけ上乗せ（z 単体は tie）。router 相手は全変種 tie（勝率 0.00 は不変）。
- 固定：`agents/v53_sched.py`＝v52＋coz（`reemit.rebuild(extra=...)`・変種 coz と router seed 90000／82000 で 1 円一致）。

### ✅ v53 の上の掃引（2026-09-24・cloud・3 回）＝**v55_sched を提出**

base＝`agents/v54_sched.py`（v53 を再生成・ノブ `sched_veto_from` 追加・v53 と router／双子 × 90000／82000 で 1 円一致）。双子の相手は v53_sched。

| 回 | 変種 | 対 v53 双子 96 試合（GHA） | 対 router 48 試合（cloud） |
|---|---|---|---|
| 2（`35981074529`） | nv10（veto は 10 日目から）／nv（veto 無し）／f5・f14（肥料販売 5・14 日）／r8（10 日目から牛 8 の床） | nv WORSE −1,777、他は tie（f5 は引き直しで +2,793 HELD だが本番 tie）。**r8 は base と完全一致＝床が効かない** | 全 tie |
| 3（`35986013280`） | q3（12 日目に 3 区画目）／q3v（＋veto 無し）／q3hv（＋12 日目から牛 8・羊 6 の床）／hv（床＋veto 無し） | q3 −7,053・q3v −9,681・q3hv −7,371 **WORSE**、hv +1,813 BETTER だが引き直し +178 NOT CONFIRMED | 全 WORSE（hv −4,438・q3 −8,939・q3hv −9,545・q3v −13,508） |
| 4（`35989955738`） | **ml11（メロンの植え付けを 11 日目まで＝2 回目のメロンを植えない）**／ml11f5／f5／st15（イチゴ ×1.5） | **ml11 勝率 0.62・margin +2,042 ± 1,179 BETTER・222000〜 +2,013 ± 932 HELD**、ml11f5 +2,008 BETTER、他 tie | ml11 −549 tie、他も tie |

- v53 対 router の traj（seed 90000）：v53 は 15 日目に現金 15,225・27 日目 45,130 を抱えたまま牛 4・羊 4・2 区画。router は 12 日目までに 3 区画・羊 10・牛 7・イチゴ 32。
  **ただし 3 区画目・群れの積み増しは両相手で WORSE**（取り合う市場を増やすだけ）＝閉じた線に追加。
- v53 は 12 日目以降に 5 区画のメロンを植え直し、24 日目に崩れた値で売っていた。**2 回目のメロンを止める（ml11）が効く**。
- メロンの売り渋り（`reserve_by_item MELON 0`）は試走で差なし（メロンは 2 日で売り切れている）。
- 固定：`agents/v55_sched.py`＝v54＋ml11（変種と router 90000／82000 で 1 円一致・md5 `860f2c802da5`）。

### ✅ 植え付けの終期（2026-09-24・cloud・v55 の上・双子の相手は v55）＝**v56_sched を提出**

| 回 | 変種（`plant_last_day`・v55 は 小麦 26・ニンジン 27・メロン 11・トマト 21・イチゴ 19） | 対 v55 双子 96 試合（GHA） | 対 router 48 試合 |
|---|---|---|---|
| 5（`35993235746`） | wh24（小麦 24・ニンジン 25）／st16（イチゴ 16）／tm16（トマト 16）／ml5・ml0（メロン 5・0） | wh24 勝率 0.80・+807 ± 190 BETTER・引き直し +868 ± 204 HELD、st16 +548・tm16 +440 BETTER、**ml5・ml0 は base と完全一致** | wh24 +482 ± 209 BETTER、他 tie |
| 6（`35998204678`） | **all（小麦 24・ニンジン 25・イチゴ 16・トマト 16）**／whst／whtm／wh24／wh22 | **all 勝率 0.71・+1,268 ± 568 BETTER・242000〜 +1,263 ± 332 HELD**、whst 0.75・+1,181、whtm +812、wh24 +750、**wh22 −367 WORSE** | all +540 tie、wh24 +397・whst +316 BETTER、**wh22 −1,624 WORSE** |

- 終盤に植えても収穫前に季節が終わる／崩れた値で売る分を止める手。小麦・ニンジンは 24／25 が良く、22／23 は早すぎ。
- 固定：`agents/v56_sched.py`＝v55＋all（変種と router 90000／82000 で 1 円一致・md5 `d04eb68980cb`）。

### v56 の上の掃引（2026-09-24・cloud・双子の相手は v56）＝**勝つ候補なし**

| 回 | 変種 | 対 v56 双子 96 試合（GHA） | 対 router 48 試合 |
|---|---|---|---|
| 7（`36001152144`） | st14／tm14／sttm14（イチゴ・トマト終期 14）・dump28・f12（肥料販売 12 日） | st14 −63 tie、**tm14 は base と完全一致**、f12 −967・**dump28 −2,070 WORSE** | 全 tie、dump28 −1,279 WORSE |
| 8（`36005569602`） | h115・h085（`sched_hands_scale`）・ap10・ap15（`animal_payback`）・st13（イチゴ ×1.3） | st13 +701・h115 +527・ap15 +181 tie（st13 引き直し −72）、**h085 −6,642 WORSE** | 全 tie |

⇒ 植え付け終期・投げ売り日・肥料販売日数・人手の倍率・回収倍率は v56 の値でほぼ局所最適。次は構造の手が要る。

### ✅ 牛を飢えさせない（2026-09-24・cloud）＝**v57_sched を提出**

- `diag/ledger.py` seed 90000：v56 の牛乳は 46 個（router 219 個）。**牛が 12 日目から餌を抜かれ 21 日目までに全頭逃げる**
  （2 日連続の未給餌で逃走）。小麦は shed に 16 あり、餌やりの価値＝牛乳の値 ×2 が他の仕事に負けていた（router が牛乳を流して値が安い）。
- 仕様（interpreter）：牛は隔日に基本 1＋「餌と世話の両方がある日」ごとに +1 の貯め＝**毎日両方で 1.5 個／日、世話なし 0.5 個／日**。
- 第 9 回（v56 上・双子の相手は v56・GHA `36011303071`）：

| 変種 | 対 v56 双子 96 試合 | 対 router 48 試合 |
|---|---|---|
| **asset**（`feed_value_rule "asset"`＝餌の価値に肥料も足す） | **勝率 0.66・+2,546 ± 1,642 BETTER・272000〜 +2,225 ± 1,843 HELD** | −2,784 ± 4,353 tie |
| asseth（＋人手 1.15） | +1,404 ± 1,365 BETTER | tie |
| esc（`"escape"`）／esch | +875／+182 tie | tie |

- 固定：`agents/v57_sched.py`＝v56＋asset（変種と router 90000／82000 で 1 円一致・md5 `ba4eaa846f8a`）。
- 🔎 **公開 router の正体**：自前の方策ではなく、**上位の公開リプレイから抜いた行動テープを決定木で選んで再生するだけ**（`_TAPES`／`_TREES`・6 日ごとに選び直し）。
  router が 2 倍稼ぐのは上位の行動そのものだから。公開リプレイの流用は禁止事項なので、この手は使わない。

### ✅ メロンを router より先に売る（2026-09-25・cloud・v57 の上・双子の相手は v57）＝**v58_sched を提出**

| 変種（v57 上） | 対 v57 双子 96 試合（GHA） | 対 router 48 試合 |
|---|---|---|
| ガチョウの床（g3d6／g6d6／g6d8／g4d3・`36070463224`） | 全 WORSE（−7,045〜−17,781） | WORSE／tie |
| 羊の床 5／6（＋veto 無し）（`36072883772`） | 全 tie（s6 +334・引き直し +343） | s6 +2,999 BETTER、再試（`36076120638`）で +419 tie |
| イチゴ ×0.5／×0（`36076120638`） | st05 −4,170・st0 −16,643 WORSE | st0 WORSE |
| **m9f6（`fert_sell_days 6`＋`harvest_at_cap ["MELON"]`＋メロン終期 8）**（`36077725511`） | **勝率 0.93・+8,672 ± 1,431 BETTER・312000〜 +10,205 ± 1,498 HELD** | **+3,292 ± 3,034 BETTER**（router 相手で初の BETTER） |
| m9（肥料 5 日・メロン終期 11）／m9L8／m9L4 | +6,540 BETTER（3 つ同値） | m9 −4,794 tie、L8/L4 +2,289 tie |

- 仕組み：メロンは齢 6〜12 に水やりで +1／日、**施肥中は +2／日**。v57 は肥料を 9 日目まで売っていたので施肥されず、上限 6 個が 12 日目。
  肥料販売を 6 日目までにして齢 6〜8 に施肥すると **8〜9 日目に上限**、`harvest_at_cap` で即収穫＝**10 日目の現金 約 9,000〜10,000（v57 は約 130）**。
  以前の「早取りは害」は施肥なしで router と同日（10 日目）に売り合ったため。先に上限へ届かせるのが鍵。
- 固定：`agents/v58_sched.py`＝v57＋m9f6（変種と router 90000／82000 で 1 円一致・md5 `19b6417923d1`）。
- 閉じた線に追加：ガチョウ（卵）の床・イチゴ削減・羊の床単独。

### ✅ メロンの収穫と売り（2026-09-25・cloud・v58 の上・双子の相手は v58）＝**v60_sched を提出**

- v58 の上の第 1 回（`36081031364`）：mel14（メロン 14 区画）・f5・f7・q3d10（10 日目に 3 区画目）・h10（10 日目から牛 8・羊 5）＝**双子で全 tie／WORSE**、router は q3d10 WORSE・他 tie。
- 糞の行方（seed 90000）：v58 は 247 回集め 124 を施肥・78 を売る。router は 362 集め 75 施肥・334 売る（家畜日 223 対 395）。
  ノブ `fert_sell_from`（その日から再び売る）：fs9／fs12／fs20／h10fs9 は**双子・router とも全 WORSE**（`36083870617`・−5,457〜−7,451）＝**施肥は売るより得**。閉じた。
- 🔑 **メロンは植えて 10 日経つまで収穫できない**（interpreter の HARVEST：`day − planted_day < first_yield_day` なら無視・メロンは 10）。
  v58 は 9 日目に上限到達 → `harvest_at_cap` が収穫を出し続け、**9 日目の午後に最大 7 人が拒否される収穫に張り付いていた**。
  さらに 10 日目の売りは slice で少しずつ（15 時〜翌 0 時に 230→104）、router は 10 時に 60 個を一気に売って高値を取っていた。
- ノブ 2 本（既定オフ・v59＝v58 を再生成して 1 円一致）：`hc_respect_first`（first_yield_day 前は収穫しない）・`sell_whole`（指定品は shed に入ったターンに全部売る）。

| 変種（v59 上・`36085756363`） | 対 v58 双子 96 試合 | 対 router 48 試合 |
|---|---|---|
| **hrsw**（両方） | **勝率 0.84・+3,885 ± 1,209 BETTER・342000〜 +4,621 ± 1,486 HELD** | −996 tie |
| sw（全部売りのみ） | 勝率 0.93・+2,708 ± 255 BETTER（引き直し無し） | −1,138 tie |
| hr（収穫待ちのみ） | +645 tie | tie |

- 固定：`agents/v60_sched.py`＝v59＋hrsw（変種と router 90000／82000 で 1 円一致）。

### ✅ 収穫したメロンを即 shed へ（2026-09-25・cloud・v60 の上）＝**v62_sched を提出**

- v60 上の `sell_whole` 拡張（`36087603353`・双子の相手 v60）：swW／swM／swS／swAll は双子で tie（swS +59 BETTER は小さすぎ・swM 引き直し −368）。router で swM +2,212・swS +639 BETTER。
- traj（v60・seed 90000／82000）：メロンは 10 日目 4〜9 時に収穫されるが、**手が持ったまま他の仕事をして 15〜16 時に売る**。router は 10 時に 60 個を 266→232 で売り、当方は 220→104。
  `drop_urgency`／`drop_load` を上げると全品で shed 往復が増えて壊滅（seed 82000 で 6,000〜10,000）。
- ノブ `rush_items`／`rush_weight`（既定オフ・v61＝v60 を再生成して 1 円一致）：指定品を持った手はすぐ shed へ戻る。

| 変種（v61 上・`36090747125`・双子の相手 v60） | 対 v60 双子 96 試合 | 対 router 48 試合 |
|---|---|---|
| **rm3**（MELON・重み 3） | **勝率 0.91・+4,412 ± 887 BETTER・362000〜 +3,475 ± 1,351 HELD** | +2,609 tie |
| rm1（重み 1）／rm1S（＋イチゴ全部売り）／rm03 | +3,361／+3,542／+3,253 BETTER | rm1 +3,826・rm1S +4,090 BETTER、rm03 tie |

- 固定：`agents/v62_sched.py`＝v61＋rm3（変種と router 90000／82000 で 1 円一致）。

### ✅ 羊毛も即 shed へ（2026-09-25・cloud・v62 の上・双子の相手 v62）＝**v63_sched を提出**

| 変種 | 1 回目（`36093820274`・seed0 370000） | 2 回目（`36097208057`・seed0 380000） | router |
|---|---|---|---|
| **rW**（`rush_items ["MELON","WOOL"]`） | 0.68・+1,619 ± 969 BETTER・引き直し +954 ± 1,605 NOT CONFIRMED | **0.62・+1,549 ± 1,058 BETTER・引き直し +1,564 ± 1,559 HELD** | −821／−1,284 tie |
| rS（イチゴ）／rWS | −3,677／−2,359 **WORSE** | — | rS WORSE |
| rM（牛乳） | −453 tie | — | tie |
| swS（イチゴ全部売り） | +46 ± 42 BETTER（小さすぎ） | — | +874 BETTER |
| mel14／mel16 | — | +14 tie／−2,637 WORSE | tie |
| h10（10 日目から牛 8・羊 5）／h10s（牛 6・羊 6） | — | −445／−650 tie | +1,770 tie／+3,950 BETTER |

- rW は 2 帯とも本番 BETTER・引き直しは +954／+1,564（合わせて +1,300 前後）。規則（BETTER＋HELD・router tie）は 2 回目で満たす＝提出。
- 固定：`agents/v63_sched.py`＝v62＋rW（変種と router 90000／82000 で 1 円一致）。

### ✅ 人手を増やす（2026-09-25・cloud・v63 の上・双子の相手 v63）＝**v64_sched を提出**

- traj（seed 82000）：v63 の手は 21 日目 225 手中 148 が移動（66%）、router は 273 手中 129（47%）・人手も多い（約 11.4 対 9.4 人）。
- 第 1 回（`36100785947`）：h10s（10 日目から牛 6・羊 6）+704 tie（引き直し +596 NOT CONFIRMED・router +2,700 BETTER＝2 回連続 router BETTER）、
  h10 +426・rw6 +387・rw1 +64 tie、swS +65 BETTER（3 回連続で小さく正・router +577）。`planner "route"` は試走で壊滅（10,121〜24,789）。
- 第 2 回（`36104142901`）：

| 変種 | 対 v63 双子 96 試合 | 対 router 48 試合 |
|---|---|---|
| **mh14s**（`max_hands 14`＋`sched_hands_scale 1.1`） | **勝率 0.73・+2,608 ± 1,075 BETTER・402000〜 +1,999 ± 1,466 HELD**（自分の金は −1,745） | +2,795 tie |
| mh14／mh12s（1.2 倍） | −1,212 WORSE（2 つ同値＝上限 12 が効いていない） | tie |
| h10sS（h10s＋イチゴ全部売り） | +431 tie | **+2,851 BETTER** |

- 固定：`agents/v64_sched.py`＝v63＋mh14s（変種と router 90000／82000 で 1 円一致）。

## ▶▶ 次の一手

0. **v64_sched を提出**（2026-09-25 の 5 本目）。最新 2 本＝v64 と v63。次の候補は v64 に双子で勝つもの（UTC 09-26 から提出可）。
   案：h10s／h10sS（router で 3 回 BETTER・双子 tie）を v64 上で、人手 1.05／1.15、`rush_items` の重み。
   案：`rush_items` を他の値崩れ品（WOOL・STRAWBERRY）へ、10 日目朝にメロンへ人手を寄せる、rm1S との比較。
   案：10 日目 0 時にメロンを収穫し 10 時前に売り切る（人手を 10 日目朝にメロンへ寄せる）、他の一発作物（小麦）の `sell_whole`。
   v58 の上で：メロンの区画数（12 → 14〜16）、施肥の日（`fert_sell_days` 5〜7）、10 日目の現金の使い道（家畜・3 区画目は以前は害だったが、現金の量が違う）。
   （旧メモ）
   次の候補は v57 に双子で勝つもの。残る大差（router 比）：羊毛 250 対 96 個・イチゴ 253 対 87 個・肥料 334 対 56 個・牛乳 219 対 46 個。
   v57 で牛が生き残ったときの牛乳と世話（CARE）の回数を `diag/ledger.py` で確かめ、世話を取りこぼしていればそこを直す。
1. v53 の上で：7 日目以降の forward veto（牛の payback 21）が効きすぎていないか、`sched_veto` を開幕期だけ外す形。
   肥料販売の日数（9 → 5／14）。どちらも v53 を base に双子（agent_b＝v53 自身か v51）＋router で測る。
2. ✅ **v51_sched は提出済み**（2026-09-24・PR #7 を cloud が merge → submit run `35965521829` success・一覧で PENDING を確認）。以下は判断の記録：双子 v48 に BETTER＋HELD、router に tie＝「v48_sched 以上だと直接対決で
   示せた」条件は満たす。出すと v45 が押し出され、最新 2 本＝v51_sched と v48_sched になる。
   **GHA で別 seed 帯（180000〜・cloud 未使用）でも再現**：双子 96 試合 勝率 0.85・margin **+6,040 ± 1,657 BETTER**（run `35961918542`）、
   router 48 試合 −5,543 ± 7,224 tie（run `35961921180`）。
   結果：`requests/results/submit-35965521829.txt`（md5 `30026330a32228bb7894bf5d22304bcb`・提出文末尾 `[run 35965521829 md5 30026330a322]`）。
   次のセッションで publicScore が付いたかを `www.kaggle.com/api/v1/competitions/submissions/list/kaggriculture` で読む（LB で版を判定はしない）。
3. ✅（→ 上の「糞を売って牛を足す開幕」で実施済み）0 日目の羊 2 頭目は閉じた・糞の販売＋牛の段階増は v53 に。

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
  **本番（`dry_run: false`）も 2026-09-24 に v51_sched で成功**（run `35965521829`・提出一覧に載ったのを workflow 自身が確認）。
- 提出 workflow の止め：main のみ／symlink 不可／単独で 720 ステップ完走／提出文に `[run <id> md5 <12桁>]` を付け、
  同じ md5 が一覧にあれば止める／再実行では提出しない／提出後に一覧を読み直して無ければ失敗（CLI は 404 でも終了コード 0）。
- 🔴 **提出の決まり（2026-09-24 user 指示で改定）**：**現提出（最新の版）に直接対決で勝てた候補は、user に聞かずに出す**。
  条件＝双子 96 試合で BETTER、かつ別 seed 帯の引き直しで HELD（router 相手は tie 以上）。1 日 5 本の範囲内・1 回に 1 本。
  **cloud が PR を作って自分で merge まで行い**、`requests/results/submit-<run id>.txt` を `git pull` で読んで結果を報告する。
  条件を満たさない版（LB を見たいだけ・単発の BETTER・中位の刻み）は出さない。

## 閉じた線（再提案しない・詳細な数字は過去の記録にあり）

capacity 一式（3 区画目・人手 1.25 倍・遊休地の埋め）／群れ拡大（通算 9 回）／肥料の重み（糞が世話を追い出す）／
`care_repeat`（同日 2 回目の世話は no-op）／開幕現金 `cash_buffer`／day-0 の家畜の顔ぶれ／市場の枠順／売りの繕い層／
`dist_weight` 0.7（0.9 が局所最適）／`lump_span`／トマト・にんじん 0／`crop_order`。
12 日目の 3 区画目・12 日目からの牛 8・羊 6 の床・veto 無し（q3／hv 系・両相手で WORSE）／0 日目の羊 2 頭目（`pending_by_species`・飼料代が消えて家畜が逃げる）／開幕メロン単品（`open_melon` だけ）／一発作物の早取り（`harvest_at_cap`・メロン限定でも害）／10〜11 日目の 3 区画目＋家畜の積み増し（r1l・r2）。
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
