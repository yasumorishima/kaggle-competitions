# ptcg — The Pokémon Company - PTCG AI Battle Challenge Playground

- コンペ：`the-pokemon-company-ptcg-ai-battle-challenge-playground`（Kaggle Playground・シミュレーション型、評価 `cabt_bo3`）。
  - ポケモンカード（PTCG）の対戦エージェントを出し、ラダーで BO3 の対戦をして、レーティング（μ0=600 から始まる）で順位が決まる。
  - 締切は 2027-01-08 23:59 UTC。提出は 1 日 5 本まで。最新 2 本だけが対戦を続け、最終評価にも使われる。
- Playground なのでメダルは無い（賞品は Kaggle グッズ）。
- 優先度：**低（user 指示：興味で 1 本出すだけ）**。大きな作業にしない。
- LB（2026-11-23 取得・808 チーム）：1 位 1542、10 位 1307、上位 25% 942、中央値 694。

## 提出の形式（出典：コンペのページ「How to Submit」・FAQ、kaggle-environments 1.33.3 の `envs/cabt`）

- **ファイル提出**（`isKernelsSubmissionsOnly: false`）。`.tar.gz` の最上位（入れ子にしない）に `main.py` と `deck.csv` を置く。
  - 原文：「Submissions need to be a .tar.gz bundle with main.py at the top level directory (not nested) and include a deck.csv.」
- 実行時、ファイルは `/kaggle_simulations/agent/` に置かれる。
- runner の動き：`main.py` を空の globals で `exec` し、**最後に定義された callable** を agent として使う。
  - そのため `agent` は必ずファイルの最後に置く。`__file__` は使えない前提で書く。
  - 引数は co_argcount に合わせて `(obs, config)` で渡される。
- 対局の最初に `obs["select"] is None` で呼ばれる。このとき 60 枚のデッキ（カード ID のリスト）を返す。
  - BO3 でもデッキを聞かれるのは 1 回だけ。2 戦目と 3 戦目は同じデッキで、先攻・後攻が入れ替わる。
- それ以降の手番は、option の添字のリストを返す（`minCount`〜`maxCount` 個・重複なし）。
  - 不正な答えを返すと、その BO3 は負けになる。
- 時間：手番ごとの制限は無い（`actTimeout 0`）。持ち時間は 1 戦ごとに 600 秒（`remainingOverageTime`）、`runTimeout` は 6000 秒。
- 提出すると、まず自分のコピーとの検証対局が 1 回ある。失敗すると Error になる。

## 方針（v1）

- **ルールで動くエージェント**（標準ライブラリだけ・エンジンの `cg` は import しない）。tar には `main.py` と `deck.csv` だけを入れる。
- デッキは水のメガユキノオー ex（Mega Abomasnow ex）。中身はユキカブリ 4・メガユキノオー ex 4・カイオーガ 2・メガシグナル 4・ハイパーボール 3・マキシマムベルト 1・リーリエの決心 4・ウェイトレス 4・なみのりビーチ 2・シラノ 1・基本水エネ 31。
  - 主催の sample デッキを土台にして組み直した。
  - 局所の対戦で比べた：sample デッキ比 158-142、Brock 入り版比 151-149（どちらも雑音の範囲）。
- **土台（出典）**：
  - 主催の sample_submission（`main.py` の枠・デッキ）。
  - 公開ノート matterhorn3838/pokemon-challenge-ultra-900 の中の対照用ルールエージェント「previous_rule」。OptionType ごとに点を付けて選ぶ考え方を借りた。コードは書き直している。
- **自分で加えた改良**：
  - 準備：エネルギーは「攻撃できるまでに足りない数」で付け先を決める（バトル場を優先）。
  - 攻撃：攻撃の前に、進化・ベンチ展開・サーチ・サポートを済ませる。
  - 攻撃：攻撃は KO できるかと、山札の残りの水の割合から期待値（Hammer-lanche）を出して選ぶ。
  - 入れ替え：なみのりビーチと逃げは、ベンチに攻撃できるポケモンがいるときだけ使う。
  - 山切れ対策：山札が 7 枚以下では Hammer-lanche を避ける。
  - 選択の種類：捨てる・戻す選択は価値の低いものから選び、相手を狙う選択は HP の低いものを選ぶ。
  - 安全策：先攻を選ぶ。答えが不正なときは、合法な乱択に落とす。

## 手元の検証（2026-11-23・エンジン libcg.so で 1 戦ずつ・座席は交互）

| 相手 | 戦数 | 勝-負 | 勝率 |
|---|---|---|---|
| 乱択（環境の既定デッキ） | 400 | 381-19 | 95.3% |
| 「常に先頭の option」（環境の既定デッキ） | 400 | 268-132 | 67.0% |
| 自分どうし | 200 | 102-98 | 51%（健全性の確認） |

- 不正な答え：0 件。例外による乱択への退避：0 件。
- 1 手あたり約 0.02 ms。
- 負けの多くは「場にポケモンがいなくなる」（理由 3）。次に直すならここ。基本ポケモンを探すカードと、ベンチを保つ方針を入れる。

## 提出ファイルの作り方

```bash
cd ptcg && tar czf submission.tar.gz main.py deck.csv
kaggle competitions submit -c the-pokemon-company-ptcg-ai-battle-challenge-playground -f submission.tar.gz -m "v1 rule agent (Mega Abomasnow ex)"
```

- `submission.tar.gz` は repo に入れない。
- エンジン（`cg/`・`ptcgProgram`）と配布物は repo に入れない。手元で試すときだけ scratchpad に置く。
