# Enveda CASMI 2026 — 作業の引き継ぎ（Claude Code 向け）

Kaggle `enveda-CASMI26-molecule-id-mass-spectra`（Featured・$50k・**締切 2026-12-14 23:59 UTC**・参加 2026-09-26）。
このファイルが作業の正本。**セッションの終わりに「現在地」と「次の一手」を更新して commit する**。
報告は日本語・です/ます。cloud セッションだけで完結させる（kaggriculture と同じ方針）。

## 課題（一次資料＝コンペのページ・2026-09-26 に読んだ）

- MS/MS スペクトルから分子の 2D 構造を当てる。**1 分子につき SMILES を最大 25 個、確からしい順**に `;` 区切り。
- 評価：**MRR@25**。正誤は RDKit の互変異性正規化（2026.03.3）→ **InChIKey 前半 14 文字**の一致（立体・互変異性は問わない）。
- 予測は**分子単位**（1 分子に 1〜16 スペクトル・中央値 3）。隠しテストは約 400 分子・約 1,500 スペクトル・**全部 Bruker timsTOF**。
  質量 157〜1,159 Da。付加イオンは 10 種（`[M+H]+` `[M+NH4]+` `[M-H2O+H]+` `[M-2H2O+H]+` `[M+Na]+` `[M+K]+` `[M-H]-` `[M-H2O-H]-` `[M+CH2O2-H]-` `[M+Cl]-`）。
- テスト分子の 3 分類（割合と所属は非公開）：
  1. 公開スペクトルライブラリにある（train との類似で当たる）
  2. 構造は PubChem か COCONUT にあるが公開スペクトルは無い（DB 検索）
  3. **PubChem に無い新規構造**（de novo で作るしかない）
- **Notebook 提出のみ**：CPU/GPU とも 9 時間以内・**インターネット不可**・公開の外部データと学習済みモデルは可・出力は `submission.csv`。1 日 5 本・最終 2 本選択。
- 見える `test.parquet`（400 分子・1,213 行）は **train の `enveda-180` から抜いた例**（precursor_mz＋adduct＋base_peak で全行 train に一致）。本番は再実行時に隠しテストに差し替わる。

## データ（train.parquet 3.0GB・2,539,608 行・27.6 万構造）

- 列：`ingest_lib` `normalized_smiles` `inchikey` `inchikey14` `molecular_formula` `ionization_mode` `instrument_type` `adduct` `adduct_orig`
  `precursor_mz` `precursor_error_ppm` `ms2_mzs` `ms2_normalized_intensities` `num_peaks` `base_peak_intensity` `collision_energy_ev` `collision_energy_orig` `collision_energy_orig_units`。
- ライブラリ：`enveda-180` 115 万行（timsTOF・18.3 万構造・うち他装置にもある構造は 2,538 だけ）、`pluskal_ms2` 53 万、`riken` 35 万、`gnps` 22 万、`massbank` 10 万、`mona` 9 万 ほか。
- `enveda-np-examples` 1,184 行・250 分子（全部 train の他ライブラリにも存在）。
- 付加イオンは 121 種（テストの 10 種以外も多い）。陽イオン 190 万・陰イオン 64 万。
- cloud コンテナでの取得：`curl -sSL -o train.parquet "https://www.kaggle.com/api/v1/competitions/data/download/<comp>/train.parquet"`（約 1 分）。
  置き場は scratchpad（git に入れない）。`pip install pyarrow rdkit` が要る。メモリ 15GB・ディスク空き約 28GB。

## 公開ノートブックの現状（2026-09-26 に読んだ・流用はしない＝読解だけ）

- 上位の公開は実質 1 系統（prvsiyan の analog-propagation と、その派生）で LB 0.33〜0.35。構成：
  - 候補＝COCONUT 2.0（46 万）＋ train 構造（28 万）を InChIKey14 で重複除去（71 万）→ 推定中性質量 ±10 ppm（中央値 56 候補）。
  - 4 チャネル：① train スペクトルとの entropy 類似（library）② 質量シフト付き類似の類縁体 × Tanimoto（analog）
    ③ 1〜2 結合切断のフラグメント説明率（MetFrag 風）④ スペクトル→フィンガープリント予測の transformer（FPNet）で `f·z`。
  - 31 特徴の HistGradientBoosting で並べ替え。
- 作者の推定：クラス 1 ≈ 16%（ほぼ満点）、クラス 2 ≈ 27%、**クラス 3 ≈ 55%（誰も手を付けていない）**＝検索だけの上限 ≈ 0.43。LB 1 位 0.425 はこの上限付近。
- 主催の de novo チュートリアル（encoder–decoder transformer・SMILES BPE）があるが、検索系と組み合わせた公開は無い。
- LB（2026-09-26）：1 位 0.425・10 位 0.383・20 位 0.370。

⇒ **金を狙う差別化はクラス 3（de novo）**。検索系は自前で 0.33 級を作り、その上に「分子式で縛った de novo 生成」を載せる。

## 方針

1. 自前の検証台：`enveda-180`（timsTOF）から構造単位で分子を抜き、C1（他ライブラリに同じ構造が残る）・C2（構造は候補 DB にだけ残る）・
   C3（どこにも無い）の 3 条件で MRR@25 を出す。公式の metric notebook と同じ InChIKey14 判定を使う。
2. 検索系（C1＋C2）：質量窓の候補プール＋ライブラリ類似＋類縁体＋フィンガープリント予測を自前で作る。
3. de novo（C3）：分子式（精密質量から）で縛った生成 → フィンガープリント予測で再順位付け → 検索系の候補の後ろの枠を埋める。
4. 提出インフラ：cloud は Kaggle CLI 不可（api.kaggle.com）＝ kaggriculture と同じく**依頼ファイルを push → GHA が kernel/dataset を push・submit**。

## 現在地（2026-09-26）

- **提出経路は開通**：`enveda-casmi26/requests/kaggle.json` を push → `.github/workflows/enveda-kaggle.yml` が kernel を push・完了待ち・
  （`"submit": "submit"` のとき）`kaggle competitions submit -k <kernel> -v <ver>` → 結果を同じブランチの `requests/results/kaggle-<run>.txt` へ。
  run `36223094376`：`kernels/sample`（sample_submission の写し）を push → 提出一覧で PENDING を確認（点数 0.000・経路確認用）。
  ⚠️ 依頼ファイルが main への merge とブランチの作り直しで再生され、**経路確認の提出が 09-26 に 3 回**走った（全部 0.000）。
  対策済み：workflow は main では動かない（`branches-ignore`）＋ **依頼には毎回新しい `id` が必須**（結果ファイルに同じ id があれば何もしない）。
  `"action": "dataset"` で `dataset-metadata.json` のあるフォルダを Kaggle dataset として作成／版上げ（未試験）。
- **cloud コンテナから外部 DB（COCONUT・PubChem・ChEBI・zenodo）は proxy が 403**。`www.kaggle.com` と pypi は届く。
  ⇒ 外部 DB の取得と加工は GHA（インターネット可）でやり、Kaggle dataset にしてから使う。
- **検証台**（`eval/`・データは `$CASMI_DATA`＝既定 `~/casmi_data` に train.parquet 等を置く）：
  - `common.py`：付加イオン→中性質量・分子式→精密質量・metric（RDKit 互変異性正規化→InChIKey14）・MRR@25。
  - `split.py`：`enveda-180`（timsTOF・テストの 10 付加イオン）から各 400 分子（≤16 スペクトル）。
    c1＝公開ライブラリにも同じ構造がある（その公開スペクトルはライブラリに残す）／c2＝enveda だけ（スペクトルは全部抜き、構造は候補に残す）／c3＝構造も候補から抜く。
  - timsTOF の精密質量誤差は 99% が 4.3 ppm 以内（候補窓 ±10 ppm で足りる）。
- **B0 `baseline_lib.py`**（train 構造の ±10 ppm 候補を train スペクトルとの entropy 類似の最大で並べる）・各 100 分子：
  c1 **0.922**（窓内 100%）／c2 0.012（窓内 100%・候補中央値 82）／c3 0（窓内 0%）。
  公開の推定比率（16/27/55%）で重み付け ≈ 0.15 ＝ 公開の「ライブラリだけ LB 0.151」と一致 ⇒ 検証台は LB と整合。
- **B1 `analog.py`**（各 60 分子・約 10 分）：類縁体＝構造×極性ごとの代表スペクトル 37 万本に FlashEntropy の hybrid 検索（ずれた一致も数える）、
  上位 100 本の類縁体について sim³ × Tanimoto(候補, 類縁体) の最大。**c2：類縁体だけで 0.815**（ライブラリだけ 0.007）、c1 0.942。
  ⚠️ 公開（NP 寄りの C2）は 0.52〜0.55。当方の c2 は `enveda-180` の無作為抽出＝合成系列の近い類縁体が多く**楽観的**。天然物寄りの c2 を作り直すこと。
  単純合成 `max(lib, 0.9·analog)` は c2 0.106 に崩れる（スペクトルを持つ異性体の lib 値が勝つ）＝「lib が高い時だけ lib、他は analog」か学習で合成する。
  メモリ：train のピーク列を丸ごと読むと落ちる（15GB）→ `common.load_peaks` で行グループごとに読む。

- **初の実提出 b1＝LB 0.275**（2026-09-26・request `b1-1`・run `36225977495`・kernel `yasunorim/casmi26-b1-library-analog` v1・Kaggle 上 約 3.5 分）。
  中身：train＋COCONUT（公開 dataset `prvsiyan/coconut-casmi26-candidates` の `coco_meta.pkl`/`coco_mass.npy`・計 71 万構造）の ±10 ppm（中央値 107 候補）を
  類縁体（上の B1）＋ライブラリ一致（0.8 以上だけ加点）で並べる。wheel は自前 dataset `yasunorim/casmi26-offline-wheels`（rdkit 2026.3.3・ms_entropy 1.5.2・`datasets/offline-pkgs/prepare.sh` で GHA が取得）。
  合成規則は `scores_analog_150.pkl` で決めた（各 150 分子）：類縁体だけ c1 0.924・c2 0.799、lib を混ぜると c2 が落ち（加算 w=1 で 0.101）、lib≥0.8 のゲートだけ同等。
  公開の推定比率（c1 16%・c2 27%）で逆算すると**本番 c2 ≈ 0.47**（当方の検証 c2 0.8 は楽観的＝近い類縁体が多すぎる）。
- 注意：wheel 用フォルダ名 `wheels/` はリポの `.gitignore` に掛かる（`offline-pkgs/` にした）。`git mv -k` は黙って何もしないことがある。

- **FP 予測 MLP**（`eval/fpnet_data.py`→`fpnet_train.py`・2026-09-26）：特徴＝1 Da ビンの断片＋中性損失（sqrt 強度・最大値）＋付加イオン one-hot（1,511 次元）、
  目標＝Morgan r2 2048 ビット、学習 98 万スペクトル（構造あたり ≤4・検証の 1,200 分子は除外）、1024×2 層・3 エポック（CPU 約 6 分・`$CASMI_DATA/fpnet.pt`）。
  各 150 分子：**FP 単独 c2 0.268**（公開の FPNet 0.47〜0.52 より弱い）、類縁体に足すと c2 0.768→0.710 以下に悪化（この楽観的な c2 では）。
- **検証の較正（済）**：`calibrate.py`（正解と Tanimoto ≥ T の類縁体を除く）では T=0.5 でも c2 0.664 ＝**近い類縁体は楽観の原因ではない**。
  原因は**問題の分子の種類**：`analog.py 150 coco np` の **class 4＝`enveda-np-examples` 250 分子（天然物）**を c2 扱い（全ライブラリから当該構造のスペクトルを抜き、構造は候補に残す）、
  候補＝train＋COCONUT（kernel と同じ・c4 の候補中央値 58）→ **b1 の式で c4 0.526 ≈ 本番推定 0.47〜0.55**。**以後の判定は class 4**（`scores_analog_150_coco_np.pkl`・約 43 分）。
  （COCONUT を入れても enveda の c2 は 0.79 のまま＝本番が難しいのは天然物だから。）
- `blend.py <dump>`：ゲート G × FP 重みの格子を class ごとに出す。c4：G 0.8→0.526、0.95→0.570、ライブラリ無し→0.576（c1 はどれも 0.924）。
  **FP（class 4 を学習から除いて再学習）は c4 で足すほど悪化**（w 0.05→0.566、0.3→0.506・単独 0.239）＝今の MLP は弱すぎる。
- **b2**＝b1 のライブラリゲートを 0.95 に（kernel `yasunorim/casmi26-b1-library-analog` v2・request `b2-1`・run `36243920088`）。**LB 0.271（b1 0.275 より悪い）**＝c4 の +0.044 は LB に出ず、0.8〜0.95 のライブラリ一致は本番では当たりを含む ⇒ **ゲートは 0.8 に戻す**。
  教訓：class 4 は c2 の分布には合うが、ライブラリ一致（c1 側）の判定には使えない。ゲートは LB で決めた値を動かさない。
- データの置き場（コンテナは消える）：`~/casmi_data`→ scratchpad の `enveda/`（train.parquet・train_meta・structures・split・coconut・fpnet_*）。
  消えていたら取り直し：train/test は curl（上）、`coco_meta.pkl`/`coco_mass.npy` は `www.kaggle.com/api/v1/datasets/download/prvsiyan/coconut-casmi26-candidates/<file>`、
  あとは `split.py`→`fpnet_data.py` の順で作り直す（structures/train_meta は `baseline_lib.py` 前の一行スクリプトと同じ内容＝common で再生成）。
- **analog の集約を改良（`analog_tune.py`・保存済みヒットから再採点・ゲート 0.8）**：b1 相当（Morgan r2・POW 3・max）c1 0.921／c2 0.764／c4 0.528 →
  **Morgan r3 カウント・POW 2・上位 400 類縁体・候補ごとに上位 3 の和**で c1 0.939／c2 0.781／c4 0.570（全クラスで上）。全表は `$CASMI_DATA/analog_tune.csv`。
  kernel に実装済み（b3・ローカル煙テスト 218 秒・400 行）。**09-27 00:01 UTC に request `b3-1` で提出予約**（send_later）。
- 提出枠：09-26 は 5 本使用（経路確認 3・b1 0.275・b2 0.271）。
- 得点の確認：`curl -sS https://www.kaggle.com/api/v1/competitions/submissions/list/enveda-CASMI26-molecule-id-mass-spectra`（cloud から届く）。

## ▶▶ 次の一手

1. **c4（天然物 c2）を上げる**：判定は `blend.py scores_analog_150_coco_np.pkl`。今の FP MLP は c4 で逆効果。
   ① ✗ フラグメント説明（`fragexp.py`・1〜2 結合切断）は単独 c4 0.120・足しても伸びない（単体では弱い＝使うなら学習合成の特徴として） ② FP を公開並み（単独 0.47 級）に強化（0.1 Da ビン・大きいモデル・学習は Kaggle GPU を GHA 経由）
   ③ ✅ 類縁体の集約（上位 3 の和）→ b3。続き：類縁体検索そのもの（代表スペクトルを構造×極性×付加イオンに増やす・MAX_PEAKS）を c4 で ④ チャネルを学習で合成（公開は HGB）。
2. **候補 DB**：GHA で COCONUT（と PubChem の天然物寄り部分集合）を取得→分子式・精密質量・InChIKey14 の表→Kaggle dataset。c2 の窓内率と候補数を再測定。
3. **c3（de novo）**：分子式で縛った生成。GPU は Kaggle Notebook（週 30 時間）を GHA 経由で使う。
4. ✅ 実提出 b1＝0.275。次は c2 を上げる手（フィンガープリント予測・フラグメント説明）と、本番に近い c2 の検証（天然物寄り）。
