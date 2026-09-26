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

- 参加済み。データ取得・スキーマ確認・公開ノートブックの読解まで。コードはまだ無い。

## ▶▶ 次の一手

1. 提出インフラ：`enveda-casmi26/requests/push.json` で GHA が kernel を push（と dataset 作成）する workflow。まず「sample_submission をそのまま出す」ノートブックで経路を確認。
2. 検証台（`enveda-casmi26/eval/`）：C1/C2/C3 の分割と MRR@25。
3. 最初の検索ベースライン：train ライブラリの entropy 類似（C1）＋ 質量窓の train 構造候補。
