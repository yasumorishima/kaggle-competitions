# CASMI26 Natural-Product Panel Benchmark — Technical notes

These notes retain the complete methods, file and column definitions, measurements, caveats, updates and source credits. The companion Kaggle card provides a short introduction.

## Terms used in these notes

**Candidate** [a possible molecular answer].

**Pool** [a collection of candidate structures].

**Tier** [the mass-sorted PubChem structure collection used here].

**Prior** [extra ranking evidence, such as how often a compound is documented].

**Fingerprint** [a list of bits recording molecular features].

**Fold** [one group reserved for checking a model while the other groups supply training data].

**InChIKey connectivity block** [the first 14 InChIKey characters, identifying connectivity without stereochemistry]. The scorer key is calculated after RDKit chooses a standard tautomer [a form differing in proton positions and bond arrangement].

**MRR@25** [the mean of 1 divided by the correct answer’s rank among the first 25 unique scored identities; a missing answer or a rank beyond 25 contributes zero].

## Dataset overview

A development benchmark for the [Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra) structure-ranking task: **250 natural products with known answers, measured on timsTOF (1,184 spectra)**, together with every candidate the public v4 pipeline considers for them and the scores it assigns. Since the version of 2026-10-13 the dataset also carries the author's identity-disjoint validation folds for the whole training file (folder `casmi26-identity-disjoint-validation-folds/`), so that the notebook [CASMI 26: From Spectra to Structures](https://www.kaggle.com/code/dmitriigluzdov/casmi-26-from-spectra-to-structures) mounts one dataset for its development tables (Section 11) and its split analysis.

The hidden test contains natural products from biological extracts. The `enveda-np-examples` library in the training file is the closest labelled material, and the training code of the public FPNet A/B models excludes this library by default. The benchmark turns it into three simulated situations and stores the complete candidate tables, so a change to retrieval depth, a prior, a merge rule or a new ranker can be replayed in seconds, without a GPU.

A replay of the PubChem channel with a custom score:

```python
import os
import pandas as pd, numpy as np
D = next(p for p in ('/kaggle/input/datasets/dmitriigluzdov/casmi26-natural-product-panel-benchmark', '/kaggle/input/casmi26-natural-product-panel-benchmark')
         if os.path.isdir(p))   # newer and older mount paths
cand = pd.read_parquet(f'{D}/pubchem_candidates.parquet')      # 1.94 M rows: every PubChem candidate of every molecule
cand = cand[np.isfinite(cand.fpnet_score)]                     # 99 rows could not be parsed

def mrr25(score):                                              # a scoring rule -> MRR@25 of the PubChem channel
    out = []
    for mol, g in cand.assign(s=score).groupby('molecule', observed=True):
        g = g[~g.in_pool | g.is_truth].sort_values('s', ascending=False)
        ident = np.where(g.scorer_key != '', g.scorer_key, g.connectivity)
        first = ~pd.Series(ident).duplicated().values           # one guess per scored identity
        rank = np.flatnonzero(g.is_truth.values[first])
        out.append(1 / (rank[0] + 1) if len(rank) and rank[0] < 25 else 0)
    return np.mean(out + [0] * (250 - len(out)))

z = cand.groupby('molecule', observed=True).fpnet_score.transform(lambda x: (x - x.mean()) / x.std())
print(mrr25(z), mrr25(z + 0.25 * (cand.log_substances + cand.log_pubmed)))   # 0.51 and 0.97 with this simplified protocol
```

The tables below follow the notebook exactly (5,000-row shortlist, its pool-exclusion rule), which gives 0.51 and 0.95.

## 2026-10-19: self-contained public prior replay

`pubchem_signals.parquet` caches 1,939,476 candidate rows in the exact original `pubchem_candidates.parquet` order, identified by `candidate_row` plus molecule/tier_row. Columns add LOTUS references, PubChemLite membership, NP/source flags, documentation and the v10 composite prior. Exact-formula flags and their frozen formula evidence are included (818,810 passing rows). `pubchemlite_annotations.parquet` provides the source-aligned 385,760-row add-on documentation table.

Run `python replay_public_priors.py --panel DATASET_DIRECTORY --out RESULTS_DIRECTORY` to reproduce `prior_rule_summary.csv`, `prior_rule_per_query.parquet` and `prior_replay.json` without a GPU or another mounted dataset. Five fixed rules use the same 250-molecule denominator, score absent answers zero, maximise within scored identity and average exact ties analytically. The annotation caches now use LOTUS frozen 2026-04-13 (Zenodo 19360665, CC BY 4.0), preserving original source lookup rules and float32 composite-prior arithmetic. v10 composite prior: MRR@25 0.831933333333 to 0.830933333333; LOTUS references only: MRR@25 0.625238327988 to 0.624332830283. At fold0 weight 1/3, the v10-prior replay changes MRR@25 0.964666666667 to 0.964800000000; top-1 remains 0.960. The paired 95% refresh interval is [0.000000000000, 0.000400000000] (20,000 molecule bootstrap replicates, seed 20261019). Other predeclared fingerprint mixtures and all historical documentation-only ranking variants retain their recorded ranks. See LOTUS_REFRESH.json for all counts, intervals and limitations. The refreshed replay visits 8 additional exact identity rows: concatenate lotus_replay_identity_additions.parquet with the unchanged panel_metric_identities.parquet on candidate_row. The supplement preserves the original identity-cache schema and is disjoint from its original rows. This experiment ranks documentation only among exact-formula candidates; it does not apply the spectral shortlist or pool-exclusion channel. POP_LAM 0.15 is documented as the notebook's spectral-blend weight; a positive scalar alone does not change these annotation-only ranks.

The notebook reads the cached summary in its panel development section. Fold-safe fingerprint panel outputs, when included, are a separate protocol with explicit lineage; earlier 499 polarity-group pool-window results are not the same as a 250-molecule PubChem replay. Competition identifiers, answer formulas and model scores follow the original competition data terms; the added replay helper is Apache-2.0. No spectra are redistributed. PubChemLite and its fingerprint distribution are credited to Schymanski et al. (2021), cosmicbe and franciscoangulo (CC0); other annotation source terms remain as documented.

Source credits: PubChem / U.S. National Library of Medicine (https://doi.org/10.1093/nar/gkae1059), LOTUS / Rutz et al. (https://doi.org/10.7554/eLife.70780), Natural Products Atlas and contributors (https://doi.org/10.1093/nar/gkab941). The LOTUS frozen export is the 2026-04-13 release (https://doi.org/10.5281/zenodo.19360665), CC BY 4.0; credit Rutz et al. and the LOTUS contributors. The current NPAtlas download page specifies CC BY-NC 4.0; retain snapshot-specific terms and attribution. Existing PubChem/source-specific and competition-derived restrictions are retained. See `SOURCE_CHECK_20261019.md`.

Public panel predictions: `panel_logits.npz` contains 250 molecule-level vectors (10,226 bits) for A+B and fold-safe FPNet. `panel_fpnet_scores.parquet` caches all 1,939,476 candidate rows (full and ECFP4-partial float32 scores); `panel_fpnet_per_molecule.csv` and `PANEL_REPLAY_PUBLIC.json` record the fixed mixtures and bootstrap intervals. `summarize_panel_fpnet.py --per-query panel_fpnet_per_molecule.csv --panel panel.parquet --out summary.json` reproduces the aggregate scores and paired 95% intervals from these files alone. The predictions reuse the completed parallel public notebook replay; no duplicate model/fingerprint inference was run for distribution. Exact source/model/input hashes and any discrepancy from the previously cached A+B scores are explicit in the report.

No new spectra or hidden-test labels are added. Public panel labels are used only for evaluation. A local documented-natural-product panel does not establish a hidden-test improvement. Earlier MANIFEST files describe earlier snapshots. The LOTUS 2026-04-13 overlay is pinned by `CHANGED_FILES.json` and validated in `VALIDATION_20261020.json`; retain every unlisted baseline file. `VALIDATION_20261019.json` remains a historical snapshot record.

## Files

| File | Rows | Content |
|---|---:|---|
| `panel.parquet` | 250 | The molecules: scorer key, measured and true neutral mass, spectra, adducts, window sizes. |
| `pubchem_candidates.parquet` | 1,939,476 | Every row of each molecule's 10-ppm PubChem window with the partial and full FPNet A+B scores, connectivity and scorer keys, truth flag, pool membership, popularity, patent and natural-product annotations. |
| `main_candidates.parquet` | 89,418 | Ranked main-channel candidates (pool and generated structures) with ranker scores for the three scenarios. |
| `results_pubchem_channel.json` | | PubChem-channel variants: retrieval depth, de-duplication, popularity weights. |
| `results_main_channel.json` | | Main-list popularity weights per scenario. |
| `results_merge_rules.json` | | End-to-end merge rules per scenario. |
| `results_fingerprint_seen_unseen.json` | | Pool-window retrieval of three public fingerprint checkpoints on the panel and on other training structures. |
| `*.py` | | The scripts that produced every file. |
| `casmi26-identity-disjoint-validation-folds/` | 277,566 / 2,539,608 | The validation folds (version of 2026-10-13): `structures.parquet`, `spectra_folds.parquet`, `SUMMARY.json`, `build_folds.py` and the original data card. |

### The three scenarios

| Scenario in `main_candidates` | What is hidden from the pipeline | Simulates |
|---|---|---|
| `reference_in_other_libraries` | the molecule's NP-example spectra | an answer with reference spectra from other instruments |
| `structure_in_pool_no_reference` | every reference spectrum of the molecule | a known natural product without any reference spectrum |
| `structure_removed_from_pool` | every spectrum and the structure itself in the candidate pool | an answer found only in PubChem, or a new structure |

## What it shows

**PubChem channel** (answer removed from the pool; 246 of 250 answers are present in the PubChem window):

| Variant | MRR@25 | Top-1 | Answer in shortlist |
|---|---:|---:|---:|
| FPNet A+B only, 1,000-row shortlist | 0.510 | 40% | 90% |
| FPNet A+B only, 5,000-row shortlist | 0.512 | 40% | 98% |
| FPNet A+B only, whole window | 0.511 | 40% | 98% |
| + popularity prior 0.25 (as in the notebook) | **0.951** | **93%** | 98% |
| + popularity prior 1.00 | 0.952 | 93% | 98% |
| + popularity and patents | 0.942 | 91% | 98% |

Retrieval depth is not the bottleneck on this panel: the answer is among the best 1,000 rows of the cheap first pass for 91% of the molecules and among the best 5,000 for 99.6%. Documentation is decisive: the correct structure is better documented than 99.7% of its mass-window neighbours (median).

**Fingerprint model alone: molecules it has seen and molecules it has not.** The answer stays in the pool and is ranked among the pool structures within 10 ppm of its mass (median window: 54 structures on the panel, 93 on the training sample) by the fingerprint score `f·z` only. Queries merge the spectra of one structure and polarity.

| Checkpoint | Trained on the panel | Training structures: top-1 | MRR@25 | Panel: top-1 | MRR@25 | Panel: top-25 |
|---|---|---:|---:|---:|---:|---:|
| FPNet A (main engine of the v4 pipeline) | no | 89% | 0.934 | **42%** | **0.581** | 95% |
| FPNet A+B (PubChem channel) | no | 90% | 0.941 | **43%** | **0.589** | 96% |
| FPNet full1 (trained on all folds) | yes | 90% | 0.938 | **89%** | **0.928** | 100% |

The same architecture puts the answer first for about 89% of the molecules it was trained on and for about 42% of the panel molecules it never saw; once the panel is part of training (full1) the panel reaches 89% as well. Most of the fingerprint model's apparent accuracy on training molecules is recall of those molecules. The 42% result measures this panel without its own reference spectra; it does not establish hidden-test accuracy. The answer is still among the best 25 for 95% of these queries, so ordering within the retrieved set is a measured opportunity for improvement.

**Merge rules** (MRR@25 of the final 25-structure list):

| Rule | Reference in other libraries | In pool, no reference | Removed from pool |
|---|---:|---:|---:|
| Main list only | 0.980 | 0.988 | 0.361 |
| Fixed slots (PubChem from rank 2 or 4) | 0.980 | 0.988 | 0.519 |
| Promote the best PubChem proposal when `S > 6` | 0.980 | 0.986 | 0.681 |
| Promote when `S > 6` and `pop ≥ 5`; early slots when `S > 5` | 0.980 | 0.988 | **0.718** |

`S = z(FPNet) + 0.25 · pop` is the score of the best PubChem proposal. With `S > 6` alone the rule fired for 100 molecules of the pool-removal scenario, always with the correct structure, and for one of 246 molecules whose answer was in the pool: a proposal with a very high fingerprint score and almost no documentation. Requiring `pop ≥ 5` as well removed that case and kept all 100 correct promotions. The replay does not include the ICEBERG and GLACIER re-ordering, and it describes the earlier configuration of the notebook rather than the expanded two-engine pipeline that its version 43 and later run.

## Limits of the numbers

* The panel is a **development proxy**. The training code of FPNet A/B excludes this library by default and the table above agrees with that (42% on the panel against 89% on training structures), yet the exact training rosters of the public checkpoints are not published, and the main ranker and its feature resources may have seen these molecules. The full1 checkpoint was trained on the panel, so the panel cannot evaluate it. The two pool scenarios (0.98–0.99) are therefore optimistic. Thresholds were chosen on these same molecules, so the paired differences describe this panel and transfer to other chemistry is untested.
* The answers are **well-documented natural products** (249 of 250 are in COCONUT). A prior based on documentation looks better here than it will for rarely studied compounds.
* The removed-from-pool scenario lets the analogue generator rebuild a known structure from its close relatives, which overstates what generation achieves for truly new chemistry.
* Masses are measured, not theoretical: the median precursor error in the panel is +1.45 ppm.

## Identity-disjoint validation folds (version of 2026-10-13)

The folder `casmi26-identity-disjoint-validation-folds/` is a byte-identical copy of the author's dataset [CASMI26 Identity-Disjoint Validation Folds](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-identity-disjoint-validation-folds). The competition scores a guess by the first 14 InChIKey characters after RDKit tautomer canonicalisation. A random row split of the training file leaks: one molecule has on average nine spectra, and 1.6% of training structures carry a stored `inchikey14` that differs from the key the scorer computes. The folds keep every spectrum of one scored identity on one side. `structures.parquet` (277,566 rows, one per distinct training structure) carries the stored and scorer keys, the Bemis–Murcko scaffold, group identifiers and two 5-fold labels; `spectra_folds.parquet` (2,539,608 rows) gives the same assignment for every spectrum of the training snapshot with SHA-256 `9423f90e…bcd98`, and a join through `inchikey14` also covers existing keys after a training-file update; unmatched new keys need assignments. `fold_identity` links structures that share the scorer key or the stored key (274,184 identity groups) and estimates performance on molecules without a reference spectrum of their own; `fold_scaffold` additionally keeps every scaffold in one fold (140,160 scaffold groups; acyclic molecules form their own groups), a harder test of chemical generalisation. Both assignments are balanced by structure count (55,512–55,514 per fold) and stratified by whether a group has Enveda spectra. `build_folds.py` (RDKit 2026.03.3, seed 20260930) asserts scorer-key and stored-key separation for both assignments, and scaffold separation for scaffold folds; `SUMMARY.json` records the counts.

A clean estimate removes held-out identities from model training, from the reference-spectrum library and from any feature table fitted on labels, states whether the candidate pool keeps the held-out structure, reports candidate coverage before ranking metrics, and uses models fitted on the four training folds only, since pretrained public checkpoints have usually seen all five.

## Method

`np_tables.py` scores every row of each PubChem window with the public FPNet A+B models (the code of the v4 pipeline); `main_tables.py` runs the public main engine with its own simulation switches (`exclude`, `exclude_sid`, `drop_pid`); `np_eval.py`, `main_eval.py` and `merge_eval.py` replay the variants and compute molecule-level bootstrap intervals; `fpnet_window_eval.py` scores the pool windows with a fingerprint checkpoint alone. Inputs: the [attributed v4g assets](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets) and the [PubChem popularity prior](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior).

## Terms

Identifiers, masses and scores are derived from the competition training file and remain under the [competition data terms](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data); PubChem annotations follow the terms described in the popularity dataset. The folds folder holds identifiers and structures derived from the same training file under the same terms. No spectra and no hidden test data are included.

## Acknowledgements

The panel molecules and their spectra come from the `enveda-np-examples` library of the [CASMI 2026 competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) data provided by Enveda. The candidates and scores were produced with the public v4 pipeline of [Ahmed Berat Özer](https://www.kaggle.com/ahmedberatozer) — its candidate pool of training and COCONUT structures, the PubChem tier, the FPNet A+B and full1 fingerprint models and the LightGBM rankers, mirrored in [Attributed v4g Reproduction Assets](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets). The documentation counts are from [PubChem](https://pubchem.ncbi.nlm.nih.gov/) via [PubChem Popularity Prior](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior), and COCONUT ([Sorokina et al., 2021](https://doi.org/10.1186/s13321-020-00478-9)) supplies the natural-product part of the pool.

## Protocol clarifications

The first-pass retrieval percentages of about 91% and 99.6% use the 246 answers present in the PubChem window: 225/246 and 245/246. Across all 250 panel molecules the corresponding counts are 90.0% and 98.0%. The documentation arm in `results_pubchem_channel.json` combines the 5,000-row fingerprint shortlist with the 200 most documented candidates. This explains why it differs from the simplified full-window code example.

The figure compares two fixed variants in that JSON, with MRR@25 0.511699 and 0.951333. Their answer counts at rank 1, ranks 2–25, and beyond rank 25 or absent are respectively 100/100/50 and 233/11/6. The dots are aggregate categories, not paired paths for individual molecules.

For the included folds, joins after a training-file update cover existing keys only. New keys need assignments, and unmatched rows must be checked before fitting. The builder checks scorer-key and source-key separation for both assignments and scaffold separation only for `fold_scaffold`. Identity folds can share ring-system relatives.

The 499-query pool-window evaluation merges by structure and polarity; it is a different protocol from the 250-molecule PubChem-channel replay. Neither protocol establishes competition-test performance.

## 2026-10-19, second update: per-molecule panel difficulty

### Files and column definitions

| File | Rows / format | Bytes | Purpose |
|---|---|---:|---|
| `DIFFICULTY_METHOD.json` | JSON | 1,847 | Panel source-cache hashes, candidate counts, reference matching, per-score identity collapse, tied-rank rules and limitations. |
| `panel_difficulty.parquet` | 250; Parquet | 75,661 | 250 panel molecules with candidate ambiguity, answer-presence flags, tied rank intervals, other-library references and natural-product labels. |

#### Columns of `panel_difficulty.parquet`

| Column | Type | Meaning |
|---|---|---|
| `molecule` | large_string | Public panel molecule scorer key; the 250-row order matches panel.parquet. |
| `formula` | large_string | Public panel answer formula, using exact RDKit formula text including charge. |
| `tier_window_rows` | uint32 | Number of cached PubChem rows within the measured neutral-mass window of +/-10 ppm. |
| `tier_window_identities` | uint32 | Distinct cached identities in the full mass window, using the public scorer-key/connectivity fallback. |
| `same_formula_window_rows` | uint32 | Cached mass-window rows whose RDKit formula equals the public truth formula. |
| `same_formula_window_identities` | uint32 | Distinct cached identities with the truth formula inside the measured window. |
| `same_formula_decoy_identities` | uint32 | Same-formula cached identities other than the panel answer. |
| `truth_in_tier_window` | bool | Whether the cached identity of the answer is represented in the measured tier window. |
| `truth_in_same_formula_window` | bool | Whether the answer is represented with its exact formula in the measured tier window. |
| `pool_window_rows` | uint32 | Original public panel count of pool rows in the measured +/-10 ppm mass window. |
| `n_query_spectra` | uint32 | Public spectra available for this panel molecule. |
| `n_reference_spectra_elsewhere` | uint32 | Number of public competition-training spectra with the same scorer key outside enveda-np-examples. |
| `reference_spectrum_elsewhere` | bool | True when at least one such other-library public reference spectrum exists. |
| `reference_libraries` | large_string | Sorted semicolon-separated public ingest_lib values with such a reference. |
| `np_pathway` | large_string | LOTUS 2026-04-13 NPClassifier pathway; sorted unique labels are joined by semicolons when the scorer identity has aliases. |
| `np_superclass` | large_string | LOTUS 2026-04-13 NPClassifier superclass, with the same alias-label convention. |
| `np_class` | large_string | LOTUS 2026-04-13 NPClassifier class, with the same alias-label convention; null when unavailable. |
| `partial_score_window_rank_min` | uint32 | Best tied answer rank for original partial-fingerprint score; full measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `partial_score_window_rank_max` | uint32 | Worst tied answer rank for original partial-fingerprint score; full measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `partial_score_same_formula_rank_min` | uint32 | Best tied answer rank for original partial-fingerprint score; exact-formula subset of measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `partial_score_same_formula_rank_max` | uint32 | Worst tied answer rank for original partial-fingerprint score; exact-formula subset of measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `fpnet_score_window_rank_min` | uint32 | Best tied answer rank for original full-fingerprint score; full measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `fpnet_score_window_rank_max` | uint32 | Worst tied answer rank for original full-fingerprint score; full measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `fpnet_score_same_formula_rank_min` | uint32 | Best tied answer rank for original full-fingerprint score; exact-formula subset of measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `fpnet_score_same_formula_rank_max` | uint32 | Worst tied answer rank for original full-fingerprint score; exact-formula subset of measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `ab_full_window_rank_min` | uint32 | Best tied answer rank for cached two-model ensemble full-fingerprint score; full measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `ab_full_window_rank_max` | uint32 | Worst tied answer rank for cached two-model ensemble full-fingerprint score; full measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `ab_full_same_formula_rank_min` | uint32 | Best tied answer rank for cached two-model ensemble full-fingerprint score; exact-formula subset of measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `ab_full_same_formula_rank_max` | uint32 | Worst tied answer rank for cached two-model ensemble full-fingerprint score; exact-formula subset of measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `fold0_full_window_rank_min` | uint32 | Best tied answer rank for public excluded-identity model full-fingerprint score; full measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `fold0_full_window_rank_max` | uint32 | Worst tied answer rank for public excluded-identity model full-fingerprint score; full measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `fold0_full_same_formula_rank_min` | uint32 | Best tied answer rank for public excluded-identity model full-fingerprint score; exact-formula subset of measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `fold0_full_same_formula_rank_max` | uint32 | Worst tied answer rank for public excluded-identity model full-fingerprint score; exact-formula subset of measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `ab_partial_window_rank_min` | uint32 | Best tied answer rank for cached two-model ensemble partial-fingerprint score; full measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `ab_partial_window_rank_max` | uint32 | Worst tied answer rank for cached two-model ensemble partial-fingerprint score; full measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `ab_partial_same_formula_rank_min` | uint32 | Best tied answer rank for cached two-model ensemble partial-fingerprint score; exact-formula subset of measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `ab_partial_same_formula_rank_max` | uint32 | Worst tied answer rank for cached two-model ensemble partial-fingerprint score; exact-formula subset of measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `fold0_partial_window_rank_min` | uint32 | Best tied answer rank for public excluded-identity model partial-fingerprint score; full measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `fold0_partial_window_rank_max` | uint32 | Worst tied answer rank for public excluded-identity model partial-fingerprint score; full measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `fold0_partial_same_formula_rank_min` | uint32 | Best tied answer rank for public excluded-identity model partial-fingerprint score; exact-formula subset of measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `fold0_partial_same_formula_rank_max` | uint32 | Worst tied answer rank for public excluded-identity model partial-fingerprint score; exact-formula subset of measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `popularity_window_rank_min` | uint32 | Best tied answer rank for cached documentation prior; full measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `popularity_window_rank_max` | uint32 | Worst tied answer rank for cached documentation prior; full measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `popularity_same_formula_rank_min` | uint32 | Best tied answer rank for cached documentation prior; exact-formula subset of measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `popularity_same_formula_rank_max` | uint32 | Worst tied answer rank for cached documentation prior; exact-formula subset of measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `prior_v10_window_rank_min` | uint32 | Best tied answer rank for cached extended documentation prior; full measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `prior_v10_window_rank_max` | uint32 | Worst tied answer rank for cached extended documentation prior; full measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `prior_v10_same_formula_rank_min` | uint32 | Best tied answer rank for cached extended documentation prior; exact-formula subset of measured mass window: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `prior_v10_same_formula_rank_max` | uint32 | Worst tied answer rank for cached extended documentation prior; exact-formula subset of measured mass window: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `pool_reference_in_other_libraries_ranker_score_rank_min` | uint32 | Best tied answer rank for cached main-channel ranker score; pool reference-available scenario: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `pool_reference_in_other_libraries_ranker_score_rank_max` | uint32 | Worst tied answer rank for cached main-channel ranker score; pool reference-available scenario: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `pool_reference_in_other_libraries_lib_max_rank_min` | uint32 | Best tied answer rank for cached library-similarity diagnostic; pool reference-available scenario: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `pool_reference_in_other_libraries_lib_max_rank_max` | uint32 | Worst tied answer rank for cached library-similarity diagnostic; pool reference-available scenario: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `pool_reference_in_other_libraries_popularity_rank_min` | uint32 | Best tied answer rank for cached documentation prior; pool reference-available scenario: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `pool_reference_in_other_libraries_popularity_rank_max` | uint32 | Worst tied answer rank for cached documentation prior; pool reference-available scenario: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `pool_structure_in_pool_no_reference_ranker_score_rank_min` | uint32 | Best tied answer rank for cached main-channel ranker score; pool structure-present, reference-removed scenario: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `pool_structure_in_pool_no_reference_ranker_score_rank_max` | uint32 | Worst tied answer rank for cached main-channel ranker score; pool structure-present, reference-removed scenario: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `pool_structure_in_pool_no_reference_lib_max_rank_min` | uint32 | Best tied answer rank for cached library-similarity diagnostic; pool structure-present, reference-removed scenario: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `pool_structure_in_pool_no_reference_lib_max_rank_max` | uint32 | Worst tied answer rank for cached library-similarity diagnostic; pool structure-present, reference-removed scenario: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `pool_structure_in_pool_no_reference_popularity_rank_min` | uint32 | Best tied answer rank for cached documentation prior; pool structure-present, reference-removed scenario: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `pool_structure_in_pool_no_reference_popularity_rank_max` | uint32 | Worst tied answer rank for cached documentation prior; pool structure-present, reference-removed scenario: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `pool_structure_removed_from_pool_ranker_score_rank_min` | uint32 | Best tied answer rank for cached main-channel ranker score; pool answer-removed scenario: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `pool_structure_removed_from_pool_ranker_score_rank_max` | uint32 | Worst tied answer rank for cached main-channel ranker score; pool answer-removed scenario: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `pool_structure_removed_from_pool_lib_max_rank_min` | uint32 | Best tied answer rank for cached library-similarity diagnostic; pool answer-removed scenario: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `pool_structure_removed_from_pool_lib_max_rank_max` | uint32 | Worst tied answer rank for cached library-similarity diagnostic; pool answer-removed scenario: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |
| `pool_structure_removed_from_pool_popularity_rank_min` | uint32 | Best tied answer rank for cached documentation prior; pool answer-removed scenario: one plus identities with strictly higher finite score. Max score per identity; descending. Null means answer has no finite cached score. |
| `pool_structure_removed_from_pool_popularity_rank_max` | uint32 | Worst tied answer rank for cached documentation prior; pool answer-removed scenario: identities with finite score greater than or equal to answer score. Same identity collapse as rank_min. |

### Method and validation

`panel_difficulty.parquet` has one row for each of the 250 panel molecules, in exactly the order of `panel.parquet`. It joins full mass-window and exact-formula candidate counts, same-formula decoy counts, answer-presence flags, other-library reference coverage and LOTUS NPClassifier labels. The mass window is the measured neutral mass ±10 parts per million. The median exact-formula window contains 859 distinct cached identities. The data table and method record total 77,509 bytes.

Each raw fingerprint or documentation score is ranked separately in the full measured window and its exact-formula subset. Pool ranker, library-similarity and popularity scores are ranked separately for each existing scenario, excluding generated candidates. Each score first takes its maximum within each cached identity and then ranks identities in descending order. `rank_min` is one plus the number of identities with strictly higher finite score; `rank_max` counts identities with score greater than or equal to the answer. Null means the answer has no finite cached score. Best and worst tied ranks are retained, without imposing an arbitrary tie order. Cached scorer keys retain the public connectivity fallback for unvisited rows.

Other-library references are exact scorer-key matches in competition training spectra outside `enveda-np-examples`. Structure matching uses the exact source key and source SMILES before joining the scorer key; no first-alias fallback is used. Counts include aliases. All 250 answers have at least one such reference. Natural-product classes are available for 200 rows; labels combine sorted unique LOTUS class annotations across scorer-identity aliases.

All 250 rows match the panel order. Window counts and truth-presence checks have zero mismatches. Every non-null rank interval has matching nullness and minimum no greater than maximum. `DIFFICULTY_METHOD.json` records the source-cache SHA-256 values, reference semantics and tie rules.

### Limits and source terms

These are diagnostic ranks of cached raw scores. They do not reproduce every shortlist, exclusion gate or final-list mixture. A reference elsewhere is an observed public-library match, not proof that a model excluded the identity from training. This public panel is enriched for documented natural products and all its molecules have reference coverage elsewhere; these results do not estimate hidden-test performance. No hidden-test labels are included.

Panel structures and spectrum-derived diagnostics retain the [competition terms](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) and original resource terms. The LOTUS 2026-04-13 frozen export is CC BY 4.0 (https://doi.org/10.5281/zenodo.19360665; Rutz et al., https://doi.org/10.7554/eLife.70780); the retained NPAtlas snapshot is conservatively CC BY-NC 4.0 (https://www.npatlas.org/download; https://doi.org/10.1093/nar/gkab941). Credit NPClassifier, Kim et al., https://doi.org/10.1021/acs.jnatprod.1c00399, and PubChem / U.S. National Library of Medicine, https://doi.org/10.1093/nar/gkae1059. No blanket CC0 relicensing is asserted.

### Embedded validation assignments

The `casmi26-identity-disjoint-validation-folds/` folder is refreshed to match the standalone [validation-fold dataset](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-identity-disjoint-validation-folds) byte for byte. The six added files below contribute 15,599,197 bytes. Original identity/scaffold assignment files remain included. The following file and column definitions and methods apply to these identical embedded copies.

| File | Rows / format | Bytes | Purpose |
|---|---|---:|---|
| `casmi26-identity-disjoint-validation-folds/SPLIT_METHOD.json` | JSON | 1,415 | Library and instrument exclusion rules, exact Morgan analogue graph, fold seed, source hashes and validation. |
| `casmi26-identity-disjoint-validation-folds/analogue_domain_summary.parquet` | 445; Parquet | 7,732 | 445 analogue-fold/category cells with spectrum and identity-group counts by library, raw instrument type and instrument family. |
| `casmi26-identity-disjoint-validation-folds/spectra_splits.parquet` | 2,539,608; Parquet | 9,272,714 | 2,539,608 aligned training-spectrum rows with library, instrument-type/family identifiers and analogue folds. |
| `casmi26-identity-disjoint-validation-folds/split_categories.csv` | 103; CSV | 4,223 | 103 library, raw instrument-type and normalized instrument-family labels, with identifiers, membership bits and family mapping. |
| `casmi26-identity-disjoint-validation-folds/split_summary.parquet` | 108; Parquet | 10,206 | 108 library, instrument and analogue split summaries, including target spectra, identity exclusions and training complements. |
| `casmi26-identity-disjoint-validation-folds/structure_splits.parquet` | 277,566; Parquet | 6,302,907 | 277,566 aligned structure rows with identity-wide library/instrument memberships, complete analogue components and five analogue folds. |

#### Columns of `casmi26-identity-disjoint-validation-folds/analogue_domain_summary.parquet`

| Column | Type | Meaning |
|---|---|---|
| `fold` | uint32 | Analogue fold number, 0 through 4. |
| `id` | uint32 | Zero-based category identifier within its axis. |
| `n_spectra` | uint32 | Number of public training spectra in this analogue-fold/category cell. |
| `identity_groups` | uint32 | Number of distinct original identity groups in this analogue-fold/category cell. |
| `axis` | large_string | Category axis: library, exact instrument type or normalized instrument family. |
| `label` | large_string | Category label for the library, original instrument type or normalized instrument family. |

#### Columns of `casmi26-identity-disjoint-validation-folds/spectra_splits.parquet`

| Column | Type | Meaning |
|---|---|---|
| `row` | uint32 | Zero-based original competition-training spectrum row. |
| `library_id` | uint8 | Original ingest_lib category ID from split_categories.csv. |
| `instrument_type_id` | uint16 | Original instrument_type category ID, including an explicit unknown category. |
| `instrument_family_id` | uint8 | Normalized instrument-family category ID from split_categories.csv. |
| `fold_analogue` | int8 | Five-way analogue fold, 0 through 4; every complete connected component belongs to one fold. |

#### Columns of `casmi26-identity-disjoint-validation-folds/split_categories.csv`

| Column | Type | Meaning |
|---|---|---|
| `axis` | str | Category axis: library, exact instrument type or normalized instrument family. |
| `id` | int64 | Zero-based category identifier within its axis. |
| `label` | str | Original category label, normalized instrument-family label or analogue-fold number. |
| `mask_bit` | str | Bit value for the library or instrument-family membership mask; absent for raw instrument types. |
| `instrument_family` | str | Normalized family assigned to an original instrument-type label; empty for other axes. |

#### Columns of `casmi26-identity-disjoint-validation-folds/split_summary.parquet`

| Column | Type | Meaning |
|---|---|---|
| `split` | large_string | Library or instrument exclusion axis, or the analogue assignment. |
| `fold` | uint32 | Category ID for domain category exclusions, or analogue fold 0 through 4. |
| `label` | large_string | Original category label, normalized instrument-family label or analogue-fold number. |
| `validation_primary_spectra` | uint32 | Number of spectra originally belonging to the held-out source or instrument category. |
| `heldout_identity_groups` | uint32 | Number of distinct identity groups represented in the held-out category. |
| `heldout_structure_rows` | uint32 | Number of structure rows excluded from training by the category exclusion. |
| `heldout_all_spectra` | uint32 | All spectra of held-out identities, including their spectra from other categories. |
| `outside_category_spectra_removed_from_training` | uint32 | Spectra outside the validation category removed because their identity occurs inside it. |
| `training_spectra` | uint32 | Training complement after removing every held-out identity or analogue component. |
| `analogue_clusters` | uint32 | Number of complete chemical-similarity components assigned to this fold. |

#### Columns of `casmi26-identity-disjoint-validation-folds/structure_splits.parquet`

| Column | Type | Meaning |
|---|---|---|
| `structure_row` | uint32 | Zero-based row in the original structures.parquet. |
| `inchikey14` | large_string | Original stored connectivity block in structures.parquet. |
| `identity_group` | uint32 | Existing identity connected component, linking stored and scorer keys; unchanged from the original folds. |
| `library_holdout_mask` | uint16 | Bit i is set if any spectrum of this identity group belongs to library category i. Remove the whole identity group from training when holding out that library. |
| `instrument_family_holdout_mask` | uint16 | Bit i is set if any spectrum of this identity group belongs to instrument family i. |
| `analogue_cluster` | uint32 | Exact connected-component ID of the Morgan Tanimoto >=0.7 graph plus existing identity links; same ID always stays in one analogue fold. |
| `fold_analogue` | int8 | Five-way analogue fold, 0 through 4; every complete connected component belongs to one fold. |
| `fingerprint_valid` | bool | Whether the stored structure produced a sanitized Morgan fingerprint. Unparseable rows retain identity links but add no fingerprint-similarity edges. |
| `instrument_type_ids` | list of uint16 | List of all raw instrument-type category IDs observed for this identity group; supports exact-type category exclusions without truncating categories. |

#### Embedded split methods, validation and limits


The 277,566 structure rows align with `structures.parquet`; the 2,539,608 spectrum rows align with `spectra_folds.parquet` and the original competition training file. The source training SHA-256 is `9423f90ec08b941a9646b535bf0606183e86ff0c5a297978f5a70693d41bcd98`. Existing identity and scaffold assignments are unchanged. The new tables and method record total 15,599,197 bytes.

`split_categories.csv` lists 11 original `ingest_lib` libraries, 83 original instrument-type labels and nine normalized instrument families. A molecular identity can occur in several categories. Library and instrument-family masks therefore combine membership across the complete original identity group; `instrument_type_ids` retains every observed original instrument-type ID. Bit i denotes category i. Unknown labels remain explicit. When loading the CSV with pandas, use `keep_default_na=False`, because the literal instrument label `null` is a category.

For a held-out library or instrument category, remove every spectrum of every identity group represented in that category from training. Evaluate the target category's spectra. Removing only the target category's spectra would retain the same molecule from another source in training. For a library or family mask, exclude groups with `(mask & mask_bit) != 0`; for an exact instrument type, exclude groups containing the selected ID in `instrument_type_ids`. `split_summary.parquet` reports target-category spectra, all removed spectra, shared-identity exclusions and the training complement. `analogue_domain_summary.parquet` reports spectrum and identity-group counts by analogue fold and each category, making composition differences visible without cutting molecular groups.

### Analogue method and validation

A fingerprint is a binary list of molecular features. RDKit 2026.03.3 computes chirality-free Morgan radius-2, 2,048-bit fingerprints. A graph links every pair with Tanimoto similarity at least 0.7, together with all existing identity-group links. Duplicate fingerprints are merged before the search. Exact bit-count bounds remove only impossible pairs; every remaining eligible pair is evaluated. Connected components [groups joined directly or through chains of links] remain intact, so transitive chains of related molecules stay in one fold.

The search contains 266,158 distinct fingerprints, 29,328,478,344 eligible pair comparisons and 195,770 threshold edges. It produces 194,357 components; the largest has 8,148 structure rows. Components are ordered by decreasing size, with seeded random tie breaking (seed 20261019), then assigned greedily to the fold with the fewest structure rows. The five counts are 55,514, 55,513, 55,513, 55,513 and 55,513. No component is cut to improve balance.

Every structure and spectrum mapping agrees with its original row, with zero alignment mismatches. Identity and analogue-component cross-fold violations are zero. All structure rows have valid sanitized fingerprints (zero failures). If a row fails in another build, it retains identity links but contributes no similarity edges. An independent full-library search for 160 fixed sampled fingerprints found zero cross-fold neighbors at or above the threshold. `SPLIT_METHOD.json` records the graph, hashes and validation.

### Limits and terms

Analogue separation holds for this documented Morgan/Tanimoto metric, not every definition of chemical similarity. Whole components can limit fold balance; library and instrument compositions can differ between analogue folds. Domain memberships deliberately preserve overlaps, so the complete identity complement must supply training data. Existing public pretrained models may have seen the evaluation identities or their analogues; clean estimates require models fitted under the corresponding exclusions. These additions contain identifiers and assignments, with no spectra or hidden-test labels.

Credit Enveda's [CASMI 2026 competition data](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) and RDKit contributors (BSD 3-Clause). Competition and original source-data terms remain applicable.



### Companion build records

`CARD_ADDITIONS.md` explains the additions, `ADDED_FILE_DESCRIPTIONS.json` records build-time file and column metadata, and `ADDITIONS_MANIFEST.json` records build-time file sizes, hashes and provenance. The current card and these notes describe the delivery layout. A distribution inventory must describe the actual distributed filenames and dataset locations.
