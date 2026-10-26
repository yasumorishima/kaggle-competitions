This dataset lets you test a ranking idea in seconds, without a GPU, on the 250 molecules that resemble the hidden test most: natural products measured by Enveda on timsTOF instruments, together with every candidate structure the public pipeline considered for each of them and the scores it gave. Two rules of the [data notebook](https://www.kaggle.com/code/dmitriigluzdov/casmi-26-the-data-behind-spectra-to-structures) were chosen on it: in the PubChem search the fingerprint model alone reaches MRR@25 0.51 and preferring well-documented structures raises it to 0.95, and when the answer is missing from the candidate pool, moving the best PubChem proposal up raises 0.36 to 0.72. The absolute numbers are optimistic because these molecules are in the training file, so trust the difference between two rules rather than a score on its own.

![Test a ranking rule in seconds: in the PubChem search the fingerprint model alone gives 0.512 and preferring well-documented structures 0.951; with the answer missing from the pool the main list gives 0.361 and moving the best PubChem proposal up 0.718; a fingerprint model ranks the right structure first for 42% of these molecules if it never saw them and for 89% once trained on them.](https://raw.githubusercontent.com/Gluzdov-Dmitrii/Enveda-CASMI-2026/main/kaggle-datasets/casmi26-natural-product-panel-benchmark/what_it_found.png)

## How to use it

- **Try a rule.** Re-sort the cached candidates in `pubchem_candidates.parquet` or `main_candidates.parquet` with your own score and compare MRR@25 with the current rule; the Quick start does this for the PubChem search.
- **Pick a threshold.** The stored merge results count how many molecules a rule moves up and down; the notebook's rule for moving a PubChem proposal up (`S > 6` and `pop ≥ 5`) was picked this way.
- **Check a model on molecules it has not seen.** Fingerprint models whose training code leaves this library out put the right structure first for 42% of the molecules; a model trained on them does so for 89%. Cached predictions of FPNet A+B and the [fold-safe FPNet](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-fold-safe-fpnet), which never saw these molecules, are included.
- **Validate on the whole training file.** `casmi26-identity-disjoint-validation-folds/` is an unchanged copy of the [identity-disjoint folds](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-identity-disjoint-validation-folds).

## The three situations

The main list was replayed in three situations that hide more and more of the answer.

| Situation | What is hidden | Like a test molecule that… | Main list MRR@25 |
|---|---|---|---:|
| Other reference | its own spectra in this library | has a reference spectrum from another instrument | 0.980 |
| No reference | every reference spectrum of it | is a known structure without any reference spectrum | 0.988 |
| Not in the pool | all its spectra and its structure in the pool | is only in PubChem, or is new | 0.361 |

The first two scores are almost perfect because the ranking models have probably seen these molecules in training; 0.988 against 0.980 is two or three molecules, which is noise. The third is still 0.361 because the pipeline rebuilds some answers by editing the structures of similar library compounds, which works better for these well-known natural products than it would for new chemistry.

## Quick start

The example reads the dataset from either supported Kaggle mount location and replays the PubChem search with the fingerprint score alone and with a preference for well-documented structures.

```python
from pathlib import Path
import numpy as np
import pandas as pd
slug = 'casmi26-natural-product-panel-benchmark'
D = next(p for p in (Path('/kaggle/input/datasets/dmitriigluzdov') / slug, Path('/kaggle/input') / slug) if p.is_dir())
cand = pd.read_parquet(D / 'pubchem_candidates.parquet')   # every PubChem candidate of every panel molecule
cand = cand[np.isfinite(cand.fpnet_score)]

def mrr25(score):                                           # a scoring rule -> MRR@25 of the PubChem search
    out = []
    for _, g in cand.assign(s=score).groupby('molecule', observed=True):
        g = g[~g.in_pool | g.is_truth].sort_values('s', ascending=False)
        ident = np.where(g.scorer_key != '', g.scorer_key, g.connectivity)
        first = ~pd.Series(ident).duplicated().values        # one guess per scored identity
        rank = np.flatnonzero(g.is_truth.values[first])
        out.append(1 / (rank[0] + 1) if len(rank) and rank[0] < 25 else 0)
    return np.mean(out + [0] * (250 - len(out)))

z = cand.groupby('molecule', observed=True).fpnet_score.transform(lambda x: (x - x.mean()) / x.std())
print(mrr25(z), mrr25(z + 0.25 * (cand.log_substances + cand.log_pubmed)))   # 0.512 and 0.969
```

This simplified replay scores the whole mass window; the notebook's 5,000-row shortlist gives 0.951 for the second rule.

## What is inside

| File | Contents |
|---|---|
| `panel.parquet` | The 250 molecules: measured masses, spectrum counts and candidate-window sizes. |
| `pubchem_candidates.parquet` | All 1.94 million PubChem candidates of the panel molecules with fingerprint and documentation scores. |
| `main_candidates.parquet` | The main list's candidates (pool and generated structures) with ranker scores for the three situations. |
| `pubchem_signals.parquet`, `pubchemlite_annotations.parquet` | Documentation and natural-product annotations aligned to the candidates. |
| `panel_logits.npz`, `panel_fpnet_scores.parquet`, `panel_fpnet_per_molecule.csv` | Fingerprint predictions for the panel by FPNet A+B and the fold-safe FPNet, and the candidate scores they give. |
| `panel_difficulty.parquet`, `DIFFICULTY_METHOD.json` | Per-molecule difficulty: candidate counts, tied ranks, reference coverage and chemical class. |
| `results_*.json`, `prior_rule_summary.csv`, `PANEL_REPLAY_PUBLIC.json` | The recorded comparisons behind the figure. |
| `replay_public_priors.py`, `summarize_panel_fpnet.py`, `*_eval.py` | Scripts that reproduce them without a GPU. |
| `casmi26-identity-disjoint-validation-folds/` | Unchanged copy of the identity-disjoint folds dataset. |

Column definitions, build steps and validation are in `TECHNICAL_NOTES.md`.

## How it was made

The panel was drawn from the `enveda-np-examples` library of the training file, and the public pipeline of Ahmed Berat Özer produced the candidate lists and scores. The scripts replay candidate selection, scoring and list merging from these tables; the replays follow the earlier single-engine configuration of the notebook, without the ICEBERG and GLACIER re-ordering. The panel was used to choose thresholds, so its differences describe this panel and transfer to other chemistry is untested. 249 of the 250 answers are in COCONUT: these are well-documented natural products, which favours documentation priors.

## Updates

- **2026-10-20, second update:** Natural-product flags, LOTUS references and classes moved to the LOTUS frozen release of 2026-04-13; the replays that use them change by at most one molecule.
- **2026-10-19, second update:** Added per-molecule difficulty diagnostics and refreshed the nested folds folder to match the standalone folds dataset byte for byte.
- **2026-10-19:** Added self-contained documentation replay, source-aligned annotations and cached model comparisons.
- **2026-10-13:** Added the complete identity-based validation assignments for the competition training file.

## Licence and credits

Identifiers, masses, answer formulas, scores and validation assignments remain under the [competition data terms](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data). The added replay helper is Apache-2.0. No spectra or hidden-test data are distributed. PubChem and other annotations retain their source-specific terms.

Credit Enveda for the panel data; [Ahmed Berat Özer](https://www.kaggle.com/ahmedberatozer) for the public pipeline, FPNet models, candidate collections and LightGBM rankers; and [Sorokina et al., COCONUT](https://doi.org/10.1186/s13321-020-00478-9) for the natural-product structures. PubChem documentation is credited to the U.S. National Library of Medicine and [Kim et al.](https://doi.org/10.1093/nar/gkae1059). The LOTUS frozen export is the 2026-04-13 release ([Zenodo 19360665](https://doi.org/10.5281/zenodo.19360665)), CC BY 4.0; credit [Rutz et al.](https://doi.org/10.7554/eLife.70780) and the LOTUS contributors; NPAtlas specifies CC BY-NC 4.0 ([van Santen et al.](https://doi.org/10.1093/nar/gkab941)), with snapshot-specific attribution retained. PubChemLite and its candidate files are credited to [Schymanski et al., 2021](https://doi.org/10.1186/s13321-021-00489-0), cosmicbe and franciscoangulo under CC0.
