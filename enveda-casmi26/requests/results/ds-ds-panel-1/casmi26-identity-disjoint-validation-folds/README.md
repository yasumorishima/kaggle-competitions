These are five ready-made folds of the CASMI 2026 training file for local validation: fit a model on four folds, score it on the fifth, and the score shows how the model handles molecules it has never seen. A random split of spectra cannot show this, because each molecule has about nine spectra and they land on both sides: in a random 20% split, 185,962 molecules appeared in both training and validation, while identity fold 0 shares none. The fuller [panel benchmark](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-natural-product-panel-benchmark) contains these folds unchanged, plus 250 Enveda natural products with cached candidates for quick ranking tests.

![One molecule has several spectra: a random split puts them on both sides, a split by molecule keeps them together.](https://raw.githubusercontent.com/Gluzdov-Dmitrii/Enveda-CASMI-2026/main/kaggle-datasets/casmi26-identity-disjoint-validation-folds/molecule_split.png)

## Quick start

The code finds the dataset and the competition files in any of the Kaggle mount locations.

```python
from pathlib import Path
import pandas as pd
slug, comp = 'casmi26-identity-disjoint-validation-folds', 'enveda-CASMI26-molecule-id-mass-spectra'
D = next(p for p in (Path('/kaggle/input/datasets/dmitriigluzdov') / slug, Path('/kaggle/input') / slug) if p.is_dir())
C = next(p for p in (Path('/kaggle/input/competitions') / comp, Path('/kaggle/input') / comp,
                     Path('/kaggle/input') / comp.lower()) if p.is_dir())
train = pd.read_parquet(C / 'train.parquet', columns=['inchikey14'])
folds = pd.read_parquet(D / 'structures.parquet').drop_duplicates('inchikey14').set_index('inchikey14')
train['fold'] = folds['fold_identity'].reindex(train['inchikey14']).values   # 'fold_scaffold' for the harder test
assert train['fold'].notna().all()     # a newer training file may bring keys without a fold
k = 0                                  # validation fold, 0-4
fit, valid = train[train['fold'] != k], train[train['fold'] == k]
assert not set(fit['inchikey14']) & set(valid['inchikey14'])
print(len(fit), 'fitting spectra,', len(valid), 'validation spectra')   # 2027776 and 511832 for fold 0
```

## How to validate locally

1. Pick one fold for validation and fit on the other four. Use `fold_identity`, or `fold_scaffold` for a harder test where molecules with the same ring framework also stay together.
2. Keep the validation molecules out of everything learned from data: model training, the reference-spectrum library used for search, and any feature fitted on labels.
3. Decide whether the validation structures stay in the candidate pool [the list of possible answers]. Keeping them tests a known structure without a reference spectrum; removing them tests a structure missing from the pool.
4. Report how often the right answer reaches the candidate list, then MRR@25 [the competition metric: the mean of 1 divided by the rank of the right answer in the first 25 guesses, zero if it is not there].
5. Public models trained on all training data have already seen every fold. Use models fitted without the validation fold, such as the [fold-safe FPNet](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-fold-safe-fpnet).

## What is inside

| File | Contents |
|---|---|
| `structures.parquet` | 277,566 training structures with their identity group, ring-system group and both fold labels. |
| `spectra_folds.parquet` | The same labels for each of the 2,539,608 training spectra, in training-file order. |
| `structure_splits.parquet`, `spectra_splits.parquet` | Further splits: hold out a whole source library or instrument family, or keep similar molecules (Morgan Tanimoto ≥ 0.7) in one of five analogue folds. |
| `split_summary.parquet`, `split_categories.csv`, `analogue_domain_summary.parquet`, `SPLIT_METHOD.json` | Category labels, the size of each held-out set and the similarity rule. |
| `SUMMARY.json`, `build_folds.py` | Counts, RDKit version, seed and training-file hash; the script that rebuilds and checks the folds. |
| `TECHNICAL_NOTES.md` | Column definitions, build steps and checks. |

Identity follows the competition scorer: the first 14 characters of the InChIKey after RDKit picks a standard tautomer [a form of the molecule differing only in proton and double-bond positions]. Structures that share the stored key or the scorer key form one group, giving 274,184 identity groups; for about 1.6% of structures the two keys differ.

## How it was made

RDKit 2026.03.3, the scorer's version, recomputes the keys and the ring systems. Linked structures form groups, and a seeded shuffle assigns whole groups to five folds of 55,512–55,514 structures, each with a similar share of molecules measured by Enveda. The build script checks that no identity crosses folds, and no ring system for the scaffold folds.

## Updates

- **2026-10-19:** Added library, instrument and analogue-disjoint assignments, with category and fold-composition summaries.

## Licence and credits

The identifiers and structures come from Enveda's competition training file and remain under the [competition data terms](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data). No spectra, hidden-test data or test-molecule labels are included. The assignments and build script are by Dmitrii Gluzdov. RDKit provides the structure standardisation and ring-system calculations.
