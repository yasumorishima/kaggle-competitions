# CASMI26 Identity-Disjoint Validation Folds — Technical notes

These notes retain the complete methods, file and column definitions, measurements, caveats, updates and source credits. The companion Kaggle card provides a short introduction.

## Terms used in these notes

**Candidate** [a possible molecular answer].

**Pool** [a collection of candidate structures].

**Fold** [one group reserved for checking a model while the other groups supply training data].

**InChIKey connectivity block** [the first 14 InChIKey characters, identifying connectivity without stereochemistry]. The scorer key is calculated after RDKit chooses a standard tautomer [a form differing in proton positions and bond arrangement].

**MRR@25** [the mean of 1 divided by the correct answer’s rank among the first 25 unique scored identities; a missing answer or a rank beyond 25 contributes zero].

## Dataset overview

Ready-made, leak-free validation folds for the training file of
[Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).

The competition scores a guess by the first 14 InChIKey characters **after RDKit tautomer canonicalisation**.
A random row split allows identity overlap: one molecule has on average nine spectra, and 1.6% of training structures carry a
stored `inchikey14` that differs from the key the scorer will compute. These folds keep every spectrum of one *scored
identity* on one side, so a model can be validated on molecules it has never seen.

## Quick start

```python
import pandas as pd
F = '/kaggle/input/casmi26-identity-disjoint-validation-folds'
train = pd.read_parquet('/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet',
                        columns=['inchikey14', 'normalized_smiles', 'precursor_mz', 'adduct'])
folds = pd.read_parquet(f'{F}/structures.parquet').drop_duplicates('inchikey14').set_index('inchikey14')

train['fold'] = folds.fold_identity.reindex(train.inchikey14).values     # or fold_scaffold for the harder split
assert train.fold.notna().all()  # assign new keys before fitting
fit, valid = train[train.fold != 0], train[train.fold == 0]
assert not set(fit.inchikey14) & set(valid.inchikey14)
```

`spectra_folds.parquet` gives the same assignment row-by-row for the training snapshot with SHA-256
`9423f90e…bcd98` (2,539,608 spectra); joining through `inchikey14` as above covers existing keys after a training-file update; new keys need assignments.

## What is inside

| File | Rows | Content |
|---|---:|---|
| `structures.parquet` | 277,566 | One row per distinct training structure: stored key, scorer key, scaffold, groups and both fold labels. |
| `spectra_folds.parquet` | 2,539,608 | One row per training spectrum in file order: `row`, `inchikey14`, group and fold labels. |
| `SUMMARY.json` | | Counts per fold, number of groups, RDKit version, seed and the training-file hash. |
| `build_folds.py` | | The complete script that produced the files. |

Two 5-fold assignments are provided:

* **`fold_identity`** keeps each scored identity in one fold. Structures are linked when they share the scorer key
  *or* the stored `inchikey14`, which merges 277,566 structure rows into **274,184 identity groups**. Use it to
  estimate performance on molecules without a reference spectrum of their own.
* **`fold_scaffold`** additionally keeps every Bemis–Murcko scaffold in one fold (**140,160 scaffold groups**; acyclic
  molecules form their own groups). Analogues with the same ring system can no longer help, so this split is a
  tougher test of chemical generalisation.

Both assignments are balanced by structure count (55,512–55,514 per fold) and stratified by whether a group has
Enveda spectra, so every fold contains a similar share of the instrument domain of the hidden test.

## Validation protocol

1. **Remove held-out identities everywhere**: from model training, from the reference-spectrum library used for
   retrieval, and from any feature table fitted on labels.
2. **Decide what the candidate pool may contain.** Keeping the held-out structure in the candidate database simulates a
   "known structure, no reference spectrum" molecule; removing it simulates a structure absent from the database.
3. **Report candidate coverage before ranking metrics**, and MRR@25 over all held-out molecules, including those whose
   answer never enters the candidate list.
4. **Pretrained public checkpoints have usually seen all five folds.** A clean estimate needs models that were fitted
   on the four training folds only.

## Method

`normalized_smiles` is parsed with RDKit 2026.03.3 (the scorer's version); the canonical tautomer from
`rdMolStandardize.TautomerEnumerator` gives the scorer key and the Murcko scaffold. Groups are connected components
of rows that share a label. Whole groups are assigned greedily to the currently smallest fold after a seeded shuffle
(seed 20260930). The script asserts scorer-key and stored-key separation for both assignments, and scaffold separation for scaffold folds.

## Terms

The identifiers and structures are derived from the competition training file and remain under the
[competition data terms](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data).
No spectra, hidden test data or labels of test molecules are included.

## Applying the assignments after a data update

A key join covers structures already represented in the assignment. Check `train.fold.notna().all()` before selecting fitting and validation rows; previously absent identities need assignments first. Direct row alignment applies only to the training-file hash recorded in `SUMMARY.json`.

The full training-file SHA-256 is `9423f90ec08b941a9646b535bf0606183e86ff0c5a297978f5a70693d41bcd98`. The builder checks scorer-key and source-key separation for both assignments, and scaffold separation only for `fold_scaffold`. The identity assignment permits related ring systems on opposite sides.

For the figure, a NumPy random assignment (`default_rng(20261019).integers(0, 5, size=n) == 0`) places 506,327 of 2,539,608 spectra in validation and shares 185,962 molecular identity groups with fitting. Supplied identity fold 0 places 511,832 spectra in validation and shares zero identity groups. Both use the same source snapshot; shared groups are not duplicated-spectrum counts or an accuracy measurement.

## 2026-10-19: library, instrument and analogue-disjoint assignments

### Files and column definitions

| File | Rows / format | Bytes | Purpose |
|---|---|---:|---|
| `SPLIT_METHOD.json` | JSON | 1,415 | Library and instrument exclusion rules, exact Morgan analogue graph, fold seed, source hashes and validation. |
| `analogue_domain_summary.parquet` | 445; Parquet | 7,732 | 445 analogue-fold/category cells with spectrum and identity-group counts by library, raw instrument type and instrument family. |
| `spectra_splits.parquet` | 2,539,608; Parquet | 9,272,714 | 2,539,608 aligned training-spectrum rows with library, instrument-type/family identifiers and analogue folds. |
| `split_categories.csv` | 103; CSV | 4,223 | 103 library, raw instrument-type and normalized instrument-family labels, with identifiers, membership bits and family mapping. |
| `split_summary.parquet` | 108; Parquet | 10,206 | 108 library, instrument and analogue split summaries, including target spectra, identity exclusions and training complements. |
| `structure_splits.parquet` | 277,566; Parquet | 6,302,907 | 277,566 aligned structure rows with identity-wide library/instrument memberships, complete analogue components and five analogue folds. |

#### Columns of `analogue_domain_summary.parquet`

| Column | Type | Meaning |
|---|---|---|
| `fold` | uint32 | Analogue fold number, 0 through 4. |
| `id` | uint32 | Zero-based category identifier within its axis. |
| `n_spectra` | uint32 | Number of public training spectra in this analogue-fold/category cell. |
| `identity_groups` | uint32 | Number of distinct original identity groups in this analogue-fold/category cell. |
| `axis` | large_string | Category axis: library, exact instrument type or normalized instrument family. |
| `label` | large_string | Category label for the library, original instrument type or normalized instrument family. |

#### Columns of `spectra_splits.parquet`

| Column | Type | Meaning |
|---|---|---|
| `row` | uint32 | Zero-based original competition-training spectrum row. |
| `library_id` | uint8 | Original ingest_lib category ID from split_categories.csv. |
| `instrument_type_id` | uint16 | Original instrument_type category ID, including an explicit unknown category. |
| `instrument_family_id` | uint8 | Normalized instrument-family category ID from split_categories.csv. |
| `fold_analogue` | int8 | Five-way analogue fold, 0 through 4; every complete connected component belongs to one fold. |

#### Columns of `split_categories.csv`

| Column | Type | Meaning |
|---|---|---|
| `axis` | str | Category axis: library, exact instrument type or normalized instrument family. |
| `id` | int64 | Zero-based category identifier within its axis. |
| `label` | str | Original category label, normalized instrument-family label or analogue-fold number. |
| `mask_bit` | str | Bit value for the library or instrument-family membership mask; absent for raw instrument types. |
| `instrument_family` | str | Normalized family assigned to an original instrument-type label; empty for other axes. |

#### Columns of `split_summary.parquet`

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

#### Columns of `structure_splits.parquet`

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

### Library and instrument method

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
