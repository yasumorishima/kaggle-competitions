# XC External Data Integration Design (ProtoSSM v4 fork)

日付: 2026-04-19
前提: xc-perch-v2-embed-birdclef-2026 kernel 完了 → `xc_perch_embeddings.npz` + `.parquet` を output dataset として取得済
目的: 現 0.926 yuriygreben/birdclef-2026-improved-ensemble fork に XC 由来 embeddings を追加学習データとして注入し 0.929 超えを狙う

## 前提: 現 fork の tensor shape

| 変数 | shape | 意味 |
|---|---|---|
| `emb_full` | (N_win_total, 1536) | 全 5 秒窓の Perch v2 embedding |
| `scores_full_raw` | (N_win_total, 234) | Perch v2 logits per window |
| `meta_full` | DataFrame(N_win_total) | filename/site/hour_utc/row_id/index |
| `Y_FULL` | (N_win_total, 234) | 5-sec window レベルラベル |
| `emb_files` | (N_files, 12, 1536) | file-level reshape (12 = N_WINDOWS) |
| `logits_files` | (N_files, 12, 234) | 同上 |
| `labels_files` | (N_files, 12, 234) | 同上 |
| `site_ids_all` | (N_files,) int | site embedding lookup |
| `hours_all` | (N_files,) int | hour embedding lookup |

## XC 側の入力 (kernel output)

`xc_perch_embeddings.npz`:
- `emb`: (N_xc_win, 1536) float32
- `logits`: (N_xc_win, 234) float32

`xc_perch_embeddings.parquet`:
- columns: `xc_id, label, scientific, window_idx, q, country, lic`
- 1 行 = 1 window。xc_id でグルーピングして per-file 構造化可

## 差し込み点: fork の Cell 6 直後に新規 Cell 追加

`full_cache_input_dir` の `perch-meta` dataset ロード直後に、追加で XC cache をロードして
`emb_full` / `scores_full_raw` / `Y_FULL` / `meta_full` に concat する。

### reshape helper

```python
def reshape_xc_to_files(xc_emb, xc_logits, xc_meta, label_to_idx, n_windows=12, n_classes=234):
    # group by xc_id, pad to n_windows
    grouped = xc_meta.groupby('xc_id', sort=False)
    file_ids, label_codes = [], []
    emb_files_list, log_files_list, lbl_files_list = [], [], []

    for xc_id, g in grouped:
        n = min(len(g), n_windows)
        idx = g.index.values[:n]
        e = xc_emb[idx]
        l = xc_logits[idx]
        label_code = g['label'].iloc[0]
        if label_code not in label_to_idx:
            continue
        ci = label_to_idx[label_code]
        # pad to n_windows
        emb_padded = np.zeros((n_windows, 1536), dtype=np.float32)
        log_padded = np.zeros((n_windows, n_classes), dtype=np.float32)
        lbl_padded = np.zeros((n_windows, n_classes), dtype=np.float32)
        emb_padded[:n] = e
        log_padded[:n] = l
        lbl_padded[:n, ci] = 1.0  # single primary label, valid windows only
        emb_files_list.append(emb_padded)
        log_files_list.append(log_padded)
        lbl_files_list.append(lbl_padded)
        file_ids.append(f"xc_{xc_id}")
        label_codes.append(label_code)

    xc_emb_files = np.stack(emb_files_list, axis=0)
    xc_log_files = np.stack(log_files_list, axis=0)
    xc_lbl_files = np.stack(lbl_files_list, axis=0)
    return xc_emb_files, xc_log_files, xc_lbl_files, file_ids, label_codes
```

### 結合 (Cell 14 [training entry] 内の reshape_to_files 直後)

```python
# 現行:
# emb_files, file_list = reshape_to_files(emb_full, meta_full)
# logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
# labels_files, _ = reshape_to_files(Y_FULL, meta_full)

# 追加:
if USE_XC_EXTERNAL:
    xc_emb_files, xc_log_files, xc_lbl_files, xc_file_ids, xc_labels = reshape_xc_to_files(
        xc_emb, xc_log, xc_meta, label_to_idx
    )
    # 結合
    emb_files = np.concatenate([emb_files, xc_emb_files], axis=0)
    logits_files = np.concatenate([logits_files, xc_log_files], axis=0)
    labels_files = np.concatenate([labels_files, xc_lbl_files], axis=0)
    file_list = file_list + xc_file_ids
    # site / hour: XC は unknown なので 0 / 12 (neutral)
    xc_sites = np.zeros(len(xc_file_ids), dtype=np.int32)
    xc_hours = np.full(len(xc_file_ids), 12, dtype=np.int32)
    site_ids_all = np.concatenate([site_ids_all, xc_sites])
    hours_all = np.concatenate([hours_all, xc_hours])
    # file_families: primary label の family を 1-hot
    xc_families = np.zeros((len(xc_file_ids), n_families), dtype=np.float32)
    for fi, code in enumerate(xc_labels):
        if code in PRIMARY_LABELS:
            ci = PRIMARY_LABELS.index(code)
            xc_families[fi, class_to_family[ci]] = 1.0
    file_families = np.concatenate([file_families, xc_families], axis=0)
```

## ⚠️ OOF 注意

- OOF cross-validation (MODE=train) は既存 competition file 単位で GroupKFold
- XC file は **training-only** にし、validation fold には含めない
- 実装: `file_groups` に XC 行は `"xc_external"` (単一大グループ) or `"xc_{xc_id}"` を割り当て、CV split の valid set から除外する後処理
- 代替 (簡単): XC 行は OOF 対象外として最終 full-data training のみに使用 → MODE=submit で full-data training 時に自動的に使われる

現 fork は MODE="submit" 運用、OOF スキップされている → 上記問題は現状起きない。後で MODE="train" を使う時に対処。

## 期待効果

- 外部データ追加は 2025 2nd place の主戦術、単独で +0.005〜0.02
- 11,563 XC files × 12 window = 138,756 窓の embedding 増 (既存 train_soundscapes の数倍)
- rare class のうち XC にある種 (Aves 159 種) に効果大

## 実装フロー

1. kernel 完了 → output dataset id 取得 (例: `yasunorim/xc-perch-v2-embed-birdclef-2026/output`)
2. 現 0.926 fork に dataset_sources 追加
3. 上記 reshape helper cell を fork に挿入
4. Cell 14 training entry に結合コード挿入
5. push → kernel run → LB 確認
6. 0.929 超え期待

## gate file 条件 (次の実 submit 時)

`kind=submit`, `self_best=0.926`, `claimed_lb=0.931`+ (期待値), `reason="XC external data 11563 files + ProtoSSM retrain expected +0.005"`
