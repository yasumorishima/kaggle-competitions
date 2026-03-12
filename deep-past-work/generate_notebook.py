"""Deep Past Challenge - ByT5 Multi-Temperature MBR Notebook Generator (v7)

Strategy:
  - Base: mattiaangeli/byt5-akkadian-mbr-v2 (already fine-tuned on competition data)
  - NO additional fine-tuning (avoids disk issues, base model is already good)
  - Multi-temperature MBR: generate candidates at T=0.6, 0.8, 1.0 → unified MBR selection
  - MBR samples: 10 per temperature = 30 total candidates per sentence
  - Context window: ±3 surrounding lines with [CURRENT] marker
  - TF-IDF similarity fallback for low-confidence translations

v7 disk fix vs v6:
  - HF_HOME/TRANSFORMERS_CACHE/TORCH_HOME → /tmp (not /kaggle/working)
  - float16 inference (halves model memory footprint)
  - MBR samples 20→10 per temp (60→30 candidates)
  - Validation 100→20 sentences
  - Periodic cache cleanup during inference
"""

import json

cells = []
cell_counter = 0


def add_md(source):
    global cell_counter
    cell_counter += 1
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({
        "cell_type": "markdown",
        "id": f"cell-{cell_counter:03d}",
        "metadata": {},
        "source": src,
    })


def add_code(source):
    global cell_counter
    cell_counter += 1
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({
        "cell_type": "code",
        "id": f"cell-{cell_counter:03d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    })


# =============================================================================
# Cell: Title
# =============================================================================
add_md("""# Deep Past Challenge: ByT5 + Multi-Temperature MBR Ensemble

**Akkadian → English translation** (private competition notebook)

## Strategy
- **Base model**: `mattiaangeli/byt5-akkadian-mbr-v2` (already fine-tuned on competition data, 28 votes)
- **No additional fine-tuning** — all GPU time invested in superior inference
- **Multi-temperature MBR**: candidates at T=0.6, 0.8, 1.0 → unified chrF++ MBR selection
- **10 samples × 3 temperatures = 30 candidates per sentence**
- **Context window**: ±3 surrounding lines with `[CURRENT]` marker
- **TF-IDF similarity fallback**: for low-confidence translations, reference training data
- **Evaluation**: BLEU + chrF++ (competition metric)""")


# =============================================================================
# Cell: Setup & Imports
# =============================================================================
add_code("""import os
import gc
import glob
import json
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

# Disable W&B entirely
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# Redirect HuggingFace cache to /tmp to avoid filling /kaggle/working disk
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["TORCH_HOME"] = "/tmp/torch_home"

# --- chrF++ implementation (no sacrebleu dependency) ---
def _extract_char_ngrams(text, n):
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))

def _extract_word_ngrams(text, n):
    words = text.split()
    if len(words) < n:
        return Counter()
    return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

def chrf_score(hypothesis, reference, char_order=6, word_order=2, beta=2):
    if not hypothesis or not reference:
        return 0.0
    total_f = 0.0
    count = 0
    for n in range(1, char_order + 1):
        hyp_ngrams = _extract_char_ngrams(hypothesis, n)
        ref_ngrams = _extract_char_ngrams(reference, n)
        common = sum((hyp_ngrams & ref_ngrams).values())
        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())
        p = common / hyp_total if hyp_total > 0 else 0.0
        r = common / ref_total if ref_total > 0 else 0.0
        if p + r > 0:
            f = (1 + beta**2) * p * r / (beta**2 * p + r)
        else:
            f = 0.0
        total_f += f
        count += 1
    for n in range(1, word_order + 1):
        hyp_ngrams = _extract_word_ngrams(hypothesis, n)
        ref_ngrams = _extract_word_ngrams(reference, n)
        common = sum((hyp_ngrams & ref_ngrams).values())
        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())
        p = common / hyp_total if hyp_total > 0 else 0.0
        r = common / ref_total if ref_total > 0 else 0.0
        if p + r > 0:
            f = (1 + beta**2) * p * r / (beta**2 * p + r)
        else:
            f = 0.0
        total_f += f
        count += 1
    return (total_f / count * 100) if count > 0 else 0.0

def chrf_corpus_score(hypotheses, references):
    scores = [chrf_score(h, r) for h, r in zip(hypotheses, references)]
    return sum(scores) / len(scores) if scores else 0.0

class ChrFScorer:
    def __init__(self, word_order=2):
        self.word_order = word_order
    def corpus_score(self, hypotheses, references_list):
        refs = [r[0] if isinstance(r, list) else r for r in references_list]
        score = chrf_corpus_score(hypotheses, refs)
        return type('Score', (), {'score': score})()

class BLEUScorer:
    def corpus_score(self, hypotheses, references_list):
        refs = [[r[0].split() if isinstance(r, list) else r.split()] for r in references_list]
        hyps = [h.split() for h in hypotheses]
        smooth = SmoothingFunction().method1
        score = corpus_bleu(refs, hyps, smoothing_function=smooth) * 100
        return type('Score', (), {'score': score})()
# --- end chrF++ implementation ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
          if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
          else f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("--- Disk at start ---")
os.system("df -h /kaggle/working | tail -1")""")


# =============================================================================
# Cell: Load competition data
# =============================================================================
add_code("""import glob as _glob
_slug = 'deep-past-initiative-machine-translation'
_matches = _glob.glob(f'/kaggle/input/**/{_slug}', recursive=True)
DATA_DIR = Path(_matches[0]) if _matches else Path(f'/kaggle/input/{_slug}')
if not DATA_DIR.exists():
    _alt = _glob.glob(f'/kaggle/input/competitions/{_slug}', recursive=False)
    if _alt:
        DATA_DIR = Path(_alt[0])
print(f"DATA_DIR: {DATA_DIR}")

train_comp = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

train_comp['transliteration'] = train_comp['transliteration'].fillna('')
train_comp['translation'] = train_comp['translation'].fillna('')
test['transliteration'] = test['transliteration'].fillna('')

print(f"Competition train: {train_comp.shape}")
print(f"Test: {test.shape}")
print(f"Columns: {list(train_comp.columns)}")""")


# =============================================================================
# Cell: Load external data for TF-IDF reference
# =============================================================================
add_code("""def load_external_data():
    dfs = []
    oracc_paths = [
        '/kaggle/input/akkadian-oracc-combined',
        '/kaggle/input/akkadian-enriched-data',
    ]
    for base in oracc_paths:
        base = Path(base)
        if not base.exists():
            print(f"  Not found: {base}")
            continue
        for csv_file in sorted(base.glob('**/*.csv')):
            try:
                df = pd.read_csv(csv_file)
                col_map = {}
                for c in df.columns:
                    cl = c.lower().strip()
                    if any(k in cl for k in ['translit', 'akkadian', 'source', 'input']):
                        col_map[c] = 'transliteration'
                    elif any(k in cl for k in ['translat', 'english', 'target', 'output']):
                        col_map[c] = 'translation'
                if col_map:
                    df = df.rename(columns=col_map)
                if 'transliteration' in df.columns and 'translation' in df.columns:
                    df = df[['transliteration', 'translation']].dropna()
                    df = df[df['transliteration'].str.len() > 2]
                    df = df[df['translation'].str.len() > 2]
                    dfs.append(df)
                    print(f"  Loaded {len(df):,} rows from {csv_file.name}")
            except Exception as e:
                print(f"  Error {csv_file.name}: {e}")
    if not dfs:
        return pd.DataFrame(columns=['transliteration', 'translation'])
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['transliteration'])
    print(f"Total external data: {len(combined):,} rows")
    return combined

print("Loading external data...")
ext_data = load_external_data()

# Build reference corpus for TF-IDF similarity fallback
ref_corpus = pd.concat([
    train_comp[['transliteration', 'translation']],
    ext_data[['transliteration', 'translation']],
], ignore_index=True).drop_duplicates(subset=['transliteration'])
print(f"Reference corpus: {len(ref_corpus):,} rows")

# Build TF-IDF index on transliterations for similarity lookup
print("Building TF-IDF index...")
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=50000)
ref_tfidf_matrix = tfidf.fit_transform(ref_corpus['transliteration'].values)
print(f"TF-IDF matrix: {ref_tfidf_matrix.shape}")""")


# =============================================================================
# Cell: Context-aware input construction
# =============================================================================
add_code("""CONTEXT_WINDOW = 3
PREFIX = "translate Akkadian to English: "

def build_context_inputs_df(df, context_col='transliteration', id_col=None):
    inputs = []
    df = df.copy().reset_index(drop=True)

    if id_col and id_col in df.columns:
        for gid, group in df.groupby(id_col, sort=False):
            idxs = group.index.tolist()
            lines = group[context_col].tolist()
            for local_i, orig_i in enumerate(idxs):
                start = max(0, local_i - CONTEXT_WINDOW)
                end = min(len(lines), local_i + CONTEXT_WINDOW + 1)
                ctx = list(lines[start:end])
                cur = local_i - start
                ctx[cur] = f"[CURRENT] {ctx[cur]}"
                inputs.append((orig_i, PREFIX + " [SEP] ".join(ctx)))
        inputs.sort(key=lambda x: x[0])
        return [x[1] for x in inputs]
    else:
        lines = df[context_col].tolist()
        for i in range(len(lines)):
            start = max(0, i - CONTEXT_WINDOW)
            end = min(len(lines), i + CONTEXT_WINDOW + 1)
            ctx = list(lines[start:end])
            cur = i - start
            ctx[cur] = f"[CURRENT] {ctx[cur]}"
            inputs.append(PREFIX + " [SEP] ".join(ctx))
        return inputs

id_col_test = 'text_id' if 'text_id' in test.columns else None
test_sorted = test.sort_values(id_col_test).reset_index(drop=True) if id_col_test else test.copy()
test_inputs = build_context_inputs_df(test_sorted, id_col=id_col_test)
test_raw_translits = test_sorted['transliteration'].tolist()

# Also build val set for evaluation
id_col_train = 'text_id' if 'text_id' in train_comp.columns else ('oare_id' if 'oare_id' in train_comp.columns else None)
train_sorted = train_comp.sort_values(id_col_train).reset_index(drop=True) if id_col_train else train_comp.copy()
all_comp_inputs = build_context_inputs_df(train_sorted, id_col=id_col_train)
all_comp_targets = train_sorted['translation'].tolist()
VAL_SIZE = min(200, len(all_comp_inputs) // 5)

print(f"Test inputs: {len(test_inputs):,}")
print(f"Val size: {VAL_SIZE}")
print(f"Context window: ±{CONTEXT_WINDOW}")""")


# =============================================================================
# Cell: Load pre-trained model
# =============================================================================
add_code("""def find_model_path(slug_patterns):
    for pattern in slug_patterns:
        for p in glob.glob(f'/kaggle/input/models/**', recursive=True):
            if os.path.isdir(p) and any(
                s in p for s in slug_patterns
            ) and any(
                os.path.exists(os.path.join(p, f))
                for f in ['config.json', 'pytorch_model.bin', 'model.safetensors']
            ):
                return p
    return "mattiaangeli/byt5-akkadian-mbr-v2"

model_path = find_model_path(['byt5-akkadian-mbr-v2', 'byt5-akkadian-mbr', 'akkadian-mbr'])
print(f"Using model: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)
model = model.to(DEVICE)
model.eval()
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print("--- Disk after model load ---")
os.system("df -h /kaggle/working | tail -1")""")


# =============================================================================
# Cell: Multi-Temperature MBR Decoding
# =============================================================================
add_code("""MAX_SOURCE_LEN = 384
MAX_TARGET_LEN = 192

# Multi-temperature MBR: generate candidates at multiple temperatures,
# then select the best by chrF++ consensus across ALL candidates.
# This produces more diverse yet high-quality translations.
MBR_SAMPLES_PER_TEMP = 10
MBR_TEMPERATURES = [0.6, 0.8, 1.0]
chrf_metric = ChrFScorer(word_order=2)

def multi_temp_mbr_decode(text, n_per_temp=MBR_SAMPLES_PER_TEMP):
    enc = tokenizer(text, max_length=MAX_SOURCE_LEN, truncation=True, return_tensors="pt").to(DEVICE)
    all_candidates = []

    with torch.no_grad():
        for temp in MBR_TEMPERATURES:
            outputs = model.generate(
                **enc,
                max_new_tokens=MAX_TARGET_LEN,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                num_return_sequences=n_per_temp,
                no_repeat_ngram_size=3,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_candidates.extend([s.strip() for s in decoded if s.strip()])

    # Deduplicate while preserving order
    seen = set()
    unique_cands = []
    for c in all_candidates:
        if c not in seen:
            seen.add(c)
            unique_cands.append(c)

    if not unique_cands:
        return "unknown", 0.0
    if len(unique_cands) == 1:
        return unique_cands[0], 100.0

    # MBR selection: pick candidate with highest average chrF++ vs all others
    scores = []
    for i, hyp in enumerate(unique_cands):
        refs = [c for j, c in enumerate(unique_cands) if j != i]
        pairwise = [chrf_score(hyp, r) for r in refs]
        scores.append(np.mean(pairwise))

    best_idx = int(np.argmax(scores))
    confidence = scores[best_idx]
    return unique_cands[best_idx], confidence

print(f"MBR config: {MBR_SAMPLES_PER_TEMP} samples × {len(MBR_TEMPERATURES)} temps = {MBR_SAMPLES_PER_TEMP * len(MBR_TEMPERATURES)} candidates/sentence")
print(f"Temperatures: {MBR_TEMPERATURES}")""")


# =============================================================================
# Cell: TF-IDF Similarity Fallback
# =============================================================================
add_code("""def tfidf_fallback(transliteration, top_k=3):
    \"\"\"Find most similar training examples and return their translations.\"\"\"
    query_vec = tfidf.transform([transliteration])
    sims = cos_sim(query_vec, ref_tfidf_matrix).flatten()
    top_idxs = sims.argsort()[-top_k:][::-1]
    results = []
    for idx in top_idxs:
        if sims[idx] > 0.1:
            results.append({
                'transliteration': ref_corpus.iloc[idx]['transliteration'],
                'translation': ref_corpus.iloc[idx]['translation'],
                'similarity': float(sims[idx]),
            })
    return results

# Test the fallback
if len(test_raw_translits) > 0:
    sample = test_raw_translits[0]
    matches = tfidf_fallback(sample)
    print(f"TF-IDF fallback test for: '{sample[:60]}'")
    for m in matches:
        print(f"  sim={m['similarity']:.3f}: {m['transliteration'][:40]} → {m['translation'][:40]}")""")


# =============================================================================
# Cell: Run inference on test set
# =============================================================================
add_code("""print("--- Disk before inference ---")
os.system("df -h /kaggle/working | tail -1")

CONFIDENCE_THRESHOLD = 30.0  # Below this, blend with TF-IDF fallback

all_predictions = []
all_confidences = []
n_fallbacks = 0

for i, (text, raw_translit) in enumerate(zip(test_inputs, test_raw_translits)):
    pred, confidence = multi_temp_mbr_decode(text)

    # If MBR confidence is low, blend with TF-IDF matches
    if confidence < CONFIDENCE_THRESHOLD:
        matches = tfidf_fallback(raw_translit, top_k=1)
        if matches and matches[0]['similarity'] > 0.5:
            # High TF-IDF similarity: use the reference translation instead
            pred = matches[0]['translation']
            n_fallbacks += 1

    all_predictions.append(pred)
    all_confidences.append(confidence)

    if (i + 1) % 20 == 0 or i == 0:
        print(f"  {i+1}/{len(test_inputs)} done, avg confidence: {np.mean(all_confidences):.1f}, fallbacks: {n_fallbacks}")
        # Periodic disk cleanup
        for cache_dir in ['/tmp/hf_cache', '/tmp/hf_home', '/tmp/torch_home', os.path.expanduser('~/.cache/huggingface')]:
            if os.path.isdir(cache_dir):
                sz = sum(f.stat().st_size for f in Path(cache_dir).rglob('*') if f.is_file()) / 1e6
                if sz > 100:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    print(f"    Cleaned {cache_dir} ({sz:.0f}MB)")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f"\\nInference complete: {len(all_predictions)} predictions")
print(f"Average MBR confidence: {np.mean(all_confidences):.1f}")
print(f"TF-IDF fallbacks used: {n_fallbacks}/{len(all_predictions)}")
print(f"Confidence distribution: min={np.min(all_confidences):.1f}, median={np.median(all_confidences):.1f}, max={np.max(all_confidences):.1f}")""")


# =============================================================================
# Cell: Validate on val set
# =============================================================================
add_code("""print("Evaluating on val set (last {VAL_SIZE} competition samples)...")
val_inputs = all_comp_inputs[-VAL_SIZE:]
val_targets = all_comp_targets[-VAL_SIZE:]

val_preds = []
for text in val_inputs[:20]:  # Evaluate first 20 for speed + disk safety
    pred, _ = multi_temp_mbr_decode(text)
    val_preds.append(pred)

val_refs = val_targets[:len(val_preds)]
bleu_metric = BLEUScorer()
chrf_eval = ChrFScorer(word_order=2)
val_bleu = bleu_metric.corpus_score(val_preds, [val_refs]).score
val_chrf = chrf_eval.corpus_score(val_preds, [val_refs]).score

print(f"Val BLEU:   {val_bleu:.4f}")
print(f"Val chrF++: {val_chrf:.4f}")

print("\\nSample predictions:")
for i in range(min(5, len(val_preds))):
    print(f"  Ref:  {val_refs[i][:80]}")
    print(f"  Pred: {val_preds[i][:80]}")
    print()""")


# =============================================================================
# Cell: Submission
# =============================================================================
add_code("""def postprocess(text):
    text = text.strip()
    return ' '.join(text.split()) if text else "unknown"

all_predictions = [postprocess(p) for p in all_predictions]

# Re-align to original test order if we sorted by id_col
if id_col_test and 'id' in test_sorted.columns and 'id' in test.columns:
    pred_series = pd.Series(all_predictions, index=test_sorted['id'].values)
    all_predictions_aligned = pred_series.reindex(test['id'].values).fillna("unknown").tolist()
else:
    all_predictions_aligned = all_predictions

submission = pd.DataFrame({
    'id': test['id'],
    'translation': all_predictions_aligned[:len(test)],
})

print(f"Submission shape: {submission.shape}")
print(submission.head(10).to_string())
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("\\n--- Final disk usage ---")
os.system("df -h /kaggle/working | tail -1")
print("\\nDone. submission.csv ready.")""")


# =============================================================================
# Build the notebook file
# =============================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = "deep-past-byt5-mbr-finetune.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Generated: {out_path} ({len(cells)} cells)")
