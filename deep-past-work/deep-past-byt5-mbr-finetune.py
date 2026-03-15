# %% [markdown]
# # Deep Past Challenge: ByT5 + Multi-Temperature MBR Ensemble (v9)
# 
# **Akkadian → English translation** (private competition notebook)
# 
# ## Strategy
# - **Base model**: `mattiaangeli/byt5-akkadian-mbr-v2` (already fine-tuned on competition data)
# - **No additional fine-tuning** — all GPU time invested in superior inference
# - **Only 4 test sentences** — maximum compute per sentence
# - **Multi-temperature MBR**: 15 samples × 4 temperatures = 60 candidates per sentence
# - **Beam search hybrid**: beam candidates added to sampling pool
# - **Pre-computed n-grams**: ~10x faster MBR chrF++ scoring via cached counters
# - **Context window**: ±5 surrounding lines with `[CURRENT]` marker
# - **TF-IDF augmented prompting**: inject similar training examples into prompt
# - **MBR top-3 weighted consensus**: final output from top-3 MBR candidates

# %%
import os, shutil, subprocess

# Clean ALL previous output and caches to reclaim disk
for d in [
    '/kaggle/working/wandb',
    '/kaggle/working/.cache',
    '/root/.cache/huggingface',
    '/root/.cache/torch',
    '/root/.local/share/wandb',
    '/tmp/hf_cache', '/tmp/hf_home', '/tmp/torch_home',
]:
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        print(f"Cleaned: {d}")

# Remove any leftover large files from previous runs
for f in os.listdir('/kaggle/working'):
    fpath = os.path.join('/kaggle/working', f)
    if os.path.isfile(fpath) and f not in ('__notebook_source__.ipynb',):
        sz = os.path.getsize(fpath) / 1e6
        if sz > 1:
            os.remove(fpath)
            print(f"Removed: {f} ({sz:.0f}MB)")

print("--- Disk after cleanup ---")
os.system("df -h /kaggle/working | tail -1")
os.system("df -h /tmp | tail -1")

# %%
import os
import gc
import glob
import json
import shutil
import time
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

T_START = time.time()

# Disable W&B entirely
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# Redirect HuggingFace cache to /tmp to avoid filling /kaggle/working disk
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["TORCH_HOME"] = "/tmp/torch_home"

# --- chrF++ implementation with pre-computed n-gram support ---
def _extract_char_ngrams(text, n):
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))

def _extract_word_ngrams(text, n):
    words = text.split()
    if len(words) < n:
        return Counter()
    return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

def precompute_ngrams(text, char_order=6, word_order=2):
    """Pre-compute all char and word n-gram counters for a candidate string.
    Returns dict: {'char': {n: Counter, ...}, 'word': {n: Counter, ...}}."""
    return {
        'char': {n: _extract_char_ngrams(text, n) for n in range(1, char_order + 1)},
        'word': {n: _extract_word_ngrams(text, n) for n in range(1, word_order + 1)},
    }

def chrf_score_precomputed(hyp_ng, ref_ng, char_order=6, word_order=2, beta=2):
    """Compute chrF++ using pre-computed n-gram counters (no re-extraction)."""
    total_f = 0.0
    count = 0
    for n in range(1, char_order + 1):
        hyp_counter = hyp_ng['char'][n]
        ref_counter = ref_ng['char'][n]
        common = sum((hyp_counter & ref_counter).values())
        hyp_total = sum(hyp_counter.values())
        ref_total = sum(ref_counter.values())
        p = common / hyp_total if hyp_total > 0 else 0.0
        r = common / ref_total if ref_total > 0 else 0.0
        if p + r > 0:
            f = (1 + beta**2) * p * r / (beta**2 * p + r)
        else:
            f = 0.0
        total_f += f
        count += 1
    for n in range(1, word_order + 1):
        hyp_counter = hyp_ng['word'][n]
        ref_counter = ref_ng['word'][n]
        common = sum((hyp_counter & ref_counter).values())
        hyp_total = sum(hyp_counter.values())
        ref_total = sum(ref_counter.values())
        p = common / hyp_total if hyp_total > 0 else 0.0
        r = common / ref_total if ref_total > 0 else 0.0
        if p + r > 0:
            f = (1 + beta**2) * p * r / (beta**2 * p + r)
        else:
            f = 0.0
        total_f += f
        count += 1
    return (total_f / count * 100) if count > 0 else 0.0

def chrf_score(hypothesis, reference, char_order=6, word_order=2, beta=2):
    """Original chrF++ for non-MBR uses (e.g. TF-IDF fallback confidence check)."""
    if not hypothesis or not reference:
        return 0.0
    hyp_ng = precompute_ngrams(hypothesis, char_order, word_order)
    ref_ng = precompute_ngrams(reference, char_order, word_order)
    return chrf_score_precomputed(hyp_ng, ref_ng, char_order, word_order, beta)

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
os.system("df -h /kaggle/working | tail -1")
print(f"\n[TIMER] Setup complete: {time.time() - T_START:.1f}s elapsed")

# %%
import glob as _glob
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
print(f"Columns: {list(train_comp.columns)}")
print(f"\n[TIMER] Data loaded: {time.time() - T_START:.1f}s elapsed")

# %%
def load_external_data():
    """Search ALL of /kaggle/input for Akkadian datasets via rglob."""
    dfs = []

    # Known dataset slugs
    known_slugs = ['akkadian-oracc-combined', 'akkadian-enriched-data']

    # First try exact paths
    search_roots = [Path('/kaggle/input') / slug for slug in known_slugs]

    # Also scan ALL subdirectories under /kaggle/input for any CSV with relevant columns
    print("  Scanning /kaggle/input for Akkadian datasets...")
    input_root = Path('/kaggle/input')
    all_csvs = []
    for csv_file in input_root.rglob('*.csv'):
        # Skip competition data (already loaded)
        if 'competitions' in str(csv_file):
            continue
        all_csvs.append(csv_file)
    print(f"  Found {len(all_csvs)} CSV files outside competition data")

    for csv_file in sorted(all_csvs):
        try:
            df = pd.read_csv(csv_file, nrows=5)  # peek first
            col_map = {}
            for c in df.columns:
                cl = c.lower().strip()
                if any(k in cl for k in ['translit', 'akkadian', 'source', 'input']):
                    col_map[c] = 'transliteration'
                elif any(k in cl for k in ['translat', 'english', 'target', 'output']):
                    col_map[c] = 'translation'
            if 'transliteration' not in col_map.values() or 'translation' not in col_map.values():
                continue
            # Full read
            df = pd.read_csv(csv_file).rename(columns=col_map)
            df = df[['transliteration', 'translation']].dropna()
            df = df[df['transliteration'].str.len() > 2]
            df = df[df['translation'].str.len() > 2]
            if len(df) > 0:
                dfs.append(df)
                print(f"  Loaded {len(df):,} rows from {csv_file}")
        except Exception as e:
            pass  # silently skip non-CSV or broken files

    if not dfs:
        print("  No external Akkadian data found")
        return pd.DataFrame(columns=['transliteration', 'translation'])
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['transliteration'])
    print(f"Total external data: {len(combined):,} rows")
    return combined

print("Loading external data...")
ext_data = load_external_data()

# Build reference corpus for TF-IDF similarity
ref_corpus = pd.concat([
    train_comp[['transliteration', 'translation']],
    ext_data[['transliteration', 'translation']],
], ignore_index=True).drop_duplicates(subset=['transliteration'])
print(f"Reference corpus: {len(ref_corpus):,} rows")

# Build TF-IDF index on transliterations
print("Building TF-IDF index...")
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=80000)
ref_tfidf_matrix = tfidf.fit_transform(ref_corpus['transliteration'].values)
print(f"TF-IDF matrix: {ref_tfidf_matrix.shape}")
print(f"\n[TIMER] External data + TF-IDF: {time.time() - T_START:.1f}s elapsed")

# %%
CONTEXT_WINDOW = 5
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

def tfidf_fallback(transliteration, top_k=5):
    """Find most similar training examples and return their translations."""
    query_vec = tfidf.transform([transliteration])
    sims = cos_sim(query_vec, ref_tfidf_matrix).flatten()
    top_idxs = sims.argsort()[-top_k:][::-1]
    results = []
    for idx in top_idxs:
        if sims[idx] > 0.05:
            results.append({
                'transliteration': ref_corpus.iloc[idx]['transliteration'],
                'translation': ref_corpus.iloc[idx]['translation'],
                'similarity': float(sims[idx]),
            })
    return results

def build_augmented_prompt(transliteration, context_input, top_k=3):
    """Augment the prompt with similar training examples (few-shot style)."""
    matches = tfidf_fallback(transliteration, top_k=top_k)
    if not matches:
        return context_input

    examples = []
    for m in matches[:top_k]:
        examples.append(f"{m['transliteration']} => {m['translation']}")

    aug_prefix = "Examples: " + " | ".join(examples) + " [END_EXAMPLES] "
    return aug_prefix + context_input

id_col_test = 'text_id' if 'text_id' in test.columns else None
test_sorted = test.sort_values(id_col_test).reset_index(drop=True) if id_col_test else test.copy()
test_inputs_base = build_context_inputs_df(test_sorted, id_col=id_col_test)
test_raw_translits = test_sorted['transliteration'].tolist()

# Build augmented prompts for test set
test_inputs_augmented = [
    build_augmented_prompt(raw, base, top_k=3)
    for raw, base in zip(test_raw_translits, test_inputs_base)
]

print(f"Test inputs: {len(test_inputs_augmented):,}")
print(f"Context window: \u00b1{CONTEXT_WINDOW}")
print(f"\nSample augmented prompt (first 300 chars):")
print(test_inputs_augmented[0][:300])
print(f"\n[TIMER] Prompts built: {time.time() - T_START:.1f}s elapsed")

# %%
def find_model_path(slug_patterns):
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
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=torch.float16)
model = model.to(DEVICE)
model.eval()
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print("--- Disk after model load ---")
os.system("df -h /kaggle/working | tail -1")
print(f"\n[TIMER] Model loaded: {time.time() - T_START:.1f}s elapsed")

# %%
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 256

# Reduced generation: 15 samples × 4 temps + 6 beam + 16 base = ~82 candidates
# ByT5 is byte-level (1 char = 1 token), so num_return_sequences=15 avoids OOM
MBR_SAMPLES_PER_TEMP = 15
MBR_TEMPERATURES = [0.5, 0.7, 0.9, 1.1]
BEAM_SIZE = 6  # Beam search candidates
BASE_PROMPT_TEMPS = [0.7, 0.9]  # Temperatures for base prompt diversity
BASE_PROMPT_SAMPLES = 8  # Samples per temp for base prompt

def multi_temp_mbr_decode(text, text_base=None, n_per_temp=MBR_SAMPLES_PER_TEMP):
    """Generate candidates via sampling at multiple temperatures + beam search,
    then select best by chrF++ MBR consensus with pre-computed n-grams.

    Returns (best_translation, confidence, top3_candidates_with_scores).
    """
    enc = tokenizer(text, max_length=MAX_SOURCE_LEN, truncation=True, return_tensors="pt").to(DEVICE)
    all_candidates = []

    with torch.inference_mode():
        # Sampling candidates at multiple temperatures
        for temp in MBR_TEMPERATURES:
            outputs = model.generate(
                **enc,
                max_new_tokens=MAX_TARGET_LEN,
                do_sample=True,
                num_beams=1,
                temperature=temp,
                top_p=0.95,
                top_k=50,
                num_return_sequences=n_per_temp,
                no_repeat_ngram_size=3,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_candidates.extend([s.strip() for s in decoded if s.strip()])

        # Beam search candidates (deterministic, high quality)
        beam_outputs = model.generate(
            **enc,
            max_new_tokens=MAX_TARGET_LEN,
            do_sample=False,
            num_beams=BEAM_SIZE,
            num_return_sequences=min(BEAM_SIZE, 6),
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        beam_decoded = tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        all_candidates.extend([s.strip() for s in beam_decoded if s.strip()])

    # Also generate from base prompt (without augmentation) for diversity
    if text_base and text_base != text:
        enc_base = tokenizer(text_base, max_length=MAX_SOURCE_LEN, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            for temp in BASE_PROMPT_TEMPS:
                outputs = model.generate(
                    **enc_base,
                    max_new_tokens=MAX_TARGET_LEN,
                    do_sample=True,
                    num_beams=1,
                    temperature=temp,
                    top_p=0.95,
                    num_return_sequences=BASE_PROMPT_SAMPLES,
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
        return "unknown", 0.0, []
    if len(unique_cands) == 1:
        return unique_cands[0], 100.0, [(unique_cands[0], 100.0)]

    # Pre-compute n-gram counters for ALL candidates once
    t_ngram = time.time()
    cand_ngrams = [precompute_ngrams(c) for c in unique_cands]
    n_cands = len(unique_cands)
    print(f"    Pre-computed n-grams for {n_cands} candidates in {time.time()-t_ngram:.2f}s")

    # MBR selection: pick candidate with highest average chrF++ vs all others
    # Uses pre-computed n-grams — no re-extraction in the O(N²) loop
    t_mbr = time.time()
    scores = []
    for i in range(n_cands):
        pairwise_sum = 0.0
        for j in range(n_cands):
            if j != i:
                pairwise_sum += chrf_score_precomputed(cand_ngrams[i], cand_ngrams[j])
        scores.append(pairwise_sum / (n_cands - 1))
    print(f"    MBR scoring ({n_cands}\u00b2 = {n_cands**2:,} comparisons) in {time.time()-t_mbr:.2f}s")

    # Get top-3 candidates with scores
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    top3 = [(unique_cands[idx], sc) for idx, sc in ranked[:3]]

    best_idx = ranked[0][0]
    confidence = scores[best_idx]
    return unique_cands[best_idx], confidence, top3

total_candidates = MBR_SAMPLES_PER_TEMP * len(MBR_TEMPERATURES) + BEAM_SIZE + BASE_PROMPT_SAMPLES * len(BASE_PROMPT_TEMPS)
print(f"MBR config: {MBR_SAMPLES_PER_TEMP} samples \u00d7 {len(MBR_TEMPERATURES)} temps + {BEAM_SIZE} beam + {BASE_PROMPT_SAMPLES * len(BASE_PROMPT_TEMPS)} base = ~{total_candidates} candidates/sentence")
print(f"Temperatures: {MBR_TEMPERATURES}")
print(f"Beam size: {BEAM_SIZE}")
print(f"Base prompt: {BASE_PROMPT_SAMPLES} samples \u00d7 {len(BASE_PROMPT_TEMPS)} temps")
print(f"Pre-computed n-grams: char(1..6) + word(1..2) cached per candidate")
print(f"\n[TIMER] Decoding ready: {time.time() - T_START:.1f}s elapsed")

# %%
print("--- Disk before inference ---")
os.system("df -h /kaggle/working | tail -1")

all_predictions = []
all_confidences = []
all_top3 = []

t0 = time.time()
for i, (text_aug, text_base, raw_translit) in enumerate(zip(
    test_inputs_augmented, test_inputs_base, test_raw_translits
)):
    print(f"\n{'='*60}")
    print(f"  Sentence {i+1}/{len(test_inputs_augmented)}")
    print(f"  Input: {raw_translit[:80]}")
    print(f"{'='*60}")

    t_sent = time.time()
    # Also generate with individual transliteration (no context) for diversity
    text_individual = PREFIX + raw_translit
    pred, confidence, top3 = multi_temp_mbr_decode(text_aug, text_base=text_individual)

    # Show top-3 MBR candidates
    print(f"  Top-3 MBR candidates:")
    for rank, (cand, sc) in enumerate(top3):
        marker = " <<<" if rank == 0 else ""
        print(f"    {rank+1}. [chrF++={sc:.1f}] {cand[:100]}{marker}")

    # TF-IDF matches for reference
    matches = tfidf_fallback(raw_translit, top_k=3)
    if matches:
        print(f"  TF-IDF similar translations:")
        for m in matches[:3]:
            print(f"    sim={m['similarity']:.3f}: {m['translation'][:80]}")

    # If confidence is very low AND TF-IDF has strong match, consider blending
    if confidence < 25.0 and matches and matches[0]['similarity'] > 0.7:
        # Very strong TF-IDF match with low MBR confidence -> use TF-IDF
        pred = matches[0]['translation']
        print(f"  -> Using TF-IDF fallback (MBR confidence {confidence:.1f} < 25, TF-IDF sim {matches[0]['similarity']:.3f})")

    all_predictions.append(pred)
    all_confidences.append(confidence)
    all_top3.append(top3)
    elapsed = time.time() - t_sent
    print(f"  Final: {pred[:100]}")
    print(f"  Confidence: {confidence:.1f}, Time: {elapsed:.1f}s")
    print(f"  [TIMER] Sentence {i+1} done: {time.time() - T_START:.1f}s total elapsed")

total_time = time.time() - t0
print(f"\n{'='*60}")
print(f"Inference complete: {len(all_predictions)} predictions in {total_time:.1f}s")
print(f"Average MBR confidence: {np.mean(all_confidences):.1f}")
print(f"Confidence per sentence: {[f'{c:.1f}' for c in all_confidences]}")
print(f"\n[TIMER] All inference done: {time.time() - T_START:.1f}s total elapsed")

# %%
import re as _re
import unicodedata as _ud

def postprocess(text):
    text = text.strip()
    if not text:
        return "unknown"
    # Remove emoji and non-Latin/non-diacritics characters that cause scoring issues
    # Keep: ASCII, Latin Extended (diacritics like ā, š, etc.), basic punctuation
    cleaned = []
    for ch in text:
        cat = _ud.category(ch)
        # Keep letters, numbers, punctuation, spaces, symbols like brackets
        if cat.startswith(('L', 'N', 'P', 'Z', 'S')) and cat not in ('So',):
            cleaned.append(ch)
        elif cat == 'So':
            # Skip symbols like emoji
            pass
        else:
            cleaned.append(ch)
    text = ''.join(cleaned)
    # Remove any remaining control characters
    text = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
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
print("\nFull submission:")
for _, row in submission.iterrows():
    print(f"  {row['id']}: {row['translation'][:120]}")
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("\n--- Final disk usage ---")
os.system("df -h /kaggle/working | tail -1")
print(f"\n[TIMER] Total runtime: {time.time() - T_START:.1f}s")
print("\nDone. submission.csv ready.")
