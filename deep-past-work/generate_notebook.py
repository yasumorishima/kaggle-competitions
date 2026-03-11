"""Deep Past Challenge - ByT5 MBR Fine-tuning Notebook Generator

Strategy:
  - Base: mattiaangeli/byt5-akkadian-mbr-v2 (already fine-tuned on competition data)
  - Additional data: ORACC parallel corpus + enriched data
  - Decoding: MBR (Minimum Bayes Risk) with chrF++ as utility function
  - Context: prepend surrounding lines (window=2)
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
add_md("""# Deep Past Challenge: ByT5 + MBR Decoding

**Akkadian → English translation** (private competition notebook)

## Strategy
- **Base model**: `mattiaangeli/byt5-akkadian-mbr-v2` (already fine-tuned on competition data, 28 votes)
- **Extra training**: ORACC parallel corpus + enriched data (~2x more data)
- **Decoding**: MBR (Minimum Bayes Risk) with chrF++ — better than beam search for MT
- **Context window**: ±2 surrounding lines with `[CURRENT]` marker for document coherence
- **Evaluation**: BLEU + chrF++ (competition metric)""")


# =============================================================================
# Cell: Setup & Imports
# =============================================================================
add_code("""import os
import gc
import glob
import json
import numpy as np
import pandas as pd
import torch
import wandb
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

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

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project="kaggle-deep-past", name="byt5-mbr-v1", config={
        "base_model": "mattiaangeli/byt5-akkadian-mbr-v2",
        "extra_data": ["oracc-combined", "enriched-v4"],
        "epochs": 3,
        "batch_size": 8,
        "max_source_len": 512,
        "max_target_len": 256,
        "context_window": 2,
        "mbr_samples": 16,
        "mbr_utility": "chrf",
    })
else:
    os.environ["WANDB_DISABLED"] = "true"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")""")


# =============================================================================
# Cell: Load competition data
# =============================================================================
add_code("""# Auto-detect competition data path
import glob as _glob
_slug = 'deep-past-initiative-machine-translation'
_matches = _glob.glob(f'/kaggle/input/**/{_slug}', recursive=True)
DATA_DIR = Path(_matches[0]) if _matches else Path(f'/kaggle/input/{_slug}')
print(f"DATA_DIR: {DATA_DIR}")

# Also try competitions subdir (Kaggle sometimes mounts here)
if not DATA_DIR.exists():
    _alt = _glob.glob(f'/kaggle/input/competitions/{_slug}', recursive=False)
    if _alt:
        DATA_DIR = Path(_alt[0])
        print(f"Using alt path: {DATA_DIR}")

print("Files in DATA_DIR:")
for f in sorted(DATA_DIR.glob('*')):
    print(f"  {f.name}")

train_comp = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

train_comp['transliteration'] = train_comp['transliteration'].fillna('')
train_comp['translation'] = train_comp['translation'].fillna('')
test['transliteration'] = test['transliteration'].fillna('')

print(f"\\nCompetition train: {train_comp.shape}")
print(f"Test: {test.shape}")
print(f"Columns: {list(train_comp.columns)}")
print(train_comp.head(3).to_string())""")


# =============================================================================
# Cell: Load external ORACC data
# =============================================================================
add_code("""def load_external_data():
    \"\"\"Load ORACC parallel corpus and enriched data.\"\"\"
    dfs = []

    oracc_paths = [
        '/kaggle/input/akkadian-oracc-combined',
        '/kaggle/input/akkadian-enriched-data',
        '/kaggle/input/oracc-akkadian-english-parallel-corpus',
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
                else:
                    print(f"  Skipped {csv_file.name} (cols: {list(df.columns)[:5]})")
            except Exception as e:
                print(f"  Error {csv_file.name}: {e}")

    if not dfs:
        print("No external data found — training on competition data only.")
        return pd.DataFrame(columns=['transliteration', 'translation'])

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['transliteration'])
    print(f"Total external data: {len(combined):,} rows")
    return combined

print("Loading external data...")
ext_data = load_external_data()

# Combine: competition train + external
train_all = pd.concat([
    train_comp[['transliteration', 'translation']],
    ext_data[['transliteration', 'translation']],
], ignore_index=True).drop_duplicates(subset=['transliteration'])
print(f"Combined train: {len(train_all):,} rows")""")


# =============================================================================
# Cell: Context-aware input construction
# =============================================================================
add_code("""CONTEXT_WINDOW = 2
PREFIX = "translate Akkadian to English: "

def build_context_inputs_df(df, context_col='transliteration', id_col=None):
    \"\"\"Build context-enriched inputs with ±2 surrounding lines.\"\"\"
    inputs = []
    df = df.copy().reset_index(drop=True)

    if id_col and id_col in df.columns:
        for gid, group in df.groupby(id_col, sort=False):
            idxs = group.index.tolist()
            lines = group[context_col].tolist()
            for local_i, orig_i in enumerate(idxs):
                start = max(0, local_i - CONTEXT_WINDOW)
                end = min(len(lines), local_i + CONTEXT_WINDOW + 1)
                ctx = lines[start:end]
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
            ctx = lines[start:end]
            cur = i - start
            ctx[cur] = f"[CURRENT] {ctx[cur]}"
            inputs.append(PREFIX + " [SEP] ".join(ctx))
        return inputs

# Detect id column for document-level context grouping
id_col_train = 'text_id' if 'text_id' in train_comp.columns else ('oare_id' if 'oare_id' in train_comp.columns else None)
id_col_test = 'text_id' if 'text_id' in test.columns else None
print(f"id_col_train: {id_col_train}, id_col_test: {id_col_test}")

train_comp_sorted = train_comp.sort_values(id_col_train).reset_index(drop=True) if id_col_train else train_comp.copy()
comp_inputs = build_context_inputs_df(train_comp_sorted, id_col=id_col_train)
comp_targets = train_comp_sorted['translation'].tolist()

ext_inputs = [PREFIX + s for s in ext_data['transliteration'].tolist()] if len(ext_data) > 0 else []
ext_targets = ext_data['translation'].tolist() if len(ext_data) > 0 else []

test_sorted = test.sort_values(id_col_test).reset_index(drop=True) if id_col_test else test.copy()
test_inputs = build_context_inputs_df(test_sorted, id_col=id_col_test)

all_train_inputs = comp_inputs + ext_inputs
all_train_targets = comp_targets + ext_targets

print(f"Total train pairs: {len(all_train_inputs):,}")
print(f"Test inputs: {len(test_inputs):,}")
print(f"\\nSample comp input: {comp_inputs[0][:200]}")
print(f"Sample comp target: {comp_targets[0][:100]}")""")


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
    # Fallback: HuggingFace Hub (requires internet=true, but try anyway)
    return "mattiaangeli/byt5-akkadian-mbr-v2"

print("Available model inputs:")
for p in sorted(glob.glob('/kaggle/input/models/*/*')):
    print(f"  {p}")

model_path = find_model_path(['byt5-akkadian-mbr-v2', 'byt5-akkadian-mbr', 'akkadian-mbr'])
print(f"\\nUsing model: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32
)
model = model.to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Model dtype: {next(model.parameters()).dtype}")""")


# =============================================================================
# Cell: Dataset
# =============================================================================
add_code("""MAX_SOURCE_LEN = 384
MAX_TARGET_LEN = 192

class TranslationDataset(Dataset):
    def __init__(self, inputs, targets=None):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.inputs[idx],
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding=False,
        )
        if self.targets is not None:
            dec = tokenizer(
                self.targets[idx],
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding=False,
            )
            enc["labels"] = dec["input_ids"]
        return enc

# Val: last 500 competition samples (same domain as test)
VAL_SIZE = min(500, len(comp_inputs) // 5)
train_ds = TranslationDataset(
    all_train_inputs[:-VAL_SIZE] if len(all_train_inputs) > VAL_SIZE else all_train_inputs,
    all_train_targets[:-VAL_SIZE] if len(all_train_targets) > VAL_SIZE else all_train_targets,
)
val_ds = TranslationDataset(comp_inputs[-VAL_SIZE:], comp_targets[-VAL_SIZE:])
test_ds = TranslationDataset(test_inputs)

print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")""")


# =============================================================================
# Cell: Fine-tuning
# =============================================================================
add_code("""data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Memory check before training
if torch.cuda.is_available():
    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    print(f"GPU free memory before training: {free_mem / 1e9:.2f} GB")
    print(f"GPU allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

training_args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/byt5-akkadian-ft",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=32,
    warmup_ratio=0.05,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    fp16=(DEVICE == 'cuda'),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    generation_num_beams=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=50,
    report_to="wandb" if WANDB_API_KEY else "none",
    dataloader_num_workers=0,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("Starting fine-tuning...")
try:
    trainer.train()
    print("Fine-tuning complete.")
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"CUDA OOM Error: {e}")
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        raise
    raise

# Free disk: delete all checkpoint dirs
import shutil
ckpt_dir = "/kaggle/working/byt5-akkadian-ft"
if os.path.exists(ckpt_dir):
    for d in sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint-*"))):
        shutil.rmtree(d, ignore_errors=True)
        print(f"Deleted checkpoint: {d}")
    # Also delete optimizer states and trainer state from best model dir
    for f in glob.glob(os.path.join(ckpt_dir, "**/*.pt"), recursive=True):
        if "optimizer" in f or "scheduler" in f:
            os.remove(f)
            print(f"Deleted: {f}")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"Disk after cleanup:")
os.system("df -h /kaggle/working | tail -1")""")


# =============================================================================
# Cell: MBR Decoding
# =============================================================================
add_code("""# MBR (Minimum Bayes Risk) Decoding
# Generate N candidate translations per sentence.
# Select the one with highest average chrF++ against all other candidates.
# This "consensus" translation avoids outlier candidates and is consistently
# better than greedy / beam search for neural MT.

MBR_SAMPLES = 12
MBR_TEMPERATURE = 0.8
BATCH_SIZE = 1

chrf_metric = ChrFScorer(word_order=2)

def mbr_decode_batch(input_ids, attention_mask, n_samples=MBR_SAMPLES):
    \"\"\"Generate n_samples candidates and select by MBR criterion.\"\"\"
    model.eval()
    batch_size = input_ids.shape[0]
    candidates_per_input = [[] for _ in range(batch_size)]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_TARGET_LEN,
            do_sample=True,
            temperature=MBR_TEMPERATURE,
            top_p=0.95,
            num_return_sequences=n_samples,
            no_repeat_ngram_size=3,
        )

    for i in range(batch_size):
        seqs = outputs[i * n_samples: (i + 1) * n_samples]
        decoded = tokenizer.batch_decode(seqs, skip_special_tokens=True)
        candidates_per_input[i] = [s.strip() for s in decoded if s.strip()]

    best_translations = []
    for cands in candidates_per_input:
        if not cands:
            best_translations.append("unknown")
            continue
        if len(cands) == 1:
            best_translations.append(cands[0])
            continue

        scores = []
        for i, hyp in enumerate(cands):
            refs = [c for j, c in enumerate(cands) if j != i]
            try:
                score = chrf_metric.corpus_score([hyp] * len(refs), [[r] for r in refs]).score
            except Exception:
                score = 0.0
            scores.append(score)

        best_idx = int(np.argmax(scores))
        best_translations.append(cands[best_idx])

    return best_translations

print("Running MBR decoding on test set...")
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)

all_predictions = []
total_batches = len(test_loader)
for batch_idx, batch in enumerate(test_loader):
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    preds = mbr_decode_batch(input_ids, attention_mask)
    all_predictions.extend(preds)
    if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
        print(f"  Batch {batch_idx+1}/{total_batches}, {len(all_predictions)} translations done")

print(f"MBR decoding complete: {len(all_predictions)} predictions")""")


# =============================================================================
# Cell: Validate on val set
# =============================================================================
add_code("""print("Evaluating val set with MBR decoding (first 160 samples)...")
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)
val_preds = []
for batch in list(val_loader)[:20]:  # 20 batches x 8 = 160 samples
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    preds = mbr_decode_batch(input_ids, attention_mask, n_samples=MBR_SAMPLES)
    val_preds.extend(preds)

val_refs = comp_targets[-VAL_SIZE:][:len(val_preds)]
bleu_metric = BLEUScorer()
chrf_eval = ChrFScorer(word_order=2)
val_bleu = bleu_metric.corpus_score(val_preds, [val_refs]).score
val_chrf = chrf_eval.corpus_score(val_preds, [val_refs]).score
val_combined = (val_bleu + val_chrf) / 2

print(f"Val BLEU:     {val_bleu:.4f}")
print(f"Val chrF++:   {val_chrf:.4f}")
print(f"Val Combined: {val_combined:.4f}")

if WANDB_API_KEY:
    wandb.log({
        "val_bleu": val_bleu,
        "val_chrf": val_chrf,
        "val_combined": val_combined,
    })

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
print("\\nSaved /kaggle/working/submission.csv")""")


# =============================================================================
# Cell: W&B finish
# =============================================================================
add_code("""if WANDB_API_KEY:
    wandb.finish()
print("Done. submission.csv ready for Kaggle browser submission.")""")


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
