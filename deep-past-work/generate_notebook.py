"""
Deep Past Challenge - ByT5 MBR Fine-tuning
Strategy:
  - Base: mattiaangeli/byt5-akkadian-mbr-v2 (already fine-tuned on competition data)
  - Additional data: ORACC parallel corpus + enriched data
  - Decoding: MBR (Minimum Bayes Risk) for maximum translation quality
  - Context: prepend surrounding lines (window=2)
"""

# %% [markdown]
# # Deep Past Challenge: ByT5 + MBR Decoding
# - Base: `mattiaangeli/byt5-akkadian-mbr-v2` (fine-tuned on Akkadian)
# - Extra training: ORACC parallel corpus + enriched data
# - Decoding: MBR (Minimum Bayes Risk) with chrF++ as utility function
# - Context window: ±2 surrounding lines

# %% Cell 1: Setup & imports
import os
import gc
import glob
import json
import numpy as np
import pandas as pd
import torch
import wandb
from pathlib import Path
from itertools import islice
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
from sacrebleu.metrics import BLEU, CHRF
import warnings
warnings.filterwarnings('ignore')

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
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% Cell 2: Data loading - competition data
_slug = 'deep-past-initiative-machine-translation'
_matches = glob.glob(f'/kaggle/input/**/{_slug}', recursive=True)
DATA_DIR = Path(_matches[0]) if _matches else Path(f'/kaggle/input/{_slug}')
print(f"DATA_DIR: {DATA_DIR}")

train_comp = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

train_comp['transliteration'] = train_comp['transliteration'].fillna('')
train_comp['translation'] = train_comp['translation'].fillna('')
test['transliteration'] = test['transliteration'].fillna('')

print(f"Competition train: {train_comp.shape}")
print(f"Test: {test.shape}")

# %% Cell 3: Load external ORACC data
def load_external_data():
    """Load ORACC parallel corpus and enriched data."""
    dfs = []

    # Try multiple ORACC sources
    oracc_paths = [
        '/kaggle/input/akkadian-oracc-combined',
        '/kaggle/input/oracc-akkadian-english-parallel-corpus',
        '/kaggle/input/akkadian-enriched-data',
    ]
    for base in oracc_paths:
        base = Path(base)
        if not base.exists():
            continue
        for csv_file in base.glob('**/*.csv'):
            try:
                df = pd.read_csv(csv_file)
                # Normalize column names
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
                print(f"  Skip {csv_file.name}: {e}")

    if not dfs:
        print("No external data found.")
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

print(f"Combined train: {len(train_all):,} rows")

# %% Cell 4: Context-aware input construction
CONTEXT_WINDOW = 2
PREFIX = "translate Akkadian to English: "

def build_context_inputs_df(df, context_col='transliteration', id_col=None):
    """Build context-enriched inputs.
    Groups by id_col if available to keep document context.
    """
    inputs = []
    df = df.copy().reset_index(drop=True)

    if id_col and id_col in df.columns:
        grouped = df.groupby(id_col, sort=False)
        idx_map = {}  # original index -> group_local_index
        for gid, group in grouped:
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
        # No grouping - still use window but without document boundary enforcement
        lines = df[context_col].tolist()
        for i in range(len(lines)):
            start = max(0, i - CONTEXT_WINDOW)
            end = min(len(lines), i + CONTEXT_WINDOW + 1)
            ctx = lines[start:end]
            cur = i - start
            ctx[cur] = f"[CURRENT] {ctx[cur]}"
            inputs.append(PREFIX + " [SEP] ".join(ctx))
        return inputs

# For competition train (has oare_id for ordering)
train_comp_sorted = train_comp.sort_values('oare_id').reset_index(drop=True)
comp_inputs = build_context_inputs_df(train_comp_sorted, id_col=None)
comp_targets = train_comp_sorted['translation'].tolist()

# For external data (no document structure - use simple inputs)
ext_inputs = [PREFIX + s for s in ext_data['transliteration'].tolist()] if len(ext_data) > 0 else []
ext_targets = ext_data['translation'].tolist() if len(ext_data) > 0 else []

# For test (has text_id)
test_inputs = build_context_inputs_df(test, id_col='text_id' if 'text_id' in test.columns else None)

all_train_inputs = comp_inputs + ext_inputs
all_train_targets = comp_targets + ext_targets

print(f"Total train pairs: {len(all_train_inputs):,}")
print(f"Test inputs: {len(test_inputs):,}")
print(f"\nSample comp input: {comp_inputs[0][:200]}")
print(f"Sample comp target: {comp_targets[0][:100]}")

# %% Cell 5: Load pre-trained model
# Find model path (mattiaangeli/byt5-akkadian-mbr-v2)
def find_model_path(slug_patterns):
    for pattern in slug_patterns:
        matches = glob.glob(f'/kaggle/input/models/**/*{pattern}*', recursive=True)
        for m in matches:
            if os.path.isdir(m) and any(
                os.path.exists(os.path.join(m, f))
                for f in ['config.json', 'pytorch_model.bin', 'model.safetensors']
            ):
                return m
    return None

model_path = find_model_path(['byt5-akkadian-mbr-v2', 'byt5-akkadian-mbr', 'akkadian-mbr'])
print(f"Model path: {model_path}")

# List what's in input dir
print("\nAvailable model inputs:")
for p in sorted(glob.glob('/kaggle/input/models/*/*')):
    print(f"  {p}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32)
model = model.to(DEVICE)
print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# %% Cell 6: Dataset
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 256

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
            with tokenizer.as_target_tokenizer():
                dec = tokenizer(
                    self.targets[idx],
                    max_length=MAX_TARGET_LEN,
                    truncation=True,
                    padding=False,
                )
            enc["labels"] = dec["input_ids"]
        return enc

# Val: last 500 competition samples (same domain as test)
VAL_SIZE = 500
train_ds = TranslationDataset(
    all_train_inputs[:-VAL_SIZE] if len(all_train_inputs) > VAL_SIZE else all_train_inputs,
    all_train_targets[:-VAL_SIZE] if len(all_train_targets) > VAL_SIZE else all_train_targets,
)
val_ds = TranslationDataset(comp_inputs[-VAL_SIZE:], comp_targets[-VAL_SIZE:])
test_ds = TranslationDataset(test_inputs)

print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")

# %% Cell 7: Fine-tuning (3 epochs on combined data)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/byt5-akkadian-ft",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    generation_num_beams=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=100,
    report_to="wandb" if WANDB_API_KEY else "none",
    dataloader_num_workers=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# %% Cell 8: MBR (Minimum Bayes Risk) Decoding
# MBR: generate N candidate translations, select the one with highest
# average chrF++ against all other candidates (consensus translation)

MBR_SAMPLES = 16  # number of candidates per sentence
MBR_TEMPERATURE = 0.8
BATCH_SIZE = 8

chrf_metric = CHRF(word_order=2)

def mbr_decode_batch(input_ids, attention_mask, n_samples=MBR_SAMPLES):
    """Generate n_samples candidates and select by MBR criterion."""
    model.eval()
    candidates_per_input = [[] for _ in range(input_ids.shape[0])]

    with torch.no_grad():
        # Generate diverse samples using sampling + temperature
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

    batch_size = input_ids.shape[0]
    for i in range(batch_size):
        seqs = outputs[i * n_samples: (i + 1) * n_samples]
        decoded = tokenizer.batch_decode(seqs, skip_special_tokens=True)
        candidates_per_input[i] = [s.strip() for s in decoded if s.strip()]

    # MBR selection: pick candidate maximizing avg chrF++ vs others
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
    if (batch_idx + 1) % 10 == 0:
        print(f"  Batch {batch_idx+1}/{total_batches}, generated {len(all_predictions)} translations")

print(f"MBR decoding complete: {len(all_predictions)} predictions")

# %% Cell 9: Evaluate on val set with MBR
print("\nEvaluating val set with MBR decoding...")
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)
val_preds = []
for batch in list(val_loader)[:20]:  # evaluate on first 160 val samples (time budget)
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    preds = mbr_decode_batch(input_ids, attention_mask, n_samples=MBR_SAMPLES)
    val_preds.extend(preds)

val_refs = comp_targets[-VAL_SIZE:][:len(val_preds)]
bleu = BLEU()
chrf = CHRF(word_order=2)
val_bleu = bleu.corpus_score(val_preds, [val_refs]).score
val_chrf = chrf.corpus_score(val_preds, [val_refs]).score
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

print("\nSample predictions:")
for i in range(min(5, len(val_preds))):
    print(f"  Ref:  {val_refs[i][:80]}")
    print(f"  Pred: {val_preds[i][:80]}")
    print()

# %% Cell 10: Submission
def postprocess(text):
    text = text.strip()
    return ' '.join(text.split()) if text else "unknown"

all_predictions = [postprocess(p) for p in all_predictions]

submission = pd.DataFrame({
    'id': test['id'],
    'translation': all_predictions[:len(test)],
})

print(f"Submission shape: {submission.shape}")
print(submission.head(10).to_string())
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("\nSaved /kaggle/working/submission.csv")

if WANDB_API_KEY:
    wandb.finish()
