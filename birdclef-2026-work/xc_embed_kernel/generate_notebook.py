"""Generate xc_embed.ipynb Kaggle kernel for downloading XC audio + Perch v2 embedding."""
import json
from pathlib import Path

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None})

md([
    "# XC audio → Perch v2 embeddings (BirdCLEF+ 2026 external data)\n",
    "\n",
    "Downloads 11,563 XC recordings (Aves, Q A|B, non-ND license, cap 500/sp, 159 species),\n",
    "resamples to 32kHz mono, runs Perch v2, saves embeddings + logits.\n",
    "\n",
    "Output: `/kaggle/working/xc_perch_embeddings.{npz,parquet}` for attaching to main fork.\n",
    "\n",
    "Requires: enable_gpu=true, enable_internet=true, dataset `yasunorim/xc-birdclef-2026-target-urls`,\n",
    "model `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1`.\n"
])

code([
    "# Cell 0 — TF 2.20 install (Perch v2 StableHLO compatibility)\n",
    "# 0.926 fork と同じ ashok205/tf-wheels から tensorboard + tensorflow 2.20 を --no-deps で入れる\n",
    "# 既定 Kaggle image の古い StableHLO runtime は Perch v2 の vhlo.cosine_v2 を認識できない\n",
    "!pip install -q --no-deps /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorboard-2.20.0-py3-none-any.whl\n",
    "!pip install -q --no-deps /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl\n"
])

code([
    "# Cell 1 — imports & config\n",
    "import os, gc, time, json\n",
    "# Perch v2 is a CPU-only SavedModel; must disable GPU before TF import to avoid InvalidArgumentError\n",
    "os.environ['CUDA_VISIBLE_DEVICES'] = ''\n",
    "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n",
    "from pathlib import Path\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import soundfile as sf\n",
    "import requests\n",
    "import tensorflow as tf\n",
    "from tqdm.auto import tqdm\n",
    "\n",
    "# Kaggle mount 構造: /kaggle/input/{competitions,datasets,notebooks,models}/{owner}/{slug}/...\n",
    "# path hardcode せず rglob で探索\n",
    "_input = Path('/kaggle/input')\n",
    "URL_CSV = next(_input.rglob('xc_filtered.csv'), None)\n",
    "# Perch v2 SavedModel は 'saved_model.pb' を含むディレクトリ\n",
    "_perch_roots = [p.parent for p in _input.rglob('saved_model.pb') if 'perch' in str(p).lower()]\n",
    "PERCH_DIR = _perch_roots[0] if _perch_roots else None\n",
    "\n",
    "WORK = Path('/kaggle/working')\n",
    "AUDIO_TMP = WORK / 'tmp_audio'\n",
    "AUDIO_TMP.mkdir(exist_ok=True)\n",
    "\n",
    "SR = 32000\n",
    "WINDOW_SEC = 5\n",
    "WINDOW_SAMPLES = SR * WINDOW_SEC\n",
    "DL_TIMEOUT = 30\n",
    "DL_SLEEP_SEC = 0.5  # rate limit\n",
    "CHECKPOINT_EVERY = 500\n",
    "\n",
    "assert URL_CSV is not None and URL_CSV.exists(), f'xc_filtered.csv not found under /kaggle/input'\n",
    "assert PERCH_DIR is not None and PERCH_DIR.exists(), f'Perch v2 saved_model.pb not found'\n",
    "print('URL_CSV :', URL_CSV)\n",
    "print('PERCH_DIR:', PERCH_DIR)\n",
    "print('TensorFlow:', tf.__version__)\n",
    "print('GPU available:', tf.config.list_physical_devices('GPU'))\n"
])

code([
    "# Cell 2 — load URL list\n",
    "df = pd.read_csv(URL_CSV)\n",
    "print('Total rows:', len(df))\n",
    "print(df.head(3))\n",
    "print()\n",
    "print('Per-species count:')\n",
    "print(df.groupby('label').size().describe())\n"
])

code([
    "# Cell 3 — Perch v2 model load\n",
    "# Perch v2 exports a SavedModel; signature returns dict with 'embedding' (1536) and 'label' (logits)\n",
    "perch = tf.saved_model.load(str(PERCH_DIR))\n",
    "sig = perch.signatures['serving_default']\n",
    "print('Perch signature:', list(sig.structured_input_signature[1].keys()))\n",
    "print('Outputs:', list(sig.structured_outputs.keys()))\n"
])

code([
    "# Cell 4 — audio DL + preprocess helpers\n",
    "def download_audio(url, out_path):\n",
    "    try:\n",
    "        r = requests.get(url, timeout=DL_TIMEOUT, stream=True, allow_redirects=True)\n",
    "        r.raise_for_status()\n",
    "        with open(out_path, 'wb') as f:\n",
    "            for chunk in r.iter_content(chunk_size=65536):\n",
    "                f.write(chunk)\n",
    "        return True\n",
    "    except Exception as e:\n",
    "        return False\n",
    "\n",
    "def load_and_resample(path, target_sr=SR):\n",
    "    # soundfile handles mp3, wav, flac, ogg\n",
    "    y, sr = sf.read(str(path), dtype='float32', always_2d=False)\n",
    "    if y.ndim > 1:\n",
    "        y = y.mean(axis=1)\n",
    "    if sr != target_sr:\n",
    "        # simple linear resample (fast, OK for Perch; for higher fidelity use librosa)\n",
    "        ratio = target_sr / sr\n",
    "        new_len = int(len(y) * ratio)\n",
    "        y = np.interp(\n",
    "            np.linspace(0, len(y)-1, new_len, dtype=np.float32),\n",
    "            np.arange(len(y), dtype=np.float32),\n",
    "            y,\n",
    "        ).astype(np.float32)\n",
    "    return y\n",
    "\n",
    "def to_windows(y, win=WINDOW_SAMPLES):\n",
    "    n = len(y)\n",
    "    if n < win:\n",
    "        pad = np.zeros(win, dtype=np.float32)\n",
    "        pad[:n] = y\n",
    "        return pad[None, :]\n",
    "    n_win = n // win\n",
    "    return y[:n_win * win].reshape(n_win, win)\n"
])

code([
    "# Cell 5 — Perch batched embed\n",
    "def perch_embed(windows, batch=16):\n",
    "    # windows: (N, WINDOW_SAMPLES) float32\n",
    "    embs, logits = [], []\n",
    "    for i in range(0, len(windows), batch):\n",
    "        w = windows[i:i+batch]\n",
    "        inp = tf.constant(w, dtype=tf.float32)\n",
    "        out = sig(inputs=inp)\n",
    "        embs.append(out['embedding'].numpy().astype(np.float32))\n",
    "        logits.append(out['label'].numpy().astype(np.float32))\n",
    "    return np.concatenate(embs, axis=0), np.concatenate(logits, axis=0)\n"
])

code([
    "# Cell 6 — main loop\n",
    "embeddings_all = []\n",
    "logits_all = []\n",
    "meta_rows = []\n",
    "fail_rows = []\n",
    "\n",
    "start = time.time()\n",
    "for idx, row in tqdm(df.iterrows(), total=len(df), desc='XC embed'):\n",
    "    url = row['file_url']\n",
    "    xc_id = row['xc_id']\n",
    "    label = row['label']\n",
    "    out_audio = AUDIO_TMP / f'{xc_id}'\n",
    "\n",
    "    if not download_audio(url, out_audio):\n",
    "        fail_rows.append({'xc_id': xc_id, 'label': label, 'reason': 'download_failed'})\n",
    "        time.sleep(DL_SLEEP_SEC)\n",
    "        continue\n",
    "\n",
    "    try:\n",
    "        y = load_and_resample(out_audio)\n",
    "    except Exception as e:\n",
    "        fail_rows.append({'xc_id': xc_id, 'label': label, 'reason': f'decode_err:{type(e).__name__}'})\n",
    "        if out_audio.exists(): out_audio.unlink()\n",
    "        time.sleep(DL_SLEEP_SEC)\n",
    "        continue\n",
    "\n",
    "    windows = to_windows(y)\n",
    "    n_win = len(windows)\n",
    "\n",
    "    try:\n",
    "        emb, log = perch_embed(windows)\n",
    "    except Exception as e:\n",
    "        fail_rows.append({'xc_id': xc_id, 'label': label, 'reason': f'embed_err:{type(e).__name__}'})\n",
    "        if out_audio.exists(): out_audio.unlink()\n",
    "        time.sleep(DL_SLEEP_SEC)\n",
    "        continue\n",
    "\n",
    "    embeddings_all.append(emb)\n",
    "    logits_all.append(log)\n",
    "    for wi in range(n_win):\n",
    "        meta_rows.append({\n",
    "            'xc_id': xc_id,\n",
    "            'label': label,\n",
    "            'scientific': row['scientific'],\n",
    "            'window_idx': wi,\n",
    "            'q': row['q'],\n",
    "            'country': row['country'],\n",
    "            'lic': row['lic'],\n",
    "        })\n",
    "\n",
    "    if out_audio.exists(): out_audio.unlink()\n",
    "    time.sleep(DL_SLEEP_SEC)\n",
    "\n",
    "    if (idx + 1) % CHECKPOINT_EVERY == 0:\n",
    "        elapsed = time.time() - start\n",
    "        print(f'[{idx+1}/{len(df)}] elapsed={elapsed/60:.1f}min, windows accumulated={sum(e.shape[0] for e in embeddings_all)}')\n",
    "        gc.collect()\n",
    "\n",
    "print('Main loop done.')\n",
    "print(f'Success: {len(embeddings_all)}  Fail: {len(fail_rows)}')\n"
])

code([
    "# Cell 7 — save outputs\n",
    "emb_arr = np.concatenate(embeddings_all, axis=0) if embeddings_all else np.zeros((0, 1536), dtype=np.float32)\n",
    "log_arr = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0, 234), dtype=np.float32)\n",
    "meta_df = pd.DataFrame(meta_rows)\n",
    "fail_df = pd.DataFrame(fail_rows)\n",
    "\n",
    "print('emb shape:', emb_arr.shape)\n",
    "print('log shape:', log_arr.shape)\n",
    "print('meta rows:', len(meta_df))\n",
    "print('fail rows:', len(fail_df))\n",
    "\n",
    "np.savez_compressed(\n",
    "    WORK / 'xc_perch_embeddings.npz',\n",
    "    emb=emb_arr,\n",
    "    logits=log_arr,\n",
    ")\n",
    "meta_df.to_parquet(WORK / 'xc_perch_embeddings.parquet', index=False)\n",
    "fail_df.to_csv(WORK / 'xc_embed_failures.csv', index=False)\n",
    "\n",
    "print('Saved to /kaggle/working/')\n",
    "print('Files:')\n",
    "for p in sorted(WORK.iterdir()):\n",
    "    print(' ', p.name, p.stat().st_size)\n"
])

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = Path("C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/birdclef-2026-work/xc_embed_kernel/xc_embed.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
