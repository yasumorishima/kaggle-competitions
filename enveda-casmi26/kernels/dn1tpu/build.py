"""Build kernels/dn1tpu/main.py = the dn1 de novo generator trained longer with JAX on a Kaggle TPU v5e-8.

Kaggle's weekly GPU quota is shared by every competition (and torch_xla segfaults on the TPU image), so the long
dn1 training goes to the TPU (free quota), the same way kernels/fp2tpu does for fp2. Data, vocabulary, the novel
condition (DN_NOVEL_T), tokens, model, loss (cross-entropy, PAD ignored, label smoothing 0.05) and optimiser
(AdamW 4e-4, wd 1e-4, clip 1.0, one-cycle 5% warm-up incl. its beta1 cycle) are kernels/dn1's; only the training
loop runs in JAX (8-core data parallel). The torch Net is still built (CPU) for its initial weights and names;
the weights are written back as dn1.pt in dn1's format ({state, vocab, D}) after every epoch (and every
DN_CKPT_SECS), so kernels/dn1r decodes it unchanged. With DN_INIT=1 (default) and a dn1.pt under /kaggle/input
(kernel source casmi26-dn1-denovo), training continues from it in its own vocabulary. No decoding here: the
log has the train loss per epoch and the teacher-forced loss on the held-out np-examples rows, and a torch-vs-JAX
parity check at the end.

    python enveda-casmi26/kernels/dn1tpu/build.py [EPOCHS] [HOURS]
"""
import json
import os
import sys

epochs = sys.argv[1] if len(sys.argv) > 1 else "30"
hours = sys.argv[2] if len(sys.argv) > 2 else "8"
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "dn1", "main.py"), encoding="utf-8").read()


def swap(old, new):
    global src
    assert src.count(old) == 1, old[:60]
    src = src.replace(old, new, 1)


swap('"""CASMI26 dn1 (ours):', f'"""CASMI26 dn1 on TPU (JAX training only; built by dn1tpu/build.py: {epochs} epochs, {hours} h). dn1 (ours):')
swap('DN_WEIGHTS = glob.glob(os.environ.get("DN_WEIGHTS_GLOB", "/kaggle/input/**/dn1.pt"), recursive=True)   # present => decode only (no training)',
     'DN_WEIGHTS = glob.glob(os.environ.get("DN_WEIGHTS_GLOB", "/kaggle/input/**/dn1.pt"), recursive=True) \\\n'
     '    if os.environ.get("DN_INIT", "1") == "1" else []   # dn1tpu: present => continue training from it')
swap("from torch import nn  # noqa: E402\n", '''from torch import nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: E402
''')
swap('DEV = "cuda" if torch.cuda.is_available() else "cpu"', 'DEV = "cpu"   # torch only builds the initial weights, writes dn1.pt and checks parity')
swap('os.environ.get("DN_EPOCHS", "8")', f'os.environ.get("DN_EPOCHS", "{epochs}")')

# ---- continue from the previous dn1.pt in its own vocabulary (dn1 would switch to decode-only here)
swap('''    tr = tr[tr.inchikey14.isin(ids)].reset_index(drop=True)
    CK = None
    if DN_WEIGHTS:                                        # decode only: the trained model's own vocabulary
        CK = torch.load(DN_WEIGHTS[0], map_location="cpu")
        vocab = CK["vocab"]
        tr = tr.head(512).reset_index(drop=True)
        log("loaded", DN_WEIGHTS[0], "- decode only")
''', '''    CK = None
    if DN_WEIGHTS:                                        # dn1tpu: continue training, in the checkpoint's vocabulary
        CK = torch.load(DN_WEIGHTS[0], map_location="cpu")
        if CK.get("D", D) != D:
            globals()["D"] = CK["D"]
        log("init from", DN_WEIGHTS[0], "D", D, "vocab: checkpoint", len(CK["vocab"]), "data", len(vocab),
            "same" if list(CK["vocab"]) == list(vocab) else "DIFFERENT (checkpoint's is used)")
        vocab = list(CK["vocab"])
        vid = {t: i for i, t in enumerate(vocab)}
        ids = {}
        for ik, t in toks.items():
            if all(x in vid for x in t):
                ids[ik] = np.array([BOS] + [vid[x] for x in t] + [EOS], np.int64)
    tr = tr[tr.inchikey14.isin(ids)].reset_index(drop=True)
''')

JAX = '''

# ---- dn1's Net in JAX, on the torch state-dict names (norm-first encoder / decoder, 8 heads, GELU)
N_HEADS = 8
LSMOOTH = 0.05


def _lin(p, k, x):
    return x @ p[k + ".weight"].T + p[k + ".bias"]


def _ln(p, k, x):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * p[k + ".weight"] + p[k + ".bias"]


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)


def _sin(p, x):
    a = x[..., None] * p["sin.w"]
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)], -1)


def _drop(x, key, rate=0.1):
    if key is None:
        return x
    return jnp.where(jax.random.bernoulli(key, 1 - rate, x.shape), x / (1 - rate), 0.0)


def _mha(p, k, xq, xkv, blocked, key):
    """torch nn.MultiheadAttention (batch_first); blocked: bool, broadcastable to (B, H, Tq, Tk)"""
    W, b = p[k + ".in_proj_weight"], p[k + ".in_proj_bias"]
    B, T, Dm = xq.shape
    S = xkv.shape[1]
    dh = Dm // N_HEADS
    q = (xq @ W[:Dm].T + b[:Dm]).reshape(B, T, N_HEADS, dh)
    kk = (xkv @ W[Dm:2 * Dm].T + b[Dm:2 * Dm]).reshape(B, S, N_HEADS, dh)
    v = (xkv @ W[2 * Dm:].T + b[2 * Dm:]).reshape(B, S, N_HEADS, dh)
    a = jnp.einsum("bthd,bshd->bhts", q, kk) / jnp.sqrt(dh)
    a = jax.nn.softmax(jnp.where(blocked, -1e9, a), -1)
    a = _drop(a, key)
    o = jnp.einsum("bhts,bshd->bthd", a, v).reshape(B, T, Dm)
    return _lin(p, k + ".out_proj", o)


def _nl(p, pre):
    return sum(1 for k in p if k.startswith(pre) and k.endswith(".norm1.weight"))


def jencode(p, mz, it, prec, add, fc, key=None):
    nl = _nl(p, "enc.layers.")
    keys = list(jax.random.split(key, 4 * nl)) if key is not None else [None] * (4 * nl)
    pad = mz <= 0
    tok = _lin(p, "peak.2", _gelu(_lin(p, "peak.0", jnp.concatenate(
        [_sin(p, mz), _sin(p, jnp.clip(prec[:, None] - mz, 0)), it[..., None]], -1))))
    pp = _lin(p, "prec.2", _gelu(_lin(p, "prec.0", _sin(p, prec)))) + p["add.weight"][add]
    f = _lin(p, "form.2", _gelu(_lin(p, "form.0", jnp.concatenate([jnp.log1p(fc), (fc > 0).astype(fc.dtype)], -1))))
    x = jnp.concatenate([pp[:, None], f[:, None], tok], 1)
    mask = jnp.concatenate([jnp.zeros_like(pad[:, :2]), pad], 1)
    blocked = mask[:, None, None, :]
    for l in range(nl):
        q = f"enc.layers.{l}."
        x = x + _drop(_mha(p, q + "self_attn", _ln(p, q + "norm1", x), _ln(p, q + "norm1", x), blocked, keys[4 * l]),
                      keys[4 * l + 1])
        h = _drop(_gelu(_lin(p, q + "linear1", _ln(p, q + "norm2", x))), keys[4 * l + 2])
        x = x + _drop(_lin(p, q + "linear2", h), keys[4 * l + 3])
    return x, mask


def jdecode(p, mem, mmask, y, key=None):
    nl = _nl(p, "dec.layers.")
    keys = list(jax.random.split(key, 6 * nl)) if key is not None else [None] * (6 * nl)
    L = y.shape[1]
    x = p["emb.weight"][y] + p["pos.weight"][:L][None]
    causal = jnp.triu(jnp.ones((L, L), bool), 1)
    self_blocked = causal[None, None] | (y == PAD)[:, None, None, :]
    mem_blocked = mmask[:, None, None, :]
    for l in range(nl):
        q = f"dec.layers.{l}."
        h = _ln(p, q + "norm1", x)
        x = x + _drop(_mha(p, q + "self_attn", h, h, self_blocked, keys[6 * l]), keys[6 * l + 1])
        x = x + _drop(_mha(p, q + "multihead_attn", _ln(p, q + "norm2", x), mem, mem_blocked, keys[6 * l + 2]),
                      keys[6 * l + 3])
        h = _drop(_gelu(_lin(p, q + "linear1", _ln(p, q + "norm3", x))), keys[6 * l + 4])
        x = x + _drop(_lin(p, q + "linear2", h), keys[6 * l + 5])
    return _lin(p, "head.1", _ln(p, "head.0", x))


def jlogits(p, mz, it, prec, add, fc, y):
    mem, mm = jencode(p, mz, it, prec, add, fc)
    return jdecode(p, mem, mm, y)


def jloss_sums(p, mz, it, prec, add, fc, y, key=None):
    """sum over non-PAD target tokens of torch's CrossEntropyLoss(label_smoothing) terms, of the plain NLL, count"""
    k1, k2 = jax.random.split(key) if key is not None else (None, None)
    mem, mm = jencode(p, mz, it, prec, add, fc, k1)
    lg = jdecode(p, mem, mm, y[:, :-1], k2).astype(jnp.float32)
    tgt = y[:, 1:]
    valid = (tgt != PAD).astype(jnp.float32)
    lsm = jax.nn.log_softmax(lg, -1)
    nll = -jnp.take_along_axis(lsm, tgt[..., None], -1)[..., 0]
    per = (1 - LSMOOTH) * nll + LSMOOTH * (-lsm.mean(-1))
    return (per * valid).sum(), (nll * valid).sum(), valid.sum()

'''
swap("\n\ndef main():\n", JAX + "\ndef main():\n")

old_tail = src[src.index("    model = Net(V).to(DEV)\n"):src.index("\n\nif __name__ == \"__main__\":")]
new_tail = '''    torch.manual_seed(SEED)
    model = Net(V).to(DEV)   # initial weights and names (torch default init)
    if CK is not None:
        model.load_state_dict(CK["state"])
    devs = jax.devices()
    nd = len(devs)
    log("jax devices", nd, devs[0].platform)
    mesh = Mesh(np.array(devs), ("b",))
    rep, shard = NamedSharding(mesh, PartitionSpec()), NamedSharding(mesh, PartitionSpec("b"))
    buffers = {"sin.w"}   # not trained
    sd = {k_: jnp.asarray(v.numpy()) for k_, v in model.state_dict().items()}
    train_p = jax.device_put({k_: v for k_, v in sd.items() if k_ not in buffers}, rep)
    fixed = jax.device_put({k_: v for k_, v in sd.items() if k_ in buffers}, rep)

    def save_ckpt(tag):
        state = {**jax.device_get(train_p), **jax.device_get(fixed)}
        model.load_state_dict({k_: torch.from_numpy(np.array(v)) for k_, v in state.items()})
        torch.save({"state": model.state_dict(), "vocab": vocab, "D": D}, "dn1.pt.tmp")
        os.replace("dn1.pt.tmp", "dn1.pt")
        log("saved dn1.pt", tag)
        return state

    bs = int(os.environ.get("DN_BS", "256"))
    assert bs % nd == 0, (bs, nd)
    LR = 4e-4
    steps = max(1, EPOCHS * (len(tr) // bs))
    warm = 0.05 * steps
    sched = optax.cosine_onecycle_schedule(steps, LR, pct_start=0.05, div_factor=25.0, final_div_factor=1e4)

    def b1_sched(c):                                      # torch OneCycleLR cycles Adam's beta1 0.95 -> 0.85 -> 0.95
        c = jnp.asarray(c, jnp.float32)
        up = 0.85 + 0.05 * (1 + jnp.cos(jnp.pi * jnp.clip(c / warm, 0, 1)))
        down = 0.95 + (0.85 - 0.95) / 2 * (1 + jnp.cos(jnp.pi * jnp.clip((c - warm) / max(steps - warm, 1), 0, 1)))
        return jnp.where(c < warm, up, down)

    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.inject_hyperparams(optax.adamw)(learning_rate=sched, b1=b1_sched, b2=0.999, eps=1e-8,
                                                            weight_decay=1e-4))
    opt_state = opt.init(train_p)

    @jax.jit
    def step(tp, ost, mz, it, pr, ad, fc, y, key):
        def lf(t):
            s, _, n_ = jloss_sums({**t, **fixed}, mz, it, pr, ad, fc, y, key)
            return s / jnp.maximum(n_, 1.0)
        loss, g = jax.value_and_grad(lf)(tp)
        upd, ost = opt.update(g, ost, tp)
        return optax.apply_updates(tp, upd), ost, loss

    jsums = jax.jit(lambda tp, mz, it, pr, ad, fc, y: jloss_sums({**tp, **fixed}, mz, it, pr, ad, fc, y))

    # padded token matrix; batches are cut to a few length buckets (extra PAD columns do not change the math)
    LMAX = MAX_TOK + 2
    Y = np.zeros((len(tr), LMAX), np.int32)
    YL = np.zeros(len(tr), np.int32)
    for r, j in enumerate(Yl):
        Y[r, :len(j)] = j
        YL[r] = len(j)
    BUCKETS = [b_ for b_ in (40, 56, 72, 88) if b_ < LMAX] + [LMAX]

    def bucket(n):
        return next(b_ for b_ in BUCKETS if b_ >= n)

    # held-out teacher-forced loss: the np-examples rows (their truths in the model's vocabulary)
    ev_ids, ev_ok = [], []
    EFC = np.zeros((len(ev_rows), len(FORM_ELEMS)), np.float32)
    for r, (s, f) in enumerate(zip(ev_rows.normalized_smiles.values, ev_rows.molecular_formula.values)):
        t = tokenize(s) if isinstance(s, str) else None
        fc = form_counts(f)
        ok = t is not None and fc is not None and len(t) <= MAX_TOK and all(x in vid for x in t)
        ev_ok.append(ok)
        ev_ids.append(np.array([BOS] + [vid[x] for x in t] + [EOS], np.int32) if ok else np.array([BOS, EOS], np.int32))
        if ok:
            EFC[r] = fc
    ev_ok = np.array(ev_ok, bool)
    EY = np.zeros((len(ev_rows), LMAX), np.int32)
    for r, j in enumerate(ev_ids):
        if ev_ok[r]:
            EY[r, :len(j)] = j
    EPR = ev_rows.precursor_mz.values.astype(np.float32)
    EAD = ev_rows.adduct.map(aidx).fillna(len(TEST_ADDUCTS)).values.astype(np.int64)
    log("held-out rows", int(ev_ok.sum()), "of", len(ev_rows), "molecules", ev_rows[ev_ok].inchikey14.nunique())
    put = lambda a: jax.device_put(a, shard)  # noqa: E731

    def heldout():
        rows = np.where(ev_ok)[0]
        if not len(rows):
            return float("nan"), float("nan")
        EB = bs
        tot = np.zeros(3)
        for i in range(0, len(rows), EB):
            b = rows[i:i + EB]
            idx = np.concatenate([b, np.full(EB - len(b), b[0])])
            y = EY[idx]
            y[len(b):] = PAD                          # padding rows add no tokens
            y = y[:, :bucket(int(max(len(ev_ids[j]) for j in b)))]
            s = jsums(train_p, put(EMZ[idx]), put(EIT[idx]), put(EPR[idx]), put(EAD[idx]), put(EFC[idx]), put(y))
            tot += np.array([float(v) for v in s])
        return tot[0] / tot[2], tot[1] / tot[2]

    budget = float(os.environ.get("DN_TRAIN_SECS", str({hours} * 3600)))
    ck_every = float(os.environ.get("DN_CKPT_SECS", "1800"))
    l0, n0 = heldout()
    log(f"held-out before training: loss {l0:.4f} nll {n0:.4f}")
    log("train spectra", len(tr), "batch", bs, "steps", steps, "buckets", BUCKETS)
    base = jax.random.PRNGKey(SEED + (1000 if CK is not None else 0))
    k = 0
    t_ck = time.time()
    stop = False
    for ep in range(EPOCHS):
        perm = rng.permutation(len(tr))
        t_ep = time.time()
        losses = []
        for i in range(0, len(perm) - bs + 1, bs):
            b = perm[i:i + bs]
            y = Y[b, :bucket(int(YL[b].max()))]
            train_p, opt_state, loss = step(train_p, opt_state, put(MZ[b]), put(IT[b]), put(PR[b]), put(AD[b]),
                                            put(FC[b]), put(y), jax.random.fold_in(base, k))
            losses.append(loss)
            k += 1
            if k % 1000 == 0:
                log(f"ep {ep} step {k}/{steps} loss {float(jnp.stack(losses[-1000:]).mean()):.4f} "
                    f"({(time.time() - t_ep) / (i // bs + 1):.3f} s/step)")
            if time.time() - T0 > budget:
                stop = True
                break
            if time.time() - t_ck > ck_every:
                save_ckpt(f"ep {ep} step {k}")
                t_ck = time.time()
        tl = float(jnp.stack(losses).mean()) if losses else float("nan")
        hl, hn = heldout()
        log(f"epoch {ep} done, loss {tl:.4f} | held-out loss {hl:.4f} nll {hn:.4f} | "
            f"{time.time() - t_ep:.0f} s, {len(losses)} steps")
        state = save_ckpt(f"epoch {ep}")
        t_ck = time.time()
        if stop:
            log("training time budget reached")
            break
    if k == 0:
        state = save_ckpt("no steps")

    # parity: the saved torch weights vs the JAX forward, on a few train rows (dropout off)
    model.eval()
    n = min(16, len(tr))
    y = Y[:n, :int(YL[:n].max()) - 1]
    with torch.no_grad():
        mem, mm = model.encode(torch.from_numpy(MZ[:n]), torch.from_numpy(IT[:n]), torch.from_numpy(PR[:n]),
                               torch.from_numpy(AD[:n]), torch.from_numpy(FC[:n]))
        t_out = model.decode(mem, mm, torch.from_numpy(y).long()).numpy()
    with jax.default_matmul_precision("highest"):
        j_out = np.asarray(jax.jit(jlogits)(state, MZ[:n], IT[:n], PR[:n], AD[:n], FC[:n], y))
    live = y != PAD
    log("parity torch vs jax: max |logit diff|", float(np.abs(t_out - j_out)[live].max()),
        "logit scale", float(np.abs(t_out[live]).max()))
    log("done")'''.replace("{hours}", hours)
src = src.replace(old_tail, new_tail, 1)
assert "scaler" not in src and "torch.multinomial" not in src and "jloss_sums" in src

open(os.path.join(HERE, "main.py"), "w", encoding="utf-8").write(src)
meta = json.load(open(os.path.join(HERE, "..", "dn1", "kernel-metadata.json")))
meta.update({"id": "yasunorim/casmi26-dn1tpu", "title": "casmi26 dn1tpu",
             "enable_gpu": "false", "enable_tpu": "true", "enable_internet": "false",
             "kernel_sources": ["yasunorim/casmi26-dn1-denovo"], "machine_shape": "TpuV5E8"})
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("wrote", os.path.join(HERE, "main.py"), "epochs", epochs, "hours", hours)
