"""Build kernels/fp2tpu/main.py = the fp2 kernel trained with JAX on a Kaggle TPU v5e-8.

Kaggle runs at most two batch GPU sessions at once, so fp2 ensemble members beyond two go to
the TPU (free quota). Data, held-out set, model, loss and schedule are kernels/fp2's; only the
training loop and the held-out prediction run in JAX. The torch Net is still built (CPU) for its
initial weights and parameter names, and the trained weights are written back as fp2.pt under
the same names, so b5 loads them unchanged. A parity check (torch vs JAX forward) is logged.

    python enveda-casmi26/kernels/fp2tpu/build.py SEED EPOCHS [D]
"""
import json
import os
import sys

seed, epochs = sys.argv[1], sys.argv[2]
d_model = sys.argv[3] if len(sys.argv) > 3 else "384"
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "fp2", "main.py"), encoding="utf-8").read()


def swap(old, new):
    global src
    assert src.count(old) == 1, old[:60]
    src = src.replace(old, new, 1)


swap('"""CASMI26 FP2:', f'"""CASMI26 FP2 on TPU (JAX; built by fp2tpu/build.py: seed {seed}, {epochs} epochs, d {d_model}):')
swap('os.environ.get("FP2_SEED", "0")', f'os.environ.get("FP2_SEED", "{seed}")')
swap('os.environ.get("FP2_EPOCHS", "16")', f'os.environ.get("FP2_EPOCHS", "{epochs}")')
swap('os.environ.get("FP2_D", "384")', f'os.environ.get("FP2_D", "{d_model}")')
swap("from torch import nn  # noqa: E402\n", '''from torch import nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: E402
''')
swap('DEV = "cuda" if torch.cuda.is_available() else "cpu"', 'DEV = "cpu"   # torch only builds the initial weights and checks parity')

JAX = '''

# ---- the same network in JAX, on the torch state-dict names (norm-first encoder, 8 heads)
N_HEADS = 8


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


def jforward(p, mz, it, prec, add, key=None):
    n_layers = sum(1 for k in p if k.endswith("norm1.weight"))
    keys = list(jax.random.split(key, 4 * n_layers)) if key is not None else [None] * (4 * n_layers)
    pad = mz <= 0
    tok = _lin(p, "peak.2", _gelu(_lin(p, "peak.0", jnp.concatenate(
        [_sin(p, mz), _sin(p, jnp.clip(prec[:, None] - mz, 0)), it[..., None]], -1))))
    pp = _lin(p, "prec.2", _gelu(_lin(p, "prec.0", _sin(p, prec)))) + p["add.weight"][add]
    x = jnp.concatenate([pp[:, None], tok], 1)
    mask = jnp.concatenate([jnp.zeros_like(pad[:, :1]), pad], 1)
    B, T, Dm = x.shape
    dh = Dm // N_HEADS
    for l in range(n_layers):
        q = f"enc.layers.{l}."
        h = _ln(p, q + "norm1", x)
        qkv = h @ p[q + "self_attn.in_proj_weight"].T + p[q + "self_attn.in_proj_bias"]
        qq, kk, vv = [t.reshape(B, T, N_HEADS, dh) for t in jnp.split(qkv, 3, -1)]
        a = jnp.einsum("bthd,bshd->bhts", qq, kk) / jnp.sqrt(dh)
        a = jax.nn.softmax(jnp.where(mask[:, None, None, :], -1e9, a), -1)
        a = _drop(a, keys[4 * l])
        o = jnp.einsum("bhts,bshd->bthd", a, vv).reshape(B, T, Dm)
        x = x + _drop(_lin(p, q + "self_attn.out_proj", o), keys[4 * l + 1])
        h = _drop(_gelu(_lin(p, q + "linear1", _ln(p, q + "norm2", x))), keys[4 * l + 2])
        x = x + _drop(_lin(p, q + "linear2", h), keys[4 * l + 3])
    y = _gelu(_lin(p, "out.1", _ln(p, "out.0", x[:, 0])))
    return _lin(p, "out.3", y)

'''
swap("\n\ndef main():\n", JAX + "\ndef main():\n")

old_train = src[src.index("    model = Net().to(DEV)\n"):src.index("    # ---- evaluation: FP alone")]
new_train = '''    model = Net().to(DEV)   # initial weights and names (torch default init, seeded above)
    devs = jax.devices()
    log("jax devices", len(devs), devs[0].platform)
    mesh = Mesh(np.array(devs), ("b",))
    rep, shard = NamedSharding(mesh, PartitionSpec()), NamedSharding(mesh, PartitionSpec("b"))
    params = jax.device_put({k: jnp.asarray(v.numpy()) for k, v in model.state_dict().items()}, rep)
    buffers = {"sin.w"}   # not trained
    bs = 512
    steps = EPOCHS * (len(tr) // bs)
    sched = optax.cosine_onecycle_schedule(steps, 5e-4, pct_start=0.05, div_factor=25.0, final_div_factor=1e4)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    train_p = {k: v for k, v in params.items() if k not in buffers}
    fixed = {k: v for k, v in params.items() if k in buffers}
    opt_state = opt.init(train_p)

    @jax.jit
    def step(tp, ost, mz, it, pr, ad, yb, key):
        def lossf(t):
            logits = jforward({**t, **fixed}, mz, it, pr, ad, key)
            y = jnp.unpackbits(yb, axis=1).astype(jnp.float32)
            return optax.sigmoid_binary_cross_entropy(logits, y).mean()
        loss, g = jax.value_and_grad(lossf)(tp)
        upd, ost = opt.update(g, ost, tp)
        return optax.apply_updates(tp, upd), ost, loss

    base = jax.random.PRNGKey(SEED)
    k = 0
    for ep in range(EPOCHS):
        perm = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(perm) - bs + 1, bs):
            b = np.sort(perm[i:i + bs])
            put = lambda a: jax.device_put(a, shard)  # noqa: E731
            train_p, opt_state, loss = step(train_p, opt_state, put(MZ[b]), put(IT[b]), put(PR[b]), put(AD[b]),
                                            put(Y[b]), jax.random.fold_in(base, k))
            k += 1
            if k % 1000 == 0:
                tot = float(loss)
                log(f"ep {ep} step {k}/{steps} loss {tot:.4f}")
        log(f"epoch {ep} done")
    params = {**jax.device_get(train_p), **jax.device_get(fixed)}
    model.load_state_dict({k_: torch.from_numpy(np.array(v)) for k_, v in params.items()})
    torch.save(model.state_dict(), "fp2.pt")
    model.eval()
    with torch.no_grad():
        n = min(64, len(tr))
        t_out = model(torch.from_numpy(MZ[:n]), torch.from_numpy(IT[:n]), torch.from_numpy(PR[:n]),
                      torch.from_numpy(AD[:n])).numpy()
    j_out = np.asarray(jax.jit(jforward)(params, MZ[:n], IT[:n], PR[:n], AD[:n]))
    log("parity torch vs jax: max |logit diff|", float(np.abs(t_out - j_out).max()), "logit scale", float(np.abs(t_out).max()))

'''
src = src.replace(old_train, new_train, 1)

old_pred = src[src.index("    with torch.no_grad():\n        P = []\n"):src.index("    P = np.concatenate(P)\n")]
new_pred = '''    jfwd = jax.jit(jforward)
    P = []
    for i in range(0, len(ev_rows), 1024):
        s = slice(i, i + 1024)
        n = len(EMZ[s])
        padn = lambda a: np.concatenate([a, np.zeros((1024 - n,) + a.shape[1:], a.dtype)])  # noqa: E731
        P.append(np.asarray(jax.nn.sigmoid(jfwd(params, padn(EMZ[s]), padn(EIT[s]), padn(EPR[s]), padn(EAD[s]))))[:n])
'''
src = src.replace(old_pred, new_pred, 1)
assert "model(torch.from_numpy(EMZ" not in src and "scaler" not in src

open(os.path.join(HERE, "main.py"), "w", encoding="utf-8").write(src)
meta = json.load(open(os.path.join(HERE, "..", "fp2", "kernel-metadata.json")))
meta.update({"id": f"yasunorim/casmi26-fp2tpu{seed}", "title": f"casmi26 fp2tpu{seed}",
             "enable_gpu": "false", "enable_tpu": "true", "machine_shape": "TpuV5E8"})
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("wrote", os.path.join(HERE, "main.py"), "seed", seed, "epochs", epochs, "d", d_model)
