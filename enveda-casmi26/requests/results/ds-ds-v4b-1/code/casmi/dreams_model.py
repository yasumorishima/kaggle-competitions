"""Self-contained PyTorch port of the DreaMS backbone + DreamsFP (spectrum -> 10,226-bit fingerprint logits).

engine v6 copy of work/analysis/dreams_ft/dreams_model.py (sha256 36a7f345...3ded3, verified faithful by
dreams_ft/verify_port.py).  The only change is patch_fpnet_load_model(), which is a no-op here because
casmi.fpnet.load_model dispatches natively on ck['arch'] == 'dreams_fp' (fpnet.NATIVE_DREAMS_FP).

Source (attribution required, see LICENCE note at the bottom):
  Bushuiev R. et al., "Self-supervised learning of molecular representations from millions of tandem mass spectra
  using DreaMS", Nat Biotechnol (2025), CC-BY-4.0.  Code: github.com/pluskal-lab/DreaMS (MIT).
  Weights: ssl_model.ckpt (HF roman-bushuiev/DreaMS, MIT; Zenodo 10997887, CC-BY-4.0), pretrained on GeMS/MassIVE only.
No pytorch_lightning / dreams / msml import is needed: the Lightning checkpoint is read with a stub unpickler.

Architecture (hyper-parameters read from the checkpoint and asserted in `check_ckpt_args`):
  input S (B, n+1, 2): token 0 = precursor (m/z, 1.1), tokens 1..n = top-n peaks by intensity (m/z in Da, intensity
        relative to the base peak), zero padding (m/z == 0 marks padding).  n = 60 in pre-training, 100 in fine-tuning.
  token = concat( ff_peak([m/1000, i]) (44),  ff_fourier(Fourier_lin_float_int(m)) (980) )  -> d = 1024
        Fourier 'lin_float_int': frequencies 1/(1e-4*k), k=2,4,..,9998 and 1/k, k=2..999 -> 5997 cos + 5997 sin
        ff_fourier: 5 x Linear (11994->512->512->512->512->980), ReLU after every layer; ff_peak: Linear(2,44)+ReLU
  7 pre-norm layers: LN -> 8-head dot-product attention (no q/k/v/o biases) -> residual; LN -> ReLU FFN 4096 (no
        biases) -> residual; final LN.  Dropout 0.1 everywhere (embeddings, attention probs, residuals, FFN, ff_fourier).
  Graphormer bias (non-parametrised): A_ij = q_i.k_j / sqrt(128) + sum_d (F_id - F_jd) = ... + s_i - s_j,
        F = ff_fourier output (980-d), s = F.sum(-1).
  Padding semantics of the original code (dreams/models/dreams/layers.py:MultiheadAttention):
        att.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(-1), -1e9) masks QUERY rows of padded tokens, not keys.
        => real tokens attend to padded tokens too (they act as extra key/value "registers"), and padded query rows
           attend uniformly (softmax of a constant row).  Faithful mode reproduces this exactly.
  SSL heads: ff_out Linear(1024, 20000) (m/z bins of 0.05 Da on [0, 1000), floor convention) on masked tokens;
        ro_out Linear(2048, 1) retention-order head on concat(precursor tokens of 2 spectra).

Efficient Graphormer bias: within a softmax row the s_i term is a constant and cancels, so the (B, n, n, 980)
tensor of the original is replaced by a per-key bias -s_j (B, 1, n, n) that is shared by all layers.  Padded
query rows are made exactly uniform by zeroing their queries and their bias row.  `graphormer='direct'` keeps the
materialised original computation for verification (verify_port.py).
"""
from __future__ import annotations

import math
import pickle
import types
from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MAX_MZ = 1000.0          # dformat A max m/z (Fourier x_max, m/z normalisation of ff_peak, SSL bin range)
X_MIN = 1e-4             # dformat A max_tbxic_stdev = Fourier x_min
PREC_INTENS = 1.1        # artificial intensity of the precursor token
HOT_BIN = 0.05           # SSL m/z bin size
N_HOT = int(round(MAX_MZ / HOT_BIN))   # 20000
PROTON = 1.007276467
N_ADDUCT_DEFAULT = 60    # len(casmi.chem.ADDUCT_LIST)
N_ELEM = 10


def _hi(t):
    """float32, or float64 if already float64 (keeps the float64 exact-math verification path)."""
    return t if t.dtype == torch.float64 else t.float()


# ================================================================================================================
# checkpoint reading without lightning / dreams / msml
# ================================================================================================================
class _Stub(dict):
    """Placeholder for any class of a package that is not installed (msml.*, dreams.*, pytorch_lightning.*)."""
    _qual = '?'

    def __new__(cls, *a, **k):
        o = dict.__new__(cls)
        o.__dict__['_args'] = a
        return o

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, st):
        if isinstance(st, tuple) and len(st) == 2 and (st[0] is None or isinstance(st[0], dict)):
            st = {**(st[0] or {}), **(st[1] or {})}
        if isinstance(st, dict):
            self.__dict__.update(st)
        else:
            self.__dict__['_state'] = st

    def __repr__(self):
        return f'<{self._qual} {self.__dict__}>'


_STUBS = {}
_SAFE = ('torch', 'collections', 'builtins', 'numpy', 'argparse', 'copyreg', '_codecs', 'typing', 'functools')


def _stub(mod, name):
    k = (mod, name)
    if k not in _STUBS:
        _STUBS[k] = type(name, (_Stub,), {'_qual': f'{mod}.{name}'})
    return _STUBS[k]


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith('pathlib') and name in ('PosixPath', 'WindowsPath', 'Path', 'PurePosixPath'):
            import pathlib
            return pathlib.PurePosixPath          # PosixPath cannot be instantiated on Windows
        if mod.split('.')[0] in _SAFE:
            try:
                return super().find_class(mod, name)
            except Exception:
                pass
        return _stub(mod, name)


_PM = types.ModuleType('dreams_stub_pickle')
_PM.Unpickler = _Unpickler
_PM.load = pickle.load


def load_lightning_ckpt(path, mmap=True):
    """-> the raw Lightning checkpoint dict (state_dict, hyper_parameters['args'] as argparse.Namespace, ...)."""
    return torch.load(path, map_location='cpu', pickle_module=_PM, weights_only=False, mmap=mmap)


# architecture hyper-parameters this port implements (asserted against the checkpoint)
EXPECTED_ARGS = dict(model='DreaMS', d_fourier=980, d_peak=44, d_mz_token=0, n_layers=7, n_heads=8,
                     ff_peak_depth=1, ff_fourier_depth=5, ff_fourier_d=512, ff_out_depth=1,
                     fourier_strategy='lin_float_int', fourier_trainable=False, fourier_num_freqs=None,
                     fourier_min_freq=None, prec_intens=1.1, charge_feature=False, graphormer_mz_diffs=True,
                     graphormer_parametrized=False, vanilla_transformer=False, pre_norm=True, scnorm=False,
                     attn_mech='dot-product', no_transformer_bias=True, no_ffs_bias=False,
                     train_objective='mask_mz_hot', hot_mz_bin_size=0.05, d_model=1024, d_graphormer_params=0)


def check_ckpt_args(args) -> dict:
    a = vars(args)
    bad = {k: (a.get(k, '<absent>'), v) for k, v in EXPECTED_ARGS.items() if a.get(k, '<absent>') != v}
    if bad:
        raise ValueError(f'checkpoint hyper-parameters differ from the port: {bad}')
    return {k: a[k] for k in ('dropout', 'att_dropout', 'residual_dropout', 'ff_dropout', 'max_peaks_n', 'frac_masks',
                              'mask_val', 'mask_prec', 'mask_peaks', 'mask_intens_strategy', 'focal_loss_gamma',
                              'ret_order_loss_w', 'mz_shift_aug_p', 'mz_shift_aug_max', 'lr', 'batch_size')}


# ================================================================================================================
# building blocks (module / parameter names mirror the original so that the state_dict maps 1:1)
# ================================================================================================================
class FeedForward(nn.Module):
    """Mirror of dreams.models.layers.feed_forward.FeedForward (int hidden_dim): Linear [Dropout ReLU] ... Linear [ReLU]."""

    def __init__(self, in_dim, out_dim, hidden_dim, depth, act_last=True, bias=True, dropout=0.0):
        super().__init__()
        hd = [hidden_dim] * depth
        layers = []
        for l in range(depth):
            d1 = hd[l - 1] if l != 0 else in_dim
            d2 = hd[l] if l != depth - 1 else out_dim
            layers.append(nn.Linear(d1, d2, bias=bias))
            if l != depth - 1:
                layers.append(nn.Dropout(p=dropout))
            if l != depth - 1 or act_last:
                layers.append(nn.ReLU())
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)


class FourierFeatures(nn.Module):
    """'lin_float_int' fixed Fourier features of m/z (Da); always evaluated in float32 (autocast off)."""

    def __init__(self, x_min=X_MIN, x_max=MAX_MZ):
        super().__init__()
        b = [1 / (x_min * i) for i in range(2, math.ceil(1 / x_min), 2)] + [1 / (1 * i) for i in range(2, math.ceil(x_max), 1)]
        self.register_buffer('b', torch.tensor(b).unsqueeze(0))          # (1, 5997) float32, same as the original

    def forward(self, x):                                                  # x (..., 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = 2 * torch.pi * _hi(x) @ self.b                             # identical op order to the original
            return torch.cat((torch.cos(x), torch.sin(x)), dim=-1)

    def num_features(self):
        return 2 * self.b.shape[1]


class Attention(nn.Module):
    """Original MultiheadAttention (dot-product, no biases): weights = [W_q; W_k; W_v; W_o] (4d x d)."""

    def __init__(self, d_model=1024, n_heads=8, att_dropout=0.1, fp32=True):
        super().__init__()
        self.d = d_model; self.h = n_heads; self.hd = d_model // n_heads; self.p = att_dropout; self.fp32 = fp32
        self.weights = nn.Parameter(torch.empty(4 * d_model, d_model))
        nn.init.normal_(self.weights, mean=0.0, std=(2 / (5 * d_model)) ** 0.5)

    def qkv(self, x):
        B, n, _ = x.shape
        return F.linear(x, self.weights[:3 * self.d]).view(B, n, 3, self.h, self.hd).permute(2, 0, 3, 1, 4).unbind(0)

    def forward(self, x, bias, qmask):
        """x (B,n,d); bias (B,1,n,n) float32 additive or None; qmask (B,1,n,1) bool = padded query rows (their bias
        rows must be 0) -> q zeroed -> exactly uniform attention, as masked_fill(-1e9) of the original."""
        B, n, d = x.shape
        q, k, v = self.qkv(x)
        if qmask is not None:
            q = q.masked_fill(qmask, 0.0)
        p = self.p if self.training else 0.0
        if self.fp32:
            with torch.autocast(device_type=x.device.type, enabled=False):
                o = F.scaled_dot_product_attention(_hi(q), _hi(k), _hi(v), attn_mask=bias, dropout_p=p)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=None if bias is None else bias.to(q.dtype), dropout_p=p)
        o = o.transpose(1, 2).reshape(B, n, d)
        return F.linear(o, self.weights[3 * self.d:])

    def forward_direct(self, x, gdists, pad):
        """Literal original computation (eval / verification): gdists (B,n,n,980) = F_i - F_j materialised."""
        B, n, d = x.shape
        q, k, v = self.qkv(x)
        att = torch.einsum('bhnd,bhdm->bhnm', q, k.transpose(-2, -1)) * (self.hd ** -0.5)
        if gdists is not None:
            att = att + gdists.sum(dim=-1).unsqueeze(1)
        if pad is not None:
            att = att.masked_fill(pad.unsqueeze(1).unsqueeze(-1), -1e9)
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.p, training=self.training)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, n, d)
        return F.linear(o, self.weights[3 * self.d:])


class FFN(nn.Module):
    def __init__(self, d_model=1024, ff_dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 4 * d_model, bias=False)
        self.out_proj = nn.Linear(4 * d_model, d_model, bias=False)
        self.p = ff_dropout
        std = (2 / (5 * d_model)) ** 0.5
        nn.init.normal_(self.in_proj.weight, std=std); nn.init.normal_(self.out_proj.weight, std=std)

    def forward(self, x):
        return self.out_proj(F.dropout(F.relu(self.in_proj(x)), p=self.p, training=self.training))


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=1024, n_layers=7, n_heads=8, att_dropout=0.1, ff_dropout=0.1, residual_dropout=0.1,
                 attn_fp32=True):
        super().__init__()
        self.n_layers = n_layers; self.rp = residual_dropout; self.grad_ckpt = False
        self.atts = nn.ModuleList([Attention(d_model, n_heads, att_dropout, attn_fp32) for _ in range(n_layers)])
        self.ffs = nn.ModuleList([FFN(d_model, ff_dropout) for _ in range(n_layers)])
        self.scales = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2 * n_layers + 1)])

    def _layer(self, i, x, bias, qmask):
        h = self.atts[i](self.scales[2 * i](x), bias, qmask)
        x = x + F.dropout(h, p=self.rp, training=self.training)
        h = self.ffs[i](self.scales[2 * i + 1](x))
        return x + F.dropout(h, p=self.rp, training=self.training)

    def _layer_direct(self, i, x, gdists, pad):
        h = self.atts[i].forward_direct(self.scales[2 * i](x), gdists, pad)
        x = x + F.dropout(h, p=self.rp, training=self.training)
        h = self.ffs[i](self.scales[2 * i + 1](x))
        return x + F.dropout(h, p=self.rp, training=self.training)

    def forward(self, x, bias, qmask):
        x = F.dropout(x, p=self.rp, training=self.training)
        for i in range(self.n_layers):
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x = checkpoint(self._layer, i, x, bias, qmask, use_reentrant=False)
            else:
                x = self._layer(i, x, bias, qmask)
        return self.scales[-1](x)

    def forward_direct(self, x, gdists, pad):
        x = F.dropout(x, p=self.rp, training=self.training)
        for i in range(self.n_layers):
            x = self._layer_direct(i, x, gdists, pad)
        return self.scales[-1](x)


@dataclass
class BackboneCfg:
    d_fourier: int = 980
    d_peak: int = 44
    n_layers: int = 7
    n_heads: int = 8
    ff_fourier_depth: int = 5
    ff_fourier_d: int = 512
    ff_peak_depth: int = 1
    dropout: float = 0.1            # ff_fourier / ff_peak internal dropout
    att_dropout: float = 0.1
    ff_dropout: float = 0.1
    residual_dropout: float = 0.1
    graphormer: str = 'key'         # 'key' (efficient per-key bias) | 'direct' (materialised, verification) | 'none'
    pad_mode: str = 'faithful'      # 'faithful' (keys NOT masked, padded rows uniform) | 'mask_keys' (diagnostic)
    fourier_fp32: bool = True       # Fourier features + ff_fourier in float32 even under autocast
    fourier_chunk: int = 4096       # tokens per chunk for Fourier features + ff_fourier (bounds the 11,994-d transient)
    fourier_ckpt: bool = True       # training: recompute each chunk's Fourier features in backward (not stored)
    attn_fp32: bool = True          # attention logits / softmax in float32 (the Graphormer bias can be large)
    ssl_heads: bool = False         # build ff_out (masked m/z bins) and ro_out (retention order)


class DreaMSBackbone(nn.Module):
    def __init__(self, cfg: BackboneCfg = None, **kw):
        super().__init__()
        cfg = cfg or BackboneCfg(**kw)
        self.cfg = cfg
        self.d_model = cfg.d_fourier + cfg.d_peak
        self.fourier_enc = FourierFeatures()
        self.ff_fourier = FeedForward(self.fourier_enc.num_features(), cfg.d_fourier, cfg.ff_fourier_d,
                                      cfg.ff_fourier_depth, dropout=cfg.dropout)
        self.ff_peak = FeedForward(2, cfg.d_peak, cfg.d_peak, cfg.ff_peak_depth, dropout=cfg.dropout)
        self.transformer_encoder = TransformerEncoder(self.d_model, cfg.n_layers, cfg.n_heads, cfg.att_dropout,
                                                      cfg.ff_dropout, cfg.residual_dropout, cfg.attn_fp32)
        if cfg.ssl_heads:
            self.ff_out = FeedForward(self.d_model, N_HOT, self.d_model, 1, act_last=False, bias=True, dropout=cfg.dropout)
            self.ro_out = nn.Linear(2 * self.d_model, 1, bias=False)
        self.grad_ckpt = False

    def set_grad_checkpointing(self, on=True):
        self.grad_ckpt = on
        self.transformer_encoder.grad_ckpt = on

    def _fourier_ffn(self, mz):
        if self.cfg.fourier_fp32:
            with torch.autocast(device_type=mz.device.type, enabled=False):
                return self.ff_fourier(self.fourier_enc(mz))
        return self.ff_fourier(self.fourier_enc(mz))

    def fourier_embed(self, mz):
        """mz (B,n,1) -> ff_fourier(Fourier(mz)) (B,n,980), computed per chunk of tokens (identical per-token math);
        in training each chunk is checkpointed so the 11,994-d features are never stored for backward."""
        shp = mz.shape[:-1]
        flat = mz.reshape(-1, 1)
        T = flat.shape[0]
        c = self.cfg.fourier_chunk if self.cfg.fourier_chunk and self.cfg.fourier_chunk > 0 else T
        use_ckpt = self.cfg.fourier_ckpt and self.training and torch.is_grad_enabled()
        outs = []
        for i in range(0, T, c):
            ch = flat[i:i + c]
            outs.append(checkpoint(self._fourier_ffn, ch, use_reentrant=False) if use_ckpt else self._fourier_ffn(ch))
        return (outs[0] if len(outs) == 1 else torch.cat(outs, 0)).view(*shp, -1)

    def embed(self, spec):
        """spec (B,n,2) float32 -> (tokens (B,n,1024), F (B,n,980) Fourier-FFN features)."""
        mz = spec[..., [0]]
        Ff = self.fourier_embed(mz)
        peak = self.ff_peak(spec / spec.new_tensor([MAX_MZ, 1.0]))
        x = torch.cat([peak.to(Ff.dtype) if peak.dtype != Ff.dtype else peak, Ff], dim=-1)
        return x, Ff

    def forward(self, spec, prec_add: Optional[torch.Tensor] = None, tok_add: Optional[torch.Tensor] = None):
        """spec: (B, n+1, 2) [m/z (Da), relative intensity], token 0 = precursor (m/z, 1.1), m/z == 0 -> padding.
        prec_add (B, d): added to the precursor token embedding; tok_add (B, d): added to every non-pad token.
        Returns final token states (B, n+1, d) (after the last LayerNorm), float32 residual stream."""
        spec = _hi(spec)
        pad = spec[..., 0] == 0
        x, Ff = self.embed(spec)
        if prec_add is not None:
            x = torch.cat([x[:, :1] + prec_add.to(x.dtype).unsqueeze(1), x[:, 1:]], 1)
        if tok_add is not None:
            x = x + tok_add.to(x.dtype).unsqueeze(1) * (~pad).unsqueeze(-1).to(x.dtype)
        g = self.cfg.graphormer
        if g == 'direct':
            gd = Ff.unsqueeze(2) - Ff.unsqueeze(1)                      # (B, n, n, 980) as in the original
            return self.transformer_encoder.forward_direct(x, gd, pad)
        B, n = pad.shape
        if g == 'key':
            s = _hi(Ff).sum(-1)                                         # (B, n)
            bias = (-s)[:, None, None, :].expand(B, 1, n, n)
        elif g == 'none':
            bias = torch.zeros(B, 1, n, n, device=spec.device, dtype=_hi(Ff).dtype)
        else:
            raise ValueError(g)
        if self.cfg.pad_mode == 'faithful':
            qmask = pad[:, None, :, None]
            bias = bias.masked_fill(qmask, 0.0).contiguous()
        elif self.cfg.pad_mode == 'mask_keys':
            qmask = None
            bias = bias.masked_fill(pad[:, None, None, :], float('-inf')).contiguous()
        else:
            raise ValueError(self.cfg.pad_mode)
        return self.transformer_encoder(x, bias, qmask)

    def ssl_logits(self, tokens):
        return self.ff_out(tokens)


# ================================================================================================================
# weight loading
# ================================================================================================================
def load_ssl_state(model: DreaMSBackbone, ckpt_path, verbose=True) -> dict:
    """Load ssl_model.ckpt into a DreaMSBackbone. Backbone keys must match exactly (raises otherwise).
    SSL head keys (ff_out.*, ro_out.*) are loaded when the module has them and reported as skipped otherwise."""
    ck = load_lightning_ckpt(ckpt_path)
    info = check_ckpt_args(ck['hyper_parameters']['args'])
    sd = dict(ck['state_dict'])
    dup = sd.pop('fourier_enc.Fourier frequencies')             # the original registers b twice
    assert torch.equal(dup, sd['fourier_enc.b'])
    ref_b = FourierFeatures().b
    b_maxdiff = float((sd['fourier_enc.b'] - ref_b).abs().max())
    head_keys = [k for k in sd if k.startswith(('ff_out.', 'ro_out.'))]
    if not model.cfg.ssl_heads:
        for k in head_keys:
            sd.pop(k)
    sd = {k: v.clone() for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    n_loaded = sum(v.numel() for v in sd.values())
    rep = dict(missing=missing, unexpected=unexpected, dropped_duplicate=['fourier_enc.Fourier frequencies'],
               ssl_head_keys=head_keys, ssl_heads_loaded=bool(model.cfg.ssl_heads), n_tensors=len(sd),
               n_params_loaded=n_loaded, fourier_b_maxdiff_vs_recomputed=b_maxdiff, global_step=ck.get('global_step'),
               epoch=ck.get('epoch'), ckpt_args=info)
    if missing or unexpected:
        raise RuntimeError(f'state_dict mismatch: missing={missing} unexpected={unexpected}')
    if verbose:
        print(f'[dreams] loaded {len(sd)} tensors ({n_loaded/1e6:.2f}M params) from {ckpt_path}; missing 0, '
              f'unexpected 0; ssl heads {"loaded" if model.cfg.ssl_heads else "skipped"} ({len(head_keys)} keys); '
              f'step {ck.get("global_step")}', flush=True)
    del ck
    return rep


# ================================================================================================================
# preprocessing / batching
# ================================================================================================================
def dreams_preprocess(mz, it, prec_mz, n_peaks=60, prec_intens=PREC_INTENS, mz_shift=0.0):
    """Raw peak list -> (n_peaks+1, 2) float32 as dreams.utils.data.SpectrumPreprocessor (no augmentation):
    relative intensities (base peak = 1), top-n peaks by intensity, zero padding, precursor prepended."""
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    ok = (mz > 0) & (it > 0)
    mz, it = mz[ok], it[ok]
    out = np.zeros((n_peaks + 1, 2), np.float32)
    out[0] = (np.float64(np.float32(prec_mz)) + mz_shift, prec_intens)   # same rounding as the GPU path
    if len(mz):
        it = it / it.max()
        if len(mz) > n_peaks:
            sel = np.sort(np.argsort(-it, kind='stable')[:n_peaks])
            mz, it = mz[sel], it[sel]
        out[1:1 + len(mz), 0] = mz.astype(np.float32).astype(np.float64) + mz_shift
        out[1:1 + len(mz), 1] = it
    return out


def fp_to_dreams_spec(mz, it, pad, prec, mode, n_peaks=100, neg_shift=True):
    """GPU conversion of fpnet.collate tensors (m/z-sorted, sqrt relative intensities, pad True = pad) to the DreaMS
    input (B, n_peaks+1, 2).  Negative mode (mode < 0): m/z + 2*proton if neg_shift ([M-H]- -> [M+H]+ scale)."""
    B, N = mz.shape
    valid = (~pad) & (mz > 0)
    inten = torch.where(valid, it.float() ** 2, torch.full_like(it, -1.0, dtype=torch.float32))
    k = min(n_peaks, N)
    top = inten.topk(k, dim=1)
    si = top.values
    sm = mz.float().gather(1, top.indices)
    keep = si > 0
    mx = torch.where(keep, si, torch.zeros_like(si)).amax(1, keepdim=True).clamp(min=1e-12)
    si = torch.where(keep, si / mx, torch.zeros_like(si))
    shift = (mode < 0).double() * (2 * PROTON) if neg_shift else torch.zeros_like(prec, dtype=torch.float64)
    sm = torch.where(keep, (sm.double() + shift[:, None]).float(), torch.zeros_like(sm))
    if k < n_peaks:
        sm = F.pad(sm, (0, n_peaks - k)); si = F.pad(si, (0, n_peaks - k))
    pm = torch.cat([(prec.float().double() + shift).float()[:, None], sm], 1)
    pi = torch.cat([torch.full((B, 1), PREC_INTENS, device=mz.device), si], 1)
    return torch.stack([pm, pi], -1)


def dreams_collate(items: Sequence[dict], device, n_peaks=100, neg_shift=True):
    """items as for fpnet.collate (mz, it = sqrt relative intensity, prec, adduct_ix, ce, ce_known, n_merged, mode)
    -> (spec (B,n+1,2), ad, ce, ce_known, merged, mode) for DreamsFP.forward_spec."""
    B = len(items)
    spec = np.zeros((B, n_peaks + 1, 2), np.float32)
    for r, x in enumerate(items):
        sh = 2 * PROTON if (neg_shift and x['mode'] < 0) else 0.0
        spec[r] = dreams_preprocess(x['mz'], np.asarray(x['it'], np.float64) ** 2, x['prec'], n_peaks, mz_shift=sh)
    T = lambda a, dt: torch.as_tensor(np.asarray(a, dt), device=device)
    return (T(spec, np.float32), T([x['adduct_ix'] for x in items], np.int64), T([x['ce'] for x in items], np.float32),
            T([x['ce_known'] for x in items], np.float32), T([x['n_merged'] / 4.0 for x in items], np.float32),
            T([x['mode'] for x in items], np.float32))


# ================================================================================================================
# DreamsFP: DreaMS backbone + conditioning + fingerprint head (FPNet interface)
# ================================================================================================================
class SinEmb(nn.Module):
    def __init__(self, dim, lo=0.0, hi=2.2):
        super().__init__()
        wav = torch.pow(10.0, (hi - lo) * torch.linspace(0, 1, dim // 2) + lo)
        self.register_buffer('inv', (2 * math.pi) / wav)

    def forward(self, x):
        a = x.float().unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


@dataclass
class FPCfg:
    nbits: int = 10226
    n_peaks: int = 100
    neg_shift: bool = True
    cond_target: str = 'prec'       # 'prec': conditioning added to the precursor token; 'all': also to every peak
    head_hidden: int = 2048
    head_dropout: float = 0.1
    n_adduct: int = N_ADDUCT_DEFAULT
    n_elem: int = N_ELEM


class DreamsFP(nn.Module):
    """forward(mz, it, pad, prec, ad, ce, ce_known, merged, mode) -> (z (B,nbits) logits, element_pred (B,10)):
    the same call signature and outputs as casmi.fpnet.FPNet (fpnet.collate batches, window_mrr, ModelBank)."""

    def __init__(self, fcfg: FPCfg = None, bcfg: BackboneCfg = None):
        super().__init__()
        self.fcfg = fcfg = fcfg or FPCfg()
        bcfg = bcfg or BackboneCfg()
        bcfg.ssl_heads = False
        self.backbone = DreaMSBackbone(bcfg)
        d = self.backbone.d_model
        self.nbits = fcfg.nbits; self.n_peaks = fcfg.n_peaks
        # conditioning (all zero-initialised: at step 0 the backbone computes exactly the pretrained function)
        self.ad_emb = nn.Embedding(fcfg.n_adduct, d)
        self.pol_emb = nn.Embedding(2, d)                        # 0 = negative (unseen in pre-training), 1 = positive
        self.ce_emb = SinEmb(32)
        self.cond_mlp = nn.Sequential(nn.Linear(32 + 2, 256), nn.GELU(), nn.Linear(256, d))
        for m in (self.ad_emb, self.pol_emb):
            nn.init.zeros_(m.weight)
        nn.init.zeros_(self.cond_mlp[-1].weight); nn.init.zeros_(self.cond_mlp[-1].bias)
        if fcfg.cond_target == 'all':
            self.tok_ad_emb = nn.Embedding(fcfg.n_adduct, d); self.tok_pol_emb = nn.Embedding(2, d)
            nn.init.zeros_(self.tok_ad_emb.weight); nn.init.zeros_(self.tok_pol_emb.weight)
        self.head = nn.Sequential(nn.Linear(2 * d, fcfg.head_hidden), nn.GELU(), nn.Dropout(fcfg.head_dropout),
                                  nn.Linear(fcfg.head_hidden, fcfg.nbits))
        self.elem_head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, fcfg.n_elem))

    # ------------------------------------------------------------------ parameter groups for layer-wise LR
    def param_groups(self):
        """-> dict name -> list of parameters: 'backbone_embed', 'backbone_L0'..'backbone_L6', 'backbone_norm', 'new'."""
        groups = {}
        for n, p in self.named_parameters():
            if n.startswith('backbone.transformer_encoder.'):
                parts = n.split('.')
                if parts[2] in ('atts', 'ffs'):
                    g = f'backbone_L{int(parts[3])}'
                else:                                           # scales.{j}: 2i, 2i+1 -> layer i; 2L -> final norm
                    j = int(parts[3]); L = self.backbone.cfg.n_layers
                    g = 'backbone_norm' if j == 2 * L else f'backbone_L{j // 2}'
            elif n.startswith('backbone.'):
                g = 'backbone_embed'
            else:
                g = 'new'
            groups.setdefault(g, []).append((n, p))
        return groups

    def condition(self, ad, ce, ce_known, merged, mode):
        pol = (mode > 0).long()
        feats = torch.cat([self.ce_emb(ce / 10.0 + 1.0) * ce_known.unsqueeze(-1), ce_known.unsqueeze(-1),
                           merged.unsqueeze(-1)], -1)
        cond = self.ad_emb(ad) + self.pol_emb(pol) + self.cond_mlp(feats)
        tok = None
        if self.fcfg.cond_target == 'all':
            tok = self.tok_ad_emb(ad) + self.tok_pol_emb(pol)
        return cond, tok

    def forward_spec(self, spec, ad, ce, ce_known, merged, mode, return_emb=False):
        cond, tok = self.condition(ad, ce, ce_known, merged, mode)
        x = self.backbone(spec, prec_add=cond, tok_add=tok)
        real = spec[..., 0] != 0
        real = torch.cat([torch.zeros_like(real[:, :1]), real[:, 1:]], 1)      # peaks only (not the precursor)
        m = real.unsqueeze(-1).to(x.dtype)
        mean = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        emb = torch.cat([x[:, 0], mean], -1)
        z = self.head(emb)
        el = self.elem_head(emb)
        if return_emb:
            return z, el, emb
        return z, el

    def forward(self, mz, it, pad, prec, ad, ce, ce_known, merged, mode, return_emb=False):
        spec = fp_to_dreams_spec(mz, it, pad, prec, mode, self.n_peaks, self.fcfg.neg_shift)
        return self.forward_spec(spec, ad, ce, ce_known, merged, mode, return_emb)


def build_dreams_fp(nbits, ssl_ckpt='data/external/dreams/ssl_model.ckpt', n_peaks=100, neg_shift=True,
                    cond_target='prec', verbose=True, **bkw):
    bcfg = BackboneCfg(**bkw)
    m = DreamsFP(FPCfg(nbits=nbits, n_peaks=n_peaks, neg_shift=neg_shift, cond_target=cond_target), bcfg)
    rep = load_ssl_state(m.backbone, ssl_ckpt, verbose=verbose) if ssl_ckpt else None
    return m, rep


def save_dreams_fp(path, model: DreamsFP, step, extra=None):
    d = dict(arch='dreams_fp', model=model.state_dict(), step=step, nbits=model.nbits,
             fcfg=asdict(model.fcfg), bcfg=asdict(model.backbone.cfg))
    if extra:
        d.update(extra)
    torch.save(d, path)


def dreams_fp_from_ckpt(ck, device='cuda'):
    bc = dict(ck['bcfg']); bc['ssl_heads'] = False
    m = DreamsFP(FPCfg(**ck['fcfg']), BackboneCfg(**bc))
    m.load_state_dict(ck['model'])
    return m.to(device).eval()


def load_dreams_fp(path, device='cuda'):
    return dreams_fp_from_ckpt(torch.load(path, map_location='cpu', weights_only=False), device)


def patch_fpnet_load_model():
    """Make casmi.fpnet.load_model (and hence fpnet.ModelBank) accept DreamsFP checkpoints next to FPNet ones.
    engine v6: casmi.fpnet already dispatches natively on ck['arch'] -> nothing to patch."""
    from casmi import fpnet
    if getattr(fpnet, 'NATIVE_DREAMS_FP', False) or getattr(fpnet.load_model, '_dreams_patched', False):
        return
    orig = fpnet.load_model

    def load_model(path, device='cuda'):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ck, dict) and ck.get('arch') == 'dreams_fp':
            return dreams_fp_from_ckpt(ck, device)
        del ck
        return orig(path, device)

    load_model._dreams_patched = True
    fpnet.load_model = load_model


# LICENCE: the DreaMS architecture/code is MIT (c) the DreaMS authors (pluskal-lab); ssl_model.ckpt is MIT (HF card) /
# CC-BY-4.0 (Zenodo 10997887).  Any redistributed derived weights must carry an MIT or CC-BY attribution (not CC0).
