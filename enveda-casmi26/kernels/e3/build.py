"""Build kernels/e3/main.ipynb = the public 0.440 notebook (E006) unchanged + our de novo part (dn1).

Base: canqiang/casmi26-e006-window-40-ice-fallback-true (version 1, public LB 0.440, Apache-2.0), copied unchanged as
base_e006.ipynb (see SOURCE.txt). Its main engine is the same v1 engine (casmi package of casmi26-v4b-models, V1FE,
public LightGBM ranker) as base_0420, so ../e1fuse/ours_denovo.py plugs in the same way: engine pass 1 -> dn1 decodes
SMILES under the top formulas -> dedupe against the candidates -> fp2 log-likelihood order -> top N injected through
pc_join (the 0.420 notebook's injection route; E006 itself has no pc_join, so its source is shipped from
../e1fuse/base_0420.ipynb) -> engine pass 2 rescoring. 90 min cap, any exception returns the pass-1 result.

No base cell is edited. With denovo > 0 two marked code cells are INSERTED:
  A. after the engine cell ("v1 engine exactly as in v3.6 ..."): write ours_fp2 / ours_denovo / pc_join to
     /kaggle/working/ours_mods, load dn1 + fp2, ours_denovo.install(E, V, public ranker), and wrap V.run once more so
     ours_denovo.CUR['mid'] is the base loop's current `mid` (the base-list cell is not touched).
  B. after the base-list cell ("all molecules -> base lists ..."): de novo summary, then release dn1 / fp2 (GPU) and
     the V.run wrapper before the base's ICEBERG / GLACIER cell.
denovo=0 inserts nothing: every code cell equals the base (only a markdown header and the metadata differ).

    python build.py NAME [denovo=N] [dns=S] [dnmin=M] [dnsrc=owner/slug] [dnforms=K] [dnlib=L]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(HERE, "..", "e1fuse")
name = sys.argv[1]
_arg = lambda k, d, f=float: next((f(a.split("=", 1)[1]) for a in sys.argv[2:] if a.startswith(k + "=")), d)  # noqa: E731
denovo = _arg("denovo", 0, int)       # max de novo rows per molecule (0 = off = the base notebook)
dns = _arg("dns", 48, int)            # samples per spectrum (4 spectra; S // 2 for the 2nd formula)
dnmin = _arg("dnmin", 90.0)           # cumulative de novo time cap (minutes): decode + engine pass 2
dnforms = _arg("dnforms", 2, int)     # formulas decoded per molecule
dnlib = _arg("dnlib", 0.0)            # > 0: no de novo when lib_max >= dnlib
dnsrc = _arg("dnsrc", "yasunorim/casmi26-dn1-denovo", str)   # kernel whose output holds dn1.pt

nb = json.load(open(os.path.join(HERE, "base_e006.ipynb"), encoding="utf-8"))
cells = nb["cells"]
src = lambda i: "".join(cells[i]["source"])  # noqa: E731


def find_cell(prefix):
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and src(i).startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def code_cell(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}


if denovo > 0:
    # ---- embedded modules: ours_fp2 / ours_denovo (e1fuse, ours) + pc_join (0.420 notebook, Apache-2.0)
    nb420 = json.load(open(os.path.join(E1, "base_0420.ipynb"), encoding="utf-8"))
    ns = {}
    exec("".join(nb420["cells"][2]["source"]), ns)
    embed = {"pc_join.py": ns["EMBED"]["pc_join.py"],
             "ours_fp2.py": open(os.path.join(E1, "ours_fp2.py"), encoding="utf-8").read(),
             "ours_denovo.py": open(os.path.join(E1, "ours_denovo.py"), encoding="utf-8").read()}
    cfg = dict(DENOVO_N=denovo, DENOVO_S=dns, DENOVO_MIN=dnmin, DENOVO_FORMS=dnforms, DENOVO_MAX_LIB=dnlib or None,
               DENOVO_SRC=dnsrc, VERSION=f"ours-{name}")
    hook_a = f'''# ==== ours (yasunorim; inserted by enveda-casmi26/kernels/e3/build.py -- every base cell is unchanged) ====
# de novo (dn1) rows for class 3: engine pass 1 -> dn1 decodes SMILES under the top formulas -> dedupe -> fp2 order ->
# top DENOVO_N injected through pc_join -> engine pass 2 (ranker, features) as for any pool-like candidate.
# Any error / empty decode / the time cap returns the pass-1 (= base) result for that molecule.
OURS_CFG = {cfg!r}
OURS_EMBED = {embed!r}
_ours_dir = '/kaggle/working/ours_mods'
os.makedirs(_ours_dir, exist_ok=True)
for _n, _s in OURS_EMBED.items():
    with open(os.path.join(_ours_dir, _n), 'w') as _fh:
        _fh.write(_s)
if _ours_dir not in sys.path:
    sys.path.insert(0, _ours_dir)
DENOVO_ON = False
try:
    import ours_denovo
    ours_denovo.MAX_NEW = OURS_CFG['DENOVO_N']; ours_denovo.SAMPLES = OURS_CFG['DENOVO_S']
    ours_denovo.MAX_SECS = 60 * OURS_CFG['DENOVO_MIN']; ours_denovo.MAX_FORMS = OURS_CFG['DENOVO_FORMS']
    ours_denovo.MAX_LIB = OURS_CFG['DENOVO_MAX_LIB']
    ours_denovo.load(log=print)
    ours_denovo.install(E, V, lambda C_, X_, F_, n_: rank_score(np.concatenate([X_, F_], 1), list(FEATURES) + list(n_)),
                        log=print)
    # the base-list loop's molecule id (global `mid` of the next cell) -> per-molecule seed and bookkeeping
    V.run = (lambda _f: (lambda *a, **k: (ours_denovo.CUR.__setitem__('mid', globals().get('mid')), _f(*a, **k))[1]))(V.run)
    DENOVO_ON = True
except Exception as _e:
    if not IS_RERUN:                 # commit run: a bug must fail the commit
        raise
    print('DENOVO FAILED -> base only', repr(_e), flush=True)
print('ours de novo', 'ON' if DENOVO_ON else 'OFF', OURS_CFG, f'{{time.time()-T0:.0f}}s', flush=True)
'''
    hook_b = '''# ==== ours (yasunorim; inserted by enveda-casmi26/kernels/e3/build.py): de novo summary, then release dn1 / fp2 ====
if DENOVO_ON:
    try:
        ours_denovo.summary(print)
        _dn_k = {m: set(v) for m, v in ours_denovo.ADDED.items() if v}
        _dn_t25 = sum(any(k in _dn_k.get(m, ()) for k in BASE[m][1][:25]) for m in BASE)
        _dn_t1 = sum(bool(BASE[m][1]) and BASE[m][1][0] in _dn_k.get(m, ()) for m in BASE)
        print(f'DENOVO in base lists: molecules with rows {len(_dn_k)} | a de novo row in the base top 25 {_dn_t25} '
              f'| at rank 1 {_dn_t1}', f'{time.time()-T0:.0f}s', flush=True)
        json.dump(dict(cfg=OURS_CFG, stats={k: v for k, v in ours_denovo.STATS.items()}, top25=_dn_t25, top1=_dn_t1),
                  open('/kaggle/working/ours_denovo_stats.json', 'w'), default=str)
    except Exception as _e:
        print('DENOVO summary failed', repr(_e), flush=True)
    import gc as _gc
    ours_denovo._M.clear(); ours_denovo.CUR.clear()      # dn1 + fp2 nets off the GPU before ICEBERG / GLACIER
    _gc.collect(); torch.cuda.empty_cache()
'''
    i_eng = find_cell("'''v1 engine exactly as in v3.6")
    i_base = find_cell("'''all molecules -> base lists")
    assert i_eng < i_base
    cells.insert(i_base + 1, code_cell(hook_b))     # later index first
    cells.insert(i_eng + 1, code_cell(hook_a))

cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                 f"# casmi26 {name} (yasunorim)\n\nBase: [CASMI26 E006 Window 40 ICE Fallback True (public LB 0.440)]"
                 "(https://www.kaggle.com/code/canqiang/casmi26-e006-window-40-ice-fallback-true) by canqiang, version 1, "
                 "Apache-2.0, and everything it credits (cells below, unchanged). "
                 + (f"Ours: formula-constrained de novo candidates (our dn1 model, ordered by our fp2) injected into the "
                    f"main engine through pc_join (huseyinemreaksoy, v4n fusion 0.420) in a second engine pass; max "
                    f"{denovo} rows per molecule, {dnmin:.0f} min cap. Two inserted cells are marked 'ours'. "
                    if denovo > 0 else "This build adds nothing (reference run). ")
                 + "Built by enveda-casmi26/kernels/e3/build.py in github.com/yasumorishima/kaggle-competitions."})
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
json.dump(nb, open(os.path.join(HERE, "main.ipynb"), "w", encoding="utf-8"), indent=1)

meta = {"id": f"yasunorim/casmi26-e3{name}", "title": f"casmi26 e3{name}", "code_file": "main.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
        "enable_tpu": "false", "machine_shape": "NvidiaTeslaT4", "enable_internet": "false",
        # the base notebook's own image (Python 3.12; its preflight asserts 3.12 and a T4)
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "dataset_sources": ["prvsiyan/casmi26-ranker-features", "prvsiyan/casmi26-fp-models-v2",
                            "dmitriigluzdov/casmi26-pubchem-popularity-prior", "megayak/casmi26-simulated-ranker-rows",
                            "ahmedberatozer/casmi26-fpnet-full1", "ahmedberatozer/casmi26-glacier",
                            "ahmedberatozer/casmi26-iceberg", "ahmedberatozer/casmi26-pubchem-tier",
                            "ahmedberatozer/casmi26-v2-pool", "ahmedberatozer/casmi26-v3-models",
                            "ahmedberatozer/casmi26-v4b-models", "prvsiyan/chebi-lipidmaps-casmi26",
                            "prvsiyan/coconut-casmi26-candidates", "hengck23/hengck23-dreams-enveda-casmi26",
                            "metric/rdkit-2026-3-3-wheel", "eveliaveldrine/rdkit-wheels"],
        "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
        "kernel_sources": ["yasunorim/casmi26-fp2-peak-transformer", "yasunorim/casmi26-fp2L3"] + ([dnsrc] if denovo > 0 else []),
        "model_sources": []}
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("wrote main.ipynb", meta["id"], "denovo", denovo, "dns", dns, "dnmin", dnmin, "dnforms", dnforms, "dnlib", dnlib,
      "dnsrc", dnsrc if denovo > 0 else "-")
