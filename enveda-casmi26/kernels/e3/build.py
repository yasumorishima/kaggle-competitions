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

With fgap > 0 one marked code cell F is appended after the last base cell (before hook C): for the molecules with no
forward-model evidence (gl_of(mid) empty and no non-None ICE_SCORES[mid] value, both from the base ICEBERG / GLACIER
cell), the top fgapk SMILES of submission.csv are re-ranked by z(-log rank) + fgap * z(our fragment explanation,
ours_frag.py = port of eval/fragexp.py); the rest of the list is kept.

    python build.py NAME [denovo=N] [dns=S] [dnmin=M] [dnsrc=owner/slug] [dnforms=K] [dnlib=L] [dnslot=a,b]
                         [mmp=N] [mmpc=C] [mmp2=N] [fgap=LAM] [fgapk=K] [fgapsec=S] [fgapmin=M] [cpu]

cpu: CPU kernel (for weeks with the GPU quota spent; CPU kernels may submit within 9 h). The only base-cell edits are the
T4 preflight check (cell "Read-only E004 observer"), device='cuda' -> CPU fallback for ICEBERG / GLACIER, the smoke
postflight's GPU acceptance checks (off), and the final pass's ICE / GLACIER status checks ('budget' allowed: on the CPU
they end on their time budget with part of the molecules scored). Time budgets are unchanged (ICEBERG covers fewer molecules on the CPU).
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
dnslot = [int(x) for x in _arg("dnslot", "", str).split(",") if x]   # fixed final slots (1-based) for the top de novo rows
dnslotlib = _arg("dnslotlib", 0.7)    # slot merge only when the engine's lib_max < this (library-weak = class-3-like)
mmp = _arg("mmp", 0, int)             # ours_mmp: max MMP class-3 products per molecule (0 = off)
mmpc = _arg("mmpc", 2, int)           # MMP rules seen at least this often in train
mmp2 = _arg("mmp2", 0, int)           # two-step MMP products (0 = off)
fgap = _arg("fgap", 0.0)              # ours_frag: weight of z(fragment explanation) where ICE/GLACIER gave no score (0 = off)
fgapk = _arg("fgapk", 25, int)        # ... re-rank only the top K of the final list
fgapsec = _arg("fgapsec", 10.0)       # ... per-molecule cap (s); the list is left as is when it is hit
fgapmin = _arg("fgapmin", 40.0)       # ... total cap (minutes)

nb = json.load(open(os.path.join(HERE, "base_e006.ipynb"), encoding="utf-8"))
cells = nb["cells"]
src = lambda i: "".join(cells[i]["source"])  # noqa: E731


def find_cell(prefix):
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and src(i).startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def code_cell(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}


cpu = "cpu" in sys.argv[2:]   # CPU kernel (GPU week quota spent): the base's T4 checks relaxed, ICEBERG / GLACIER on the CPU
if cpu:   # the only base-cell edits: device strings and the T4-only checks (time budgets unchanged)
    def _cpu_swap(i, old, new, n=1):
        s_ = src(i)
        assert s_.count(old) == n, (i, old, s_.count(old))
        cells[i]["source"] = s_.replace(old, new)
    _ip = find_cell('"""Read-only E004 observer')
    _cpu_swap(_ip, """    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    assert 'T4' in gpu, gpu
""", """    gpu = 'cpu'   # ours (cpu build): no GPU on this kernel
""")
    _ii = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "ice_fuse.run_ice(" in src(i))
    _cpu_swap(_ii, "device='cuda'", "device=('cuda' if torch.cuda.is_available() else 'cpu')", 2)
    _io = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "def _e004_postflight" in src(i))
    _cpu_swap(_io, """        if not full:
            assert meta['status'] == 'ok' and meta['device'].startswith('cuda'), (stage, meta)""",
              """        if not full and False:   # ours (cpu build): smoke-only GPU acceptance checks off (ICE hits its 300 s budget on the CPU)
            assert meta['status'] == 'ok' and meta['device'].startswith('cuda'), (stage, meta)""")
    _if = find_cell("METRIC_SOURCE_SHA256 = ")   # final pass: on the CPU ICEBERG / GLACIER may end on their time budget
    _cpu_swap(_if, "assert _stage_meta['ice'].get('status') == 'ok', _stage_meta['ice']",
              "assert _stage_meta['ice'].get('status') in ('ok', 'budget'), _stage_meta['ice']   # ours (cpu build): budget allowed")
    _cpu_swap(_if, "assert _stage_meta['glacier'].get('status') == 'ok', _stage_meta['glacier']",
              "assert _stage_meta['glacier'].get('status') in ('ok', 'budget'), _stage_meta['glacier']   # ours (cpu build)")


if fgap > 0:     # F. after the last base cell, BEFORE hook C (appended below), so de novo slots are applied last
    FCFG = dict(FGAP_LAM=fgap, FGAP_K=fgapk, FGAP_SECS=fgapsec, FGAP_MIN=fgapmin)
    hook_f = f'''# ==== ours (yasunorim; inserted by enveda-casmi26/kernels/e3/build.py -- every base cell is unchanged) ====
# fragment explanation where the forward models are blind: molecules with no GLACIER score (gl_of(mid) empty) and no
# ICEBERG score (no non-None value in ICE_SCORES[mid]) get their final top FGAP_K re-ranked by
# z(-log rank) + FGAP_LAM * z(ours_frag: explained share of sqrt intensity by 1-2 bond-cut fragments, port of
# enveda-casmi26/eval/fragexp.py). Ranks beyond FGAP_K are kept. Any error on the scoring rerun: submission as is.
OURS_FCFG = {FCFG!r}
_ours_dir = '/kaggle/working/ours_mods'
os.makedirs(_ours_dir, exist_ok=True)
with open(os.path.join(_ours_dir, 'ours_frag.py'), 'w') as _fh:
    _fh.write({open(os.path.join(HERE, "ours_frag.py"), encoding="utf-8").read()!r})
if _ours_dir not in sys.path:
    sys.path.insert(0, _ours_dir)
try:
    _ft0 = time.time()
    if 'ICE_SCORES' not in globals() or 'gl_of' not in globals() or 'te' not in globals():
        print('FGAP: ICE_SCORES / gl_of / te missing -> nothing done', flush=True)
    else:
        import ours_frag
        _ice_all = ICE_SCORES if isinstance(ICE_SCORES, dict) else {{}}
        def _f_cov(m_):
            _i = _ice_all.get(str(m_), {{}})
            _i = _i if isinstance(_i, dict) and any(v is not None for v in _i.values()) else {{}}
            return bool(gl_of(str(m_))) or bool(_i)
        _fspec = {{}}
        for _r in te.itertuples():
            _fspec.setdefault(str(_r.molecule_id), []).append(
                (np.column_stack([np.asarray(_r.ms2_mzs, np.float64), np.asarray(_r.ms2_normalized_intensities, np.float64)]),
                 float(_r.precursor_mz), ours_frag.polarity_of(_r.adduct, getattr(_r, 'ionization_mode', None))))
        _sp = '/kaggle/working/submission.csv'
        _sub = pd.read_csv(_sp, dtype=str, keep_default_na=False)
        _out, _nt, _n1, _nskip, _nunc = [], 0, 0, 0, 0
        _fdl = _ft0 + 60 * OURS_FCFG['FGAP_MIN']
        for _m, _s in zip(_sub.molecule_id, _sub.smiles):
            _l = [x for x in str(_s).split(';') if x]
            if _f_cov(_m) or len(_l) < 2 or not _fspec.get(str(_m)):
                _out.append(_s); continue
            _nunc += 1
            _top = _l[:OURS_FCFG['FGAP_K']]
            _fx = ours_frag.score_list(_top, _fspec[str(_m)], OURS_FCFG['FGAP_SECS']) if time.time() < _fdl else None
            if _fx is None:
                _nskip += 1; _out.append(_s); continue
            _o = ours_frag.rerank(_top, _fx, OURS_FCFG['FGAP_LAM'])
            _new = [_top[i] for i in _o] + _l[len(_top):]
            _nt += 1; _n1 += _new[0] != _l[0]
            _out.append(';'.join(_new))
        _sub['smiles'] = _out
        _sub.to_csv(_sp, index=False)
        print(f'FGAP lam {{OURS_FCFG["FGAP_LAM"]}} K {{OURS_FCFG["FGAP_K"]}}: uncovered molecules {{_nunc}} of {{len(_sub)}} | '
              f're-ranked {{_nt}} | top-1 changed {{_n1}} | skipped (time cap) {{_nskip}} | {{time.time()-_ft0:.0f}}s', flush=True)
except Exception as _e:
    if not IS_RERUN:                 # commit run: a bug must fail the commit
        raise
    print('FGAP FAILED -> submission as is', repr(_e), flush=True)
'''
    cells.append(code_cell(hook_f))

if denovo > 0:
    # ---- embedded modules: ours_fp2 / ours_denovo (e1fuse, ours) + pc_join (0.420 notebook, Apache-2.0)
    nb420 = json.load(open(os.path.join(E1, "base_0420.ipynb"), encoding="utf-8"))
    ns = {}
    exec("".join(nb420["cells"][2]["source"]), ns)
    embed = {"pc_join.py": ns["EMBED"]["pc_join.py"],
             "ours_fp2.py": open(os.path.join(E1, "ours_fp2.py"), encoding="utf-8").read(),
             "ours_denovo.py": open(os.path.join(E1, "ours_denovo.py"), encoding="utf-8").read()}
    cfg = dict(DENOVO_N=denovo, DENOVO_S=dns, DENOVO_MIN=dnmin, DENOVO_FORMS=dnforms, DENOVO_MAX_LIB=dnlib or None,
               DENOVO_SLOTS=dnslot, DENOVO_SLOT_LIB=dnslotlib,
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
    if dnslot:      # C. after the last base cell: the top de novo rows into fixed final slots of submission.csv
        hook_c = """# ==== ours (yasunorim; inserted by enveda-casmi26/kernels/e3/build.py): de novo rows into fixed final slots ====
# library-weak molecules (engine lib_max < DENOVO_SLOT_LIB): the top de novo rows (our fp2 order) are moved to the final
# slots DENOVO_SLOTS of submission.csv, the way the public line slots PubChem-only proposals, instead of the ranker's place.
if DENOVO_ON and OURS_CFG.get('DENOVO_SLOTS'):
    try:
        from rdkit import Chem as _OC
        def _oik(x):
            m_ = _OC.MolFromSmiles(x)
            return _OC.MolToInchiKey(m_)[:14] if m_ is not None else x
        _sp = '/kaggle/working/submission.csv'
        _sub = pd.read_csv(_sp)
        _as = {str(k): v for k, v in ours_denovo.ADDED_SMI.items()}
        _out, _ns = [], 0
        for _m, _s in zip(_sub.molecule_id, _sub.smiles.fillna('')):
            _lm, _dn = _as.get(str(_m), (1.0, []))
            if _dn and _lm < OURS_CFG['DENOVO_SLOT_LIB']:
                _dk = {_oik(x) for x in _dn[:len(OURS_CFG['DENOVO_SLOTS'])]}
                _l = [x for x in str(_s).split(';') if x and _oik(x) not in _dk]
                for _slot, _d in zip(OURS_CFG['DENOVO_SLOTS'], _dn):
                    _l.insert(min(_slot - 1, len(_l)), _d)
                _s = ';'.join(_l[:25]); _ns += 1
            _out.append(_s)
        _sub['smiles'] = _out
        _sub.to_csv(_sp, index=False)
        print('DENOVO slots', OURS_CFG['DENOVO_SLOTS'], 'lib <', OURS_CFG['DENOVO_SLOT_LIB'], 'molecules', _ns, flush=True)
    except Exception as _e:
        if not IS_RERUN:
            raise
        print('DENOVO SLOTS FAILED -> submission as is', repr(_e), flush=True)
"""
        cells.append(code_cell(hook_c))
    cells.insert(i_base + 1, code_cell(hook_b))     # later index first
    cells.insert(i_eng + 1, code_cell(hook_a))

if mmp > 0:      # M. after the engine cell, BEFORE hook A (ours_mmp must wrap E.generate before pc_join does)
    MCFG = dict(MMP_N=mmp, MMP_MINC=mmpc, MMP_TWO=mmp2)
    hook_m = f'''# ==== ours (yasunorim; inserted by enveda-casmi26/kernels/e3/build.py -- every base cell is unchanged) ====
# MMP class-3 generator (ours_mmp): replacement rules mined from all train structures (one acyclic single bond, same
# larger side) applied to the spectral analogs; products that land on the target mass join the v1 engine's candidates
# and are scored by its ranker like the base generator's products. Any error: the base generator only.
OURS_MCFG = {MCFG!r}
_ours_dir = '/kaggle/working/ours_mods'
os.makedirs(_ours_dir, exist_ok=True)
with open(os.path.join(_ours_dir, 'ours_mmp.py'), 'w') as _fh:
    _fh.write({open(os.path.join(E1, "ours_mmp.py"), encoding="utf-8").read()!r})
if _ours_dir not in sys.path:
    sys.path.insert(0, _ours_dir)
MMP_ON = False
try:
    import ours_mmp
    ours_mmp.mine(os.path.join(COMP, 'train.parquet'), workers=4, min_count=OURS_MCFG['MMP_MINC'], log=print)
    ours_mmp.install(E, max_new=OURS_MCFG['MMP_N'], two_step=OURS_MCFG['MMP_TWO'])
    MMP_ON = True
except Exception as _e:
    if not IS_RERUN:                 # commit run: a bug must fail the commit
        raise
    print('MMP FAILED -> base generator only', repr(_e), flush=True)
print('ours MMP', 'ON' if MMP_ON else 'OFF', OURS_MCFG, f'{{time.time()-T0:.0f}}s', flush=True)
'''
    _ie = find_cell("'''v1 engine exactly as in v3.6")
    cells.insert(_ie + 1, code_cell(hook_m))
    _ib = find_cell("'''all molecules -> base lists")
    cells.insert(_ib + 1, code_cell("# ==== ours (yasunorim): MMP summary ====\nif MMP_ON:\n    print('MMP stats', ours_mmp.STATS, flush=True)\n"))

cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                 f"# casmi26 {name} (yasunorim)\n\nBase: [CASMI26 E006 Window 40 ICE Fallback True (public LB 0.440)]"
                 "(https://www.kaggle.com/code/canqiang/casmi26-e006-window-40-ice-fallback-true) by canqiang, version 1, "
                 "Apache-2.0, and everything it credits (cells below, unchanged). "
                 + (f"Ours: formula-constrained de novo candidates (our dn1 model, ordered by our fp2) injected into the "
                    f"main engine through pc_join (huseyinemreaksoy, v4n fusion 0.420) in a second engine pass; max "
                    f"{denovo} rows per molecule, {dnmin:.0f} min cap. Two inserted cells are marked 'ours'. "
                    if denovo > 0 else "")
                 + (f"Ours: MMP class-3 generator (replacement rules mined from train, applied to the spectral analogs; max "
                    f"{mmp} products per molecule) inside the main engine's generator. " if mmp > 0 else "")
                 + (f"Ours: where neither GLACIER nor ICEBERG scored a molecule, its final top {fgapk} are re-ranked by "
                    f"z(-log rank) + {fgap} z(our 1-2 bond-cut fragment explanation of the spectra). " if fgap > 0 else "")
                 + ("This build adds nothing (reference run). " if denovo == 0 and mmp == 0 and fgap == 0 else "")
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
if cpu:
    meta.update(enable_gpu="false")
    meta.pop("machine_shape")
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("wrote main.ipynb", meta["id"], "denovo", denovo, "dns", dns, "dnmin", dnmin, "dnforms", dnforms, "dnlib", dnlib,
      "dnsrc", dnsrc if denovo > 0 else "-", "mmp", mmp, "fgap", fgap, "fgapk", fgapk, "cpu", cpu)
