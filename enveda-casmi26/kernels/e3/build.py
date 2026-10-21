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
                         [mmp=N] [mmpc=C] [mmp2=N] [fgap=LAM] [fgapk=K] [fgapsec=S] [fgapmin=M] [fgapcov=both|ice] [fgapdec=0|1]
                         [nl=LAM] [nlk=K] [nlcov=all|blind] [nlion=0|1] [fp2gap=LAM] [fp2k=K] [np=LAM] [npk=K] [npcov=all|blind] [tx=LAM] [txk=K] [txcov=all|blind] [iceb=S] [cpu]

With fp2gap > 0 one marked code cell P is appended after cell N: for the molecules neither GLACIER nor ICEBERG scored
(as cell F), the top fp2k SMILES are re-ranked by z(-log rank) + fp2gap * z(our fp2 log likelihood: Morgan r2 bits of the
candidate under our fp2 nets' bit probabilities, ../e1fuse/ours_fp2.py, nets from the kernel sources).

With nl > 0 one marked code cell N is appended after cell F: for each molecule whose spectra show a diagnostic neutral
loss (precursor m/z - peak m/z = hexose, glucuronic acid, SO3, H3PO4, CO2, NH3, CH3, ...; ours_nl.py, rates measured on
train), the top nlk SMILES are re-ranked by z(-log rank) + nl * nl * (naive-Bayes log likelihood of the seen / unseen
losses given each candidate's groups, raw nats). nlcov=blind limits it to the molecules neither GLACIER nor ICEBERG scored.

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
pcl = _arg("pcl", 0, int)             # ours_pcl (10-19): > 0 = PubChemLite rows into engine 1 through the de novo route (max per molecule)
pclcap = _arg("pclcap", 0, int)       # ours (10-20), 1: rows injected into engine 1 (pcl / de novo) may not hold rank 1 of submission.csv
dnnoinj = _arg("dnnoinj", 0, int)     # ours (10-20), 1: the proposed rows are not injected into engine 1, only kept for the dnslot cell
if pcl > 0:
    assert denovo == 0, "pcl replaces the de novo provider"
    denovo = pcl
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
fgapcov = _arg("fgapcov", "both", str)  # 'both': only where neither GLACIER nor ICEBERG scored; 'ice': wherever ICEBERG gave no score (GLACIER-scored too)
assert fgapcov in ("both", "ice"), fgapcov
fgapdec = _arg("fgapdec", 0, int)     # 1: decoy-corrected fragment score (ours_frag.DECOY_OFFSETS = -9,-5,-3,3,5,9 Da)
nl = _arg("nl", 0.0)                  # ours_nl: weight of z(neutral-loss substructure log likelihood) (0 = off)
nlk = _arg("nlk", 25, int)            # ... re-rank only the top K of the final list
nlcov = _arg("nlcov", "all", str)     # 'all': every molecule with a diagnostic loss; 'blind': only where neither GLACIER nor ICEBERG scored
assert nlcov in ("all", "blind"), nlcov
nlion = _arg("nlion", 0, int)         # 1: diagnostic product ions (ours_nl.ION_RULES) join the neutral-loss evidence
fp2gap = _arg("fp2gap", 0.0)          # ours_fp2: weight of z(fp2 log likelihood) where ICE/GLACIER gave no score (0 = off)
fp2k = _arg("fp2k", 25, int)
npl = _arg("np", 0.0)                  # ours_np: weight of z(Ertl NP-likeness) in the final top-K re-rank (0 = off)
npk = _arg("npk", 25, int)            # ... re-rank only the top K of the final list
npcov = _arg("npcov", "all", str)     # 'all': every molecule; 'blind': only where neither GLACIER nor ICEBERG scored
txl = _arg("tx", 0.0)                  # ours_tx: weight of z(similarity to the other test molecules' rank-1 structures) (0 = off)
txk = _arg("txk", 25, int)            # ... re-rank only the top K of the final list
txcov = _arg("txcov", "all", str)     # 'all': every molecule; 'blind': only molecules neither GLACIER nor ICEBERG scored, anchored on the scored ones
assert txcov in ("all", "blind"), txcov
base = _arg("base", "e006", str)      # 'e006': canqiang E006 (0.440); 'a451': saddamaditya Apex Synthesis (0.451 as titled)
assert base in ("e006", "a451", "v15"), base   # v15: ahmedberatozer casmi26-v15-n1a (v14 LB 0.455 + N1A)
bfrag = _arg("bfrag", 0.0)            # a451 only: weight of z(the base's frag_rescore) on molecules the forward models DID score (0 = off)
bfragk = _arg("bfragk", 25, int)      # ... inside same-formula groups of the final top K
bfragmin = _arg("bfragmin", 30.0)     # ... total cap (minutes)
val = _arg("val", 0, int)             # > 0: validation run, no submission (build_val.py): fake test = val enveda-180 molecules held out of train, stage-wise scores
valseed = _arg("valseed", 0, int)
valfull = _arg("valfull", 0, int)     # 1: an np val run keeps the leaky fpnet_full1 (to measure its leak)
valsrc = _arg("valsrc", "e180", str)
valc3 = _arg("valc3", 0, int)         # 1: the c2 half also loses its structure from the engine-1/2 pools (only the PubChem tier keeps it)   # e180: enveda-180 held out (leaks: the public FPNet/ranker saw it) | np: enveda-np-examples (unseen by the public FPNet)
pcfix = _arg("pcfix", 0, int)         # a451: PubChem-only candidates return to their gated-merge rank at the end (build_pc.py)
latecut = _arg("latecut", 0, int)
pcglib = _arg("pcglib", 9.0)          # pcguard only where lib_max < pcglib (ours, e3val7)
pcguard = _arg("pcguard", 0, int)     # a451: per molecule, the post-fusion list when its top-1 is PubChem-only, else the final list (build_pc.py)
pcdump = _arg("pcdump", 0, int)       # validation: write val_pconly.json (build_pc.py)     # a451: at the end, back to the list of the final metric pass (later overlays undone; build_pc.py)
pcvote = _arg("pcvote", 0, int)       # a451: PubChem-only candidates get engine 2's vote at their main rank in the fusion (build_pc.py)
fins = _arg("fins", 0, int)           # a451 only, > 0: the frag_rescore top pick among the final top K of rank 1's formula goes to this rank (ours)
lfrag = _arg("lfrag", 0, int)         # a451 only, 1: the base's frag_rescore.score_molecule -> our learned peak credits (ours_lfrag)
iceb = _arg("iceb", 0, int)           # > 0: the base's ICEBERG time budget (s) on the rerun (base 5400; the CPU needs more)          # ... re-rank only the top K of the final list

nb = json.load(open(os.path.join(HERE, f"base_{base}.ipynb"), encoding="utf-8"))
cells = nb["cells"]
src = lambda i: "".join(cells[i]["source"])  # noqa: E731


def find_cell(prefix):
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and src(i).startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def code_cell(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}


cpu = "cpu" in sys.argv[2:]   # CPU kernel (GPU week quota spent): the base's T4 checks relaxed, ICEBERG / GLACIER on the CPU
assert not (cpu and base not in ("e006", "v15")), "cpu edits are written for the E006-lineage cells (e006, v15)"
assert bfrag == 0 or base in ("a451", "v15"), "bfrag uses the a451 base's frag_rescore"
assert fins == 0 or base in ("a451", "v15"), "fins uses the a451 base's frag_rescore"
assert lfrag == 0 or base in ("a451", "v15"), "lfrag patches the a451 base's frag_rescore"
bx = _arg("bx", 0, int)               # 1: port of ahmedberatozer v19 BX (casmi26-v19-bioext: PubChem structures from biological depositors into engine 2's pool; not ours)
ownpc = _arg("ownpc", 0, int)         # ours (10-21), K > 0: our own PubChem tier (yasunorim/casmi26-own-pubchem-tier), structures with >= K PubMed links, into engine 2's pool
ownco = _arg("ownco", 0, int)         # ours (10-21), 1: our own COCONUT table (yasunorim/casmi26-own-coconut-candidates) adds its structures not yet in engine 2's pool
gg = _arg("gg", 0, int)               # v15 only, 1: saddamaditya's group-gap fragment step (a451 cells 32-33, Apache-2.0) ported in
if gg:
    assert base == "v15", "gg ports the a451 group gap into v15"
    _a451 = json.load(open(os.path.join(HERE, "base_a451.ipynb"), encoding="utf-8"))["cells"]
    for _pre in ("# v7c: MetFrag-lite fragment evidence", "def _factorial_entry(scope):"):
        _iv = find_cell(_pre)
        _ia = [i for i, c in enumerate(_a451) if c["cell_type"] == "code" and "".join(c["source"]).startswith(_pre)]
        assert len(_ia) == 1, _pre
        cells[_iv]["source"] = "".join(_a451[_ia[0]]["source"])
if lfrag:   # the one base-cell edit of lfrag: right after the base imports frag_rescore, our learned credits replace its score
    _old = "    import frag_rescore, math\n" if any("    import frag_rescore, math\n" in src(i) for i in range(len(cells))) else "    import frag_rescore\n"
    _il = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and _old in src(i))
    assert src(_il).count(_old) == 1
    _lsrc = open(os.path.join(HERE, "ours_lfrag.py"), encoding="utf-8").read()
    cells[_il]["source"] = src(_il).replace(_old, _old + (
        "    # ==== ours (yasunorim, build.py lfrag=1): learned peak credits (ours_lfrag) replace frag_rescore.score_molecule ====\n"
        f"    open('/kaggle/working/ours_lfrag.py', 'w').write({_lsrc!r})\n"
        "    import ours_lfrag\n"
        "    ours_lfrag.install(frag_rescore)\n"
        "    print('ours_lfrag installed: learned fragment credits', flush=True)\n"))
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
    if base == "v15":   # v15's N1 commit-only self-test needs a finite ICEBERG score, which the CPU smoke (300 s) does not reach
        _in1 = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "complement filter not exercised" in src(i))
        _cpu_swap(_in1, "assert _n1_xd, ('complement filter not exercised", "assert _n1_xd or True, ('complement filter not exercised")


if fgap > 0:     # F. after the last base cell, BEFORE hook C (appended below), so de novo slots are applied last
    FCFG = dict(FGAP_LAM=fgap, FGAP_K=fgapk, FGAP_SECS=fgapsec, FGAP_MIN=fgapmin, FGAP_COV=fgapcov, FGAP_DEC=fgapdec)
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
        if OURS_FCFG.get('FGAP_DEC'):
            ours_frag.DECOY_OFFSETS = (-9.0, -5.0, -3.0, 3.0, 5.0, 9.0)
        _ice_all = ICE_SCORES if isinstance(ICE_SCORES, dict) else {{}}
        def _f_cov(m_):
            _i = _ice_all.get(str(m_), {{}})
            _i = _i if isinstance(_i, dict) and any(v is not None for v in _i.values()) else {{}}
            if OURS_FCFG.get('FGAP_COV') == 'ice':   # fgapcov=ice: GLACIER-scored molecules are re-ranked too
                return bool(_i)
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
        print(f'FGAP lam {{OURS_FCFG["FGAP_LAM"]}} K {{OURS_FCFG["FGAP_K"]}} cov {{OURS_FCFG.get("FGAP_COV")}}: uncovered molecules {{_nunc}} of {{len(_sub)}} | '
              f're-ranked {{_nt}} | top-1 changed {{_n1}} | skipped (time cap) {{_nskip}} | {{time.time()-_ft0:.0f}}s', flush=True)
except Exception as _e:
    if not IS_RERUN:                 # commit run: a bug must fail the commit
        raise
    print('FGAP FAILED -> submission as is', repr(_e), flush=True)
'''
    cells.append(code_cell(hook_f))

if bfrag > 0:    # B. a451 base only, after cell F, BEFORE cell N
    BCFG = dict(BF_LAM=bfrag, BF_K=bfragk, BF_MIN=bfragmin)
    hook_b = f'''# ==== ours (yasunorim; inserted by enveda-casmi26/kernels/e3/build.py -- every base cell is unchanged) ====
# the base's own fragment scorer (frag_rescore, MetFrag-lite) as a third re-scorer where the forward models DID score:
# the base uses it only where neither GLACIER nor ICEBERG scored (whole molecule or a whole same-formula group).
# Here every molecule with a GLACIER or ICEBERG score gets its final top BF_K re-ranked inside same-formula groups by
# z(-log rank) + BF_LAM * z(frag_rescore) (gl_fuse.rerank_multi). Our check: on 171 enveda-np-examples molecules
# (truth among 20 same-formula train isomers) frag_rescore alone ranks the truth at MRR 0.568 (random 0.305).
OURS_BCFG = {BCFG!r}
try:
    _bt0 = time.time()
    _ice_all = ICE_SCORES if isinstance(ICE_SCORES, dict) else {{}}
    def _b_cov(m_):
        _i = _ice_all.get(str(m_), {{}})
        return bool(gl_of(str(m_))) or (isinstance(_i, dict) and any(v is not None for v in _i.values()))
    _bby = {{str(m): g for m, g in te.groupby('molecule_id', sort=False)}}
    _bcache = _fcache if '_fcache' in globals() and isinstance(_fcache, dict) else {{}}
    _sp = '/kaggle/working/submission.csv'
    _sub = pd.read_csv(_sp, dtype=str, keep_default_na=False)
    _out, _nc, _nt, _n1, _nskip = [], 0, 0, 0, 0
    _bdl = _bt0 + 60 * OURS_BCFG['BF_MIN']
    for _m, _s in zip(_sub.molecule_id, _sub.smiles):
        _l = [x for x in str(_s).split(';') if x]
        _g = _bby.get(str(_m))
        if not _b_cov(_m) or len(_l) < 2 or _g is None:
            _out.append(_s); continue
        _nc += 1
        if time.time() > _bdl:
            _nskip += 1; _out.append(_s); continue
        _top = _l[:OURS_BCFG['BF_K']]
        _spec = [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                      adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == 'positive' else -1)
                 for r in _g.itertuples()][:16]
        _fx = frag_rescore.score_molecule(_spec, _top, cache=_bcache)
        _fd = {{t: float(v) for t, v in zip(_top, _fx) if v is not None}}
        _o = gl_fuse.rerank_multi(_top, list(_top), [-float(np.log(i + 1)) for i in range(len(_top))],
                                  [_form2(t) for t in _top], [_fd], [OURS_BCFG['BF_LAM']], top_n=len(_top))
        _new = [_top[i] for i in _o] + _l[len(_top):]
        _nt += _new != _l; _n1 += _new[0] != _l[0]
        _out.append(';'.join(_new))
    _sub['smiles'] = _out
    _sub.to_csv(_sp, index=False)
    print(f'BFRAG lam {{OURS_BCFG["BF_LAM"]}} K {{OURS_BCFG["BF_K"]}}: covered molecules {{_nc}} of {{len(_sub)}} | order changed {{_nt}} | '
          f'top-1 changed {{_n1}} | skipped (time cap) {{_nskip}} | {{time.time()-_bt0:.0f}}s', flush=True)
except Exception as _e:
    if not IS_RERUN:                 # commit run: a bug must fail the commit
        raise
    print('BFRAG FAILED -> submission as is', repr(_e), flush=True)
'''
    cells.append(code_cell(hook_b))

if pcfix or pcvote or latecut or pcguard or pcdump:   # P. PubChem-only protection (build_pc.py)
    exec(open(os.path.join(HERE, "build_pc.py"), encoding="utf-8").read())
if val > 0:      # V. validation rig (build_val.py): fake test before the smoke switch, stage snapshots, scores at the end
    exec(open(os.path.join(HERE, "build_val.py"), encoding="utf-8").read())

if fins > 0:     # I. a451 base only, appended after the last base cell (after the base's E2R2)
    ICFG = dict(FI_SLOT=fins, FI_K=bfragk, FI_MIN=bfragmin)
    hook_i = f'''# ==== ours (yasunorim; inserted by enveda-casmi26/kernels/e3/build.py -- every base cell is unchanged) ====
# fragment-hedge insert: a third opinion placed at a fixed rank, like the base's engine-2 rank-2 insert. Ranks 1 and 2
# are kept. Among the final top FI_K candidates with rank 1's molecular formula, the one the fragment scorer
# (frag_rescore.score_molecule, with lfrag=1 our learned credits) likes best moves to rank FI_SLOT when it is not already
# above it. Isomer check (truth among 20 same-formula train isomers): learned scorer MRR 0.748 on enveda-180 held out.
OURS_ICFG = {ICFG!r}
try:
    _it0 = time.time()
    _iby = {{str(m): g for m, g in te.groupby('molecule_id', sort=False)}}
    _icache = _fcache if '_fcache' in globals() and isinstance(_fcache, dict) else {{}}
    _sp = '/kaggle/working/submission.csv'
    _sub = pd.read_csv(_sp, dtype=str, keep_default_na=False)
    _out, _nm, _nins, _nskip, _nalr = [], 0, 0, 0, 0
    _idl = _it0 + 60 * OURS_ICFG['FI_MIN']
    _slot = OURS_ICFG['FI_SLOT']
    for _m, _s in zip(_sub.molecule_id, _sub.smiles):
        _l = [x for x in str(_s).split(';') if x]
        _g = _iby.get(str(_m))
        if len(_l) <= _slot or _g is None or _l[0] == 'CCO':
            _out.append(_s); continue
        if time.time() > _idl:
            _nskip += 1; _out.append(_s); continue
        _f1 = _form2(_l[0])
        _top = [t for t in _l[:OURS_ICFG['FI_K']] if _f1 is not None and _form2(t) == _f1]
        if len(_top) < 2:
            _out.append(_s); continue
        _nm += 1
        _spec = [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                      adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == 'positive' else -1)
                 for r in _g.itertuples()][:16]
        _fx = frag_rescore.score_molecule(_spec, _top, cache=_icache)
        _fd = [(float(v), t) for t, v in zip(_top, _fx) if v is not None and np.isfinite(v)]
        if not _fd:
            _out.append(_s); continue
        _best = max(_fd, key=lambda z: z[0])[1]
        _pos = _l.index(_best)
        if _pos < _slot:
            _nalr += 1; _out.append(_s); continue
        _new = [t for t in _l if t != _best]
        _new.insert(_slot - 1, _best)
        _nins += 1
        _out.append(';'.join(_new[:25]))
    _sub['smiles'] = _out
    _sub.to_csv(_sp, index=False)
    print(f'FINS slot {{_slot}} K {{OURS_ICFG["FI_K"]}}: scored molecules {{_nm}} of {{len(_sub)}} | inserted {{_nins}} | '
          f'already above {{_nalr}} | skipped (time cap) {{_nskip}} | {{time.time()-_it0:.0f}}s', flush=True)
except Exception as _e:
    if not IS_RERUN:                 # commit run: a bug must fail the commit
        raise
    print('FINS FAILED -> submission as is', repr(_e), flush=True)
'''
    cells.append(code_cell(hook_i))

if nl > 0:       # N. after cell F, BEFORE hook C
    NCFG = dict(NL_LAM=nl, NL_K=nlk, NL_COV=nlcov, NL_IONS=bool(nlion))
    hook_n = f'''# ==== ours (yasunorim; inserted by enveda-casmi26/kernels/e3/build.py -- every base cell is unchanged) ====
# neutral-loss substructure evidence (ours_nl): a peak at precursor m/z - L (hexose, glucuronic acid, SO3, H3PO4, CO2,
# NH3, CH3, ...) says the molecule carries the group. The final top NL_K are re-ranked by z(-log rank) + NL_LAM *
# (naive-Bayes log likelihood of the seen / unseen losses given each candidate's groups, raw nats; rates measured on train).
# Molecules with no diagnostic loss, or whose candidates all agree on every group, are unchanged.
OURS_NCFG = {NCFG!r}
_ours_dir = '/kaggle/working/ours_mods'
os.makedirs(_ours_dir, exist_ok=True)
with open(os.path.join(_ours_dir, 'ours_nl.py'), 'w') as _fh:
    _fh.write({open(os.path.join(HERE, "ours_nl.py"), encoding="utf-8").read()!r})
if _ours_dir not in sys.path:
    sys.path.insert(0, _ours_dir)
try:
    _nt0 = time.time()
    if 'te' not in globals():
        print('NL: te missing -> nothing done', flush=True)
    else:
        import ours_nl
        ours_nl.USE_IONS = OURS_NCFG.get('NL_IONS', False)
        _ice_all = ICE_SCORES if 'ICE_SCORES' in globals() and isinstance(ICE_SCORES, dict) else {{}}
        def _n_cov(m_):              # True: the molecule is skipped
            if OURS_NCFG['NL_COV'] != 'blind':
                return False
            _i = _ice_all.get(str(m_), {{}})
            _i = _i if isinstance(_i, dict) and any(v is not None for v in _i.values()) else {{}}
            return ('gl_of' in globals() and bool(gl_of(str(m_)))) or bool(_i)
        _nspec = {{}}
        for _r in te.itertuples():
            _nspec.setdefault(str(_r.molecule_id), []).append(
                (np.column_stack([np.asarray(_r.ms2_mzs, np.float64), np.asarray(_r.ms2_normalized_intensities, np.float64)]),
                 float(_r.precursor_mz)))
        _sp = '/kaggle/working/submission.csv'
        _sub = pd.read_csv(_sp, dtype=str, keep_default_na=False)
        _out, _nseen, _nt, _n1, _nrule = [], 0, 0, 0, {{}}
        for _m, _s in zip(_sub.molecule_id, _sub.smiles):
            _l = [x for x in str(_s).split(';') if x]
            if _n_cov(_m) or len(_l) < 2 or not _nspec.get(str(_m)):
                _out.append(_s); continue
            _seen = ours_nl.observed(_nspec[str(_m)])
            if not any(_seen.values()):
                _out.append(_s); continue
            _nseen += 1
            for _k, _v in _seen.items():
                _nrule[_k] = _nrule.get(_k, 0) + int(_v)
            _top = _l[:OURS_NCFG['NL_K']]
            _o = ours_nl.rerank(_top, ours_nl.score_list(_top, _seen), OURS_NCFG['NL_LAM'])
            _new = [_top[i] for i in _o] + _l[len(_top):]
            _nt += _new != _l; _n1 += _new[0] != _l[0]
            _out.append(';'.join(_new))
        _sub['smiles'] = _out
        _sub.to_csv(_sp, index=False)
        print(f'NL lam {{OURS_NCFG["NL_LAM"]}} K {{OURS_NCFG["NL_K"]}} cov {{OURS_NCFG["NL_COV"]}} ions {{ours_nl.USE_IONS}}: molecules with a diagnostic loss '
              f'{{_nseen}} of {{len(_sub)}} | order changed {{_nt}} | top-1 changed {{_n1}} | rules {{ {{k: v for k, v in _nrule.items() if v}} }} | '
              f'{{time.time()-_nt0:.0f}}s', flush=True)
except Exception as _e:
    if not IS_RERUN:                 # commit run: a bug must fail the commit
        raise
    print('NL FAILED -> submission as is', repr(_e), flush=True)
'''
    cells.append(code_cell(hook_n))

if fp2gap > 0:   # P. after cell N, BEFORE hook C
    PCFG = dict(FP2_LAM=fp2gap, FP2_K=fp2k)
    hook_p = f'''# ==== ours (yasunorim; inserted by enveda-casmi26/kernels/e3/build.py -- every base cell is unchanged) ====
# our fp2 where the forward models are blind: molecules with no GLACIER and no ICEBERG score get their final top FP2_K
# re-ranked by z(-log rank) + FP2_LAM * z(log likelihood of the candidate's Morgan r2 bits under our fp2 nets' bit
# probabilities; ours_fp2.py, nets = the fp2.pt files of the kernel sources). Any error on the rerun: submission as is.
OURS_PCFG = {PCFG!r}
_ours_dir = '/kaggle/working/ours_mods'
os.makedirs(_ours_dir, exist_ok=True)
with open(os.path.join(_ours_dir, 'ours_fp2.py'), 'w') as _fh:
    _fh.write({open(os.path.join(E1, "ours_fp2.py"), encoding="utf-8").read()!r})
if _ours_dir not in sys.path:
    sys.path.insert(0, _ours_dir)
try:
    _pt0 = time.time()
    if 'te' not in globals() or 'ICE_SCORES' not in globals() or 'gl_of' not in globals():
        print('FP2GAP: te / ICE_SCORES / gl_of missing -> nothing done', flush=True)
    else:
        import ours_fp2
        _ice_all = ICE_SCORES if isinstance(ICE_SCORES, dict) else {{}}
        def _p_cov(m_):
            _i = _ice_all.get(str(m_), {{}})
            _i = _i if isinstance(_i, dict) and any(v is not None for v in _i.values()) else {{}}
            return bool(gl_of(str(m_))) or bool(_i)
        _sp = '/kaggle/working/submission.csv'
        _sub = pd.read_csv(_sp, dtype=str, keep_default_na=False)
        _lists = {{}}
        for _m, _s in zip(_sub.molecule_id, _sub.smiles):
            _l = [x for x in str(_s).split(';') if x]
            if not _p_cov(_m) and len(_l) >= 2:
                _lists[str(_m)] = _l[:OURS_PCFG['FP2_K']]
        _te2 = te.copy(); _te2['molecule_id'] = _te2['molecule_id'].astype(str)
        _te2 = _te2[_te2.molecule_id.isin(_lists)]
        _ll = ours_fp2.score(_te2, _lists, log=print) if _lists else {{}}
        _out, _nt, _n1 = [], 0, 0
        for _m, _s in zip(_sub.molecule_id, _sub.smiles):
            _d = _ll.get(str(_m))
            if not _d:
                _out.append(_s); continue
            _l = [x for x in str(_s).split(';') if x]
            _top = _lists[str(_m)]
            _v = np.array([_d.get(x) if _d.get(x) is not None else np.nan for x in _top], np.float64)
            if np.all(np.isnan(_v)):
                _out.append(_s); continue
            _v = np.where(np.isnan(_v), np.nanmin(_v), _v)
            _r = -np.log(np.arange(1, len(_top) + 1, dtype=np.float64))
            _z = lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.zeros_like(x)
            _o = list(np.argsort(-(_z(_r) + OURS_PCFG['FP2_LAM'] * _z(_v)), kind='stable'))
            _new = [_top[i] for i in _o] + _l[len(_top):]
            _nt += 1; _n1 += _new[0] != _l[0]
            _out.append(';'.join(_new))
        _sub['smiles'] = _out
        _sub.to_csv(_sp, index=False)
        print(f'FP2GAP lam {{OURS_PCFG["FP2_LAM"]}} K {{OURS_PCFG["FP2_K"]}}: blind molecules {{len(_lists)}} of {{len(_sub)}} | '
              f're-ranked {{_nt}} | top-1 changed {{_n1}} | {{time.time()-_pt0:.0f}}s', flush=True)
except Exception as _e:
    if not IS_RERUN:                 # commit run: a bug must fail the commit
        raise
    print('FP2GAP FAILED -> submission as is', repr(_e), flush=True)
'''
    cells.append(code_cell(hook_p))

if iceb > 0:     # the base's ICEBERG budget on the rerun (on the CPU 5400 s scores only part of the molecules)
    _ib = next(i for i, c in enumerate(cells) if c["cell_type"] == "code"
                                            and "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 5400, True" in src(i))
    cells[_ib]["source"] = src(_ib).replace("TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 5400, True",
                                            f"TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, {iceb}, True   # ours: ICE budget (base 5400)")

if npl > 0:      # Q. after cells F / N / P, BEFORE hook C
    QCFG = dict(NP_LAM=npl, NP_K=npk, NP_COV=npcov)
    hook_q = f'''# ==== ours (yasunorim; inserted by enveda-casmi26/kernels/e3/build.py -- every base cell is unchanged) ====
# natural-product-likeness prior (ours_np): the final top NP_K of each molecule are re-ranked by
# z(-log rank) + NP_LAM * z(Ertl NP-likeness score; Ertl et al. 2008, model read from the installed RDKit Contrib/NP_Score).
# The hidden test is Enveda's natural products; the candidate windows mix them with synthetic isomers.
OURS_QCFG = {QCFG!r}
_ours_dir = '/kaggle/working/ours_mods'
os.makedirs(_ours_dir, exist_ok=True)
with open(os.path.join(_ours_dir, 'ours_np.py'), 'w') as _fh:
    _fh.write({open(os.path.join(HERE, "ours_np.py"), encoding="utf-8").read()!r})
if _ours_dir not in sys.path:
    sys.path.insert(0, _ours_dir)
try:
    _qt0 = time.time()
    import ours_np
    ours_np.load(log=print)
    _ice_all = ICE_SCORES if 'ICE_SCORES' in globals() and isinstance(ICE_SCORES, dict) else {{}}
    def _q_cov(m_):                  # True: the molecule is skipped
        if OURS_QCFG['NP_COV'] != 'blind':
            return False
        _i = _ice_all.get(str(m_), {{}})
        _i = _i if isinstance(_i, dict) and any(v is not None for v in _i.values()) else {{}}
        return ('gl_of' in globals() and bool(gl_of(str(m_)))) or bool(_i)
    _sp = '/kaggle/working/submission.csv'
    _sub = pd.read_csv(_sp, dtype=str, keep_default_na=False)
    _out, _nt, _nc, _n1 = [], 0, 0, 0
    for _m, _s in zip(_sub.molecule_id, _sub.smiles):
        _l = [x for x in str(_s).split(';') if x]
        if _q_cov(_m) or len(_l) < 2:
            _out.append(_s); continue
        _top = _l[:OURS_QCFG['NP_K']]
        _o = ours_np.rerank(_top, OURS_QCFG['NP_LAM'])
        _new = [_top[i] for i in _o] + _l[len(_top):]
        _nt += 1; _nc += _new != _l; _n1 += _new[0] != _l[0]
        _out.append(';'.join(_new))
    _sub['smiles'] = _out
    _sub.to_csv(_sp, index=False)
    print(f'NP lam {{OURS_QCFG["NP_LAM"]}} K {{OURS_QCFG["NP_K"]}} cov {{OURS_QCFG["NP_COV"]}}: re-ranked {{_nt}} of {{len(_sub)}} | '
          f'order changed {{_nc}} | top-1 changed {{_n1}} | {{time.time()-_qt0:.0f}}s', flush=True)
except Exception as _e:
    if not IS_RERUN:                 # commit run: a bug must fail the commit
        raise
    print('NP FAILED -> submission as is', repr(_e), flush=True)
'''
    cells.append(code_cell(hook_q))

if txl > 0:      # T. after cell Q (the NP prior), BEFORE hook C: anchors are the final rank-1 structures
    TCFG = dict(TX_LAM=txl, TX_K=txk, TX_COV=txcov)
    hook_t = f'''# ==== ours (yasunorim; inserted by enveda-casmi26/kernels/e3/build.py -- every base cell is unchanged) ====
# test-set chemotype prior (ours_tx): natural products come in families. Each molecule's final top TX_K are re-ranked
# by z(-log rank) + TX_LAM * z(mean top-3 Morgan r2 Tanimoto to the OTHER test molecules' rank-1 structures).
# The anchors are the rank-1 structures before this cell (one pass, no iteration).
# TX_COV 'blind': only the molecules neither GLACIER nor ICEBERG scored are re-ranked, and only the scored molecules
# (whose rank-1 the forward models back) serve as anchors.
OURS_TCFG = {TCFG!r}
_ours_dir = '/kaggle/working/ours_mods'
os.makedirs(_ours_dir, exist_ok=True)
with open(os.path.join(_ours_dir, 'ours_tx.py'), 'w') as _fh:
    _fh.write({open(os.path.join(HERE, "ours_tx.py"), encoding="utf-8").read()!r})
if _ours_dir not in sys.path:
    sys.path.insert(0, _ours_dir)
try:
    _tt0 = time.time()
    import ours_tx
    _sp = '/kaggle/working/submission.csv'
    _sub = pd.read_csv(_sp, dtype=str, keep_default_na=False)
    _lists = {{str(_m): [x for x in str(_s).split(';') if x] for _m, _s in zip(_sub.molecule_id, _sub.smiles)}}
    _ice_all = ICE_SCORES if 'ICE_SCORES' in globals() and isinstance(ICE_SCORES, dict) else {{}}
    def _t_blind(m_):
        _i = _ice_all.get(str(m_), {{}})
        _i = _i if isinstance(_i, dict) and any(v is not None for v in _i.values()) else {{}}
        return not (('gl_of' in globals() and bool(gl_of(str(m_)))) or bool(_i))
    _bl = {{k for k in _lists if _t_blind(k)}} if OURS_TCFG['TX_COV'] == 'blind' else set()
    _anc = ours_tx.anchors({{k: v for k, v in _lists.items() if k not in _bl}})
    _out, _nt, _nc, _n1 = [], 0, 0, 0
    for _m, _s in zip(_sub.molecule_id, _sub.smiles):
        _l = _lists[str(_m)]
        if len(_l) < 2 or (OURS_TCFG['TX_COV'] == 'blind' and str(_m) not in _bl):
            _out.append(_s); continue
        _top = _l[:OURS_TCFG['TX_K']]
        _o = ours_tx.rerank(ours_tx.score_list(str(_m), _top, _anc), OURS_TCFG['TX_LAM'])
        _new = [_top[i] for i in _o] + _l[len(_top):]
        _nt += 1; _nc += _new != _l; _n1 += _new[0] != _l[0]
        _out.append(';'.join(_new))
    _sub['smiles'] = _out
    _sub.to_csv(_sp, index=False)
    print(f'TX lam {{OURS_TCFG["TX_LAM"]}} K {{OURS_TCFG["TX_K"]}} cov {{OURS_TCFG["TX_COV"]}} (blind {{len(_bl)}}): anchors {{len(_anc)}} | re-ranked {{_nt}} of {{len(_sub)}} | '
          f'order changed {{_nc}} | top-1 changed {{_n1}} | {{time.time()-_tt0:.0f}}s', flush=True)
except Exception as _e:
    if not IS_RERUN:                 # commit run: a bug must fail the commit
        raise
    print('TX FAILED -> submission as is', repr(_e), flush=True)
'''
    cells.append(code_cell(hook_t))

if denovo > 0:
    # ---- embedded modules: ours_fp2 / ours_denovo (e1fuse, ours) + pc_join (0.420 notebook, Apache-2.0)
    nb420 = json.load(open(os.path.join(E1, "base_0420.ipynb"), encoding="utf-8"))
    ns = {}
    exec("".join(nb420["cells"][2]["source"]), ns)
    embed = {"pc_join.py": ns["EMBED"]["pc_join.py"],
             "ours_fp2.py": open(os.path.join(E1, "ours_fp2.py"), encoding="utf-8").read(),
             "ours_denovo.py": open(os.path.join(E1, "ours_denovo.py"), encoding="utf-8").read()}
    if pcl > 0:
        embed["ours_pcl.py"] = open(os.path.join(E1, "ours_pcl.py"), encoding="utf-8").read()
    cfg = dict(DENOVO_N=denovo, DENOVO_S=dns, DENOVO_MIN=dnmin, DENOVO_FORMS=dnforms, DENOVO_MAX_LIB=dnlib or None,
               DENOVO_SLOTS=dnslot, DENOVO_SLOT_LIB=dnslotlib,
               DENOVO_SRC=dnsrc, PCL=pcl, PCL_CAP=pclcap, DENOVO_NOINJ=dnnoinj, VERSION=f"ours-{name}")
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
    ours_denovo.NO_INJECT = bool(OURS_CFG.get('DENOVO_NOINJ'))
    if OURS_CFG.get('PCL'):          # ours_pcl: PubChemLite structures instead of dn1 samples (fp2 nets only)
        import ours_pcl
        ours_pcl.install(ours_denovo, log=print)
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
    if pclcap:      # D. rank 1 of submission.csv only for candidates engine 1 had before the injection (ours, 10-20)
        hook_d = """# ==== ours (yasunorim; inserted by enveda-casmi26/kernels/e3/build.py): injected rows may not hold rank 1 ====
# e3v15pcl-1 (PubChemLite into engine 1, both engines vote) lost 0.022 on the LB: wrong injected structures took rank 1.
# Here an injected row at rank 1 swaps with the best row engine 1 already had (pass 1); the injected row keeps rank 2.
if DENOVO_ON and OURS_CFG.get('PCL_CAP'):
    try:
        from rdkit import Chem as _OC
        def _oik(x):
            m_ = _OC.MolFromSmiles(x)
            return _OC.MolToInchiKey(m_)[:14] if m_ is not None else x
        _sp = '/kaggle/working/submission.csv'
        _sub = pd.read_csv(_sp)
        _as = {str(k): v for k, v in ours_denovo.ADDED_SMI.items()}
        _out, _nc = [], 0
        for _m, _s in zip(_sub.molecule_id, _sub.smiles.fillna('')):
            _lm, _dn = _as.get(str(_m), (1.0, []))
            _l = [x for x in str(_s).split(';') if x]
            if _dn and _l:
                _dk = {_oik(x) for x in _dn}
                if _oik(_l[0]) in _dk:
                    _j = next((j for j, x in enumerate(_l) if _oik(x) not in _dk), None)
                    if _j is not None:
                        _l.insert(0, _l.pop(_j)); _nc += 1
            _out.append(';'.join(_l[:25]))
        _sub['smiles'] = _out
        _sub.to_csv(_sp, index=False)
        print('PCL CAP: injected rank-1 rows moved down in', _nc, 'molecules of', len(_sub), flush=True)
    except Exception as _e:
        if not IS_RERUN:
            raise
        print('PCL CAP FAILED -> submission as is', repr(_e), flush=True)
"""
        cells.append(code_cell(hook_d))
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

if bx:           # X. port (not ours): ahmedberatozer casmi26-v19-bioext cell 12 block, verbatim, before the training structures
    _ic = find_cell("\'\'\'Our engine (two-ranker + BIO + AFIX) in a subprocess")
    _anc = 'tr = pd.DataFrame({"ik": L["ik"], "smi": L["smi"]}).dropna().drop_duplicates("ik")\\n'
    _c = src(_ic)
    assert _c.count(_anc) == 1, "bx anchor"
    cells[_ic]["source"] = _c.replace(_anc, open(os.path.join(HERE, "bx_block.txt"), encoding="utf-8").read() + _anc)

if ownco or ownpc:   # O. engine 2's candidate pool also takes our own COCONUT structures it does not have (ours, 10-21)
    _ic = find_cell("\'\'\'Our engine (two-ranker + BIO + AFIX) in a subprocess")
    _old = 'tr = pd.DataFrame({"ik": L["ik"], "smi": L["smi"]}).dropna().drop_duplicates("ik")\\n'
    _c = src(_ic)
    assert _c.count(_old) == 1, "ownco anchor"
    _new = (_old
            + '    try:  # ours (yasunorim, 10-21): own COCONUT table, structures not yet in the pool, fingerprinted like the training structures\\n'
            + '        _od = [p for p in glob.glob("/kaggle/input/**/coco_meta.pkl", recursive=True) if "own-coconut" in p]\\n'
            + '        if _od:\\n'
            + '            _om = pickle.load(open(_od[0], "rb"))\\n'
            + '            _ot = pd.DataFrame({"ik": np.asarray(_om["keys"], dtype=object), "smi": np.asarray(_om["smiles"], dtype=object)}).dropna()\\n'
            + '            _ot = _ot[~_ot.ik.isin(set(co_keys)) & ~_ot.ik.isin(set(tr.ik))].drop_duplicates("ik")\\n'
            + '            tr = pd.concat([tr, _ot], ignore_index=True)\\n'
            + '            log(f"+ ours: own COCONUT {len(_ot):,} new structures")\\n'
            + '        else:\\n'
            + '            log("own COCONUT not mounted")\\n'
            + '    except Exception as _oe:\\n'
            + '        log(f"own COCONUT skipped: {_oe!r}")\\n')
    if not ownco:
        _new = _old
    if ownpc:
        _new += ('    try:  # ours (yasunorim, 10-21): own PubChem tier, literature-linked structures not yet in the pool\\n'
                 + '        _pp = [p for p in glob.glob("/kaggle/input/**/pubchem_tier.parquet", recursive=True) if "own-pubchem-tier" in p]\\n'
                 + '        if _pp:\\n'
                 + '            _pt = pd.read_parquet(_pp[0], columns=["inchikey14", "smiles", "n_pmid"])\\n'
                 + f'            _pt = _pt[_pt.n_pmid >= {ownpc}].rename(columns={{"inchikey14": "ik", "smiles": "smi"}})[["ik", "smi"]].dropna()\\n'
                 + '            _pt = _pt[~_pt.ik.isin(set(co_keys)) & ~_pt.ik.isin(set(tr.ik))].drop_duplicates("ik")\\n'
                 + '            tr = pd.concat([tr, _pt], ignore_index=True)\\n'
                 + '            log(f"+ ours: own PubChem tier {len(_pt):,} new structures")\\n'
                 + '        else:\\n'
                 + '            log("own PubChem tier not mounted")\\n'
                 + '    except Exception as _pe2:\\n'
                 + '        log(f"own PubChem tier skipped: {_pe2!r}")\\n')
    cells[_ic]["source"] = _c.replace(_old, _new)

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

# Z. last cell (every build): drop the ICEBERG site tree from /kaggle/working. It holds > 500 files, and the kernel
# output listing is paged at 500, so the workflow's `kaggle kernels output` never reached submission.csv
# (e3mmp100c-2, 10-14). Only scratch directories go; submission.csv and the audit files stay.
cells.append(code_cell("""# ==== ours (yasunorim): drop scratch trees from /kaggle/working (the output listing is paged at 500 files) ====
import os as _os, shutil as _sh
for _d in ('ice_site', 'gl_site', '__pycache__', 'ours_mods'):
    _p = _os.path.join('/kaggle/working', _d)
    if _os.path.isdir(_p):
        _sh.rmtree(_p, ignore_errors=True)
_n = sum(len(f) for _, _, f in _os.walk('/kaggle/working'))
print('files left in /kaggle/working', _n, '| submission.csv', _os.path.isfile('/kaggle/working/submission.csv'), flush=True)
"""))
cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                 f"# casmi26 {name} (yasunorim)\n\n"
                 + ("Base: [CASMI26 E006 Window 40 ICE Fallback True (public LB 0.440)]"
                    "(https://www.kaggle.com/code/canqiang/casmi26-e006-window-40-ice-fallback-true) by canqiang, version 1, "
                    "Apache-2.0, and everything it credits (cells below, unchanged). " if base == "e006" else
                    "Base: [Forensics Lab: Apex Synthesis (public LB 0.451 as titled)]"
                    "(https://www.kaggle.com/code/saddamaditya/0-451-sota-forensics-lab-apex-synthesis) by saddamaditya, "
                    "version 1 (E006 + PubChem promotion rel>150 + engine-2 rank-2 + fragment gap / group gap), Apache-2.0, "
                    "and everything it credits (cells below, unchanged). " if base == "a451" else
                    "Base: [casmi26-v15-n1a](https://www.kaggle.com/code/ahmedberatozer/casmi26-v15-n1a) by Ahmed Berat Ozer, "
                    "version 1 (its v14, public LB 0.455 = E006 lineage + PubChemLite CC0 candidates in engine 2, + N1A GLACIER proxies "
                    "from seyitkaangunes), Apache-2.0 lineage, and everything it credits (cells below, unchanged). ")
                 + ("Ported (not ours): the group-gap fragment step of saddamaditya's Apex Synthesis (public 0.451, Apache-2.0): "
                    "the v7c and E005-E007 cells are taken from it (fragment evidence also for same-formula groups no simulator scored). " if gg else "")
                 + ("Ours: the base's fragment scorer (frag_rescore) scores with our learned peak credits (bucket = cleavage "
                    "level x hydrogen shift of the cheapest explanation, linear and sqrt intensity; listwise fit on 4,432 "
                    "train isomer panels; held out MRR 0.723 -> 0.748 on enveda-180, 0.567 -> 0.584 on np-examples). " if lfrag else "")
                 + (f"Ours: the base's fragment scorer (frag_rescore) also re-ranks molecules GLACIER or ICEBERG DID score: "
                    f"final top {bfragk}, inside same-formula groups, z(-log rank) + {bfrag} z(frag_rescore). " if bfrag > 0 else "")
                 + (f"VALIDATION RUN (val={val}, valsrc={valsrc}): the test is replaced by {val} molecules held out of train; no submission. " if val > 0 else "")
                 + (f"Ours (fins={fins}): ranks 1-{fins-1} kept; the fragment scorer's top pick among the final top {bfragk} with rank 1's formula moves to rank {fins}. " if fins > 0 else "")
                 + (f"Ours (pcl={pcl}): PubChemLite CC0 structures (the base gives them to engine 2 only) also enter engine 1: "
                    f"after engine 1's pass 1, PubChemLite structures with its top formulas (+-5 ppm, RDKit formula check), not "
                    f"already candidates, ordered by our fp2 log likelihood, top {pcl} injected through pc_join (huseyinemreaksoy, "
                    f"v4n fusion 0.420) into a second engine-1 pass, so both engines can vote for them. " if pcl > 0 else "")
                 + ("Ours (ownco=1): engine 2's candidate pool also takes the structures of our own COCONUT table "
                    "(yasunorim/casmi26-own-coconut-candidates, built from the official COCONUT dump; 38k structures more than the "
                    "public COCONUT table) that the pool does not have, fingerprinted like the training structures. " if ownco else "")
                 + ("Port (not ours): ahmedberatozer casmi26-v19-bioext's BX block (engine 2's pool also takes PubChem structures from "
                    "biological depositors, output of ahmedberatozer/casmi26-bioext-build), verbatim. " if bx else "")
                 + (f"Ours (ownpc={ownpc}): engine 2's candidate pool also takes the structures of our own PubChem tier "
                    f"(yasunorim/casmi26-own-pubchem-tier, built from the PubChem FTP CID-PMID / CID-SMILES / InChIKey files) with at least "
                    f"{ownpc} PubMed links that the pool does not have, fingerprinted like the training structures. " if ownpc else "")
                 + ("Ours (pclcap=1): an injected row at rank 1 of the final list swaps with the best row engine 1 already had "
                    "before the injection (the injected row keeps rank 2). " if pclcap else "")
                 + (f"Ours (dnnoinj=1): the proposed rows are not injected into the engines; only the slot cell uses them (final "
                    f"slots {dnslot} where lib_max < {dnslotlib}). " if dnnoinj else "")
                 + (f"Ours: formula-constrained de novo candidates (our dn1 model, ordered by our fp2) injected into the "
                    f"main engine through pc_join (huseyinemreaksoy, v4n fusion 0.420) in a second engine pass; max "
                    f"{denovo} rows per molecule, {dnmin:.0f} min cap. Two inserted cells are marked 'ours'. "
                    if denovo > 0 and not pcl else "")
                 + (f"Ours: MMP class-3 generator (replacement rules mined from train, applied to the spectral analogs; max "
                    f"{mmp} products per molecule) inside the main engine's generator. " if mmp > 0 else "")
                 + (f"Ours: where neither GLACIER nor ICEBERG scored a molecule, its final top {fgapk} are re-ranked by "
                    f"z(-log rank) + {fgap} z(our 1-2 bond-cut fragment explanation of the spectra)"
                    + (" (fgapcov=ice: every molecule without an ICEBERG score, GLACIER-scored ones included)" if fgapcov == "ice" else "")
                    + (" minus its chance rate (mean over the spectra shifted by +-3/5/9 Da, our decoy correction)" if fgapdec else "")
                    + ". " if fgap > 0 else "")
                 + (f"Ours: neutral-loss substructure evidence (diagnostic losses such as hexose, glucuronic acid, SO3, CO2 "
                    f"from the precursor; rates measured on train) re-ranks the final top {nlk} by z(-log rank) + {nl} "
                    f"x (naive-Bayes log likelihood, nats){' (only where neither GLACIER nor ICEBERG scored)' if nlcov == 'blind' else ''}"
                    f"{'; diagnostic product ions (phosphocholine, benzoyl, caffeoyl, immonium ions, HSO4-, ...) join the evidence' if nlion else ''}. "
                    if nl > 0 else "")
                 + (f"Ours: where neither GLACIER nor ICEBERG scored a molecule, its final top {fp2k} are re-ranked by "
                    f"z(-log rank) + {fp2gap} z(our fp2 Morgan-bit log likelihood). " if fp2gap > 0 else "")
                 + (f"Ours: natural-product-likeness prior: the final top {npk} are re-ranked by z(-log rank) + {npl} "
                    f"z(Ertl NP-likeness, RDKit Contrib/NP_Score model){' (only where neither GLACIER nor ICEBERG scored)' if npcov == 'blind' else ''}. "
                    if npl > 0 else "")
                 + (f"Ours: test-set chemotype prior: the final top {txk} are re-ranked by z(-log rank) + {txl} "
                    f"z(mean top-3 Tanimoto to the other test molecules' rank-1 structures)"
                    f"{' (only where neither GLACIER nor ICEBERG scored; anchors = the scored molecules)' if txcov == 'blind' else ''}. " if txl > 0 else "")
                 + (f"Ours (CPU): the base's ICEBERG budget raised from 5400 s to {iceb} s so the CPU scores more molecules. " if iceb > 0 else "")
                 + ("This build adds nothing (reference run). " if denovo == 0 and mmp == 0 and fgap == 0 and nl == 0 and fp2gap == 0 and npl == 0 and txl == 0 and iceb == 0 and bfrag == 0 and not lfrag and fins == 0 else "")
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
                            "ahmedberatozer/casmi26-v4b-models",
                            "prvsiyan/chebi-lipidmaps-casmi26" if base == "e006" else "sergiomalv/casmi-chebi-lmsd-structures",
                            "prvsiyan/coconut-casmi26-candidates"] + (["yasunorim/casmi26-own-coconut-candidates"] if ownco else []) + (["yasunorim/casmi26-own-pubchem-tier"] if ownpc else [])
                           + (["franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints", "metric/rdkit-2026-3-3-wheel"] if base == "v15" else
                              ["hengck23/hengck23-dreams-enveda-casmi26", "metric/rdkit-2026-3-3-wheel", "eveliaveldrine/rdkit-wheels"]),
        "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
        "kernel_sources": ["yasunorim/casmi26-fp2-peak-transformer", "yasunorim/casmi26-fp2L3"] + (["ahmedberatozer/casmi26-bioext-build"] if bx else []) + ([dnsrc] if denovo > 0 and not pcl else []),
        "model_sources": []}
if cpu:
    meta.update(enable_gpu="false")
    meta.pop("machine_shape")
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("wrote main.ipynb", meta["id"], "denovo", denovo, "dns", dns, "dnmin", dnmin, "dnforms", dnforms, "dnlib", dnlib,
      "dnsrc", dnsrc if denovo > 0 else "-", "mmp", mmp, "fgap", fgap, "fgapk", fgapk, "nl", nl, nlcov, "nlion", nlion, "base", base, "bfrag", bfrag, "lfrag", lfrag, "fins", fins, "pcfix", pcfix, "pcvote", pcvote, "latecut", latecut, "pcguard", pcguard, "pclcap", pclcap, "bx", bx, "ownco", ownco, "ownpc", ownpc, "dnnoinj", dnnoinj, "dnslot", dnslot, "val", val, "fp2gap", fp2gap, "cpu", cpu)
