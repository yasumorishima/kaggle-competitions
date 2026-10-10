"""Build kernels/e1fuse/main.ipynb = the public 0.420 notebook + our changes.

Base: huseyinemreaksoy/casmi26-v4n-fusion-pubchem-on-public-0-421 (version 1, LB 0.420, Apache-2.0; itself built on
ahmedberatozer's v4 engine, prvsiyan's engine 2, ICEBERG/GLACIER, the PubChem tier and the popularity prior), copied
unchanged as base_0420.ipynb. Our changes:
  1. ours_fp2.py: our fp2 (spectrum -> Morgan bits) scores every candidate (base top-N, engine-2, PubChem lists).
  2. fusion_core: fp2 is one more re-scoring term inside same-formula groups, jointly with ICEBERG / GLACIER /
     fragment: z(ranker) + ICE_LAM z(ice) + GL_LAM z(gl) + FP2_LAM z(fp2). It also covers the adducts and negative-mode
     spectra the forward models cannot score. FP2_LIB_OFF keeps library hits untouched.
  3. validation mode: the fusion is replayed for every FP2_LAMS value and the MRR of each is logged.

    python build.py NAME FP2_LAM [val] [only=fp2L3,fp2-peak] [pc=PC_FP2_LAM]
  4. (pc=) fp2 also orders the PubChem-only proposals: z(f.z) + PC_FP2_LAM z(fp2 ll) inside the pass-1 set; the
     gate keeps the raw best f.z.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
name, lam = sys.argv[1], float(sys.argv[2])
pclam = next((float(a.split("=", 1)[1]) for a in sys.argv[3:] if a.startswith("pc=")), 0.0)
val = "val" in sys.argv[3:]
only = next((a.split("=", 1)[1].split(",") for a in sys.argv[3:] if a.startswith("only=")), None)
mmp = next((int(a.split("=", 1)[1]) for a in sys.argv[3:] if a.startswith("mmp=")), 0)
mmpc = next((int(a.split("=", 1)[1]) for a in sys.argv[3:] if a.startswith("mmpc=")), 2)   # min rule count
mmpfp = next((int(a.split("=", 1)[1]) for a in sys.argv[3:] if a.startswith("mmpfp=")), 0)  # FP-retrieved parents
c3full = "c3full" in sys.argv[3:]   # class-3 simulation through ICE / GLACIER / fusion (not only the base lists)
c3val = "c3val" in sys.argv[3:] or c3full
c3keep = "c3keep" in sys.argv[3:]   # with c3val: the truths stay in the pool (harm check on in-pool molecules)
c3par = "c3par" in sys.argv[3:]     # with c3val: parent-retrieval study (ours_mmp.study) per molecule
c3ord = "c3ord" in sys.argv[3:]     # with c3val: where the truth falls among the valid MMP products under each order
c3hard = next((float(a.split("=", 1)[1]) for a in sys.argv[3:] if a.startswith("c3hard=")), 0.0)   # with c3val: MMP parents farther than this Tanimoto from the truth
iceb = next((int(a.split("=", 1)[1]) for a in sys.argv[3:] if a.startswith("ice=")), 0)   # ICEBERG time budget (s) override
valn = next((int(a.split("=", 1)[1]) for a in sys.argv[3:] if a.startswith("valn=")), 0)   # validation molecules override
mmp2 = next((int(a.split("=", 1)[1]) for a in sys.argv[3:] if a.startswith("mmp2=")), 0)   # two-step intermediates per parent
mmpord = next((a.split("=", 1)[1] for a in sys.argv[3:] if a.startswith("mmpord=")), "count")   # count | fp | mix
nb = json.load(open(os.path.join(HERE, "base_0420.ipynb"), encoding="utf-8"))
cells = nb["cells"]
src = lambda i: "".join(cells[i]["source"])  # noqa: E731


def setsrc(i, s):
    cells[i]["source"] = s


def swap(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old))
    return s.replace(old, new)


# ---- 1. embedded sources: our fp2 module + the patched fusion_core
c2 = src(2)
ns = {}
exec(c2, ns)
E = ns["EMBED"]
E["ours_fp2.py"] = open(os.path.join(HERE, "ours_fp2.py"), encoding="utf-8").read()
E["ours_mmp.py"] = open(os.path.join(HERE, "ours_mmp.py"), encoding="utf-8").read()
fc = E["fusion_core.py"]
fc = swap(fc, "                PC_FWD_T1=None, PC_FWD_N=25)",
          "                PC_FWD_T1=None, PC_FWD_N=25,\n"
          "                FP2_LAM=0.0, FP2_LIB_OFF=True)          # ours: fp2 re-scoring term (0 = off)")
fc = swap(fc, "                     frag_scores=None):\n",
          "                     frag_scores=None, fp2_scores=None):\n")
fc = swap(fc, "    extra = {}                                   # FILL_25",
          '''    def fp2_of(m, lib_max):
        """ours: fp2 scores of molecule m, or {} when off / library hit with FP2_LIB_OFF."""
        if c['FP2_LAM'] <= 0 or gl_fuse is None:
            return {}
        if c['FP2_LIB_OFF'] and lib_max >= GATE_TAU:
            return {}
        d = (fp2_scores or {}).get(str(m), {})
        return d if covered(d) else {}

    def joint_terms(m, lib_max, ice, gl, il, gll, f2):
        """ours: every re-scorer of the molecule in ONE z-sum (ice, gl, frag as the base notebook uses them, + fp2)."""
        terms, lams = [], []
        if covered(ice):
            terms.append(ice); lams.append(il)
        if covered(gl):
            terms.append(gl); lams.append(gll)
        fr = (frag_scores or {}).get(str(m), {})
        if (c['FRAG_LAM'] > 0 and covered(fr) and (c['FRAG_MODE'] != 'gap' or not covered(ice))
                and not (c['FRAG_LIB_OFF'] and lib_max >= GATE_TAU)):
            terms.append(fr); lams.append(c['FRAG_LAM'])
        terms.append(f2); lams.append(c['FP2_LAM'])
        return terms, lams

    extra = {}                                   # FILL_25''')
fc = swap(fc, '''        base[m] = (smis[:25], keys[:25], lib_max)
        extra[m] = smis[25:]''', '''        f2 = fp2_of(m, lib_max)
        if f2:                                   # ours: joint re-rank with fp2 replaces the ICE/GL/frag order
            try:
                terms, lams = joint_terms(m, lib_max, ice, gl, ICE_LAM, GL_LAM, f2)
                o = gl_fuse.rerank_multi(smis0, keys0, scs0, forms0, terms, lams, top_n=TOPN)
                st = stats.setdefault('fp2', dict(molecules=0, changed_top1=0))
                st['molecules'] += 1; st['changed_top1'] += int(bool(smis) and smis0[o[0]] != smis[0])
                smis, keys = [smis0[i] for i in o], [keys0[i] for i in o]
            except Exception as e:
                log('fp2 rerank failed', m, repr(e))
        base[m] = (smis[:25], keys[:25], lib_max)
        extra[m] = smis[25:]''')
fc = swap(fc, '''        if ice_fuse is not None and p and p.get('pc') and lib_max < LIB_TAU and ice:
            try:
                if not p.get('pc_form'):
                    p['pc_form'] = [formula(x) for x in p['pc']]
                o = ice_fuse.rerank(p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, lam=ICE_LAM, top_n=25)
                o = gl_rerank(o, p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, gl, 25, 'pc_')''',
          '''        if ice_fuse is not None and p and p.get('pc') and lib_max < LIB_TAU and (ice or f2):
            try:
                if not p.get('pc_form'):
                    p['pc_form'] = [formula(x) for x in p['pc']]
                if f2:                           # ours: fp2 joins the PubChem-list re-rank
                    terms, lams = joint_terms(m, lib_max, ice, gl, ICE_LAM, GL_LAM, f2)
                    o = gl_fuse.rerank_multi(p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], terms, lams, top_n=25)
                else:
                    o = ice_fuse.rerank(p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, lam=ICE_LAM, top_n=25)
                    o = gl_rerank(o, p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, gl, 25, 'pc_')''')
fc = swap(fc, '''            stats['fuse']['top1_changed'] += int(bool(vs) and fsm[:1] != vs[:1])''',
          '''            f2 = fp2_of(mid, lib_of.get(mid, 0.0))
            if f2:                               # ours: joint re-rank of the fused list with fp2
                try:
                    terms, lams = joint_terms(mid, lib_of.get(mid, 0.0), ice, gl_of(mid), P_ICE, P_GL, f2)
                    o = gl_fuse.rerank_multi(fsm0, fk0, fsc0, [formula(x) for x in fsm0], terms, lams, top_n=len(fsm0))
                    fsm = [fsm0[i] for i in o]
                except Exception as ex:
                    log('fused fp2 rerank failed', mid, repr(ex))
            stats['fuse']['top1_changed'] += int(bool(vs) and fsm[:1] != vs[:1])''')
E["fusion_core.py"] = fc

# ours (b): fp2 joins the ordering of the PubChem-only proposals (z(f.z) + PC_FP2_LAM z(fp2 ll)) inside the pass-1 set
pc = E["pc/probe_core2.py"]
pc += '''

# ==== ours (yasunorim): fp2 term in the PubChem-only channel; active when CASMI_FP2_LAM > 0. The task carries the
# molecule's fp2 bit probabilities (2048, Morgan r2) as a 4th element. ====
_FP2_LAM = float(os.environ.get('CASMI_FP2_LAM', '0'))
_init_worker_base = init_worker


def init_worker_fp2(pc_dir, bits_path, pool_meta_path, code_dir=None):
    _init_worker_base(pc_dir, bits_path, pool_meta_path, code_dir)
    from rdkit.Chem import rdFingerprintGenerator
    _W['g2'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def probe_one_fp2(task):
    mid, target, z = task[:3]
    p2 = task[3] if len(task) > 3 else None
    t0 = time.time()
    chem = _W['chem']; Chem = _W['Chem']; gen = _W['ecfp4']
    smis = _window(float(target))
    n = len(smis)
    diag = dict(molecule_id=mid, n_window=n, n_pass1=0, n_pass2=0, n_keyed=0, n_in_pool=0, n_listed=0, fz_top=None)
    if n == 0:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    raw_e = _W['raw_e']; z_e = z[_W['sel_e']].astype(np.float32)
    E = np.zeros((n, len(raw_e)), np.float32); ok = np.zeros(n, bool); mols = [None] * n
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        mols[i] = m
        E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]; ok[i] = True
    sc1 = E @ z_e
    sc1[~ok] = -np.inf
    n1 = min(PC_N1, int(ok.sum()))
    diag['n_pass1'] = n1
    if n1 == 0:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    top1 = np.argpartition(-sc1, n1 - 1)[:n1] if n1 < n else np.where(ok)[0]
    bits = _W['bits']
    fz, idx, ll = [], [], []
    lp, lq = (np.log(p2), np.log(1 - p2)) if p2 is not None else (None, None)
    for i in top1:
        fp = chem.raw_fingerprint(smis[i])
        if fp is None:
            continue
        fz.append(float(fp[bits].astype(np.float32) @ z)); idx.append(int(i))
        if lp is not None:
            b = _W['g2'].GetFingerprintAsNumPy(mols[i]).astype(np.float64)
            ll.append(float(b @ lp + (1 - b) @ lq))
    diag['n_pass2'] = len(idx)
    if not idx:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    fz = np.asarray(fz, np.float64); idx = np.asarray(idx)
    keyc = {}

    def key(j):
        if j not in keyc:
            keyc[j] = chem.score_key(smis[idx[j]]); diag['n_keyed'] += 1
        return keyc[j]
    for j in np.argsort(-fz, kind='stable'):        # gate quantity stays the best non-pool structure by raw f.z
        k = key(j)
        if k is not None and k not in _W['pool_keys']:
            diag['fz_top'] = float(fz[j]); break
    sc = (fz - fz.mean()) / (fz.std() + 1e-9)
    if ll:
        ll = np.asarray(ll); sc = sc + _FP2_LAM * (ll - ll.mean()) / (ll.std() + 1e-9)
    order = np.argsort(-sc, kind='stable')
    out, out_s, out_k, seen = [], [], [], set()
    for j in order:
        k = key(j)
        if k is None or k in seen:
            continue
        seen.add(k)
        if k in _W['pool_keys']:
            diag['n_in_pool'] += 1
            continue
        out.append(smis[idx[j]]); out_s.append(float(fz[j])); out_k.append(k)
        if len(out) >= K:
            break
    diag['n_listed'] = len(out)
    diag['secs'] = time.time() - t0
    return mid, out, out_s, out_k, diag


if _FP2_LAM > 0:
    init_worker, probe_one = init_worker_fp2, probe_one_fp2
'''
E["pc/probe_core2.py"] = pc
pr = E["pc/pc_runner.py"]
pr = swap(pr, "    del bank; torch.cuda.empty_cache()\n", '''    if float(os.environ.get('CASMI_FP2_LAM', '0')) > 0:     # ours: fp2 probabilities per molecule (4th task element)
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import ours_fp2
            _paths, _nets = ours_fp2.load(only=[x for x in os.environ.get('CASMI_FP2_ONLY', '').split(',') if x] or None, device=dev)
            _g = dict(list(te.groupby('molecule_id', sort=False)))
            tasks = [(t[0], t[1], t[2], ours_fp2.mol_probs(_nets, _g[t[0]], dev)) for t in tasks]
            print('fp2 probabilities for the PubChem channel', len(tasks), _paths, flush=True)
            del _nets
        except Exception as e:
            print('FP2 IN PUBCHEM CHANNEL FAILED -> f.z only:', repr(e), flush=True)
    del bank; torch.cuda.empty_cache()
''')
E["pc/pc_runner.py"] = pr
setsrc(2, "# embedded sources (base notebook's EMBED + ours_fp2.py; fusion_core.py patched by kernels/e1fuse/build.py)\n"
       "EMBED = " + repr(E) + "\n")

# ---- 2. config: our knobs on top of the base notebook's final CFG.update
c4 = src(4)
upd = {"VERSION": f"ours-{name}", "FP2_LAM": lam, "PC_FP2_LAM": pclam, "FP2_LIB_OFF": True, "FP2_ONLY": only,
       "FP2_LAMS": [0.0, 0.25, 0.5, 1.0, 1.5]}
upd["MMP_N"] = mmp
upd["MMP_MINC"] = mmpc
upd["MMP_FPPAR"] = mmpfp
upd["MMP_ORDER"] = mmpord
upd["MMP_TWO"] = mmp2
if iceb:
    upd["ICE_BUDGET"] = iceb
if val:
    upd.update(VALIDATION=True, FP_BANK="A")
if c3val:       # class-3 simulation: held-out truths leave the pool; only the base lists are built and scored
    upd.update(VALIDATION=True, C3VAL=True, BASE_ONLY=True, VAL_SET="fold0_np", FP_BANK="fold0", VAL_MAX_SPEC=6,
               USE_ENG=False, USE_PC=False, PC_JOIN_N=0, C3KEEP=c3keep, C3PAR=c3par, C3ORD=c3ord, C3HARD=c3hard)
    if c3full:
        upd.update(BASE_ONLY=False, FP2_LAMS=[0.0])
if valn:
    upd["VAL_N"] = valn
setsrc(4, c4 + "\nCFG.update(" + repr(upd) + ")   # ours (kernels/e1fuse/build.py)\n")

# ---- 2a. engine: MMP generator (ours_mmp) and, for c3val, the held-out truths removed from the pool
i14 = next(i for i, c in enumerate(cells) if "".join(c["source"]).startswith("# v1 engine (casmi package"))
c14 = src(i14)
c14 = swap(c14, "import pc_join\n", """if CFG.get('MMP_N', 0) > 0:                  # ours: MMP class-3 generator (before pc_join wraps E.generate)
    try:
        import ours_mmp
        _xik = ()
        if CFG.get('C3VAL') and not IS_RERUN and os.path.exists(os.path.join(STAGE, 'val_labels.csv')):
            from rdkit import Chem as _Ch   # class-3 validation: the held-out truths give no MMP rules
            _xik = set(_Ch.MolToInchiKey(_Ch.MolFromSmiles(s_))[:14] for s_ in pd.read_csv(os.path.join(STAGE, 'val_labels.csv')).smiles
                       if _Ch.MolFromSmiles(s_) is not None)
        ours_mmp.mine(os.path.join(COMP, 'train.parquet'), workers=4, min_count=CFG.get('MMP_MINC', 2), log=log, exclude_ik14=_xik)
        ours_mmp.install(E, max_new=CFG['MMP_N'], n_fp_parent=CFG.get('MMP_FPPAR', 0), order=CFG.get('MMP_ORDER', 'count'),
                         two_step=CFG.get('MMP_TWO', 0))
        log('MMP generator installed, max new per molecule', CFG['MMP_N'])
    except Exception as e:
        if not IS_RERUN:
            raise
        print('MMP FAILED -> base generator only', repr(e))
if CFG.get('C3VAL') and not CFG.get('C3KEEP') and CFG['VALIDATION'] and not IS_RERUN:   # ours: class-3 simulation
    _lab3 = pd.read_csv(os.path.join(STAGE, 'val_labels.csv'))
    C3_HOLD = set(k for k in (chem.score_key(s) for s in _lab3.smiles) if k)
    _drop3 = np.fromiter((k in C3_HOLD for k in P.key), bool, len(P.key))
    _w03 = P.window
    P.window = lambda t, ppm: (lambda w: w[~_drop3[w]])(_w03(t, ppm))
    log('C3VAL: held-out structures removed from the pool:', int(_drop3.sum()), 'of', len(C3_HOLD))
if (CFG.get('C3PAR') or CFG.get('C3ORD')) and CFG.get('MMP_N', 0) > 0:   # ours: parent / product-order studies
    ours_mmp.STUDY = bool(CFG.get('C3PAR'))
    _lab3 = pd.read_csv(os.path.join(STAGE, 'val_labels.csv'))
    ours_mmp.DIAG.update(dict(zip(_lab3.molecule_id, _lab3.smiles)))
    ours_mmp.DIAG_HOLD.update(k for k in (chem.score_key(s) for s in _lab3.smiles) if k)
    log('PARENT STUDY on', len(ours_mmp.DIAG), 'molecules')
if CFG.get('C3HARD') and CFG.get('MMP_N', 0) > 0 and not IS_RERUN:   # ours: "novel truth" simulation
    _lab3 = pd.read_csv(os.path.join(STAGE, 'val_labels.csv'))
    ours_mmp.DIAG.update(dict(zip(_lab3.molecule_id, _lab3.smiles)))
    ours_mmp.STUDY = bool(CFG.get('C3PAR'))
    ours_mmp.HARD_T = float(CFG['C3HARD'])
    log('C3HARD: MMP parents within Tanimoto', ours_mmp.HARD_T, 'of the truth are dropped')
import pc_join
""")
setsrc(i14, c14)
i16 = next(i for i, c in enumerate(cells) if "".join(c["source"]).startswith("# all molecules -> base lists"))
c16 = src(i16)
c16 = swap(c16, "for gi, (mid, sub) in enumerate(mols):\n", """MMPK = {}
for gi, (mid, sub) in enumerate(mols):
    if CFG.get('MMP_N', 0) > 0:
        ours_mmp.GEN_KEYS.clear(); ours_mmp.CUR['mid'] = mid
""")
c16 = swap(c16, "    BASE[mid] = [smis, keys, lib_max, scs, forms, pids]\n", """    BASE[mid] = [smis, keys, lib_max, scs, forms, pids]
    MMPK[mid] = list(ours_mmp.GEN_KEYS.get('last', ())) if CFG.get('MMP_N', 0) > 0 else []
""")
c16 += """
if CFG.get('MMP_N', 0) > 0:
    log('MMP stats', ours_mmp.STATS)
    if CFG.get('C3PAR'):
        ours_mmp.study_summary(log)
    if CFG.get('C3ORD'):
        ours_mmp.order_summary(CFG['MMP_N'], log)
    if CFG.get('C3HARD'):
        ours_mmp.hard_summary(log)
if CFG.get('C3VAL') and os.path.exists(os.path.join(STAGE, 'val_labels.csv')):   # ours: class-3 base-list scores
    _lab3 = pd.read_csv(os.path.join(STAGE, 'val_labels.csv'))
    _tk = {m: chem.score_key(s) for m, s in zip(_lab3.molecule_id, _lab3.smiles)}
    def _rr(ks, t, n=25):
        return next((1.0 / i for i, k in enumerate(ks[:n], 1) if k == t), 0.0)
    _w, _wo, _h60, _h60o, _gen = [], [], [], [], []
    for _m, _t in _tk.items():
        _ks = BASE.get(_m, [[], []])[1]; _mm = set(MMPK.get(_m, []))
        _kso = [k for k in _ks if k not in _mm]
        _w.append(_rr(_ks, _t)); _wo.append(_rr(_kso, _t))
        _h60.append(_t in _ks); _h60o.append(_t in _kso); _gen.append(_t in _mm)
    log(f'C3VAL n={len(_w)} | with MMP: MRR@25 {np.mean(_w):.4f} hit@25 {np.mean(np.array(_w) > 0):.3f} '
        f'top1 {np.mean(np.array(_w) == 1):.3f} hit@TOPN {np.mean(_h60):.3f} | without MMP rows: MRR@25 {np.mean(_wo):.4f} '
        f'hit@25 {np.mean(np.array(_wo) > 0):.3f} hit@TOPN {np.mean(_h60o):.3f} | truth generated by MMP {np.mean(_gen):.3f}')
    json.dump(dict(MMPK=MMPK, TK=_tk), open(os.path.join(STAGE, 'c3val.json'), 'w'))
"""
setsrc(i16, c16)

# ---- 2b. PubChem channel: pass CASMI_FP2_LAM to the runner
i12 = next(i for i, c in enumerate(cells) if "".join(c["source"]).startswith("# PubChem-only channel in its own process"))
c12 = src(i12)
c12 = swap(c12, "if CFG['USE_PC']:\n", """if CFG.get('PC_FP2_LAM', 0) > 0:             # ours: fp2 term in the PubChem-only channel
    _pc_env.update(CASMI_FP2_LAM=str(CFG['PC_FP2_LAM']), CASMI_FP2_ONLY=','.join(CFG['FP2_ONLY'] or []))
if CFG['USE_PC']:
""")
setsrc(i12, c12)

# ---- 3. fp2 scores next to the ICE / GLACIER / frag scores
i18 = next(i for i, c in enumerate(cells) if "".join(c["source"]).startswith("# ICEBERG + GLACIER scoring"))
c18 = src(i18)
c18 = swap(c18, "log('ICE molecules', len(ICE_SCORES)", '''FP2 = {}                                    # ours: fp2 log-likelihood of every candidate list
if CFG['FP2_LAM'] > 0 or CFG['VALIDATION']:
    try:
        import ours_fp2
        _t0 = time.time()
        _cl = {}
        for mid in BASE:
            _e = (ENG.get(str(mid)) or {}).get('smiles', [])[:CFG['FUSE_ENG_K']]
            _cl[mid] = list(dict.fromkeys(list(BASE[mid][0][:TOPN]) + list(_e) + list((PC.get(mid) or {}).get('pc', []))))
        FP2 = ours_fp2.score(te, _cl, only=CFG['FP2_ONLY'], log=log)
        log('fp2 scores', len(FP2), 'molecules', f'{time.time()-_t0:.0f}s')
    except Exception as e:
        FP2 = {}
        print('FP2 FAILED -> off:', repr(e))
dump('fp2', FP2)
log('ICE molecules', len(ICE_SCORES)''')
setsrc(i18, c18)

tpu = "tpu" in sys.argv[3:]   # TPU v5e-8 VM: no GPU quota; ICEBERG / GLACIER on its many CPU cores
if "cpu" in sys.argv[3:] or tpu:   # CPU kernel: ICEBERG / GLACIER fall back to the CPU (time budgets unchanged)
    c18 = src(i18)
    assert c18.count("device='cuda'") == 3
    setsrc(i18, "import os as _os, torch as _t; print('CPU cores', _os.cpu_count(), 'torch threads', _t.get_num_threads(), flush=True)\n"
           + c18.replace("device='cuda'", "device=('cuda' if torch.cuda.is_available() else 'cpu')"))

# ---- 4. fusion: pass fp2; in validation mode replay the fusion for every FP2_LAMS value
i20 = next(i for i, c in enumerate(cells) if "".join(c["source"]).startswith("# fusion_core:"))
c20 = src(i20)
c20 = swap(c20, "    frag_scores=FRAG)\n", "    frag_scores=FRAG, fp2_scores=FP2)\n")
c20 += '''
# ours: validation replay over FP2_LAMS (same stage caches, only the fp2 weight changes)
if os.path.exists(os.path.join(STAGE, 'val_labels.csv')):
    for _lam in CFG['FP2_LAMS']:
        _sub, _st = fusion_core.build_submission(
            BASE, PC, ENG, ICE_SCORES, GL_SCORES, dict(CFG, FP2_LAM=_lam), MOL_ORDER, list(samp.molecule_id),
            ice_fuse=ice_fuse, gl_fuse=gl_fuse, score_key=chem.score_key, formula=formula_of, log=print,
            pool_pop=POOL_POP, frag_scores=FRAG, fp2_scores=FP2)
        _rr = []
        for _m, _s in zip(_sub.molecule_id, _sub.smiles):
            _t = _ik14(_lab.smiles[_m]); _r = 0.0
            for _i, _g in enumerate(_s.split(';'), 1):
                if _ik14(_g) == _t:
                    _r = 1.0 / _i; break
            _rr.append(_r)
        log(f'OURS FP2_LAM={_lam}: MRR@25 = {np.mean(_rr):.4f} | top1 {np.mean(np.array(_rr) == 1):.3f} | '
            f'hit@25 {np.mean(np.array(_rr) > 0):.3f} | fp2 stats {_st.get("fp2")}')
if CFG.get('C3VAL') and os.path.exists(os.path.join(STAGE, 'val_labels.csv')):
    # ours: class-3 after ICE / GLACIER / fusion, with and without the MMP rows (same scores, rows dropped)
    def _drop_mmp(b):
        out = {}
        for _m, _v in b.items():
            _mm = set(MMPK.get(_m, []))
            _keep = [i for i, k in enumerate(_v[1]) if k not in _mm]
            _n = len(_v[1])
            out[_m] = [[x[i] for i in _keep] if hasattr(x, '__len__') and not isinstance(x, str) and len(x) == _n else x
                       for x in _v]
        return out
    for _tag, _b in (('with MMP', BASE), ('without MMP rows', _drop_mmp(BASE))):
        _sub, _st = fusion_core.build_submission(
            _b, PC, ENG, ICE_SCORES, GL_SCORES, dict(CFG, FP2_LAM=0.0), MOL_ORDER, list(samp.molecule_id),
            ice_fuse=ice_fuse, gl_fuse=gl_fuse, score_key=chem.score_key, formula=formula_of, log=print,
            pool_pop=POOL_POP, frag_scores=FRAG, fp2_scores=FP2)
        _rr = []
        for _m, _s in zip(_sub.molecule_id, _sub.smiles):
            _t = _ik14(_lab.smiles[_m]); _r = 0.0
            for _i, _g in enumerate(_s.split(';'), 1):
                if _ik14(_g) == _t:
                    _r = 1.0 / _i; break
            _rr.append(_r)
        log(f'C3FULL {_tag}: n={len(_rr)} MRR@25 = {np.mean(_rr):.4f} | top1 {np.mean(np.array(_rr) == 1):.3f} | '
            f'hit@25 {np.mean(np.array(_rr) > 0):.3f} | ice {_st.get("ice")}')
'''
setsrc(i20, c20)

if c3val and not c3full:   # base lists only: the later stages are skipped
    for i in range(i16 + 1, len(cells)):
        if cells[i]["cell_type"] == "code":
            setsrc(i, "if not CFG.get('BASE_ONLY'):\n    exec(compile(" + repr(src(i)) + ", 'cell', 'exec'))\n")
cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                 "# casmi26 ours (yasunorim)\n\nBase: [CASMI26 | v4n Fusion + PubChem Join (LB 0.420)]"
                 "(https://www.kaggle.com/code/huseyinemreaksoy/casmi26-v4n-fusion-pubchem-on-public-0-421) "
                 "and everything it credits. Ours: an independent spectrum-to-fingerprint model (fp2, trained by us) as a "
                 "re-scoring term inside same-formula groups, jointly with ICEBERG / GLACIER / fragments. "
                 "Built by enveda-casmi26/kernels/e1fuse/build.py in github.com/yasumorishima/kaggle-competitions."})
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
json.dump(nb, open(os.path.join(HERE, "main.ipynb"), "w", encoding="utf-8"), indent=1)
meta = {"id": f"yasunorim/casmi26-{name}", "title": f"casmi26 {name}", "code_file": "main.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "machine_shape": "NvidiaTeslaT4", "enable_internet": "false",
        # the base notebook's image (Python 3.12): ICEBERG / GLACIER ship cp312 wheels and fail on the 3.13 image
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "dataset_sources": ["prvsiyan/casmi26-fp-models-v2", "dmitriigluzdov/casmi26-fold-safe-fpnet",
                            "dmitriigluzdov/casmi26-pubchem-popularity-prior", "prvsiyan/casmi26-ranker-features",
                            "megayak/casmi26-simulated-ranker-rows", "ahmedberatozer/casmi26-fpnet-full1",
                            "ahmedberatozer/casmi26-glacier", "ahmedberatozer/casmi26-iceberg",
                            "ahmedberatozer/casmi26-pubchem-tier", "ahmedberatozer/casmi26-v2-pool",
                            "ahmedberatozer/casmi26-v3-models", "ahmedberatozer/casmi26-v4b-models",
                            "prvsiyan/chebi-lipidmaps-casmi26", "prvsiyan/coconut-casmi26-candidates",
                            "metric/rdkit-2026-3-3-wheel"],
        "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
        "kernel_sources": ["yasunorim/casmi26-fp2-peak-transformer", "yasunorim/casmi26-fp2L3"],
        "model_sources": []}
if (c3val or "cpu" in sys.argv[3:] or tpu) and "gpu" not in sys.argv[3:]:   # CPU only (gpu: keep the T4 for a full validation)
    meta.update(enable_gpu="false")
    meta.pop("machine_shape")
if tpu:
    meta.update(enable_tpu="true", machine_shape="TpuV5E8")
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("wrote main.ipynb", name, "FP2_LAM", lam, "val" if val else "", "only", only)
