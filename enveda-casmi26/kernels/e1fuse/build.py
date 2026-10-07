"""Build kernels/e1fuse/main.ipynb = the public 0.420 notebook + our changes.

Base: huseyinemreaksoy/casmi26-v4n-fusion-pubchem-on-public-0-421 (version 1, LB 0.420, Apache-2.0; itself built on
ahmedberatozer's v4 engine, prvsiyan's engine 2, ICEBERG/GLACIER, the PubChem tier and the popularity prior), copied
unchanged as base_0420.ipynb. Our changes:
  1. ours_fp2.py: our fp2 (spectrum -> Morgan bits) scores every candidate (base top-N, engine-2, PubChem lists).
  2. fusion_core: fp2 is one more re-scoring term inside same-formula groups, jointly with ICEBERG / GLACIER /
     fragment: z(ranker) + ICE_LAM z(ice) + GL_LAM z(gl) + FP2_LAM z(fp2). It also covers the adducts and negative-mode
     spectra the forward models cannot score. FP2_LIB_OFF keeps library hits untouched.
  3. validation mode: the fusion is replayed for every FP2_LAMS value and the MRR of each is logged.

    python build.py NAME FP2_LAM [val] [only=fp2L3,fp2-peak]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
name, lam = sys.argv[1], float(sys.argv[2])
val = "val" in sys.argv[3:]
only = next((a.split("=", 1)[1].split(",") for a in sys.argv[3:] if a.startswith("only=")), None)
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
setsrc(2, "# embedded sources (base notebook's EMBED + ours_fp2.py; fusion_core.py patched by kernels/e1fuse/build.py)\n"
       "EMBED = " + repr(E) + "\n")

# ---- 2. config: our knobs on top of the base notebook's final CFG.update
c4 = src(4)
upd = {"VERSION": f"ours-{name}", "FP2_LAM": lam, "FP2_LIB_OFF": True, "FP2_ONLY": only,
       "FP2_LAMS": [0.0, 0.25, 0.5, 1.0, 1.5]}
if val:
    upd.update(VALIDATION=True, FP_BANK="A")
setsrc(4, c4 + "\nCFG.update(" + repr(upd) + ")   # ours (kernels/e1fuse/build.py)\n")

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
'''
setsrc(i20, c20)

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
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("wrote main.ipynb", name, "FP2_LAM", lam, "val" if val else "", "only", only)
