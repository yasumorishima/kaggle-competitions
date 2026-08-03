"""Validate the seed-bagging OOF/test aggregation logic (the restructured ensemble
TCN loop) on synthetic well sequences. Mirrors rogii-wellbore-ensemble TCN cell."""
import numpy as np, torch, torch.nn as nn, pandas as pd
from sklearn.model_selection import GroupKFold
torch.manual_seed(0); np.random.seed(0)
_dev = "cpu"
T_CH, T_NB, T_DROP, T_LR, T_WD, T_CLIP, T_EPOCHS, T_PAT = 16, 2, 0.15, 5e-4, 1e-4, 1.0, 3, 2
NF = 195; N_SEED = 2

def _gn(c): return nn.GroupNorm(8 if c % 8 == 0 else 1, c)
class _Blk(nn.Module):
    def __init__(s,c,d,dr):
        super().__init__(); s.c1=nn.Conv1d(c,c,3,padding=d,dilation=d); s.n1=_gn(c)
        s.c2=nn.Conv1d(c,c,3,padding=d,dilation=d); s.n2=_gn(c); s.a=nn.ReLU(); s.do=nn.Dropout(dr)
    def forward(s,x):
        y=s.do(s.a(s.n1(s.c1(x)))); y=s.n2(s.c2(y)); return s.a(x+y)
class _TCN(nn.Module):
    def __init__(s,cin,c,nb,dr):
        super().__init__(); s.inp=nn.Conv1d(cin,c,1)
        s.bl=nn.ModuleList([_Blk(c,2**i,dr) for i in range(nb)]); s.h=nn.Conv1d(c,1,1)
    def forward(s,x):
        x=s.inp(x)
        for b in s.bl: x=b(x)
        return s.h(x).squeeze(1)
def _hub(p,t,d=1.0):
    e=p-t; a=e.abs(); return torch.where(a<=d,0.5*e*e,d*(a-0.5*d)).mean()

# synthetic train wells + a fake train_df 'id' order + test
wids = [f"w{k}" for k in range(12)]
_tr = []
ids_all = []
for wi, w in enumerate(wids):
    L = [1, 4, 200, 50][wi % 4]
    ids = [f"{w}_{i}" for i in range(L)]; ids_all += ids
    _tr.append({'wid': w, 'X': np.random.randn(L, NF).astype(np.float32),
                'ids': np.array(ids), 't': np.random.randn(L).astype(np.float32)})
_te = [{'wid': f"t{k}", 'X': np.random.randn(30, NF).astype(np.float32),
        'ids': np.array([f"t{k}_{i}" for i in range(30)])} for k in range(3)]
_ymu, _ysd = 0.0, 1.0
_grp = np.array([s['wid'] for s in _tr])
def _tx(s): return torch.tensor(s['X'].T[None], dtype=torch.float32, device=_dev)
CFGcv = GroupKFold(n_splits=3)

def _train_one(_tri,_vai,_seed):
    torch.manual_seed(_seed); np.random.seed(_seed)
    _m=_TCN(NF,T_CH,T_NB,T_DROP).to(_dev)
    _opt=torch.optim.Adam(_m.parameters(),lr=T_LR,weight_decay=T_WD)
    _sch=torch.optim.lr_scheduler.CosineAnnealingLR(_opt,T_max=T_EPOCHS)
    _best=1e9;_bs=None;_bad=0;_trl=np.array(_tri)
    for ep in range(T_EPOCHS):
        _m.train(); np.random.shuffle(_trl)
        for j in _trl:
            s=_tr[j]; x=_tx(s); t=torch.tensor(s['t'][None],dtype=torch.float32,device=_dev)
            _opt.zero_grad(); _hub(_m(x),t).backward()
            torch.nn.utils.clip_grad_norm_(_m.parameters(),T_CLIP); _opt.step()
        _sch.step()
        _m.eval(); P=[];T=[]
        with torch.no_grad():
            for j in _vai:
                s=_tr[j]; P.append(_m(_tx(s)).cpu().numpy()[0]*_ysd+_ymu); T.append(s['t']*_ysd+_ymu)
        vr=float(np.sqrt(np.mean((np.concatenate(P)-np.concatenate(T))**2)))
        if vr<_best-1e-4: _best=vr;_bad=0;_bs={k:v.detach().cpu().clone() for k,v in _m.state_dict().items()}
        else:_bad+=1
        if _bad>=T_PAT: break
    _m.load_state_dict(_bs); _m.eval(); return _m

_oof_id={};_test_sum={};_nf=0;_fb=[]
_idx=np.arange(len(_tr))
for _f,(_tri,_vai) in enumerate(CFGcv.split(_idx,groups=_grp)):
    _va_sum={};_te_fold={}
    for _seed in range(N_SEED):
        _m=_train_one(_tri,_vai,1000*_seed+_f)
        with torch.no_grad():
            for j in _vai:
                s=_tr[j]; pr=_m(_tx(s)).cpu().numpy()[0]*_ysd+_ymu
                for i,_id in enumerate(s['ids']): _va_sum[_id]=_va_sum.get(_id,0.0)+float(pr[i])/N_SEED
            for s in _te:
                pr=_m(_tx(s)).cpu().numpy()[0]*_ysd+_ymu
                for i,_id in enumerate(s['ids']): _te_fold[_id]=_te_fold.get(_id,0.0)+float(pr[i])/N_SEED
    for _id,v in _va_sum.items(): _oof_id[_id]=v
    for _id,v in _te_fold.items(): _test_sum[_id]=_test_sum.get(_id,0.0)+v
    _nf+=1
    _vp=[];_vt=[]
    for j in _vai:
        s=_tr[j]; _vp.extend([_va_sum[i] for i in s['ids']]); _vt.extend(list(s['t']*_ysd+_ymu))
    _fr=float(np.sqrt(np.mean((np.array(_vp)-np.array(_vt))**2))); _fb.append(_fr)
    print(f"fold{_f} bagged({N_SEED}) RMSE={_fr:.4f} (vsize={len(_vp)})")

# align checks
train_id = pd.Series(ids_all)
tcn_oof = train_id.map(_oof_id).to_numpy(dtype=np.float32)
test_id = pd.Series([i for s in _te for i in s['ids']])
te_mean = {k: v/_nf for k,v in _test_sum.items()}
tcn_test = test_id.map(te_mean).to_numpy(dtype=np.float32)
assert not np.isnan(tcn_oof).any(), "OOF unmapped"
assert not np.isnan(tcn_test).any(), "test unmapped"
assert len(tcn_oof) == len(ids_all)
assert _nf == 3
print(f"OOF mapped {len(tcn_oof)}/{len(ids_all)}, test mapped {len(tcn_test)}, folds={_nf}")
print("LOCAL BAG TEST PASSED (seed-avg OOF/test, id-align, fold RMSE, incl L=1)")
