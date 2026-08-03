"""Local validation of the iterated TCN-hybrid model/train logic (GroupNorm +
grad clip + cosine schedule) on synthetic well sequences. Catches API/shape bugs
without a Kaggle cycle. Mirrors the appended cells in
rogii-wellbore-tcn-hybrid/generate_notebook.py."""
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)
DEVICE = "cpu"
CH, NB, DROP, LR, WD, CLIP = 32, 3, 0.15, 5e-4, 1e-4, 1.0
N_FEAT = 195
EPOCHS = 3

def _gn(c):
    g = 8 if c % 8 == 0 else 1
    return nn.GroupNorm(g, c)

class TCNBlock(nn.Module):
    def __init__(s, c, d, drop):
        super().__init__()
        s.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d); s.n1 = _gn(c)
        s.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d); s.n2 = _gn(c)
        s.act = nn.ReLU(); s.do = nn.Dropout(drop)
    def forward(s, x):
        y = s.do(s.act(s.n1(s.c1(x)))); y = s.n2(s.c2(y)); return s.act(x + y)

class TCN(nn.Module):
    def __init__(s, cin, c, nb, drop=0.15):
        super().__init__()
        s.inp = nn.Conv1d(cin, c, 1)
        s.blocks = nn.ModuleList([TCNBlock(c, 2 ** i, drop) for i in range(nb)])
        s.head = nn.Conv1d(c, 1, 1)
    def forward(s, x):
        x = s.inp(x)
        for b in s.blocks:
            x = b(x)
        return s.head(x).squeeze(1)

def huber(p, t, d=1.0):
    e = p - t; a = e.abs()
    return torch.where(a <= d, 0.5 * e * e, d * (a - 0.5 * d)).mean()

# synthetic wells: variable toe length, incl. a length-1 and a short one
lengths = [1, 5, 3836, 200, 4717, 50]
seqs = [{'X': np.random.randn(L, N_FEAT).astype(np.float32),
         't': np.random.randn(L).astype(np.float32)} for L in lengths]

def to_x(s):
    return torch.tensor(s['X'].T[None], dtype=torch.float32, device=DEVICE)

from sklearn.model_selection import GroupKFold
groups = np.array([f"w{i}" for i in range(len(seqs))])
idx = np.arange(len(seqs))
gkf = GroupKFold(n_splits=2)
for fold, (tr, va) in enumerate(gkf.split(idx, groups=groups)):
    model = TCN(N_FEAT, CH, NB, DROP).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    tr = np.array(tr)
    for ep in range(EPOCHS):
        model.train(); np.random.shuffle(tr)
        for j in tr:
            s = seqs[j]
            x = to_x(s); t = torch.tensor(s['t'][None], dtype=torch.float32, device=DEVICE)
            opt.zero_grad(); loss = huber(model(x), t); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            for j in va:
                p = model(to_x(seqs[j])).cpu().numpy()[0]
                assert p.shape[0] == seqs[j]['X'].shape[0], (p.shape, seqs[j]['X'].shape)
    print(f"fold{fold} OK lr_end={opt.param_groups[0]['lr']:.2e}")
print("LOCAL TCN TEST PASSED (GroupNorm + clip + cosine, incl L=1 well)")
