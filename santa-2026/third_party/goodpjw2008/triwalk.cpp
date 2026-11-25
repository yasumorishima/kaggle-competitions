// triwalk: annealing over VALUE SETS instead of programs.
// A state is only the set V of values a program computes. a+b=c and a*b=c inside V are "triples": any two members give the third
// (c=a+b, a=c-b, b=c-a; c=a*b, a=c/b, b=c/a), so V is a valid program iff the triple closure of the starting values is V.
// Cost = |V \ starts|. Derivation choice and instruction order are implicit, so all programs with the same values are one state.
// Moves: add a value w that takes part in >= 2 triples of V (one to derive it, one to help somebody else), then drop a value that
// is no longer needed. Candidates w are kept incrementally in a pair table (counts of every y+z, |y-z|, y*z, y/z over V).
// Input: our P-block text format (src/pfmt.py); output: R blocks for strictly shorter, register-feasible programs.
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64; typedef long long ll;
static bool CPU_CLOCK = false;
static inline double now() { timespec t; clock_gettime(CPU_CLOCK ? CLOCK_PROCESS_CPUTIME_ID : CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
struct Rng { u64 s; u64 next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; } u64 operator()(u64 n) { return (u64)(((unsigned __int128)next() * n) >> 64); } double u() { return (next() >> 11) * 0x1.0p-53; } } rng{88172645463325252ULL};
struct Puzzle { string id; vector<u64> start, targets; u64 L; int regs; vector<array<int, 4>> prog; };
static bool readPuzzle(istream& in, Puzzle& p) {
    string tag; if (!(in >> tag) || tag != "P") return false;
    int k, m; in >> p.id >> k; p.start.resize(k); for (auto& x : p.start) in >> x;
    in >> m; p.targets.resize(m); for (auto& x : p.targets) in >> x;
    int n; in >> p.L >> p.regs >> n; p.prog.resize(n);
    for (auto& x : p.prog) { string op; in >> op >> x[1] >> x[2] >> x[3]; x[0] = op[0]; }
    return true;
}
static const int MV = 512, MT = 1 << 14, HC = 4096;
static inline u64 hsh(u64 v) { return v * 0x9E3779B97F4A7C15ULL; }

struct PairTable {   // value -> number of relations with pairs of V; candidates = values with count >= 2
    int bits = 0; u64 mask = 0; vector<u64> key; vector<int> cnt, pos; size_t used = 0; vector<u64> cand;
    void init(int b) { bits = b; mask = (1ULL << b) - 1; key.assign(mask + 1, 0); cnt.assign(mask + 1, 0); pos.assign(mask + 1, -1); used = 0; cand.clear(); }
    inline size_t find(u64 v) const { size_t i = hsh(v) >> (64 - bits); while (key[i] && key[i] != v) i = (i + 1) & mask; return i; }
    inline void inc(u64 v) {
        size_t i = find(v); if (!key[i]) { key[i] = v; used++; }
        if (++cnt[i] == 2) { pos[i] = cand.size(); cand.push_back(v); }
    }
    inline void dec(u64 v) {
        size_t i = find(v);
        if (--cnt[i] == 1) { int p = pos[i]; u64 last = cand.back(); cand.pop_back(); if (last != v) { cand[p] = last; pos[find(last)] = p; } pos[i] = -1; }
    }
};

struct Engine {
    const Puzzle& P; u64 L;
    u64 val[MV]; bool alive[MV], isS[MV], isT[MV]; int apos[MV], al[MV]; u64 av[MV]; int nal = 0; vector<int> freeS; int nStart = 0, one = -1;
    u64 hk[HC]; int16_t hv[HC];
    int16_t ta[MT], tb[MT], tc[MT], tx[MT]; uint8_t tt[MT]; int16_t tpa[MT], tpb[MT], tpc[MT]; vector<int> freeT; int tHigh = 0, nTri = 0;
    vector<int> inc[MV]; int deg[MV], crit[MV]; PairTable pt;
    vector<int> zeroed;   // values worth testing for removal after a move
    int sup[MV];          // acyclic choice of one deriving triple per value (-1 for starts and 1)
    ll stProp = 0, stU = 0, stSwap = 0, stUp = 0, stClos = 0, stClos2 = 0, stDown = 0, stReb = 0, stSched = 0, stSchedFail = 0;

    Engine(const Puzzle& p) : P(p), L(p.L) {
        memset(hk, 0, sizeof hk); memset(alive, 0, sizeof alive); memset(isS, 0, sizeof isS); memset(isT, 0, sizeof isT);
        for (int i = MV - 1; i >= 0; i--) freeS.push_back(i);
        for (int i = MT - 1; i >= 0; i--) freeT.push_back(i);
        memset(deg, 0, sizeof deg); memset(crit, 0, sizeof crit);
    }
    inline int look(u64 v) const { size_t i = hsh(v) >> 52; while (hk[i]) { if (hk[i] == v) return hv[i]; i = (i + 1) & (HC - 1); } return -1; }
    void hput(u64 v, int s) { size_t i = hsh(v) >> 52; while (hk[i]) i = (i + 1) & (HC - 1); hk[i] = v; hv[i] = s; }
    void hdel(u64 v) {
        size_t i = hsh(v) >> 52; while (hk[i] != v) i = (i + 1) & (HC - 1);
        size_t j = i;
        for (;;) { j = (j + 1) & (HC - 1); if (!hk[j]) break; size_t k = hsh(hk[j]) >> 52; if (((j - k) & (HC - 1)) >= ((j - i) & (HC - 1))) { hk[i] = hk[j]; hv[i] = hv[j]; i = j; } }
        hk[i] = 0;
    }
    inline bool fre(int x) const { return isS[x] || x == one; }
    inline bool derives(int t, int x) const { return ta[t] != tb[t] || tc[t] == x; }
    int sole(int x) const { for (int t : inc[x]) if (derives(t, x)) return t; return -1; }
    inline void critAdd(int t, int x, int d) {
        int a = ta[t], b = tb[t], c = tc[t];
        if (a != x) crit[a] += d;
        if (b != x && b != a) crit[b] += d;
        if (c != x) crit[c] += d;
    }
    void addTri(int a, int b, int c, int type) {
        int t = freeT.back(); freeT.pop_back(); tHigh = max(tHigh, t + 1); nTri++;
        int D[3], nd = 0; if (a == b) D[nd++] = c; else { D[0] = a; D[1] = b; D[2] = c; nd = 3; }
        for (int i = 0; i < nd; i++) if (!fre(D[i]) && deg[D[i]] == 1) critAdd(sole(D[i]), D[i], -1);
        ta[t] = a; tb[t] = b; tc[t] = c; tt[t] = type; tx[t] = a ^ b ^ c;
        tpa[t] = inc[a].size(); inc[a].push_back(t);
        if (b != a) { tpb[t] = inc[b].size(); inc[b].push_back(t); }
        tpc[t] = inc[c].size(); inc[c].push_back(t);
        for (int i = 0; i < nd; i++) { deg[D[i]]++; if (!fre(D[i]) && deg[D[i]] == 1) critAdd(t, D[i], +1); }
    }
    inline void unlinkAt(int x, int p) {
        int l = inc[x].back(); inc[x].pop_back(); if (p == (int)inc[x].size()) return;
        inc[x][p] = l; if (ta[l] == x) tpa[l] = p; else if (tb[l] == x) tpb[l] = p; else tpc[l] = p;
    }
    void delTri(int t) {
        int a = ta[t], b = tb[t], c = tc[t];
        int D[3], nd = 0; if (a == b) D[nd++] = c; else { D[0] = a; D[1] = b; D[2] = c; nd = 3; }
        for (int i = 0; i < nd; i++) if (!fre(D[i]) && deg[D[i]] == 1) critAdd(t, D[i], -1);
        unlinkAt(a, tpa[t]); if (b != a) unlinkAt(b, tpb[t]); unlinkAt(c, tpc[t]);
        for (int i = 0; i < nd; i++) { deg[D[i]]--; if (!fre(D[i]) && deg[D[i]] == 1) critAdd(sole(D[i]), D[i], +1); }
        freeT.push_back(t); nTri--;
    }
    // triples of value w (not in V) with pairs of V: (y slot, z slot, kind): 0 y+w=z, 1 y+z=w, 2 y*w=z, 3 y*z=w, 4 w+w=z, 5 w*w=z
    struct Rel { int y, z, k; };
    Rel rel[4 * MV]; int nrel = 0;
    void enumerate(u64 w) {
        nrel = 0; u64 Lw = L / w; int z;
        for (int i = 0; i < nal; i++) {
            u64 y = av[i]; int s = al[i];
            if (y <= L - w && (z = look(w + y)) >= 0) rel[nrel++] = {s, z, 0};
            if (w > y) {
                u64 d = w - y;
                if (d >= y && (z = look(d)) >= 0) rel[nrel++] = {s, z, 1};
                if (y > 1) { u64 q = w / y; if (q >= y && q * y == w && (z = look(q)) >= 0) rel[nrel++] = {s, z, 3}; }
            }
            if (w > 1 && y > 1 && y <= Lw && (z = look(w * y)) >= 0) rel[nrel++] = {s, z, 2};
        }
        if (w <= L - w && (z = look(w + w)) >= 0) rel[nrel++] = {-1, z, 4};
        if (w <= Lw && w > 1 && (z = look(w * w)) >= 0) rel[nrel++] = {-1, z, 5};
    }
    // put value into V (slot, hash, triples); pair table not touched
    int addLight(u64 w) {   // uses rel[] from enumerate(w)
        int s = freeS.back(); freeS.pop_back(); val[s] = w; alive[s] = true; apos[s] = nal; al[nal] = s; av[nal] = w; nal++; hput(w, s); deg[s] = 0; crit[s] = 0; inc[s].clear();
        if (w == 1) one = s;
        for (int i = 0; i < nrel; i++) { auto& r = rel[i];
            switch (r.k) { case 0: addTri(r.y, s, r.z, 0); break; case 1: addTri(r.y, r.z, s, 0); break; case 2: addTri(r.y, s, r.z, 1); break;
                           case 3: addTri(r.y, r.z, s, 1); break; case 4: addTri(s, s, r.z, 0); break; case 5: addTri(s, s, r.z, 1); break; } }
        return s;
    }
    void delLight(int s) {
        while (!inc[s].empty()) delTri(inc[s].back());
        hdel(val[s]); int p = apos[s]; nal--; if (p != nal) { al[p] = al[nal]; av[p] = av[nal]; apos[al[p]] = p; }
        alive[s] = false; freeS.push_back(s); if (s == one) one = -1;
    }
    template <bool ADD> inline void pairs(u64 w) {   // all relations of w with the current V (w itself must be in V)
        u64 Lw = L / w;
        auto f = [&](u64 r) { if (ADD) pt.inc(r); else pt.dec(r); };
        for (int i = 0; i < nal; i++) {
            u64 y = av[i];
            if (y <= L - w) f(w + y);
            if (y <= Lw) f(w * y);
            if (w > y) { f(w - y); u64 q = w / y; if (q * y == w) f(q); }
            else if (y > w) { f(y - w); u64 q = y / w; if (q * w == y) f(q); }
        }
        if (!(w & 1)) f(w >> 1);
        u64 r = (u64)sqrtl((long double)w); if (r * r == w && r > 1) f(r);
    }
    void rebuildPairs() {
        int b = 14; while ((1ULL << b) < (size_t)nal * nal * 5) b++;
        pt.init(min(b, 20)); stReb++;
        int n = nal;
        for (nal = 0; nal < n;) { u64 w = av[nal]; nal++; pairs<true>(w); }
    }
    void addFull(u64 w) { enumerate(w); addLight(w); pairs<true>(w); }
    void delFull(int s) { pairs<false>(val[s]); delLight(s); }
    int cost() const { return nal - nStart; }
    // closure of the starting values without slot u covers everything else?
    int16_t rem_[MT]; uint8_t tk[MT]; bool vis[MV]; int que[MV];
    bool closureOK(int u) {
        stClos++;
        memcpy(rem_, tx, sizeof(int16_t) * tHigh); memset(tk, 0, tHigh); memset(vis, 0, sizeof vis);
        if (u >= 0) vis[u] = true;
        int qh = 0, qt = 0;
        for (int i = 0; i < nStart; i++) { int s = startSlots[i]; vis[s] = true; que[qt++] = s; }
        if (one >= 0 && one != u && !vis[one]) { vis[one] = true; que[qt++] = one; }
        while (qh < qt) {
            int v = que[qh++];
            for (int t : inc[v]) {
                int th;
                if (ta[t] == tb[t]) { if (ta[t] != v) continue; th = tc[t]; }
                else { rem_[t] ^= v; if (++tk[t] != 2) continue; th = rem_[t]; }
                if (!vis[th]) { vis[th] = true; que[qt++] = th; }
            }
        }
        return qt == nal - (u >= 0 ? 1 : 0);
    }
    void buildSupport() {
        for (int i = 0; i < nal; i++) sup[al[i]] = -1;
        vector<int> q; memset(vis, 0, sizeof vis); memcpy(rem_, tx, sizeof(int16_t) * tHigh); memset(tk, 0, tHigh);
        for (int s : startSlots) { vis[s] = true; q.push_back(s); }
        if (one >= 0 && !vis[one]) { vis[one] = true; q.push_back(one); }
        for (size_t h = 0; h < q.size(); h++) { int v = q[h];
            for (int t : inc[v]) { int th;
                if (ta[t] == tb[t]) { if (ta[t] != v) continue; th = tc[t]; } else { rem_[t] ^= v; if (++tk[t] != 2) continue; th = rem_[t]; }
                if (!vis[th]) { vis[th] = true; sup[th] = t; q.push_back(th); } } }
    }
    // remove-test of u on the support DAG: mark everything supported through u as lost, then try to re-derive it without u
    bool lost[MV], gone[MV] = {}; vector<int> L1, Q1; vector<pair<int, int>> ns;
    inline bool partnersOK(int t, int x) const {   // can t derive x from non-lost values?
        int a = ta[t], b = tb[t], c = tc[t];
        if (a == b) return c == x && !lost[a];
        return (a == x || !lost[a]) && (b == x || !lost[b]) && (c == x || !lost[c]);
    }
    int freeX = -1;
    bool tryRemove(int u) {
        stClos++;
        stClos2++; L1.clear(); ns.clear(); L1.push_back(u); lost[u] = true;
        for (size_t h = 0; h < L1.size(); h++) { int v = L1[h];
            for (int t : inc[v]) { int a = ta[t], b = tb[t], c = tc[t];
                if (a != v && sup[a] == t && !lost[a] && a != freeX) { lost[a] = true; L1.push_back(a); }
                if (b != v && sup[b] == t && !lost[b] && b != freeX) { lost[b] = true; L1.push_back(b); }
                if (c != v && sup[c] == t && !lost[c] && c != freeX) { lost[c] = true; L1.push_back(c); } } }
        size_t need = L1.size() - 1; Q1.clear();
        for (size_t i = 1; i < L1.size(); i++) { int x = L1[i]; if (!lost[x]) continue;
            for (int t : inc[x]) if (partnersOK(t, x)) { lost[x] = false; ns.push_back({x, t}); Q1.push_back(x); break; }
            while (!Q1.empty()) { int v = Q1.back(); Q1.pop_back();
                for (int t : inc[v]) { int a = ta[t], b = tb[t], c = tc[t];
                    for (int y : {a, b, c}) if (y != v && y != u && lost[y] && !gone[y] && partnersOK(t, y)) { lost[y] = false; ns.push_back({y, t}); Q1.push_back(y); } } } }
        bool ok = ns.size() == need;
        for (int x : L1) lost[x] = false;
        return ok;
    }
    void commitRemove(int u) {   // after a successful tryRemove(u)
        auto touch = [&](int t) { if (t < 0) return; for (int y : {ta[t], tb[t], tc[t]}) zeroed.push_back(y); };
        for (auto& [x, t] : ns) { touch(sup[x]); sup[x] = t; }
        touch(sup[u]); delFull(u);
    }
    vector<int> startSlots;
    inline bool removable(int u) const { return !isS[u] && !isT[u]; }
    // ---------------------------------------------------------------- scheduling
    // State of the scheduler: an order of all non-start values plus one derivation per value (both operands earlier in the order).
    // Annealing moves: relocate a value inside its precedence window, or switch its derivation. Cost = register overflow.
    double posv[MV];
    struct Sch { vector<int> seq; int da[MV], db[MV]; uint8_t op[MV]; } sc;
    vector<array<int, 3>> ders[MV];   // (a, b, op) of a value
    int idxOf[MV], lastU[MV]; vector<int> die_;
    static inline int opOf(int t2, int kind) { return t2 == 0 ? (kind == 0 ? '+' : '-') : (kind == 0 ? '*' : '/'); }
    bool initSchedule() {   // guided closure: smallest position first among the derivable values
        sc.seq.clear(); typedef pair<double, int> PD; priority_queue<PD, vector<PD>, greater<PD>> pq; vector<char> inq(MV, 0), done(MV, 0);
        int anyS = startSlots[0];
        auto pushReady = [&](int v) { for (int t : inc[v]) { int a = ta[t], b = tb[t], c = tc[t];
            if (a == b) { if (a == v && !inq[c]) { inq[c] = 1; pq.push({posv[c], c}); } continue; }
            int cnt = done[a] + done[b] + done[c]; if (cnt != 2) continue; int th = !done[a] ? a : !done[b] ? b : c; if (!inq[th]) { inq[th] = 1; pq.push({posv[th], th}); } } };
        for (int s : startSlots) { done[s] = 1; inq[s] = 1; }
        for (int s : startSlots) pushReady(s);
        if (one >= 0 && !isS[one]) { if (!inq[one]) { inq[one] = 1; pq.push({posv[one], one}); } }
        while (!pq.empty()) {
            int x = pq.top().second; pq.pop(); bool ok = false;
            if (x == one) { sc.da[x] = sc.db[x] = anyS; sc.op[x] = '/'; ok = true; }
            int nd = 0;
            for (int t : inc[x]) { int a = ta[t], b = tb[t], c = tc[t], p = -1, q = -1, op = 0;
                if (a == b) { if (c != x || !done[a]) continue; p = q = a; op = opOf(tt[t], 0); }
                else if (c == x) { if (!done[a] || !done[b]) continue; p = a; q = b; op = opOf(tt[t], 0); }
                else { int o = a == x ? b : a; if (!done[c] || !done[o]) continue; p = c; q = o; op = opOf(tt[t], 1); }
                if (rng(++nd) == 0) { sc.da[x] = p; sc.db[x] = q; sc.op[x] = op; ok = true; } }
            if (!ok) return false;
            done[x] = 1; sc.seq.push_back(x); pushReady(x);
        }
        return (int)sc.seq.size() == nal - nStart;
    }
    int overflow(int limit) {   // sum over steps of max(0, registers needed - limit), weighted
        int n = sc.seq.size(); if ((int)die_.size() < n + 1) die_.assign(n + 1, 0);
        memset(lastU, 255, sizeof lastU);
        for (int i = 0; i < n; i++) { int x = sc.seq[i]; lastU[sc.da[x]] = i; lastU[sc.db[x]] = i; }
        std::fill(die_.begin(), die_.begin() + n + 1, 0); int cur = 0, ov = 0;
        for (int s : startSlots) if (lastU[s] >= 0) { cur++; die_[lastU[s]]++; }
        for (int i = 0; i < n; i++) { int x = sc.seq[i]; int need = cur - die_[i] + 1; if (need > limit) ov += (need - limit); cur -= die_[i]; if (lastU[x] > i) { cur++; die_[lastU[x]]++; } }
        return ov;
    }
    bool annealOrder(int iters) {
        int n = sc.seq.size(); int cur = overflow(P.regs); if (!cur) return true;
        for (int i = 0; i < n; i++) idxOf[sc.seq[i]] = i; for (int s : startSlots) idxOf[s] = -1;
        vector<vector<int>> users(MV); for (int x : sc.seq) { users[sc.da[x]].push_back(x); if (sc.db[x] != sc.da[x]) users[sc.db[x]].push_back(x); }
        auto rmUser = [&](int p, int x) { auto& v = users[p]; for (size_t i = 0; i < v.size(); i++) if (v[i] == x) { v[i] = v.back(); v.pop_back(); return; } };
        double Ts = 1.5;
        for (int it = 0; it < iters; it++) {
            double T = Ts * (1.0 - (double)it / iters) + 0.05;
            int i = rng(n), x = sc.seq[i];
            if (rng(8) == 0 && x != one) {   // other derivation
                auto& I = inc[x]; if (I.size() < 2) continue; int t = I[rng(I.size())]; int a = ta[t], b = tb[t], c = tc[t], p, q, op;
                if (a == b) { if (c != x) continue; p = q = a; op = opOf(tt[t], 0); } else if (c == x) { p = a; q = b; op = opOf(tt[t], 0); } else { p = c; q = a == x ? b : a; op = opOf(tt[t], 1); }
                if (idxOf[p] >= i || idxOf[q] >= i || (p == sc.da[x] && q == sc.db[x])) continue;
                int oa = sc.da[x], ob = sc.db[x], oo = sc.op[x]; sc.da[x] = p; sc.db[x] = q; sc.op[x] = op; int c2 = overflow(P.regs);
                if (c2 <= cur || rng.u() < exp((cur - c2) / T)) { cur = c2; rmUser(oa, x); if (ob != oa) rmUser(ob, x); users[p].push_back(x); if (q != p) users[q].push_back(x); if (!cur) return true; }
                else { sc.da[x] = oa; sc.db[x] = ob; sc.op[x] = oo; }
                continue;
            }
            int lo = max(idxOf[sc.da[x]], idxOf[sc.db[x]]) + 1, hi = n - 1; for (int u : users[x]) hi = min(hi, idxOf[u] - 1);
            if (hi <= lo) continue;
            int j = lo + rng(hi - lo + 1); if (j == i) continue;
            auto mv = [&](int f, int t) { if (f < t) rotate(sc.seq.begin() + f, sc.seq.begin() + f + 1, sc.seq.begin() + t + 1); else rotate(sc.seq.begin() + t, sc.seq.begin() + f, sc.seq.begin() + f + 1); };
            mv(i, j); int c2 = overflow(P.regs);
            if (c2 <= cur || rng.u() < exp((cur - c2) / T)) { cur = c2; for (int k = min(i, j); k <= max(i, j); k++) idxOf[sc.seq[k]] = k; if (!cur) return true; }
            else mv(j, i);
        }
        return false;
    }
    bool emitSchedule(vector<array<int, 4>>& out) {
        int n = sc.seq.size(); memset(lastU, 255, sizeof lastU);
        for (int i = 0; i < n; i++) { int x = sc.seq[i]; lastU[sc.da[x]] = i; lastU[sc.db[x]] = i; }
        vector<int> reg(MV, -1), occ(P.regs, -1);
        for (int k = 0; k < (int)P.start.size(); k++) { int s = look(P.start[k]); if (reg[s] < 0 && lastU[s] >= 0) { reg[s] = k; occ[k] = s; } }
        out.clear();
        for (int i = 0; i < n; i++) {
            int x = sc.seq[i], a = sc.da[x], b = sc.db[x], r1 = reg[a], r2 = reg[b];
            if (lastU[a] == i) occ[r1] = -1;
            if (lastU[b] == i) occ[r2] = -1;
            int d = -1; for (int r = 0; r < P.regs; r++) if (occ[r] < 0) { d = r; break; }
            if (d < 0) return false;
            out.push_back({sc.op[x], r1, r2, d}); reg[x] = d; if (lastU[x] > i) occ[d] = x;
        }
        return true;
    }
    double schedTime = 0;
    bool schedule(vector<array<int, 4>>& out, int tries, int iters) {
        stSched++; double t0 = now(); bool ok = false;
        for (int i = 0; i < tries && !ok; i++) if (initSchedule() && (overflow(P.regs) == 0 || annealOrder(iters)) && emitSchedule(out)) ok = true;
        schedTime += now() - t0; if (!ok) stSchedFail++; return ok;
    }
    vector<u64> orderedSet() const { vector<u64> r; for (int x : sc.seq) r.push_back(val[x]); return r; }
    // ---------------------------------------------------------------- walk
    vector<u64> bestSet, feasSet, endSet; vector<array<int, 4>> endProg; int best = 0; vector<array<int, 4>> bestProg;
    void loadSet(const vector<u64>& S) {
        while (nal) { int s = al[nal - 1]; delLight(s); }
        memset(isS, 0, sizeof isS); memset(isT, 0, sizeof isT); nStart = 0; startSlots.clear(); one = -1;
        auto put = [&](u64 v) { int s = look(v); if (s >= 0) return s; enumerate(v); return addLight(v); };
        // starts first; flags are set before triples exist so crit bookkeeping stays consistent
        for (u64 v : P.start) if (look(v) < 0) { nrel = 0; int s = addLight(v); isS[s] = true; startSlots.push_back(s); nStart++; }
        // triples among the starts themselves
        { vector<u64> st; for (int s : startSlots) st.push_back(val[s]);
          while (nal) delLight(al[nal - 1]);
          startSlots.clear(); nStart = 0;
          for (u64 v : st) { int s = freeS.back(); isS[s] = true; enumerate(v); int s2 = addLight(v); assert(s == s2); startSlots.push_back(s2); nStart++; } }
        for (int i = 0; i < nal; i++) posv[al[i]] = -1;
        { int ix = 0; for (u64 v : S) if (look(v) < 0) { int sl = put(v); posv[sl] = ix++; } }
        for (u64 t : P.targets) { int s = look(t); assert(s >= 0); isT[s] = true; }
        rebuildPairs(); zeroed.clear(); memset(lost, 0, sizeof lost); buildSupport();
    }
    vector<u64> curSet() const { vector<u64> r; for (int i = 0; i < nal; i++) if (!isS[al[i]]) r.push_back(av[i]); return r; }
    int mark[MV], mark2[MV], mark3[MV], ep = 0; vector<int> lst, U;
    double T = 0.25, big = 0, rbFrac = 0; int cap = 2; int critFull = 64;
    vector<vector<u64>> plateau; int plateauMax = 0;
    ll accSince = 0;
    int pruneZeroed(int keep) {   // drop values that became unneeded; returns number removed
        int removed = 0;
        while (!zeroed.empty()) {
            int j = rng(zeroed.size()); int u = zeroed[j]; zeroed[j] = zeroed.back(); zeroed.pop_back();
            if (u == keep || !alive[u] || !removable(u) || crit[u] != 0) continue;
            if (tryRemove(u)) { commitRemove(u); removed++; }
        }
        return removed;
    }
    int pruneAll(int keep) {
        int removed = 0; bool ch = true;
        while (ch) { ch = false; vector<int> o; for (int i = 0; i < nal; i++) if (removable(al[i]) && crit[al[i]] == 0 && al[i] != keep) o.push_back(al[i]);
            for (size_t i = o.size(); i > 1; i--) swap(o[i - 1], o[rng(i)]);
            for (int u : o) if (alive[u] && crit[u] == 0 && tryRemove(u)) { commitRemove(u); removed++; ch = true; } }
        zeroed.clear(); return removed;
    }
    // ---------------------------------------------------------------- rebuild of one value with up to two new values
    vector<u64> bits_; vector<u64> s1; vector<u64> sorted_; double deepP = 0.2, deepTime = 0; ll stDeep = 0, stDeepHit = 0, stDeepAcc = 0, lastF[8] = {}; ll stRb = 0, stRbHit = 0, stRbAcc = 0, stRbGain = 0; double rbTime = 0;
    static const int BB = 21;
    inline bool bit(u64 v) const { u64 h = hsh(v) >> (64 - BB); return bits_[h >> 6] >> (h & 63) & 1; }
    u64 kv[MV]; int nk = 0; static const int SB = 18; vector<u64> s1k = vector<u64>(1 << SB, 0); vector<uint32_t> s1pos;
    inline bool s1has(u64 v) const { size_t i = hsh(v) >> (64 - SB); while (s1k[i]) { if (s1k[i] == v) return true; i = (i + 1) & ((1 << SB) - 1); } return false; }
    inline bool s1put(u64 v) { size_t i = hsh(v) >> (64 - SB); while (s1k[i]) { if (s1k[i] == v) return false; i = (i + 1) & ((1 << SB) - 1); } if (s1pos.size() > (1u << SB) * 3 / 4) return false; s1k[i] = v; s1pos.push_back(i); return true; }
    template <class F> static inline void inv(u64 x, u64 a, u64 L, F f) {   // all w with x derivable from (w, a) in one triple
        if (x > a) f(x - a); else if (a > x) f(a - x);
        if (x <= L - a) f(x + a);
        if (a > 1) { u64 q = x / a; if (q * a == x) f(q); if (a > x && a % x == 0) f(a / x); if (x <= L / a) f(x * a); }
    }
    inline bool inS1K(u64 w) const { return bit(w) && s1has(w); }
    bool rebuildMove() {
        stRb++;
        int x; do x = al[rng(nal)]; while (fre(x));
        // what becomes removable if x were free?
        vector<int> Rm; vector<pair<int, int>> undo; vector<int> qu; freeX = x;
        auto pushT = [&](int t) { if (t < 0) return; int m[3] = {ta[t], tb[t], tc[t]}; for (int y : m) if (y != x) qu.push_back(y); };
        pushT(sup[x]);
        while (!qu.empty() && Rm.size() < 6) {
            int j = rng(qu.size()), r = qu[j]; qu[j] = qu.back(); qu.pop_back();
            if (gone[r] || !removable(r)) continue;
            if (tryRemove(r)) { for (auto& [y, t] : ns) { undo.push_back({y, sup[y]}); pushT(sup[y]); sup[y] = t; } pushT(sup[r]); gone[r] = lost[r] = true; Rm.push_back(r); }
        }
        int f = Rm.size(); bool found = false; u64 W1 = 0, W2 = 0; int g = 0; vector<u64> Wn;
        bool up1 = cost() + 1 <= best + cap;
        if (f >= 2 || (f == 1 && up1 && rng(4) == 0)) {
            // K = values derivable without x
            memcpy(rem_, tx, sizeof(int16_t) * tHigh); memset(tk, 0, tHigh); memset(vis, 0, sizeof vis); int qh = 0, qt = 0;
            for (int s : startSlots) { vis[s] = true; que[qt++] = s; }
            if (one >= 0 && !vis[one] && !gone[one] && one != x) { vis[one] = true; que[qt++] = one; }
            while (qh < qt) { int v = que[qh++];
                for (int t : inc[v]) { int th; if (ta[t] == tb[t]) { if (ta[t] != v) continue; th = tc[t]; } else { rem_[t] ^= v; if (++tk[t] != 2) continue; th = rem_[t]; }
                    if (!vis[th] && th != x && !gone[th]) { vis[th] = true; que[qt++] = th; } } }
            nk = qt; for (int i = 0; i < qt; i++) kv[i] = val[que[i]];
            bits_.assign((1ULL << BB) >> 6, 0); s1.clear(); for (uint32_t i : s1pos) s1k[i] = 0; s1pos.clear();
            auto put = [&](u64 r) { u64 h = hsh(r) >> (64 - BB); u64& w = bits_[h >> 6]; bool seen = w >> (h & 63) & 1; w |= 1ULL << (h & 63); if (s1put(r)) s1.push_back(r); (void)seen; };
            for (int i = 0; i < nk; i++) { u64 y = kv[i], Ly = L / y; for (int j = 0; j <= i; j++) { u64 z = kv[j];
                if (z <= L - y) put(y + z); if (z <= Ly && z > 1 && y > 1) put(y * z);
                if (y > z) { put(y - z); if (z > 1) { u64 q = y / z; if (q * z == y) put(q); } } else if (z > y) { put(z - y); if (y > 1) { u64 q = z / y; if (q * y == z) put(q); } } } }
            u64 X = val[x]; int nh = 0; int nOK = 0;
            auto offer = [&](u64 w1, u64 w2, int gg) {   // w2 == 0: single new value
                if (nh > 400) return; nh++;
                if (w1 == X || w2 == X) return; int s1_ = look(w1); if (s1_ >= 0 && !gone[s1_]) return; if (w2) { int s2_ = look(w2); if (s2_ >= 0 && !gone[s2_]) return; }
                if (s1_ >= 0 && gone[s1_] && w2 && look(w2) >= 0 && gone[look(w2)]) return;   // the old way
                if (!inS1K(w1)) return;
                if (gg == 3 && !inS1K(w2)) return;
                int real = (gg == 3 ? 2 : gg);
                if (real > f + (up1 ? 1 : 0)) return;
                if (!found || real < g || (real == g && rng(++nOK) == 0)) { if (!found || real < g) nOK = 1; found = true; W1 = w1; W2 = w2; g = real; }
            };
            for (int i = 0; i < nk; i++) inv(X, kv[i], L, [&](u64 w) { if (bit(w)) offer(w, 0, 1); });
            if (!found || g > 1) {
                if (f >= 2 || up1) {
                    int off = rng(nk);
                    for (int ii = 0; ii < nk && nh < 400; ii++) { int i = ii + off; if (i >= nk) i -= nk;
                        inv(X, kv[i], L, [&](u64 w2) { if (w2 == X || look(w2) >= 0) return; for (int j = 0; j < nk; j++) inv(w2, kv[j], L, [&](u64 w1) { if (bit(w1)) offer(w1, w2, 2); }); }); }
                    size_t ns1 = s1.size(), o2 = ns1 ? rng(ns1) : 0;
                    for (size_t q = 0; q < ns1 && nh < 400; q++) { u64 w1 = s1[q + o2 < ns1 ? q + o2 : q + o2 - ns1]; inv(X, w1, L, [&](u64 w2) { if (bit(w2) && w2 != w1) offer(w1, w2, 3); }); }
                }
            }
            // three or four new values: product near-misses x = A*B +- R and sampled longer chains
            int lim = f + (up1 ? 1 : 0); lastF[min(f, 7)]++;
            if (!found && lim >= 3 && f >= 2 && rng.u() < deepP) {
                stDeep++; double td0 = now();
                sorted_ = s1; for (int i = 0; i < nk; i++) sorted_.push_back(kv[i]); sort(sorted_.begin(), sorted_.end()); sorted_.erase(unique(sorted_.begin(), sorted_.end()), sorted_.end());
                auto inK = [&](u64 v) { int z = look(v); return z >= 0 && vis[z]; };
                auto isNew = [&](u64 v) { int z = look(v); return z < 0; };
                int bestN = lim + 1, ties = 0, checks = 0;
                auto take = [&](initializer_list<u64> l) {   // values needed; those in K are free, others must be in S1(K) unless listed last (the product)
                    vector<u64> nv; size_t i = 0; for (u64 v : l) { i++; if (v == X) return; if (inK(v)) continue; if (look(v) >= 0 && !gone[look(v)]) return; if (find(nv.begin(), nv.end(), v) != nv.end()) continue; if (i < l.size() && !inS1K(v)) return; nv.push_back(v); }
                    int n = nv.size(); if (n > lim || n > bestN) return;
                    bool old = true; for (u64 v : nv) if (look(v) < 0) old = false; if (old) return;
                    if (n < bestN) { bestN = n; ties = 0; } if (rng(++ties) == 0) Wn = nv; };
                // A*B +- R with B in K or S1, A near X/B in S1+K, R in K or S1
                size_t nb = sorted_.size(); size_t st = nb > 6000 ? nb / 6000 : 1, o = rng(st);
                auto tryB = [&](u64 B) { if (B < 2 || B > X) return; u64 q = X / B; auto itp = lower_bound(sorted_.begin(), sorted_.end(), q); size_t c = itp - sorted_.begin();
                    for (int d = -3; d <= 3; d++) { ll ix = (ll)c + d; if (ix < 0 || ix >= (ll)nb) continue; u64 A = sorted_[ix]; if (A < 2 || A > L / B) continue; u64 Pp = A * B; if (Pp == X || Pp == 0) continue; u64 R = Pp > X ? Pp - X : X - Pp;
                        if (R != X && (inK(R) || bit(R)) && checks < 20000) { checks++; take({A, B, R, Pp}); } } };
                for (int i = 0; i < nk; i++) tryB(kv[i]);
                for (size_t i = o; i < nb; i += st) tryB(sorted_[i]);
                // (w1 op w2) op a for sampled a; ((w1 op c) op b) op a for sampled a
                if (lim >= 3 && X < (1ULL << 40)) for (int rep = 0; rep < 6; rep++) { u64 a = kv[rng(nk)];
                    inv(X, a, L, [&](u64 y) { if (look(y) >= 0) return;
                        if (rep < 2) { for (u64 w1 : s1) inv(y, w1, L, [&](u64 w2) { if (w2 != w1 && bit(w2) && checks < 20000) { checks++; take({w1, w2, y}); } }); }
                        else { u64 b = 0; for (int r2 = 0; r2 < 12; r2++) { b = kv[rng(nk)]; inv(y, b, L, [&](u64 z) { if (look(z) >= 0) return; for (int j = 0; j < nk; j++) inv(z, kv[j], L, [&](u64 w1) { if (bit(w1) && checks < 20000) { checks++; if (inS1K(w1)) { vector<u64> nv{w1, z, y}; bool bad = false; for (u64 v : nv) if (v == X || (look(v) >= 0 && !gone[look(v)])) bad = true; if (bad) return; bool old = true; for (u64 v : nv) if (look(v) < 0) old = false; if (old || 3 > lim || 3 > bestN) return; if (3 < bestN) { bestN = 3; ties = 0; } if (rng(++ties) == 0) Wn = nv; } } }); }); } } }); }
                if (!Wn.empty() && bestN <= lim) { found = true; g = bestN; stDeepHit++; } else Wn.clear();
                deepTime += now() - td0;
            }
        }
        // restore
        for (size_t i = undo.size(); i-- > 0;) sup[undo[i].first] = undo[i].second;
        for (int r : Rm) gone[r] = lost[r] = false;
        freeX = -1;
        if (!found) return false;
        stRbHit++;
        // apply for real: add the new values, then remove what became unnecessary
        vector<int> added; zeroed.clear();
        auto addReal = [&](u64 w) { if (look(w) >= 0) return; enumerate(w); int s = addLight(w); sup[s] = -1; int nd = 0; for (int t : inc[s]) if (derives(t, s) && rng(++nd) == 0) sup[s] = t;
            posv[s] = posv[x] - 1e-3 * (1 + rng.u()); if (sup[s] < 0) { delLight(s); return; } pairs<true>(w); added.push_back(s); };
        if (!Wn.empty()) { for (u64 w : Wn) addReal(w); if ((int)added.size() < (int)Wn.size()) { /* some value could not be derived in this order */ for (u64 w : Wn) if (look(w) < 0) addReal(w); } }
        else { addReal(W1); if (W2) addReal(W2); }
        if (nal >= MV - 4) return false;
        int removed = 0;
        for (int r : Rm) if (alive[r] && removable(r) && find(added.begin(), added.end(), r) == added.end() && crit[r] == 0 && tryRemove(r)) { commitRemove(r); removed++; }
        int net = (int)added.size() - removed;
        if (net > (up1 ? 1 : 0)) { for (size_t i = added.size(); i-- > 0;) if (alive[added[i]] && tryRemove(added[i])) commitRemove(added[i]); zeroed.clear(); return false; }
        stRbAcc++; if (!Wn.empty()) stDeepAcc++; int r2 = pruneZeroed(-1); if (net - r2 < 0) { stRbGain++; }
        onState(); return true;
    }
    void onState() {
        int c = cost();
        if (c < best) {
            vector<array<int, 4>> out;
            if (schedule(out, 3, 60000)) { best = c; bestProg = out; bestSet = orderedSet(); stDown++; feasSet = bestSet; }
        }
    }
    double pu0 = 0, pu1 = 0; int cycles = 1, trace = 0, chkEvery = 0, chkIters = 4000; ll stBack = 0;
    void run(double secs, double resetEvery) {
        double t0 = now(), tReset = t0, pUpCur = 0; int cyc = 0, lastTr = -1; memset(mark, 0, sizeof mark); memset(mark2, 0, sizeof mark2); memset(mark3, 255, sizeof mark3);
        const double pUp0 = exp(-1.0 / T); feasSet = bestSet; if (nal >= 150) chkEvery = 400;
        for (ll it = 0;; it++) {
            if ((it & 63) == 0) { double t = now(); if (t - t0 > secs) break;
                if (pu0 > 0) {   // annealing cycles: up-move probability decays geometrically from pu0 to pu1
                    double f = (t - t0) / secs * cycles; int c = min((int)f, cycles - 1); f = min(1.0, f - c);
                    if (c != cyc) { cyc = c; loadSet(bestSet); }
                    pUpCur = pu0 * pow(pu1 / pu0, f);
                    if (trace) { int tr = (int)((t - t0) / secs * trace); if (tr != lastTr) { lastTr = tr; cerr << "  t " << fixed << setprecision(1) << t - t0 << " pUp " << setprecision(5) << pUpCur << " cost " << cost() << " best " << best << " cand " << pt.cand.size() << " tri " << nTri << "\n"; } }
                } else if (t - tReset > resetEvery) { tReset = t; loadSet(chkEvery ? feasSet : bestSet); } }
            if (pt.cand.empty()) { break; }
            if (pt.used > pt.mask * 0.6) rebuildPairs();
            if (rbFrac > 0 && (it & 15) == 0 && rbTime < rbFrac * (now() - t0)) { double tb0 = now(); rebuildMove(); rbTime += now() - tb0; if (pt.used > pt.mask * 0.6) rebuildPairs(); }
            u64 w = pt.cand[rng(pt.cand.size())];
            if (big > 0 && rng.u() < big) for (int q = 0; q < 5; q++) { u64 w2 = pt.cand[rng(pt.cand.size())]; if (w2 > w) w = w2; }
            if (w == 1 || look(w) >= 0) continue;
            stProp++;
            enumerate(w);
            bool der = false; for (int i = 0; i < nrel; i++) if (rel[i].k < 4) { der = true; break; }
            if (!der || nrel < 2) continue;
            if (nal >= MV - 2 || nTri + nrel > MT - 8) continue;
            // values that w helps: the partners in their current support may become unnecessary
            lst.clear(); U.clear(); ep++;
            auto gain = [&](int x) { if (fre(x) || mark[x] == ep) return; mark[x] = ep;
                int t = sup[x]; for (int p : {ta[t], tb[t], tc[t]}) if (p != x && removable(p) && mark2[p] != ep) { mark2[p] = ep; lst.push_back(p); } };
            for (int i = 0; i < nrel; i++) { auto& r = rel[i]; if (r.k < 4) { gain(r.y); gain(r.z); } else gain(r.z); }
            // hard filter: p may still be the only way to somebody that w does not help
            for (int p : lst) { int c = crit[p]; if (c) for (int t : inc[p]) { for (int x : {ta[t], tb[t], tc[t]}) if (x != p && !fre(x) && deg[x] == 1 && mark[x] == ep && derives(t, x) && mark3[x] != p) { mark3[x] = p; c--; } } for (int t : inc[p]) for (int x : {ta[t], tb[t], tc[t]}) mark3[x] = -1; if (c <= 0) U.push_back(p); }
            bool upOK = pu0 > 0 ? rng.u() < pUpCur : (cost() + 1 <= best + cap && rng.u() < pUp0);
            if (U.empty() && !upOK) continue;
            zeroed.clear();
            int s = addLight(w); int u = -1; sup[s] = -1;
            { int nd = 0; for (int t : inc[s]) if (derives(t, s) && rng(++nd) == 0) sup[s] = t;
              double pm = -1, um = 1e18; for (int y : {ta[sup[s]], tb[sup[s]], tc[sup[s]]}) if (y != s) pm = max(pm, posv[y]);
              for (int t : inc[s]) for (int y : {ta[t], tb[t], tc[t]}) if (y != s && posv[y] > pm) um = min(um, posv[y]);
              posv[s] = um < 1e17 ? (rng(2) ? um - 1e-3 * rng.u() : pm + (um - pm) * rng.u()) : pm + 1e-3 * rng.u(); }
            if (!U.empty()) { stU++; for (size_t i = U.size(); i > 1; i--) swap(U[i - 1], U[rng(i)]); for (int x : U) if (crit[x] == 0 && tryRemove(x)) { u = x; break; } }
            if (u < 0 && !upOK) { delLight(s); zeroed.clear(); continue; }
            pairs<true>(w);
            if (u >= 0) {
                stSwap++; commitRemove(u); accSince++;
                int r = pruneZeroed(s);
                if (accSince % critFull == 0) r += pruneAll(s);
                if (r) onState();
                if (chkEvery > 0 && accSince % chkEvery == 0 && cost() <= best + 1) {   // stay where a 48-register schedule exists
                    vector<array<int, 4>> o2; if (schedule(o2, 1, chkIters)) feasSet = orderedSet(); else { stBack++; loadSet(feasSet); } }
            } else { stUp++; zeroed.clear(); }
        }
    }
};

int main(int argc, char** argv) {
    double tper = 10, T = 0.25, reset = 5; u64 seed = 1; int cap = 2, verbose = 0, cycles = 1, trace = 0; double pu0 = 0, pu1 = 1e-5, big = 0, rbFrac = 0, deepP = 0;
    for (int i = 1; i < argc; i++) { string a = argv[i];
        if (a == "--time") tper = atof(argv[++i]); else if (a == "--seed") seed = atoll(argv[++i]); else if (a == "--T") T = atof(argv[++i]);
        else if (a == "--cap") cap = atoi(argv[++i]); else if (a == "--reset") reset = atof(argv[++i]); else if (a == "--verbose") verbose = 1; else if (a == "--cpu") CPU_CLOCK = true;
        else if (a == "--pu0") pu0 = atof(argv[++i]); else if (a == "--pu1") pu1 = atof(argv[++i]); else if (a == "--big") big = atof(argv[++i]); else if (a == "--rb") rbFrac = atof(argv[++i]); else if (a == "--deep") deepP = atof(argv[++i]); else if (a == "--cycles") cycles = atoi(argv[++i]); else if (a == "--trace") trace = atoi(argv[++i]); }
    ios::sync_with_stdio(false);
    Puzzle p; u64 idx = 0;
    while (readPuzzle(cin, p)) {
        rng.s = (seed * 0x9E3779B97F4A7C15ULL) ^ (hash<string>{}(p.id) + idx++); if (!rng.s) rng.s = 1; for (int i = 0; i < 10; i++) rng.next();
        auto E = make_unique<Engine>(p); double t0 = now();
        vector<u64> regs(p.regs, 0), S; for (size_t i = 0; i < p.start.size(); i++) regs[i] = p.start[i];
        bool bad = false; set<u64> seen(p.start.begin(), p.start.end());
        for (auto& x : p.prog) { u64 a = regs[x[1]], b = regs[x[2]], r = x[0] == '+' ? a + b : x[0] == '-' ? a - b : x[0] == '*' ? a * b : (b ? a / b : 0); regs[x[3]] = r; if (!r) bad = true; if (seen.insert(r).second) S.push_back(r); }
        for (u64 t : p.targets) if (!seen.count(t)) bad = true;
        if (bad || S.size() + p.start.size() > MV - 8) { cerr << p.id << " skipped\n"; continue; }
        E->T = T; E->big = big; E->rbFrac = rbFrac; E->deepP = deepP; E->cap = cap; E->pu0 = pu0; E->pu1 = pu1; E->cycles = cycles; E->trace = trace; E->best = p.prog.size(); E->bestProg = p.prog; E->bestSet = S;
        E->loadSet(S);
        if (!E->closureOK(-1)) { cerr << p.id << " closure-fail\n"; continue; }
        int c0 = E->cost(); E->pruneAll(-1); E->onState();
        int tri0 = E->nTri, cand0 = E->pt.cand.size(); int z = 0; for (int i = 0; i < E->nal; i++) if (E->removable(E->al[i]) && E->crit[E->al[i]] == 0) z++;
        E->run(tper, reset);
        // equal-length end state (for the archive / the next round)
        if (E->best == (int)p.prog.size()) { for (int k = 0; k < 2 && E->endProg.empty(); k++) { if (k) E->loadSet(E->feasSet.empty() ? E->bestSet : E->feasSet); E->pruneAll(-1);
            vector<array<int, 4>> o; if (E->cost() <= E->best && E->schedule(o, 2, 20000) && (int)o.size() <= E->best) E->endProg = o; } }
        double dt = now() - t0;
        if (verbose) cerr << p.id << " prog " << p.prog.size() << " set " << c0 << " tri " << tri0 << " cand " << cand0 << " crit0 " << z << " | prop/s " << (ll)(E->stProp / dt) << " U/s " << (ll)(E->stU / dt) << " swap/s " << (ll)(E->stSwap / dt)
            << " up/s " << (ll)(E->stUp / dt) << " clos/s " << (ll)(E->stClos / dt) << "(" << (ll)(E->stClos2 / dt) << ")" << " rebuilds " << E->stReb << " sched " << E->stSched << " fail " << E->stSchedFail << " rb " << E->stRb << "/" << E->stRbHit << "/" << E->stRbAcc << "/" << E->stRbGain << " rbT " << fixed << setprecision(1) << E->rbTime << " deep " << E->stDeep << "/" << E->stDeepHit << "/" << E->stDeepAcc << " deepT " << E->deepTime << " f " << E->lastF[2] << "," << E->lastF[3] << "," << E->lastF[4] << "," << E->lastF[5] << "," << E->lastF[6] << " back " << E->stBack << " schedT " << fixed << setprecision(2) << E->schedTime << " ->";
        cerr << " " << p.id << " " << p.prog.size() << " -> " << E->best << "\n";
        if (E->best < (int)p.prog.size()) {
            cout << "R " << p.id << " " << p.prog.size() << " " << E->bestProg.size() << "\n";
            for (auto& x : E->bestProg) cout << (char)x[0] << " " << x[1] << " " << x[2] << " " << x[3] << "\n";
            cout.flush();
        } else if (!E->endProg.empty() && E->endProg.size() <= p.prog.size()) {   // "R" if it is even shorter, "V" for an equal-length variant
            cout << (E->endProg.size() < p.prog.size() ? "R " : "V ") << p.id << " " << p.prog.size() << " " << E->endProg.size() << "\n";
            for (auto& x : E->endProg) cout << (char)x[0] << " " << x[1] << " " << x[2] << " " << x[3] << "\n";
            cout.flush();
        }
    }
}
