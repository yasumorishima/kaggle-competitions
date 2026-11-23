#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace small_factor {
using U = std::uint64_t;

inline U mul(U x, U y, U m) {
#if defined(_MSC_VER) && defined(_M_X64)
    U hi, residue;
    U lo = _umul128(x, y, &hi);
    if (!hi) return lo % m;
    _udiv128(hi, lo, m, &residue);
    return residue;
#elif defined(__SIZEOF_INT128__)
    return (U)((__uint128_t)x * y % m);
#else
    U r = 0;
    auto add = [m](U a, U b) { return a >= m - b ? a - (m - b) : a + b; };
    for (; y; y >>= 1, x = add(x, x))
        if (y & 1) r = add(r, x);
    return r;
#endif
}

inline U power(U a, U n, U m) {
    U out = 1;
    for (; n; n >>= 1, a = mul(a, a, m)) if (n & 1) out = mul(out, a, m);
    return out;
}

inline U rng(U& state) {
    state += 0x9e3779b97f4a7c15ULL;
    U x = state;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline bool prime(U x) {
    if (x < 2) return false;
    for (U d : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
        if (x % d == 0) return x == d;
    U exp = x - 1;
    int s = 0;
    while (!(exp & 1)) { exp >>= 1; ++s; }
    for (U a : {2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL}) {
        if (a % x == 0) continue;
        U z = power(a % x, exp, x);
        if (z == 1 || z == x - 1) continue;
        bool tested_prime = false;
        for (int i = 1; i < s; ++i) {
            z = mul(z, z, x);
            if (z == x - 1) { tested_prime = true; break; }
        }
        if (!tested_prime) return false;
    }
    return true;
}

inline U brent(U n, int effort) {
    U state = n;
    for (int trial = 0; trial < 3; ++trial) {
        U y = rng(state) % (n - 1) + 1, c = rng(state) % (n - 1) + 1;
        U g = 1, x = 0, saved = 0, span = 1;
        auto step = [&](U x) {
            U s = mul(x, x, n);
            return s >= n - c ? s - (n - c) : s + c;
        };
        int remaining = effort;
        while (g == 1 && remaining > 0) {
            x = y;
            for (U i = 0; i < span && remaining-- > 0; ++i) y = step(y);
            for (U k = 0; k < span && g == 1 && remaining > 0; k += 64) {
                saved = y;
                U product = 1;
                for (U i = 0; i < std::min<U>(64, span - k) && remaining-- > 0; ++i) {
                    y = step(y);
                    product = mul(product, x > y ? x - y : y - x, n);
                }
                g = std::gcd(product, n);
            }
            span *= 2;
        }
        if (g == n) {
            g = 1;
            for (int i = 0; i < 64 && g == 1; ++i) {
                saved = step(saved);
                g = std::gcd(x > saved ? x - saved : saved - x, n);
            }
        }
        if (g > 1 && g < n) return g;
    }
    return n;
}

inline void decompose(U u, std::vector<U>& f, int effort) {
    if (u == 1) return;
    for (U d : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
        while (u % d == 0) { f.push_back(d); u /= d; }
    if (u == 1) return;
    U d = prime(u) ? u : brent(u, effort);
    if (d == u) f.push_back(u);
    else { decompose(d, f, effort); decompose(u / d, f, effort); }
}

// Remaining unsplit composites are valid atoms. Completeness is unnecessary.
inline std::vector<U> divisors(U n, int effort = 6000) {
    std::vector<U> f, out = {1};
    decompose(n, f, effort);
    std::sort(f.begin(), f.end());
    for (std::size_t i = 0; i < f.size();) {
        std::size_t j = i + 1, size = out.size();
        while (j < f.size() && f[j] == f[i]) ++j;
        U p = 1;
        for (std::size_t count = i; count < j; ++count) {
            p *= f[i];
            for (std::size_t k = 0; k < size; ++k) out.push_back(out[k] * p);
        }
        i = j;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}
}
