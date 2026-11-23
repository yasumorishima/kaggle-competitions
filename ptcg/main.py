"""Rule-based agent for the PTCG AI Battle Challenge Playground (cabt engine).

Stand-alone: no import of the engine's `cg` package, only the standard library.
The observation is read as a plain dict (same fields as cg.api.Observation).

Deck: Mega Abomasnow ex / Snover (water), see deck.csv and CLAUDE.md.
Idea taken from the public rule agent "previous_rule" in matterhorn3838/pokemon-challenge-ultra-900
(option scoring per OptionType); rewritten and extended here (readiness-aware energy, switching via
Surfing Beach / retreat, deck-out guard for Hammer-lanche, KO-aware attack choice, safe fallback).
"""
import os
import random

# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------
WATER = 3
SNOVER, ABOMASNOW, KYOGRE = 722, 723, 721
MEGA_SIGNAL, MAX_BELT, CYRANO, LILLIE, WAITRESS, SURF_BEACH = 1145, 1158, 1205, 1227, 1235, 1262
ULTRA_BALL, BROCK, SWITCH = 1121, 1210, 1123

DECK = ([SNOVER] * 4 + [ABOMASNOW] * 4 + [KYOGRE] * 2 + [MEGA_SIGNAL] * 4 + [ULTRA_BALL] * 3
        + [MAX_BELT] + [LILLIE] * 4 + [WAITRESS] * 4 + [SURF_BEACH] * 2 + [CYRANO])
DECK += [WATER] * (60 - len(DECK))


def _read_deck():
    # kaggle-environments exec()s this file with an empty globals dict, so __file__ may be undefined.
    paths = ["/kaggle_simulations/agent/deck.csv", "deck.csv"]
    try:
        paths.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv"))
    except NameError:
        pass
    for p in paths:
        try:
            with open(p) as f:
                d = [int(x) for x in f.read().split() if x.strip()]
            if len(d) == 60:
                return d
        except Exception:
            pass
    return list(DECK)


DECK = _read_deck()
WATER_IN_DECK = sum(1 for c in DECK if c == WATER)

# Known card facts for our own cards (from the engine's AllCard/AllAttack).
POKEMON = {
    SNOVER: dict(hp=90, retreat=3, need=2, stage=0),
    ABOMASNOW: dict(hp=350, retreat=4, need=2, stage=1),
    KYOGRE: dict(hp=150, retreat=3, need=3, stage=0),
}
ATTACK_COST = {1042: 1, 1043: 3, 1044: 1, 1045: 2, 1046: 2, 1047: 3}
ATTACK_DMG = {1043: 130, 1044: 10, 1045: 30, 1047: 200}

# Enum values (cg.api)
A_DECK, A_HAND, A_DISCARD, A_ACTIVE, A_BENCH, A_PRIZE, A_STADIUM = 1, 2, 3, 4, 5, 6, 7
A_LOOKING = 12
O_NUMBER, O_YES, O_NO, O_CARD, O_TOOL, O_ENERGYCARD, O_ENERGY = 0, 1, 2, 3, 4, 5, 6
O_PLAY, O_ATTACH, O_EVOLVE, O_ABILITY, O_DISCARD, O_RETREAT, O_ATTACK, O_END = 7, 8, 9, 10, 11, 12, 13, 14
# Contexts where the selected cards are lost / sent away (pick the least valuable).
NEG_CONTEXTS = {8, 9, 10, 11, 23, 26, 27, 28, 29, 30, 31, 32, 33, 36}
# Contexts where an opponent's Pokemon is usually the target (prefer weak/low HP).
HARM_CONTEXTS = {13, 14, 15, 25, 20}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _g(d, k, default=None):
    try:
        v = d.get(k, default)
    except AttributeError:
        v = getattr(d, k, default)
    return default if v is None else v


class Ctx:
    def __init__(self, obs):
        self.obs = obs
        self.sel = _g(obs, "select", {})
        self.cur = _g(obs, "current", {})
        self.me = int(_g(self.cur, "yourIndex", 0))
        players = _g(self.cur, "players", [{}, {}])
        self.my = players[self.me]
        self.op = players[1 - self.me]

    # cards ---------------------------------------------------------------
    def zone(self, area, owner):
        p = self.cur["players"][owner] if owner in (0, 1) else self.my
        if area == A_DECK:
            return _g(self.sel, "deck", [])
        if area == A_HAND:
            return _g(p, "hand", [])
        if area == A_DISCARD:
            return _g(p, "discard", [])
        if area == A_ACTIVE:
            return _g(p, "active", [])
        if area == A_BENCH:
            return _g(p, "bench", [])
        if area == A_PRIZE:
            return _g(p, "prize", [])
        if area == A_STADIUM:
            return _g(self.cur, "stadium", [])
        if area == A_LOOKING:
            return _g(self.cur, "looking", [])
        return []

    def card(self, opt, target=False):
        area = opt.get("inPlayArea" if target else "area")
        idx = opt.get("inPlayIndex" if target else "index")
        owner = self.me if target else opt.get("playerIndex", self.me)
        if owner is None:
            owner = self.me
        if area is None or idx is None:
            return None
        z = self.zone(area, owner)
        if 0 <= idx < len(z):
            return z[idx]
        return None

    def my_inplay(self):
        return [p for p in (_g(self.my, "active", []) + _g(self.my, "bench", [])) if p]

    def my_active(self):
        a = _g(self.my, "active", [])
        return a[0] if a else None

    def op_active(self):
        a = _g(self.op, "active", [])
        return a[0] if a else None

    def hand_ids(self):
        return [c.get("id") for c in _g(self.my, "hand", []) if c]

    def water_remaining_frac(self):
        seen = sum(1 for c in _g(self.my, "hand", []) if c and c.get("id") == WATER)
        seen += sum(1 for c in _g(self.my, "discard", []) if c and c.get("id") == WATER)
        for p in self.my_inplay():
            seen += sum(1 for c in _g(p, "energyCards", []) if c and c.get("id") == WATER)
        for c in _g(self.my, "prize", []):
            if c and c.get("id") == WATER:
                seen += 1
        unknown = _g(self.my, "deckCount", 0) + sum(1 for c in _g(self.my, "prize", []) if c is None)
        return max(0, WATER_IN_DECK - seen) / max(1, unknown)


def energy_count(p):
    return len(_g(p, "energies", [])) if p else 0


def need_of(p):
    info = POKEMON.get(p.get("id")) if p else None
    return info["need"] if info else 2


def ready(p):
    return p is not None and energy_count(p) >= need_of(p) and p.get("id") in (ABOMASNOW, KYOGRE)


def attack_value(c, attack_id):
    deck_n = _g(c.my, "deckCount", 0)
    if attack_id == 1046:
        dmg = 100 * min(6, deck_n) * c.water_remaining_frac()
    elif attack_id == 1042:
        dmg = 20 * sum(1 for x in _g(c.my, "discard", []) if x and x.get("id") == WATER)
    else:
        dmg = ATTACK_DMG.get(attack_id, 30)
    oa = c.op_active()
    v = dmg
    if oa is not None and dmg >= oa.get("hp", 999):
        v += 1000
        my_prizes = len(_g(c.my, "prize", []))
        if my_prizes <= 3:
            v += 2000  # may be game-winning
    if attack_id == 1046 and deck_n <= 7 and v < 3000:
        v -= 2000  # deck-out guard: milling 6 leaves us unable to draw
    if attack_id == 1043:
        v -= 20
    return v


def keep_value(cid, c):
    """How much we want to keep / obtain this card (own perspective)."""
    hand = c.hand_ids()
    inplay = [p.get("id") for p in c.my_inplay()]
    if cid == ABOMASNOW:
        return 320 if (SNOVER in inplay or SNOVER in hand) else 220
    if cid == SNOVER:
        if not any(x in (SNOVER, ABOMASNOW) for x in inplay):
            return 330
        return 260 if ABOMASNOW in hand else 200
    if cid == KYOGRE:
        return 120
    if cid in (WAITRESS, LILLIE, BROCK):
        return 150
    if cid == SWITCH:
        return 100
    if cid in (MEGA_SIGNAL, ULTRA_BALL):
        return 130
    if cid == CYRANO:
        return 125
    if cid == MAX_BELT:
        return 110
    if cid == SURF_BEACH:
        return 90
    if cid == WATER:
        return 60
    return 80


def pokemon_score(p, c, for_active=False):
    """Value of one of our in-play Pokemon as the attacker."""
    if not p:
        return -1
    pid = p.get("id")
    s = energy_count(p) * 150 + p.get("hp", 0) * 0.5
    if pid == ABOMASNOW:
        s += 400
    elif pid == KYOGRE:
        s += 150
    if ready(p):
        s += 600
    return s


def attach_target_value(p, c, area):
    if not p:
        return 0
    pid = p.get("id")
    e = energy_count(p)
    need = need_of(p)
    if pid not in POKEMON:
        return 100
    if e >= (3 if pid == ABOMASNOW else need):
        return 50
    v = 1000 + 100 * (need - e)
    if area == A_ACTIVE:
        v += 400
    if pid == ABOMASNOW:
        v += 300
    elif pid == SNOVER:
        v += 150
    if e + 1 >= need:
        v += 200
    return v


# ---------------------------------------------------------------------------
# MAIN phase
# ---------------------------------------------------------------------------
def main_value(c, o):
    t = o.get("type")
    if t == O_END:
        return 0
    if t == O_EVOLVE:
        tgt = c.card(o, target=True)
        return 5000 + (300 if o.get("inPlayArea") == A_ACTIVE else 0) + energy_count(tgt) * 10
    if t == O_ATTACK:
        return 900 + attack_value(c, o.get("attackId")) / 10.0
    if t == O_ATTACH:
        src = c.card(o) or {}
        tgt = c.card(o, target=True)
        if src.get("id") == MAX_BELT:
            return 2500 + (500 if tgt and tgt.get("id") == ABOMASNOW else 0) + (200 if o.get("inPlayArea") == A_ACTIVE else 0)
        v = attach_target_value(tgt, c, o.get("inPlayArea"))
        return 1500 + v if v >= 1000 else 200 + v
    if t == O_PLAY:
        cid = (c.card({"area": A_HAND, "index": o.get("index"), "playerIndex": c.me}) or {}).get("id")
        return play_value(c, cid)
    if t == O_ABILITY:
        src = c.card(o) or {}
        if src.get("id") == SURF_BEACH or o.get("area") == A_STADIUM:
            return switch_benefit(c, 2800)
        return 300
    if t == O_RETREAT:
        act = c.my_active()
        if act is None:
            return -1
        info = POKEMON.get(act.get("id"), {})
        if energy_count(act) < info.get("retreat", 9):
            return -1
        return switch_benefit(c, 1200, cost=info.get("retreat", 3))
    if t == O_DISCARD:
        return -1
    return 10


def switch_benefit(c, base, cost=0):
    act = c.my_active()
    bench = [p for p in _g(c.my, "bench", []) if p]
    if not bench:
        return -1
    best = max(bench, key=lambda p: pokemon_score(p, c))
    if ready(act) and energy_count(act) - cost >= 0 and act.get("id") == ABOMASNOW:
        return -1
    if not ready(best):
        return -1
    if pokemon_score(best, c) <= pokemon_score(act, c) + 100:
        return -1
    return base


def play_value(c, cid):
    hand = c.hand_ids()
    inplay = [p.get("id") for p in c.my_inplay()]
    bench = [p for p in _g(c.my, "bench", []) if p]
    bench_room = _g(c.my, "benchMax", 5) - len(bench)
    deck_n = _g(c.my, "deckCount", 0)
    supporter_done = bool(_g(c.cur, "supporterPlayed", False))
    if cid == SNOVER:
        return 4200 if bench_room > 0 else -1
    if cid == KYOGRE:
        if bench_room <= 0:
            return -1
        return 3000 if inplay.count(KYOGRE) == 0 else 1200
    if cid == MEGA_SIGNAL:
        if ABOMASNOW in hand:
            return -1
        snovers = inplay.count(SNOVER) + hand.count(SNOVER)
        return 3600 if snovers > 0 else 400
    basics_in_hand = sum(1 for x in hand if x in (SNOVER, KYOGRE))
    lines = inplay.count(SNOVER) + inplay.count(ABOMASNOW)
    if cid == BROCK:
        if supporter_done or bench_room <= 0:
            return -1
        if not bench and basics_in_hand == 0:
            return 3900
        return 3000 if lines + hand.count(SNOVER) < 3 else 400
    if cid == SWITCH:
        return switch_benefit(c, 2700)
    if cid == ULTRA_BALL:
        if len(hand) < 3:
            return -1
        if not any(p.get("id") == SNOVER for p in c.my_inplay()) and SNOVER not in hand and bench_room > 0:
            return 3700
        snovers = inplay.count(SNOVER) + hand.count(SNOVER)
        return 3500 if (ABOMASNOW not in hand and snovers > 0) else -1
    if cid == CYRANO:
        if supporter_done:
            return -1
        snovers = inplay.count(SNOVER) + hand.count(SNOVER)
        return 3300 if (ABOMASNOW not in hand and snovers > 0) else 300
    if cid == WAITRESS:
        if supporter_done or deck_n < 6:
            return -1
        need = any(energy_count(p) < need_of(p) for p in c.my_inplay())
        return 3100 if need else 700
    if cid == LILLIE:
        if supporter_done or deck_n <= 8:
            return -1
        n = len(hand) - 1
        if not bench and basics_in_hand == 0:
            return 3400
        if n <= 2:
            return 3200
        if n <= 4:
            return 2000
        return 500
    if cid == SURF_BEACH:
        st = _g(c.cur, "stadium", [])
        if st and st[0] and st[0].get("id") == SURF_BEACH:
            return -1
        return 2200
    if cid == MAX_BELT:
        return 2400
    if cid == WATER:
        return 1600
    return 800


# ---------------------------------------------------------------------------
# Non-MAIN selections
# ---------------------------------------------------------------------------
def option_value(c, o, context):
    t = o.get("type")
    if t == O_YES:
        return 1
    if t == O_NO:
        return 0
    if t == O_NUMBER:
        n = o.get("number", 0) or 0
        return n
    if t in (O_CARD, O_TOOL, O_ENERGYCARD, O_ENERGY):
        card = c.card(o) or {}
        cid = card.get("id", o.get("cardId"))
        owner = o.get("playerIndex", c.me)
        mine = owner == c.me or owner is None
        area = o.get("area")
        if context in (1,):  # setup active
            return {SNOVER: 300, KYOGRE: 200}.get(cid, 100) + card.get("hp", 0) * 0.1
        if context in (3, 4) and mine and area in (A_ACTIVE, A_BENCH):
            return pokemon_score(card, c)
        if context in (21, 22) and area in (A_ACTIVE, A_BENCH) and mine:
            return attach_target_value(card, c, area)
        if context in (16, 17):  # heal / remove damage: most damaged own Pokemon
            return (card.get("maxHp", 0) - card.get("hp", 0)) if mine else -100
        if context in HARM_CONTEXTS or (not mine and area in (A_ACTIVE, A_BENCH)):
            if mine:
                return -pokemon_score(card, c) if area in (A_ACTIVE, A_BENCH) else -keep_value(cid, c)
            hp = card.get("hp", 300)
            return 1000 - hp + (100 if area == A_ACTIVE else 0)
        if context == 3 and not mine:  # gust-like: bring up the weakest opponent
            return 1000 - card.get("hp", 300)
        v = keep_value(cid, c) if mine else 100
        if t == O_ENERGY or t == O_ENERGYCARD:
            v = 60 + (o.get("count", 1) or 1)
            tgt = card
            if tgt:
                v += energy_count(tgt) * 0 + (200 if tgt.get("id") == ABOMASNOW else 0)
                v += 100 if area == A_ACTIVE else 0
        return v
    return 0


def choose(obs):
    c = Ctx(obs)
    sel = c.sel
    opts = _g(sel, "option", [])
    n = len(opts)
    lo = int(_g(sel, "minCount", 0))
    hi = int(_g(sel, "maxCount", 1))
    hi = max(0, min(hi, n))
    lo = max(0, min(lo, hi))
    context = _g(sel, "context", 0)
    if n == 0 or hi == 0:
        return []
    if context == 0:
        vals = [main_value(c, o) for o in opts]
        best = max(range(n), key=lambda i: (vals[i], -i))
        if vals[best] < 0:
            ends = [i for i, o in enumerate(opts) if o.get("type") == O_END]
            if ends:
                best = ends[0]
        ans = [best]
        if len(ans) < lo:
            ans += [i for i in range(n) if i != best][: lo - len(ans)]
        return ans
    if context == 41:  # go first?
        return [next((i for i, o in enumerate(opts) if o.get("type") == O_YES), 0)]
    vals = [option_value(c, o, context) for o in opts]
    neg = context in NEG_CONTEXTS
    if neg:
        order = sorted(range(n), key=lambda i: (vals[i], i))
        return order[:lo]
    order = sorted(range(n), key=lambda i: (-vals[i], i))
    k = lo
    while k < hi and vals[order[k]] > 0:
        k += 1
    if k == 0 and lo == 0 and context in (2, 5, 6, 7, 24, 37):
        k = min(hi, 1)
    return order[:k]


def _legal(ans, obs):
    sel = _g(obs, "select", {})
    n = len(_g(sel, "option", []))
    lo = int(_g(sel, "minCount", 0))
    hi = min(int(_g(sel, "maxCount", 1)), n)
    if not isinstance(ans, list) or len(set(ans)) != len(ans):
        return False
    if any((not isinstance(x, int)) or x < 0 or x >= n for x in ans):
        return False
    return lo <= len(ans) <= hi


def agent(obs, config=None):
    sel = _g(obs, "select", None)
    if not sel:
        return list(DECK)
    try:
        ans = [int(x) for x in choose(obs)]
        if _legal(ans, obs):
            return ans
    except Exception:
        pass
    n = len(_g(sel, "option", []))
    lo = int(_g(sel, "minCount", 0))
    hi = min(int(_g(sel, "maxCount", 1)), n)
    k = max(lo, min(1, hi))
    return random.sample(list(range(n)), k)
