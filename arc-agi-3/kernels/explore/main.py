"""ARC-AGI-3 explorer (no LLM): a state graph over frames, tried action by action.

Each distinct frame is a node; every (node, action) is tried once and the edge is
remembered. When the current node has nothing left to try, the agent walks the
known edges to the nearest node that does, and RESETs (back to the level start)
when no such node is reachable. A new level clears the graph.

Cells that change after almost every action (move counters, timers) would make
every frame a new node, so they are masked out of the node key once seen.
ACTION6 (click) is tried on one cell per connected same-colour object.

On a competition rerun the games come from the Kaggle gateway; otherwise the 25
public games in environment_files are played offline and the scorecard printed.
"""
import os
import subprocess
import sys
import time
from collections import deque

COMP_DIR = os.getenv("ARC3_COMP_DIR", "/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
WORK = os.getenv("ARC3_WORK", "/kaggle/working")   # both overridable for a local run
RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

if not os.getenv("ARC3_LOCAL"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links",
                           COMP_DIR + "/arc_agi_3_wheels", "arc-agi", "python-dotenv"])

import numpy as np  # noqa: E402

# Budgets: 110 games in under 9 hours (leave margin for the gateway and start-up).
TOTAL_BUDGET_S = 7.5 * 3600 if RERUN else float(os.getenv("ARC3_BUDGET_S", 2.0 * 3600))  # the offline check (25 games) stays short
MAX_ACTIONS = int(os.getenv("ARC3_MAX_ACTIONS", "2500"))
MAX_CLICKS = 24          # click targets per node
VOLATILE_FRAC = 0.6      # a cell changing on this share of moves is masked
NOOP_MIN = int(os.getenv("ARC3_NOOP_MIN", "0"))  # tries of a move kind before its no-op rate counts (0 = off; LB: off 0.15, 2 0.09)
EDGE = 3                 # rows/columns this close to the border may hold a counter bar
LEVEL_LOG = {}           # game -> actions (RESETs included) spent on each completed level, for comp_score


def write_env():
    if RERUN:
        subprocess.call("curl -s --fail --retry 999 --retry-all-errors --retry-delay 5 "
                        "--retry-max-time 600 http://gateway:8001/api/games > /dev/null", shell=True)
    mode = "online" if RERUN else "offline"
    envdir = "" if RERUN else COMP_DIR + "/environment_files/"
    with open(WORK + "/.env", "w") as f:
        f.write("SCHEME=http\nHOST=gateway\nPORT=8001\nARC_API_KEY=test-key-123\n"
                "ARC_BASE_URL=http://gateway:8001/\n"
                f"OPERATION_MODE={mode}\nENVIRONMENTS_DIR={envdir}\n"
                f"RECORDINGS_DIR={WORK}/server_recording\n")


def grid_of(resp):
    fr = getattr(resp, "frame", None)
    if fr is None or len(fr) == 0:
        return None
    return np.asarray(fr[-1], dtype=np.int16)


def click_targets(g):
    """One cell per 4-connected same-colour object, small objects first (buttons)."""
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    bg = np.bincount(g.ravel(), minlength=16).argmax()
    objs = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or g[y, x] == bg:
                continue
            c = g[y, x]
            q = [(y, x)]
            seen[y, x] = True
            cells = []
            while q:
                cy, cx = q.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] == c:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            cy, cx = cells[len(cells) // 2]
            objs.append((len(cells), int(cx), int(cy)))
    objs.sort()
    return [(x, y) for _, x, y in objs[:MAX_CLICKS]]


class Explorer:
    def __init__(self, actions):
        self.actions = actions          # GameAction members other than RESET
        self.reset_level()

    noop = None                         # move kind -> [no-op count, tries]; kept across levels

    def reset_level(self):
        if self.noop is None:
            self.noop = {}
        self.edges = {}                 # node -> {move: node}
        self.todo = {}                  # node -> [move, ...] not yet tried
        self.changes = None             # per-cell change counts
        self.n_moves = 0
        self.mask = None

    def key(self, g):
        if self.mask is not None and self.mask.shape == g.shape:
            g = np.where(self.mask, -1, g)
        return g.tobytes() + bytes(str(g.shape), "ascii")

    def moves_for(self, g):
        out = []
        for a in self.actions:
            if a.is_complex():
                out += [(a, xy) for xy in click_targets(g)]
            else:
                out.append((a, None))
        return out

    def observe(self, prev_g, g):
        if prev_g is None or g is None or prev_g.shape != g.shape:
            return
        if self.changes is None or self.changes.shape != g.shape:
            self.changes = np.zeros(g.shape, dtype=np.int32)
            self.row_ch = np.zeros(g.shape[0], dtype=np.int32)
            self.col_ch = np.zeros(g.shape[1], dtype=np.int32)
        d = prev_g != g
        self.changes += d
        self.row_ch += d.any(1)
        self.col_ch += d.any(0)
        self.n_moves += 1
        if self.n_moves >= 8:
            lim = VOLATILE_FRAC * self.n_moves
            m = self.changes >= lim
            # a counter bar near the edge changes one cell per move: mask the whole edge row/column
            h, w = g.shape
            for r in list(range(min(EDGE, h))) + list(range(max(0, h - EDGE), h)):
                if self.row_ch[r] >= lim:
                    m[r, :] = True
            for c in list(range(min(EDGE, w))) + list(range(max(0, w - EDGE), w)):
                if self.col_ch[c] >= lim:
                    m[:, c] = True
            # a mask over most of the screen means the whole scene moves: do not mask
            self.mask = m if m.mean() < 0.2 else None

    def node(self, g):
        k = self.key(g)
        if k not in self.todo:
            self.todo[k] = self.moves_for(g)
            self.edges[k] = {}
        return k

    @staticmethod
    def kind(g, mv):
        """What a move is, independent of the state: the action, plus the clicked colour."""
        a, xy = mv
        return (int(a.value), int(g[xy[1], xy[0]]) if xy is not None else -1)

    def pick(self, g, todo):
        """Untried move whose kind has most often changed the frame (unknown kinds first)."""
        def rate(mv):
            n0, n = self.noop.get(self.kind(g, mv), (0, 0))
            return (n0 + 0.5) / (n + 1.0) if n >= NOOP_MIN else 0.0
        i = min(range(len(todo)), key=lambda j: (rate(todo[j]), j))
        return todo.pop(i)

    def record(self, g, mv, changed):
        k = self.kind(g, mv)
        n0, n = self.noop.get(k, (0, 0))
        self.noop[k] = (n0 + (not changed), n + 1)

    def path_to_todo(self, start):
        """Shortest known move sequence from start to a node with untried moves."""
        prev = {start: None}
        dq = deque([start])
        while dq:
            u = dq.popleft()
            if self.todo.get(u):
                path = []
                while prev[u] is not None:
                    p, mv = prev[u]
                    path.append(mv)
                    u = p
                return path[::-1]
            for mv, v in self.edges.get(u, {}).items():
                if v not in prev:
                    prev[v] = (u, mv)
                    dq.append(v)
        return None


def play(env, game_id, deadline):
    from arcengine import GameAction, GameState

    resp = env._last_response
    if resp is None or resp.state in (GameState.NOT_PLAYED,):
        resp = env.step(GameAction.RESET, {})
    avail = getattr(resp, "available_actions", None) or []
    acts = [a for a in GameAction if a is not GameAction.RESET]
    if avail:
        ids = set(int(getattr(x, "value", x)) for x in avail)
        acts = [a for a in acts if int(a.value) in ids] or acts
    ex = Explorer(acts)
    level = resp.levels_completed
    plan = []
    n = 0
    start_n, LEVEL_LOG[game_id] = 0, []
    while n < MAX_ACTIONS and time.time() < deadline:
        if resp is None:
            break
        if resp.state == GameState.WIN:
            break
        if resp.state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
            resp = env.step(GameAction.RESET, {})
            n += 1
            plan = []
            continue
        g = grid_of(resp)
        if g is None:
            resp = env.step(GameAction.RESET, {})
            n += 1
            continue
        cur = ex.node(g)
        if plan:
            mv = plan.pop(0)
        elif ex.todo[cur]:
            mv = ex.pick(g, ex.todo[cur]) if NOOP_MIN else ex.todo[cur].pop(0)
        else:
            path = ex.path_to_todo(cur)
            if not path:
                resp = env.step(GameAction.RESET, {})
                n += 1
                continue
            mv, plan = path[0], path[1:]
        a, xy = mv
        data = {"x": xy[0], "y": xy[1]} if xy is not None else {}
        nxt = env.step(a, data)
        n += 1
        if nxt is None:
            break
        g2 = grid_of(nxt)
        if nxt.levels_completed > level:
            level = nxt.levels_completed
            LEVEL_LOG[game_id].append(n - start_n)
            start_n = n
            ex.reset_level()
            plan = []
        elif g2 is not None and nxt.state == GameState.NOT_FINISHED:
            ex.record(g, mv, g2.shape != g.shape or bool((g2 != g).any()))
            ex.observe(g, g2)
            nk = ex.node(g2)
            if ex.edges[cur].get(mv, nk) != nk:
                plan = []       # not deterministic from here: re-plan
            ex.edges[cur][mv] = nk
        resp = nxt
    return f"{game_id}: actions={n} state={resp.state.name if resp else '?'} levels={resp.levels_completed if resp else 0}"


def comp_score(arcade):
    """The competition's score from LEVEL_LOG: levels weighted 1..n, each min(115, 100 (baseline / actions)^2).

    Needed offline with ONLY_RESET_LEVELS=true (the competition's setting: RESET only restarts the level,
    and costs an action), where the local scorecard never opens a play.
    """
    base = {e.game_id: e.baseline_actions or [] for e in arcade.available_environments}
    out = {}
    for gid, acts in LEVEL_LOG.items():
        b = base.get(gid) or []
        if not b:
            continue
        w = np.arange(1, len(b) + 1)
        s = np.array([min(115.0, 100.0 * (b[i] / acts[i]) ** 2) if i < len(acts) and acts[i] > 0 else 0.0
                      for i in range(len(b))])
        out[gid] = min(float((w * s).sum() / w.sum()), 100.0 * float(w[s > 0].sum()) / w.sum())
    return float(np.mean([out.get(g, 0.0) for g in base])), out


def main():
    import arc_agi
    import dotenv

    os.chdir(WORK)
    write_env()
    dotenv.load_dotenv(dotenv_path=".env", override=True)
    arcade = arc_agi.Arcade()
    games = [e.game_id for e in arcade.available_environments]
    print(f"{len(games)} games, rerun={RERUN}", flush=True)
    t0 = time.time()
    for i, gid in enumerate(games):
        left = TOTAL_BUDGET_S - (time.time() - t0)
        per = left / max(1, len(games) - i)
        try:
            print(play(arcade.make(gid), gid, time.time() + per), f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # one bad game must not lose the others
            print(f"{gid}: error {type(e).__name__}: {e}", flush=True)
    if not RERUN:
        cs, per = comp_score(arcade)
        print(f"Comp score (level resets only): {cs:.4f}  levels {sum(len(v) for v in LEVEL_LOG.values())}")
        for g, v in sorted(per.items()):
            if v > 0:
                print(f"  {g:<20} {v:8.2f} {LEVEL_LOG[g]}")
        sc = arcade.get_scorecard()
        print(f"Score: {sc.score:.4f}  levels {sc.total_levels_completed}/{sc.total_levels}  actions {sc.total_actions}")
        for e in sc.environments:
            print(f"  {e.id:<20} {e.score:8.2f} {e.levels_completed:4} {e.actions:6}")
        import pandas as pd
        pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]) \
            .to_parquet(WORK + "/submission.parquet", index=False)


if __name__ == "__main__":
    main()
