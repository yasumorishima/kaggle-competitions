"""Brief-reading explorer (ours, v1): no model, CPU only.

What it does, per game:
- Reads info["controls"] and info["description"] to pick its input families: arrow keys / WASD held (real-time
  movement) or pressed once per step (turn-based, "one grid step per press"), mouse aim with the left button held
  ("hold the left mouse button"), clicks on the board ("click", "dig"), and a start key ("press Space", "any key",
  "click Play", "press R to play again").
- Opens with the start inputs the brief names, then plays.
- Watches the screen: a frame that stops changing for STUCK_S seconds is a menu, a game-over screen or a dead end,
  so it replays the start inputs, then clicks a sweep of likely button spots.
- Learns, per game, which input families change the screen most (a bandit on the mean frame difference after each
  choice) and picks them more often. A changing screen is a weak but universal sign that the input does something.
"""

import io
import math
import random
import re

try:
    from gga.protocol import Action
except Exception:  # local unit use without the harness
    class Action:  # type: ignore
        def __init__(self, tool, params=None, meta=None):
            self.tool, self.params, self.meta = tool, params or {}, meta

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

STUCK_S = 2.5          # a screen this long unchanged is treated as a menu / game over
DIFF_STILL = 1.2       # mean absolute grey difference (0-255) below which two frames count as the same
RESTARTS = ("reload", "relaunch")


def _thumb(jpeg: bytes):
    if Image is None or not jpeg:
        return None
    try:
        im = Image.open(io.BytesIO(jpeg)).convert("L").resize((48, 32))
        return list(im.getdata())
    except Exception:
        return None


def _diff(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _menu_spots(jpeg: bytes, w: int, h: int):
    """Likely buttons on a still screen: bands of rows with strong horizontal detail (text, button frames),
    returned top to bottom as (x, y) in pixels. Menus usually list Play / Start first, under a title."""
    if Image is None or not jpeg:
        return []
    try:
        im = Image.open(io.BytesIO(jpeg)).convert("L").resize((160, 100))
    except Exception:
        return []
    px = im.load()
    rows = []
    for y in range(100):
        e = sum(abs(px[x + 1, y] - px[x, y]) for x in range(159))
        rows.append(e)
    thr = max(rows) * 0.25 if max(rows) > 0 else 1
    bands, start = [], None
    for y, e in enumerate(rows + [0]):
        if e > thr and start is None:
            start = y
        elif e <= thr and start is not None:
            if y - start >= 2:
                bands.append((start, y))
            start = None
    spots = []
    for a, b in bands:
        cols = [sum(abs(px[x + 1, y] - px[x, y]) for y in range(a, b)) for x in range(159)]
        tot = sum(cols)
        if tot <= 0:
            continue
        cx = sum(i * c for i, c in enumerate(cols)) / tot
        spots.append((int((cx + 0.5) * w / 160), int(((a + b) / 2) * h / 100)))
    return spots[:8]


class Brief:
    """What the brief says about inputs."""

    def __init__(self, info: dict):
        t = ((info.get("controls") or "") + " \n " + (info.get("description") or "")).lower()
        self.text = t
        self.arrows = "arrow" in t
        self.wasd = "wasd" in t or re.search(r"\bw\s*,?\s*a\s*,?\s*s\s*,?\s*d\b", t) is not None
        self.step_keys = "per press" in t or "turn-based" in t or "one grid step" in t or "each press" in t
        self.mouse_aim = "aim" in t or "hold the left mouse" in t or "fire" in t
        self.clicks = bool(re.search(r"\bclick|\bdig|\bselect|\btap", t))
        self.right_click = "right-click" in t or "right click" in t
        self.space = "space" in t
        self.enter = "enter" in t
        self.any_key = "any key" in t
        self.restart_key = None
        m = re.search(r"press\s+'?\"?([a-z0-9])'?\"?\s+to\s+(?:play|restart|try|start)", t)
        if m:
            self.restart_key = m.group(1)
        self.build_keys = sorted(set(re.findall(r"\b([qwer])\b(?=[^.]*build)", t))) if "build" in t else []
        self.play_button = "play" in t and "click" in t
        self.jump = "jump" in t
        self.shift = "shift" in t

    def openers(self):
        o = []
        if self.play_button:
            o.append(("click_center",))
        if self.space or self.any_key:
            o.append(("key", "Space"))
        if self.enter:
            o.append(("key", "Enter"))
        if self.restart_key:
            o.append(("key", self.restart_key))
        if not o:
            o = [("key", "Space"), ("key", "Enter"), ("click_center",)]
        return o


class Agent:
    def __init__(self) -> None:
        self.rng = random.Random(12345)
        self.last_elapsed = float("inf")
        self._new_game(None)

    # ---- per-game state -------------------------------------------------------------------------------------
    def _new_game(self, obs) -> None:
        self.turn = 0
        self.restarts = 0
        self.brief = Brief(obs.info) if obs is not None else None
        self.prev = None
        self.still_since = None
        self.queue: list = []
        self.arm = None
        self.arm_stats: dict = {}
        self.sweep_i = 0
        self.tried: set = set()
        self.w = self.h = None

    def _families(self):
        b = self.brief
        fam = []
        if b.arrows or b.wasd:
            fam += ["step"] if b.step_keys else ["hold_move", "hold_move", "tap_move"]
        if b.mouse_aim:
            fam += ["aim_fire"]
        if b.clicks:
            fam += ["click_rand"]
        if b.right_click:
            fam += ["right_click"]
        if b.build_keys:
            fam += ["build"]
        if b.jump or b.space:
            fam += ["space"]
        if not fam:
            fam = ["hold_move", "click_rand", "space", "tap_move"]
        return sorted(set(fam), key=fam.index)

    def _move_keys(self):
        b = self.brief
        keys = []
        if b.arrows or not b.wasd:
            keys += ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]
        if b.wasd and not b.arrows:
            keys += ["w", "a", "s", "d"]
        return keys

    def _xy(self):
        # board-ish area: keep away from the very edges (edge scrolling, letterbox bars)
        return (int(self.w * self.rng.uniform(0.12, 0.88)), int(self.h * self.rng.uniform(0.12, 0.88)))

    # ---- action builders ------------------------------------------------------------------------------------
    def _act(self, fam):
        r = self.rng
        if fam == "step":
            return Action("press_key", {"key": r.choice(self._move_keys())})
        if fam == "tap_move":
            return Action("press_key", {"key": r.choice(self._move_keys())})
        if fam == "hold_move":
            ks = self._move_keys()
            k = [r.choice(ks)]
            if r.random() < 0.3:
                k.append(r.choice(ks))
            return Action("sequence", {"steps": [{"keys": sorted(set(k)), "duration_ms": r.choice([400, 700, 1000, 1500])},
                                                 {"keys": [], "duration_ms": 0}]})
        if fam == "aim_fire":
            x, y = self._xy()
            ks = self._move_keys()
            k = [r.choice(ks)] if r.random() < 0.7 else []
            return Action("sequence", {"steps": [{"keys": k, "buttons": ["left"], "x": x, "y": y,
                                                  "duration_ms": r.choice([500, 900, 1300])},
                                                 {"keys": [], "buttons": ["left"], "x": x, "y": y, "duration_ms": 0}]})
        if fam == "click_rand":
            x, y = self._xy()
            return Action("click", {"x": x, "y": y})
        if fam == "right_click":
            x, y = self._xy()
            return Action("click", {"x": x, "y": y, "button": "right"})
        if fam == "build":
            x, y = self._xy()
            return Action("sequence", {"steps": [{"keys": [], "x": x, "y": y, "duration_ms": 100},
                                                 {"keys": [r.choice(self.brief.build_keys)], "x": x, "y": y, "duration_ms": 100},
                                                 {"keys": [], "x": x, "y": y, "duration_ms": 0}]})
        if fam == "space":
            return Action("press_key", {"key": "Space"})
        return Action("skip", {})

    def _opener_action(self, o):
        if o[0] == "click_center":
            return Action("click", {"x": self.w // 2, "y": self.h // 2})
        if o[0] == "click":
            return Action("click", {"x": o[1], "y": o[2]})
        return Action("press_key", {"key": o[1]})

    def _unstick(self, jpeg=None):
        """Start inputs again, then a sweep of likely button spots: the detected text/button bands first (top to
        bottom, the title band last), then fixed spots (centre, lower centre, ...)."""
        seq = list(self.brief.openers())
        found = _menu_spots(jpeg, self.w, self.h) if jpeg else []
        if len(found) >= 2:
            found = found[1:] + found[:1]       # the top band is usually the title
        for x, y in found[:4]:
            if ("click", x, y) not in self.tried:
                self.tried.add(("click", x, y))
                seq.insert(0, ("click", x, y))
                break
        spots = [(0.5, 0.5), (0.5, 0.62), (0.5, 0.72), (0.5, 0.4), (0.5, 0.82), (0.35, 0.6), (0.65, 0.6), (0.5, 0.3)]
        fx, fy = spots[self.sweep_i % len(spots)]
        self.sweep_i += 1
        seq.append(("click", int(self.w * fx), int(self.h * fy)))
        for k in ("Space", "Enter", "r"):
            if ("key", k) not in seq:
                seq.append(("key", k))
        return seq

    # ---- bandit ---------------------------------------------------------------------------------------------
    def _choose(self):
        fams = self._families()
        best, bv = None, -1.0
        for f in fams:
            n, s = self.arm_stats.get(f, (0, 0.0))
            v = (s / n if n else 50.0) + 6.0 * math.sqrt(math.log(self.turn + 2) / (n + 1))
            if v > bv:
                best, bv = f, v
        return best

    # ---- the contract ---------------------------------------------------------------------------------------
    def step(self, obs, tools):
        info = obs.info or {}
        if obs.elapsed_s < self.last_elapsed or (obs.elapsed_s == self.last_elapsed and not obs.last_action):
            self._new_game(obs)
        elif self.restarts < 3 and any(d.get("code") in RESTARTS for d in (obs.last_action or {}).get("dropped", [])):
            self.restarts += 1
            self.queue = list(self.brief.openers())
        self.last_elapsed = obs.elapsed_s
        self.turn += 1
        if self.w is None:
            self.w, self.h = int(info.get("width", 1600)), int(info.get("height", 1000))
            self.queue = list(self.brief.openers())
            if self.brief.play_button:
                spots = _menu_spots(obs.image, self.w, self.h)
                if len(spots) >= 2:
                    x, y = spots[1]
                    self.tried.add(("click", x, y))
                    self.queue = [("click", x, y)] + [q for q in self.queue if q[0] != "click_center"]

        cur = _thumb(obs.image)
        d = _diff(cur, self.prev) if self.prev is not None else 0.0
        self.prev = cur
        if self.arm is not None:
            n, s = self.arm_stats.get(self.arm, (0, 0.0))
            self.arm_stats[self.arm] = (n + 1, s + min(d, 60.0))
            self.arm = None
        if d < DIFF_STILL:
            self.still_since = obs.elapsed_s if self.still_since is None else self.still_since
        else:
            self.still_since = None

        if self.queue:
            return self._opener_action(self.queue.pop(0))
        if self.still_since is not None and obs.elapsed_s - self.still_since > STUCK_S:
            self.still_since = obs.elapsed_s
            self.queue = self._unstick(obs.image)
            return self._opener_action(self.queue.pop(0))

        # let a running hold finish before replacing it (a reply replaces the rest of the program)
        if (obs.program_left_s or 0) > 0.25:
            return Action("skip", {})
        fam = self._choose()
        self.arm = fam
        return self._act(fam)
