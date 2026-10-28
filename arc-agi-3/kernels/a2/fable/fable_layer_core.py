r"""Fable layer, core (pure functions, standard library only).

What strong runs do that ours does not, turned into things the harness can compute from the recorded frames
(no model call, nothing written by the model):

  * watch the legend / tray / counter every turn and note which action changed it   -> small_far_changes, Ledger.feed
  * keep per-key notes of what each control did, with the state it was pressed in    -> Ledger.key_lines
  * remember what the previous levels taught (the keys' effects, the clearing run)  -> Ledger.digest_lines
  * record the step budget early and warn when little is left                        -> detect_budget, budget_alert

The text goes into the agent's context only in two places: one short line per turn when an indicator patch changed or the
budget runs low, and one pinned LEVEL LOG when a level has taken a long time (see fable_layer_patch.py). Everything is
conservative: a detector that is unsure says nothing.

Frames are 2-D integer grids (rows of ints 0-15). History entries are (action_name, result_dict, grid_after) in order.
"""
from __future__ import annotations

from collections import Counter

LETTERS = "WwgGcBMPRbSYOrNp"
COLOR_NAMES = {
    "W": "white", "w": "light gray", "g": "gray", "G": "dark gray", "c": "charcoal", "B": "black", "M": "magenta", "P": "pink",
    "R": "red", "b": "blue", "S": "sky blue", "Y": "yellow", "O": "orange", "r": "dark red", "N": "light green", "p": "purple",
}
BORDER = 4  # the harness treats changes within 4 cells of the grid edge as "edge only" (timer bars)


def letter(v):
    try:
        return LETTERS[int(v)]
    except (IndexError, ValueError, TypeError):
        return "?"


def key_of(action):
    """'MOUSE(row=3, col=4)' -> 'MOUSE'; 'UP' -> 'UP'; '' -> ''."""
    a = str(action or "").strip()
    return a.split("(")[0].strip().upper()


def diff_cells(a, b):
    out = []
    for r, (ra, rb) in enumerate(zip(a, b)):
        if ra == rb:
            continue
        for c, (va, vb) in enumerate(zip(ra, rb)):
            if va != vb:
                out.append((r, c))
    return out


def inner_changed(cells, rows=64, cols=64):
    return sum(1 for (r, c) in cells if BORDER <= r < rows - BORDER and BORDER <= c < cols - BORDER)


def components8(cells):
    """8-connected components of a set of (r, c); returns a list of sorted lists."""
    left = set(cells)
    comps = []
    while left:
        seed = left.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    n = (r + dr, c + dc)
                    if n in left:
                        left.remove(n)
                        stack.append(n)
                        comp.append(n)
        comps.append(sorted(comp))
    return comps


def bbox_of(cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return (min(rs), min(cs), max(rs), max(cs))


def bbox_gap(b1, b2):
    """Chebyshev gap between two boxes (0 when they touch or overlap)."""
    dr = max(b2[0] - b1[2], b1[0] - b2[2], 0)
    dc = max(b2[1] - b1[3], b1[1] - b2[3], 0)
    return max(dr, dc)


def _run_len(grid, r, c, dr, dc, value):
    """Number of consecutive cells equal to `value` starting next to (r, c) in direction (dr, dc)."""
    n = 0
    r += dr
    c += dc
    while 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == value:
        n += 1
        r += dr
        c += dc
    return n


def _bar_like(before, after, comp, box):
    """A budget / timer bar changes a few cells at the end of a long straight run of ONE colour.

    The run is measured in the grid where that colour is present (the colour the cells had before, in the grid before;
    the colour they have after, in the grid after), so a dashed beam or a legend icon is not mistaken for a bar.
    """
    r0, c0, r1, c1 = box
    rows, cols = len(before), len(before[0])
    if min(r0, c0, rows - 1 - r1, cols - 1 - c1) > 6:
        return False  # a bar sits at the edge of the frame; a beam or wire in the middle of the field is not a bar
    for grid, pick in ((before, before), (after, after)):
        r, c = comp[0]
        value = pick[r][c]
        if any(pick[rr][cc] != value for rr, cc in comp):
            continue
        horiz = _run_len(grid, r0, c0, 0, -1, value) + _run_len(grid, r0, c1, 0, 1, value) + (c1 - c0 + 1)
        vert = _run_len(grid, r0, c0, -1, 0, value) + _run_len(grid, r1, c0, 1, 0, value) + (r1 - r0 + 1)
        if horiz >= 12 or vert >= 12:
            return True
    return False


def _dominant(grid, comp):
    cnt = Counter(letter(grid[r][c]) for (r, c) in comp)
    return "".join(ch for ch, _ in cnt.most_common(2)), cnt.most_common(1)[0][0]


def _clusters(comps, gap=8):
    """Group components whose boxes are within `gap` cells of each other."""
    boxes = [bbox_of(c) for c in comps]
    parent = list(range(len(comps)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if bbox_gap(boxes[i], boxes[j]) <= gap:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(comps)):
        groups.setdefault(find(i), []).append(i)
    return [[comps[i] for i in idxs] for idxs in groups.values()]


def small_far_changes(before, after, *, small_max=16, box_max=6, gap_min=8, max_events=2):
    """Small patches that changed far away from the main activity of one action (a legend mark, a counter tick).

    The changed cells are grouped into clusters; bar-like cells (timer bars) are dropped first. The biggest cluster (ties:
    the one made of more pieces, a moving thing breaks into several) is the main activity. Every other small, compact
    cluster at least `gap_min` cells away from it is reported, unless it looks like a small object that moved (a vacated and
    an occupied piece of the same size). Returns [] whenever the picture is ambiguous.
    """
    cells = diff_cells(before, after)
    if len(cells) < 3:
        return []
    raw = components8(cells)
    if len(raw) > 60:
        return []  # a busy frame (animation, big redraw): nothing here is a clean indicator signal
    comps = [c for c in raw if not _bar_like(before, after, c, bbox_of(c))]
    clusters = _clusters(comps)
    if len(clusters) < 2:
        return []
    clusters.sort(key=lambda g: (sum(len(c) for c in g), len(g)), reverse=True)
    main = clusters[0]
    mbox = bbox_of([p for c in main for p in c])
    out = []
    for g in clusters[1:]:
        pts = [p for c in g for p in c]
        box = bbox_of(pts)
        if len(pts) < 2 or len(pts) > small_max or box[2] - box[0] + 1 > box_max or box[3] - box[1] + 1 > box_max:
            continue  # one cell is a timer tick; a legend or tray mark is several cells
        if bbox_gap(mbox, box) < gap_min:
            continue
        if len(g) >= 2:
            sizes = sorted(len(c) for c in g)
            if sizes[0] >= 1 and abs(sizes[0] - sizes[-1]) <= 1:
                continue  # a vacated and an occupied piece: a small object moved
        bl, _ = _dominant(before, pts)
        al, _ = _dominant(after, pts)
        out.append({"box": box, "size": len(pts), "before": bl, "after": al})
    if len(out) > max_events:
        return []
    return out


def detect_budget(frames, min_actions=3):
    """Same detector as budget_bar_patch.py: a straight line whose run of equal cells loses cells from one end."""
    k = len(frames) - 1
    if k < min_actions:
        return None
    n = len(frames[0])
    best = None
    lines = [("row", i) for i in range(n)] + [("col", j) for j in range(len(frames[0][0]))]
    for kind, idx in lines:
        def line(frame):
            return list(frame[idx]) if kind == "row" else [row[idx] for row in frame]

        first = line(frames[0])
        runs, start = [], 0
        for i in range(1, len(first) + 1):
            if i == len(first) or first[i] != first[start]:
                if i - start >= 12:
                    runs.append((start, i - 1))
                start = i
        if not runs:
            continue
        if line(frames[-1]) == first:
            continue  # a bar that is being used up differs between the first and the last frame; most lines do not
        series = [line(f) for f in frames]
        changed_prev = set()
        ok = True
        steps_with_change = 0
        max_step = 0
        for t in range(1, len(series)):
            changed = {i for i in range(len(first)) if series[t][i] != first[i]}
            if not changed >= changed_prev:
                ok = False
                break
            if len(changed) > len(changed_prev):
                steps_with_change += 1
                max_step = max(max_step, len(changed) - len(changed_prev))
            changed_prev = changed
        if not ok or not changed_prev or max_step > 6 or steps_with_change < 0.3 * k:
            continue
        for lo, hi in runs:
            d = {i for i in changed_prev if lo <= i <= hi}
            if not d or d != changed_prev:
                continue
            cells = sorted(d)
            contiguous = cells[-1] - cells[0] + 1 == len(cells)
            at_end = cells[0] == lo or cells[-1] == hi
            if not (contiguous and at_end):
                continue
            new_colors = {series[-1][i] for i in cells}
            if len(new_colors) != 1:
                continue
            used, cap = len(cells), hi - lo + 1
            rate = used / k
            if not (0.15 <= rate <= 4.0):
                continue
            cand = dict(kind=kind, index=idx, cap=cap, used=used, left=cap - used, rate=rate,
                        remaining=int((cap - used) / rate), actions=k)
            if best is None or (cap, used) > (best["cap"], best["used"]):
                best = cand
    return best


def budget_alert(b, *, first_time):
    """One line, or '' when nothing is worth saying: the first time a bar is found, then only when it runs low."""
    if not b or b["left"] <= 0:
        return ""  # an empty bar means the attempt is already over; there is nothing left to warn about
    frac = b["left"] / float(b["cap"])
    low = frac <= 0.25 or b["remaining"] <= 12
    if not (first_time or low):
        return ""
    where = f"row {b['index']}" if b["kind"] == "row" else f"column {b['index']}"
    tail = ""
    if low:
        tail = f" At this rate about {b['remaining']} more actions empty it, and the level then resets (actions already spent still count)."
    return (
        f"Step budget (read from the frames by the harness): the bar along {where} has {b['left']} of {b['cap']} cells left "
        f"after {b['actions']} actions on this attempt (about {b['rate']:.1f} cells per action).{tail}"
    )


def _move_of(before, after):
    """(colour, cells, d_row, d_col) of the colour that shifted most between two grids, or None.

    Cells that left a colour and cells that took it are compared by their centres; a colour only counts when both sets are of
    similar size (an object moved, not painted over), which keeps counters and animations out.
    """
    cells = diff_cells(before, after)
    if not cells or len(cells) > 400:
        return None
    left, took = {}, {}
    for r, c in cells:
        left.setdefault(before[r][c], []).append((r, c))
        took.setdefault(after[r][c], []).append((r, c))
    total = sum(len(row) for row in before)
    census = Counter(v for row in before for v in row)
    best = None
    for col in set(left) & set(took):
        if census[col] > 0.25 * total:
            continue  # the background "moves" the opposite way to every object; it is not what the key moves
        v, e = left[col], took[col]
        big = max(len(v), len(e))
        if min(len(v), len(e)) < 4 or abs(len(v) - len(e)) > 0.25 * big:
            continue
        dr = round(sum(p[0] for p in e) / len(e) - sum(p[0] for p in v) / len(v))
        dc = round(sum(p[1] for p in e) / len(e) - sum(p[1] for p in v) / len(v))
        if dr == 0 and dc == 0:
            continue
        if best is None or big > best[1]:
            best = (col, big, dr, dc)
    return best


def _span(lo, hi):
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def describe_box(box):
    r0, c0, r1, c1 = box
    return f"rows {_span(r0, r1)}, cols {_span(c0, c1)}"


class Ledger:
    """Per-level facts, fed one transition at a time (action name, result dict, grid before, grid after)."""

    def __init__(self, level):
        self.level = level
        self.n_actions = 0           # real actions on this level (RESET not counted)
        self.keys = {}               # key -> {"n", "changed", "noop", "first_noop", "moves": Counter, "samples": [(before, after)]}
        self.events = []             # {"idx", "action", "box", "before", "after", "revert_of"}
        self.marks = {}              # box -> list of (idx, before_letters, after_letters)
        self.undo = 0
        self.resets = 0
        self.mouse_targets = []      # (row, col) of clicks
        self.noisy = False

    # ---- feeding -------------------------------------------------------------------------------------------------
    def feed(self, action, result, before, after):
        """Returns the new indicator events produced by this transition (usually an empty list)."""
        key = key_of(action)
        if key == "RESET":
            self.resets += 1
            self.marks = {}          # a restarted level legitimately clears its marks
            return []
        self.n_actions += 1
        idx = self.n_actions
        rec = self.keys.setdefault(key, {"n": 0, "changed": 0, "noop": 0, "first_noop": None, "samples": []})
        rec["n"] += 1
        cells = diff_cells(before, after)
        inner = inner_changed(cells, len(after), len(after[0]) if after else 64)
        if inner > 0:
            rec["changed"] += 1
            if len(rec["samples"]) < 40:
                rec["samples"].append((before, after))
            else:
                rec["samples"][rec["n"] % 40] = (before, after)
        else:
            rec["noop"] += 1
            if rec["first_noop"] is None:
                rec["first_noop"] = idx
        if key == "UNDO":
            self.undo += 1
        if key == "MOUSE":
            s = str(action)
            try:
                rr = int(s.split("row=")[1].split(",")[0])
                cc = int(s.split("col=")[1].split(")")[0])
                if len(self.mouse_targets) < 12:
                    self.mouse_targets.append((rr, cc))
            except (IndexError, ValueError):
                pass
        if self.noisy:
            return []
        patches = small_far_changes(before, after) if inner else []
        out = []
        for p in patches:
            ev = {"idx": idx, "action": key, "box": p["box"], "before": p["before"], "after": p["after"],
                  "revert_of": None, "kind": "first", "original": p["before"]}
            hist = self.marks.setdefault(p["box"], [])
            if hist:
                original = hist[0][1]            # the colour the patch had before anything changed it
                ev["original"] = original
                if p["after"] == original:
                    ev["kind"] = "lost"          # back to the original colour: a mark that was earned may be gone
                    ev["revert_of"] = next((i for (i, _b, a) in reversed(hist) if a != original), None)
                else:
                    earlier = next((i for (i, _b, a) in hist if a == p["after"]), None)
                    ev["kind"] = "again" if earlier is not None else "other"
                    ev["revert_of"] = earlier
            hist.append((idx, p["before"], p["after"]))
            self.events.append(ev)
            out.append(ev)
        if len(self.marks) > 6:
            # many different places flicker (a busy board, not a legend): say nothing rather than add noise.
            # One place that changes back and forth many times is a signal, so repeats at one place do not count.
            self.noisy = True
            self.events = []
            out = []
        return out

    # ---- text ----------------------------------------------------------------------------------------------------
    @staticmethod
    def _names(ev):
        b, a, o = ev["before"][:1], ev["after"][:1], ev.get("original", ev["before"])[:1]
        return COLOR_NAMES.get(b, b), COLOR_NAMES.get(a, a), COLOR_NAMES.get(o, o)

    @staticmethod
    def event_line(ev):
        box = describe_box(ev["box"])
        bn, an, on = Ledger._names(ev)
        kind = ev.get("kind", "first")
        if kind == "lost":
            since = f" (it had changed to {bn} after action {ev['revert_of']})" if ev["revert_of"] is not None else ""
            return (
                f"Indicator note: after action {ev['idx']} on this level ({ev['action']}), the small patch at {box} went back to "
                f"its original colour, {on}{since}. A mark you earned may have been lost: compare what you did between those "
                "two actions."
            )
        if kind == "again":
            return (
                f"Indicator note: after action {ev['idx']} on this level ({ev['action']}), the small patch at {box} changed to "
                f"{an} again ({bn} before; it was {an} after action {ev['revert_of']})."
            )
        return (
            f"Indicator note: after action {ev['idx']} on this level ({ev['action']}), a small patch away from the moved objects "
            f"changed: {box}, {bn} to {an}. Legends, trays and counters often report progress; note which action did this."
        )

    def key_lines(self, valid_actions=None):
        lines = []
        order = [k for k in ("UP", "DOWN", "LEFT", "RIGHT") if k in self.keys] + sorted(k for k in self.keys if k not in ("UP", "DOWN", "LEFT", "RIGHT"))
        for k in order:
            rec = self.keys[k]
            line = f"- {k}: pressed {rec['n']}, changed the board {rec['changed']}, did nothing {rec['noop']}"
            if rec["first_noop"] is not None and rec["noop"]:
                line += f" (first at action {rec['first_noop']})"
            sig = self.move_signature(k)
            if sig:
                line += f"; typical effect: {sig}"
            if k == "MOUSE" and self.mouse_targets:
                line += "; clicked at " + ", ".join(f"({r},{c})" for r, c in self.mouse_targets[:6])
            lines.append(line)
        few = [k for k in order if self.keys[k]["n"] < 2 and k not in ("RESET",)]
        if few:
            lines.append("- Pressed fewer than twice (one result says little about other board states): " + ", ".join(few) + ".")
        if valid_actions:
            unused = [a for a in valid_actions if key_of(a) not in self.keys and key_of(a) not in ("RESET",)]
            if unused:
                lines.append("- Valid but not pressed on this level: " + ", ".join(sorted(set(key_of(a) for a in unused))) + ".")
        return lines

    def move_signature(self, key, limit=8):
        """Most common shift of the cells of one colour over up to `limit` recent effective presses of one key.

        Pure grid arithmetic: the cells that left a colour and the cells that took it, compared by their centres.
        """
        rec = self.keys.get(key)
        if not rec or not rec["samples"]:
            return ""
        sig = Counter()
        sizes = {}
        used = 0
        for before, after in rec["samples"][-limit:]:
            used += 1
            m = _move_of(before, after)
            if m is None:
                continue
            color, n, dr, dc = m
            sig[(color, dr, dc)] += 1
            sizes[(color, dr, dc)] = max(sizes.get((color, dr, dc), 0), n)
        if not sig or not used:
            return ""
        (color, dr, dc), cnt = sig.most_common(1)[0]
        if cnt * 2 < used:
            return ""  # no consistent move: say nothing rather than a number that most presses contradict
        parts = []
        if dr:
            parts.append(f"{'down' if dr > 0 else 'up'} {abs(dr)}")
        if dc:
            parts.append(f"{'right' if dc > 0 else 'left'} {abs(dc)}")
        name = COLOR_NAMES.get(letter(color), str(color))
        return f"{name} cells shifted {' and '.join(parts)} ({cnt} of {used} presses checked)"

    def event_lines(self, last=5):
        out = []
        for ev in self.events[-last:]:
            box = describe_box(ev["box"])
            bn, an, on = Ledger._names(ev)
            kind = ev.get("kind", "first")
            if kind == "lost":
                out.append(f"- after action {ev['idx']} ({ev['action']}): the patch at {box} went back to its original colour, {on} (a mark may have been lost).")
            elif kind == "again":
                out.append(f"- after action {ev['idx']} ({ev['action']}): the patch at {box} changed to {an} again.")
            else:
                out.append(f"- after action {ev['idx']} ({ev['action']}): the patch at {box} changed from {bn} to {an}.")
        return out

    def digest_lines(self, clear_actions):
        """Short facts of a cleared level, for the LEVEL LOG of a later level."""
        bits = []
        for k in ("UP", "DOWN", "LEFT", "RIGHT"):
            if k in self.keys:
                sig = self.move_signature(k, limit=4)
                if sig:
                    bits.append(f"{k}: {sig}")
        extra = [f"{k} pressed {r['n']} times" for k, r in self.keys.items() if k not in ("UP", "DOWN", "LEFT", "RIGHT", "RESET")]
        text = f"- Level {self.level} was cleared with {self.n_actions} actions."
        if bits:
            text += " Effects seen there: " + "; ".join(bits) + "."
        if extra:
            text += " Other keys used: " + ", ".join(extra[:4]) + "."
        if clear_actions:
            text += " The last actions before clearing it: " + " ".join(clear_actions[-10:]) + "."
        return [text]


ACK_TEXT = "Noted: these are harness-computed facts about this level, and I will check my plan against them."

PROTOCOL_LINES = [
    "How strong runs work on a level like this (use it now):",
    "1. Keep two or three readings of the goal side by side. For each, name the cheapest probe (1-2 actions) that would refute it and what you expect to see.",
    "2. After every probe look at three things: the objects that moved, any legend/tray/counter marks that changed (and which action changed them), and the step bar.",
    "3. Write results as 'action -> what changed, in which board state'. One press with no effect in one state says nothing about the other states.",
    "4. If a mark you had earned disappears, undo or avoid the last action before building on that state.",
    "5. Send short batches (1-3 actions) while a rule is unconfirmed; longer batches only for sequences you have already verified.",
]


def build_level_log(ledger, *, level_tokens, valid_actions, budget_text, earlier, undo_note=True):
    lines = [
        f"LEVEL LOG - facts the harness computed from the recorded frames of this level (not from your notes): level {ledger.level}, "
        f"{ledger.n_actions} actions here, about {level_tokens} generated tokens so far.",
        "Keys on this level:",
        *ledger.key_lines(valid_actions),
    ]
    if undo_note and ledger.undo:
        lines.append(f"- UNDO was pressed {ledger.undo} times; every press counts as one action.")
    if ledger.events:
        lines.append("Small patches that changed away from the moved objects (legend/tray/counter candidates):")
        lines.extend(ledger.event_lines())
    elif not ledger.noisy:
        lines.append("No indicator patch (legend, tray, counter) has changed on this level so far; if the screen has one, no action of yours has moved it.")
    if budget_text:
        lines.append(budget_text)
    if earlier:
        lines.append("Earlier levels of this game:")
        lines.extend(earlier[-2:])
    lines.append("")
    lines.extend(PROTOCOL_LINES)
    return "\n".join(lines)