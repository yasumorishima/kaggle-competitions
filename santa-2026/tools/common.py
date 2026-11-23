"""Shared helpers for Santa 2026 (The Elf Calculator): load puzzles/submissions and replay programs exactly."""
import csv, json
csv.field_size_limit(1 << 30)
NREG = 48

def load_puzzles(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        out[r["id"]] = (json.loads(r["starting_values"]), json.loads(r["targets"]), int(r["value_limit"]))
    return out

def load_sub(path):
    return {r["id"]: json.loads(r["program"]) for r in csv.DictReader(open(path, encoding="utf-8-sig"))}

def write_sub(path, sub):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "program"])
        for pid in sorted(sub):
            w.writerow([pid, json.dumps(sub[pid], separators=(",", ":"))])

def replay(start, targets, limit, prog):
    """Raise AssertionError if prog is invalid under the scorer's rules; return the length."""
    assert isinstance(prog, list) and len(prog) <= 4096
    regs = [None] * NREG; regs[:len(start)] = start
    need = set(targets) - set(start)
    for k, ins in enumerate(prog):
        op, i, j, d = ins
        assert all(type(x) is int and 0 <= x < NREG for x in (i, j, d)), f"step {k}: register"
        a, b = regs[i], regs[j]
        assert a is not None and b is not None, f"step {k}: undefined register"
        if op == "+": v = a + b
        elif op == "-": v = a - b
        elif op == "*": v = a * b
        elif op == "/":
            assert a % b == 0, f"step {k}: inexact division"; v = a // b
        else: raise AssertionError(f"step {k}: op {op}")
        assert 1 <= v <= limit, f"step {k}: value out of range"
        regs[d] = v; need.discard(v)
    assert not need, f"{len(need)} targets missing"
    return len(prog)
