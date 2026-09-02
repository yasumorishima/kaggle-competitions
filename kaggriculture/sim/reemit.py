#!/usr/bin/env python
"""Rebuild an emitted agent on top of the current main.py.

An emitted agent is a frozen copy of main.py with a calendar and a few knob
overrides appended, which is what makes it a fair sparring partner -- and what
makes it useless the moment a knob is added: the sweep refuses it with "has no
such knob", correctly, because the frozen copy has never heard of it.

The calendar lives on a branch and is not in the checkout, so the honest source
for it is the frozen agent itself. This lifts the calendar and the overrides
back out and re-emits them, so the new agent is today's policy under exactly
the capital the old one ran.

Usage:
    python sim/reemit.py agents/v38_sched.py --out agents/v39_sched.py
    python sim/reemit.py agents/v38_sched.py --out agents/v39.py --set fill_idle=true
"""
import argparse
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import emit_sched  # noqa: E402


def extract(path):
    """Return (calendar, overrides) from an emitted agent.

    The trailing `SCHEDULE = {...}` and `P.update({...})` are read with ast,
    not exec: this file is run over agents that came off a search, and nothing
    here needs to execute one to read two literals off the end of it.
    """
    with open(path, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    sched, over = None, {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "SCHEDULE"):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(value, dict):
                sched = value          # the last binding wins, as at run time
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "update"
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "P"
                and node.value.args):
            over.update(ast.literal_eval(node.value.args[0]))
    return sched, over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", help="the frozen agent to lift the calendar out of")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="",
                    help="extra knob overrides, k=v,k=v (JSON values)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    sched, over = extract(os.path.join(ROOT, args.agent)
                          if not os.path.isabs(args.agent) else args.agent)
    if sched is None:
        print(f"{args.agent} carries no calendar", file=sys.stderr)
        return 1
    for pair in filter(None, (s.strip() for s in args.set.split(","))):
        k, _, v = pair.partition("=")
        over[k.strip()] = json.loads(v)

    note = args.note or f"Re-emitted from {os.path.basename(args.agent)}"
    source = emit_sched.emit(sched, note=note)
    if over:
        source += "P.update(" + emit_sched.py_literal(over) + ")\n"
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(source)
    print(f"WROTE {args.out}  days={len(sched)}  overrides={json.dumps(over, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
