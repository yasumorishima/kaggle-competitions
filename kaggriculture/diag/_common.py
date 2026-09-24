"""Shared paths for the diagnostics: everything is relative to this checkout.

The default opponent is the public router, which is fetched, never committed:
    python sim/fetch_opponent.py --exec thomastschinkel/kaggriculture-93-8-win-rate-public-state-router
(needs KAGGLE_API_TOKEN). Point OPP at any other agent file to use that instead.
"""
import os
import sys

K = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(K, "sim"))
OUT = os.path.join(K, "diag", "_variants")
ROUTER = os.path.join(
    K, "opponents",
    "thomastschinkel__kaggriculture-93-8-win-rate-public-state-router.py")


def opponent():
    opp = os.environ.get("OPP", ROUTER)
    if not os.path.isabs(opp):
        opp = os.path.join(K, opp)
    if not os.path.exists(opp):
        sys.exit(f"opponent not found: {opp}\n" + __doc__)
    return opp


def agent_path(rel):
    return rel if os.path.isabs(rel) else os.path.join(K, rel)
