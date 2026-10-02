

# ---------------------------------------------------------------------------------------------
# LLM agent v0 (appended to the explorer by kernels/llm/build.py).
#
# A vLLM server (own wheel dataset, separate process) serves Qwen3-30B-A3B-Instruct-2507-FP8.
# Each move the LLM sees the frame as hex rows, what the recent moves changed, and which move
# kinds have done nothing so far, and names the next action. The explorer's state graph keeps
# running underneath: an unparsable answer, or a move already known to change nothing from
# this state, falls back to the explorer's choice. After LLM_STEPS moves in a game the
# explorer plays on alone.
# ---------------------------------------------------------------------------------------------
import glob as _glob  # noqa: E402
import json as _json  # noqa: E402
import re as _re  # noqa: E402
import threading as _threading  # noqa: E402
import urllib.request as _url  # noqa: E402

LLM_STEPS = int(os.getenv("ARC3_LLM_STEPS", "250"))
LLM_PORT = 8011
LLM_URL = f"http://127.0.0.1:{LLM_PORT}/v1/chat/completions"
HEX = "0123456789abcdef"
SYSTEM = """You are playing an unknown turn-based puzzle game on a grid (at most 64x64 cells, colours 0-f).
Each game has several levels; finishing a level shows a new grid. You must discover the rules yourself:
which actions move what, what the goal looks like, what ends the game. Use the effects of earlier moves.
Actions: ACTION1..ACTION5 are simple actions (often up, down, left, right, interact), ACTION6 clicks a cell
(give x = column, y = row, both 0-based), ACTION7 is undo when available. Fewer moves score higher, so
do not repeat moves that changed nothing and head for the goal once you have a theory.
Answer with at most three short sentences of reasoning, then a final line exactly like
ACTION: ACTION3
or
ACTION: ACTION6 12 40"""


def start_llm_server():
    wheels = os.path.dirname((_glob.glob("/kaggle/input/**/vllm-0.30.0-*.whl", recursive=True) or [""])[0])
    model = os.path.dirname((_glob.glob("/kaggle/input/models/qwen-lm/**/config.json", recursive=True) or [""])[0])
    print("wheels", wheels, "model", model, flush=True)
    t = time.time()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", wheels, "vllm"])
    print(f"vllm installed in {time.time() - t:.0f}s", flush=True)
    subprocess.call([sys.executable, "-c", "import torch;print('torch', torch.__version__, torch.version.cuda, "
                     "torch.cuda.is_available(), torch.cuda.get_device_capability())"])
    for extra in ([], ["--enforce-eager"]):
        log = open(WORK + "/vllm.log", "w")
        proc = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
                                 "--served-model-name", "q", "--port", str(LLM_PORT), "--max-model-len", "16384",
                                 "--gpu-memory-utilization", "0.88", "--max-num-seqs", "32"] + extra,
                                stdout=log, stderr=subprocess.STDOUT)
        for _ in range(180):
            time.sleep(10)
            try:
                _url.urlopen(f"http://127.0.0.1:{LLM_PORT}/v1/models", timeout=5)
                print(f"vllm up after {time.time() - t:.0f}s {extra}", flush=True)
                return proc
            except Exception:
                if proc.poll() is not None:
                    break
        proc.kill()
        lines = open(WORK + "/vllm.log").read().splitlines()
        print(f"vllm failed {extra}: {len(lines)} log lines; error lines:", flush=True)
        keep = [i for i, x in enumerate(lines) if any(k in x for k in ("Error", "error", "Exception", "CUDA", "sm_", "not supported"))]
        for i in keep[:60]:
            print("   ", lines[i][:400])
        print("first lines:\n" + "\n".join(x[:300] for x in lines[:40]), flush=True)
    raise RuntimeError("vllm server did not start")


def ask(messages, max_tokens=200):
    body = _json.dumps({"model": "q", "messages": messages, "max_tokens": max_tokens, "temperature": 0.3}).encode()
    req = _url.Request(LLM_URL, body, {"Content-Type": "application/json"})
    with _url.urlopen(req, timeout=120) as r:
        return _json.load(r)["choices"][0]["message"]["content"] or ""


def grid_text(g):
    return "\n".join("".join(HEX[int(v) & 15] for v in row) for row in g)


def change_text(a, b):
    if a is None or b is None:
        return "no frame"
    if a.shape != b.shape:
        return f"the whole screen changed (new size {b.shape[1]}x{b.shape[0]})"
    d = np.argwhere(a != b)
    if not len(d):
        return "nothing changed"
    (y0, x0), (y1, x1) = d.min(0), d.max(0)
    cols = sorted(set(int(v) for v in b[a != b]))[:6]
    return f"{len(d)} cells changed in rows {y0}-{y1}, columns {x0}-{x1} (new colours {','.join(HEX[c] for c in cols)})"


def parse_action(text, acts):
    m = _re.findall(r"ACTION:\s*(ACTION[1-7])(?:\s+(\d+)\s+(\d+))?", text or "")
    if not m:
        return None
    name, x, y = m[-1]
    for a in acts:
        if a.name == name:
            if a.is_complex():
                if not x:
                    return None
                return (a, (min(int(x), 63), min(int(y), 63)))
            return (a, None)
    return None


def play_llm(env, game_id, deadline):
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
    plan, hist = [], []
    n = n_llm = n_fallback = 0
    prev_g = None
    while n < MAX_ACTIONS and time.time() < deadline:
        if resp is None or resp.state == GameState.WIN:
            break
        if resp.state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
            resp = env.step(GameAction.RESET, {})
            n += 1
            plan = []
            hist.append("RESET after GAME OVER")
            continue
        g = grid_of(resp)
        if g is None:
            resp = env.step(GameAction.RESET, {})
            n += 1
            continue
        cur = ex.node(g)
        mv = None
        if n_llm < LLM_STEPS and not plan:
            known_noop = [f"{m[0].name}" + (f" {m[1][0]} {m[1][1]}" if m[1] else "")
                          for m, v in ex.edges[cur].items() if v == cur][:12]
            user = (f"Level {level + 1}. Available actions: {', '.join(a.name for a in acts)}.\n"
                    f"Recent moves and their effects (oldest first):\n" + ("\n".join(hist[-15:]) or "none yet") +
                    (f"\nMoves that changed nothing from this exact state: {', '.join(known_noop)}" if known_noop else "") +
                    f"\nCurrent grid ({g.shape[1]} columns x {g.shape[0]} rows, row 0 at the top):\n{grid_text(g)}")
            try:
                mv = parse_action(ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]), acts)
            except Exception as e:  # a slow or failed call falls back to the explorer
                print(game_id, "llm error", type(e).__name__, flush=True)
            n_llm += 1
            if mv is not None and ex.edges[cur].get(mv) == cur:
                mv = None
            if mv is None:
                n_fallback += 1
        if mv is None:
            if plan:
                mv = plan.pop(0)
            elif ex.todo[cur]:
                mv = ex.pick(g, ex.todo[cur]) if NOOP_MIN else ex.todo[cur].pop(0)
            else:
                path = ex.path_to_todo(cur)
                if not path:
                    resp = env.step(GameAction.RESET, {})
                    n += 1
                    hist.append("RESET (nothing left to try)")
                    continue
                mv, plan = path[0], path[1:]
        if mv in ex.todo.get(cur, []):
            ex.todo[cur].remove(mv)
        a, xy = mv
        nxt = env.step(a, {"x": xy[0], "y": xy[1]} if xy is not None else {})
        n += 1
        if nxt is None:
            break
        g2 = grid_of(nxt)
        name = a.name + (f" {xy[0]} {xy[1]}" if xy is not None else "")
        if nxt.levels_completed > level:
            hist.append(f"{name}: LEVEL {level + 1} COMPLETED")
            level = nxt.levels_completed
            ex.reset_level()
            plan = []
        else:
            hist.append(f"{name}: " + ("GAME OVER" if nxt.state == GameState.GAME_OVER else change_text(g, g2)))
            if g2 is not None and nxt.state == GameState.NOT_FINISHED:
                ex.record(g, mv, g2.shape != g.shape or bool((g2 != g).any()))
                ex.observe(g, g2)
                nk = ex.node(g2)
                if ex.edges[cur].get(mv, nk) != nk:
                    plan = []
                ex.edges[cur][mv] = nk
        resp = nxt
    return (f"{game_id}: actions={n} llm={n_llm} fallback={n_fallback} state={resp.state.name if resp else '?'} "
            f"levels={resp.levels_completed if resp else 0}")


def main_llm():
    import arc_agi
    import dotenv

    server = start_llm_server()
    os.chdir(WORK)
    write_env()
    dotenv.load_dotenv(dotenv_path=".env", override=True)
    arcade = arc_agi.Arcade()
    games = [e.game_id for e in arcade.available_environments]
    print(f"{len(games)} games, rerun={RERUN}", flush=True)
    t0 = time.time()
    deadline = t0 + TOTAL_BUDGET_S
    lock = _threading.Lock()
    queue = list(games)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                gid = queue.pop(0)
            try:
                env = arcade.make(gid)
                print(play_llm(env, gid, deadline), f"({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"{gid}: error {type(e).__name__}: {e}", flush=True)

    threads = [_threading.Thread(target=worker) for _ in range(int(os.getenv("ARC3_PAR", "25")))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not RERUN:
        sc = arcade.get_scorecard()
        print(f"Score: {sc.score:.4f}  levels {sc.total_levels_completed}/{sc.total_levels}  actions {sc.total_actions}")
        for e in sc.environments:
            print(f"  {e.id:<20} {e.score:8.2f} {e.levels_completed:4} {e.actions:6}")
    import pandas as pd
    pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]) \
        .to_parquet(WORK + "/submission.parquet", index=False)
    server.terminate()


if __name__ == "__main__":
    main_llm()
