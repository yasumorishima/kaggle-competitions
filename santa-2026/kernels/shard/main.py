# Santa 2026 search shard (yasunorim). One of N kernels; each searches its share of the 1,000 puzzles.
# Inputs: the competition data and our private dataset santa26-work (our code in search/ and tools/, the public
# C++ engines in third_party/ with their NOTICE.md, our pool.csv and archive.jsonl).
# Engines: Wang Jakie's graph search and Atakan Aldemir's optimizers (Apache-2.0, unchanged; see NOTICE.md).
# Ours: search/runner.py (fresh-first scheduling, near-length archive as crossover tapes, sharding) and tools/.
# Output: improved.csv (puzzles that got shorter) and archive_new.csv (near-length distinct programs).
import glob, os, shutil, subprocess, sys, time

SHARD = "__SHARD__"          # "i/n", filled in by the workflow
MINUTES = float("__MINUTES__")
PLAN = "__PLAN__"
T0 = time.time()

def find(pattern):
    hits = sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True))
    assert hits, pattern
    return hits[0]

runner = find("search/runner.py"); root = os.path.dirname(os.path.dirname(runner))
puzzles = [p for p in glob.glob("/kaggle/input/**/puzzles.csv", recursive=True) if "practice" not in p][0]
pool, archive = os.path.join(root, "pool.csv"), os.path.join(root, "archive.jsonl")
print("root", root, "puzzles", puzzles, "cpus", os.cpu_count(), flush=True)

work = "/kaggle/temp_bin" if os.access("/kaggle", os.W_OK) else "/tmp/bin"
os.makedirs(work, exist_ok=True)
src = os.path.join(root, "third_party")
builds = {s: f"wangjakie/{s}.cpp" for s in ("anneal", "anneal_seeded", "incremental", "incremental_seeded", "quartet",
                                             "compact_rotate", "compact_mask", "baseline_mask")}
builds.update(frontier="atakan/elf_multiplier_frontier.cpp", closure="atakan/elf_interface_closure.cpp",
              triwalk="goodpjw2008/triwalk.cpp")
procs = [subprocess.Popen(["g++", "-O3", "-march=native", "-std=c++17", os.path.join(src, f), "-o", os.path.join(work, s)])
         for s, f in builds.items()]
for p in procs: p.wait()
print("built", sorted(os.listdir(work)), f"{time.time() - T0:.0f}s", flush=True)

minutes = MINUTES - (time.time() - T0) / 60
subprocess.run([sys.executable, runner, "--pool", pool, "--puzzles", puzzles, "--bin", work,
                "--out", "/kaggle/working/improved.csv", "--archive", archive,
                "--archive-out", "/kaggle/working/archive_new.csv", "--shard", SHARD,
                "--minutes", str(minutes), "--jobs", str(os.cpu_count()), "--plan", PLAN]
               + (["--fresh-ids", os.path.join(root, "fresh.txt")] if os.path.exists(os.path.join(root, "fresh.txt")) else []),
               check=True)
shutil.rmtree(work, ignore_errors=True)
print("total", f"{(time.time() - T0) / 60:.1f} min", flush=True)
