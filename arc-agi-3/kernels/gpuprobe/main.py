"""ARC-AGI-3 GPU probe: what the RTX Pro 6000 image offers for an LLM agent.

GPU, torch/CUDA, whether vllm is preinstalled or pip-installable, the Qwen3 model mount,
and the speed of a short vLLM generation (prompt like a 64x64 grid as text).
"""
import glob
import importlib
import os
import subprocess
import sys
import time


def sh(c):
    print("$", c, flush=True)
    print(subprocess.run(c, shell=True, capture_output=True, text=True).stdout[-3000:], flush=True)


sh("nvidia-smi")
sh("python -V; df -h /kaggle/working | tail -1; free -g | head -2; nproc")
for m in ["torch", "transformers", "vllm", "sglang", "flash_attn", "accelerate"]:
    try:
        x = importlib.import_module(m)
        print(m, getattr(x, "__version__", "?"), flush=True)
    except Exception as e:
        print(m, "missing", type(e).__name__, flush=True)
import torch  # noqa: E402
print("cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

cfg = glob.glob("/kaggle/input/**/config.json", recursive=True)
print("model configs", cfg[:5])
MODEL = os.path.dirname(cfg[0]) if cfg else None
sh(f"ls -la {MODEL} | head -30; du -sh {MODEL}") if MODEL else None
sh("ls /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ | head")

try:
    import vllm  # noqa: F401
except Exception:
    t = time.time()
    sh(f"{sys.executable} -m pip install -q vllm 2>&1 | tail -5")
    print("pip install vllm took", round(time.time() - t), "s", flush=True)
    sh(f"{sys.executable} -m pip show vllm torch | grep -E 'Name|Version'")

try:
    from vllm import LLM, SamplingParams
    t = time.time()
    llm = LLM(model=MODEL, max_model_len=32768, gpu_memory_utilization=0.85)
    print("load", round(time.time() - t), "s", flush=True)
    grid = "\n".join("".join("0123456789abcdef"[(x * y + x) % 16] for x in range(64)) for y in range(64))
    msgs = [[{"role": "user", "content": f"Here is a 64x64 grid (hex digits are colours):\n{grid}\n"
              f"Describe the objects you see and propose one action among ACTION1-ACTION5. Variant {i}."}] for i in range(8)]
    sp = SamplingParams(max_tokens=512, temperature=0.7)
    t = time.time()
    outs = llm.chat(msgs, sp)
    dt = time.time() - t
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"8 chats: {dt:.1f}s, {ntok} tokens out, prompt tokens {len(outs[0].prompt_token_ids)}", flush=True)
    print(outs[0].outputs[0].text[:800])
except Exception as e:
    import traceback
    traceback.print_exc()
    print("vllm run failed", e)
