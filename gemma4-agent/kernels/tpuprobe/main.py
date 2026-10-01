"""TPU probe: what the Kaggle TPU machine offers for LoRA training of Gemma 4 31B."""
import glob
import importlib
import os
import shutil
import subprocess

print("== env")
for k in ("TPU_NAME", "TPU_ACCELERATOR_TYPE", "XRT_TPU_CONFIG", "PJRT_DEVICE", "KAGGLE_KERNEL_RUN_TYPE"):
    print(k, os.environ.get(k))
print(subprocess.run("nproc; free -g; df -h /kaggle/working /tmp | tail -2", shell=True, capture_output=True, text=True).stdout)

print("== packages")
for m in ("jax", "jaxlib", "flax", "optax", "keras", "keras_hub", "torch", "torch_xla", "transformers", "peft",
          "safetensors", "accelerate", "orbax", "qwix", "tunix"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)

print("== jax devices")
try:
    import jax
    devs = jax.devices()
    print(len(devs), devs[:2])
    for d in devs[:1]:
        try:
            print("memory", d.memory_stats())
        except Exception as e:
            print("memory_stats", e)
except Exception as e:
    print("jax", e)

print("== model files")
for p in glob.glob("/kaggle/input/**/config.json", recursive=True)[:5]:
    d = os.path.dirname(p)
    fs = sorted(os.listdir(d))
    print(d, len(fs), fs[:12])
    print(open(p).read()[:1500])
print("total input GB", round(sum(os.path.getsize(f) for f in glob.glob("/kaggle/input/**/*", recursive=True) if os.path.isfile(f)) / 1e9, 1))
print(shutil.disk_usage("/kaggle/working"))
