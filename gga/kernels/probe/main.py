# Probe of the TPU image for the GGA inference server: what it has for gRPC stubs and an LLM.
import importlib, subprocess, sys
print("python", sys.version)
for m in ("grpc", "grpc_tools", "google.protobuf", "polars", "PIL", "jax", "keras", "keras_hub", "torch", "torch_xla", "transformers", "vllm"):
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
print(subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True).stdout[:6000])
import os; print("cpus", os.cpu_count())
open("/kaggle/working/submission.parquet", "wb").close()
