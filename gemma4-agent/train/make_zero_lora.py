"""Write a rank-r PEFT LoRA adapter for gemma-4-31b-it whose B matrices are zero (the model's output is unchanged).

Used for the LoRA feasibility probe (10-24 plan audit): does a submission with adapters/ start vLLM (LoRA enabled,
max_lora_rank = r) and finish the hidden run within 12 h with the smaller KV cache? Only q_proj of the 60 text layers
is targeted (sliding layers: 5376 -> 32 x 256, full layers: 5376 -> 32 x 512, from the model's config.json).
No numpy/torch needed: safetensors is written by hand (bf16).

    python make_zero_lora.py OUT_DIR [rank]
"""
import json
import os
import random
import struct
import sys

out = sys.argv[1]
r = int(sys.argv[2]) if len(sys.argv) > 2 else 8
H, NH, HD, GHD, L = 5376, 32, 256, 512, 60
full = {i for i in range(L) if i % 6 == 5}      # layer_types: every 6th layer is full_attention (5, 11, ..., 59)


def bf16(x):
    return struct.unpack(">I", struct.pack(">f", x))[0] >> 16


tensors = {}
rng = random.Random(0)
for i in range(L):
    out_f = NH * (GHD if i in full else HD)
    pre = f"base_model.model.model.language_model.layers.{i}.self_attn.q_proj"
    a = bytearray()
    bound = 1.0 / (H ** 0.5)
    for _ in range(r * H):
        a += struct.pack("<H", bf16(rng.uniform(-bound, bound)))
    tensors[pre + ".lora_A.weight"] = ([r, H], bytes(a))
    tensors[pre + ".lora_B.weight"] = ([out_f, r], bytes(2 * out_f * r))

header, off, blobs = {}, 0, []
for k in sorted(tensors):
    shape, b = tensors[k]
    header[k] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + len(b)]}
    off += len(b)
    blobs.append(b)
header["__metadata__"] = {"format": "pt"}
hj = json.dumps(header, separators=(",", ":")).encode()
hj += b" " * ((8 - len(hj) % 8) % 8)
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "adapter_model.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(hj)))
    f.write(hj)
    for b in blobs:
        f.write(b)
cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": "google/gemma-4-31b-it",
       "r": r, "lora_alpha": 2 * r, "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
       "target_modules": ["q_proj"], "layers_to_transform": None, "modules_to_save": None,
       "inference_mode": True, "use_rslora": False, "use_dora": False, "init_lora_weights": True}
json.dump(cfg, open(os.path.join(out, "adapter_config.json"), "w"), indent=2)
print("wrote", out, "rank", r, "tensors", len(tensors), "bytes", off)
