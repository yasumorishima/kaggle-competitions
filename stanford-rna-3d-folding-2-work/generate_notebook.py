"""Generate Stanford RNA 3D Folding 2 — v5 Smart Template Matching notebook.

Converts the .py percent-format script to .ipynb using jupytext.

5 structure slots:
1. Best template by k-mer + sequence identity (linear interpolation)
2. 2nd best template (linear interpolation)
3. 3rd best template (linear interpolation)
4. Weighted average of top-5 templates (Kabsch-aligned)
5. Best template with cubic spline interpolation

Strategy:
- Two-stage template search: k-mer pre-filter (top-50) -> combined scoring (top-10)
- Combined similarity: 0.6*kmer_cosine + 0.3*seq_identity + 0.1*length_similarity
- Validation on validation set before test predictions
- W&B offline logging for experiment tracking
"""

import jupytext

PY_FILE = "s6e3-rna-multi-approach-work.py"
IPYNB_FILE = "s6e3-rna-multi-approach-work.ipynb"

nb = jupytext.read(PY_FILE)
jupytext.write(nb, IPYNB_FILE)

print(f"Generated {IPYNB_FILE} from {PY_FILE}")
print(f"Cells: {len(nb.cells)}")
for i, cell in enumerate(nb.cells):
    src_preview = cell.source[:60].replace("\n", " ").encode("ascii", "replace").decode()
    print(f"  [{i}] {cell.cell_type:10s} | {src_preview}...")
