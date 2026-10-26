"""Small consumer helpers for aligned public CASMI26 resources."""
import pathlib
import numpy as np

def load_lotus_refs(pop_dir,tier_rows=105875489):
    """Return raw uint16 counts, aligned to pc_mass / pc_ik14 source rows."""
    a=np.load(pathlib.Path(pop_dir)/'pc_lotus_refs.npy',mmap_mode='r')
    if a.shape!=(tier_rows,) or a.dtype!=np.dtype('uint16'):
        raise ValueError(f'LOTUS count alignment mismatch: shape={a.shape}, dtype={a.dtype}')
    return a

def lotus_log_prior(counts):
    """Preserve the original notebook's float64-log1p -> float32 precision."""
    return np.log1p(np.asarray(counts,np.float64)).astype(np.float32)

def validate_pubchemlite_annotations(frame,meta):
    """Check positional compatibility with the source pubchemlite_meta.pkl."""
    if len(frame)!=len(meta['keys']):raise ValueError('PubChemLite row count mismatch')
    if not np.array_equal(frame.pubchemlite_row.to_numpy(),np.arange(len(frame))):raise ValueError('PubChemLite row order mismatch')
    if not np.array_equal(frame.candidate_key.astype(str).to_numpy(),np.asarray(meta['keys']).astype(str)):raise ValueError('PubChemLite candidate-key mismatch')
    return frame
