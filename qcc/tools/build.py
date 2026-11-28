"""Assemble a QCC submission from public kernel outputs (by sha256) plus our edits.

usage: python build.py recipe.json <dir with downloaded kernel outputs> <out dir>
needs kickmix on sys.path when the recipe has edits (KICKMIX env var: the folder holding kickmix.so).

recipe keys:
  members  {path in the submission: sha256}  every file must be found by its sha256
  edits    {path: {"base": sha256, "replace": [[old, new], ...], "sha256": result sha256}}
           text edits on the kmx form of the base circuit; each old must occur exactly once
"""
import hashlib, io, json, os, shutil, sys, zipfile

def sha(b): return hashlib.sha256(b).hexdigest()

def index(root):
    """sha256 -> bytes source for every file under root, and inside every zip under root"""
    idx = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            if f.endswith('.zip'):
                with zipfile.ZipFile(p) as z:
                    for n in z.namelist():
                        if n.endswith('/'): continue
                        h = hashlib.sha256()
                        with z.open(n) as fh:
                            for ch in iter(lambda: fh.read(1 << 24), b''): h.update(ch)
                        idx.setdefault(h.hexdigest(), ('zip', p, n))
            else:
                h = hashlib.sha256()
                with open(p, 'rb') as fh:
                    for ch in iter(lambda: fh.read(1 << 24), b''): h.update(ch)
                idx.setdefault(h.hexdigest(), ('file', p, None))
    return idx

def read(src):
    kind, p, n = src
    if kind == 'file': return open(p, 'rb').read()
    with zipfile.ZipFile(p) as z: return z.read(n)

def main():
    rec = json.load(open(sys.argv[1])); root, out = sys.argv[2], sys.argv[3]
    idx = index(root)
    os.makedirs(out, exist_ok=True)
    edits = rec.get('edits', {})
    need = set(rec['members'].values()) | {e['base'] for e in edits.values()}
    miss = [h for h in need if h not in idx and h not in {e['sha256'] for e in edits.values()}]
    assert not miss, f'not found by sha256: {miss}'
    for path, h in rec['members'].items():
        dst = os.path.join(out, path); os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
        if path in edits:
            e = edits[path]; assert e['sha256'] == h
            if h in idx:
                data = read(idx[h])
            else:
                sys.path.insert(0, os.environ['KICKMIX']); import kickmix as km
                tmp = os.path.join(out, '_base.kmb'); open(tmp, 'wb').write(read(idx[e['base']]))
                c = km.Circuit.from_file(tmp, 'kmb'); os.remove(tmp)
                buf = io.BytesIO(); c.to_file(buf, format='kmx'); del c
                txt = buf.getvalue(); del buf
                for old, new in e['replace']:
                    o, n = ('\n' + old + '\n').encode(), ('\n' + new + '\n').encode()
                    assert txt.count(o) == 1, f'edit must match once: {old!r} ({txt.count(o)})'
                    txt = txt.replace(o, n, 1)
                tmp = os.path.join(out, '_edit.kmx'); open(tmp, 'wb').write(txt); del txt
                c = km.Circuit.from_file(tmp, 'kmx'); os.remove(tmp)
                buf = io.BytesIO(); c.to_file(buf, format='kmb'); data = buf.getvalue(); del c, buf
        else:
            data = read(idx[h])
        assert sha(data) == h, f'{path}: sha256 {sha(data)} != {h}'
        open(dst, 'wb').write(data); print(f'{path:28s} {h[:8]}  {len(data):>11,} bytes')
    zp = os.path.join(out, 'submission.zip')
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(rec['members']): z.write(os.path.join(out, path), path)
    print('wrote', zp, os.path.getsize(zp))

if __name__ == '__main__':
    main()
