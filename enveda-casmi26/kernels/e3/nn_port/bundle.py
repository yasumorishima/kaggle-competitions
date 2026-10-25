import hashlib,json,zipfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def find_bundle(input_root,output_root,pins):
    roots=[]
    for path in Path(input_root).rglob('final_model.pt'):
        if (path.parent/'config.json').is_file():roots.append(path.parent)
    if not roots:
        archives=list(Path(input_root).rglob('casmi_spectrum_model_v1.zip'))
        assert len(archives)==1, 'Attach the supplied casmi_spectrum_model_v1.zip as a Kaggle dataset (zipped or extracted)'
        dest=Path(output_root)/'uploaded_model';dest.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archives[0]) as z:
            for item in z.infolist():
                target=(dest/item.filename).resolve()
                assert target.is_relative_to(dest.resolve()),'Unsafe ZIP member'
            z.extractall(dest)
        roots=[p.parent for p in dest.rglob('final_model.pt')]
    matching=[r for r in roots if all((r/n).is_file() and sha(r/n)==h for n,h in pins.items())]
    assert len(matching)==1, f'Expected exactly one matching model bundle, found {len(matching)}. Use the attached v1 artifacts.'
    return matching[0]
