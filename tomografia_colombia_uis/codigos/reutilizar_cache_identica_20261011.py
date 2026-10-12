"""Reuse only existing fields with exactly matching full model identity."""
from robustez_20261011 import *
import shutil

if __name__=='__main__':
    original=Path(CFG['cache_directory']);target=OUT/'inverse_synthetic_fields';records=[]
    for folder in target.iterdir():
        if not folder.is_dir():continue
        src=original/folder.name
        if not (src/'model_identity.json').is_file():continue
        a=json.loads((src/'model_identity.json').read_text());b=json.loads((folder/'model_identity.json').read_text())
        if a!=b:continue
        count=0
        for f in src.glob('*.npy'):
            dest=folder/f.name
            if dest.exists():continue
            shutil.copy2(f.with_suffix('.json'),dest.with_suffix('.json'))
            tmp=dest.with_suffix('.clone_tmp');shutil.copy2(f,tmp)
            try:
                if not dest.exists():tmp.replace(dest);count+=1
                else:tmp.unlink()
            except PermissionError:
                if dest.exists():tmp.unlink(missing_ok=True)
                else:raise
        records.append(dict(model_key=a['model_key'],fields_copied=count))
    save(OUT/'identical_cache_reuse.json',dict(records=records,script_sha256=digest(__file__),changes_to_algorithm=False));print(json.dumps(records),flush=True)
