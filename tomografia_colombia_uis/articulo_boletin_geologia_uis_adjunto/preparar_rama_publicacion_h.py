"""Stage only the verified reproduction package in the dedicated Git branch.

This command neither commits nor pushes. It rejects an incomplete campaign,
an unreviewed Word document, or a changed delivery archive.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / 'resultados/campana_perfil_suavidad_20261011h'
CHECKOUT = PROJECT.parent / 'publicacion_tomografia_colombia_uis'
BRANCH = 'tomografia-colombia-uis-h'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=CHECKOUT, text=True).strip()

assert git('branch', '--show-current') == BRANCH
assert git('remote', 'get-url', 'origin') == 'https://github.com/romusan/relocation-pwave-tomography.git'
assert json.loads((RESULTS / 'completed.json').read_text())['status'] == 'completed'
assert json.loads((RESULTS / 'document_delivery_qa.json').read_text())['visual_review_all_pages'] is True
qa = json.loads((RESULTS / 'archive_delivery_qa.json').read_text())
archive = Path(qa['zip'])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == qa['sha256']
target = CHECKOUT / 'tomografia_colombia_uis'
manifest = []
with zipfile.ZipFile(archive) as package:
    assert package.testzip() is None
    for entry in package.infolist():
        if entry.is_dir():
            continue
        rel = Path(entry.filename)
        assert not rel.is_absolute() and '..' not in rel.parts
        data = package.read(entry)
        assert len(data) < 100_000_000, f'Exceeds GitHub file limit: {rel}'
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        manifest.append(dict(path=destination.relative_to(CHECKOUT).as_posix(),
                             bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
readme = target / 'README.md'
readme.write_text((PROJECT / 'articulo_boletin_geologia_uis_adjunto/README_reproduccion_h.md').read_text(encoding='utf-8'), encoding='utf-8')
report = dict(branch=BRANCH, repository='romusan/relocation-pwave-tomography',
              base_commit=git('rev-parse', 'HEAD'), archive_sha256=qa['sha256'],
              files=manifest, committed=False, published=False)
(RESULTS / 'publication_staging.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(dict(checkout=str(CHECKOUT), branch=BRANCH, files=len(manifest),
                      committed=False, published=False), indent=2))
