"""Portable delivery package; exclude all field caches and unused optimizers."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'articulo_boletin_geologia_uis_adjunto';C=ROOT/'resultados/campana_perfil_suavidad_20261011h';G=ROOT/'resultados/campana_factorizada_cotas_20261011g'
assert json.loads((C/'completed.json').read_text())['status']=='completed'
assert json.loads((C/'document_delivery_qa.json').read_text())['visual_review_all_pages'] is True
assert json.loads((C/'document_delivery_qa.json').read_text())['docx_sha256']==hashlib.sha256((A/'Manuscrito_tomografia_Vp_Colombia_UIS_revision_20261011h.docx').read_bytes()).hexdigest()
files=set()
for name in ['Manuscrito_tomografia_Vp_Colombia_UIS_revision_20261011h.docx','crear_manuscrito_revision_20261011h.py','crear_manuscrito_revision_20261011f.py','figuras_revision_h.py','suplementos_revision_h.py','empaquetar_revision_h.py','verificar_manuscrito_h.py','README_reproduccion_h.md','preparar_rama_publicacion_h.py','cerrar_revision_h.ps1']:
    files.add(A/name)
files.update(p for p in (A/'suplementos').glob('*') if p.is_file());files.update((A/'figuras_revision_h').glob('*.png'));files.update((A/'fuentes').glob('*.json'));files.add(A/'fuentes/plantilla_uis.docx')
files.update((ROOT/'codigos').glob('*.py'));files.update((ROOT/'codigos/regional').glob('*.py'))
files.discard(ROOT/'codigos/acelerar_perfil_h.py')
for suffix in ['*.cpp','*.dll','*.json','*.py']:
    files.update(p for p in (ROOT/'codigos/native').glob(suffix) if 'depth_tangent' not in p.name)
for path in C.rglob('*'):
    if path.is_file() and path.suffix in ['.json','.csv','.npz','.md'] and path.name not in ['archive_delivery_qa.json','publication_staging.json','publication_delivery_qa.json'] and not path.name.endswith('_publication_delivery_qa.json') and not path.name.startswith(('depth_acceleration','depth_benchmark')) and path.relative_to(C).parts[0] not in ['memory_cache_reference','memory_cache_check']:files.add(path)
for path in G.glob('*'):
    if path.is_file() and path.suffix in ['.json','.csv']:files.add(path)
for folder in ['real_elevation','synthetic_03pct','spatial_fold_1','spatial_fold_2','spatial_fold_3','codigo_congelado']:
    files.update(p for p in (G/folder).iterdir() if p.is_file() and p.suffix in ['.json','.csv','.npz','.py','.cpp','.dll'])
old=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
for name in ['config.json','domain.json','vp_final.npz','predictions.csv']:files.add(old/name)
files.add(ROOT/'resultados/revision_independiente_20261011f/estaticas_y_elevacion.csv')
files.add(ROOT/'resultados/robustez_20261011/synthetic_03pct/synthetic_observations.csv')
files.add(ROOT/'datos/cartografia/departamentos_dane_2025.geojson')
files.add(ROOT/'datos/limite_diagonal_nazca_caribe_20261010.json')
for name in ['sam_slab2_dep_02.23.18.grd','sam_slab2_unc_02.23.18.grd','descarga.json']:
    files.add(ROOT/'datos/slab2_2026-10-10'/name)
for name in ['campana_picks.py','campana_picks_v2.py','campana_carma_caribe.py','campana_choco_caribe.py','campana_choco_caribe_senales_fuertes.py']:
    files.add(ROOT/name)
audit=json.loads((C/'arrival_source_audit.json').read_text())
for item in audit['files']:files.add(ROOT/item['path'])
for rel in json.loads((C/'data_description_audit.json').read_text())['source_catalogs']:files.add(ROOT/rel)
files.update((ROOT/'datos').glob('*/campana_manifest.json'))
missing=[str(p) for p in files if not p.is_file()];assert not missing,missing
manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
checks=A/'Manifiesto_revision_20261011h.json';checks.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
dest=A/'Articulo_UIS_revision_20261011h.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
    z.write(checks,checks.relative_to(ROOT).as_posix())
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for item in manifest:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256'],item['path']
report=dict(zip=str(dest),files=len(files)+1,bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),all_archived_hashes_verified=True)
(C/'archive_delivery_qa.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
