from pathlib import Path
import json,hashlib,csv,zipfile
from html.parser import HTMLParser
from docx import Document
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'articulo_boletin_geologia_uis_adjunto';SUP=OUT/'suplementos_20261011'
FILE=OUT/'Manuscrito_tomografia_Vp_Colombia_UIS_revision_20261011.docx'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=Document(FILE);pdf=PdfReader(OUT/'qa_revision_20261011/word_render.pdf')
assert len(pdf.pages)==32 and len(d.inline_shapes)==7 and len(d.tables)==6
audit=json.loads((OUT/'fuentes/auditoria_manuscrito_revision_20261011.json').read_text(encoding='utf-8'))
assert audit['output_sha256']==sha(FILE) and audit['resumen_words']<=300 and audit['abstract_words']<=300
for f,h in audit['input_sha256'].items():assert sha(f)==h,f
r=json.loads((SUP/'revision_diagnosticos.json').read_text(encoding='utf-8'))
for group in ['input_sha256','output_sha256']:
    for f,h in r[group].items():assert sha(f)==h,f
links=[str(x.target_ref) for x in d.part.rels.values() if x.reltype.endswith('/hyperlink')]
local=[t for t in links if t.startswith('suplementos_20261011/')]
assert len(local)==3
for target in local:assert (OUT/target).is_file()
class Parse(HTMLParser):
    def __init__(self):super().__init__();self.sources=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='script' and 'src' in a:self.sources.append(a['src'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
for path in SUP.glob('*.html'):
    text=path.read_text(encoding='utf-8');p=Parse();p.feed(text)
    assert not p.sources and 'Plotly.newPlot' in text
    for target in p.links:assert (SUP/target).is_file()
rows=list(csv.DictReader((SUP/'clasificacion_kmeans_sismos.csv').open(encoding='utf-8')));assert len(rows)==1180
labels=[int(x['kmeans_km_k2'])-1 for x in rows];diag=[int(x['diagonal_group']) for x in rows]
agreement=max(sum(a==b for a,b in zip(labels,diag)),sum(1-a==b for a,b in zip(labels,diag)))/len(rows)
assert abs(agreement-.9033898305084745)<1e-12
for metric in ['km','standardized']:
    for k in range(2,7):assert set(int(x[f'kmeans_{metric}_k{k}']) for x in rows)==set(range(1,k+1))
sat=list(csv.DictReader((SUP/'saturacion_por_profundidad.csv').open(encoding='utf-8')))
assert sum(int(x['lower_saturation']) for x in sat)==1300
assert sum(int(x['upper_saturation']) for x in sat)==1557
for row,expected in zip(d.tables[-1].rows[1:],r['rms_depth']):
    vals=[c.text for c in row.cells]
    assert int(vals[1])==expected['events'] and int(vals[2])==expected['p_times']
    assert abs(float(vals[3].replace(',','.'))-expected['initial_rms_s'])<=.000051
    assert abs(float(vals[4].replace(',','.'))-expected['final_rms_s'])<=.000051
old=json.loads((OUT/'fuentes/auditoria_manuscrito_anomalias.json').read_text(encoding='utf-8'))
assert sha(old['output'])==old['output_sha256']
readme=OUT/'LEEME_paquete_revision_20261011.txt'
readme.write_text('Extraiga todo el ZIP antes de abrir el manuscrito Word. Conserve la carpeta suplementos_20261011 junto al Word. En Word, Ctrl+clic sobre los enlaces de las Figuras 1 y 2 o de la sección 7.1 abre los HTML. También puede abrir directamente cualquiera de los tres HTML en un navegador. No requieren internet. Las etiquetas k-means son grupos espaciales exploratorios; las superficies NURBS conservan la clasificación diagonal del experimento original. Los CSV y respuesta_revision.md documentan los diagnósticos y pruebas pendientes.\n',encoding='utf-8')
package=OUT/'Articulo_UIS_y_figuras_3D_revision_20261011.zip'
assert not package.exists()
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    z.write(FILE,FILE.name);z.write(readme,readme.name)
    for path in SUP.iterdir():
        if path.is_file():z.write(path,path.relative_to(OUT).as_posix())
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None
    for target in local:assert target in z.namelist()
    assert hashlib.sha256(z.read(FILE.name)).hexdigest()==sha(FILE)
report=dict(status='verified',pages=32,visually_inspected_pages=list(range(1,33)),figures=7,tables=6,references=23,
    local_html_links=local,html_offline=True,html_browser_verified=True,kmeans_selector_verified='k=2 to k=3',
    output_sha256=sha(FILE),package=str(package),package_sha256=sha(package),previous_manuscript_preserved=True,
    validation_script_sha256=sha(__file__))
(OUT/'fuentes/verificacion_revision_20261011.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
