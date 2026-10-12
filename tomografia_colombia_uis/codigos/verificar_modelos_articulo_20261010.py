from pathlib import Path
import json,hashlib
from docx import Document
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'articulo_boletin_geologia_uis_adjunto'
FILE=OUT/'Manuscrito_tomografia_Vp_Colombia_UIS_modelos_inicial_final.docx'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=Document(FILE);pdf=PdfReader(OUT/'qa_modelos/word_render.pdf')
assert len(pdf.pages)==25 and len(d.inline_shapes)==4 and len(d.tables)==5
audit=json.loads((OUT/'fuentes/auditoria_manuscrito_modelos.json').read_text(encoding='utf-8'))
assert audit['output_sha256']==sha(FILE)
for f,h in audit['input_sha256'].items():assert sha(f)==h,f
models=json.loads((OUT/'fuentes/auditoria_modelos_velocidad.json').read_text())
for f,h in models['inputs'].items():assert sha(f)==h,f
for f,h in models['outputs'].items():assert sha(f)==h,f
rows=models['rows'];table=d.tables[-1]
assert len(table.rows)==len(rows)+1
for r,expected in zip(table.rows[1:],rows):
    values=[float(c.text.replace(',','.')) for c in r.cells]
    reference=[expected[k] for k in ['depth_km','vp_initial_km_s','vp_final_min_km_s','vp_final_median_km_s','vp_final_max_km_s','active_cells']]
    assert all(abs(a-b)<=.000051 for a,b in zip(values,reference)),values
assert sum(int(r.cells[-1].text) for r in table.rows[1:])==10011
text='\n'.join(p.text for p in d.paragraphs)
assert '4.3 Modelos de velocidad inicial y final' in text and 'Figura 4.' in text and 'Tabla 5.' in text
assert audit['references']==23 and audit['resumen_words']<=300 and audit['abstract_words']<=300
for n in range(1,15):assert sha(OUT/'qa_modelos'/f'page-{n}.png')==sha(OUT/'qa_verificado'/f'page-{n}.png')
for n in range(15,26):assert (OUT/'qa_modelos'/f'page-{n}.png').is_file()
old=json.loads((OUT/'fuentes/auditoria_manuscrito_verificado.json').read_text())
assert sha(old['output'])==old['output_sha256']
report={'status':'verified','pages':25,'figures':4,'tables':5,'references':23,
    'new_render_pages_visually_inspected':list(range(15,26)),
    'identical_to_previously_inspected_pages':list(range(1,15)),
    'explicit_velocity_models_section':'4.3','spatial_comparison_figure':3,'profile_comparison_figure':4,'depth_values_table':5,
    'table_values_verified_against_model':True,'active_nodes':10011,
    'output_sha256':sha(FILE),'pdf_sha256':sha(OUT/'qa_modelos/word_render.pdf'),
    'validation_script_sha256':sha(__file__),'previous_delivery_preserved':True}
(OUT/'fuentes/verificacion_modelos_entrega.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
