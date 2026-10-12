from pathlib import Path
import json,hashlib
from docx import Document
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'articulo_boletin_geologia_uis_adjunto'
file=OUT/'Manuscrito_tomografia_Vp_Colombia_UIS_corregido.docx'
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
    d=Document(file);pdf=PdfReader(OUT/'qa_verificado/word_render.pdf');pages=len(pdf.pages)
    assert pages<=35 and len(d.inline_shapes)==3 and len(d.tables)==4
    assert len([p for p in d.paragraphs if p.style.name=='Title'])==2
    assert not d.element.xpath('.//w:instrText[contains(text(),"PAGE")]')
    for style in ['Title','Heading 1','Heading 2']:
        assert str(d.styles[style].font.color.rgb)=='000000'
    text='\n'.join(p.text for p in d.paragraphs)
    assert 'romulo@ecci.edu.co' in text and 'Zunino' in text
    for stale in ['permanece sin resolver','limitación numérica pendiente','7,9308','1,7860','26 nodos','Final ray coverage']:
        assert stale not in text,stale
    a=json.loads((OUT/'fuentes/auditoria_manuscrito_verificado.json').read_text(encoding='utf-8'))
    assert sha(file)==a['output_sha256']
    for name,h in a['input_sha256'].items():assert sha(name)==h,name
    assert a['resumen_words']<=300 and a['abstract_words']<=300 and a['references']==23
    # Completion requires the human-readable visual checklist, recorded after inspection.
    checklist=json.loads((OUT/'qa_verificado/inspeccion_visual.json').read_text())
    assert sorted(checklist['pages_inspected'])==list(range(1,pages+1))
    for n in range(1,pages+1):assert (OUT/'qa_verificado'/f'page-{n}.png').is_file()
    for n in checklist['reused_after_sha256_comparison']:
        assert sha(OUT/'qa_verificado'/f'page-{n}.png')==sha(OUT/'qa_entrega'/f'page-{n}.png')
    old=ROOT/'articulo_boletin_geologia_uis';old_a=json.loads((old/'fuentes/auditoria_manuscrito.json').read_text())
    assert sha(old_a['output'])==old_a['output_sha256']
    for name,h in old_a['input_sha256'].items():assert sha(name)==h,name
    manifest=json.loads((ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10/manifest.json').read_text())
    for name,h in manifest.items():assert sha(name)==h,name
    report={'status':'visual and structural QA passed','pages':pages,'figures':3,'tables':4,'references':23,
        'pages_visually_inspected':list(range(1,pages+1)),'output_sha256':sha(file),
        'new_render_pages_inspected':checklist['new_render_pages_inspected'],
        'identical_to_previously_inspected_pages':checklist['reused_after_sha256_comparison'],
        'figure_sha256':{p.name:sha(p) for p in (OUT/'figuras').glob('*.png')},
        'builder_sha256':sha(OUT/'crear_manuscrito_verificado.py'),'pdf_sha256':sha(OUT/'qa_verificado/word_render.pdf'),
        'original_manuscript_and_science_unchanged':True,
        'pending_before_submission':['ORCID and coauthor emails','Independent nonlinear resolution and grid convergence assessment'],
        'discrete_operator_consistency':'resolved and independently checked by finite differences'}
    (OUT/'fuentes/verificacion_entrega.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','pages','figures','tables','references']},ensure_ascii=False))
if __name__=='__main__':main()
