from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'articulo_boletin_geologia_uis_adjunto'
source=(OUT/'crear_manuscrito_adjunto.py').read_text(encoding='utf-8')
normalizer="""def numeric_typography(text):
    text=re.sub(r'(?<![\\w])-(?=\\d)', '−\\u2060',str(text))
    return re.sub(r'([0-9,]+) × (10[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)',r'\\1\\u00a0×\\u00a0\\2',text)

"""
# A replacement string cannot contain a literal backslash-u in re.sub.
normalizer=normalizer.replace("r'\\1\\u00a0×\\u00a0\\2'","'\\\\1\\u00a0×\\u00a0\\\\2'")
source=source.replace("def p(text='',style=None,single=False,center=False):",normalizer+"def p(text='',style=None,single=False,center=False):")
source=source.replace('q=D.add_paragraph(text,style)','q=D.add_paragraph(numeric_typography(text),style)')
source=source.replace('q.text=str(text);','q.text=numeric_typography(text);')
source=source.replace('metricas_articulo_auditadas.json','metricas_articulo_verificado.json')
source=source.replace('Manuscrito_tomografia_Vp_Colombia_UIS_adjunto_consistente.docx','Manuscrito_tomografia_Vp_Colombia_UIS_corregido.docx')
source=source.replace('auditoria_manuscrito.json','auditoria_manuscrito_verificado.json')
dest=OUT/'crear_manuscrito_verificado.py';assert not dest.exists();dest.write_text(source,encoding='utf-8')
m=json.loads((OUT/'fuentes/metricas_articulo_auditadas.json').read_text())
m['input_sha256'][str(dest)]=hashlib.sha256(dest.read_bytes()).hexdigest()
m['input_sha256'][str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT/'fuentes/metricas_articulo_verificado.json').write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf-8')
print(dest)
