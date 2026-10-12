from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'articulo_boletin_geologia_uis_adjunto'
t=(OUT/'crear_manuscrito_revision_20261011b.py').read_text(encoding='utf-8')
for a,b in [
    ('(28,54 %), casi tres de cada diez, la solución','(28,54 %; casi tres de cada diez) la solución'),
    ('La cobertura final activó 10011 celdas.', 'La cobertura final activó 10011 celdas; las amplitudes quedan condicionadas por los límites impuestos en el 28,5 % de ellas.'),
    ('Final discrete sensitivity coverage activated 10011 cells.', 'Final discrete sensitivity coverage activated 10011 cells; recovered amplitudes are constrained by the imposed bounds in 28.5% of them.'),
    ('Manuscrito_tomografia_Vp_Colombia_UIS_revision_20261011b.docx','Manuscrito_tomografia_Vp_Colombia_UIS_revision_20261011c.docx'),
    ('auditoria_manuscrito_revision_20261011b.json','auditoria_manuscrito_revision_20261011c.json'),
]:
    assert a in t,a
    t=t.replace(a,b)
a="    for t in text.strip().split('\\n\\n'):p(t.replace('\\n',' '))"
b="    for t in text.strip().split('\\n\\n'):\n        q=p(t.replace('\\n',' '))\n        if t.startswith('De los 1180 eventos elegibles'):q.paragraph_format.keep_together=True"
assert a in t
t=t.replace(a,b)
(OUT/'crear_manuscrito_revision_20261011c.py').write_text(t,encoding='utf-8')
print('Prepared revision c')
