from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'articulo_boletin_geologia_uis_adjunto'
source = OUT / 'crear_manuscrito_modelos.py'
builder = OUT / 'crear_manuscrito_anomalias.py'
text = source.read_text(encoding='utf-8')
addition = """heading('4.4 Cortes de anomalías de velocidad',True)
blocks('''La Figura 5 presenta las anomalías porcentuales de Vp del modelo final respecto al volumen inicial 1D. En cada posición se calculó cien veces la diferencia entre Vp final y Vp inicial, dividida por Vp inicial. Se emplean las mismas coordenadas, interpolación y máscara activa de la Figura 3, con una escala divergente común de −16 a +16 %. Los valores positivos indican velocidades mayores que la referencia 1D; los negativos indican velocidades menores.

Los cortes permiten distinguir el contraste lateral del gradiente vertical del modelo de referencia. Los colores no identifican por sí solos una placa o una interfaz tectónica. El gris corresponde a posiciones sin cobertura activa y no a anomalías nulas; los tonos próximos al blanco indican valores cercanos a cero dentro de la máscara. La interpretación conserva las limitaciones de resolución y de las cotas de velocidad descritas anteriormente.''')
figure('Figura_5_cortes_anomalias_Vp.png','Figura 5. Cortes de anomalías porcentuales de Vp final respecto al modelo inicial 1D a 77°, 76° y 74° O. Azul: velocidades mayores que la referencia; rojo: velocidades menores. La escala común abarca −16 a +16 %. El gris identifica posiciones fuera de la máscara activa de inversión. Los puntos negros son sismos SGC relocalizados de entrenamiento dentro de una banda longitudinal de ±0,25° de cada sección. Latitud en grados norte y profundidad positiva hacia abajo; muestreo de aproximadamente 0,18° en latitud y 20 km en profundidad. Las anomalías expresan contrastes relativos y no delimitan automáticamente las superficies candidatas de Nazca y Caribe.')

"""
anchor = "heading('5 Discusión de resultados')"
assert text.count(anchor) == 1
text = text.replace(anchor, addition + anchor)
text = text.replace('metricas_articulo_modelos.json', 'metricas_articulo_anomalias.json')
text = text.replace('Manuscrito_tomografia_Vp_Colombia_UIS_modelos_inicial_final.docx', 'Manuscrito_tomografia_Vp_Colombia_UIS_con_anomalias.docx')
text = text.replace('auditoria_manuscrito_modelos.json', 'auditoria_manuscrito_anomalias.json')
builder.write_text(text, encoding='utf-8')
metrics = json.loads((OUT / 'fuentes/metricas_articulo_modelos.json').read_text(encoding='utf-8-sig'))
paths = [builder, Path(__file__), source,
         OUT / 'Manuscrito_tomografia_Vp_Colombia_UIS_modelos_inicial_final.docx',
         ROOT / 'codigos/cortes_anomalias_articulo_20261010.py',
         OUT / 'fuentes/auditoria_cortes_anomalias.json',
         OUT / 'fuentes/cortes_anomalias_vp.npz',
         OUT / 'figuras/Figura_5_cortes_anomalias_Vp.png']
for path in paths:
    metrics['input_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT / 'fuentes/metricas_articulo_anomalias.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
(OUT / 'artifact_anomalias.md').write_text('Actualización local del manuscrito UIS: sección 4.4 y Figura 5 con tres cortes de anomalías porcentuales respecto al modelo inicial 1D. Se conservan las cuatro figuras previas, cinco tablas, referencias, plantilla y resultados del experimento verificado. Escala divergente común ±16 %, máscara inactiva gris, eventos SGC relocalizados. Originales preservados. Verificación visual en qa_anomalias.\n', encoding='utf-8')
print(builder)
