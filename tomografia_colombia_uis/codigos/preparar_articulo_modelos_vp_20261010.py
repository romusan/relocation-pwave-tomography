from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'articulo_boletin_geologia_uis_adjunto'
source=(OUT/'crear_manuscrito_verificado.py').read_text(encoding='utf-8')
source=source.replace("heading('4.3 Comparación mediante cortes de velocidad',True)","heading('4.3 Modelos de velocidad inicial y final',True)")
lead="""blocks('''El modelo de velocidad inicial es un volumen 3D homogéneo lateralmente, construido al extender el perfil 1D de la Tabla 2 a todo el dominio. Vp aumenta de 5,31 km/s en superficie a 8,03 km/s a 40 km y 8,28 km/s a 100 km; permanece en 8,28 km/s hasta 180 km. Entre los nodos del perfil se utiliza interpolación lineal. No se introducen placas en este volumen inicial.

El modelo de velocidad final es el volumen heterogéneo obtenido después de las tres rondas de inversión, conservado en la malla directa de 10 km. Su representación espacial se muestra en la fila inferior de la Figura 3; la fila superior muestra el mismo volumen inicial en posiciones equivalentes. Las velocidades finales evaluadas en los centros activos de la parametrización abarcan 4,5527–8,8000 km/s. La heterogeneidad depende de la posición horizontal y la profundidad, por lo que no se sustituye el modelo final por una sola curva 1D.''')
"""
source=source.replace("blocks('''La Figura 3 compara",lead+"blocks('''La Figura 3 compara",1)
addition="""blocks('''La Figura 4 presenta de forma explícita el perfil de velocidad inicial y una síntesis por profundidad del modelo final 3D. La Tabla 5 compara la velocidad inicial con el mínimo, la mediana y el máximo finales en los nodos activos de cada nivel de inversión. Para obtener estos valores se interpoló cada volumen sobre las mismas coordenadas de los 10011 nodos activos; no se incluyeron las zonas sin cobertura. La mediana y las bandas resumen variación lateral del modelo, no constituyen un nuevo modelo 1D ni intervalos de incertidumbre. Las líneas entre niveles facilitan la lectura y no añaden información a la discretización de 20 km. La disminución de nodos activos de 1678 a 40 km a 119 a 180 km muestra la cobertura desigual con la profundidad.''')
DEPTH_MODEL=json.loads((OUT/'fuentes/auditoria_modelos_velocidad.json').read_text(encoding='utf-8'))['rows']
vpformat=lambda v:f'{v:.4f}'.replace('.',',')
table('Tabla 5. Modelos de velocidad inicial y final por profundidad. Todas las velocidades se expresan en km/s y los valores finales corresponden a los nodos activos de cada nivel.',
    ['Profundidad\\n(km)','Vp inicial\\n(km/s)','Vp final\\nmínima','Vp final\\nmediana','Vp final\\nmáxima','Celdas\\nactivas'],
    [[str(int(r['depth_km'])),vpformat(r['vp_initial_km_s']),vpformat(r['vp_final_min_km_s']),vpformat(r['vp_final_median_km_s']),vpformat(r['vp_final_max_km_s']),str(r['active_cells'])] for r in DEPTH_MODEL],
    [2.75,2.6,2.65,2.65,2.65,2.75])
figure('Figura_4_modelos_velocidad.png','Figura 4. Modelo de velocidad inicial y distribución por profundidad del modelo final. a) Perfil inicial 1D extendido lateralmente; los puntos indican los seis nodos de la Tabla 2. b) Valores finales del volumen 3D evaluados en los nodos activos de cada nivel: mediana, percentiles 5–95 y mínimo–máximo. La línea azul discontinua reproduce el perfil inicial para comparar. Las bandas representan heterogeneidad lateral de Vp, no intervalos de confianza. La Figura 3 conserva la información espacial de los modelos mediante cortes coincidentes.')

"""
source=source.replace("heading('5 Discusión de resultados')",addition+"heading('5 Discusión de resultados')",1)
source=source.replace('metricas_articulo_verificado.json','metricas_articulo_modelos.json')
source=source.replace('Manuscrito_tomografia_Vp_Colombia_UIS_corregido.docx','Manuscrito_tomografia_Vp_Colombia_UIS_modelos_inicial_final.docx')
source=source.replace('auditoria_manuscrito_verificado.json','auditoria_manuscrito_modelos.json')
dest=OUT/'crear_manuscrito_modelos.py';assert not dest.exists();dest.write_text(source,encoding='utf-8')
m=json.loads((OUT/'fuentes/metricas_articulo_verificado.json').read_text())
paths=[dest,Path(__file__),OUT/'fuentes/auditoria_modelos_velocidad.json',OUT/'fuentes/comparacion_modelos_vp_profundidad.csv',OUT/'fuentes/modelos_vp_celdas_activas.csv',OUT/'figuras/Figura_4_modelos_velocidad.png',ROOT/'codigos/modelos_velocidad_articulo_20261010.py']
m['input_sha256'].update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(OUT/'fuentes/metricas_articulo_modelos.json').write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf-8')
print(dest)
