"""Close a completed numerical campaign without turning failed gates into success."""
from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'resultados/campana_factorizada_cotas_20261011g'
A=ROOT/'articulo_boletin_geologia_uis_adjunto'
load=lambda p:json.loads((C/p).read_text(encoding='utf-8'))
s=load('nonlinear_synthetics.json');f=load('spatial_validation.json');p=load('spatial_pooled_comparison.json')
assert len(s)==2 and len(f)==3 and p['events']==3621 and p['picks']==32304
lines=['# Cierre de la campaña factorizada con cotas reales','',
'Se completaron la comparación real, los dos dameros no lineales, las tres reservas espaciales, el refinamiento regional y las auditorías complementarias. El cierre no implica que las puertas de resolución y convergencia hayan sido superadas. No se presenta una imagen tectónica validada.','',
'## Resultados comprobados','',
'- Población: 3621 eventos con identificador SGC, 32304 lecturas normalizadas P, 267 identificadores de estación y 264 posiciones físicas. Periodo 2015-04-07 a 2026-10-02. Sin duplicaciones exactas adicionales por evento, posición a 1 m y llegada a 1 ms; mínimo cuatro posiciones por evento.',
'- Comparación real: RMS 1D 1,959693 s; 3D 2,028320 s; diferencia +0,068626 s; IC95 por eventos [+0,041594; +0,092835]. Predicciones condicionadas al catálogo reservado.',
'- Refinamiento 5→2,5 km del 3D final: diferencia RMS 0,171031 s y p95 0,379615 s, por encima de 0,100 y 0,250 s. El perfil inicial sí satisface ambas tolerancias.',
'- En 915 lecturas de 464 eventos de seis estaciones, con modelos y estáticas congelados, ΔRMS 3D−1D cambia de −0,075075 a +0,067140 y +0,185724 s a 10, 5 y 2,5 km. No se extrapola esta prueba al total de lecturas ni a un reajuste a malla fina.',
'- Verificación analítica, manufacturada, adjunto proyectado y equivalencia serial/paralelo satisfactoria. Esto no sustituye una prueba de recuperación.',
'- 2449/9373 celdas activas finales (26,13 %) contienen algún nodo en 8,8 km/s o en el límite inferior acumulado de tres pasos de −5 %. No hay nodos en la cota física inferior de 4 km/s.',
'- Las asociaciones estática–elevación nuevas son débiles y sus intervalos de pendiente incluyen cero. No se acredita una explicación causal por elevación.','']
for x in s:
 lines.append(f"- Damero ±{100*x['amplitude_fraction']:.0f} %: r={x['correlation']:.6f}, L2 relativo={x['relative_l2_error']:.6f}, ganancia={x['least_squares_amplitude_ratio']:.6f}; RMS hipocentral 3D={x['final_3d_hypocentre_rmse_km']:.6f} km y 1D={x['final_1d_hypocentre_rmse_km']:.6f} km. Puerta superada: {x['passes_gate']}.")
lines += ['', 'El RMS inicial continuo es 4,960554 km y el inicial llevado a nodos es 7,101306 km. La degradación acreditada es sintética, no una medida del error real. Los bloques son de 40 km; no se concluye ausencia de información a todas las escalas. La verdad ±8 % alcanza 8,9424 km/s y excede la cota de inversión.', '']
for x in f:
 lines.append(f"- Reserva región {x['fold']}: entrenamiento={x['train_events']}, reserva={x['holdout_events']}, excluidos={x['excluded_buffer_events']}; RMS 1D={x['control_1d']['rms_s']:.6f} s; 3D={x['model_3d']['rms_s']:.6f} s.")
lines += [f"- Agregado de tres reservas: RMS 1D={p['control_1d']['rms_s']:.6f} s; 3D={p['model_3d']['rms_s']:.6f} s; Δ={p['paired_bootstrap']['delta_rms_3d_minus_1d_s']:.6f} s; IC95={p['paired_bootstrap']['ci95_s']}. No es un único modelo ajustado.", '',
'## Revisión propia y alcance', '',
'Se comprueba el resultado adverso del 3D con población y operador comunes. Se comprueba degradación hipocentral en las verdades sintéticas, reconociendo el efecto inicial del redondeo. Se rechaza extrapolarla al catálogo real. La factorización trata la singularidad de fuente, pero el 3D heterogéneo no ha convergido según la tolerancia fijada. Se rechaza extrapolar un error continuo a partir de un orden asintótico no demostrado. Las NURBS históricas y k-means son geometría exploratoria, no fronteras validadas.', '',
'El control mantiene fijo el perfil 1D: no es un modelo mínimo 1D optimizado. Se utiliza P normalizada, sin seguimiento separado Pn/Pg; no hay inversión S. La relocalización de entrenamiento usa nodos de 10 km, la propagación es cartesiana y la cota se incorpora mediante extrapolación de roca, sin superficie topográfica completa ni curvatura terrestre. Los bootstrap no incorporan error numérico ni toda la incertidumbre del modelo.', '',
'## Documentos y reproducción', '',
'El Word g contiene cortes inicial/final y anomalías nuevas, tablas de controles, interpretación diagnóstica y enlaces relativos a tres HTML históricos identificados como tales. El ZIP conserva modelos, predicciones, datos seleccionados, configuraciones, código C++/Python, bibliografía y manifiesto SHA256. No contiene las cachés voluminosas de tiempos ni afirma un depósito público o DOI inexistente. Las rutas absolutas de las configuraciones registran el entorno original y requieren adaptación en otro equipo.', '',
'La ejecución sintética final utilizó factorization_parallel con OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1 y OMP_NUM_THREADS=4. Las regiones se ejecutaron separadamente con cachés por región y el mismo protocolo. El agregador cerrar_metricas_factorizadas_g.py reúne únicamente resultados terminados.', '',
'## Declaraciones confirmadas', '',
'Sin financiación específica; sin conflictos de intereses. Los tres autores participaron por igual en conceptualización, metodología, análisis formal y redacción. Autores y afiliaciones confirmados: Rómulo Sandoval Florez, Universidad ECCI, romulo@ecci.edu.co; Claudia Patricia Parra Medina, Universidad Militar Nueva Granada; Francisco Cabrera Zambrano, Grupo PANGEA, Universidad de Pamplona. La asistencia de IA se declara sin atribuir aprobación externa o editorial.', '']
text='\n'.join(lines)
(A/'Cierre_campana_20261011g.md').write_text(text,encoding='utf-8')
(C/'ESTADO_CAMPANA.md').write_text(text,encoding='utf-8')
(C/'REVISION_TECNICA.md').write_text(text,encoding='utf-8')
protocol=load('protocol.json')
verified={}
for name,expected in protocol['sha256'].items():
    actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert actual==expected,f'Frozen protocol file changed: {name}'
    verified[name]=actual
manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'codigos').glob('*factorizado*20261011g.py')}
(C/'execution_final_manifest.json').write_text(json.dumps(dict(status='all finite experiments completed; structural and mesh gates not passed',runtime_threads=dict(OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,OMP_NUM_THREADS=4),code_sha256=manifest,verified_frozen_protocol_sha256=verified),indent=2),encoding='utf-8')
print(text)
