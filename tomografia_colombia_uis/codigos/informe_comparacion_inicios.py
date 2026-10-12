"""Write final scientific comparison without selecting parameters on holdout."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=ROOT/'resultados/comparacion_1d_placas_continuas_2026-10-10';r=json.loads((p/'comparacion.json').read_text(encoding='utf-8'))
    model=ROOT/'resultados/modelo_1d_placas_continuas_slab2_2026-10-10';method=json.loads((model/'metodologia.json').read_text(encoding='utf-8'));adjust=json.loads((model/'ajuste_slab2.json').read_text())
    outcome='El inicio 3D reduce el RMS reservado frente al inicio 1D.' if r['holdout_improvement_percent']>0 else 'El inicio 3D es más continuo geométricamente, pero no reduce el RMS reservado frente al inicio 1D en esta prueba.'
    rows='\n'.join(f"| {s['name']} | {s['sample_points']:,} | {s['faces']:,} | {s['source']} |" for s in method['sheets'])
    ci=r['event_cluster_bootstrap_delta_95percent_ci_s']
    report=f'''# Comparación del inicio 1D con el inicio 1D más placas continuas

{outcome} La prueba conserva el perfil 1D original, los mismos sismos, observaciones, mallas, regularización y tres rondas. Las placas se agregan únicamente como una perturbación tridimensional con espesor; no se reemplaza el fondo por el perfil 1D derivado de la corrida anterior.

## Resultado de la inversión

| RMS | Inicio 1D | Inicio 1D más placas |
|---|---:|---:|
| Reservados antes de invertir | {r['initial_holdout_1d_rms_s']:.6f} s | {r['initial_holdout_3d_rms_s']:.6f} s |
| Entrenamiento después de tres rondas | {r['final_train_1d_rms_s']:.6f} s | {r['final_train_3d_rms_s']:.6f} s |
| Reservados después de tres rondas | {r['final_holdout_1d_rms_s']:.6f} s | {r['final_holdout_3d_rms_s']:.6f} s |

Cambio relativo del RMS reservado: **{r['holdout_improvement_percent']:.3f} % de mejora**, donde un valor negativo significa deterioro. Cambio del RMS de entrenamiento: **{r['train_improvement_percent']:.3f} % de mejora**.

La comparación usa los mismos **{r['reserved_events']} eventos y {r['reserved_P']} P reservadas**, y {r['train_P']} P de entrenamiento. Las observaciones UTC, coordenadas y grupos se comprobaron uno a uno. Los hipocentros y orígenes reservados permanecen fijos; cada modelo aprende sus correcciones de estación únicamente con entrenamiento. Ambos modelos completaron tres rondas con los mismos límites de velocidad y algoritmos.

La diferencia pareada 3D menos 1D es {r['holdout_delta_3d_minus_1d_s']:.6f} s. El intervalo del 95 % por 3.000 remuestreos de eventos reservados es [{ci[0]:.6f}, {ci[1]:.6f}] s. Mejoran {r['held_events_with_lower_rms_3d']} de {r['reserved_events']} eventos. Este intervalo mide la variación por muestreo de eventos, no errores sistemáticos de la geometría, el solver o los catálogos.

Las celdas activas finales son {r['active_final_cells_1d']} para el inicio 1D y {r['active_final_cells_3d']} para el inicio 3D. El número de celdas activas no demuestra resolución ni continuidad real de placas.

## Cómo se construyó el inicio 3D

1. Se conserva exactamente el perfil VMM de referencia: 5.31 km/s a 0 km; 6.48 a 10; 7.22 a 20; 8.03 a 40; 8.28 a 100 y 180 km. Es el mismo fondo utilizado por la corrida 1D de control.
2. Se descargaron de USGS las mallas Slab2 de profundidad e incertidumbre SAM, CAR y CAM, con versiones fechadas en febrero de 2018. Las longitudes 0–360 se convierten a WGS84 y los valores negativos de profundidad a profundidad positiva hacia abajo.
3. SAM tiene cobertura válida en el dominio colombiano. CAR y CAM no tienen valores válidos dentro de latitud 2–13 y longitud −79 a −71. No se trasladan esas mallas para inventar cobertura Caribe. SAM se utiliza como referencia geométrica y no como identificación automática de una placa.
4. La superficie SAM se ajusta con una NURBS cúbica global de 14 por 18 puntos de control, con pesos unitarios. Se pondera el prior por su incertidumbre y se incorporan sismos relocalizados de entrenamiento, profundidad ≥40 km y gap ≤180°. Los eventos cuya diferencia vertical respecto al centro aproximado Slab2 es ≤30 km contribuyen al ajuste. La mediana de diferencia con la profundidad de interfaz Slab2 es {adjust['Slab2_top_depth_fit_median_difference_km']:.3f} km y el percentil 90 absoluto es {adjust['Slab2_top_depth_fit_p90_absolute_km']:.3f} km.
5. Los sismos más someros y más profundos respecto de esa referencia se conservan como ramas separadas. Se utilizan celdas 3D de 20 km, componentes con vecindad 100 km, al menos 12 controles, recortes con aristas de hasta 150 km y distancia al control de hasta 100 km. Se ajusta una spline global por rama, eliminando las costuras de las ventanas de 150 km de la versión anterior.
6. El espesor es 30 km perpendicular a la superficie. Para SAM, Slab2 representa la interfaz superior; la superficie media se desplaza 15 km siguiendo la normal y la inferior otros 15 km. Para las ramas sísmicas se considera la nube como aproximación a la superficie media. Se guarda la geometría de cada sólido con paredes laterales.
7. Los contrastes iniciales de Vp se obtienen del Vp final de entrenamiento respecto al 1D original, mediante hasta ocho celdas activas a ≤60 km, limitados a ±3 %. Donde no hay apoyo de Vp local se aplica el escenario de +3 % del antecedente FMM. Slab2 aporta geometría, no una medición de Vp. La transición cosenoidal tiene 15 km y los contrastes no se suman en superposición: se selecciona la superficie media más cercana.

| Superficie geométrica | Puntos | Triángulos | Apoyo |
|---|---:|---:|---|
{rows}

Las tres mallas tienen una sola componente conectada. Se modifican {method['changed_nodes']:,} nodos del volumen; fuera de las bandas, el inicio 3D conserva **exactamente** el Vp 1D. El muestreo superficial es de 5 km y el volumen de 10 km; la distancia normal a superficies curvas es aproximada. Las superficies NURBS, nudos, puntos de control, normales y recortes se guardan para inspección.

Las clases de apoyo distinguen: 1, cerca de sismos locales; 2, interpolación entre controles Benioff; 3, prior Slab2 sin sismos locales próximos. La continuidad del inicio es una hipótesis impuesta y no un resultado observado. Las ramas no se etiquetan automáticamente como Nazca o Caribe. La placa suramericana anfitriona se representa por el fondo 1D; Q requiere datos espectrales y no se infiere de Vp.

## Interpretación y límites

Un inicio más continuo puede guiar la inversión, pero no garantiza menor RMS. Debe evaluarse la predicción reservada y no solo el ajuste de entrenamiento o la apariencia de las superficies. La información usada para construir geometría y contrastes procede de una inversión anterior del mismo entrenamiento; se trata de una comparación condicionada por ese prior. Los datos de Slab2 pueden compartir entradas sísmicas con los catálogos utilizados. Esta prueba no constituye una confirmación independiente de identidad tectónica.

No se eligieron espesores, amplitudes o parámetros usando el nuevo RMS reservado. Se mantuvieron el espesor acordado de 30 km, contraste máximo 3 %, sigma provisional, regularización y solver anteriores. El FSM de primer orden mantiene sus limitaciones analíticas y el trazado de rayos aproximado. Una siguiente evaluación debería usar recuperación sintética y otro conjunto externo para estudiar la sensibilidad de la geometría y del contraste.

## Archivos y reproducción

- Inicio 3D: `resultados/modelo_1d_placas_continuas_slab2_2026-10-10/vp_inicial_1d_mas_placas_30km.npz`.
- Modelo continuo interactivo: `resultados/modelo_1d_placas_continuas_slab2_2026-10-10/modelo_inicial_continuo_3d.html`.
- Tomografía ejecutada: `resultados/tomografia_1d_mas_placas_continuas_2026-10-10`.
- Comparación de métricas y observaciones: `resultados/comparacion_1d_placas_continuas_2026-10-10`.
- Sección interactiva comparativa: `resultados/comparacion_1d_placas_continuas_2026-10-10/comparacion_1d_3d.html`.

Las figuras calculan todas las anomalías respecto al mismo perfil 1D original. Las secciones muestran el volumen completo, incluidas regiones que conservan el prior por falta de cobertura; no deben interpretarse como estructura resuelta en esas regiones.

```powershell
py -3.14 codigos/test_modelo_continuo.py
py -3.14 codigos/modelo_3d_continuo.py --output resultados/modelo_continuo_nuevo
py -3.14 invertir_desde_nurbs.py --run resultados/migracion_isc_choco_2026-10-10 --initial-model resultados/modelo_continuo_nuevo/vp_inicial_1d_mas_placas_30km.npz --output resultados/inversion_continua_nueva --experimental-first-order
py -3.14 codigos/verify_run.py resultados/inversion_continua_nueva
```

Las salidas exigen directorios nuevos para preservar las corridas anteriores. La tentativa que utilizaba el fondo 1D derivado de Vp final fue detenida al aclarar el usuario que quería el mismo 1D original y no forma parte de esta comparación.

Fuentes: [mallas oficiales USGS Slab2](https://apps.usgs.gov/shakemap_geodata/slabs/), [Hayes et al. 2018](https://doi.org/10.1126/science.aat4723), [datos Slab2](https://doi.org/10.5066/F7PV6JNV), [repositorio oficial del código](https://github.com/usgs/slab2).
'''
    (ROOT/'COMPARACION_INICIOS_1D_3D_2026-10-10.md').write_text(report,encoding='utf-8');print(outcome)
if __name__=='__main__':main()
