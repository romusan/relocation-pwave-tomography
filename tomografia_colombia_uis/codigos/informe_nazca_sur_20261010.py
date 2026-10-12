"""Measured department acquisition and domain-controlled 1D comparison."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10'
CONTROL=ROOT/'resultados/migracion_control_sur_1d_2026-10-10'
def main():
    def read(run,name):return json.loads((run/f'{name}.json').read_text(encoding='utf-8'))
    p=read(NEW,'preparation');d=read(NEW,'comparacion_control_iluminacion');s=read(NEW,'summary');v=read(NEW,'verification');vc=read(CONTROL,'verification')
    added=pd.read_csv(NEW/'added_events.csv');borders=[]
    for zone,g in added.groupby('addition_group',sort=True):borders.append(f"| {zone} | {len(g)} | {g.lat.min():.3f}–{g.lat.max():.3f}° N | {g.lon.min():.3f}–{g.lon.max():.3f}° | {int(g.depth_km.ge(40).sum())} |")
    comparisons=[]
    for key,name in [('domain_effect','Cambio de malla sin añadir datos'),('data_effect','Adición de datos sobre la misma malla ampliada'),('total_effect','Cambio total respecto al experimento anterior')]:
        r=d[key];ci=r['delta_ci95_s'];comparisons.append(f"| {name} | {r['RMS_before_s']:.6f} | {r['RMS_after_s']:.6f} | {r['RMS_after_s']-r['RMS_before_s']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
    r=d['data_effect'];ci=r['delta_ci95_s']
    judgment='El intervalo del cambio incluye cero; no se demuestra una mejora del RMS común por la adición.' if ci[0]<=0<=ci[1] else ('La comparación común favorece la ampliación: el intervalo del cambio queda por debajo de cero.' if ci[1]<0 else 'La adición empeora el RMS común en esta comparación: el intervalo queda por encima de cero.')
    before,after=[c['department_zones'] for c in d['coverage']];coverage=[]
    for zone,a in before.items():
        b=after[zone];coverage.append(f"| {zone} | {a['events']} → {b['events']} | {a['P']} → {b['P']} | {a['station_epochs']} → {b['station_epochs']} | {a['active_deep_cells']} → {b['active_deep_cells']} | {a['deep_cells_hits20_sectors3']} → {b['deep_cells_hits20_sectors3']} |")
    corridors=[]
    for zone,c in d['corridor_counts'].items():corridors.append(f"| {zone} | {c['inactive_targets']} | {c['targets_now_active']} | {c['targets_now_hits20_sectors3']} |")
    cb,ca=[c['Caribe_8_13N'] for c in d['coverage']]
    text=f'''# Ampliación de Nazca en Cauca Valle del Cauca y Eje Cafetero desde 1D

Se añadieron **{p['additional_events']} sismos SGC**, para un conjunto de **{p['selected_events']} eventos y {p['all_P_station_epochs']} estaciones/épocas con P**. Se conservan los eventos y llegadas de Caribe de la corrida anterior. El modelo vuelve a empezar desde el perfil 1D original, sin placas NURBS, Slab2 ni anomalías impuestas. La ampliación se evalúa por cobertura y predicción; no se atribuye identidad tectónica a un evento por su departamento.

## Cobertura de los departamentos y del sur

| Grupo definido por polígonos DANE | Nuevos eventos | Latitudes | Longitudes WGS84 | Nuevos eventos ≥40 km |
|---|---:|---|---|---:|
{chr(10).join(borders)}

Se buscaron hasta 200 adiciones por grupo. En Cauca y el Eje Cafetero se seleccionaron todos los nuevos eventos que pasaron la asociación, el mínimo de cuatro estaciones P diferentes y el gap ≤240°. Eje Cafetero comprende Caldas, Quindío y Risaralda. Los polígonos y el catálogo previo se conservan como fuentes trazables.

Hay **{p['new_south_of2_events']} adiciones al sur de 2° N**, de las que {len(added[(added.lat<2)&added.depth_km.ge(40)])} son de profundidad ≥40 km. La ampliación aporta {p['new_P']} P para invertir y {p['new_S_retained_QC']} S conservadas para control de calidad. La escasez de sismos profundos nuevos en el extremo sur continúa; los rayos procedentes de otras zonas se incluyen en la cobertura medida.

Se utilizaron llegadas ISC publicadas de 2018–septiembre de 2024, asociadas a hipocentros SGC por coincidencia única de tiempo y posición. Se reutilizaron las llegadas del Eje ya descargadas y eventos asociados antes no seleccionados. También se descargó el catálogo SGC de la franja 0,8–2° N y metadatos SGC/EarthScope para sus estaciones. Se respetaron épocas de operación y se excluyeron códigos de redes sintéticas y duplicaciones del mismo sitio de adquisición. No se impuso un máximo de estaciones por evento.

## Por qué se amplió la malla

El dominio anterior empezaba en 2° N, mientras que el límite DANE de Cauca llega a 0,958° N. La nueva malla va de **0,8° a 13° N, −79° a −71°**, hasta 180 km de profundidad. Se mantiene el origen AEQD anterior, 7,5° N / −75°: las coordenadas físicas y observaciones heredadas no cambian. Las mallas tienen pasos de 10 km para el solver y 20 km para invertir. Ambos experimentos ampliados usan exactamente los mismos ejes y el mismo Vp inicial, igualdad comprobada.

Se calcularon dos experimentos completos en esta malla: un control con los 3362 eventos anteriores y otro con las adiciones. Así se distingue el efecto de abrir el dominio del efecto de añadir datos. Cada uno tiene tres rondas y la misma regularización. Se preserva la reserva anterior y se fija la reserva adicional antes de invertir.

## Comparación sobre las mismas observaciones reservadas

Las tres comparaciones usan los mismos {r['events']} eventos y {r['P']} P heredados; sus hipocentros y tiempos de origen no se optimizan. Los estáticos se ajustan con entrenamiento. El IC 95 % se estima mediante 3000 remuestreos por evento.

| Comparación | RMS anterior s | RMS posterior s | Cambio s | IC 95 % del cambio s |
|---|---:|---:|---:|---|
{chr(10).join(comparisons)}

{judgment} El RMS sobre toda la reserva ampliada es {s['history'][-1]['holdout_rms_fixed_catalog_origin_s']:.6f} s y el de entrenamiento {s['history'][-1]['train_rms_after_update_s']:.6f} s. La población de la reserva ampliada difiere de la anterior; no debe confundirse esa comparación agregada con la comparación común.

## Qué ganamos en iluminación

Los valores siguientes se calculan con rayos curvos finales de **entrenamiento** sobre la misma malla ampliada. Los conteos de eventos, P y estaciones incluyen toda la selección de cada departamento; las celdas corresponden a su huella geográfica y profundidad ≥40 km. La cobertura de reserva no se emplea para activar celdas.

| Departamento o grupo | Eventos control → ampliado | P | Estaciones/épocas P | Celdas activas ≥40 km | Celdas ≥40 km con ≥20 rayos y ≥3 sectores |
|---|---:|---:|---:|---:|---:|
{chr(10).join(coverage)}

En los corredores amplios de búsqueda de Nazca pasan a activas **{d['targets_now_active']} de {d['targets_before_inactive']} celdas objetivo** antes inactivas; {d['targets_now_hits20_sectors3']} cumplen además el criterio de diversidad de orientación. La región evaluada incluye el Pacífico adyacente; no se limita la propagación a las fronteras administrativas.

| Corredor de búsqueda | Objetivos antes inactivos | Ahora activos | Ahora con ≥20 rayos y ≥3 sectores |
|---|---:|---:|---:|
{chr(10).join(corridors)}

Los corredores se superponen y sus filas no deben sumarse; el total anterior cuenta celdas únicas. En todo el dominio, {d['previous_active_cells_lost']} celdas del control pasan a inactivas con el nuevo modelo, porque también cambian las trayectorias. En Caribe, 8–13° N, las celdas activas ≥40 km cambian de {cb['active_deep_cells']} a {ca['active_deep_cells']} y las de ≥20 rayos/≥3 sectores de {cb['deep_cells_hits20_sectors3']} a {ca['deep_cells_hits20_sectors3']}.

La selección previa usó trayectorias rectas solo como aproximación de adquisición. El diagnóstico compara inactividad del control ampliado con el nuevo modelo: objetivos entre 40 y 160 km, próximos a cobertura activa a ≤80 km, o ≤160 km en la extensión al sur de 2° N. Los sectores son orientaciones horizontales fuente–receptor, módulo 180°, agrupadas en seis bins de 30°; no son ángulos de incidencia locales ni resolución de una interfaz.

## Figuras y reproducibilidad

- [Visor 3D actualizado](resultados/nazca_cauca_valle_eje_1d_vista_2026-10-10/caribe_nazca_vp_final_3d.html), con latitud, longitud y profundidad, fronteras de Colombia y departamentos y eventos SGC relocalizados. Los fragmentos Benioff siguen siendo locales y provisionales.
- [Imagen 3D](resultados/nazca_cauca_valle_eje_1d_vista_2026-10-10/caribe_nazca_vp_final.png), con exageración vertical 3× declarada; el visor usa proporciones físicas sin exageración.
- [Iluminación control frente a ampliación](resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10/comparacion_iluminacion_sur.png).
- [Perfiles de Vp y sismicidad](resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10/perfiles_transicion.png), desde 0,8° N. Las celdas se dibujan sin interpolar a través de los huecos.
- [Resultados cuantitativos y hashes](resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10/comparacion_control_iluminacion.json).

La verificación del control comprobó {vc['source_files_checked']} archivos fuente y {vc['predictions_checked']} predicciones; la ampliación, {v['source_files_checked']} fuentes y {v['predictions_checked']} predicciones. Se comprobaron coherencia de residuales, reserva fija, unicidad de eventos y fases y convergencia de LSMR. No se modificaron entradas de corridas anteriores.

El modelo sigue siendo experimental: FSM de primer orden, rayos discretos aproximados, estaciones a z=0 e incertidumbres de llegadas aún sin calibración. Esta actividad aumenta adquisición y evalúa su efecto, pero no certifica una placa Nazca completa. Para definir continuidad o límites Caribe–Nazca, falta contrastar modelos sintéticos continuos, separados y superpuestos y evidencia independiente, siguiendo la [revisión bibliográfica anterior](referencias/REVISION_HUECOS_CARIBE_NAZCA_2026-10-10.md). No se producen Vs ni Q nuevos.
'''
    (ROOT/'AMPLIACION_NAZCA_CAUCA_VALLE_EJE_1D_2026-10-10.md').write_text(text,encoding='utf-8');print('Nazca expansion report exported',flush=True)
if __name__=='__main__':main()
