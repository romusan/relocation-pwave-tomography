"""Summarize measured expansion results without asserting plate identity."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def main():
    run=ROOT/'resultados/migracion_eje_caribe_1d_2026-10-10'
    def read(name):return json.loads((run/(name+'.json')).read_text(encoding='utf-8'))
    p=read('preparation');d=read('comparacion_iluminacion');s=read('summary');v=read('verification')
    before,after=[x['zones'] for x in d['coverage']]
    table=[]
    for zone in before:
        a,b=before[zone],after[zone]
        table.append(f"| {zone} | {a['events']} → {b['events']} | {a['P']} → {b['P']} | {a['active_deep_cells']} → {b['active_deep_cells']} | {a['cells_hits20_sectors3_deep']} → {b['cells_hits20_sectors3_deep']} |")
    targets=[]
    for zone,x in d['target_zones'].items():
        targets.append(f"| {zone} | {x['previous_inactive_targets']} | {x['targets_now_active']} | {x['targets_now_hits20_sectors3']} | {x['previous_active_cells_lost']} |")
    ci=d['delta_bootstrap_95percent_ci_s'];delta=d['common_RMS_after_s']-d['common_RMS_before_s']
    judgment='No se demuestra una mejora del RMS sobre la reserva común: el intervalo incluye cero.' if ci[0]<=0<=ci[1] else ('La reserva común favorece la ampliación en esta comparación; el intervalo queda por debajo de cero.' if ci[1]<0 else 'El RMS de la reserva común empeora; el intervalo queda por encima de cero.')
    text=f'''# Ampliación del Eje Cafetero y Caribe desde el modelo inicial 1D

Se incorporaron **{p['groups']['Eje_Cafetero']} sismos del Eje Cafetero y {p['groups']['Caribe']} de Caribe**: {p['selected_events']} eventos seleccionados, {s['train_events']} de entrenamiento y {s['holdout_events']} de reserva. Hay {p['all_P_station_epochs']} estaciones/épocas P utilizables en el conjunto. Los hipocentros de entrada son SGC; las llegadas adicionales proceden de eventos ISC vinculados de forma única por tiempo y posición. No se mezclan hipocentros ISC en la visualización. No se impone un límite tectónico en 8° N.

## Datos y comparación controlada

Las adiciones aportan {p['new_P']} llegadas P y {p['new_S_retained_QC']} S conservadas para control de calidad. Se agregaron {p['new_deep_events']} eventos con profundidad ≥40 km. Las consultas abarcan 2018–septiembre de 2024, profundidad ≤180 km y magnitud ≥2 para ISC. Se recuperaron también eventos del Eje ya asociados y antes no seleccionados. Se eliminaron duplicaciones de sitios de adquisición en los candidatos; no se alteraron observaciones heredadas.

Se comprobó igualdad exacta del Vp inicial con el experimento anterior: perfil 1D original, sin NURBS ni perturbaciones de placas. Se conservaron mallas, parámetros, observaciones y división de reserva heredada. Las adiciones se dividieron antes de invertir, con la semilla anterior. Inversión P únicamente, tres rondas, campos FSM y relocalización/backtrace con C++ donde ya estaba habilitado. No se calcula Vs ni Q nuevos.

## Ajuste y validación

Sobre los mismos **{d['common_holdout_events']} eventos / {d['common_holdout_P']} llegadas P** reservados del experimento anterior:

| Medida | Resultado |
|---|---:|
| RMS anterior | {d['common_RMS_before_s']:.6f} s |
| RMS ampliado sobre las mismas observaciones | {d['common_RMS_after_s']:.6f} s |
| Cambio ampliado − anterior | {delta:+.6f} s |
| IC 95 % del cambio, bootstrap por evento (3000 muestras) | [{ci[0]:+.6f}, {ci[1]:+.6f}] s |
| RMS de toda la nueva reserva | {d['all_new_holdout_RMS_s']:.6f} s |
| RMS final de entrenamiento | {s['history'][-1]['train_rms_after_update_s']:.6f} s |

{judgment} El RMS de toda la nueva reserva corresponde a una población ampliada y no es directamente intercambiable con el anterior. Los hipocentros y tiempos de origen de reserva quedan fijos; sus tiempos de origen no se ajustan. Los estáticos de estación se obtienen del entrenamiento.

## Qué cambia en los huecos

La selección priorizó aproximaciones de rayos hacia celdas antes inactivas entre 40 y 160 km, situadas a ≤60 km de celdas activas, con diversidad azimutal y menor redundancia espacial. La ganancia siguiente se mide con los **rayos curvos del modelo final**, no con la aproximación de selección.

| Zona | Eventos antes → después | P antes → después | Celdas activas ≥40 km | Celdas ≥40 km con ≥20 rayos y ≥3 sectores |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

| Zona | Objetivos antes inactivos | Objetivos ahora activos | Objetivos con ≥20 rayos y ≥3 sectores | Celdas antes activas que pasan a inactivas |
|---|---:|---:|---:|---:|
{chr(10).join(targets)}

En total, **{d['targets_now_active']} de {d['previous_inactive_targets']} celdas objetivo** pasan a activas; {d['targets_now_hits20_sectors3']} cumplen además el criterio de diversidad de orientación. En todo el dominio {d['previous_active_cells_lost']} celdas antes activas pasan a inactivas: las trayectorias también cambian al actualizar el modelo. Los sectores son seis bins de orientación horizontal fuente–receptor de 30°, módulo 180°; no son ángulos locales de incidencia ni una medida de resolución. Los datos reservados no aportan rayos a esta cobertura de entrenamiento.

No se rellenan los vacíos mediante interpolación para simular continuidad. Se conservan triángulos Benioff locales con restricciones de apoyo. Las celdas activas y la nube de eventos no asignan automáticamente identidad Nazca/Caribe; tampoco demuestran una interfaz de Suramérica.

## Bibliografía y próximos contrastes

Se creó un [complemento Word con ocho referencias anotadas y DOI](referencias/Complemento_referencias_huecos_Caribe_Nazca_2026-10-10.docx), comprobado en Crossref y verificado visualmente en sus tres páginas. Conserva el Word anterior sin modificaciones.

La [revisión con ocho referencias y DOI](referencias/REVISION_HUECOS_CARIBE_NAZCA_2026-10-10.md) complementa el Word existente. Sun et al. (2022), [10.1016/j.epsl.2021.117253](https://doi.org/10.1016/j.epsl.2021.117253), y Cornthwaite et al. (2021), [10.1029/2020GC009431](https://doi.org/10.1029/2020GC009431), motivan contrastar segmentación y superposición. Bishop et al. (2025), [10.1029/2024JB030067](https://doi.org/10.1029/2024JB030067), exige considerar sismicidad supraslab en Cauca: toda la nube no tiene necesariamente la misma identidad tectónica.

Para distinguir un hueco de adquisición de una separación real, el siguiente contraste requiere sintéticos de placa continua, separada y superpuesta a diferentes escalas, con ruido y relocalización; los checkerboards lineales disponibles no sustituyen esa prueba. Rawlinson y Spakman (2016), [10.1093/gji/ggw084](https://doi.org/10.1093/gji/ggw084), fundamenta esa distinción. También deben contrastarse funciones receptoras y mecanismos focales que no se hayan impuesto primero al modelo.

## Figuras y trazabilidad

- [Visor 3D geográfico](resultados/eje_caribe_1d_vista_final_2026-10-10/caribe_nazca_vp_final_3d.html): latitud, longitud, profundidad; fronteras de Colombia y departamentos; sismos SGC relocalizados; fragmentos provisionales y Vp final.
- [Imagen 3D](resultados/eje_caribe_1d_vista_final_2026-10-10/caribe_nazca_vp_final.png).
- [Cobertura antes/después](resultados/migracion_eje_caribe_1d_2026-10-10/comparacion_iluminacion.png): proyección, no superficie de placa.
- [Perfiles de transición](resultados/migracion_eje_caribe_1d_2026-10-10/perfiles_transicion.png): longitudes −77°, −76°, −75° y −74°, corredor ±0.25°. Banda 7–9° N solo para comparar.
- [Diagnóstico cuantitativo](resultados/migracion_eje_caribe_1d_2026-10-10/comparacion_iluminacion.json).
- [Verificación](resultados/migracion_eje_caribe_1d_2026-10-10/verification.json): {v['source_files_checked']} fuentes verificadas por SHA256, {v['predictions_checked']} predicciones comprobadas y reserva fija.

El estado científico sigue siendo experimental. El FSM de primer orden tiene errores numéricos conocidos, las estaciones se proyectan a z=0, las incertidumbres de las llegadas no están calibradas y las trayectorias discretas son aproximadas. La ampliación evalúa adquisición y predicción; todavía no certifica un límite Caribe–Nazca ni resolución suficiente para Q1.
'''
    (ROOT/'AMPLIACION_EJE_CARIBE_1D_2026-10-10.md').write_text(text,encoding='utf-8')
    print('Report exported',flush=True)
if __name__=='__main__':main()
