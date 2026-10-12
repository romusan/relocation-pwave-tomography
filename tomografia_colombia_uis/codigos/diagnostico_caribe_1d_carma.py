"""Common holdout and regional coverage for the northern original-1D expansion."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reconstruir_benioff_ampliado import model_cells
from regional.geometry import Projection
from regional.inputs import sha256
def main():
    old=ROOT/'resultados/migracion_isc_choco_2026-10-10';new=ROOT/'resultados/migracion_caribe_1d_carma_2026-10-10'
    before=pd.read_csv(old/'predictions.csv',low_memory=False);after=pd.read_csv(new/'predictions.csv',low_memory=False)
    h0=before[before.split.eq('holdout')];h1=after[after.split.eq('holdout')]
    pair=h0.merge(h1,on=['event_id','station','fase','raw_row'],suffixes=('_before','_after'),validate='one_to_one');assert len(pair)==len(h0)
    np.testing.assert_allclose(pair.observed_s_before,pair.observed_s_after,atol=1e-10,rtol=0)
    rms=lambda x:float(np.sqrt(np.mean(np.asarray(x)**2))) if len(x) else None
    added=pd.read_csv(new/'added_events.csv');newh=h1[~h1.event_id.isin(h0.event_id)]
    zones={'Caribe_9_13N':(9,13,-79,-71),'Transicion_8_9N':(8,9,-79,-71),'Nazca_suroccidente':(2,6.5,-79,-75)}
    coverage=[]
    for run in [old,new]:
        domain=json.loads((run/'domain.json').read_text());p=pd.read_csv(run/'associated_picks_selected.csv',low_memory=False);e=pd.read_csv(run/'events_split.csv');rel=pd.read_csv(run/'relocated_train_events.csv')
        with np.load(run/'vp_final.npz') as m:c=model_cells(m,Projection(domain))
        counts={}
        for name,(a,b,x,y) in zones.items():
            ce=c[c.lat.ge(a)&c.lat.lt(b)&c.lon.between(x,y)];ee=e[e.lat.ge(a)&e.lat.lt(b)&e.lon.between(x,y)];rr=rel[rel.lat.ge(a)&rel.lat.lt(b)&rel.lon.between(x,y)];pp=p[p.event_id.isin(ee.event_id)&p.fase.eq('P')]
            counts[name]=dict(events=len(ee),events_depth_ge40=int(ee.depth_km.ge(40).sum()),P=len(pp),station_epochs=pp.station.nunique(),active_cells=len(ce),active_cells_depth_ge40=int(ce.depth_km.ge(40).sum()),relocated=len(rr[rr.status.eq('relocated')]))
        coverage.append(dict(run=run.name,zones=counts,total_P_station_epochs=p[p.fase.eq('P')].station.nunique()))
    with np.load(new/'vp_final.npz') as m,np.load(old/'vp_final.npz') as b:np.testing.assert_array_equal(m['vp_initial'],b['vp_initial'])
    result=dict(common_holdout_events=pair.event_id.nunique(),common_holdout_P=len(pair),common_RMS_before_s=rms(pair.residual_s_before),common_RMS_after_s=rms(pair.residual_s_after),all_holdout_RMS_after_s=rms(h1.residual_s),new_holdout_events=newh.event_id.nunique(),new_holdout_P=len(newh),new_holdout_RMS_s=rms(newh.residual_s),original_1d_exact=True,coverage=coverage,new_by_group=added.addition_group.value_counts().to_dict(),limits='Counts measure illumination, not resolution or plate identity; total holdout RMS differs in composition.')
    pair['sq_before']=pair.residual_s_before**2;pair['sq_after']=pair.residual_s_after**2
    grouped=pair.groupby('event_id').agg(a=('sq_before','sum'),b=('sq_after','sum'),n=('sq_before','size')).to_numpy()
    rng=np.random.default_rng(20261009);draw=rng.integers(0,len(grouped),size=(3000,len(grouped)));sums=grouped[draw].sum(axis=1)
    delta=np.sqrt(sums[:,1]/sums[:,2])-np.sqrt(sums[:,0]/sums[:,2]);result['paired_event_bootstrap_delta_95percent_ci_s']=np.quantile(delta,[.025,.975]).tolist()
    result['common_RMS_delta_s']=result['common_RMS_after_s']-result['common_RMS_before_s']
    result['comparison_sha256']={str(f):sha256(f) for f in [Path(__file__),old/'predictions.csv',new/'predictions.csv',old/'vp_final.npz',new/'vp_final.npz']}
    intermediate=ROOT/'resultados/migracion_caribe_1d_estaciones_v2_2026-10-10'
    hm=pd.read_csv(intermediate/'predictions.csv',low_memory=False);hm=hm[hm.split.eq('holdout')]
    cm=hm.merge(h1,on=['event_id','station','fase','raw_row'],suffixes=('_published','_carma'),validate='one_to_one');assert len(cm)==len(hm)
    np.testing.assert_allclose(cm.observed_s_published,cm.observed_s_carma,atol=1e-10,rtol=0)
    result['CARMA_comparison']=dict(common_events=cm.event_id.nunique(),common_P=len(cm),published_only_RMS_s=rms(cm.residual_s_published),with_CARMA_RMS_s=rms(cm.residual_s_carma),limitation='CARMA receivers plus four new events; not a pure station-count experiment.')
    result['comparison_sha256'][str(intermediate/'predictions.csv')]=sha256(intermediate/'predictions.csv')
    pair.to_csv(new/'common_holdout_comparison.csv',index=False);(new/'comparacion_cobertura.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(13,6),sharex=True,sharey=True)
    for ax,run in zip(axs,[old,new]):
        e=pd.read_csv(run/'events_split.csv');s=pd.read_csv(run/'associated_picks_selected.csv',low_memory=False);s=s[s.fase.eq('P')][['station','st_lon_current','st_lat_current']].drop_duplicates()
        ax.scatter(e.lon,e.lat,c=e.depth_km,s=7,cmap='viridis',vmin=0,vmax=180,alpha=.6);ax.scatter(s.st_lon_current,s.st_lat_current,marker='^',s=16,c='red',label='Estaciones con P');ax.axhline(9,c='black',lw=.7);ax.axhline(8,c='black',lw=.7,ls='--');ax.set(xlim=(-79,-71),ylim=(2,13),xlabel='Longitud WGS84',title=run.name);ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(loc='upper left',fontsize=8)
    axs[0].set_ylabel('Latitud WGS84');fig.suptitle('Sismos y estaciones utilizados: mismo inicio 1D');fig.tight_layout();fig.savefig(new/'cobertura_caribe_1d.png',dpi=150);plt.close(fig)
    prep=json.loads((new/'preparation.json').read_text());summary=json.loads((new/'summary.json').read_text());quality=json.loads((new/'added_data_quality.json').read_text())
    selected=pd.read_csv(new/'associated_picks_selected.csv',low_memory=False)
    actual_counts=selected[selected.fase.eq('P')&selected.event_id.isin(added.event_id)].groupby('event_id').station.nunique()
    quality['P_stations_quantiles']={str(k):float(v) for k,v in actual_counts.quantile([0,.25,.5,.75,1]).items()}
    report=f'''# Ampliación Caribe con inicio 1D original

Se ejecutó una nueva inversión con {prep['selected_events']} sismos, incluidos {prep['additional_events']} adicionales. Se conserva exactamente el inicio 1D original, el dominio 2–13 N / −79 a −71, profundidad 0–180 km y tres rondas. No se incorporan perturbaciones NURBS al inicio.

## Comparación justa

Las {result['common_holdout_P']} P de {result['common_holdout_events']} sismos reservados anteriores permanecen idénticas, con los mismos hipocentros y orígenes fijos.

| Métrica | Valor |
|---|---:|
| RMS reservado común anterior | {result['common_RMS_before_s']:.6f} s |
| RMS reservado común nuevo | {result['common_RMS_after_s']:.6f} s |
| RMS nuevos reservados | {result['new_holdout_RMS_s']:.6f} s |
| RMS total reservado nuevo | {result['all_holdout_RMS_after_s']:.6f} s |

La diferencia nueva menos anterior en el conjunto reservado común es {result['common_RMS_delta_s']:.6f} s. Su intervalo del 95 % por 3.000 remuestreos pareados de eventos es [{result['paired_event_bootstrap_delta_95percent_ci_s'][0]:.6f}, {result['paired_event_bootstrap_delta_95percent_ci_s'][1]:.6f}] s. Es incertidumbre de muestreo, no incertidumbre geológica ni del solver.

El RMS total no se compara directamente con el anterior porque cambia la población. El número de estaciones de P pasa de {prep['previous_P_stations']} a {prep['total_P_stations']} identidades de estación/época. Se revisaron {prep['metadata_epochs']} épocas regionales. No se limita el número por evento: se usan todas las llegadas publicadas que pasan asociación, metadatos y dominio. No se afirma disponer de señales de todas las estaciones existentes.

Se eliminaron {prep['duplicate_site_arrivals_removed']} llegadas bajo alias CM/IR con posiciones a ≤100 m para el mismo evento y fase, dando prioridad al metadato CM. Tres eventos dejaron de cumplir el mínimo de cuatro P tras esa revisión. Las identidades de estación/época no son un conteo de sitios físicos independientes. La preparación anterior, con 2.961 eventos, se detuvo y quedó marcada como sustituida; la comparación usa exclusivamente esta versión corregida.

Los nuevos eventos incluyen {quality['depth_ge40']} de profundidad ≥40 km y {quality['depth_ge100']} de profundidad ≥100 km. La mediana del gap es {quality['gap_median_deg']:.1f}°; {quality['gap_le180']} de {quality['additional_events']} tienen gap ≤180°. La mediana es de {quality['P_stations_quantiles']['0.5']:.0f} estaciones P por evento y el máximo es {quality['P_stations_quantiles']['1.0']:.0f}. Los eventos con gap grande no se presentan como superficies bien determinadas; el contraste de calidad y la resolución deben evaluarse sin retunar sobre los reservados.

## Adición de receptores CARMA

Se descargaron señales de 32 eventos SGC del norte, enero–abril de 2018, compatibles con la red temporal CARMA (YU). Se solicitaron todas las estaciones activas regionales disponibles, sin seleccionar solo las más cercanas. Se exige un sensor Z/N/E completo, ventana completa y ausencia de huecos mayores de 0.5 s. Se detectan P/S con PhaseNet instance, umbrales 0.3 y control de asociación al 1D original. Los miniseed y parámetros de consulta quedan registrados con SHA-256.

Se agregan {prep['carma_retained_P']} P y {prep['carma_retained_S']} S de CARMA, con {prep['carma_P_station_epochs']} identidades de estación/época de P. Se amplían {prep['carma_augmented_existing_training_events']} eventos de entrenamiento existentes y se incorporan {prep['carma_new_event_count']} sismos nuevos. Los eventos reservados heredados se excluyeron de esta ampliación de receptores; sus observaciones permanecen congeladas. Los sismos completamente nuevos se dividen antes de invertir. La condición de cuatro P se verifica en el conjunto combinado por evento, permitiendo sumar receptores individuales a eventos existentes que ya cumplen ese mínimo.

La red CARMA se ha utilizado en estudios independientes del Caribe: [Cornthwaite et al. 2021, DOI 10.1029/2020GC009431](https://doi.org/10.1029/2020GC009431) y [Miao et al. 2024, DOI 10.1029/2024GC011612](https://doi.org/10.1029/2024GC011612). Compartir la red no constituye validación independiente de nuestro resultado.

En los {result['CARMA_comparison']['common_events']} reservados y {result['CARMA_comparison']['common_P']} P comunes a la etapa publicada y a la etapa CARMA, el RMS pasa de {result['CARMA_comparison']['published_only_RMS_s']:.6f} a {result['CARMA_comparison']['with_CARMA_RMS_s']:.6f} s. Este contraste incorpora receptores y cuatro eventos nuevos, por lo que no aísla únicamente el efecto del número de estaciones.

## Datos y trazabilidad

Consulta ISC 2018–2023, latitud 8–13 N, longitud −79 a −71, magnitud ≥2.5, profundidad 0–180 km, incluyendo llegadas. Se recuperan orígenes SGC del mismo periodo y se exige coincidencia única dentro de 12 s, 30 km y 40 km de profundidad. Los hipocentros de entrada y las visualizaciones siguen siendo SGC. Las fases locales P/Pg/Pn y S/Sg/Sn publicadas se conservan con agencia y pick ID; Pn/Pg se modelan como primera llegada P en el solver vigente. Probabilidad 1 indica elegibilidad de una llegada publicada, no confianza neuronal. S no entra en la inversión Vp.

Estaciones: metadatos CM del SGC, registro regional ISC y EarthScope. Alias IR→CM solo si las posiciones coinciden a ≤1 km. El registro puede carecer de fechas de época para algunas estaciones IR: esa incertidumbre debe resolverse antes de una interpretación definitiva. Las estaciones fuera del dominio físico se excluyen para evitar recortarlas a los bordes del solver.

Los datos, scripts y respuestas se versionan; se verifican sus SHA-256. Se mantienen las observaciones anteriores y el reparto entrenamiento/reserva; las adiciones se reservan por grupos antes de invertir. Los estáticos se aprenden exclusivamente con entrenamiento.

## Cobertura por zona

'''
    report+='| Zona | Sismos antes/después | P antes/después | Celdas activas ≥40 km antes/después |\n|---|---:|---:|---:|\n'
    for name in zones:
        b=coverage[0]['zones'][name];a=coverage[1]['zones'][name];report+=f"| {name} | {b['events']} / {a['events']} | {b['P']} / {a['P']} | {b['active_cells_depth_ge40']} / {a['active_cells_depth_ge40']} |\n"
    report+='''
## Camino científico hacia una publicación Q1

Esta ampliación mejora el conjunto de datos disponible; no demuestra por sí sola la separación de Nazca, Caribe y Suramericana. La costa y el norte deben evaluarse por profundidad, azimut y cruces de rayos, no solo por número de eventos. La Suramericana requiere delimitar corteza y manto de la placa superior con restricciones independientes; el mapa de Colombia no es una interfaz tectónica.

La pregunta publicable debe discriminar geometrías alternativas: una losa Caribe continua y de bajo ángulo bajo el norte; segmentos Caribe separados o superpuestos a Nazca; y sismicidad en material de la placa superior. Para el suroccidente no se supone que todos los eventos intermedios estén dentro de Nazca. Estas hipótesis necesitan diferencias recuperables por nuestros rayos y contrastes independientes de funciones receptoras, mecanismos focales y geología regional. No se selecciona una interpretación por la apariencia de una superficie interpolada.

1. Corregir y validar el solver frente a soluciones analíticas y un solver independiente: el FSM de primer orden vigente no supera los umbrales estrictos y sus rayos son aproximados.
2. Calibrar incertidumbres P, asociación y estáticos con revisión de llegadas y calidad por estación; incorporar elevación y evaluar curvatura terrestre.
3. Realizar recuperación sintética no lineal con placas separadas, placas conectadas y modelo sin placa, usando la misma geometría de adquisición y ruido realista. El checkerboard lineal existente no basta.
4. Evaluar estabilidad frente al perfil 1D, amortiguamiento, tamaño de celda y estaciones; reservar datos externos o bloques espaciales antes de ajustar parámetros.
5. Definir superficies con incertidumbre y comparar con Slab2, mecanismos focales, discontinuidades y estudios regionales. No atribuir automáticamente cada rama de Benioff a una placa.
6. Para Q, obtener medidas espectrales con respuesta instrumental, incertidumbres y separación fuente/sitio/trayectoria; no inferir Q de Vp.

Como extensión metodológica, las S disponibles pueden sustentar una inversión conjunta Vp/Vs una vez calibradas sus incertidumbres y el forward de S. Esto permitiría contrastar interpretaciones que Vp por sí sola no separa. No se presenta Vs ni Q como invertidas en esta corrida.

Lecturas prioritarias para contrastar la separación de placas y la placa superior:

- [Overlapping slabs: Untangling subduction in NW South America through finite-frequency teleseismic tomography, DOI 10.1016/j.epsl.2021.117253](https://doi.org/10.1016/j.epsl.2021.117253).
- [Lithospheric and Slab Configurations From Receiver Function Imaging in Northwestern South America, Colombia, DOI 10.1029/2022JB024475](https://doi.org/10.1029/2022JB024475).
- [Segmented flat-slab Caribbean subduction beneath northernmost South America from teleseismic receiver function imaging, DOI 10.1016/j.jsames.2025.105501](https://doi.org/10.1016/j.jsames.2025.105501).
- [The Deep Lithospheric Structure of Terrane Accretion ... Beneath Cauca, Colombia, DOI 10.1029/2024JB030067](https://doi.org/10.1029/2024JB030067).
- [Subducting Slab Thermo-Petrologic Structure and Intermediate-Depth Supra-Slab Earthquakes Beneath Southern Colombia, DOI 10.1029/2025GL118610](https://doi.org/10.1029/2025GL118610).

Estos trabajos permiten formular hipótesis alternativas y pruebas; no se utiliza una asignación de placa por apariencia de la nube. La profundidad máxima actual de 180 km limita el seguimiento profundo de las losas.

Referencias metodológicas y datos: [ISC Bulletin](https://isc.ac.uk/iscbulletin/), [USGS Slab Models: guía técnica](https://ghsc.code-pages.usgs.gov/esi/usgs-slab-models/manual/s2_technical_guide.html), [Seismic tomography 2023](https://www.usgs.gov/publications/seismic-tomography-2023). La bibliografía DOI previa se conserva en `referencias/Referencias_tomografia_placas_Colombia_DOI.docx`.

## Reproducción

```powershell
py -3.14 codigos/ampliacion_caribe_1d_datos.py
py -3.14 ampliar_caribe_1d_estaciones_v2.py
py -3.14 campana_carma_caribe.py
py -3.14 ampliar_caribe_1d_carma.py
py -3.14 codigos/verify_run.py resultados/migracion_caribe_1d_carma_2026-10-10
py -3.14 codigos/diagnostico_caribe_1d_carma.py
```

La inversión exige una salida nueva para repetirla. Los archivos originales permanecen preservados.
'''
    (ROOT/'AMPLIACION_CARIBE_1D_CARMA_2026-10-10.md').write_text(report,encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
