"""Audit and figures derived exclusively from completed robustness outputs."""
from robustez_20261011 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from regional.rays import mapping

def main():
    mesh=json.loads((OUT/'mesh/summary.json').read_text())
    syn=json.loads((OUT/'nonlinear_synthetics.json').read_text())
    spatial=json.loads((OUT/'spatial_validation.json').read_text())
    for folder in [*OUT.glob('spatial_fold_*'),*OUT.glob('synthetic_*pct')]:
        path=folder/'summary.json';metadata=json.loads(path.read_text())
        metadata['linear_lcurve']={'performed':False,'reason':'fixed nonlinear/spatial protocol; no retuning of regularization'}
        metadata['linear_resolution']['performed']=False
        metadata['metadata_note']='Linear diagnostics omitted in the archived inversion engine; no phantom CSV is referenced.'
        save(path,metadata)
    figure_dir=OUT/'figures';figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(14,4.6))
    labels={'homogeneous':'Homogéneo, Vp=6 km/s','initial_1d':'Modelo inicial 1D','final_3d':'Modelo final 3D'}
    for a,m in zip(ax,mesh['models']):
        a.bar([0,1],[m['delta_10_5']['rms_s'],m['delta_5_2p5']['rms_s']],color=['#6c58a3','#ec963f'])
        a.axhline(.1,color='black',ls='--',label='Umbral RMS predefinido: 0,10 s')
        a.set_xticks([0,1],['10 → 5 km','5 → 2,5 km']);a.set_title(labels[m['model']]);a.set_ylabel('RMS del cambio de tiempo (s)')
        a.text(.03,.98,f"P95 5–2,5 km: {m['p95_abs_5_2p5_s']:.3f} s\nOrden aparente: {m['apparent_order']:.2f}",transform=a.transAxes,va='top',fontsize=9)
        a.set_ylim(0,max(m['delta_10_5']['rms_s']*1.35,.2))
    fig.suptitle('Convergencia directa sobre el dominio completo y seis estaciones fijas',y=1.02)
    fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(figure_dir/'Figura_S3_convergencia.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
    depth_metrics=[];oracle_metrics=[]
    for row,amp in enumerate([.03,.08]):
        f=np.load(OUT/f'synthetic_{int(amp*100):02d}pct/nonlinear_recovery.npz')
        mapflat=mapping(GRID,COARSE);count=np.bincount(mapflat,minlength=f['active'].size)
        red=lambda a:(np.bincount(mapflat,weights=a.ravel(),minlength=f['active'].size)/count).reshape(COARSE.shape)
        imposed=100*red((f['truth_vp']-f['base_vp'])/f['base_vp']);got=100*red((f['recovered_vp']-f['base_vp'])/f['base_vp'])
        # Representation oracle: a coarse additive-slowness model can recover
        # the declared coarse-average target without fitting any observations.
        # This checks attainability of the frozen metric, not inversion success.
        delta=np.zeros(f['active'].size);base=f['base_vp'];target=imposed/100
        for _ in range(12):
            oracle=1/(1/base+delta[mapflat].reshape(GRID.shape))
            value=red((oracle-base)/base);derivative=red(-oracle**2/base)
            delta-=((value-target)/derivative).ravel()
        oracle=1/(1/base+delta[mapflat].reshape(GRID.shape));value=red((oracle-base)/base)
        mask=f['active'];a=target[mask];b=value[mask]
        oracle_metrics.append(dict(amplitude_percent=int(100*amp),coarse_target_correlation=float(np.corrcoef(a,b)[0,1]),relative_l2_error=float(np.linalg.norm(b-a)/np.linalg.norm(a)),oracle_vp_min=float(oracle.min()),oracle_vp_max=float(oracle.max()),within_global_velocity_bounds=bool(oracle.min()>=CFG['vp_min_km_s'] and oracle.max()<=CFG['vp_max_km_s']),kind='algebraic representation of target, no observations or refit'))
        for k,depth in enumerate(COARSE.z):
            mask=f['active'][k];target=imposed[k][mask];estimated=got[k][mask]
            if len(target)<2 or np.std(target)==0:continue
            depth_metrics.append(dict(amplitude_percent=int(amp*100),depth_km=float(depth),active_cells=int(mask.sum()),correlation=float(np.corrcoef(target,estimated)[0,1]),amplitude_ratio=float(np.dot(target,estimated)/np.dot(target,target)),relative_l2=float(np.linalg.norm(estimated-target)/np.linalg.norm(target))))
        for col,(depth,kind) in enumerate([(60,'Impuesto'),(60,'Recuperado'),(120,'Impuesto'),(120,'Recuperado')]):
            k=int(np.argmin(abs(COARSE.z-depth)));a=axes[row,col];v=imposed[k] if kind=='Impuesto' else got[k]
            v=np.where(f['active'][k],v,np.nan)
            im=a.pcolormesh(COARSE.x,COARSE.y,v,cmap='RdBu',vmin=-16,vmax=16,shading='nearest')
            title_kind='Referencia agregada' if kind=='Impuesto' else kind
            a.set_title(f'{title_kind}, {depth} km\nEscenario ±{int(amp*100)} %');a.set_aspect('equal');a.set_xlabel('X proyectada (km)')
            if col==0:a.set_ylabel('Y proyectada (km)')
    fig.suptitle('Datos generados a 5 km; inversión no lineal a 10 km, celdas de 20 km',y=.98)
    fig.subplots_adjust(left=.05,right=.86,top=.86,bottom=.08,wspace=.30,hspace=.35)
    color_axis=fig.add_axes([.915,.13,.018,.70])
    fig.colorbar(im,cax=color_axis,label='Anomalía de Vp respecto al 1D (%)')
    fig.savefig(figure_dir/'Figura_S4_sinteticos.png',bbox_inches='tight');plt.close(fig)
    pd.DataFrame(depth_metrics).to_csv(OUT/'depth_recovery_metrics.csv',index=False)
    save(OUT/'representation_oracle.json',dict(cases=oracle_metrics,definition='Both truth and recovered anomalies are sampled on common 10 km nodes and averaged using identical direct-to-coarse mapping; not a pointwise fine-grid truth metric. Checker edges are at x0/y0/z0 + multiples of 40 km and may cross coarse cells.'))
    p=pd.read_csv(OUT/'spatial_events.csv');fig,ax=plt.subplots(1,2,figsize=(12,5.5))
    for fold,color,name in zip(range(3),['#7750ac','#ba9d42','#e5792e'],['Sur <4,5° N','Centro 4,5–8° N','Norte ≥8° N']):
        d=p[p.band.eq(fold)];ax[0].scatter(d.ev_lon,d.ev_lat,s=4,color=color,alpha=.65,label=f'{name}: {len(d)} eventos')
    geo=json.loads((ROOT/'datos/cartografia/departamentos_dane_2025.geojson').read_text(encoding='utf-8'))
    for feature in geo['features']:
        g=feature['geometry'];polys=[g['coordinates']] if g['type']=='Polygon' else g['coordinates']
        for poly in polys:
            ring=np.asarray(poly[0]);ax[0].plot(ring[:,0],ring[:,1],color='gray',lw=.35)
    ax[0].set_xlim(-79,-71);ax[0].set_ylim(.8,13);ax[0].set_xlabel('Longitud (°)');ax[0].set_ylabel('Latitud (°)');ax[0].legend(markerscale=2,fontsize=8);ax[0].set_title('Regiones retiradas del entrenamiento')
    x=np.arange(3);r=spatial['folds'];ax[1].bar(x-.18,[v['control_1d']['rms_s'] for v in r],.36,label='Control 1D',color='#717d8c');ax[1].bar(x+.18,[v['model_3d']['rms_s'] for v in r],.36,label='Modelo 3D',color='#7154a2')
    ax[1].set_xticks(x,['Sur','Centro','Norte']);ax[1].set_ylabel('RMS reservado (s)');ax[1].set_title('Mismos eventos retirados para ambos modelos');ax[1].legend(loc='lower center',bbox_to_anchor=(.5,-.16),ncol=2)
    fig.tight_layout();fig.savefig(figure_dir/'Figura_S5_validacion_espacial.png',bbox_inches='tight');plt.close(fig)
    # Recompute all spatial RMS from individual retained observations.
    checks=[]
    for fold in [1,2,3]:
        folder=OUT/f'spatial_fold_{fold}';r=pd.read_csv(folder/'exact_holdout_residuals.csv');expected=json.loads((folder/'comparison.json').read_text())
        for model,col in [('control_1d','residual_1d_s'),('model_3d','residual_3d_s')]:
            value=metrics(r[col].to_numpy())['rms_s'];assert abs(value-expected[model]['rms_s'])<1e-12
        part=pd.read_csv(folder/'partition.csv');assert not set(part[part.split.eq('train')].event_id)&set(part[part.split.eq('holdout')].event_id)
        xy=p.set_index('event_id');tr=xy.loc[part[part.split.eq('train')].event_id];te=xy.loc[part[part.split.eq('holdout')].event_id]
        nearest=cKDTree(te[['source_x_km','source_y_km']]).query(tr[['source_x_km','source_y_km']])[0].min();assert nearest>=20-1e-10
        checks.append(dict(fold=fold,minimum_train_test_horizontal_distance_km=float(nearest),disjoint_events=True,recomputed_rms=True))
    combined=pd.concat([pd.read_csv(OUT/f'spatial_fold_{fold}/exact_holdout_residuals.csv') for fold in [1,2,3]],ignore_index=True)
    assert combined.event_id.nunique()==3734
    pooled=dict(events=combined.event_id.nunique(),picks=len(combined),control_1d=metrics(combined.residual_1d_s.to_numpy()),model_3d=metrics(combined.residual_3d_s.to_numpy()),bootstrap=bootstrap(combined,combined.residual_3d_s.to_numpy(),combined.residual_1d_s.to_numpy()))
    save(OUT/'verified_summary.json',dict(mesh=mesh,synthetics=syn,spatial=spatial,pooled_spatial=pooled,representation_oracle=oracle_metrics,checks=checks,script_sha256=digest(__file__),original_inputs_unchanged=all(digest(path)==h for path,h in json.loads((OUT/'protocol.json').read_text())['sha256'].items())))
    text=['# Controles de robustez de la tomografía Vp','', 'Se ejecutaron los tres controles con parámetros congelados. Las particiones espaciales son retrospectivas sobre los eventos ya seleccionados y asociados; no representan un catálogo ciego nuevo.','', '## Convergencia directa de malla','', '| Modelo | RMS 10–5 km (s) | RMS 5–2,5 km (s) | P95 5–2,5 km (s) | Orden aparente | Cumple umbral |','|---|---:|---:|---:|---:|---|']
    for m in mesh['models']:text.append(f"| {labels[m['model']]} | {m['delta_10_5']['rms_s']:.3f} | {m['delta_5_2p5']['rms_s']:.3f} | {m['p95_abs_5_2p5_s']:.3f} | {m['apparent_order']:.2f} | {'Sí' if m['passes_frozen_gate'] else 'No'} |")
    text+=['','Se usaron seis estaciones seleccionadas por número de lecturas, dos por banda de latitud, y todos sus trayectos. Las fuentes de estación permanecen en los mismos nodos físicos del modelo original; fuentes de evento interpoladas en XYZ exactas. Los límites del dominio, ocho barridos y el interpolante trilineal de lentitud se mantienen. Los umbrales predefinidos son RMS ≤0,10 s, P95 ≤0,25 s y reducción del error entre refinamientos. Esta prueba evalúa cálculo directo; no se reinvirtieron los 3734 eventos a 2,5 km.','', '## Sintéticos no lineales','', '| Amplitud | Correlación | Razón de amplitud | Error L2 relativo | RMS hipocentral (km) | Cumple umbral |','|---|---:|---:|---:|---:|---|']
    for r in syn['experiments']:text.append(f"| ±{100*r['amplitude_fraction']:.0f} % | {r['correlation']:.3f} | {r['least_squares_amplitude_ratio']:.3f} | {r['relative_l2_error']:.3f} | {r['hypocenter_3d_rmse_km']:.2f} | {'Sí' if r['passes_gate'] else 'No'} |")
    text+=['','Generación FSM a 5 km desde posiciones exactas; inversión a 10 km con celdas de 20 km y tres rondas nuevas de relocalización, estaciones y Vp. Damero de 40 km, ruido gaussiano σ=0,05 s, errores de catálogo uniformes de ±5 km, perturbaciones de origen σ=0,2 s y estáticas σ=0,15 s. Se conservaron los pesos de inversión del experimento real (σ=1 s), sin optimizarlos para estos sintéticos. La prueba ±8 % alcanza Vp=8,94 km/s en parte del modelo verdadero, superior a la cota de inversión de 8,8 km/s; ese contraste cuantifica también la restricción de amplitud. Generar en otra malla evita reutilizar el Jacobiano, aunque ambos cálculos usan la misma familia de solucionador de primer orden.','', '## Validación espacial','', '| Región retirada | Entrenamiento | Reserva | Excluidos por margen | RMS 1D (s) | RMS 3D (s) | Δ3D–1D e IC95 (s) |','|---|---:|---:|---:|---:|---:|---|']
    for name,r in zip(['Sur','Centro','Norte'],spatial['folds']):
        b=r['paired_bootstrap'];lo,hi=b['ci95_s'];text.append(f"| {name} | {r['train_events']} | {r['holdout_events']} | {r['buffer_excluded_events']} | {r['control_1d']['rms_s']:.3f} | {r['model_3d']['rms_s']:.3f} | {b['delta_rms_3d_minus_1d_s']:+.3f} [{lo:+.3f}; {hi:+.3f}] |")
    text+=['',f"RMS espacial agregado sobre {pooled['events']} eventos y {pooled['picks']} lecturas: control 1D {pooled['control_1d']['rms_s']:.3f} s, 3D {pooled['model_3d']['rms_s']:.3f} s. Cada evento se evalúa una sola vez. El bootstrap remuestrea eventos completos; no sustituye incertidumbre entre regiones ni elimina el sesgo de una asociación común previa.",'', 'Las tres bandas son <4,5° N, 4,5–8° N y ≥8° N. Cada inversión empieza otra vez desde el 1D original. Se excluyen eventos de entrenamiento a menos de 20 km horizontales de cualquier evento reservado. Los orígenes y posiciones de reserva permanecen fijos; los tiempos se interpolan en sus coordenadas exactas. Las estaciones exclusivas de reserva reciben corrección cero. No se ajustan restricciones ni regularización usando esta reserva.','', '## Alcance para el envío','', 'Realizar un ensayo no implica que el modelo supere su criterio. Los resultados anteriores determinan si la precisión numérica y la recuperación permiten interpretar amplitudes. La geometría de las superficies no se convierte en una identificación independiente de placas por estos controles. Elevaciones, curvatura, ponderación de lecturas, perfil regional, comparación tectónica y declaraciones de autores conservan el alcance descrito en el manuscrito.']
    (OUT/'Informe_robustez_20261011.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    report=OUT/'Informe_robustez_20261011.md'
    report.write_text(report.read_text(encoding='utf-8').replace('ocho barridos','un máximo de ocho ciclos de ocho direcciones (64 barridos direccionales)'),encoding='utf-8')
    with report.open('a',encoding='utf-8') as f:
        f.write('\n## Verificación de la optimización de barridos\n\nLa parada anticipada solo se activa cuando un ciclo completo no cambia ningún tiempo, sin tolerancia aproximada. El estado es entonces un punto fijo del mismo operador y los ciclos restantes darían el mismo resultado. Se certificó igualdad bit a bit frente al cálculo original en un volumen aleatorio, el modelo 1D real y tres campos sintéticos completos a 5 km; los cinco máximos de diferencia son exactamente cero. Esta optimización no corrige el error de discretización; su evidencia se conserva en fsm_equivalence.json.\n')
        f.write('\n## Definición del objetivo de recuperación\n\nLas métricas comparan anomalías verdaderas muestreadas en los nodos comunes de 10 km y promediadas con la misma asignación a celdas de 20 km que las recuperadas. No son una distancia punto a punto al modelo generador de 5 km. Los bordes del damero pueden cruzar celdas de inversión. Un control algebraico adicional, sin usar tiempos ni reinvertir datos, construye una perturbación de lentitud por celda que reproduce ese promedio: sus métricas se conservan en representation_oracle.json. Sirve para comprobar que el objetivo agregado es representable; no altera los criterios congelados ni acredita recuperación a partir de observaciones.\n')
    print(json.dumps(pooled,indent=2),flush=True)
if __name__=='__main__':main()
