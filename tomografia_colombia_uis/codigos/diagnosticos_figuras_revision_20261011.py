from ablacion_revision_20261011 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    r=pd.read_csv(OUT/'holdout_ablation_residuals.csv');p=pd.read_csv(RUN/'predictions.csv',low_memory=False)
    ci={};rng=np.random.default_rng(20261011)
    for coord in ['node','continuous']:
        cols=[coord+'_1d_relocated_statics',coord+'_3d_final3d_statics'];g=r.groupby('event_id');ss=g[cols].agg(lambda x:np.sum(x*x)).to_numpy();nn=g.size().to_numpy();vals=[]
        for _ in range(3000):
            ix=rng.integers(0,len(nn),len(nn));rr=np.sqrt(ss[ix].sum(axis=0)/nn[ix].sum());vals.append(rr[1]-rr[0])
        ci[coord]=dict(delta_rms_3d_minus_1d_s=float(np.sqrt(np.mean(r[cols[1]]**2))-np.sqrt(np.mean(r[cols[0]]**2))),ci95_s=np.quantile(vals,[.025,.975]).tolist())
    (OUT/'bootstrap_1d_relocalizado.json').write_text(json.dumps(ci,indent=2),encoding='utf-8')
    summary=json.loads((RUN/'summary.json').read_text());base=json.loads((OUT/'1d_relocated_ablation.json').read_text());rounds=[]
    for i,h in enumerate(summary['history'],1):rounds.append(dict(round=i,train_3d=h['train_rms_after_update_s'],holdout_3d=h['holdout_rms_fixed_catalog_origin_s'],train_1d=base['history'][i-1]['train']['rms_s'],holdout_1d=base['history'][i-1]['holdout_node']['rms_s']))
    pd.DataFrame(rounds).to_csv(OUT/'rms_por_ronda.csv',index=False)
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained');colors=['#555555','#167890','#9b3480']
    for k,c,label in zip(['1d_no_statics','1d_relocated_statics','3d_final3d_statics'],colors,['1D sin correcciones','1D con relocalización y estaciones','3D con relocalización y estaciones']):
        ax[0,0].hist(r['continuous_'+k],bins=np.linspace(-8,8,65),histtype='step',color=c,label=label,density=True)
    ax[0,0].set(xlabel='Residuo P en reserva (s)',ylabel='Densidad',title='a) Coordenadas exactas de catálogo');ax[0,0].legend(fontsize=7)
    for model,c in [('1d',colors[1]),('3d',colors[2])]:
        for split,style in [('train','--'),('holdout','-')]:
            start=summary['baseline']['train_rms_fixed_catalog_s' if split=='train' else 'holdout_rms_fixed_catalog_origin_s']
            ax[0,1].plot(range(4),[start]+[h[split+'_'+model] for h in rounds],marker='o',ls=style,color=c,label=model.upper()+' '+('ajuste' if split=='train' else 'reserva'))
    ax[0,1].set(xlabel='Ronda',ylabel='RMS (s)',title='b) Evaluación original en nodos',xticks=range(4));ax[0,1].legend(fontsize=7)
    h=p[p.split.eq('holdout')].copy();h['r1d']=r.continuous_1d_relocated_statics.to_numpy();h['r3d']=r.continuous_3d_final3d_statics.to_numpy()
    h['bin']=pd.cut(h.dist_epi_km,[0,50,100,200,400,800,1600],right=False)
    distances=[]
    for b,g in h.groupby('bin',observed=True):
        distances.append(dict(distance_bin_km=str(b),picks=len(g),rms_1d=metrics(g.r1d.to_numpy())['rms_s'],rms_3d=metrics(g.r3d.to_numpy())['rms_s']))
    pd.DataFrame(distances).to_csv(OUT/'rms_por_distancia.csv',index=False)
    for key,c in [('rms_1d',colors[1]),('rms_3d',colors[2])]:ax[1,0].plot(range(len(distances)),[d[key] for d in distances],'o-',color=c,label=key[-2:].upper())
    ax[1,0].set_xticks(range(len(distances)),[d['distance_bin_km'] for d in distances],rotation=25,ha='right',fontsize=8);ax[1,0].set(ylabel='RMS (s)',xlabel='Distancia epicentral (km)',title='c) Reserva con posiciones exactas');ax[1,0].legend()
    stations=[]
    for sta,g in h.groupby('station'):
        stations.append(dict(station=sta,picks=len(g),lat=g.st_lat_current.iloc[0],lon=g.st_lon_current.iloc[0],rms_1d=metrics(g.r1d.to_numpy())['rms_s'],rms_3d=metrics(g.r3d.to_numpy())['rms_s']))
    st=pd.DataFrame(stations);st.to_csv(OUT/'rms_por_estacion.csv',index=False)
    im=ax[1,1].scatter(st.lon,st.lat,c=st.rms_3d-st.rms_1d,s=15+np.sqrt(st.picks)*5,cmap='RdBu_r',vmin=-.5,vmax=.5,edgecolors='black',linewidths=.2)
    ax[1,1].set(xlabel='Longitud (°)',ylabel='Latitud (° N)',title='d) Diferencia RMS por estación');fig.colorbar(im,ax=ax[1,1],label='3D menos 1D (s)')
    fig.savefig(OUT/'Figura_S1_ablacion.png',dpi=180);plt.close(fig)
    # Recompute depth statistics at aligned coarse nodes to verify coincidences.
    f=np.load(RUN/'vp_final.npz');v=f['vp'];mask=f['active'];stats=[]
    for k,z in enumerate(f['inversion_zg']):
        vals=v[2*k,::2,::2][mask[k]];initial=f['vp_initial'][2*k,0,0]
        stats.append(dict(depth_km=float(z),active_nodes=len(vals),median_full_precision=float(np.median(vals)),equal_initial_nodes=int(np.count_nonzero(np.isclose(vals,initial,rtol=0,atol=1e-12)))))
    (OUT/'tabla5_verificacion.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');print(json.dumps(dict(bootstrap=ci,depth_stats=stats),indent=2),flush=True)
if __name__=='__main__':main()
