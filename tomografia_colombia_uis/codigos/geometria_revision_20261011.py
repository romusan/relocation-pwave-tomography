"""Surface areas and held-out catalog-depth residuals without refitting sheets."""
from ablacion_revision_20261011 import ROOT,RUN,OUT
import json,sys
import numpy as np,pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from modelo_3d_continuo import evaluate,grd
from regional.geometry import gap_deg
from scipy.cluster.vq import kmeans2
GEO=ROOT/'resultados/placas_adjunto_consistente_2026-10-10'
def main():
    p=pd.read_csv(RUN/'predictions.csv',low_memory=False);h=p[p.split.eq('holdout')]
    events=[]
    for eid,g in h.groupby('event_id'):
        r=g.iloc[0];events.append(dict(event_id=eid,lat=r.ev_lat,lon=r.ev_lon,x=r.source_x_km,y=r.source_y_km,z=r.ev_prof_km,gap=gap_deg(r.ev_lat,r.ev_lon,g.st_lat_current.to_numpy(),g.st_lon_current.to_numpy())))
    e=pd.DataFrame(events);e['eligible']=e.z.ge(40)&e.gap.le(180)
    e['north']=e.lat.ge(6+(e.lon+72.5)*3.2/(-3.8));rows=[];areas=[]
    for name in ['nazca','caribe']:
        mesh=np.load(GEO/(name+'_superficie.npz'));v=mesh['middle'];fs=mesh['faces'];supported=mesh['supported'][fs].all(axis=1)
        a=.5*np.linalg.norm(np.cross(v[fs[:,1]]-v[fs[:,0]],v[fs[:,2]]-v[fs[:,0]]),axis=1)
        areas.append(dict(surface=name,area_km2=float(a.sum()),supported_km2=float(a[supported].sum()),interpolated_km2=float(a[~supported].sum()),interpolated_percent=float(100*a[~supported].sum()/a.sum())))
        sheet=json.loads((GEO/(name+'_nurbs.json')).read_text());sheet={k:np.array(val) if isinstance(val,list) else val for k,val in sheet.items()}
        selected=e[e.eligible&(e.north if name=='caribe' else ~e.north)].copy();xy=selected[['x','y']].to_numpy();uv=(xy-sheet['lo'])/sheet['span'];inside=((uv>=0)&(uv<=1)).all(axis=1)
        fit,_=evaluate(sheet,xy[inside]);selected['in_parameter_domain']=inside;selected['vertical_residual_km']=np.nan;selected.loc[inside,'vertical_residual_km']=fit[:,2]-selected.loc[inside,'z']
        # Domain coverage of exported surface, without silently extrapolating.
        dist=cKDTree(v[:,:2]).query(xy)[0];selected['in_exported_footprint']=dist<=np.sqrt(50)
        selected.to_csv(OUT/(name+'_holdout_geometry.csv'),index=False)
        use=inside&selected.in_exported_footprint.to_numpy();r=selected.loc[use,'vertical_residual_km'].to_numpy()
        rows.append(dict(surface=name,eligible_events=len(selected),evaluated_events=int(use.sum()),outside_footprint=int((~use).sum()),vertical_rms_km=float(np.sqrt(np.mean(r*r))),vertical_p90_abs_km=float(np.quantile(abs(r),.9)),slab2_selection_applied=False))
    (OUT/'geometria.json').write_text(json.dumps(dict(areas=areas,holdout=rows,notes=['Areas are triangle surface areas in local Cartesian km; supported means all three vertices satisfy the original mask.','Heldout catalog coordinates and depths are not relocated or used to fit NURBS; gaps use heldout P stations.','No Slab2 compatibility filter applied to either heldout population; vertical, not orthogonal distances.','Export footprint membership approximated by nearest surface sample within sqrt(50) km, the diagonal of a 5 km cell.','Fixed catalog depths have different location errors than relocated training controls; this is a predictive geometry check, not plate-interface ground truth.']),indent=2),encoding='utf-8')
    # Density control using the same 20 km cell medians used by geometry.
    rel=pd.read_csv(GEO/'asignacion_sismos.csv');sel=rel.status.eq('relocated')&rel.depth_km.ge(40)&rel.gap_deg.le(180);rel=rel[sel].copy();xyz=rel[['x_km','y_km','depth_km']].to_numpy()
    ix=np.floor(xyz/20).astype(int);rel[['ix','iy','iz']]=ix;cells=rel.groupby(['ix','iy','iz'])[['x_km','y_km','depth_km']].median().to_numpy();results=[]
    raw=pd.read_csv(ROOT/'articulo_boletin_geologia_uis_adjunto/suplementos/clasificacion_kmeans_sismos.csv')
    for metric in ['km','standardized']:
        scale=np.ones(3) if metric=='km' else cells.std(axis=0);data=cells/scale;best=None
        for seed in range(20261011,20261041):
            c,l=kmeans2(data,2,minit='++',iter=100,seed=seed);score=((data-c[l])**2).sum()
            if best is None or score<best[0]:best=(score,c,l)
        _,c,l=best;labels=np.argmin(((xyz[:,None,:]/scale-c[None,:,:])**2).sum(axis=2),axis=1)
        geographic=(rel.lat>=6+(rel.lon+72.5)*3.2/(-3.8)).to_numpy();agreement=max(np.mean(labels==geographic),np.mean((1-labels)==geographic))
        results.append(dict(metric=metric,original_events=len(rel),unique20km_cells=len(cells),agreement_diagonal_percent=float(100*agreement),centroids_km=(c*scale).tolist(),note='Cells rather than events equally weighted; internal sensitivity, not independent plate classification.'))
    (OUT/'kmeans_declustering.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(dict(areas=areas,holdout=rows,decluster=results),indent=2),flush=True)
if __name__=='__main__':main()
