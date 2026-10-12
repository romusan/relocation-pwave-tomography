from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'resultados/campana_perfil_suavidad_20261011h'
p=pd.read_csv(ROOT/'resultados/campana_factorizada_cotas_20261011g/selected_picks.csv',low_memory=False)
events=p[['event_id','ev_lat','ev_lon','ev_prof_km','split']].drop_duplicates('event_id')
records=[];paths=[]
for path in sorted((ROOT/'datos').rglob('catalogo_sgc.csv')):
    q=pd.read_csv(path,low_memory=False)
    if {'event_id','magnitude','magnitude_type'}<=set(q):
        q=q[q.event_id.isin(events.event_id)][['event_id','magnitude','magnitude_type']].copy()
        if len(q):
            q['source_catalog']=path.relative_to(ROOT).as_posix();records.append(q);paths.append(q.source_catalog.iloc[0])
q=pd.concat(records,ignore_index=True).dropna(subset=['magnitude']);q['magnitude']=pd.to_numeric(q.magnitude,errors='coerce');q=q.dropna(subset=['magnitude'])
q['value_key']=q.magnitude.round(3).astype(str)+' '+q.magnitude_type.fillna('unknown')
count=q.groupby('event_id').value_key.nunique();good=count[count.eq(1)].index
unique=q[q.event_id.isin(good)].drop_duplicates('event_id')
events=events.merge(unique[['event_id','magnitude','magnitude_type','source_catalog']],on='event_id',how='left',validate='one_to_one')
events.to_csv(C/'event_descriptors.csv',index=False)
result=dict(events=len(events),magnitude_known=int(events.magnitude.notna().sum()),magnitude_missing_or_conflicting=int(events.magnitude.isna().sum()),magnitude_conflicting_events=int(count.gt(1).sum()),magnitude_types=events.magnitude_type.fillna('not_available').value_counts().to_dict(),source_catalogs=paths,depth_range_km=[float(events.ev_prof_km.min()),float(events.ev_prof_km.max())],probability_equal_one=int(p.prob.eq(1).sum()),probability_below_one=int(p.prob.lt(1).sum()),campaign_groups=p.campaign_group.fillna('not_recorded').value_counts().to_dict(),association_window_max_abs_retained_s=float(abs(p.association_residual_s).max()),published_phase_missing=int(p.published_phase.isna().sum()),picker_note='prob=1 is assigned by published-arrival importers; it is not a calibrated neural-picker score; individual picker identity unavailable for imported published arrivals')
result['magnitude_range_mixed_types']=[float(events.magnitude.min()),float(events.magnitude.max())]
(C/'data_description_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(result,indent=2,ensure_ascii=False))
