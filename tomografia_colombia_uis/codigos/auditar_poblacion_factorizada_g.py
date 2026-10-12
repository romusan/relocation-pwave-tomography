from pathlib import Path
import json,pandas as pd
root=Path(__file__).resolve().parents[1]/'resultados/campana_factorizada_cotas_20261011g'
p=pd.read_csv(root/'selected_picks.csv');e=p.drop_duplicates('event_id')
dates=pd.to_datetime(e.origin_time,utc=True,errors='coerce',format='mixed')
xyz=p[['st_x_km','st_y_km','st_elev_km']].round(3)
locations=xyz.astype(str).agg('|'.join,axis=1);q=p[['event_id','observed_s']].copy();q['location_1m']=locations
sites=q.groupby('event_id').location_1m.nunique();duplicate=q.assign(arrival_ms=q.observed_s.round(3)).duplicated(['event_id','location_1m','arrival_ms'],keep='first')
r=dict(events=len(e),picks=len(p),origin_start_utc=str(dates.min()),origin_end_utc=str(dates.max()),undated_events=int(dates.isna().sum()),phase_counts=p.fase.fillna('not_recorded').value_counts().to_dict(),published_phase_counts=p.published_phase.fillna('not_recorded').value_counts().to_dict(),reporting_agency_counts=p.reporting_agency.fillna('not_recorded').value_counts().to_dict(),origin_agency_counts=e.origin_agency.fillna('not_recorded').value_counts().to_dict(),event_id_samples=e.event_id.head(4).tolist(),minimum_distinct_locations_per_event=int(sites.min()),events_with_fewer_than_four_distinct_locations_1m=int((sites<4).sum()),duplicate_event_location_arrival_at_1ms=int(duplicate.sum()),duplicate_definition='extra rows sharing event, receiver position rounded to 1 m, and observed_s rounded to 1 ms; this audit does not remove records or identify instrumentation provenance')
r['events_without_sgc_identifier']=int((~e.event_id.str.startswith('SGC')).sum())
(root/'dataset_scope_audit.json').write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(r,indent=2,ensure_ascii=False))
