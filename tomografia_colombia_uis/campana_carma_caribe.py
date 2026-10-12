"""All active regional CARMA receivers, fixed SGC cohort, no heldout augmentation."""
from pathlib import Path
import sys,json,time,io
import numpy as np
import pandas as pd
import requests
from obspy import read,Stream,UTCDateTime
import torch
import seisbench.models as sbm
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'codigos'))
from regional.inputs import active_stations,sha256,load_profile,straight_time,spatial_order
from regional.geometry import Projection
from campana_picks_v2 import components
OUT=ROOT/'datos/carma_caribe_2026-10-10'
SERVICE='https://service.earthscope.org/fdsnws/dataselect/1/query'
def main():
    OUT.mkdir(exist_ok=True);(OUT/'ondas').mkdir(exist_ok=True);(OUT/'estados').mkdir(exist_ok=True)
    prev=ROOT/'resultados/migracion_caribe_1d_estaciones_v2_2026-10-10';domain=json.loads((prev/'domain.json').read_text());pr=Projection(domain)
    base=pd.read_csv(prev/'events_split.csv');held=set(base[base.split.eq('holdout')].event_id)
    source=ROOT/'datos/caribe_1d_estaciones_completas_2026-10-10/catalogo_sgc_completo.csv';c=pd.read_csv(source);c['t']=pd.to_datetime(c.time_utc,utc=True,format='mixed')
    c=c[c.t.between('2018-01-01','2018-04-28')&c.lat.between(8,13)&c.magnitude.ge(2.5)&c.phases.ge(6)&c.depth_km.between(0,180)&~c.event_id.isin(held)].copy()
    order=spatial_order(c,pr);c=c.set_index('event_id').loc[order].reset_index();c.to_csv(OUT/'eventos_campana.csv',index=False)
    stsource=ROOT/'datos/caribe_1d_estaciones_completas_2026-10-10/estaciones_todas_regionales.csv';stations=pd.read_csv(stsource);stations=stations[stations.net.eq('YU')].copy();stations.to_csv(OUT/'estaciones_carma.csv',index=False)
    profile=load_profile(ROOT/'datos/perfil_vp_vmm_transferencia.csv');torch.set_num_threads(2);model=sbm.PhaseNet.from_pretrained('instance');model.eval()
    inputs=[source,stsource,prev/'events_split.csv',Path(__file__),ROOT/'campana_picks_v2.py',ROOT/'datos/perfil_vp_vmm_transferencia.csv']+list((Path.home()/'.seisbench/models/v3/phasenet').glob('instance.*'))
    signature={str(p):sha256(p) for p in inputs};mp=OUT/'campana_manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=signature:raise ValueError('Campaign sources changed')
    mp.write_text(json.dumps(signature,indent=2),encoding='utf-8');session=requests.Session()
    for index,e in enumerate(c.itertuples(),1):
        recordpath=OUT/'estados'/f'{e.event_id}.json';pickpath=recordpath.with_suffix('.csv')
        if recordpath.exists() and json.loads(recordpath.read_text()).get('complete'):continue
        origin=UTCDateTime(e.t.timestamp());active=active_stations(stations,str(origin));ex,ey=pr.xy(e.lat,e.lon);sx,sy=pr.xy(active.lat,active.lon)
        active=active.assign(x=sx,y=sy);windows={};durations=[]
        for st in active.itertuples():
            tt=float(straight_time(np.array([[ex,ey,e.depth_km]]),np.array([[st.x,st.y,0.]]),profile)[0]);windows[st.sta]=(origin+tt-20,origin+tt*1.73+30);durations.append(tt*1.73+30)
        q=dict(net='YU',sta=','.join(active.sta),loc='*',cha='BH?',starttime=(origin-20).strftime('%Y-%m-%dT%H:%M:%S'),endtime=(origin+max(durations)).strftime('%Y-%m-%dT%H:%M:%S'))
        wf=OUT/'ondas'/f'{e.event_id}.mseed';record=dict(event_id=e.event_id,complete=False,station_requests=len(active),query=q,service=SERVICE)
        try:
            if not wf.exists():
                response=session.get(SERVICE,params=q,timeout=(10,45));record.update(http=response.status_code,bytes=len(response.content))
                if response.status_code==204:record.update(complete=True,error='No waveform data');raise ValueError('No waveform data')
                response.raise_for_status();read(io.BytesIO(response.content));wf.write_bytes(response.content)
            stream=read(str(wf));clean=Stream();valid=[];rejections=[]
            for st in active.itertuples():
                try:
                    part=components(stream,'YU',st.sta);t0,t1=windows[st.sta];part.trim(t0,t1)
                    if not part or any(t.stats.starttime>t0+.5 or t.stats.endtime<t1-.5 for t in part):raise ValueError('Window incomplete')
                    clean+=part;valid.append(st.sta)
                except Exception as exc:rejections.append(dict(sta=st.sta,error=str(exc)))
            rows=[]
            if clean:
                result=model.classify(clean,P_threshold=.3,S_threshold=.3,batch_size=128)
                for pick in result.picks:
                    net,sta,*_=pick.trace_id.split('.')
                    rows.append(dict(event_id=e.event_id,origin_time=str(origin),ev_lat=e.lat,ev_lon=e.lon,ev_prof_km=e.depth_km,net=net,sta=sta,fase=pick.phase[:1],pick_time=str(pick.peak_time),prob=float(pick.peak_value),campaign_group='CARMA todos los receptores regionales'))
            pd.DataFrame(rows,columns=['event_id','origin_time','ev_lat','ev_lon','ev_prof_km','net','sta','fase','pick_time','prob','campaign_group']).to_csv(pickpath,index=False)
            record.update(complete=True,valid_station_streams=len(valid),valid_stations=valid,rejected_stations=rejections,picks=len(rows),waveform_sha256=sha256(wf))
        except Exception as exc:record['error']=str(exc)
        recordpath.write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(dict(index=index,total=len(c),event_id=e.event_id,complete=record['complete'],valid_station_streams=record.get('valid_station_streams',0),picks=record.get('picks',0),error=record.get('error'))),flush=True);time.sleep(.3)
    frames=[pd.read_csv(f) for f in (OUT/'estados').glob('*.csv')];p=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame();p.to_csv(OUT/'picks_nuevos.csv',index=False)
    print('CARMA campaign exported '+str(len(p))+' candidate picks',flush=True)
if __name__=='__main__':main()
