"""New SGC waveform campaign; isolated, resumable, fixed input identities."""
from pathlib import Path
import json,sys,hashlib,time
import numpy as np
import pandas as pd
import requests
from obspy import UTCDateTime,read,Stream
import torch
import seisbench.models as sbm
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'codigos'))
from regional.geometry import Projection
from regional.inputs import active_stations,load_profile,straight_time,sha256,spatial_order
from campana_picks_v2 import components
OUT=ROOT/'datos/ampliacion_choco_caribe_senales_fuertes_2026-10-10'
SERVICE='https://sismo.sgc.gov.co:8443/fdsnws/dataselect/1/query'


def main():
    domain=json.loads((ROOT/'dominio.json').read_text()); domain.update(lat_max=13.,lon_max=-71.)
    domain.pop('placas_fuera',None); domain.pop('placas_cubiertas',None)
    (OUT/'dominio_ampliado.json').write_text(json.dumps(domain,indent=2))
    projection=Projection(domain)
    initial=pd.read_csv(ROOT/'datos/ampliacion_choco_caribe_2026-10-10/eventos_campana.csv')
    old=pd.read_csv(ROOT/'resultados/migracion_2200_quibdo_antioquia_2026-10-10/events_split.csv')
    ch=pd.read_csv(ROOT/'datos/catalogo_sgc/catalogo_crudo.csv')
    ch=ch[ch.region.str.contains(r' - Choc., Colombia$',regex=True,na=False)&~ch.event_id.isin(old.event_id)
        &~ch.event_id.isin(initial.event_id)&ch.magnitude.ge(2.5)&ch.depth_km.between(40,180)&ch.phases.ge(12)&ch.gap_deg.le(200)&ch.error_depth_km.le(20)].copy()
    coast=pd.read_csv(OUT/'catalogo_caribe_2024.csv')
    coast=coast[~coast.event_id.isin(initial.event_id)&coast.magnitude.ge(2.5)&coast.depth_km.between(0,180)&coast.phases.ge(10)&coast.gap_deg.le(220)&coast.error_depth_km.le(20)].copy()
    # Larger signals first within geographic cells; priorities precede tomography.
    groups=[]
    for d,n,name in [(ch,20,'Choco nuevo profundo'),(coast,40,'Caribe 9–13 N')]:
        d=d.drop_duplicates('event_id').sort_values(['magnitude','phases'],ascending=False)
        d['gap_deg']=d.gap_deg-10*d.magnitude
        order=d.sort_values(['magnitude','phases'],ascending=False).event_id.tolist()
        chosen=d.set_index('event_id').loc[order[:n]].reset_index(); chosen['campaign_group']=name
        groups.append(chosen)
    chosen=pd.concat(groups,ignore_index=True)
    chosen.to_csv(OUT/'eventos_campana.csv',index=False)
    stations=pd.read_csv(OUT/'estaciones_ampliadas.csv')
    profile=load_profile(ROOT/'datos/perfil_vp_vmm_transferencia.csv')
    inputs=[ROOT/'datos/catalogo_sgc/catalogo_crudo.csv',OUT/'catalogo_caribe_2024.csv',
            OUT/'estaciones_ampliadas.csv',ROOT/'datos/perfil_vp_vmm_transferencia.csv',Path(__file__)]
    identity={str(p):sha256(p) for p in inputs}; mp=OUT/'campana_manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=identity: raise ValueError('Campaign sources changed')
    mp.write_text(json.dumps(identity,indent=2))
    torch.set_num_threads(2)
    model=sbm.PhaseNet.from_pretrained('instance'); model.eval()
    session=requests.Session()
    rawdir=OUT/'ondas'; rawdir.mkdir(exist_ok=True)
    states=OUT/'estados'; states.mkdir(exist_ok=True)
    for index,e in enumerate(chosen.itertuples(),1):
        statepath=states/f'{e.event_id}.json'; pickpath=states/f'{e.event_id}.csv'
        if statepath.exists() and json.loads(statepath.read_text()).get('complete'): continue
        origin=UTCDateTime(str(e.time_utc)); active=active_stations(stations,str(origin))
        ex,ey=projection.xy(e.lat,e.lon); sx,sy=projection.xy(active.lat,active.lon)
        active=active.assign(km=np.hypot(sx-ex,sy-ey),x=sx,y=sy)
        active=active[active.km<=400].nsmallest(20,'km')
        requests_lines=[]; windows={}
        for st in active.itertuples():
            p=float(straight_time(np.array([[float(ex),float(ey),e.depth_km]]),np.array([[st.x,st.y,0.]]),profile)[0])
            t0,t1=origin+p-20,origin+p*1.73+30; windows[(st.net,st.sta)]=(t0,t1)
            requests_lines.append(f'{st.net} {st.sta} * * {t0.strftime("%Y-%m-%dT%H:%M:%S")} {t1.strftime("%Y-%m-%dT%H:%M:%S")}')
        payload='\n'.join(requests_lines)+'\n'; wf=rawdir/f'{e.event_id}.mseed'
        record=dict(event_id=e.event_id,complete=False,station_requests=len(active))
        try:
            if not wf.exists():
                response=session.post(SERVICE,data=payload,headers={'Content-Type':'text/plain'},timeout=(5,40))
                record.update(http=response.status_code,bytes=len(response.content))
                if response.status_code==204: record['complete']=True; raise ValueError('No waveform data')
                response.raise_for_status(); read(__import__('io').BytesIO(response.content))
                wf.write_bytes(response.content)
            stream=read(str(wf)); clean=Stream(); valid={}
            for st in active.itertuples():
                try:
                    part=components(stream,st.net,st.sta); t0,t1=windows[(st.net,st.sta)]; part.trim(t0,t1)
                    clean+=part; valid[(st.net,st.sta)]=st
                except Exception: pass
            rows=[]
            if clean:
                result=model.classify(clean,P_threshold=.3,S_threshold=.3,batch_size=128)
                for pick in result.picks:
                    net,sta,*_=pick.trace_id.split('.'); st=valid[(net,sta)]
                    rows.append(dict(event_id=e.event_id,origin_time=str(origin),ev_lat=e.lat,ev_lon=e.lon,
                        ev_prof_km=e.depth_km,net=net,sta=sta,fase=pick.phase[:1],pick_time=str(pick.peak_time),
                        prob=float(pick.peak_value),campaign_group=e.campaign_group))
            pd.DataFrame(rows,columns=['event_id','origin_time','ev_lat','ev_lon','ev_prof_km','net','sta','fase','pick_time','prob','campaign_group']).to_csv(pickpath,index=False)
            record.update(complete=True,valid_station_streams=len(valid),picks=len(rows),waveform_sha256=sha256(wf),request=payload)
        except Exception as exc: record['error']=str(exc)
        statepath.write_text(json.dumps(record,indent=2))
        print(json.dumps(dict(index=index,total=len(chosen),**record)),flush=True)
        time.sleep(.3)
    frames=[pd.read_csv(p) for p in states.glob('*.csv')]
    result=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    result.to_csv(OUT/'picks_nuevos.csv',index=False)
    print(f'Finished {len(result)} candidates in {result.event_id.nunique() if len(result) else 0} events',flush=True)


if __name__=='__main__': main()
