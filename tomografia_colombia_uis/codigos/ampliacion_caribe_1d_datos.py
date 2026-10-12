"""Fetch northern ISC arrivals and SGC origins, keeping versioned raw responses."""
from pathlib import Path
import sys,json,datetime,concurrent.futures,xml.etree.ElementTree as ET
import requests
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from descarga_catalogo import params,CONSULT_URL,HEADERS_HTTP
from arma_catalogo import lee
OUT=ROOT/'datos/caribe_1d_estaciones_completas_2026-10-10'
NS={'b':'http://quakeml.org/xmlns/bed/1.2'}
def value(el,path,default=''):
    return el.findtext('/'.join('b:'+p for p in path.split('/')),default=default,namespaces=NS)
def fetch(url,path,query=None,post=False):
    if path.exists():return
    r=requests.post(url,data=query,headers=HEADERS_HTTP,timeout=(15,180)) if post else requests.get(url,params=query,timeout=(15,240))
    r.raise_for_status()
    if 'Software error' in r.text:raise ValueError('Server error, not data')
    path.write_bytes(r.content)
    path.with_suffix(path.suffix+'.query.json').write_text(json.dumps(dict(url=r.url,params=query,status=r.status_code,bytes=len(r.content),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2),encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=True)
    q=dict(format='xml',starttime='2018-01-01',endtime='2023-12-31T23:59:59',minlatitude=8,maxlatitude=13,minlongitude=-79,maxlongitude=-71,minmagnitude=2.5,mindepth=0,maxdepth=180,includearrivals='true')
    def isc():fetch('https://www.isc.ac.uk/fdsnws/event/1/query',OUT/'isc_norte.xml',q)
    def sgc(year):
        p=params(dict(lon_min=-79,lon_max=-71,lat_min=8,lat_max=13,prof_min=0,prof_max=180));p.update(inicial=f'01/01/{year}',final=f'31/12/{year}')
        fetch(CONSULT_URL,OUT/f'sgc_{year}.html',p,True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(isc)]+[pool.submit(sgc,y) for y in range(2018,2024)]
        for f in concurrent.futures.as_completed(fs):f.result();print('Response saved',flush=True)
    sgc=pd.concat([pd.read_csv(ROOT/'datos/catalogo_sgc/catalogo_crudo.csv')]+[pd.DataFrame(lee(p.read_text(encoding='utf-8',errors='replace'),p.stem)) for p in OUT.glob('sgc_*.html')],ignore_index=True).drop_duplicates('event_id')
    sgc['time_utc']=pd.to_datetime(sgc.time_utc,utc=True,format='mixed',errors='coerce');sgc.to_csv(OUT/'catalogo_sgc_completo.csv',index=False)
    origins=[];picks=[];matches=[]
    for _,ev in ET.iterparse(OUT/'isc_norte.xml',events=['end']):
        if ev.tag!='{'+NS['b']+'}event':continue
        eid='ISC'+ev.attrib['publicID'].split('=')[-1];orig=ev.findall('b:origin',NS)
        orig=sorted(orig,key=lambda o:(value(o,'creationInfo/agencyID') not in ['RSNC','SGC'],o.attrib['publicID']!=value(ev,'preferredOriginID')))
        o=orig[0];t=pd.Timestamp(value(o,'time/value'));la=float(value(o,'latitude/value'));lo=float(value(o,'longitude/value'));z=float(value(o,'depth/value'))/1000
        origins.append(dict(event_id=eid,time_utc=str(t),lat=la,lon=lo,depth_km=z,agency=value(o,'creationInfo/agencyID')))
        dt=(sgc.time_utc-t).dt.total_seconds().abs();dist=np.hypot((sgc.lat-la)*111,(sgc.lon-lo)*110)
        hit=sgc[(dt<=12)&(dist<=30)&((sgc.depth_km-z).abs()<=40)].copy()
        if len(hit)!=1:ev.clear();continue
        c=hit.iloc[0];matches.append(dict(isc_event_id=eid,sgc_event_id=c.event_id,time_difference_s=float(dt.loc[c.name]),distance_km=float(dist.loc[c.name])))
        pk={p.attrib['publicID']:p for p in ev.findall('b:pick',NS)}
        # All exact local P/Pg/Pn and S/Sg/Sn arrivals; no reflected/teleseismic phases.
        seen=set()
        for origin in orig:
            for a in origin.findall('b:arrival',NS):
                phase=value(a,'phase');pid=value(a,'pickID');p=pk.get(pid)
                if p is None or pid in seen or phase not in ['P','Pg','Pn','S','Sg','Sn']:continue
                seen.add(pid);w=p.find('b:waveformID',NS)
                if w is None:continue
                picks.append(dict(event_id=c.event_id,origin_time=str(c.time_utc),ev_lat=c.lat,ev_lon=c.lon,ev_prof_km=c.depth_km,net=w.get('networkCode','IR'),sta=w.get('stationCode',''),fase=phase[0],pick_time=value(p,'time/value'),prob=1.,published_phase=phase,pick_id=pid,reporting_agency=value(p,'creationInfo/agencyID'),isc_event_id=eid))
        ev.clear()
    pd.DataFrame(origins).to_csv(OUT/'catalogo_isc.csv',index=False);pd.DataFrame(matches).to_csv(OUT/'event_crossmatch.csv',index=False);pd.DataFrame(picks).to_csv(OUT/'picks_publicados.csv',index=False)
    print(json.dumps(dict(isc_events=len(origins),unique_sgc_matches=len(matches),published_picks=len(picks))),flush=True)
if __name__=='__main__':main()
