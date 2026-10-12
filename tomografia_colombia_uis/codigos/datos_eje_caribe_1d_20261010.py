"""New regional arrival acquisition, preserving previous raw files."""
from pathlib import Path
import sys,json,datetime,concurrent.futures,xml.etree.ElementTree as ET
import requests
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from descarga_catalogo import params,CONSULT_URL,HEADERS_HTTP
from arma_catalogo import lee
OUT=ROOT/'datos/eje_caribe_1d_2026-10-10';OUT.mkdir(exist_ok=True)
NS={'b':'http://quakeml.org/xmlns/bed/1.2'}
def val(el,path,default=''):
    return el.findtext('/'.join('b:'+p for p in path.split('/')),default=default,namespaces=NS)
def fetch(url,path,q,post=False):
    if path.exists():return
    r=requests.post(url,data=q,headers=HEADERS_HTTP,timeout=(15,180)) if post else requests.get(url,params=q,timeout=(15,240))
    (path.with_suffix(path.suffix+'.query.json')).write_text(json.dumps(dict(url=r.url,params=q,status=r.status_code,bytes=len(r.content),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2),encoding='utf-8')
    r.raise_for_status()
    if 'Software error' in r.text[:4000]:raise ValueError('Server error')
    path.write_bytes(r.content);print('Downloaded',path.name,len(r.content),flush=True)
def main():
    def isc(group,lat0,lat1,lon0,lon1):
        q=dict(format='xml',starttime='2018-01-01',endtime='2024-09-30T23:59:59',minlatitude=lat0,maxlatitude=lat1,minlongitude=lon0,maxlongitude=lon1,minmagnitude=2.0,mindepth=0,maxdepth=180,includearrivals='true',includeallorigins='true')
        fetch('https://www.isc.ac.uk/fdsnws/event/1/query',OUT/f'isc_{group}.xml',q)
    def sgc():
        q=params(dict(lon_min=-79,lon_max=-71,lat_min=8,lat_max=13,prof_min=0,prof_max=180));q.update(inicial='01/01/2024',final='30/09/2024')
        fetch(CONSULT_URL,OUT/'sgc_norte_2024.html',q,True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(isc,'eje',4,6.6,-77,-74.5),pool.submit(isc,'caribe',8,13,-79,-71),pool.submit(sgc)]
        for t in concurrent.futures.as_completed(tasks):t.result()
    oldcat=ROOT/'datos/caribe_1d_estaciones_completas_2026-10-10/catalogo_sgc_completo.csv'
    cat=pd.concat([pd.read_csv(oldcat),pd.DataFrame(lee((OUT/'sgc_norte_2024.html').read_text(encoding='utf-8',errors='replace'),'norte_2024'))],ignore_index=True).drop_duplicates('event_id')
    cat['time_utc']=pd.to_datetime(cat.time_utc,utc=True,format='mixed',errors='coerce');cat.to_csv(OUT/'catalogo_sgc.csv',index=False)
    picks=[];matches=[];counters={}
    for group in ['eje','caribe']:
        c=cat[cat.lat.between(4,6.6)&cat.lon.between(-77,-74.5)] if group=='eje' else cat[cat.lat.between(8,13)]
        c=c[c.time_utc.between(pd.Timestamp('2018-01-01',tz='UTC'),pd.Timestamp('2024-10-01',tz='UTC'))].copy();times=np.array([t.timestamp() for t in c.time_utc],dtype=float)
        total=0;matched=0
        for _,ev in ET.iterparse(OUT/f'isc_{group}.xml',events=['end']):
            if ev.tag!='{'+NS['b']+'}event':continue
            total+=1;orig=ev.findall('b:origin',NS)
            orig.sort(key=lambda o:(val(o,'creationInfo/agencyID') not in ['RSNC','SGC'],o.attrib['publicID']!=val(ev,'preferredOriginID')))
            if not orig:ev.clear();continue
            o=orig[0];t=pd.Timestamp(val(o,'time/value'));la=float(val(o,'latitude/value'));lo=float(val(o,'longitude/value'));dep=float(val(o,'depth/value'))/1000
            dt=np.abs(times-t.timestamp());ix=np.where(dt<=12)[0]
            if not len(ix):ev.clear();continue
            dist=np.hypot((c.iloc[ix].lat.to_numpy()-la)*111,(c.iloc[ix].lon.to_numpy()-lo)*110)
            ix=ix[(dist<=30)&(np.abs(c.iloc[ix].depth_km.to_numpy()-dep)<=40)]
            if len(ix)!=1:ev.clear();continue
            row=c.iloc[ix[0]];matched+=1;eid=ev.attrib['publicID'];pk={p.attrib['publicID']:p for p in ev.findall('b:pick',NS)};seen=set()
            matches.append(dict(group=group,event_id=row.event_id,isc_event_id=eid,time_difference_s=float(dt[ix[0]]),horizontal_difference_km=float(np.hypot((row.lat-la)*111,(row.lon-lo)*110)),depth_difference_km=abs(row.depth_km-dep),origin_agency=val(o,'creationInfo/agencyID')))
            for origin in orig:
                for a in origin.findall('b:arrival',NS):
                    phase=val(a,'phase');pid=val(a,'pickID');p=pk.get(pid)
                    if p is None or pid in seen or phase not in ['P','Pg','Pn','S','Sg','Sn']:continue
                    seen.add(pid);w=p.find('b:waveformID',NS)
                    if w is None:continue
                    picks.append(dict(event_id=row.event_id,origin_time=str(row.time_utc),ev_lat=row.lat,ev_lon=row.lon,ev_prof_km=row.depth_km,net=w.attrib.get('networkCode',''),sta=w.attrib.get('stationCode',''),fase=phase[0],pick_time=val(p,'time/value'),prob=1.0,published_phase=phase,pick_id=pid,origin_agency=val(origin,'creationInfo/agencyID'),isc_event_id=eid,reporting_agency=val(p,'creationInfo/agencyID'),campaign_group=group))
            ev.clear()
        counters[group]=dict(ISC_events=total,SGC_unique_crossmatches=matched)
        print(group,counters[group],flush=True)
    pd.DataFrame(matches).to_csv(OUT/'crossmatch.csv',index=False);pd.DataFrame(picks).drop_duplicates(['event_id','pick_id']).to_csv(OUT/'picks_publicados.csv',index=False)
    (OUT/'adquisicion.json').write_text(json.dumps(counters,indent=2),encoding='utf-8')
if __name__=='__main__':main()
