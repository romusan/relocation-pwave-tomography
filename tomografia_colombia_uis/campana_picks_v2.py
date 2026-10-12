"""Reanudación transaccional, épocas activas y ventanas P/S con caché propia.

No sobrescribe picks/trazas de la campaña antigua. No interpreta la confianza
PhaseNet como sigma temporal. Ejecutar explícitamente; no es parte del piloto.
"""
from pathlib import Path
import argparse
import json
import sqlite3
import sys
import time
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'codigos'))
from regional.downloads import download
from regional.geometry import Projection
from regional.inputs import active_stations, load_profile, straight_time, sha256

SERVICE='https://sismo.sgc.gov.co:8443/fdsnws/dataselect/1/query'


def components(stream, network, station, max_gap=0.5):
    """Choose one complete Z/N/E sensor; do not invent 1/2 orientations."""
    stream=stream.select(network=network,station=station)
    groups={}
    for trace in stream:
        key=(trace.stats.location,trace.stats.channel[:2])
        groups.setdefault(key,stream.__class__()).append(trace.copy())
    for key in sorted(groups):
        group=groups[key]
        suffix={t.stats.channel[-1] for t in group}
        if not {'Z','N','E'}<=suffix:
            continue
        group=group.__class__([t for t in group if t.stats.channel[-1] in 'ZNE'])
        if any(float(g[6])>max_gap for g in group.get_gaps()):
            continue
        group.merge(method=1,fill_value='interpolate')
        if len(group)==3:
            return group
    raise ValueError('No complete ZNE sensor without gaps >0.5 s; rotation of 1/2 requires orientation metadata')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eventos',type=int,default=100)
    ap.add_argument('--estaciones',type=int,default=25)
    ap.add_argument('--umbral',type=float,default=.3)
    ap.add_argument('--reintentar-incompletos',action='store_true')
    ap.add_argument('--export-only',action='store_true',help='Export existing SQLite picks without network/model loading')
    ap.add_argument('--database',type=Path,default=ROOT/'datos/campana_v2.sqlite3')
    ap.add_argument('--output',type=Path,default=ROOT/'datos/picks_phasenet_v2.csv')
    a=ap.parse_args()
    if a.eventos<1 or a.estaciones<1 or not 0<a.umbral<1:
        ap.error('eventos/estaciones must be positive and umbral in (0,1)')
    database=a.database.resolve()
    database.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY,status TEXT,detail TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS picks(event_id TEXT,net TEXT,sta TEXT,fase TEXT,pick_time TEXT,payload TEXT,PRIMARY KEY(event_id,net,sta,fase,pick_time))')
        db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)')
        if a.export_only:
            export(db,a.output)
            return
        signature=json.dumps({'threshold':a.umbral,'station_limit':a.estaciones,'model':'PhaseNet.instance',
            'files':{str(path):sha256(path) for path in [ROOT/'dominio.json',ROOT/'datos/perfil_vp_vmm_transferencia.csv',
                ROOT/'datos/catalogo_seleccionado.csv',ROOT/'datos/estaciones_cm.csv',Path(__file__)]}},sort_keys=True)
        saved=db.execute("SELECT value FROM metadata WHERE key='campaign_signature'").fetchone()
        if saved and saved[0]!=signature:
            raise ValueError('Campaign settings/input/code changed. Use a new --database and --output to keep provenance separate.')
        with db:
            db.execute("INSERT OR IGNORE INTO metadata VALUES('campaign_signature',?)",(signature,))
        from obspy import UTCDateTime, read
        import seisbench.models as sbm
        model=sbm.PhaseNet.from_pretrained('instance')
        model.eval()
        dom=json.loads((ROOT/'dominio.json').read_text(encoding='utf-8'))
        proj=Projection(dom)
        profile=load_profile(ROOT/'datos/perfil_vp_vmm_transferencia.csv')
        cat=pd.read_csv(ROOT/'datos/catalogo_seleccionado.csv').set_index('event_id')
        order=pd.read_csv(ROOT/'datos/campana_orden.csv').event_id
        # Preserve the archived campaign order, with stale IDs removed and new
        # catalog IDs appended deterministically.
        ids=[e for e in order if e in cat.index]
        seen=set(ids)
        ids += sorted(set(cat.index)-seen)
        stations=pd.read_csv(ROOT/'datos/estaciones_cm.csv')
        done=dict(db.execute('SELECT event_id,status FROM events'))
        session=requests.Session()
        session.headers.update({'User-Agent':'tomografia-colombia-Q-v2 (academico)'})
        count=0
        for eid in ids:
            if eid in done and (done[eid]!='incomplete' or not a.reintentar_incompletos):
                continue
            if count>=a.eventos:
                break
            e=cat.loc[eid]
            origin=UTCDateTime(str(e.time_utc).replace(' ','T'))
            active=active_stations(stations,str(origin))
            ex,ey=proj.xy(e.lat,e.lon)
            sx,sy=proj.xy(active.lat,active.lon)
            active=active.assign(d_km=np.hypot(sx-ex,sy-ey),st_x=sx,st_y=sy)
            active=active[active.d_km<=250.].nsmallest(a.estaciones,'d_km')
            detail=[]; event_picks=[]; incomplete=False
            # Reruns with new settings get a separate download cache identity;
            # classify only the precise requested time window.
            for st in active.itertuples():
                p=float(straight_time(np.array([[float(ex),float(ey),e.depth_km]]),
                    np.array([[st.st_x,st.st_y,0.]]),profile)[0])
                t0,t1=origin+p-20.,origin+p*1.73+30.
                query={'network':st.net,'station':st.sta,'location':'*','channel':'*',
                    'starttime':t0.strftime('%Y-%m-%dT%H:%M:%S'),'endtime':t1.strftime('%Y-%m-%dT%H:%M:%S')}
                path,status=download(session,SERVICE,query,ROOT/'datos/trazas_campana_v2'/str(eid))
                entry={'net':st.net,'sta':st.sta,'download':status}
                if path is None:
                    incomplete |= status['status']!='no_data'
                    detail.append(entry)
                    continue
                try:
                    stream=components(read(str(path)),st.net,st.sta)
                    stream.trim(t0,t1)
                    result=model.classify(stream,P_threshold=a.umbral,S_threshold=a.umbral)
                    entry['classification']='ok'
                    for pick in result.picks:
                        phase=pick.phase.upper()[:1]
                        if phase not in ('P','S'):
                            continue
                        event_picks.append({'event_id':eid,'origin_time':str(origin),
                            'ev_lat':e.lat,'ev_lon':e.lon,'ev_prof_km':e.depth_km,
                            'net':st.net,'sta':st.sta,'st_lat':st.lat,'st_lon':st.lon,
                            'dist_epi_km':float(st.d_km),'fase':phase,'pick_time':str(pick.peak_time),
                            'prob':float(pick.peak_value),'t_viaje_s':float(pick.peak_time-origin)})
                except Exception as exc:
                    incomplete=True
                    entry['classification']='error'
                    entry['error']=f'{type(exc).__name__}: {exc}'
                detail.append(entry)
                time.sleep(.4)
            status='incomplete' if incomplete or active.empty else ('complete_with_picks' if event_picks else 'complete_without_picks')
            # Picks and event status are committed together; a crash before
            # commit can be retried without duplicate CSV appends.
            with db:
                for pick in event_picks:
                    db.execute('INSERT OR IGNORE INTO picks VALUES(?,?,?,?,?,?)',
                        (eid,pick['net'],pick['sta'],pick['fase'],pick['pick_time'],json.dumps(pick)))
                db.execute('INSERT OR REPLACE INTO events VALUES(?,?,?)',(eid,status,json.dumps(detail)))
            count+=1
            print(f'{count}/{a.eventos} {eid}: {status}, {len(event_picks)} candidates',flush=True)
            if count%25==0:
                export(db,a.output)
        export(db,a.output)


def export(db,dest):
    rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM picks ORDER BY event_id,net,sta,fase,pick_time')]
    columns=['event_id','origin_time','ev_lat','ev_lon','ev_prof_km','net','sta','st_lat','st_lon',
             'dist_epi_km','fase','pick_time','prob','t_viaje_s']
    dest=Path(dest).resolve()
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix('.tmp')
    pd.DataFrame(rows,columns=columns).to_csv(temp,index=False)
    temp.replace(dest)
    print(f'Exported {len(rows)} candidates to {dest}',flush=True)


if __name__=='__main__':
    main()
