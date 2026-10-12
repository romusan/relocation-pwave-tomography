r"""Campana completa: bajar trazas, picar con PhaseNet, escribir los picks.

Corre durante dias. Se lanza DESACOPLADA de la consola (ver lanzar.ps1) o
muere al cerrar la sesion.

ORDEN DE LOS EVENTOS: no cronologico ni aleatorio, sino POR TURNOS entre
celdas del dominio. Asi cualquier prefijo de la campana --- se pare cuando
se pare --- cubre el dominio de forma pareja. Si se ordenara por fecha, una
interrupcion dejaria 2015-2018 completo y el resto vacio; si fuera al azar,
la cobertura seguiria la densidad de sismicidad y el nido se lo comeria
todo otra vez.

REANUDABLE: los eventos ya procesados se saltan, las trazas quedan en
cache. Volver a lanzarlo continua donde iba.

VERSION HISTORICA: usar campana_picks_v2.py para nuevas descargas.
El archivo de referencia contiene 56 081 picks de analista, pero la
validacion efectiva tiene 228 coincidencias P y 142 S, seleccionadas por
cercania al analista. No demuestra rendimiento regional ni calibra sigma.
La confianza del detector se conserva, sin convertirla en peso temporal.

Uso:  py campana_picks.py [--eventos 12000] [--estaciones 25]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
from obspy import UTCDateTime, read                            # noqa: E402

AQUI = Path(__file__).resolve().parent
TRAZAS = AQUI / "datos" / "trazas_campana"
PICKS = AQUI / "datos" / "picks_phasenet.csv"
HECHOS = AQUI / "datos" / "campana_eventos_hechos.csv"
ORDEN = AQUI / "datos" / "campana_orden.csv"
DAT = "https://sismo.sgc.gov.co:8443/fdsnws/dataselect/1/query"

VP_MEDIO, GRADO = 6.2, 111.32
ANTES, DESPUES = 15.0, 60.0
RADIO_MAX_KM = 250.0
CELDA_ORDEN_KM = 25.0       # celda del reparto por turnos
PAUSA = 0.4                 # s entre peticiones al SGC


def xy(lat, lon, lat0):
    return ((lon + 79.0) * GRADO * np.cos(np.radians(lat0)),
            (lat - 2.0) * GRADO)


def orden_por_turnos(cat, lat0):
    """Eventos ordenados tomando uno de cada celda por turnos."""
    if ORDEN.exists():
        o = pd.read_csv(ORDEN)
        return cat.set_index("event_id").loc[o.event_id].reset_index()
    x, y = xy(cat.lat.to_numpy(), cat.lon.to_numpy(), lat0)
    z = cat.depth_km.to_numpy()
    celda = list(zip((x // CELDA_ORDEN_KM).astype(int),
                     (y // CELDA_ORDEN_KM).astype(int),
                     (z // CELDA_ORDEN_KM).astype(int)))
    c = cat.assign(_c=celda)
    # dentro de cada celda, primero los mejor localizados
    c["_r"] = (c.gap_deg.rank(pct=True) + c.rms_s.rank(pct=True)
               - c.phases.rank(pct=True))
    colas = {k: list(g.sort_values("_r").index) for k, g in c.groupby("_c")}
    orden = []
    while colas:
        for k in list(colas):
            if not colas[k]:
                del colas[k]
                continue
            orden.append(colas[k].pop(0))
    out = c.loc[orden].drop(columns=["_c", "_r"]).reset_index(drop=True)
    out[["event_id"]].to_csv(ORDEN, index=False)
    return out


def baja(s, net, sta, t0, t1, destino):
    if destino.exists():
        return destino.stat().st_size > 2000
    q = {"network": net, "station": sta, "location": "*", "channel": "*",
         "starttime": t0.strftime("%Y-%m-%dT%H:%M:%S"),
         "endtime": t1.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        r = s.get(DAT, params=q, timeout=(10, 120), verify=False)
    except Exception:
        return False
    time.sleep(PAUSA)
    if r.status_code != 200 or len(r.content) < 2000:
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(b"")        # marca de "probado y vacio"
        return False
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(r.content)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eventos", type=int, default=12000)
    ap.add_argument("--estaciones", type=int, default=25)
    ap.add_argument("--umbral", type=float, default=0.3)
    a = ap.parse_args()

    import seisbench.models as sbm
    print("[%s] cargando PhaseNet" % time.strftime("%H:%M:%S"))
    modelo = sbm.PhaseNet.from_pretrained("instance")
    modelo.eval()

    dom = json.loads((AQUI / "dominio.json").read_text(encoding="utf-8"))
    lat0 = 0.5 * (dom["lat_min"] + dom["lat_max"])
    cat = pd.read_csv(AQUI / "datos" / "catalogo_seleccionado.csv")
    est = pd.read_csv(AQUI / "datos" / "estaciones_cm.csv")
    cat = orden_por_turnos(cat, lat0)
    print("[%s] %s eventos ordenados por turnos espaciales"
          % (time.strftime("%H:%M:%S"), "{:,}".format(len(cat))))

    hechos = set()
    if HECHOS.exists():
        hechos = set(pd.read_csv(HECHOS).event_id)
        print("[%s] %s eventos ya procesados, se continua"
              % (time.strftime("%H:%M:%S"), "{:,}".format(len(hechos))))

    s = requests.Session()
    s.headers.update({"User-Agent": "tomografia-colombia-Q (academico)"})
    buf_picks, buf_hechos = [], []
    n_ev = n_picks = 0
    t_ini = time.time()

    for _, e in cat.iterrows():
        if n_ev >= a.eventos:
            break
        if e.event_id in hechos:
            continue
        org = UTCDateTime(str(e.time_utc).replace(" ", "T"))
        d = np.hypot((est.lat - e.lat) * GRADO,
                     (est.lon - e.lon) * GRADO * np.cos(np.radians(e.lat)))
        cerca = est.assign(d_km=d)
        cerca = cerca[cerca.d_km <= RADIO_MAX_KM].nsmallest(a.estaciones, "d_km")
        n_ev += 1
        for _, st_ in cerca.iterrows():
            dist = float(np.sqrt(st_.d_km ** 2 + float(e.depth_km) ** 2))
            t_p = org + dist / VP_MEDIO
            f = TRAZAS / str(e.event_id) / ("%s.mseed" % st_.sta)
            if not baja(s, st_.net, st_.sta, t_p - ANTES, t_p + DESPUES, f):
                continue
            try:
                stream = read(f)
                stream.merge(method=1, fill_value="interpolate")
                if len(stream) < 3:
                    continue
                res = modelo.classify(stream, P_threshold=a.umbral,
                                      S_threshold=a.umbral)
            except Exception:
                continue
            for p in res.picks:
                fase = p.phase.upper()[:1]
                if fase not in ("P", "S"):
                    continue
                buf_picks.append({
                    "event_id": e.event_id, "origin_time": str(org),
                    "ev_lat": e.lat, "ev_lon": e.lon, "ev_prof_km": e.depth_km,
                    "net": st_.net, "sta": st_.sta,
                    "st_lat": st_.lat, "st_lon": st_.lon,
                    "dist_epi_km": round(float(st_.d_km), 2),
                    "fase": fase, "pick_time": str(p.peak_time),
                    "prob": round(float(p.peak_value), 4),
                    "t_viaje_s": round(float(p.peak_time - org), 4)})
                n_picks += 1
        buf_hechos.append({"event_id": e.event_id,
                           "cuando": time.strftime("%Y-%m-%d %H:%M:%S")})
        if len(buf_hechos) >= 25:
            vuelca(buf_picks, buf_hechos)
            buf_picks, buf_hechos = [], []
            h = (time.time() - t_ini) / 3600
            print("[%s] %6d eventos  %8s picks  %5.2f h  %.1f ev/h"
                  % (time.strftime("%H:%M:%S"), n_ev, "{:,}".format(n_picks),
                     h, n_ev / max(h, 1e-9)))
    vuelca(buf_picks, buf_hechos)
    print("[%s] FIN: %d eventos, %s picks en %.2f h"
          % (time.strftime("%H:%M:%S"), n_ev, "{:,}".format(n_picks),
             (time.time() - t_ini) / 3600))
    return 0


def vuelca(picks, hechos):
    if picks:
        d = pd.DataFrame(picks)
        d.to_csv(PICKS, mode="a", header=not PICKS.exists(), index=False)
    if hechos:
        d = pd.DataFrame(hechos)
        d.to_csv(HECHOS, mode="a", header=not HECHOS.exists(), index=False)


if __name__ == "__main__":
    raise SystemExit(main())
