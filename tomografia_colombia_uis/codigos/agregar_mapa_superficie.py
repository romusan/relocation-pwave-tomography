"""Add DANE surface geography to an existing viewer without changing tomography.

Requires shapely (tested with 2.2.0), plus the project's existing dependencies.
Run from the project root with --viewer, --run and --output directories.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from shapely.geometry import shape, box, Point, Polygon
from shapely.ops import unary_union
from shapely import make_valid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codigos'))
from regional.geometry import Projection

SOURCE = 'https://geoportal.dane.gov.co/mparcgis/rest/services/Territoriales_DANE/Serv_Territoriales_DANE_2025/MapServer/1'


def boundaries():
    path = ROOT / 'datos/cartografia/departamentos_dane_2025.geojson'
    metadata = path.with_suffix('.metadata.json')
    if not path.exists():
        params = dict(where='1=1', outFields='dpto_ccdgo,dpto_cnmbr',
                      outSR=4326, geometryPrecision=6, maxAllowableOffset=.002,
                      returnGeometry='true', f='geojson')
        response = requests.get(SOURCE + '/query', params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
        if data.get('type') != 'FeatureCollection' or len(data['features']) != 33:
            raise ValueError(f'Incomplete DANE response: {str(data)[:300]}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        metadata.write_text(json.dumps(dict(source=SOURCE, query_url=response.url,
            retrieved_utc=datetime.now(timezone.utc).isoformat(),
            generalization_degrees=.002, coordinates='WGS84 longitude/latitude',
            feature_count=33), indent=2), encoding='utf-8')
    data = json.loads(path.read_text(encoding='utf-8'))
    features = {str(f['properties']['dpto_ccdgo']).zfill(2):
                (f['properties']['dpto_cnmbr'], make_valid(shape(f['geometry']))) for f in data['features']}
    if len(features) != 33 or not features['76'][1].covers(Point(-76.52, 3.44)):
        raise ValueError('Boundary validation failed: missing departments or Cali outside Valle')
    return path, metadata, features


def lines(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type in ('LineString', 'LinearRing'):
        yield np.asarray(geometry.coords)
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from lines(part)


def projected(geometry, projection):
    for points in lines(geometry):
        x, y = projection.xy(points[:, 1], points[:, 0])
        yield np.asarray(x), np.asarray(y)


def trace(geometry, projection, name, color, width, visible=True):
    x, y, z = [], [], []
    for xx, yy in projected(geometry, projection):
        x.extend(xx.tolist() + [None])
        y.extend(yy.tolist() + [None])
        z.extend([0.] * len(xx) + [None])
    return dict(type='scatter3d', mode='lines', name=name, x=x, y=y, z=z,
                line=dict(color=color, width=width), visible=visible,
                hovertemplate=name + '<br>Superficie: 0 km<extra></extra>')


def plotly_arguments(html):
    # The final call is the exported figure, following the embedded Plotly library.
    pos = html.rfind('Plotly.newPlot(') + len('Plotly.newPlot(')
    if pos < len('Plotly.newPlot('):
        raise ValueError('No exported Plotly figure found')
    decoder = json.JSONDecoder()
    result = []
    for _ in range(3):
        while html[pos].isspace() or html[pos] == ',':
            pos += 1
        start = pos
        value, pos = decoder.raw_decode(html, pos)
        result.append((start, pos, value))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--viewer', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise ValueError('Use a new output directory to preserve existing artifacts')
    domain = json.loads((ROOT / 'dominio.json').read_text())
    projection = Projection(domain)
    boundary_path, metadata_path, features = boundaries()
    region = box(domain['lon_min'], domain['lat_min'], domain['lon_max'], domain['lat_max'])
    country = unary_union([g for _, g in features.values()])
    parts = list(country.geoms) if hasattr(country, 'geoms') else [country]
    mainland = max(parts, key=lambda g: g.area)
    shell = Polygon(mainland.exterior)
    # National outlines omit interior rings (lakes and small gaps introduced by
    # independent department generalization). Preserve mainland and offshore islands.
    outline = unary_union([g.exterior for g in parts if g is mainland or
                           not shell.covers(g.representative_point())])
    # Clip original boundary lines, not clipped polygons: study edges are not national borders.
    coast = outline.intersection(region)
    departments = unary_union([g.boundary for _, g in features.values()]).intersection(region)
    cauca = features['19'][1].boundary.intersection(region)
    valle = features['76'][1].boundary.intersection(region)
    styles = [(departments, 'Departamentos (superficie)', '#89939b', 1.5),
              (coast, 'Colombia: costa y frontera', '#182b35', 5),
              (cauca, 'Cauca (superficie)', '#bb5c12', 4),
              (valle, 'Valle del Cauca (superficie)', '#9c27b0', 4)]
    places = [('Cali', 3.44, -76.52), ('Bogotá', 4.71, -74.07),
              ('Quibdó', 5.69, -76.66), ('Bucaramanga (referencia)', 6.81, -73.10)]
    xx, yy = projection.xy([q[1] for q in places], [q[2] for q in places])
    html_path = a.viewer / 'benioff_3d.html'
    html = html_path.read_text(encoding='utf-8')
    args = plotly_arguments(html)
    original_data = args[1][2]
    original_trace_count = len(original_data)
    for geo, name, color, width in styles:
        original_data.append(trace(geo, projection, name, color, width))
    original_data.append(dict(type='scatter3d', mode='markers+text',
        name='Ciudades (superficie)', x=xx.tolist(), y=yy.tolist(), z=[0]*len(places),
        text=[q[0] for q in places], textposition='top center',
        textfont=dict(size=12, color='#182b35'), marker=dict(size=4, color='#182b35'),
        hovertemplate='%{text}<br>Superficie: 0 km<extra></extra>'))
    # National overview is optional so the default retains the regional scale.
    original_data.append(trace(outline, projection,
        'Colombia completa (contexto)', '#324d5c', 3, visible='legendonly'))
    layout = args[2][2]
    layout.update(title=dict(text='Colombia en superficie · sismicidad y Vp experimental'),
                  height=800, margin=dict(l=35, r=65, t=165, b=25))
    layout['scene']['zaxis']['range'] = [180, 0]
    # All scientific arrays and previous layers remain exactly as exported.
    for start, end, value in reversed(args[1:]):
        html = html[:start] + json.dumps(value, ensure_ascii=False) + html[end:]
    note = ('<p style="font-family:Arial;margin:12px">Mapa DANE 2025 a profundidad 0 km; '
            'Cauca en naranja y Valle del Cauca en violeta. Pulse la leyenda para mostrar u '
            'ocultar capas. «Colombia completa» amplía el contexto; el modelo sigue siendo '
            'regional (2–9° N, 79–73° O). <a href="' + SOURCE + '">Fuente cartográfica: DANE</a>.</p>')
    html = html.replace('<body>', '<body>' + note)
    a.output.mkdir(parents=True)
    (a.output / 'benioff_3d.html').write_text(html, encoding='utf-8')

    catalog_path = ROOT / 'datos/catalogo_seleccionado.csv'
    catalog = pd.read_csv(catalog_path)
    d = catalog.sample(min(len(catalog), 15000), random_state=20261009)
    x, y = projection.xy(d.lat, d.lon)
    e = pd.read_csv(a.run / 'relocated_train_events.csv')
    c = pd.read_csv(a.viewer / 'vp_celdas_activas.csv')
    with np.load(a.viewer / 'fragmentos_geometricos.npz') as mesh:
        triangles = mesh['xyz'][mesh['triangles']]
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_axes([.02, .08, .83, .83], projection='3d')
    ax.scatter(x, y, d.depth_km, s=1, c=d.depth_km, cmap='viridis',
               vmin=0, vmax=180, alpha=.12)
    ax.add_collection3d(Poly3DCollection(triangles, facecolor='gold', edgecolor='none', alpha=.18))
    ax.scatter(e.x_km, e.y_km, e.depth_km, s=5, c='black', alpha=.55)
    color = ax.scatter(c.x_km, c.y_km, c.depth_km, s=16, c=c.dvp_percent,
        cmap='RdBu_r', vmin=-15, vmax=15, marker='s', alpha=.85)
    for geo, name, col, width in styles:
        segments = [np.column_stack([px, py, np.zeros(len(px))])
                    for px, py in projected(geo, projection)]
        ax.add_collection3d(Line3DCollection(segments, colors=col,
                            linewidths=width*.4, label=name))
    ax.scatter(xx, yy, [0]*len(xx), c='#182b35', s=12)
    for px, py, (name, _, _) in zip(xx, yy, places):
        ax.text(px, py, 0, name.replace(' (referencia)', ''), fontsize=8, color='#182b35')
    oceanx, oceany = projection.xy(5.5, -78.6)
    ax.text(oceanx, oceany, 0, 'Océano Pacífico', color='#1d728c', fontsize=9)
    ax.set(xlabel='Este AEQD (km)', ylabel='Norte AEQD (km)', zlabel='Profundidad (km)',
           zlim=(180, 0))
    ax.set_box_aspect([np.ptp(x), np.ptp(y), 180])
    ax.view_init(elev=28, azim=-130)
    ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Colombia en superficie: sismos, fragmentos geométricos y Vp\n'
                 '960 relocalizaciones · 520 celdas activas · profundidad 0–180 km', fontsize=14)
    barax = fig.add_axes([.90, .14, .018, .30])
    fig.colorbar(color, cax=barax, label='dVp/Vp₀ (%)')
    inset = fig.add_axes([.81, .60, .17, .28])
    for points in lines(outline):
        inset.plot(points[:, 0], points[:, 1], color='#324d5c', lw=.55)
    inset.add_patch(Rectangle((domain['lon_min'], domain['lat_min']),
        domain['lon_max']-domain['lon_min'], domain['lat_max']-domain['lat_min'],
        fill=False, edgecolor='#c43a31', lw=1.5))
    inset.set(xlim=(-82, -66), ylim=(-5, 14), title='Región estudiada', xlabel='Longitud', ylabel='Latitud')
    inset.set_aspect('equal')
    inset.tick_params(labelsize=7)
    fig.text(.04, .035, 'Cartografía: DANE 2025 · límites sobre z = 0 km · '
             'Vp y geometría originales; interpretación experimental sin asignación de placa.', fontsize=9)
    fig.savefig(a.output / 'vista_3d_mapa.png', dpi=180)
    fig.savefig(a.output / 'vista_3d_mapa.pdf')
    plt.close(fig)
    inputs = [html_path, a.run/'relocated_train_events.csv', a.viewer/'vp_celdas_activas.csv',
        a.viewer/'fragmentos_geometricos.npz', catalog_path, ROOT/'dominio.json',
        ROOT/'codigos/regional/geometry.py', boundary_path, metadata_path, Path(__file__)]
    manifest = dict(source=SOURCE, surface_depth_km=0, geographic_clip=domain,
        original_trace_count=original_trace_count, added_trace_count=6,
        relocated_events=len(e), active_vp_cells=len(c), mesh_triangles=len(triangles),
        science_arrays_unchanged=True,
        input_sha256={str(q.resolve()): hashlib.sha256(q.read_bytes()).hexdigest() for q in inputs})
    (a.output/'mapa_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(a.output.resolve()), departments=len(features),
                         relocated_events=len(e), active_vp_cells=len(c), triangles=len(triangles))))


if __name__ == '__main__':
    main()
