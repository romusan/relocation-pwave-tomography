"""Geographic 3D export: longitude, latitude, depth; physical approximate aspect."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from pyproj import Geod
from agregar_mapa_superficie_norte import boundaries, lines, plotly_arguments, Projection, ROOT


def decode(values):
    if isinstance(values,dict) and 'bdata' in values:
        return np.frombuffer(base64.b64decode(values['bdata']),dtype=values['dtype'])
    return np.asarray(values,float)


def geo_trace(geo,name,color,width):
    x,y,z=[],[],[]
    for coords in lines(geo):
        x.extend(coords[:,0].tolist()+[None]); y.extend(coords[:,1].tolist()+[None])
        z.extend([0.]*len(coords)+[None])
    return dict(type='scatter3d',mode='lines',name=name,x=x,y=y,z=z,
        line=dict(color=color,width=width),hovertemplate=name+'<br>Superficie: 0 km<extra></extra>')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--viewer',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists(): raise ValueError('Choose a new output directory')
    domain=json.loads((a.run/'summary.json').read_text())['domain']
    projection=Projection(domain)
    source=a.viewer/'benioff_3d.html'
    html=source.read_text(encoding='utf-8').replace('Distancias en km, sin exageracion vertical.','Coordenadas WGS84; profundidad en km, sin exageracion vertical.')
    args=plotly_arguments(html); data=args[1][2]; layout=args[2][2]
    original_count=len(data)
    data[:]=[trace for trace in data if trace.get('name')!='Eje de seccion (referencia)']
    max_roundtrip=0.
    for trace in data:
        x,y=decode(trace['x']),decode(trace['y'])
        lat,lon=projection.latlon(x,y)
        rx,ry=projection.xy(lat,lon)
        max_roundtrip=max(max_roundtrip,float(np.max(np.hypot(rx-x,ry-y))))
        trace['x']=lon.tolist(); trace['y']=lat.tolist()
    boundary_path,metadata_path,features=boundaries()
    region=box(domain['lon_min'],domain['lat_min'],domain['lon_max'],domain['lat_max'])
    country=unary_union([g for _,g in features.values()])
    parts=list(country.geoms) if hasattr(country,'geoms') else [country]
    mainland=max(parts,key=lambda g:g.area); shell=Polygon(mainland.exterior)
    outline=unary_union([g.exterior for g in parts if g is mainland or not shell.covers(g.representative_point())])
    maps=[(unary_union([g.boundary for _,g in features.values()]).intersection(region),'Fronteras departamentales','#89939b',1.5),
          (outline.intersection(region),'Fronteras de Colombia','#182b35',5)]
    for geo,name,color,width in maps: data.append(geo_trace(geo,name,color,width))
    cities=[('Cali',3.44,-76.52),('Bogotá',4.71,-74.07),('Quibdó',5.69,-76.66),('Bucaramanga',6.81,-73.1),('Cartagena',10.40,-75.50),('Santa Marta',11.24,-74.21)]
    data.append(dict(type='scatter3d',mode='markers+text',name='Ciudades',
        x=[c[2] for c in cities],y=[c[1] for c in cities],z=[0]*len(cities),
        text=[c[0] for c in cities],textposition='top center',marker=dict(size=4,color='#182b35')))
    e=pd.read_csv(a.run/'relocated_train_events.csv')
    data[0]['name']=f'Catálogo SGC ({len(data[0]["x"]):,})'
    data[1]['name']=f'Sismos relocalizados ({len(e):,})'
    data[1]['hovertemplate']='%{text}<br>Lat %{y:.3f}, lon %{x:.3f}<br>Prof. %{z:.1f} km<extra>Relocalización experimental</extra>'
    geod=Geod(ellps='WGS84'); lat0=(domain['lat_min']+domain['lat_max'])/2; lon0=(domain['lon_min']+domain['lon_max'])/2
    width=geod.inv(domain['lon_min'],lat0,domain['lon_max'],lat0)[2]/1000
    height=geod.inv(lon0,domain['lat_min'],lon0,domain['lat_max'])[2]/1000
    aspect=[width/height,1.,domain['prof_max']/height]
    layout.update(title=dict(text='Colombia: catálogo SGC, relocalizaciones y Vp experimental'),height=820)
    layout['scene'].update(xaxis=dict(title='Longitud (°)',range=[domain['lon_min'],domain['lon_max']],dtick=1),
        yaxis=dict(title='Latitud (° N)',range=[domain['lat_min'],domain['lat_max']],dtick=1),
        zaxis=dict(title='Profundidad (km)',range=[domain['prof_max'],0]),
        aspectmode='manual',aspectratio=dict(zip(['x','y','z'],aspect)))
    for start,end,value in reversed(args[1:]): html=html[:start]+json.dumps(value,ensure_ascii=False)+html[end:]
    note='<p style="font-family:Arial">Longitud y latitud WGS84; profundidad positiva hacia abajo. Aspecto físico aproximado sin exageración vertical. Las capas se pueden ocultar en la leyenda. Fragmentos provisionales; no identifican placas por sí solos.</p>'
    html=html.replace('<body>','<body>'+note)
    a.output.mkdir(parents=True); (a.output/'benioff_3d.html').write_text(html,encoding='utf-8')
    c=pd.read_csv(a.viewer/'vp_celdas_activas.csv')
    catalog_path=a.run/'catalogo_visualizacion_ampliado.csv'
    if not catalog_path.exists(): catalog_path=ROOT/'datos/catalogo_seleccionado.csv'
    catalog=pd.read_csv(catalog_path).sample(15000,random_state=20261009)
    with np.load(a.viewer/'fragmentos_geometricos.npz') as mesh:
        xyz=mesh['xyz'].copy(); faces=mesh['triangles'].copy()
    lat,lon=projection.latlon(xyz[:,0],xyz[:,1]); xyz[:,0]=lon; xyz[:,1]=lat
    fig=plt.figure(figsize=(14,10)); ax=fig.add_axes([.02,.08,.83,.83],projection='3d')
    ax.scatter(catalog.lon,catalog.lat,catalog.depth_km,s=1,c=catalog.depth_km,cmap='viridis',vmin=0,vmax=180,alpha=.12,label='Catálogo SGC (15.000 puntos de contexto)')
    ax.add_collection3d(Poly3DCollection(xyz[faces],facecolor='gold',edgecolor='none',alpha=.18))
    ax.scatter(e.lon,e.lat,e.depth_km,s=5,c='black',alpha=.5,label=f'Sismos relocalizados ({len(e):,})')
    cb=ax.scatter(c.lon,c.lat,c.depth_km,s=16,c=c.dvp_percent,cmap='RdBu_r',vmin=-15,vmax=15,marker='s',alpha=.8)
    for geo,name,color,widthline in maps:
        segments=[np.column_stack([q[:,0],q[:,1],np.zeros(len(q))]) for q in lines(geo)]
        ax.add_collection3d(Line3DCollection(segments,colors=color,linewidths=widthline*.4,label=name))
    for name,lat,lon in cities: ax.text(lon,lat,0,name,fontsize=8,color='#182b35')
    ax.set(xlabel='Longitud (°)',ylabel='Latitud (° N)',zlabel='Profundidad (km)',
        xlim=(domain['lon_min'],domain['lon_max']),ylim=(domain['lat_min'],domain['lat_max']),zlim=(domain['prof_max'],0))
    ax.set_box_aspect(aspect); ax.view_init(elev=28,azim=-130); ax.legend(loc='upper left',fontsize=8)
    fig.suptitle(f'Colombia: sismicidad y Vp en longitud, latitud y profundidad\n{len(e)} relocalizaciones · {len(c)} celdas activas · {len(faces)} fragmentos triangulares',fontsize=14)
    cax=fig.add_axes([.91,.19,.017,.4]); fig.colorbar(cb,cax=cax,label='dVp/Vp₀ (%)')
    inset=fig.add_axes([.82,.70,.16,.23])
    for q in lines(outline): inset.plot(q[:,0],q[:,1],color='#324d5c',lw=.5)
    lo,hi,la,lb=domain['lon_min'],domain['lon_max'],domain['lat_min'],domain['lat_max']
    inset.plot([lo,hi,hi,lo,lo],[la,la,lb,lb,la],color='red',lw=1)
    inset.set(xlim=(-82,-66),ylim=(-5,14),title='Región estudiada'); inset.set_aspect('equal'); inset.tick_params(labelsize=7)
    fig.text(.04,.035,'Cartografía DANE 2025 · WGS84 · aspecto físico aproximado · geometría provisional; sin asignación de placa.',fontsize=9)
    fig.savefig(a.output/'vista_3d_geografica.png',dpi=180); fig.savefig(a.output/'vista_3d_geografica.pdf'); plt.close(fig)
    inputs=[source,a.run/'relocated_train_events.csv',a.run/'summary.json',
        a.viewer/'vp_celdas_activas.csv',a.viewer/'fragmentos_geometricos.npz',boundary_path,metadata_path,
        catalog_path,Path(__file__),ROOT/'codigos/agregar_mapa_superficie_norte.py',ROOT/'codigos/regional/geometry.py']
    manifest=dict(coordinates='WGS84 longitude, latitude; depth km positive down',surface_depth_km=0,
        original_trace_count=original_count,removed_reference_section_axis=True,
        boundary_layers=[m[1] for m in maps],original_depths_and_topology_unchanged=True,
        projection_roundtrip_max_km=max_roundtrip,physical_aspect_ratio=aspect,
        relocated_events=len(e),active_vp_cells=len(c),triangles=len(faces),
        input_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (a.output/'manifest_geografico.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='input_sha256'},indent=2))


if __name__=='__main__': main()
