from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'articulo_boletin_geologia_uis_adjunto'
src=OUT/'crear_manuscrito_revision_20261011.py'
t=src.read_text(encoding='utf-8')
def change(a,b):
    global t
    assert a in t,a
    t=t.replace(a,b)
change("p(title,'Title',center=True)","q=p(title,'Title',center=True)\nfor r in q.runs:r.font.hidden=False;r.font.color.rgb=RGBColor(0,0,0)\nD.styles['Title'].font.hidden=False")
change('La candidata Nazca se ajustó con 570 eventos', 'De los 1180 eventos elegibles por profundidad y brecha azimutal, 795 quedaron al lado Nazca de la diagonal y 385 al lado Caribe. Nazca retuvo 570 después del filtro de compatibilidad con la profundidad media Slab2; los 225 restantes no alimentaron controles de superficie. Caribe utilizó los 385 eventos elegibles de su lado: la envolvente lateral de 75 km delimita la representación y sus anclajes, pero no es un filtro de descarte de esos eventos.\n\nLa candidata Nazca se ajustó con 570 eventos')
change('Las semillas fueron 20261011 a 20261040.', 'Las mismas 30 semillas, 20261011 a 20261040, se reutilizaron para cada valor de k y para cada métrica: 150 ajustes por métrica y 300 en total. La exageración vertical de 3× utilizada en las figuras no interviene en el ajuste, que emplea distancias en km reales o coordenadas estandarizadas según el caso.')
change('el bisector estimado a 80 km no coincide con la diagonal propuesta', 'los bisectores estimados a 60, 80 y 100 km no coinciden con la diagonal propuesta')
change("figure('Figura_7_clasificacion_kmeans.png'", "figure('Figura_7_clasificacion_kmeans_revision_b.png'")
change('la línea negra es el bisector de los centroides 3D evaluado a 80 km de profundidad.', 'las líneas negras son los bisectores de los centroides 3D evaluados a 60, 80 y 100 km de profundidad, con estilos diferenciados. Los niveles abarcan el entorno del corte de 80 km presentado en la Figura 6 y permiten visualizar la dependencia con la profundidad.')
start=t.index("blocks('''La reserva también muestra diferencias")
end=t.index("blocks('''Como comprobación limitada",start)
reserve=t[start:end]
t=t[:start]+t[end:]
target="heading('5 Discusión de resultados')"
assert target in t
t=t.replace(target,"heading('4.6 Evaluación de reserva por profundidad',True)\n"+reserve+'\n'+target)
change('la incertidumbre uniforme de 1 s', 'el divisor uniforme de residuos de 1 s')
change('Las derivadas ya verificadas permiten realizar esos contrastes', 'En 2857 de las 10011 celdas activas (28,54 %), casi tres de cada diez, la solución alcanza una de las dos cotas: 1300 llegan al límite acumulado de descenso por ronda y 1557 al máximo de 8,8 km/s. Sus amplitudes quedan condicionadas por las cotas y no pueden interpretarse como estimaciones libres del contraste de velocidad; los datos no permiten cuantificar cuánto continuarían fuera de ellas. Esto limita directamente la lectura de las anomalías de la Figura 5. Las derivadas ya verificadas permiten realizar esos contrastes')
change('suplementos_20261011', 'suplementos')
change('Las consultas internacionales tienen registros de recuperación del 10 de octubre de 2026; el catálogo combina campañas locales de los días 9 y 10. El archivo final no conserva una fecha única de descarga verificable para todas las consultas SGC, por lo que el corte temporal de eventos no se presenta como fecha de adquisición de cada registro.', 'Las consultas internacionales tienen registros de recuperación del 10 de octubre de 2026; el catálogo combina campañas locales de los días 9 y 10.\n\nLimitación de trazabilidad del catálogo SGC. El archivo final no conserva una fecha única de descarga verificable para todas las consultas SGC, por lo que el corte temporal de eventos no se presenta como fecha de adquisición de cada registro. Los registros existentes se conservan; esta revisión no vuelve a descargar ni sustituye el catálogo usado en el experimento. Un depósito público deberá adjuntar los archivos originales y los registros de consulta disponibles, y las adquisiciones futuras deberán registrar consulta y fecha por fuente.')
change("DEST=OUT/'Manuscrito_tomografia_Vp_Colombia_UIS_revision_20261011.docx'", "DEST=OUT/'Manuscrito_tomografia_Vp_Colombia_UIS_revision_20261011b.docx'")
change("fuentes/auditoria_manuscrito_revision_20261011.json", "fuentes/auditoria_manuscrito_revision_20261011b.json")
(OUT/'crear_manuscrito_revision_20261011b.py').write_text(t,encoding='utf-8')
shutil.copytree(OUT/'suplementos_20261011',OUT/'suplementos',dirs_exist_ok=True)

f=(ROOT/'codigos/mapa_kmeans_revision_20261011.py').read_text(encoding='utf-8')
a=f.index('lo,la=np.meshgrid')
b=f.index('ax.set(',a)
f=f[:a]+'''lo,la=np.meshgrid(lon,np.linspace(.8,13,300));x,y=pr.xy(la,lo)
for depth,style in [(60,':'),(80,'-'),(100,'--')]:
    z=np.full_like(x,depth)
    a=((x-centers[0,0])**2+(y-centers[0,1])**2+(z-centers[0,2])**2)
    b=((x-centers[1,0])**2+(y-centers[1,1])**2+(z-centers[1,2])**2)
    ax.contour(lo,la,a-b,levels=[0],colors='black',linewidths=1.5,linestyles=style)
    ax.plot([],[],color='black',linestyle=style,label=f'Bisector a {depth} km')
'''+f[b:]
f=f.replace('Figura_7_clasificacion_kmeans.png','Figura_7_clasificacion_kmeans_revision_b.png')
(ROOT/'codigos/mapa_kmeans_revision_20261011b.py').write_text(f,encoding='utf-8')
print('Prepared revision b without changing frozen revision inputs')
