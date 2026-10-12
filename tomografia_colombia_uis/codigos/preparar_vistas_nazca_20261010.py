"""Version new geographic viewers while preserving frozen previous sources."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'vista_eje_caribe_1d_20261010.py').read_text(encoding='utf-8')
changes={'from regional.geometry import Projection':'from nazca_campana_20261010 import projection as Projection','migracion_eje_caribe_1d_2026-10-10':'migracion_nazca_cauca_valle_eje_1d_2026-10-10','eje_caribe_1d_vista_final_2026-10-10':'nazca_cauca_valle_eje_1d_vista_2026-10-10','nuevos Eje Cafetero y Caribe':'nuevos Cauca Valle y Eje Cafetero','del Eje Cafetero y Caribe':'de Cauca Valle y Eje Cafetero','ampliación Eje Cafetero y Caribe':'ampliación Cauca Valle y Eje Cafetero','box(-79,2,-71,13)':'box(-79,.8,-71,13)','range=[2,13]':'range=[.8,13]','ylim=(2,13)':'ylim=(.8,13)','x=.726,y=1,z=.148':'x=.655,y=1,z=.133','[.726,1,.444]':'[.655,1,.399]','inputs=[Path(__file__),':"inputs=[Path(__file__),ROOT/'codigos/nazca_campana_20261010.py',"}
for before,after in changes.items():s=s.replace(before,after)
(HERE/'vista_nazca_sur_20261010.py').write_text(s,encoding='utf-8')
s=(HERE/'perfiles_transicion_eje_caribe_20261010.py').read_text(encoding='utf-8')
changes={'from regional.geometry import Projection':'from nazca_campana_20261010 import projection as Projection','migracion_eje_caribe_1d_2026-10-10':'migracion_nazca_cauca_valle_eje_1d_2026-10-10','xlim=(4,13)':'xlim=(.8,13)','Perfiles para comparar continuidad, separación y superposición':'Perfiles de Nazca y transición Caribe desde inicio 1D','inputs=[Path(__file__),':"inputs=[Path(__file__),ROOT/'codigos/nazca_campana_20261010.py',"}
for before,after in changes.items():s=s.replace(before,after)
(HERE/'perfiles_nazca_sur_20261010.py').write_text(s,encoding='utf-8')
print('New geographic viewer sources prepared')
