"""Recompute geometry/figures from the corrected inversion, retaining originals."""
from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'codigos'))
RUN=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
GEO=ROOT/'resultados/placas_adjunto_consistente_2026-10-10'
VIEW=ROOT/'resultados/fragmentos_adjunto_consistente_2026-10-10'
OUT=ROOT/'articulo_boletin_geologia_uis_adjunto'
OLD=ROOT/'articulo_boletin_geologia_uis'
def main():
    assert (RUN/'summary.json').is_file(),'Inversion must finish before scientific postprocessing'
    OUT.mkdir(exist_ok=True);(OUT/'fuentes').mkdir(exist_ok=True);(OUT/'figuras').mkdir(exist_ok=True)
    for name in ['plantilla_uis.docx','bibliografia_verificada.json','fong_saunders_crossref.json']:
        shutil.copy2(OLD/'fuentes'/name,OUT/'fuentes'/name)
    import placas_nazca_caribe_diagonal_20261010 as geo
    geo.RUN=RUN;geo.OUT=GEO;geo.main()
    from reconstruir_benioff_ampliado import controls,supported_mesh
    e=pd.read_csv(RUN/'relocated_train_events.csv');c=controls(e);xyz=c[c.accepted][['x_km','y_km','depth_km']].to_numpy();faces=supported_mesh(xyz)
    VIEW.mkdir(exist_ok=False);c.to_csv(VIEW/'controles.csv',index=False);np.savez(VIEW/'fragmentos_finales.npz',xyz=xyz,triangles=faces)
    import figuras_articulo_uis_20261010 as figures
    figures.RUN=RUN;figures.GEO=GEO;figures.OLDVIEW=VIEW;figures.OUT=OUT;figures.main()
    # The audited selection is identical; retain its corrected dates and metadata.
    m=json.loads((OUT/'fuentes/metricas_articulo.json').read_text());oldm=json.loads((OLD/'fuentes/metricas_articulo_auditadas.json').read_text())
    for key in ['selected_catalog_dates','minimum_p_stations_selected','max_gap_selected_deg','unique_station_identifiers_P','origin_agencies']:m[key]=oldm[key]
    m['input_sha256'][str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for f in [GEO/'metodologia.json',RUN/'config.json',RUN/'domain.json',RUN/'manifest.json',OLD/'fuentes/metricas_articulo_auditadas.json']:
        m['input_sha256'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    (OUT/'fuentes/metricas_articulo_auditadas.json').write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf-8')
    fonts=ROOT/'codigos/cortes_articulo_uis_tipografia_20261010.py'
    text=fonts.read_text(encoding='utf-8').replace("RUN=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10'","RUN=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'").replace("OUT=ROOT/'articulo_boletin_geologia_uis'","OUT=ROOT/'articulo_boletin_geologia_uis_adjunto'")
    exec(compile(text,str(fonts),'exec'),{'__file__':str(fonts),'__name__':'__render_corrected_sections__'})
    print('Corrected scientific manuscript inputs prepared',flush=True)
if __name__=='__main__':main()
