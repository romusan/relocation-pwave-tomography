from pathlib import Path
HERE=Path(__file__).resolve().parent
source=HERE/'auditar_sensibilidad_real_20261010.py'
text=source.read_text(encoding='utf-8').replace("BASE=ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10'","BASE=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'").replace("'finite_difference_real_model.json'","'finite_difference_updated_model.json'")
exec(compile(text,str(source),'exec'),{'__file__':str(source),'__name__':'__audit_final_consistent_model__'})
