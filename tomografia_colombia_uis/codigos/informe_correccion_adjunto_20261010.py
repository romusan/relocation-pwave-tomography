from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
def main():
    old=json.loads((ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10/summary.json').read_text())
    new=json.loads((RUN/'summary.json').read_text());d=new['final_ray_diagnostic'];fd=json.loads((RUN/'finite_difference_updated_model.json').read_text())
    e=max(x['relative_l2_directional_error'] for x in fd['independent_real_model_fd_checks'])
    final=new['history'][-1];previous=old['history'][-1];initial=new['baseline']
    report=f'''# Corrección del operador de sensibilidad de la tomografía P

Se conservan los resultados originales y se ejecuta nuevamente el experimento iniciado con el perfil 1D. La discrepancia original provenía de integrar un recorrido de 26 vecinos, mientras que la predicción utilizaba campos calculados con Godunov de primer orden y fast sweeping. Esos recorridos no proporcionaban la derivada de ese problema directo discretizado.

## Solución implementada

El módulo C++ `codigos/native/eikonal_adjoint.cpp` diferencia la ecuación discreta en cada nodo. Sean T el tiempo del nodo, a_j los tiempos de sus vecinos causales, h el espaciamiento y s la lentitud. Para D = Σ_j (T − a_j)/h², las derivadas locales son ∂T/∂a_j = (T − a_j)/(h² D) y ∂T/∂s = s/D. Se propagan en orden decreciente de tiempo y se acumulan en las mismas celdas de 20 km utilizadas por las actualizaciones directas de lentitud. El nodo fuente mantiene T = 0 y no recibe una derivada de lentitud local.

No se recortan ni se reescalan las derivadas. La identidad Σ_i (∂T/∂s_i)s_i = T se comprueba sobre la malla directa completa. Es una identidad de homogeneidad del problema discreto, no una reinterpretación del núcleo como longitud de un rayo geométrico. Los campos se rechazan si su residual eikonal relativo supera 2 × 10⁻⁷.

El indicador de cobertura utiliza al menos 20 contribuciones de sensibilidad no menores de 0,1 km, umbral equivalente al 1 % del espaciamiento directo de 10 km. Solo determina las celdas que pueden actualizarse. La matriz y las comprobaciones conservan todas las derivadas. La mayor extensión del núcleo discreto refleja también difusión de la malla y no representa sismos nuevos ni una mejora probada de resolución física.

## Comprobaciones independientes

Cuatro pruebas automatizadas verifican homogeneidad, perturbaciones centrales independientes, invariancia de escala, no negatividad y rechazo de campos inconsistentes. Además, se recalcularon campos completos con perturbaciones positivas y negativas en seis direcciones distribuidas entre tres estaciones y doce eventos por estación. Sobre el modelo final corregido, el error L2 relativo direccional máximo fue {e:.9e}. Las perturbaciones no se generaron con el Jacobiano sometido a prueba.

Para las ecuaciones finales de entrenamiento, el desajuste entre la identidad ponderada y el campo tiene mediana {d['kernel_time_minus_field_median_s']:.9e} s, percentil 95 absoluto {d['kernel_time_minus_field_p95_abs_s']:.9e} s y máximo absoluto {d['kernel_time_minus_field_max_abs_s']:.9e} s. No hubo filas rechazadas. El mayor residual local eikonal fue {d['max_local_eikonal_relative_residual']:.9e}.

## Comparación controlada

Se mantuvieron catálogo, estaciones, perfil 1D, proyección, mallas, partición de eventos, tres rondas y pesos de regularización. La nueva definición de cobertura se declara explícitamente; por ello la comparación evalúa la implementación consistente completa y no aísla únicamente el efecto del algoritmo C++. Los eventos reservados mantienen sus orígenes e hipocentros. No se seleccionó una ronda por el resultado de reserva.

| RMS (s) | Inicial 1D | Implementación anterior | Implementación consistente |
|---|---:|---:|---:|
| Entrenamiento | {initial['train_rms_fixed_catalog_s']:.6f} | {previous['train_rms_after_update_s']:.6f} | {final['train_rms_after_update_s']:.6f} |
| Reserva | {initial['holdout_rms_fixed_catalog_origin_s']:.6f} | {previous['holdout_rms_fixed_catalog_origin_s']:.6f} | {final['holdout_rms_fixed_catalog_origin_s']:.6f} |

Las correcciones de estación y la relocalización contribuyen al ajuste de entrenamiento; su reducción no se atribuye exclusivamente a Vp. El manuscrito actualizado presenta solo el experimento consistente desde 1D, sin convertir esta revisión técnica en una bitácora del artículo. Las superficies y los cortes se recalculan a partir del nuevo resultado.

## Alcance y trazabilidad

La discrepancia entre el operador de sensibilidad y los campos queda corregida a precisión numérica. Persisten las limitaciones de precisión espacial del solucionador de primer orden, elevaciones omitidas, aproximación cartesiana y evaluación de resolución no lineal. Esta corrección no valida por sí sola la identidad de las placas ni la diagonal utilizada como hipótesis de visualización. No se implementó una inversión Q mediante estos núcleos P.

Los archivos `manifest.json`, `finite_difference_real_model.json` y `finite_difference_updated_model.json` documentan fuentes y pruebas. La compilación se registra en `codigos/native/build_adjoint_info.json`. Los resultados y documentos originales se conservan y sus SHA-256 se verifican.

Referencia primaria: Zunino, A., Keating, S. y Fichtner, A. (2025). A discrete adjoint method for deterministic and probabilistic eikonal-equation-based inversion of traveltime for velocity and source location. Geophysical Journal International, 243(1), ggaf283. https://doi.org/10.1093/gji/ggaf283. Se utiliza el principio de diferenciación discreta; no se afirma implementar su solucionador FMM de segundo orden.
'''
    out=RUN/'informe_correccion_numerica.md';out.write_text(report,encoding='utf-8')
    audit={'report_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'summary.json',RUN/'finite_difference_updated_model.json',ROOT/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10/summary.json']}}
    (RUN/'auditoria_informe.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(out)
if __name__=='__main__':main()
