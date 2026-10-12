# Reproducción de los controles de tomografía Vp

Repositorio público: https://github.com/romusan/relocation-pwave-tomography/tree/tomografia-colombia-uis-h/tomografia_colombia_uis

El manuscrito presenta cinco brazos reales: perfil 1D fijo, perfil 1D ajustable y modelos 3D con pesos de suavidad 2, 20 y 200. Las figuras principales utilizan el peso intermedio fijado antes de los resultados. Consulte `resultados/campana_perfil_suavidad_20261011h/protocol.json` para conocer las reglas de parada, el presupuesto sintético y las limitaciones.

## Contenido

El Word y sus enlaces están en `articulo_boletin_geologia_uis_adjunto`. Mantenga la carpeta `suplementos` junto al manuscrito. Los HTML geométricos tienen su propio aviso de alcance y no representan la inversión de este artículo. Las figuras científicas suplementarias sí corresponden a los controles descritos.

Las observaciones seleccionadas y sus elevaciones están en `resultados/campana_factorizada_cotas_20261011g/selected_picks.csv`; ese nombre identifica el archivo fuente, sin cambiar el protocolo científico del manuscrito. La procedencia de cada llegada está en `arrival_source_matches.csv`. La reconciliación de redes usa también `original_net`; las llegadas publicadas tienen probabilidad convencional 1, no confianza calibrada del detector.

Los resultados por ronda, estáticas, relocalizaciones, modelos, predicciones, recuperaciones y pruebas de malla están en `resultados/campana_perfil_suavidad_20261011h`. El manifiesto SHA256 permite comprobar todos los archivos del paquete. Los cachés de campos se excluyen por tamaño y se regeneran a partir del modelo y de las coordenadas.

## Ejecución

La campaña se ejecutó en Windows con Python 3.14, NumPy 2.4.6, pandas 3.0 y SciPy 1.18, con biblioteca nativa C++ incluida. El núcleo numérico utiliza `codigos/campana_perfil_suavidad_20261011h.py`; las bibliotecas y sus fuentes están en `codigos/native`. Se requiere instalar las dependencias científicas y, si se reconstruye la biblioteca, un compilador compatible con OpenMP. No se depende de una licencia de Word para invertir; la exportación editorial de DOCX es independiente.

Para una repetición que conserve los resultados entregados, desde la raíz del paquete ejecute:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='4'
python codigos/reproducir_campana_h.py
```

La repetición escribe en `resultados/reproduccion_perfil_suavidad_h`, sin borrar los resultados de entrega. Es una campaña costosa; no es una prueba de humo. Los campos intermedios requieren espacio de disco considerable. El contraste pareado de altura está definido en `codigos/contraste_cotas_20261011h.py` y sus resultados se incluyen; ese script original utiliza la carpeta de entrega y no debe ejecutarse para modificarla inadvertidamente.

Si LSMR alcanza 1000 iteraciones (código 7), la misma solución lineal se repite desde cero con límites de 10000 y, si es necesario, 30000, conservando las tolerancias `atol=btol=1e-6`. No se acepta el código 7 como solución convergente. La reanudación y sus intentos se documentan en `lsmr_execution_amendment.json` y `lsmr_resume_audit.json`; los brazos ya terminados se preservan.

La reanudación conserva hasta 2048 MB de campos por instancia en memoria, lee los cachés existentes y calcula los nuevos sin escribirlos en disco. `memory_cache_verification.json` documenta igualdad bit a bit de factor y tiempos frente al cálculo con caché. Los primeros brazos utilizaron campos en disco; el cambio posterior afecta al almacenamiento, no al operador. La reproducción utiliza esta modalidad para limitar el consumo de disco.

Los sintéticos conservan columnas de inyecciones de su archivo fuente como trazabilidad. Las observaciones utilizadas son `true_time_s + effective_error_s`: no se vuelven a sumar las antiguas columnas de ruido, origen o estáticas. Las cuatro rondas sintéticas son un presupuesto común, no prueba de convergencia.

## Alcance

Una parada por límite no equivale a convergencia. La máscara gris indica soporte de parámetros, no resolución. La reserva espacial auxiliar tiene tres rondas y regularización de incrementos; no se presenta como validación espacial de todos los brazos con regularización absoluta. El remuestreo reservado es condicional al catálogo y a los modelos ajustados. La rama pública indicada permite descargar el suplemento; no dispone de DOI.

## Resultados de la campaña terminada

| Modelo | RMS reservado (s) | Rondas | Parada |
|---|---:|---:|---|
| 1D fijo | 1.881488 | 11 | Criterio de estabilización |
| 1D ajustable | 1.763292 | 12 | Límite predefinido, sin estabilización |
| 3D lambda 2 | 1.692492 | 12 | Límite predefinido, sin estabilización |
| 3D lambda 20 | 1.935247 | 12 | Límite predefinido, sin estabilización |
| 3D lambda 200 | 1.700911 | 12 | Límite predefinido, sin estabilización |

El modelo de peso intermedio, fijado para las figuras antes de obtener resultados, satisface la tolerancia empírica del último refinamiento, pero no recupera el damero. El modelo de peso 2 pierde su ventaja frente al perfil ajustable en el subconjunto al refinar la malla. Las superficies de Nazca y Caribe son representaciones geométricas exploratorias conservadas de la etapa anterior; esta campaña no las valida como interfaces tectónicas.

El manuscrito final está en `articulo_boletin_geologia_uis_adjunto/Manuscrito_tomografia_Vp_Colombia_UIS_revision_20261011h.docx`. Para visualizar los HTML, descargue el repositorio y abra los archivos de `articulo_boletin_geologia_uis_adjunto/suplementos` en un navegador; GitHub muestra su código fuente.
