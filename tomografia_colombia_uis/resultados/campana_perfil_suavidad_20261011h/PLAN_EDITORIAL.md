# Revisión propia y experimento ampliado

Los resultados de la versión g se preservan. El manuscrito siguiente incorporará únicamente métricas nuevas terminadas y mantendrá separadas las hipótesis causales de los hallazgos observados.

Se acepta reportar entrenamiento y reserva por ronda, ajustar un perfil 1D, variar regularización, añadir errores efectivos comparables y explicitar ecuaciones, procedencia y máscara de parámetros. Se acepta centrar el refinamiento y la sensibilidad del ranking predictivo. Los cortes primarios usarán el peso intermedio 20, definido antes de disponer de resultados 3D de este barrido, sin seleccionar un ganador con la reserva.

La regularización previa penalizaba incrementos: aumentar ese peso no asegura limitar la rugosidad acumulada. El experimento ampliado penaliza la desviación total de lentitud respecto al perfil de referencia y las estáticas totales. Esta diferencia metodológica se declarará. Los parámetros de convergencia, máximo de rondas y pesos están en protocol.json. Si se alcanza el máximo sin cumplir la puerta, se informará como tal.

No se acepta que el incremento de RMS entre campañas demuestre un perfil demasiado rápido: cambiaron población, operador y ajuste. La disminución de la pendiente estáticas–elevación tampoco era un contraste limpio. Se añadió un contraste computacional pareado a cota cero y física con la misma población y operador. Su interpretación será condicionada a sus resultados y no a las cifras históricas.

El error reservado incluye incertidumbre del catálogo y del tiempo de origen, discretización, selección de fases y deficiencias del modelo. No se equipara directamente a error de picks. El sintético ampliado es una prueba de estrés con errores efectivos empíricos calibrados solo con entrenamiento; no estima el error intrínseco del picker. No se reutilizan adicionalmente las inyecciones antiguas de ruido, estática y origen, para evitar duplicarlas con la distribución efectiva.

La ganancia era una condición conjunta junto con r y L2, no una prueba suficiente por sí sola. Se omite de la tabla principal para claridad y se conserva en los archivos de evidencia previos. No se atribuye fracaso a todas las escalas ni degradación sintética al catálogo real.

Se retirará del cuerpo la interpretación geométrica histórica, se preservarán HTML exploratorios en un suplemento con nombres neutros, y se nombrará Codex de OpenAI como herramienta de asistencia. No se inventará una revisión humana de todos los autores ni un DOI de depósito público. Un depósito con DOI requiere una cuenta de repositorio con capacidad de publicación que no está disponible en esta sesión.

Treister y Haber 2016: Journal of Computational Physics 324, 210–225, DOI 10.1016/j.jcp.2016.08.012. Verificado en el registro editorial https://www.sciencedirect.com/science/article/pii/S0021999116303552 y en el repositorio de la institución autora https://www.bgu.ac.il/en/researcher/eran-treister/publications/324875767/.
