# Respuesta a la revisión de la versión 20261011

La versión 20261011b mantiene los datos, la inversión, los resultados numéricos y las superficies NURBS del experimento. Se revisó el texto y se amplió la Figura 7 sin repetir la tomografía.

1. Título español: el DOCX 20261011 ya contiene el título como primer párrafo con estilo Title. La nueva versión lo conserva, hace explícita su visibilidad y se verifica en la primera página renderizada. No se duplica.
2. Selección de eventos: se contrastó asignacion_sismos.csv con el código de las superficies. Los 1180 elegibles se dividen en 795 Nazca y 385 Caribe; Nazca conserva 570 compatibles con Slab2 y excluye 225. Caribe usa sus 385. La envolvente de Caribe recorta la representación y sus anclajes, no los eventos. Se explica en §3.4.
3. Peso uniforme: se sustituye incertidumbre uniforme por divisor uniforme de residuos en §5.1.
4. Inicializaciones: las mismas 30 semillas se reutilizan en cada k y métrica; 150 ajustes por métrica, 300 en total. Se aclara que 3× de exageración gráfica no altera las distancias del ajuste.
5. Evaluación por profundidad: la Tabla 6 y su interpretación pasan a §4.6, separadas de resolución y clasificación.
6. Saturación: §5.1 cuantifica 2857/10011 = 28,54 %. Se señala que las cotas condicionan la amplitud, sin afirmar que las celdas carezcan de toda información.
7. Bisector: la Figura 7 incorpora 60, 80 y 100 km con estilos diferentes; no modifica grupos ni NURBS. El texto explica la elección de niveles alrededor del corte de 80 km.
8. Portabilidad: los enlaces pasan a la carpeta sin fecha suplementos. El paquete ZIP contiene el Word y esa carpeta contigua. La extracción completa conserva los enlaces y los HTML funcionan sin conexión.
9. Trazabilidad SGC: se separa explícitamente como limitación. No se inventa una fecha única ni se sustituye el catálogo original con una adquisición posterior. Esta revisión registra fecha y hashes de sus insumos; no los presenta como fechas de descarga histórica.

Pendiente antes del envío editorial: depositar públicamente los datos permitidos, el código y los HTML en un repositorio con identificador persistente y reemplazar enlaces locales. No se ha publicado ni emitido DOI. Permanecen pendientes las reinversiones y pruebas de resolución adicionales descritas en la revisión previa. Una descarga nueva con registro fechado constituiría otra adquisición y requeriría comprobar su equivalencia con el catálogo del experimento.
