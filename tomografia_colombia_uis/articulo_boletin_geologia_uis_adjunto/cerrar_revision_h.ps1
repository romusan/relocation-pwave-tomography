param([int]$MainProcessId)
$ErrorActionPreference='Stop'
$taskProjectRoot=Split-Path $PSScriptRoot -Parent
$taskResults=Join-Path $taskProjectRoot 'resultados/campana_perfil_suavidad_20261011h'
$taskComplete=Join-Path $taskResults 'completed.json'
Set-Location -LiteralPath $taskProjectRoot
while (-not (Test-Path -LiteralPath $taskComplete)) {
    if (-not (Get-Process -Id $MainProcessId -ErrorAction SilentlyContinue)) { throw 'Numerical process ended without a complete report; no manuscript will be authored.' }
    Start-Sleep -Seconds 10
}
$taskNumericPython='C:/Users/claudia.parra/AppData/Local/Programs/Python/Python314/python.exe'
$taskDocumentPython='C:/Users/claudia.parra/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$taskNode='C:/Users/claudia.parra/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='4'
& $taskNumericPython -X utf8 codigos/resumir_experimentos_revision_h.py
if ($LASTEXITCODE -ne 0) { throw 'Completed numerical results failed aggregation checks.' }
& $taskNumericPython -X utf8 articulo_boletin_geologia_uis_adjunto/figuras_revision_h.py velocities
if ($LASTEXITCODE -ne 0) { throw 'Main figures failed.' }
& $taskNumericPython -X utf8 articulo_boletin_geologia_uis_adjunto/suplementos_revision_h.py
if ($LASTEXITCODE -ne 0) { throw 'Supplement figures failed.' }
$taskReceipt=Join-Path $taskResults 'artifact_operation_receipt.json'
if (-not (Test-Path -LiteralPath $taskReceipt)) {
    & $taskNode 'C:/Users/claudia.parra/.codex/plugins/cache/openai-primary-runtime/documents/26.1007.11041/skills/documents/container_tools/mark_artifact_operation_started.mjs' --operation-kind create --expected-output-count 1 --output-format docx
    if ($LASTEXITCODE -ne 0) { throw 'Document authoring marker failed.' }
    & $taskDocumentPython -s -X utf8 articulo_boletin_geologia_uis_adjunto/crear_manuscrito_revision_20261011h.py
    $taskAuthorExit=$LASTEXITCODE
    @{ marker_success=$true; authoring_exit=$taskAuthorExit } | ConvertTo-Json | Set-Content -LiteralPath $taskReceipt -Encoding utf8
    if ($taskAuthorExit -ne 0) { throw 'Manuscript builder failed; marker already recorded.' }
} else {
    & $taskDocumentPython -s -X utf8 articulo_boletin_geologia_uis_adjunto/crear_manuscrito_revision_20261011h.py
    if ($LASTEXITCODE -ne 0) { throw 'Manuscript builder failed.' }
}
& $taskDocumentPython -s -X utf8 articulo_boletin_geologia_uis_adjunto/verificar_manuscrito_h.py
if ($LASTEXITCODE -ne 0) { throw 'Manuscript content checks failed.' }
$taskQa=Join-Path $PSScriptRoot 'qa_revision_20261011h'
New-Item -ItemType Directory -Path $taskQa -Force | Out-Null
$taskDocx=Join-Path $PSScriptRoot 'Manuscrito_tomografia_Vp_Colombia_UIS_revision_20261011h.docx'
$taskPdf=Join-Path $taskQa 'word_render.pdf'
& (Join-Path $taskProjectRoot 'articulo_boletin_geologia_uis/exportar_word_pdf.ps1') -DocxPath $taskDocx -PdfPath $taskPdf
& $taskDocumentPython -s -X utf8 articulo_boletin_geologia_uis/render_articulo.py $taskDocx $taskQa
if ($LASTEXITCODE -ne 0) { throw 'Manuscript render failed.' }
@{ status='ready_for_visual_review'; docx=$taskDocx; qa_directory=$taskQa; visual_review_all_pages=$false } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskResults 'render_ready.json') -Encoding utf8
Write-Output 'Scientific campaign and rendering complete. Every rendered page still requires visual review before delivery.'
