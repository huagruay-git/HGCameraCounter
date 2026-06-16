# Nightly CLEAN retrain of the chair-service classifier (haircut vs not-haircut).
# Runs off-hours (salon closed) so it does not starve the live counting runtime.
# Trains on the de-poisoned feedback folders and writes a CANDIDATE model
# (models/chair_service_cls_clean.pt) — it does NOT overwrite the production model,
# so the result must be validated before re-enabling the classifier.
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\พีซี\code\HGCameraCounter'
Set-Location $root
$ts  = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $root "logs\retrain_chair_cls_$ts.log"
"[$ts] starting clean chair-cls retrain" | Out-File -FilePath $log -Encoding utf8
& "$root\.venv\Scripts\python.exe" "$root\scripts\train_chair_service_classifier.py" `
    --positive-dir 'data/performance_feedback/haircut' `
    --negative-dirs 'data/performance_feedback/no haircut' `
    --output-model 'models/chair_service_cls_clean.pt' `
    --epochs 30 --imgsz 224 --batch 32 --patience 12 --workers 2 `
    --project 'runs/classify' --name 'chair_service_cls_clean' --exist-ok *>> $log
"[$(Get-Date -Format 'yyyyMMdd_HHmmss')] done (exit $LASTEXITCODE)" | Out-File -FilePath $log -Append -Encoding utf8
