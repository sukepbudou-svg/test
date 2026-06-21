Set-Location "C:\Users\user\Desktop\test\boatrace-predictor"
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  ボートレース予想システム" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[1] コードを最新に更新中..." -ForegroundColor Yellow
git pull origin claude/update-april-7-summary-ceGQV
Write-Host ""
Write-Host "[2] 予想スタート（Ctrl+C で停止 → ウィンドウはそのまま残ります）" -ForegroundColor Green
Write-Host ""
& ".\.venv\Scripts\python.exe" main.py --mode auto
Write-Host ""
Write-Host "予想が停止しました。git pull や再起動が可能です。" -ForegroundColor Yellow
