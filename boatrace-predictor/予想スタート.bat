@echo off
start "ボートレース予想" cmd /k "cd /d C:\Users\user\Desktop\test\boatrace-predictor & echo. & echo ================================ & echo   ボートレース予想システム & echo ================================ & echo. & echo [1] コードを最新に更新中... & git pull origin claude/update-april-7-summary-ceGQV & echo. & echo [2] 予想スタート（Ctrl+C で停止 → ウィンドウはそのまま残ります） & echo. & .venv\Scripts\python.exe main.py --mode auto"
