@echo off
cd /d C:\Users\user\Desktop\test\boatrace-predictor
echo コードを最新に更新中...
git pull origin claude/update-april-7-summary-ceGQV
echo.
echo 予想を開始します...
echo 終了するときは Ctrl+C を押してください
echo.
.venv\Scripts\python.exe main.py --mode auto
pause
