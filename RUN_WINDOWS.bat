@echo off
setlocal
cd /d "%~dp0"
echo Setting up GridPulse Team 11...
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
if errorlevel 1 goto failed
streamlit run app.py
goto end
:failed
echo.
echo Setup or model training failed. Read the error above.
:end
pause
