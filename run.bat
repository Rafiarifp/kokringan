@echo off
setlocal
cd /d "%~dp0"

python -m streamlit run app/dashboard.py

if errorlevel 1 (
    echo.
    echo Dashboard gagal dijalankan. Periksa instalasi Python/Streamlit dan lokasi app\dashboard.py.
    pause
)
endlocal
