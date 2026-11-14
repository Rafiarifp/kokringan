@echo off
setlocal
cd /d "%~dp0"
title KOKRINGAN Command Center
set "PORT=8501"

rem --- 1. Periksa Python 3.10+ ---
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 atau lebih baru tidak ditemukan.
    echo Pasang dari https://www.python.org/downloads/ dan centang "Add Python to PATH".
    goto :fail
)
