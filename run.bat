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

rem --- 2. Pastikan file utama ada ---
if not exist "app\dashboard.py" (
    echo [ERROR] app\dashboard.py tidak ditemukan. Jalankan run.bat dari folder proyek.
    goto :fail
)

rem --- 3. Buat virtual environment sekali saja ---
if not exist ".venv\Scripts\python.exe" (
    echo Membuat virtual environment...
    python -m venv .venv || goto :fail
)
set "PY=.venv\Scripts\python.exe"

rem --- 4. Pasang dependensi saat pertama kali atau saat requirements.txt berubah ---
fc /b requirements.txt .venv\requirements.installed >nul 2>nul
if errorlevel 1 (
    echo Memasang dependensi. Pemasangan pertama bisa memakan waktu beberapa menit
    echo karena PyTorch berukuran besar. Mohon tunggu...
    "%PY%" -m pip install --upgrade pip --disable-pip-version-check || goto :fail
    "%PY%" -m pip install -r requirements.txt --disable-pip-version-check || goto :fail
    copy /y requirements.txt .venv\requirements.installed >nul
)

rem --- 5. Info jika .env belum ada (dashboard tetap jalan, scraper butuh API key) ---
if not exist ".env" (
    echo [INFO] File .env tidak ditemukan. Dashboard tetap berjalan, tetapi scraper
    echo        membutuhkan API key. Lihat README bagian konfigurasi .env.
)

rem --- 6. Lewati prompt email Streamlit pada peluncuran pertama ---
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

rem --- 7. Mulai server Streamlit di jendela terminal terpisah yang diminimalkan ---
set "KOKRINGAN_LOCAL=1"
start "KOKRINGAN Server" /min "%PY%" -m streamlit run app/dashboard.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false

rem --- 8. Tunggu server siap (maksimal 60 detik) ---
set /a TRIES=0
:waitloop
curl -s -o nul http://localhost:%PORT%/_stcore/health
if not errorlevel 1 goto :ready
set /a TRIES+=1
if %TRIES% geq 60 goto :timeout
timeout /t 1 /nobreak >nul
goto :waitloop

:timeout
echo [ERROR] Server tidak merespons dalam 60 detik.
call :stopserver
goto :fail

:ready
rem --- 9. Cari Edge atau Chrome untuk membuka dashboard sebagai jendela aplikasi ---
set "BROWSER="
if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not defined BROWSER if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if not defined BROWSER if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "BROWSER=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "BROWSER=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set "BROWSER=%LocalAppData%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER goto :nobrowser

echo Dashboard berjalan. Tutup jendela dashboard untuk menghentikan semuanya.
start /wait "" "%BROWSER%" --app=http://localhost:%PORT% --user-data-dir="%LocalAppData%\KOKRINGAN\browser" --no-first-run --no-default-browser-check
call :stopserver
goto :end

:nobrowser
start "" http://localhost:%PORT%
echo Edge/Chrome tidak ditemukan. Dashboard dibuka di browser bawaan.
echo Tekan tombol apa saja di jendela ini untuk menghentikan dashboard.
pause >nul
call :stopserver
goto :end

:stopserver
rem Matikan proses server beserta semua proses turunannya (termasuk scraper)
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do taskkill /PID %%P /T /F >nul 2>nul
exit /b 0

:fail
echo.
echo Gagal dijalankan. Periksa pesan di atas.
pause
:end
endlocal
