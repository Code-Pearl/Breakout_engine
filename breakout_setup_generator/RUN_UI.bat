@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM --- find python (double-click often has a thinner PATH than your terminal)
set PYCMD=python
where python >nul 2>nul
if errorlevel 1 (
  set PYCMD=py -3
  where py >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] No Python found. Install Python 3.11+ and tick "Add to PATH".
    pause
    exit /b 1
  )
)

REM --- pick the first free port 5000-5009
set PORT=
for /f %%p in ('powershell -noprofile -command "$c=New-Object Net.Sockets.TcpClient; foreach ($pt in 5000..5009) { try { $c.Connect('127.0.0.1',$pt); $c.Close(); $c=New-Object Net.Sockets.TcpClient } catch { echo $pt; break } }"') do set PORT=%%p
if "%PORT%"=="" (
  echo [ERROR] Ports 5000-5009 are all busy. Close the other BreakoutLab window and try again.
  pause
  exit /b 1
)

echo Starting BreakoutLab server on port %PORT% ...
start "BreakoutLab server (port %PORT%)" %PYCMD% ui_server.py --port %PORT%

REM --- wait until the server answers (up to ~40s), THEN open the browser
set URL=http://127.0.0.1:%PORT%/
set READY=0
for /l %%i in (1,1,20) do (
  curl -sf -o nul --max-time 2 %URL%api/status >nul 2>nul
  if not errorlevel 1 (
    set READY=1
    goto :openbrowser
  )
  timeout /t 2 /nobreak >nul
)
echo [ERROR] Server did not start. See the "BreakoutLab server" window for the error.
pause
exit /b 1

:openbrowser
echo UI ready at %URL%
if not "%BL_NO_BROWSER%"=="1" start "" %URL%
echo Keep this window and the server window open while you use the site.
pause
