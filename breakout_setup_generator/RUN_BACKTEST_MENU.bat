@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

:MENU
cls
echo ============================================================
echo  BREAKOUT SETUP GENERATOR - Backtest Menu
echo ============================================================
echo.
echo  LIVE DATA ONLY - synthetic random data has been removed.
echo  Just press Enter to accept the [default].
echo.
echo  [1] Quick backtest (live cache, 150 ideas + equity JPGs)
echo  [2] Standard backtest (choose data + size + checks + plots)
echo  [3] Real data backtest (real_data\ - 5 real markets)
echo  [4] Real intraday backtest (real_intraday_data\)
echo  [5] Evolve mode (genetic algorithm)
echo  [6] ML tools (train / evaluate / prescreen)
echo  [7] Validate + Export + Amipy check + Plots on live data
echo  [8] Custom data folder backtest
echo  [9] Download fresh real data (cached, Yahoo Finance)
echo  [0] Exit
echo.
set /p CHOICE="Pick 1-9 (0 to exit): "

if "%CHOICE%"=="1" goto QUICK
if "%CHOICE%"=="2" goto STANDARD
if "%CHOICE%"=="3" goto REALDATA
if "%CHOICE%"=="4" goto INTRADATA
if "%CHOICE%"=="5" goto EVOLVE
if "%CHOICE%"=="6" goto MLMENU
if "%CHOICE%"=="7" goto CHECKS
if "%CHOICE%"=="8" goto CUSTOM
if "%CHOICE%"=="9" goto FETCH
if "%CHOICE%"=="0" exit /b 0
goto MENU

:ASK_COMMON
echo.
echo --- Live data source ---
echo  [1] Downloaded cache, auto-refreshed (default, current to this week)
echo  [2] real_data\ (bundled snapshot)
echo  [3] real_intraday_data\ (bundled hourly snapshot)
echo  [4] Custom folder
set /p DATAC="Data [1]: "
if "%DATAC%"=="" set DATAC=1
set DATAARGS=--data-dir .\data_cache
set MINTRADES=15
if "%DATAC%"=="1" python fetch_data.py
if "%DATAC%"=="2" set DATAARGS=--data-dir .\real_data
if "%DATAC%"=="3" set DATAARGS=--data-dir .\real_intraday_data
if "%DATAC%"=="3" set MINTRADES=100
if "%DATAC%"=="4" goto ASK_CUSTOMDIR
goto ASK_COMMON2
:ASK_CUSTOMDIR
set /p CUSTDIR="Folder with your CSVs [.\my_csvs]: "
if "%CUSTDIR%"=="" set CUSTDIR=.\my_csvs
set DATAARGS=--data-dir "%CUSTDIR%"
goto ASK_COMMON2

:ASK_COMMON2
echo.
echo --- Position sizing ---
echo  [1] full_compounding (default, simple)
echo  [2] fixed_fractional (realistic risk per trade)
set /p SIZEC="Sizing [1]: "
if "%SIZEC%"=="" set SIZEC=1
set SIZEARGS=
if "%SIZEC%"=="2" (
  set /p RISKP="Risk %% per trade [10]: "
  if "!RISKP!"=="" set RISKP=10
  for /f "tokens=*" %%a in ('python -c "print(float('!RISKP!')/100)"') do set RISKDEC=%%a
  set SIZEARGS=--position-sizing fixed_fractional --risk-pct !RISKDEC!
)
set /p SEED="Random seed [blank=random]: "
set SEEDARG=
if not "%SEED%"=="" set SEEDARG=--seed %SEED%
exit /b 0

:QUICK
python fetch_data.py
python main.py --data-dir .\data_cache --num 150 --plot-top 10
pause
goto MENU

:STANDARD
call :ASK_COMMON
set /p NUM="How many strategies? [150]: "
if "%NUM%"=="" set NUM=150
set /p TOP="Leaderboard size [15]: "
if "%TOP%"=="" set TOP=15
set /p MT="Min trades per asset [%MINTRADES%]: "
if "%MT%"=="" set MT=%MINTRADES%
set /p VAL="Validate top N? (walk-forward+MonteCarlo+neighbors) [0=skip]: "
if "%VAL%"=="" set VAL=0
set /p EXP="Export top N to code? [0=skip]: "
if "%EXP%"=="" set EXP=0
set EXPARG=
if not "%EXP%"=="0" (
  echo  Formats: all ^(default^), easylanguage, afl, pinescript ^(comma-separated^)
  set /p EXPF="Export format [all]: "
  if "!EXPF!"=="" set EXPF=all
  set EXPARG=--export-top %EXP% --export-format !EXPF!
)
set /p AMI="Amipy cross-check top N? [0=skip]: "
if "%AMI%"=="" set AMI=0
set /p PLOT="Equity-curve JPGs for top N? [10, 0=skip]: "
if "%PLOT%"=="" set PLOT=10
python main.py --num %NUM% --top %TOP% --min-trades %MT% %DATAARGS% %SIZEARGS% %SEEDARG% --validate-top %VAL% %EXPARG% --amipy-check-top %AMI% --plot-top %PLOT%
pause
goto MENU

:REALDATA
set DATAARGS=--data-dir .\real_data
set SIZEARGS=
set SEEDARG=
set /p NUM="How many strategies? [400]: "
if "%NUM%"=="" set NUM=400
set /p VAL="Validate top N? [10]: "
if "%VAL%"=="" set VAL=10
set /p AMI="Amipy cross-check top N? [5]: "
if "%AMI%"=="" set AMI=5
set /p EXP="Export top N to code? [5]: "
if "%EXP%"=="" set EXP=5
python main.py --data-dir .\real_data --num %NUM% --validate-top %VAL% --amipy-check-top %AMI% --export-top %EXP% --plot-top %EXP%
pause
goto MENU

:INTRADATA
set /p NUM="How many strategies? [400]: "
if "%NUM%"=="" set NUM=400
set /p MT="Min trades per asset [100]: "
if "%MT%"=="" set MT=100
set /p VAL="Validate top N? [10]: "
if "%VAL%"=="" set VAL=10
python main.py --data-dir .\data_cache --num %NUM% --validate-top %VAL% --plot-top 10
pause
goto MENU

:EVOLVE
call :ASK_COMMON
set /p GENS="Generations [15]: "
if "%GENS%"=="" set GENS=15
set /p POP="Population [60]: "
if "%POP%"=="" set POP=60
set /p MT="Min trades per asset [%MINTRADES%]: "
if "%MT%"=="" set MT=%MINTRADES%
set /p ML="Use ML prescreening? (y/N): "
set MLARG=
if /i "%ML%"=="y" set MLARG=--ml-prescreen
python evolve.py --generations %GENS% --population %POP% --min-trades %MT% %DATAARGS% %SIZEARGS% %SEEDARG% %MLARG%
pause
goto MENU

:MLMENU
cls
echo --- ML tools ---
echo  [1] Bootstrap training log to 1000 rows
echo  [2] Train model
echo  [3] Evaluate model (recall@K test)
echo  [4] Prescreen: generate 500, backtest best 50
echo  [0] Back
set /p MLC="Pick: "
if "%MLC%"=="1" python ml_rank.py --bootstrap 1000 & pause & goto MENU
if "%MLC%"=="2" python ml_rank.py --train & pause & goto MENU
if "%MLC%"=="3" (
  set /p MLN="Candidates N [300]: "
  if "!MLN!"=="" set MLN=300
  set /p MLK="Top K [50]: "
  if "!MLK!"=="" set MLK=50
  python ml_rank.py --evaluate !MLN! !MLK! & pause & goto MENU
)
if "%MLC%"=="4" (
  set /p PSN="Generate N [500]: "
  if "!PSN!"=="" set PSN=500
  set /p KEEP="Backtest top KEEP [50]: "
  if "!KEEP!"=="" set KEEP=50
  python ml_rank.py --prescreen !PSN! --keep !KEEP! & pause & goto MENU
)
goto MENU

:CHECKS
echo Fresh run on LIVE cache (auto-refreshed), with all checks + plots.
python fetch_data.py
set /p NUM="How many strategies? [300]: "
if "%NUM%"=="" set NUM=300
set /p VAL="Validate top N? [10]: "
if "%VAL%"=="" set VAL=10
set /p EXP="Export top N? [5]: "
if "%EXP%"=="" set EXP=5
set /p AMI="Amipy cross-check top N? [5]: "
if "%AMI%"=="" set AMI=5
python main.py --data-dir .\data_cache --num %NUM% --validate-top %VAL% --export-top %EXP% --amipy-check-top %AMI% --plot-top %EXP%
pause
goto MENU

:CUSTOM
set /p CUSTDIR="Folder with your CSVs [.\my_csvs]: "
if "%CUSTDIR%"=="" set CUSTDIR=.\my_csvs
set /p NUM="How many strategies? [300]: "
if "%NUM%"=="" set NUM=300
set /p MT="Min trades per asset [15, raise to 100+ for intraday]: "
if "%MT%"=="" set MT=15
python main.py --data-dir "%CUSTDIR%" --num %NUM% --min-trades %MT%
pause
goto MENU

:FETCH
echo.
echo --- Download real data (Yahoo Finance, cached in data_cache\) ---
echo  [1] Daily bars, 5-asset set: SPY GLD BTC-USD EURUSD TNX (default)
echo  [2] Hourly bars (last ~2 years, Yahoo limit)
set /p FREQC="Bars [1]: "
if "%FREQC%"=="" set FREQC=1
set FETCHINT=1d
if "%FREQC%"=="2" set FETCHINT=1h
set /p FORCE="Force re-download even if cache is fresh? (y/N): "
set FORCEARG=
if /i "%FORCE%"=="y" set FORCEARG=--refresh
python fetch_data.py --interval %FETCHINT% %FORCEARG%
echo.
set /p NOWBT="Backtest on the downloaded data now? (Y/n): "
if /i "%NOWBT%"=="n" goto MENU
set /p NUM="How many strategies? [400]: "
if "%NUM%"=="" set NUM=400
set /p VAL="Validate top N? [10]: "
if "%VAL%"=="" set VAL=10
python main.py --data-dir .\data_cache --num %NUM% --validate-top %VAL%
pause
goto MENU
