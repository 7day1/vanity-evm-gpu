@echo off
REM ============================================================
REM  vanity-evm-gpu on Windows 11 - interactive launcher
REM
REM  Double-click this file (or run it from PowerShell) and
REM  pick a mode. Each command resolves the binary relative to
REM  this script so it works from any cwd, including Explorer.
REM ============================================================
setlocal enableextensions

REM Resolve the folder containing this script (works in cmd / explorer)
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Search for the .exe in common locations (CI artifact layout vs
REM cargo build layout).
set "EXE_CLI="
set "EXE_GUI="
set "CANDIDATES=^
  "%SCRIPT_DIR%\..\..\target\x86_64-pc-windows-msvc\release\vanity-evm-gpu.exe";^
  "%SCRIPT_DIR%\..\..\target\release\vanity-evm-gpu.exe";^
  "%SCRIPT_DIR%\vanity-evm-gpu.exe";^
  "%SCRIPT_DIR%\..\release\vanity-evm-gpu.exe""
for %%C in (%CANDIDATES%) do (
    if not defined EXE_CLI if exist %%~C set "EXE_CLI=%%~C"
)
set "CANDIDATES_GUI=^
  "%SCRIPT_DIR%\..\..\target\x86_64-pc-windows-msvc\release\vanity-evm-gui.exe";^
  "%SCRIPT_DIR%\..\..\target\release\vanity-evm-gui.exe";^
  "%SCRIPT_DIR%\vanity-evm-gui.exe";^
  "%SCRIPT_DIR%\..\release\vanity-evm-gui.exe""
for %%C in (%CANDIDATES_GUI%) do (
    if not defined EXE_GUI if exist %%~C set "EXE_GUI=%%~C"
)

if not defined EXE_CLI (
    echo [run.bat] Could not find vanity-evm-gpu.exe.
    echo   Looked in:
    echo     %SCRIPT_DIR%\..\..\target\x86_64-pc-windows-msvc\release\
    echo     %SCRIPT_DIR%\..\..\target\release\
    echo     %SCRIPT_DIR%\
    echo   Either build locally (cargo build --release) or download the
    echo   "vanity-evm-gpu-windows-x64" artifact from GitHub Actions.
    pause
    exit /b 1
)

:menu
cls
echo ============================================================
echo   vanity-evm-gpu on Windows 11
echo   CLI: %EXE_CLI%
echo   GUI: %EXE_GUI%
echo ============================================================
echo   1. List OpenCL GPU devices
echo   2. Self-test  (validate GPU kernel vs CPU reference)
echo   3. Benchmark  (10 second, pure-rate, no candidates written)
echo   4. Dry-run    (one GPU dispatch, exit; safe beginner practice)
echo   5. GUI        (open the eframe window)
echo   6. Custom args (advanced)
echo   7. Search 8x8 + 8x7  (one command: 88888888 OR 77777777)
echo   8. Each group once   (88888888 AND 77777777, --all-groups)
echo   9. Search + custom result dir
echo   0. Quit
echo ============================================================
set /p CHOICE=Pick [0-9]:
if "%CHOICE%"=="1" goto list
if "%CHOICE%"=="2" goto selftest
if "%CHOICE%"=="3" goto bench
if "%CHOICE%"=="4" goto dryrun
if "%CHOICE%"=="5" goto gui
if "%CHOICE%"=="6" goto custom
if "%CHOICE%"=="7" goto multi
if "%CHOICE%"=="8" goto allgroups
if "%CHOICE%"=="9" goto customdir
if "%CHOICE%"=="0" goto end
goto menu

:list
"%EXE_CLI%" --list-devices
echo.
pause
goto menu

:selftest
echo [run.bat] Running --self-test (typically 1-2 minutes on first run)...
"%EXE_CLI%" --self-test
echo.
pause
goto menu

:bench
echo [run.bat] Running --benchmark 10 (pure rate, no file written)...
"%EXE_CLI%" --benchmark 10 --batch 65536
echo.
pause
goto menu

:dryrun
echo [run.bat] Running --dry-run --prefix cafe...
"%EXE_CLI%" --prefix cafe --dry-run
echo.
pause
goto menu

:gui
if not defined EXE_GUI (
    echo [run.bat] GUI binary not found next to the CLI binary.
    echo            Skipping option 5.
    pause
    goto menu
)
start "" "%EXE_GUI%"
exit /b 0

:custom
set /p ARGS=Args (e.g. --prefix cafe --device 0):
"%EXE_CLI%" %ARGS%
echo.
pause
goto menu

:multi
REM One command, two patterns: tail 8x8 OR tail 8x7. Stops at the first hit.
if not exist results mkdir results
echo [run.bat] --suffix 88888888 --suffixes 77777777 (stops at first match)
"%EXE_CLI%" --suffix 88888888 --suffixes 77777777 --result-dir results
echo.
pause
goto menu

:allgroups
REM One run, one address per group: 88888888 AND 77777777 (--all-groups).
if not exist results mkdir results
echo [run.bat] --all-groups: collect 88888888 AND 77777777 in a single run
"%EXE_CLI%" --suffix 88888888 --suffixes 77777777 --all-groups --result-dir results
echo.
pause
goto menu

:customdir
set /p RDIR=Result directory name (e.g. run_8x8_8x7):
if "%RDIR%"=="" set "RDIR=results"
if not exist "%RDIR%" mkdir "%RDIR%"
echo [run.bat] results will be written to: %RDIR%\
set /p ARGS=Extra args (e.g. --suffix 88888888 --suffixes 77777777 --all-groups):
"%EXE_CLI%" %ARGS% --result-dir "%RDIR%"
echo.
pause
goto menu

:end
endlocal
