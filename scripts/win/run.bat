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
echo   0. Quit
echo ============================================================
set /p CHOICE=Pick [0-6]:
if "%CHOICE%"=="1" goto list
if "%CHOICE%"=="2" goto selftest
if "%CHOICE%"=="3" goto bench
if "%CHOICE%"=="4" goto dryrun
if "%CHOICE%"=="5" goto gui
if "%CHOICE%"=="6" goto custom
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

:end
endlocal
