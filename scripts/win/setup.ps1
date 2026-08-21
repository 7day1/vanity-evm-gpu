# vanity-evm-gpu on Windows 11 — one-shot setup helpers
#
# Run from PowerShell (right-click `setup.ps1` -> Run with PowerShell,
# or `powershell -ExecutionPolicy Bypass -File .\setup.ps1`). This script
# only **checks** what's installed and prints the next step; it does NOT
# silently install drivers.

$ErrorActionPreference = 'Stop'

function Test-Command($name) {
    $found = $null
    try { Get-Command $name -ErrorAction Stop | Out-Null; $found = $true }
    catch { $found = $false }
    return $found
}

Write-Host '== vanity-evm-gpu :: Windows 11 setup check ==' -ForegroundColor Cyan
Write-Host ''

# 1. Rust toolchain
Write-Host '[1/4] Rust toolchain' -ForegroundColor Yellow
if (Test-Command 'cargo') {
    $v = cargo --version
    Write-Host "  OK: $v"
} else {
    Write-Host '  Missing.  Install with:' -ForegroundColor Red
    Write-Host '    winget install Rustlang.Rustup' -ForegroundColor White
    Write-Host '  Then open a fresh PowerShell so cargo is on PATH.'
}
Write-Host ''

# 2. MSVC build tools (only needed if you build locally)
Write-Host '[2/4] MSVC build tools (only needed for local build)' -ForegroundColor Yellow
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $ver = & $vswhere -property installationVersion 2>$null
    Write-Host "  OK: Visual Studio install found (version $ver)"
} else {
    Write-Host '  Missing (or not the standard layout).' -ForegroundColor Red
    Write-Host '  Install: https://visualstudio.microsoft.com/visual-cpp-build-tools/' -ForegroundColor White
    Write-Host '  Tick "Desktop development with C++" -> MSVC v143, Windows 11 SDK.'
    Write-Host '  (Skip this step entirely if you ONLY use the prebuilt .exe artifact.)'
}
Write-Host ''

# 3. AMD Software / OpenCL ICD
Write-Host '[3/4] AMD OpenCL ICD (or another vendor ICD)' -ForegroundColor Yellow
$amdPaths = @(
    "${env:ProgramFiles}\AMD\AMD Software\amdocl.dll",
    "${env:ProgramFiles}\AMD\amdocl.dll",
    "${env:ProgramFiles}\Common Files\ATI Technologies\Shared\OpenCL\amdocl64.dll",
    "${env:ProgramFiles}\Common Files\AMD\OpenCL\amdocl64.dll"
)
$amdFound = $false
foreach ($p in $amdPaths) { if (Test-Path $p) { $amdFound = $true; break } }
if ($amdFound) {
    Write-Host "  OK: amdocl.dll detected at $p" -ForegroundColor Green
} else {
    Write-Host '  Not detected (no amdocl.dll in common AMD paths).' -ForegroundColor Red
    Write-Host '  Install one of:' -ForegroundColor White
    Write-Host '    - AMD Software: Adrenalin Edition (https://www.amd.com/en/support)  -- recommended, ships amdocl.dll'
    Write-Host '    - Standalone AMD OpenCL ICD from your GPU vendor / Khronos'
    Write-Host ''
    Write-Host '  (NVIDIA users: install CUDA Toolkit. Intel users: Intel oneAPI runtime.)'
}
Write-Host ''

# 4. The .exe itself
Write-Host '[4/4] vanity-evm-gpu .exe binary' -ForegroundColor Yellow
$candidates = @(
    ".\target\x86_64-pc-windows-msvc\release\vanity-evm-gpu.exe",
    ".\target\release\vanity-evm-gpu.exe",
    ".\vanity-evm-gpu.exe"
)
$exeFound = $false
foreach ($c in $candidates) {
    $abs = (Resolve-Path $c -ErrorAction SilentlyContinue)
    if ($abs) { Write-Host "  OK: $abs"; $exeFound = $true; break }
}
if (-not $exeFound) {
    Write-Host "  Not found in standard cargo target/ locations." -ForegroundColor Red
    Write-Host '  Either:' -ForegroundColor White
    Write-Host '    1. Download the "vanity-evm-gpu-windows-x64" artifact from the'
    Write-Host '       GitHub Actions run, unzip next to run.bat, and you are done; or'
    Write-Host '    2. Build locally:'
    Write-Host '         rustup default stable-x86_64-pc-windows-msvc'
    Write-Host '         cargo build --release'
}
Write-Host ''

Write-Host '== Done ==' -ForegroundColor Cyan
Write-Host 'Next step: double-click run.bat (in scripts\win\) and pick "List devices" first.'
