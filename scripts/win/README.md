# scripts/win

Windows 11 helpers for `vanity-evm-gpu`.

## What is here

| File | Use |
|---|---|
| `setup.ps1` | PowerShell pre-flight: checks Rust / MSVC / AMD OpenCL ICD / `.exe` are present and prints the missing pieces. **Read-only — does not install anything for you.** |
| `run.bat` | Interactive launcher. Double-click → menu with: list devices / self-test / benchmark / dry-run / GUI / custom args. Resolves `.exe` in `target\release\`, `target\x86_64-pc-windows-msvc\release\`, or next to itself — works whether you build locally or extracted the CI artifact. |

## Recommended first-run flow

1. PowerShell: `.\setup.ps1` — confirm checks are all green (or follow the printed install commands).
2. Run `run.bat` → pick **1. List OpenCL GPU devices** → confirm `AMD Radeon RX 6750 GRE` shows up.
3. Pick **2. Self-test** → confirm `PASS — GPU kernel matches CPU reference` on the last line.
4. Pick **3. Benchmark 10** → note the Mkeys/s for your batch setting.
5. Pick **5. GUI** to open the windowed front-end.

## If you want to ship a tiny zip

After `cargo build --release`, the only files needed to run on Win11 are:

```
target\release\vanity-evm-gpu.exe
target\release\vanity-evm-gui.exe
README.md
LICENSE
```

`kernel.cl` is embedded into both binaries via `include_str!` — no separate copy needed.
