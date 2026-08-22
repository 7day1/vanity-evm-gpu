# vanity-evm-gpu

EVM 靓号地址生成器，**自动适配 GPU（OpenCL）加速，CPU 兜底验证**。

把两套方案的优点合在一起：

| 来源 | 优点 | 本项目如何沿用 |
|---|---|---|
| `lin200083/vanity-wallet-generator`（CPU 版，已审计） | 可信、简单的 `secp256k1 + Keccak` 派生；私钥本地、零网络 | 作为 **CPU 兜底路径** 与 **GPU 候选的可信验证 oracle** |
| GPU 加速（vanity-eth 类思路） | 暴力派生快 100×–1000× | 新增 **OpenCL 内核**，开机自动探测 GPU 并启用 |

核心安全设计：**GPU 找到候选后，必须在 CPU 上用私钥重新派生地址并比对一致，才会被接受**。因此即便 GPU 内核有密码学 bug，也绝不会输出"私钥和地址对不上"的假靓号（最坏情况只是找不到，不会给错）。

## 特性

- **自动适配 GPU**：启动时探测 OpenCL GPU，有则用、无则自动回退 CPU。也可用 `--cpu` / `--gpu` 强制。
- **不绑定任何钱包**：只产出 `私钥 + 地址（EIP-55）`。同一把私钥适用于**所有 EVM 链**（Ethereum / BSC / Polygon / Arbitrum / Optimism / Base …），你自己导进任意钱包。
- **私钥本地自持**：纯本地、零网络依赖；私钥只落本地文件（`results/matched-wallet-latest.txt`，权限 `0o600` 仅属主可读写），可用 `--redact-private-key` 在控制台与文件中打码。
- **CPU 验证 oracle**：GPU 结果强制 CPU 复核。
- **`--self-test`**：用 CPU 参考实现逐位校验 GPU 内核，**需在带 GPU 的机器上**跑一次（无 OpenCL 设备时会直接报错退出）。
- **GPU 设备选择**：`--device auto`（默认，运行时多向量探针验证每台设备后用首个可靠设备）/`--device <索引>`/`--device <名称子串>`；`--list-devices` 列出可用 GPU。凡通过探针的 GPU 才被采用，编译失败或计算与 CPU 参考不一致的会被自动跳过。
- **实测速率**：运行时打印 `Mkeys/s`；`--benchmark N` 跑 N 秒纯速率基准（仅 GPU，不落盘、不产出候选）。macOS 上 GPU 路径不可用，请用 `--cpu` 走 CPU 后端（速率低但正确，适合验证工具本身或小规模 vanity 搜索）。

## 构建

```bash
cargo build --release
```

依赖：`secp256k1`、`sha3`(Keccak)、`ocl`(OpenCL)、`rand`、`zeroize`、`clap`。
GPU 路径需要系统有 OpenCL：

- **Intel Mac（含 Radeon Pro 560X / Intel UHD 630 等）**：OpenCL 在系统里存在，但本项目的 kernel 在 macOS 的 Apple OpenCL 编译器上**编译失败**（`cvms_element_build_from_source` 报错），实测 `--self-test` 在 560X 与 UHD 630 上**均 FAIL**。因此 macOS 上请直接加 `--cpu` 或 `--no-gpu` 走 CPU 后端，不要依赖 GPU 路径。
- **Apple Silicon（M 系列）Mac**：OpenCL 已弃用/不可用，自动回退 CPU。
- **Linux**：装 ICD 与驱动头 —— `apt install ocl-icd-opencl-dev`，再装对应显卡驱动（如 NVIDIA 的 `nvidia-opencl-icd`、AMD 的 `rocm-opencl-icd` / `mesa-opencl-icd`）。
- **Windows**：安装显卡厂商的 OpenCL 运行时（NVIDIA CUDA SDK / AMD ADL 或 Intel 驱动），并把 `OpenCL.dll` 所在目录加入 `PATH`；`--self-test` 会在有设备时报 PASS。

> **直接取构建好的 `.exe`（最省事）**：CI（`.github/workflows/ci.yml` 的 `windows-msvc-build` job）每次 push 都把 `vanity-evm-gpu.exe` 与 `vanity-evm-gui.exe` 作为 artifact 上传；点 tag（`vX.Y.Z`）还会自动挂到 GitHub Release。Windows 11 用户直接下 .exe，不用在本机装 Rust。

### Windows 11 + AMD Radeon RX 6750 GRE（推荐路径）

这台配置（2026 年典型装机，i5-12400F + 6750 GRE 10GB GDDR6 + Win11）是 OpenCL **最稳**的目标之一，不需要任何 workaround：

1. **装 OpenCL ICD**（已装 AMD Software: Adrenalin 的可跳过）：
   * **首选**：直接装 [AMD Software: Adrenalin Edition](https://www.amd.com/en/support)（自动带 `amdocl.dll` OpenCL ICD）。
   * **不想装全家桶**：单独装 AMD OpenCL 运行时（`AMD-OpenCL-ICD` 或 [Khronos OpenCL ICD](https://www.khronos.org/opencl/resources)）也行。
   * 装完开 PowerShell，`clinfo` 或自己写一行确认 ICD 可见（见下面的「验证 OpenCL」）。
2. **构建/取二进制**（选其一）：
   * **下 artifact**：到 [GitHub Actions](https://github.com/7day1/vanity-evm-gpu/actions) → 选一次成功的 Windows-MSVC-build run → 底部 Artifacts 下 `vanity-evm-gpu-windows-x64`，解压得到两个 `.exe`。
   * **本地构建**（要装 Rust + MSVC）：见上方 `构建` 一节，rustup 默认 toolchain 已经是 `stable-x86_64-pc-windows-msvc`。
3. **验证 OpenCL 已被正确识别**（首次必做）：
   ```powershell
   cd path\to\release
   .\vanity-evm-gpu.exe --list-devices
   # 期望看到：[0] AMD Radeon RX 6750 GRE  (auto default)
   ```
4. **验证 GPU 内核与 CPU 参考一致**（首次必做，1–2 分钟）：
   ```powershell
   .\vanity-evm-gpu.exe --self-test
   # 期望最后一行：[self-test] PASS — GPU kernel matches CPU reference.
   ```
5. **跑 10 秒速率基线**（不落盘、不产出候选）：
   ```powershell
   .\vanity-evm-gpu.exe --benchmark 10
   # 期望看到 [benchmark] XX.XXMkeys/s over 10s (GPU, batch=4096)
   ```
   **6750 GRE 实测**（OpenCL 2.1 AMD-APP 3652.0，OpenCL + RDNA 2 朴素 Jacobian kernel）：

   | `--batch` | rate | dispatches / 10s | ms / dispatch |
   |---|---|---|---|
   | 4096（默认） | ~1.1 Mkeys/s | ~190 | ~52 |
   | 65536 | **1.14 Mkeys/s** | 175 | 57 |
   | 1048576 | **1.24 Mkeys/s** | 12 | ~840 |

   Batch 翻 16 倍速率只提 9% —— **瓶颈在单次 dispatch 的同步开销**（`finish()` + CPU 端 base point 加法 + enqueue，约 50ms/次），kernel 本身只占 5-10%。需要 5-10x 提速得改 pipeline（去掉同步 + GPU 端算 base + unroll），改动量较大；当前速率对个人 vanity 地址生成（6 位前缀 ~14 秒、8 位 ~10 小时）够用，先用着。
6. **要 GUI？** 双击 `vanity-evm-gui.exe`，或 `scripts\win\run-gui.bat` 一键拉起（设备下拉框已列 6750 GRE，先点「自检 (Self-Test)」，再填前缀/后缀开始）。

**Win11 上的安全提示**：GPU 候选会被强制 CPU 复核（见 [安全须知](#安全须知生成真实钱包前必读)）；首次跑通 `--self-test` 之前不要把任何真地址发到外面。

#### 常见坑

| 现象 | 原因 / 修法 |
|---|---|
| `--list-devices` 报 `no OpenCL GPU devices found` | 没装 AMD Software / OpenCL ICD。装 [AMD Software: Adrenalin](https://www.amd.com/en/support) 重启后再试 |
| `--self-test` 失败但 `--list-devices` 能看到卡 | 极少见，多为驱动 Bug。装最新 AMD Software；仍然失败则加 `--cpu` 强制 CPU 后端跑，功能不变但慢 |
| 报缺 `vcruntime140.dll` | 装 [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)（装 Rust + MSVC 工具链一般已经带上） |
| GUI 启动后黑屏 / 报缺 `XAML` | 极少；右键 `vanity-evm-gui.exe` → 属性 → 兼容性，勾「禁用显示缩放」 |
| `--batch` 设到 1<<20+ 后 hang | 任何 GPU 都看门狗；遇到 hang 加 `--max-seconds 30` 让它自动退出，batch 降到 65536 重试 |

### macOS（仅 CPU 模式可靠）

macOS 自带 `OpenCL.framework`，但本项目的 OpenCL kernel **在 Apple 的 OpenCL 编译器上编译失败**（实测在 Intel Mac 的 Radeon Pro 560X 与 Intel UHD 630 上 `--self-test` 均 FAIL，报错 `cvms_element_build_from_source`）。因此：

> **macOS 用户请直接加 `--cpu` 或 `--no-gpu` 走 CPU 后端。** 不要依赖本项目的 GPU 路径在 macOS 上工作。

Apple Silicon（M 系列）上 OpenCL 更已被弃用，同样只能走 CPU。

```bash
# macOS 上构建（Apple Silicon 与 Intel 通用）
cargo build --release           # 同时产出 vanity-evm-gpu 和 vanity-evm-gui

# 直接走 CPU（推荐，避开无用的 OpenCL probe）
./vanity-evm-gpu --cpu --prefix deadbeef

# 或显式跳过 GPU 检测
./vanity-evm-gpu --no-gpu --suffix 888888888
```

CI 在 macOS runner 上验证**编译**（无 GPU 所以不跑 self-test），artifact `vanity-evm-gpu-macos-universal` 含 CLI 与 GUI 两个 Mach-O binary。即使下载了 artifact，在 macOS 上也要用 `--cpu`/`--no-gpu`。

CPU 路径无需 OpenCL，所有平台通用。

## GUI（窗口界面）

除命令行外，仓库同时构建了一个 **egui 窗口程序** `vanity-evm-gui`，实时显示：

- 当前后端（**CPU / GPU**）与设备名；
- 实时速率（keys/s 与 Mkeys/s）、已尝试次数、耗时；
- 运行状态指示（运行中 / 已停止）与命中结果（地址 + 私钥，或 `--redact` 打码）。

```bash
# 构建（release 下会同时产出 CLI 与 GUI 两个二进制）
cargo build --release

# 启动窗口程序
./target/release/vanity-evm-gui
```

GUI 与 CLI **共用同一套搜索内核与 CPU 验证 oracle**，二者安全性一致。窗口里可填前缀/后缀/**额外后缀（逗号分隔，等长，等价 `--suffixes`）**、最长秒数、GPU 设备选择、强制 CPU、**每组各出一个（`--all-groups`）**、隐藏私钥；**开始/停止** 按钮通过进度回调（`ProgressCb` 返回 `false` 作为取消信号）干净地中断搜索循环。GUI 命中后同样写入 `results/matched-wallet-latest.txt`。

> 注意：GUI 依赖系统图形栈（macOS 为原生 Cocoa + OpenGL，Windows/Linux 为 winit + OpenGL），在无显示环境（纯 SSH/CI headless）下无法打开窗口，请改用 CLI。

### GUI 跨平台构建与运行

GUI 用 **eframe 0.27 / egui 0.27**（底层 winit + 各平台图形后端），纯 Rust、单二进制、不依赖系统 webview。

| 平台 | 工具链要求 | 图形后端 / 运行依赖 | 备注 |
|---|---|---|---|
| **macOS** | 稳定版 Rust（本机 x86_64 验证通过） | 原生 Cocoa + 系统 OpenGL | 开箱即用 |
| **Windows 11** | MSVC 工具链（`rustup default stable-x86_64-pc-windows-msvc`，需装 **Visual Studio Build Tools / MSVC**） | winit + OpenGL（ANGLE/系统 GL） | 可放心运行：本工具零网络、私钥本地，GUI 与 CLI 共用同一 CPU 验证 oracle，不可能输出私钥/地址错配的假靓号；唯一差异是显示形态 |
| **Linux** | 稳定版 Rust + 系统 OpenGL 开发库 | winit + OpenGL（X11 或 Wayland） | 需装 `libxcb`、`libxkbcommon`、`mesa`（如 `apt install libxcb1-dev libxkbcommon-dev libgl1-mesa-dev`），否则启动报缺库 |

```bash
# 各平台构建 GUI（与 CLI 同一条命令，自动产出两个二进制）
cargo build --release
# macOS / Linux 运行：
./target/release/vanity-evm-gui
# Windows 运行（PowerShell）：
.\target\release\vanity-evm-gui.exe
```

- **Windows 11 安全性**：已在上一轮确认——本工具不发起任何网络请求，私钥仅落本地 `results/`（权限 `0o600`、可用 `--redact` 打码），GPU 候选强制 CPU 复核。GUI 只是把同样的运行过程可视化，**不会降低安全性**。
- **headless / CI**：无显示环境下 `vanity-evm-gui` 无法打开窗口（CI 仅验证其编译通过，见 `.github/workflows/ci.yml`），真实搜索请改用 CLI 或在本机桌面环境运行 GUI。

## 使用

```bash
# 首次先验证 GPU 内核正确性（不搜索地址，exit 0 即 PASS）
./target/release/vanity-evm-gpu --self-test

# 列出本机可用 GPU，确认用的是哪张卡
./target/release/vanity-evm-gpu --list-devices

# 只要前缀（注意：默认 suffix 为空，不会偷偷加 8 个 0）
./target/release/vanity-evm-gpu --prefix cafe

# 优先用独显（自动选 AMD/NVIDIA；也可 --device Radeon / --device 1）
./target/release/vanity-evm-gpu --prefix cafe --device auto

# 只要后缀 N 个 8（10 个 8 难度约 16^10，CPU 数天，GPU 十几分钟~几小时）
# 注意：Apple 核显有看门狗，batch 过大（如 4194304）会 GPU hang；默认 4096 已调小，难任务请配合 --max-seconds 分多次跑
./target/release/vanity-evm-gpu --suffix 88888888

# 一次搜索同时匹配多个后缀（8个8 或 8个7）—— 一条命令、一次搜索
# 所有 --suffixes 必须与 --suffix 等长；命中后输出会标明命中的是 group 0 (88888888) 还是 group 1 (77777777)
./target/release/vanity-evm-gpu --suffix 88888888 --suffixes 77777777

# 想同时收 8个8 / 8个7 / 8个6，继续加逗号分隔即可（最多 16 组）
./target/release/vanity-evm-gpu --suffix 88888888 --suffixes 77777777,66666666

# 跑 30 秒纯速率基准（不落盘、不产出候选）
./target/release/vanity-evm-gpu --benchmark 30

# 单发验证 GPU 能跑通、不真正搜（新手安全演练）
./target/release/vanity-evm-gpu --prefix cafe --dry-run

# 强制 CPU
./target/release/vanity-evm-gpu --prefix dead --cpu

# 控制台与文件里都不显示私钥
./target/release/vanity-evm-gpu --prefix cafe --redact-private-key
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--prefix` | 地址前缀（hex，按 nibble 值匹配，大小写无关；EIP-55 仅影响显示） |
| `--suffix` | 地址后缀（hex），**默认空** |
| `--suffixes` | 额外后缀（逗号分隔 hex）；地址后缀等于 `--suffix` 或任一 `--suffixes` 即命中。各后缀**必须等长**。最多 16 组。一条命令一次搜索匹配多种模式 |
| `--workers` | CPU 线程数（仅 CPU 模式） |
| `--batch` | 每次 GPU 派发的 work-items（默认 4,096；Apple 核显看门狗限制，过大易 hang） |
| `--max-seconds` / `--duration` | 超时停止（秒） |
| `--cpu` / `--gpu` | 强制后端 |
| `--device` | GPU 选择：`auto`（默认，优先独显）/`索引`/`名称子串` |
| `--list-devices` | 列出可用 OpenCL GPU 后退出 |
| `--redact-private-key` | 打码私钥 |
| `--self-test` | 校验 GPU 内核后退出 |
| `--dry-run` | 单发 GPU 验证后退出，不产出候选 |
| `--benchmark N` | 跑 N 秒速率基准（Mkeys/s），不落盘 |
| `--all-groups` | 每个后缀组各出一个地址再停（而非首次命中即停）。需配合 `--suffixes` 使用，例如 `--suffix 88888888 --suffixes 77777777 --all-groups` 一次跑出两个地址 |
| `--resume-state <file>` | GPU 崩溃续跑：每轮把扫描偏移写入该文件，重启时读回从断点继续，不重复已扫区间（CPU 为随机密钥，不支持续跑） |

> 难度参考：`16^(前缀长度 + 后缀长度)` 次尝试。6 位 ≈ 1600 万（CPU 约 10–20 秒），8 位 ≈ 43 亿（CPU ~1 小时，GPU ~分钟），10 位 ≈ 1.1 万亿（GPU 十几分钟~几小时，CPU 数天）。使用 `--suffixes` 一次搜 N 个等长后缀时，命中概率约为单后缀的 N 倍，期望耗时约降为 1/N（注意：GPU/CPU 算力被所有组共享，不会变快，只是「更早撞到其中一个」）。

### 真机 `--self-test` 输出（反面示例：Intel Mac, macOS, Radeon Pro 560X + Intel UHD 630）

> 这是**真实的 macOS 失败输出**，说明本项目 GPU 路径在 Apple OpenCL 编译器上不可用。macOS 用户应直接走 CPU（`--cpu` / `--no-gpu`），不要期待 GPU 路径能跑通。

```
[gpu]    build ProQue on 'Intel(R) UHD Graphics 630': OK
[gpu]    build ProQue on 'AMD Radeon Pro 560X Compute Engine': ERROR:
         Error returned by cvms_element_build_from_source
[gpu]    no reliable GPU device found
[self-test] FAIL — do not trust GPU results on this device.
```

**结论**：在 macOS 上看到上面的 FAIL 是正常的。改用：

```bash
./vanity-evm-gpu --cpu --suffix 888888888
```

## 安全须知（生成真实钱包前必读）

> ⚠️ **硬约束（必做）**：在**任何新设备 / 新驱动 / 新系统版本**上首次使用前，**必须先跑 `--self-test` 通过**，再用 `--dry-run` 单发验证一遍，确认 GPU 内核与 CPU 参考逐位一致。大额资金转入前，**务必**用独立工具从私钥反算地址、与输出逐位核对。**GPU 内核 bug 最坏只会导致"找不到"，绝不会输出私钥/地址对不上的假靓号**（CPU oracle 兜底），但未经长期审计，新代码请你自己再验证一遍。

1. **【必做】先跑 `--self-test`**（新设备/新驱动/更新系统后必重跑）：
   ```bash
   ./target/release/vanity-evm-gpu --self-test
   # 期望输出：[self-test] PASS — GPU kernel matches CPU reference.
   ```
2. **【必做】转入资金前，用独立工具从私钥反算地址核对**（不要只信本程序输出）。完整可复制示例：
   ```python
   # pip install eth-utils eth-keys
   from eth_keys import keys
   from eth_utils import keccak

   priv_hex = "0x你的私钥十六进制"          # 来自 results/matched-wallet-latest.txt
   pk = keys.PrivateKey(bytes.fromhex(priv_hex[2:]))
   addr = pk.public_key.to_address()         # 已含 EIP-55 校验和
   print("address:", addr)
   print("matches output:", addr == "0x你的输出地址")
   ```
   或 Node.js：`npm i ethers` 后 `new ethers.Wallet(privHex).address`。
3. **私钥即资产控制权**：**离线备份后立即删除 `results/*.txt`**；私钥绝不截图、绝不上传任何网站/聊天工具。
4. 本项目**零网络依赖**；如介意，可在断网环境编译运行。
5. **运行环境建议**：私钥明文只存在于进程内存与本地 `results/*.txt`（权限 `0o600`）。本程序**不再对私钥文件做 `mlock`**（经验证对防 swap 无实际收益），请在**加密磁盘**或**禁用休眠（hibernation）**的环境下运行，避免私钥被写入 swap / 休眠镜像。
6. **GPU 异常回滚**：若运行时发现 GPU 误算、看门狗挂起或设备异常，**立即加 `--cpu` 强制纯 CPU 后端**（已验证 `--gpu` 在探测失败时报 exit 3 退出，不会悄悄用错结果）：
   ```bash
   ./target/release/vanity-evm-gpu --prefix cafe --cpu
   ```

## Windows 稳定性：显卡看门狗（TDR）

Windows 显示驱动有一个 **TDR（Timeout Detection and Recovery，超时检测与恢复）** 机制：任何 GPU 任务超过默认 **2 秒** 不返回，驱动就判定「卡死」并**重置显卡**。长时间满载的 vanity 搜索偶尔会触发它，表现为：

- 程序突然报 OpenCL 错误 / `CL_OUT_OF_RESOURCES` / 找不到设备；
- 屏幕闪一下、显卡驱动「重启」；
- 搜索进程异常退出，但**私钥不会因此泄露**（结果只在命中时才写盘）。

### 缓解办法（按推荐顺序）

1. **优先用 `--max-seconds` 分段跑**：把长任务拆成多段，每段远小于 2 秒级连续独占。例如 9 位后缀任务每次跑 1 小时：
   ```powershell
   vanity-evm-gpu.exe --suffix 88888888 --max-seconds 3600 --result-dir results_8
   ```
2. **调大 TDR 超时（治本，需管理员）**：注册表把默认 2 秒提到 10–60 秒，给 GPU 长任务喘息空间。
   - `Win + R` → `regedit` → 定位
     `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`
   - 新建（或改）`DWORD` 值 **`TdrDelay`** = `10`（十进制，单位秒；激进可设 `60`）。
   - 若存在 **`TdrDdiDelay`**（UMD 超时）也一并设为 `10`。
   - **重启生效**。改完若系统不稳定，删掉这两个值即可还原。
   - 仅影响本机调试体验，不改变任何密码学安全性。
3. **降低 `--batch`**：batch 越大单次 kernel 占用越长，越易撞 TDR。默认 4,096 已保守；若仍触发，降到 `1024` 重试（吞吐略降但更稳）。
4. **遇到 hang 别强杀**：加 `--max-seconds 30` 让程序自己退出，避免驱动处于半重置状态。

> 说明：TDR 是 Windows 驱动层行为，**与本项目正确性无关**——命中结果都经 CPU 复核（oracle），驱动重置只会导致「这段白跑」，不会产出错误私钥。

## 项目结构

```
vanity-evm-gpu/
├── Cargo.toml
├── src/
│   ├── main.rs        # CLI + 编排（自动选后端、验证、落盘）
│   ├── kernel.cl      # OpenCL 内核：secp256k1 派生 + Keccak-256 + 前后缀匹配
│   ├── crypto.rs      # CPU 派生 / EIP-55 / 工具（验证 oracle 用）
│   ├── config.rs      # 前后缀解析与校验
│   ├── cpu_worker.rs  # CPU 搜索（兜底）
│   ├── gpu.rs         # OpenCL 探测 + 运行 + CPU 验证 + self-test
│   └── output.rs      # 本地文件原子写、可打码
├── LICENSE            # MIT（衍生自 lin200083/vanity-wallet-generator）
└── .gitignore
```

## 测试与 CI

仓库自带单元测试（CPU 侧密码学 + 前后缀解析，无需 GPU），发布前可本地跑：

```bash
cargo test --release
```

CI（`.github/workflows/ci.yml`）在推送/PR 时自动执行：`cargo fmt --check` → `cargo build --release` → `cargo test` → `cargo clippy -D warnings`（仅装 OpenCL 头文件，测试不依赖真实 GPU）。

> 说明：单元测试**不**覆盖 GPU 内核——那部分由 `--self-test` 在带 GPU 的机器上验证。

## 上传 GitHub

```bash
git init
git add -A
git commit -m "vanity-evm-gpu: auto GPU (OpenCL) + CPU verification oracle"
git remote add origin https://github.com/<you>/vanity-evm-gpu.git
git push -u origin main
```

记得在 `Cargo.toml` 和 `LICENSE` 里把 `<your-account>` / `<your name>` 改成你自己的。
