# vanity-evm-gpu — 生产级收尾总结

## 本次完成的工作

### 1. 清理残留临时诊断代码（质量门禁阻塞根因）
- **`src/gpu.rs`**：删除 `dbg_mul` kernel 诊断块（约 25 行，曾尝试在 Radeon 上直接比对 `fe_mul`）与 `ref_mul_host` 参考函数（约 45 行）。
- **`tests/fe_mul_ref.rs`**：删除整个临时诊断测试文件，连带空 `tests/` 目录一并移除。该文件此前含一个运行超 60 秒未结束的大整数比对测试，会卡死 `cargo test --release`。
- 删除前先 kill 掉上一轮卡住的 `cargo test` 后台进程，避免资源争用。

### 2. 质量门禁三连全绿
| 门禁 | 结果 |
|---|---|
| `cargo fmt --check` | ✅ PASS（已 `cargo fmt` 自动格式化） |
| `cargo clippy --all-targets -- -D warnings` | ✅ 无警告 |
| `cargo test --release` | ✅ 12/12 通过，29s 完成（无 hang） |

### 3. 真机正确性复验
- `--self-test` 仍 **8/8 PASS**，自动跳过编译失败的 Radeon Pro 560X（`cvms_element_build_from_source` 错误），落到 Intel UHD 630。
- 此前已落地的 Jacobian 投影坐标 kernel + 多向量设备可靠性探针未受影响。

### 4. README 同步更新
- 默认 `--batch` 由 `4,194,304` 改为 `4,096`（Apple 核显看门狗限制，过大易 hang）。
- 新增 Radeon Pro 560X 本机不可用说明（驱动编译失败，运行时探针自动跳过）。
- self-test 示例改为实际输出（带 `probing device` / `selected reliable device`）。
- 新增设备可靠性探针说明（编译失败或计算与 CPU 参考不一致即跳过）与核显吞吐提示（Intel UHD 630 约 0.001 Mkeys/s 量级）。
- `--suffix 88888888` 示例去掉了会触发 hang 的 `--batch 4194304`，补充看门狗风险提示。

## 当前项目状态
- **功能正确性**：Jacobian kernel + 多向量探针已在真机验证（8/8 PASS）。
- **质量**：fmt / clippy / test 全绿；CI（`.github/workflows/ci.yml`）覆盖 fmt→build→test→clippy。
- **安全**：GPU 候选强制 CPU 复核；私钥零化 + 原子写 + 0o600 权限。
- **文档**：README 完整，已反映本机实测行为。

## 遗留（用户侧 P0，上传 GitHub 前需填）
- `Cargo.toml`：`repository = "https://github.com/<your-account>/vanity-evm-gpu"`
- `LICENSE`：`Copyright (c) 2026 <your name / handle>`

> 注：内核文件 `src/kernel.cl` 中曾引用的 `dbg_mul` 函数从未存在（Rust 端 `kernel_builder("dbg_mul")` 因找不到 kernel 返回 Err 被跳过），故 kernel.cl 无需改动，本身已干净。
