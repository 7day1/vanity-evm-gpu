# Changelog

> ⚠️ **版本状态说明**：当前为 `0.x` 版本，**未经长期独立安全审计**。本程序采用
> "GPU 候选强制 CPU 重新派生地址并逐位比对" 的 oracle 设计，GPU 内核 bug 最坏只
> 导致"找不到"，绝不会输出私钥/地址对不上的假靓号；但请在新设备/新驱动上先用
> `--self-test` 验证通过，再用于真实资金。使用即表示你已知晓并自担风险。

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-08-21

### Added
- EVM 靓号地址生成，自动适配 GPU（OpenCL）加速、CPU 兜底验证。
- 混合架构：GPU 暴力派生 + CPU 验证 oracle（私钥重新派生地址逐位比对才接受）。
- Jacobian 投影坐标 secp256k1 标量乘法内核，免模逆，规避 Apple 核显看门狗挂起。
- 运行时设备可靠性探针：多向量（key=1/2/3 + 随机）GPU/CPU 比对，自动跳过编译失败或计算不一致的设备。
- 私钥本地自持：原子写（tmp+rename）、`0o600` 权限、`ZeroizeOnDrop` 落盘后零化、`--redact-private-key` 打码。
- `--self-test` / `--list-devices` / `--dry-run` / `--benchmark` / `--device` / `--cpu` / `--gpu` 等 CLI。
- 12 个 CPU 侧单元测试（Keccak / EIP-55 / 私钥1→已知地址 / 前后缀解析 / 大数进位）。
- CI：`.github/workflows/ci.yml`（fmt → build → test → clippy `-D warnings`）。

### Security
- 移除了对私钥文件无效且无收益的 `mlock_file` 调用（不防 swap）。
- `private_key_hex()` 与落盘内容改用 `Zeroizing<String>`，消除明文私钥的 `String` 堆残留。
- `privkey_to_address` 中间非压缩公钥缓冲区 `arr` 显式 `zeroize()`。

### Known Limitations
- Apple Silicon（M 系列）OpenCL 已弃用/不可用，自动回退 CPU。
- 本机（Intel Mac）Radeon Pro 560X 因驱动编译限制（`cvms_element_build_from_source`）无法编译本内核，运行时自动跳过，落到可用的 Intel UHD 630 核显；核显吞吐远低于独立 GPU（约 0.001 Mkeys/s 量级）。
- GPU 内核正确性由带 GPU 机器上的 `--self-test` 保证；CI 无 GPU，仅跑 CPU 单测。

[0.1.0]: https://github.com/<your-account>/vanity-evm-gpu/releases/tag/v0.1.0
