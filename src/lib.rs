//! vanity-evm-gpu library crate.
//!
//! Shared logic for the CLI binary (`src/main.rs`) and the GUI binary
//! (`src/bin/vanity-evm-gpu-gui.rs`). Design (combines two approaches):
//! * CPU path  — trusted, audited secp256k1 + Keccak on the host (fallback / --cpu).
//! * GPU path  — OpenCL kernel brute-forces derivation at speed (auto-detected).
//! * Oracle    — every GPU candidate is re-derived on the CPU before it is
//!   accepted, so a buggy kernel can never emit a mismatched key.

pub const KERNEL_SRC: &str = include_str!("kernel.cl");

pub mod config;
pub mod cpu_worker;
pub mod crypto;
pub mod ec;
pub mod gpu;
pub mod mont;
pub mod output;
pub mod progress;

// Re-exports commonly used by front-ends (CLI, GUI).
pub use config::Pattern;
pub use cpu_worker::{run_cpu, Match as CpuMatch};
pub use gpu::{run_gpu, GpuMatch};
// Experimental Radeon multi-dispatch path (gated — see gpu.rs).
pub use gpu::radeon_self_test;
pub use output::Found;
pub use progress::{Progress, ProgressCb};
