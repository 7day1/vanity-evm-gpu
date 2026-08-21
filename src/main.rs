//! vanity-evm-gpu — EVM vanity address generator.
//!
//! Design (combines two approaches):
//! * CPU path  — trusted, audited secp256k1 + Keccak on the host (fallback / --cpu).
//! * GPU path  — OpenCL kernel brute-forces derivation at speed (auto-detected).
//! * Oracle    — every GPU candidate is re-derived on the CPU before it is
//!   accepted, so a buggy kernel can never emit a mismatched key.
//!
//! Wallet-unbound: only private key + EIP-55 address are produced; the same
//! key works on every EVM chain. Private keys stay local (atomic file write,
//! optional redaction, result files restricted to owner-only). No network
//! access

use clap::Parser;
use std::path::{Path, PathBuf};
use vanity_evm_gpu::{config, cpu_worker, gpu, output};

#[derive(Parser)]
#[command(
    name = "vanity-evm-gpu",
    about = "EVM vanity address generator with auto GPU (OpenCL) + CPU verification"
)]
struct Cli {
    /// Address prefix to match (hex, case-insensitive on nibble value).
    #[arg(long, default_value = "")]
    prefix: String,

    /// Address suffix to match (hex). Defaults to empty (no suffix).
    #[arg(long, default_value = "")]
    suffix: String,

    /// Number of CPU worker threads (CPU mode only).
    #[arg(long)]
    workers: Option<usize>,

    /// Stop after this many seconds.
    #[arg(long)]
    max_seconds: Option<u64>,

    /// Redact the private key in console output and result files.
    #[arg(long)]
    redact_private_key: bool,

    /// Force CPU mode (skip GPU detection).
    #[arg(long)]
    cpu: bool,

    /// Force GPU mode (error if no OpenCL GPU).
    #[arg(long)]
    gpu: bool,

    /// GPU work-items per batch (default 4,096). The safe value depends on
    /// the GPU; integrated display GPUs (e.g. Intel UHD 630 on macOS) need
    /// small batches to stay under the OS watchdog. Raise this on discrete GPUs.
    #[arg(long)]
    batch: Option<usize>,

    /// Directory for result files (default: ./results).
    #[arg(long)]
    result_dir: Option<PathBuf>,

    /// Validate the OpenCL kernel against the CPU reference, then exit.
    #[arg(long)]
    self_test: bool,

    /// EXPERIMENTAL: validate the Radeon multi-dispatch kernel
    /// (radeon_init_inf/radeon_step_bit/radeon_finalize_affine) against the CPU
    /// reference, then exit.
    ///
    /// **macOS-only workaround**: the Apple OpenCL 1.2 compiler (used on Intel
    /// Macs with discrete AMD GPUs such as the Radeon Pro 560X) crashes inside
    /// `cvms_element_build_from_source` when compiling the default single-pass
    /// `scalar_mul()` kernel. This multi-dispatch path sidesteps that crash
    /// by moving the 256-iteration double-and-add loop into the host. It is
    /// slower than the default Jacobian kernel and only useful on macOS.
    /// On Windows (AMD/NVIDIA/Intel Adrenalin or vendor OpenCL ICD) the
    /// default Jacobian path compiles and runs fine — do not use this flag.
    #[arg(long)]
    radeon_self_test: bool,

    /// GPU device selection: "auto" (default, prefer discrete), "list" to show
    /// available GPUs, or an index / name substring (e.g. "1", "Radeon").
    #[arg(long, default_value = "auto")]
    device: String,

    /// Stop after this many seconds of searching (convenience alias for
    /// --max-seconds).
    #[arg(long)]
    duration: Option<u64>,

    /// Prove the GPU kernel/device works on a single dispatch, then exit
    /// without emitting a (false) candidate. No file is written.
    #[arg(long)]
    dry_run: bool,

    /// Measure raw GPU keys/s for N seconds. Prints Mkeys/s, emits nothing,
    /// writes no file.
    #[arg(long)]
    benchmark: Option<u64>,

    /// List available OpenCL GPU devices and exit.
    #[arg(long)]
    list_devices: bool,
}

fn default_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn parse_device(sel: &str) -> gpu::DeviceSelection {
    if sel.eq_ignore_ascii_case("auto") {
        gpu::DeviceSelection::Auto
    } else if let Ok(i) = sel.parse::<usize>() {
        gpu::DeviceSelection::Index(i)
    } else {
        gpu::DeviceSelection::Name(sel.to_string())
    }
}

fn main() {
    let cli = Cli::parse();

    if cli.list_devices {
        let gpus = gpu::list_gpus();
        if gpus.is_empty() {
            eprintln!("[gpu] no OpenCL GPU devices found");
            std::process::exit(3);
        }
        println!("Available OpenCL GPU devices:");
        for (i, name) in gpus {
            let mark = if i == 0 { " (auto default)" } else { "" };
            println!("  [{}] {}{}", i, name, mark);
        }
        return;
    }

    let device = parse_device(&cli.device);

    if cli.self_test {
        eprintln!("[self-test] validating OpenCL kernel against CPU reference...");
        let ok = gpu::self_test(device);
        if ok {
            println!("[self-test] PASS — GPU kernel matches CPU reference.");
            return;
        } else {
            eprintln!("[self-test] FAIL — do not trust GPU results on this device.");
            std::process::exit(1);
        }
    }

    if cli.radeon_self_test {
        eprintln!(
            "[radeon-self-test] validating EXPERIMENTAL Radeon multi-dispatch kernel \
             against CPU reference..."
        );
        let ok = gpu::radeon_self_test(device);
        if ok {
            println!("[radeon-self-test] PASS — Radeon multi-dispatch kernel matches CPU.");
            return;
        } else {
            eprintln!("[radeon-self-test] FAIL — do not trust this path on this device.");
            std::process::exit(1);
        }
    }

    if let Some(secs) = cli.benchmark {
        let batch = cli.batch.unwrap_or(1 << 12);
        match gpu::benchmark(secs, batch, device) {
            Some(rate) => {
                println!(
                    "[benchmark] {:.2}Mkeys/s over {}s (GPU, batch={})",
                    rate / 1e6,
                    secs,
                    batch
                );
                return;
            }
            None => {
                eprintln!("error: no OpenCL GPU device available for benchmark");
                std::process::exit(3);
            }
        }
    }

    let pattern = match config::Pattern::parse(&cli.prefix, &cli.suffix) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(2);
        }
    };

    let result_dir = cli.result_dir.unwrap_or_else(|| PathBuf::from("results"));
    let max_seconds = cli.max_seconds.or(cli.duration);

    let backend = if cli.cpu {
        Backend::Cpu
    } else if cli.gpu {
        if !gpu::gpu_available() {
            eprintln!("error: --gpu requested but no OpenCL GPU device found");
            std::process::exit(3);
        }
        Backend::Gpu
    } else if gpu::gpu_available() {
        Backend::Gpu
    } else {
        eprintln!("[info] no OpenCL GPU found — falling back to CPU.");
        Backend::Cpu
    };

    if cli.dry_run {
        println!(
            "target: prefix='{}' suffix='{}'  (expected ~{:.0} attempts)",
            cli.prefix,
            cli.suffix,
            pattern.expected_attempts()
        );
        println!(
            "backend: {} (dry-run — single GPU dispatch, no candidate emitted)",
            backend.label()
        );
        match backend {
            Backend::Gpu => {
                let batch = cli.batch.unwrap_or(1 << 12);
                match gpu::run_gpu(&pattern, Some(1), batch, device, true, None) {
                    Some(_) => {
                        println!("[dry-run] OK — GPU kernel returned a CPU-verified candidate.")
                    }
                    None => println!("[dry-run] OK — GPU dispatch completed (no match in sample)."),
                }
            }
            Backend::Cpu => {
                println!("[dry-run] CPU path is always available; nothing to probe.")
            }
        }
        return;
    }

    println!(
        "target: prefix='{}' suffix='{}'  (expected ~{:.0} attempts)",
        cli.prefix,
        cli.suffix,
        pattern.expected_attempts()
    );
    println!("backend: {}", backend.label());

    match backend {
        Backend::Gpu => {
            let batch = cli.batch.unwrap_or(1 << 12);
            match gpu::run_gpu(&pattern, max_seconds, batch, device, false, None) {
                Some(m) => finish(m.priv32, m.addr, &result_dir, cli.redact_private_key),
                None => println!("no match found (stopped)."),
            }
        }
        Backend::Cpu => {
            let workers = cli.workers.unwrap_or_else(default_workers);
            println!("[cpu] workers={}", workers);
            match cpu_worker::run_cpu(&pattern, max_seconds, workers, None) {
                Some(m) => finish(m.priv32, m.addr, &result_dir, cli.redact_private_key),
                None => println!("no match found (stopped)."),
            }
        }
    }
}

enum Backend {
    Cpu,
    Gpu,
}
impl Backend {
    fn label(&self) -> &'static str {
        match self {
            Backend::Cpu => "CPU (secp256k1 + Keccak, host)",
            Backend::Gpu => "GPU (OpenCL kernel) + CPU verification oracle",
        }
    }
}

fn finish(priv32: [u8; 32], addr: [u8; 20], result_dir: &Path, redact: bool) {
    let found = output::Found {
        priv_reduced: priv32,
        raw_addr: addr,
    };
    let address = found.address_eip55();
    let key_hex = found.private_key_hex();

    if redact {
        println!("Address: 0x{}", address);
        println!("PrivateKey: [redacted by --redact-private-key]");
    } else {
        println!("Address: 0x{}", address);
        println!("PrivateKey: 0x{}", key_hex.as_str());
    }

    if let Err(e) = output::write_result(result_dir, &found, redact) {
        eprintln!("warning: failed to write result file: {}", e);
    } else {
        println!("saved: {}/matched-wallet-latest.txt", result_dir.display());
    }

    println!(
        "\n[security] Verify before funding: re-derive the address from the private key\n\
         with ethers/web3.py/alloy and confirm it matches 0x{} (EIP-55).",
        address
    );
    if !redact {
        println!("[security] Back up the private key offline, then delete the result file.");
    }
}
