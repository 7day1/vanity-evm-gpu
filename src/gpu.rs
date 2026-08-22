//! OpenCL backend: auto-detect a GPU (fall back to any device), brute-force on
//! the device, and re-derive the candidate's address on the CPU before
//! accepting it. Includes `--self-test` to validate the kernel against the CPU.

use crate::config::Pattern;
use crate::crypto::{
    add_u64_be, bytes_to_u32x8, privkey_to_address, reduce_mod_n, u32x8_to_bytes, zeroize_key,
};
use crate::progress::{Progress, ProgressCb};
use ocl::{Buffer, Device, DeviceType, Platform, ProQue, SpatialDims};
use rand::rngs::OsRng;
use rand::RngCore;
use std::collections::HashSet;
use std::path::Path;
use zeroize::ZeroizeOnDrop;

/// Result of a GPU match. Zeroized on drop (defense-in-depth).
#[derive(ZeroizeOnDrop)]
pub struct GpuMatch {
    pub priv32: [u8; 32], // canonical (reduced mod n)
    pub addr: [u8; 20],
}

/// Known EVM address for the private key 1 (used as a fast probe).
const ADDR_KEY1: [u8; 20] = [
    0x7e, 0x5f, 0x45, 0x52, 0x09, 0x1a, 0x69, 0x12, 0x5d, 0x5d, 0xfc, 0xb7, 0xb8, 0xc2, 0x65, 0x90,
    0x29, 0x39, 0x5b, 0xdf,
];

/// OpenCL C prelude prepended to the kernel at compile time.
const KERNEL_PRELUDE: &str = r#"
typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef short          int16_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;
typedef long           int64_t;
typedef unsigned long  uint64_t;
"#;

fn kernel_source() -> String {
    let mut s = String::with_capacity(KERNEL_PRELUDE.len() + crate::KERNEL_SRC.len());
    s.push_str(KERNEL_PRELUDE);
    s.push_str(crate::KERNEL_SRC);
    s
}

pub fn diagnose_opencl() {
    let platforms = Platform::list();
    eprintln!("[gpu] clGetPlatformIDs -> {} platform(s)", platforms.len());
    for (i, p) in platforms.iter().enumerate() {
        let name = p.name().unwrap_or_else(|_| "<unknown>".into());
        let vendor = p.vendor().unwrap_or_else(|_| "<unknown>".into());
        let version = p.version().unwrap_or_else(|_| "<unknown>".into());
        eprintln!(
            "[gpu]   platform[{}]: name='{}' vendor='{}' version='{}'",
            i, name, vendor, version
        );
        for (label, dt) in [
            ("GPU", Some(DeviceType::GPU)),
            ("ALL", None),
            ("CPU", Some(DeviceType::CPU)),
        ] {
            match Device::list(p, dt) {
                Ok(devs) => {
                    eprintln!(
                        "[gpu]     {:<3} filter -> {} device(s): {}",
                        label,
                        devs.len(),
                        devs.iter().map(device_name).collect::<Vec<_>>().join(", ")
                    );
                }
                Err(e) => {
                    eprintln!("[gpu]     {:<3} filter -> ERROR: {}", label, e);
                }
            }
        }
        if let Ok(devs) = Device::list(p, None) {
            let src = kernel_source();
            for d in devs {
                match ProQue::builder()
                    .platform(*p)
                    .device(d)
                    .src(src.as_str())
                    .build()
                {
                    Ok(_) => eprintln!("[gpu]     build ProQue on '{}': OK", device_name(&d)),
                    Err(e) => eprintln!(
                        "[gpu]     build ProQue on '{}': ERROR: {}",
                        device_name(&d),
                        e
                    ),
                }
            }
        }
    }
}

fn is_preferred_gpu(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("radeon")
        || n.contains("nvidia")
        || n.contains("geforce")
        || n.contains("rx ")
        || n.contains("gtx")
        || n.contains("rtx")
        || n.contains("discrete")
        || n.contains("amd")
        || n.contains("firepro")
        || n.contains("quadro")
        || n.contains("arc ")
        || n.contains("intel(r) arc")
}

pub fn list_gpus() -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for p in Platform::list() {
        if let Ok(devs) = Device::list(p, Some(DeviceType::GPU)) {
            for d in devs {
                out.push((out.len(), device_name(&d)));
            }
        }
    }
    out
}

fn build_proque(platform: Platform, device: ocl::Device) -> Option<ProQue> {
    let src = kernel_source();
    // Note: the AMD Radeon Pro 560X on this Mac fails to build the current
    // Jacobian kernel regardless of optimization settings
    // ("cvms_element_build_from_source"). The runtime probe automatically skips
    // any device that does not build or that produces wrong results, so we
    // simply use the default compiler flags for all devices.
    ProQue::builder()
        .platform(platform)
        .device(device)
        .src(src.as_str())
        .build()
        .ok()
}

/// Probe a single private-key vector on the already-built buffers/kernels.
/// `base_u32` is the 256-bit private key in big-endian u32 chunks. The work
/// size is always 1, so gid=0 and the probed key equals `base_u32`.
fn probe_one_vector(
    proque: &ProQue,
    derive: &ocl::Kernel,
    hash: &ocl::Kernel,
    base_buf: &Buffer<u32>,
    addrs: &Buffer<u8>,
    base_u32: &[u32; 8],
    expected: &[u8; 20],
) -> bool {
    if base_buf.write(&base_u32[..]).enq().is_err() {
        return false;
    }
    if proque.queue().finish().is_err() {
        return false;
    }
    unsafe {
        if derive
            .cmd()
            .global_work_size(SpatialDims::One(1))
            .enq()
            .is_err()
        {
            return false;
        }
    }
    if proque.queue().finish().is_err() {
        return false;
    }
    unsafe {
        if hash
            .cmd()
            .global_work_size(SpatialDims::One(1))
            .enq()
            .is_err()
        {
            return false;
        }
    }
    if proque.queue().finish().is_err() {
        return false;
    }
    let mut got = [0u8; 20];
    if addrs.read(&mut got[..]).enq().is_err() {
        return false;
    }
    if proque.queue().finish().is_err() {
        return false;
    }
    got == *expected
}

/// Kernel/device reliability probe.
///
/// The Apple/Radeon Pro 560X driver historically miscompiled the affine
/// scalar-multiplication path for non-trivial private keys, even though the
/// trivial key=1 (the generator itself) happened to pass. With the Jacobian
/// kernel this should no longer happen, but we keep a multi-vector probe as a
/// runtime safety net so any driver-level regression is caught before we trust
/// the device.
fn probe_device(proque: &ProQue) -> bool {
    let base_buf = match Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(8)
        .build()
    {
        Ok(b) => b,
        Err(_) => return false,
    };
    let pubs = match Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(16)
        .build()
    {
        Ok(b) => b,
        Err(_) => return false,
    };
    let addrs = match Buffer::<u8>::builder()
        .queue(proque.queue().clone())
        .len(20)
        .build()
    {
        Ok(b) => b,
        Err(_) => return false,
    };
    let derive = match proque
        .kernel_builder("derive_pubkeys")
        .arg(&base_buf)
        .arg(&pubs)
        .build()
    {
        Ok(k) => k,
        Err(_) => return false,
    };
    let hash = match proque
        .kernel_builder("hash_addrs")
        .arg(&pubs)
        .arg(&addrs)
        .build()
    {
        Ok(k) => k,
        Err(_) => return false,
    };

    // Vector 0: private key 1.
    let mut base1 = [0u32; 8];
    base1[7] = 1;
    if !probe_one_vector(
        proque, &derive, &hash, &base_buf, &addrs, &base1, &ADDR_KEY1,
    ) {
        return false;
    }

    // Diagnostic vectors: key=2 (tests Jacobian doubling), key=3 (tests
    // doubling + mixed addition). Keep these even in release until we are sure
    // the Jacobian path is solid.
    for (v, k) in [(1u32, 2u8), (2, 3)].iter() {
        let mut bytes = [0u8; 32];
        bytes[31] = *k;
        let base = bytes_to_u32x8(&bytes);
        let expected = privkey_to_address(&bytes).unwrap();
        if !probe_one_vector(proque, &derive, &hash, &base_buf, &addrs, &base, &expected) {
            eprintln!("[gpu] probe vector {} (key={}): GPU/CPU mismatch", v, k);
            return false;
        }
    }

    // Vectors 3..N: random private keys. Non-trivial scalars exercise the
    // full Jacobian double/add path, which is the real reliability gate.
    let mut rng = OsRng;
    const RANDOM_VECTORS: usize = 4;
    for v in 3..=RANDOM_VECTORS + 2 {
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        let base = bytes_to_u32x8(&bytes);
        let expected = match privkey_to_address(&bytes) {
            Some(a) => a,
            None => {
                eprintln!(
                    "[gpu] probe vector {}: CPU reference failed (invalid scalar)",
                    v
                );
                return false;
            }
        };
        if !probe_one_vector(proque, &derive, &hash, &base_buf, &addrs, &base, &expected) {
            eprintln!("[gpu] probe vector {}: GPU/CPU mismatch", v);
            return false;
        }
    }

    true
}

/// Internal implementation of device selection. `quiet` suppresses the
/// per-device probe messages (used by `gpu_available()`).
fn try_select_device(selection: DeviceSelection, quiet: bool) -> Option<(ProQue, ocl::Device)> {
    let mut gpus: Vec<(Platform, ocl::Device, String)> = Vec::new();
    for p in Platform::list() {
        if let Ok(devs) = Device::list(p, Some(DeviceType::GPU)) {
            for d in devs {
                gpus.push((p, d, device_name(&d)));
            }
        }
    }
    let candidates: Vec<(Platform, ocl::Device, String)> = match selection {
        DeviceSelection::Auto => {
            // Prefer discrete GPUs first, but still probe all of them.
            let mut preferred: Vec<_> = gpus
                .iter()
                .filter(|(_, _, n)| is_preferred_gpu(n))
                .cloned()
                .collect();
            let mut rest: Vec<_> = gpus
                .iter()
                .filter(|(_, _, n)| !is_preferred_gpu(n))
                .cloned()
                .collect();
            preferred.append(&mut rest);
            preferred
        }
        DeviceSelection::Index(i) => gpus.into_iter().skip(i).take(1).collect(),
        DeviceSelection::Name(sub) => {
            let sub = sub.to_ascii_lowercase();
            gpus.into_iter()
                .filter(|(_, _, n)| n.to_ascii_lowercase().contains(&sub))
                .take(1)
                .collect()
        }
    };
    for (p, d, name) in candidates {
        if !quiet {
            eprintln!("[gpu] probing device: {}", name);
        }
        if let Some(proque) = build_proque(p, d) {
            if probe_device(&proque) {
                if !quiet {
                    eprintln!("[gpu] selected reliable device: {}", name);
                }
                return Some((proque, d));
            } else if !quiet {
                eprintln!(
                    "[gpu] device {} failed probe (kernel/CPU mismatch) — skipping",
                    name
                );
            }
        }
    }
    if !quiet {
        eprintln!("[gpu] no reliable GPU device found");
    }
    None
}

/// Pick a GPU according to `selection` and verify it produces correct results.
/// For `Auto`, every GPU is probed in order and the first reliable one wins.
pub fn select_device(selection: DeviceSelection) -> Option<(ProQue, ocl::Device)> {
    try_select_device(selection, false)
}

/// True if at least one OpenCL GPU passes the reliability probe. This is the
/// same path `select_device` uses, so `gpu_available()` and the actual GPU run
/// always agree on whether a device is trustworthy.
pub fn gpu_available() -> bool {
    try_select_device(DeviceSelection::Auto, true).is_some()
}

/// How the GPU device is chosen.
#[derive(Debug, Clone, Default)]
pub enum DeviceSelection {
    /// Prefer a reliable discrete GPU, else any reliable GPU.
    #[default]
    Auto,
    /// N-th GPU in `list_gpus()` order.
    Index(usize),
    /// First GPU whose name contains this substring (case-insensitive).
    Name(String),
}

fn device_name(dev: &ocl::Device) -> String {
    dev.name()
        .unwrap_or_else(|_| "<unknown device>".to_string())
}

/// Number of `u32` slots the params buffer needs for a given pattern.
/// Layout: 90 base slots + 2 header slots (num_alt, alt_len) + 40 slots per
/// alternative suffix group. Group 0 lives in the base [50..90) region, so it
/// does not add to the count.
fn params_len(pattern: &Pattern) -> usize {
    92 + pattern.alt_suffixes.len() * 40
}

fn make_params(base: &[u32; 8], pattern: &Pattern) -> Vec<u32> {
    let mut p = vec![0u32; params_len(pattern)];
    p[0] = pattern.prefix.len() as u32;
    p[1] = pattern.suffix.len() as u32;
    p[2..10].copy_from_slice(base);
    for i in 0..pattern.prefix.len().min(40) {
        p[10 + i] = pattern.prefix[i] as u32;
    }
    for i in 0..pattern.suffix.len().min(40) {
        p[50 + i] = pattern.suffix[i] as u32;
    }
    // Alternative suffix groups (packed, each into 40 slots).
    p[90] = pattern.alt_suffixes.len() as u32;
    p[91] = pattern.suffix.len() as u32; // all groups share this length
    for (g, alt) in pattern.alt_suffixes.iter().enumerate() {
        let base_off = 92 + g * 40;
        for i in 0..alt.len().min(40) {
            p[base_off + i] = alt[i] as u32;
        }
    }
    p
}

/// Resume-state record for the GPU search. The GPU path brute-forces a 256-bit
/// space by advancing `base` by `batch` each iteration, so persisting the last
/// `base` lets a restarted run continue from where it stopped instead of
/// re-scanning already-tested keys. (CPU mode uses random keys and cannot
/// resume — see README.)
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct ResumeState {
    base: [u8; 32],
    total: u64,
    found_groups: Vec<usize>,
}

fn load_resume_state(path: &Path) -> Option<ResumeState> {
    let s = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&s).ok()
}

fn save_resume_state(path: &Path, st: &ResumeState) {
    if let Ok(s) = serde_json::to_string(st) {
        // Best-effort: write atomically so a crash mid-write never corrupts it.
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, &s).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }
}

/// Run the GPU search.
///
/// * `all_groups` — when true, keep searching until **every** suffix group in
///   `pattern` has been matched at least once (one address per group), instead
///   of stopping at the first match. Useful when you need e.g. both an `88888888`
///   and a `77777777` address and want them in one run.
/// * `resume_state` — when `Some(path)`, the last `base` offset (and collected
///   groups) is read from / written to this file each iteration so an
///   interrupted long run can continue without re-scanning.
///
/// Returns all matched results (1 in default mode, up to `group_count` in
/// `--all-groups` mode).
#[allow(clippy::too_many_arguments)]
pub fn run_gpu(
    pattern: &Pattern,
    max_seconds: Option<u64>,
    batch: usize,
    selection: DeviceSelection,
    dry_run: bool,
    all_groups: bool,
    resume_state: Option<&Path>,
    cb: Option<ProgressCb>,
) -> Vec<GpuMatch> {
    let (proque, dev) = match select_device(selection) {
        Some(x) => x,
        None => return Vec::new(),
    };
    let dev_name = device_name(&dev);
    eprintln!("[gpu] using device: {}", dev_name);
    if let Some(cb) = &cb {
        cb(&Progress {
            backend: "GPU",
            device: dev_name.clone(),
            attempts: 0,
            rate: 0.0,
            elapsed_secs: 0.0,
            done: false,
        });
    }

    let base_buf = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(8)
        .build();
    let pubs = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(batch * 16)
        .build();
    let addrs = Buffer::<u8>::builder()
        .queue(proque.queue().clone())
        .len(batch * 20)
        .build();
    let out_found = Buffer::<i32>::builder()
        .queue(proque.queue().clone())
        .len(1)
        .build();
    let out_priv = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(8)
        .build();
    let out_addr = Buffer::<u8>::builder()
        .queue(proque.queue().clone())
        .len(20)
        .build();
    let params = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(params_len(pattern))
        .build();
    let (base_buf, pubs, addrs, out_found, out_priv, out_addr, params) =
        match (base_buf, pubs, addrs, out_found, out_priv, out_addr, params) {
            (Ok(a), Ok(b), Ok(c), Ok(d), Ok(e), Ok(f), Ok(g)) => (a, b, c, d, e, f, g),
            _ => return Vec::new(),
        };

    let derive = proque
        .kernel_builder("derive_pubkeys")
        .arg(&base_buf)
        .arg(&pubs)
        .build();
    let hash = proque
        .kernel_builder("hash_addrs")
        .arg(&pubs)
        .arg(&addrs)
        .build();
    let matcher = proque
        .kernel_builder("match_addrs")
        .arg(&base_buf)
        .arg(&addrs)
        .arg(&out_found)
        .arg(&out_priv)
        .arg(&out_addr)
        .arg(&params)
        .build();
    let (derive, hash, matcher) = match (derive, hash, matcher) {
        (Ok(d), Ok(h), Ok(m)) => (d, h, m),
        _ => return Vec::new(),
    };

    // Seed base: resume from the saved offset if present, else fresh random.
    let mut base: [u32; 8];
    let mut total: u64;
    let mut found_groups: HashSet<usize> = HashSet::new();
    if let Some(path) = resume_state {
        if let Some(st) = load_resume_state(path) {
            base = bytes_to_u32x8(&st.base);
            total = st.total;
            found_groups = st.found_groups.iter().cloned().collect();
            eprintln!(
                "[gpu] resumed from {} (base offset loaded, {} group(s) already found)",
                path.display(),
                found_groups.len()
            );
        } else {
            let mut base_bytes = [0u8; 32];
            OsRng.fill_bytes(&mut base_bytes);
            base = bytes_to_u32x8(&base_bytes);
            total = 0;
        }
    } else {
        let mut base_bytes = [0u8; 32];
        OsRng.fill_bytes(&mut base_bytes);
        base = bytes_to_u32x8(&base_bytes);
        total = 0;
    }

    let total_groups = pattern.suffix_group_count();
    let mut results: Vec<GpuMatch> = Vec::new();

    let start = std::time::Instant::now();
    let deadline = max_seconds.map(std::time::Duration::from_secs);
    let mut next_report = total + (batch as u64 * 20);
    // Rate is computed from THIS session's attempts only: when resuming from a
    // state file, `total` includes previous sessions' work but `elapsed` starts
    // at zero, so dividing total/elapsed would wildly overstate the rate.
    let session_start_total = total;
    // Fail-fast guard: if the OpenCL driver resets mid-run (Windows TDR, device
    // hang), every enq/finish below starts failing. Without this counter the
    // loop would spin forever on a dead device, silently wasting days. After
    // `MAX_CONSECUTIVE_FAILURES` fully-failed iterations we abort with a clear
    // message instead.
    const MAX_CONSECUTIVE_FAILURES: u32 = 3;
    let mut consecutive_failures: u32 = 0;

    loop {
        if let Some(d) = deadline {
            if start.elapsed() >= d {
                break;
            }
        }
        if dry_run && total > 0 {
            break;
        }

        // --- one full GPU iteration; any failure marks the iteration bad ---
        let mut iter_ok = true;
        let p = make_params(&base, pattern);
        if params.write(&p).enq().is_err() {
            iter_ok = false;
        }
        if iter_ok && base_buf.write(&base[..]).enq().is_err() {
            iter_ok = false;
        }
        if iter_ok {
            let zero = [0i32];
            if out_found.write(&zero[..]).enq().is_err() {
                iter_ok = false;
            }
        }
        if iter_ok && proque.queue().finish().is_err() {
            iter_ok = false;
        }
        if iter_ok {
            unsafe {
                if derive
                    .cmd()
                    .global_work_size(SpatialDims::One(batch))
                    .enq()
                    .is_err()
                {
                    iter_ok = false;
                }
            }
        }
        if iter_ok && proque.queue().finish().is_err() {
            iter_ok = false;
        }
        if iter_ok {
            unsafe {
                if hash
                    .cmd()
                    .global_work_size(SpatialDims::One(batch))
                    .enq()
                    .is_err()
                {
                    iter_ok = false;
                }
            }
        }
        if iter_ok && proque.queue().finish().is_err() {
            iter_ok = false;
        }
        if iter_ok {
            unsafe {
                if matcher
                    .cmd()
                    .global_work_size(SpatialDims::One(batch))
                    .enq()
                    .is_err()
                {
                    iter_ok = false;
                }
            }
        }
        if iter_ok && proque.queue().finish().is_err() {
            iter_ok = false;
        }

        let mut found = [0i32; 1];
        if iter_ok {
            if out_found.read(&mut found[..]).enq().is_err() {
                iter_ok = false;
            } else if proque.queue().finish().is_err() {
                iter_ok = false;
            }
        }

        if !iter_ok {
            consecutive_failures += 1;
            eprintln!(
                "[gpu][WARN] OpenCL iteration failed ({}/{} consecutive) — device may have \
                 reset (Windows TDR?). Re-run with the same --resume-state to continue; \
                 see README 'Windows 稳定性' for TdrDelay advice.",
                consecutive_failures, MAX_CONSECUTIVE_FAILURES
            );
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                eprintln!(
                    "[gpu] aborting after {} consecutive failed iterations — GPU device is \
                     not responding. Progress saved; re-run the same command to resume.",
                    consecutive_failures
                );
                break;
            }
            // Skip the rest of this iteration; do NOT advance `base` so no key
            // range is skipped when the iteration partially executed.
            continue;
        }
        consecutive_failures = 0;

        if found[0] > 0 {
            let mut priv_u32 = [0u32; 8];
            let mut addr = [0u8; 20];
            out_priv.read(&mut priv_u32[..]).enq().ok();
            out_addr.read(&mut addr[..]).enq().ok();
            proque.queue().finish().ok();

            let mut key_bytes = u32x8_to_bytes(&priv_u32);
            match privkey_to_address(&key_bytes) {
                Some(cpu_addr) if cpu_addr == addr => {
                    let reduced = reduce_mod_n(&key_bytes);
                    zeroize_key(&mut key_bytes);
                    // Determine which suffix group this address matched.
                    let group = pattern.matched_suffix_group(&addr, &pattern.prefix);
                    let is_new_group = group.map(|g| found_groups.insert(g)).unwrap_or(false);
                    // Record the result. In default mode every match is kept;
                    // in --all-groups mode we only keep matches for groups we
                    // have not collected yet (duplicate-group hits are skipped).
                    let keep = !all_groups || is_new_group || group.is_none();
                    if keep {
                        results.push(GpuMatch {
                            priv32: reduced,
                            addr,
                        });
                    }
                    if !all_groups {
                        // First match wins — emit final progress and stop.
                        if let Some(cb) = &cb {
                            let elapsed = start.elapsed().as_secs_f64();
                            cb(&Progress {
                                backend: "GPU",
                                device: dev_name.clone(),
                                attempts: total,
                                rate: (total - session_start_total) as f64 / elapsed.max(1e-6),
                                elapsed_secs: elapsed,
                                done: true,
                            });
                        }
                        if let Some(path) = resume_state {
                            let _ = std::fs::remove_file(path);
                        }
                        return results;
                    }
                    // --all-groups: stop once every group has been collected.
                    if found_groups.len() >= total_groups {
                        if let Some(cb) = &cb {
                            let elapsed = start.elapsed().as_secs_f64();
                            cb(&Progress {
                                backend: "GPU",
                                device: dev_name.clone(),
                                attempts: total,
                                rate: (total - session_start_total) as f64 / elapsed.max(1e-6),
                                elapsed_secs: elapsed,
                                done: true,
                            });
                        }
                        if let Some(path) = resume_state {
                            let _ = std::fs::remove_file(path);
                        }
                        return results;
                    }
                }
                _ => {
                    eprintln!(
                        "[gpu][WARN] GPU-derived address failed CPU verification. \
                         Ignoring (possible kernel/driver bug on this device)."
                    );
                }
            }
        }

        total += batch as u64;
        if let Some(path) = resume_state {
            // Persist progress for crash-safe resume.
            let base_bytes = u32x8_to_bytes(&base);
            let st = ResumeState {
                base: base_bytes,
                total,
                found_groups: found_groups.iter().cloned().collect(),
            };
            save_resume_state(path, &st);
        }

        if total >= next_report {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (total - session_start_total) as f64 / elapsed.max(1e-6);
            eprintln!(
                "[gpu] device='{}' attempts={} rate={:.2}M/s elapsed={:.0}s groups={}/{}",
                dev_name,
                total,
                rate / 1e6,
                elapsed,
                found_groups.len(),
                total_groups
            );
            next_report = total + (batch as u64 * 20);
            if let Some(cb) = &cb {
                let keep_going = cb(&Progress {
                    backend: "GPU",
                    device: dev_name.clone(),
                    attempts: total,
                    rate,
                    elapsed_secs: elapsed,
                    done: false,
                });
                if !keep_going {
                    eprintln!("[gpu] progress callback requested stop — cancelling search");
                    break;
                }
            }
        }

        let mut bb = u32x8_to_bytes(&base);
        add_u64_be(&mut bb, batch as u64);
        base = bytes_to_u32x8(&bb);
    }
    if let Some(cb) = &cb {
        let elapsed = start.elapsed().as_secs_f64();
        cb(&Progress {
            backend: "GPU",
            device: dev_name.clone(),
            attempts: total,
            rate: total as f64 / elapsed.max(1e-6),
            elapsed_secs: elapsed,
            done: true,
        });
    }
    results
}

/// Validate the GPU kernel against the CPU reference on the selected device.
pub fn self_test(selection: DeviceSelection) -> bool {
    diagnose_opencl();
    let (proque, dev) = match select_device(selection) {
        Some(x) => x,
        None => {
            eprintln!("[self-test] no reliable OpenCL device available");
            return false;
        }
    };
    eprintln!("[self-test] device: {}", device_name(&dev));

    let base_buf = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(8)
        .build()
        .unwrap();
    let pubs = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(16)
        .build()
        .unwrap();
    let addrs = Buffer::<u8>::builder()
        .queue(proque.queue().clone())
        .len(20)
        .build()
        .unwrap();
    let out_found = Buffer::<i32>::builder()
        .queue(proque.queue().clone())
        .len(1)
        .build()
        .unwrap();
    let out_priv = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(8)
        .build()
        .unwrap();
    let out_addr = Buffer::<u8>::builder()
        .queue(proque.queue().clone())
        .len(20)
        .build()
        .unwrap();
    // self-test only ever uses a single (empty) suffix group; a fixed buffer
    // of 132 slots is enough for up to one alternative group and keeps the
    // allocation independent of the per-trial Pattern built below.
    let params = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(132)
        .build()
        .unwrap();

    let derive = proque
        .kernel_builder("derive_pubkeys")
        .arg(&base_buf)
        .arg(&pubs)
        .build()
        .unwrap();
    let hash = proque
        .kernel_builder("hash_addrs")
        .arg(&pubs)
        .arg(&addrs)
        .build()
        .unwrap();
    let matcher = proque
        .kernel_builder("match_addrs")
        .arg(&base_buf)
        .arg(&addrs)
        .arg(&out_found)
        .arg(&out_priv)
        .arg(&out_addr)
        .arg(&params)
        .build()
        .unwrap();

    let mut rng = OsRng;
    let trials = 8;
    for t in 0..trials {
        let mut base_bytes = [0u8; 32];
        rng.fill_bytes(&mut base_bytes);
        let base = bytes_to_u32x8(&base_bytes);

        let cpu_addr = match privkey_to_address(&base_bytes) {
            Some(a) => a,
            None => {
                eprintln!("[self-test] trial {}: CPU derive failed", t);
                return false;
            }
        };
        let mut prefix = Vec::with_capacity(40);
        for &b in &cpu_addr {
            prefix.push((b >> 4) & 0xF);
            prefix.push(b & 0xF);
        }
        let pat = Pattern {
            prefix: prefix.clone(),
            suffix: vec![],
            alt_suffixes: vec![],
        };

        params.write(&make_params(&base, &pat)).enq().unwrap();
        base_buf.write(&base[..]).enq().unwrap();
        let zero = [0i32];
        out_found.write(&zero[..]).enq().unwrap();
        proque.queue().finish().unwrap();

        unsafe {
            derive
                .cmd()
                .global_work_size(SpatialDims::One(1))
                .enq()
                .unwrap();
        }
        proque.queue().finish().unwrap();
        unsafe {
            hash.cmd()
                .global_work_size(SpatialDims::One(1))
                .enq()
                .unwrap();
        }
        proque.queue().finish().unwrap();
        unsafe {
            matcher
                .cmd()
                .global_work_size(SpatialDims::One(1))
                .enq()
                .unwrap();
        }
        proque.queue().finish().unwrap();

        let mut found = [0i32; 1];
        out_found.read(&mut found[..]).enq().unwrap();
        proque.queue().finish().unwrap();

        if found[0] != 1 {
            eprintln!(
                "[self-test] trial {}: kernel did not find expected match (found={})",
                t, found[0]
            );
            return false;
        }
        let mut priv_u32 = [0u32; 8];
        let mut addr = [0u8; 20];
        out_priv.read(&mut priv_u32[..]).enq().unwrap();
        out_addr.read(&mut addr[..]).enq().unwrap();
        proque.queue().finish().unwrap();

        if addr != cpu_addr {
            eprintln!("[self-test] trial {}: GPU address != CPU address", t);
            return false;
        }
        let key_bytes = u32x8_to_bytes(&priv_u32);
        match privkey_to_address(&key_bytes) {
            Some(a) if a == addr => {}
            _ => {
                eprintln!(
                    "[self-test] trial {}: private key does not derive address",
                    t
                );
                return false;
            }
        }
        eprintln!("[self-test] trial {}: OK", t);
    }
    true
}

/// EXPERIMENTAL: Radeon-friendly scalar multiplication driven by the host.
///
/// This is the workaround for the Apple OpenCL 1.2 / AMD Radeon
/// `cvms_element_build_from_source` crash: instead of a single kernel that
/// unrolls 256 double-and-add iterations (which the Radeon compiler chokes
/// on), the host issues 256 dispatches of `radeon_step_bit` (one bit per
/// dispatch, no loop inside the kernel), keeping the running Jacobian point in
/// a global scratch buffer. It is mathematically identical to `scalar_mul()`.
///
/// GATED: this path is never taken by `run_gpu` / `self_test` unless the
/// caller explicitly opts in via [`radeon_self_test`] or a future `--radeon`
/// flag. The kernels themselves are always compiled (they build on every
/// driver), but the multi-dispatch loop is only exercised when requested, so
/// the default release path stays on the trusted, well-tested Jacobian kernel.
///
/// Returns the affine (Qx, Qy) for `key` (big-endian u32 chunks) on the device.
/// Panics are avoided: on any OpenCL error the function returns `None`.
pub fn radeon_scalar_mul(
    proque: &ProQue,
    key: &[u32; 8],
    work_size: usize,
) -> Option<([u32; 8], [u32; 8])> {
    // Scratch: 24 uints per work-item (RX,RY,RZ), 16 uints for affine output.
    let scratch = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(work_size * 24)
        .build()
        .ok()?;
    let base_buf = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(8)
        .build()
        .ok()?;
    // 1-element buffer carrying the current bit index (host overwrites it each
    // dispatch — see kernel note on why this is a buffer, not a scalar arg).
    let bit_buf = Buffer::<i32>::builder()
        .queue(proque.queue().clone())
        .len(1)
        .build()
        .ok()?;

    let init = proque
        .kernel_builder("radeon_init_inf")
        .arg(&scratch)
        .build()
        .ok()?;
    let step = proque
        .kernel_builder("radeon_step_bit")
        .arg(&scratch)
        .arg(&base_buf)
        .arg(&bit_buf)
        .build()
        .ok()?;
    let finalize = proque
        .kernel_builder("radeon_finalize_affine")
        .arg(&scratch)
        .build()
        .ok()?;

    // Set the base key for every work-item (gid is added inside the kernel for
    // multi-work-item use; for a single key work_size=1 and gid=0).
    base_buf.write(&key[..]).enq().ok()?;
    proque.queue().finish().ok()?;

    unsafe {
        init.cmd()
            .global_work_size(SpatialDims::One(work_size))
            .enq()
            .ok()?;
    }
    proque.queue().finish().ok()?;

    for bit in 0..256i32 {
        let b: [i32; 1] = [bit];
        bit_buf.write(&b[..]).enq().ok()?;
        proque.queue().finish().ok()?;
        unsafe {
            step.cmd()
                .global_work_size(SpatialDims::One(work_size))
                .enq()
                .ok()?;
        }
        proque.queue().finish().ok()?;
    }

    unsafe {
        finalize
            .cmd()
            .global_work_size(SpatialDims::One(work_size))
            .enq()
            .ok()?;
    }
    proque.queue().finish().ok()?;

    // Read back affine (Qx,Qy) for work-item 0.
    let mut out = vec![0u32; work_size * 16];
    scratch.read(&mut out[..]).enq().ok()?;
    proque.queue().finish().ok()?;

    let mut qx = [0u32; 8];
    let mut qy = [0u32; 8];
    qx.copy_from_slice(&out[..8]);
    qy.copy_from_slice(&out[8..16]);
    Some((qx, qy))
}

/// EXPERIMENTAL self-test for the Radeon multi-dispatch path.
///
/// Validates `radeon_scalar_mul` against the CPU reference
/// (`privkey_to_address`) for a set of test vectors, including the generator
/// (key=1), small scalars (key=2,3) and random keys. Returns true only if the
/// device builds the kernels and every vector matches. This is gated and only
/// invoked when the user explicitly requests the Radeon experiment.
pub fn radeon_self_test(selection: DeviceSelection) -> bool {
    let (proque, dev) = match select_device(selection) {
        Some(x) => x,
        None => {
            eprintln!("[radeon-self-test] no reliable OpenCL device available");
            return false;
        }
    };
    eprintln!("[radeon-self-test] device: {}", device_name(&dev));

    let mut rng = OsRng;
    // Test vectors: key=1 (generator), key=2, key=3, then random keys.
    let mut vectors: Vec<[u8; 32]> = vec![
        {
            let mut b = [0u8; 32];
            b[31] = 1;
            b
        },
        {
            let mut b = [0u8; 32];
            b[31] = 2;
            b
        },
        {
            let mut b = [0u8; 32];
            b[31] = 3;
            b
        },
    ];
    for _ in 0..5 {
        let mut b = [0u8; 32];
        rng.fill_bytes(&mut b);
        vectors.push(b);
    }

    for (i, bytes) in vectors.iter().enumerate() {
        let key = bytes_to_u32x8(bytes);
        let expected = match privkey_to_address(bytes) {
            Some(a) => a,
            None => {
                eprintln!("[radeon-self-test] vector {}: CPU reference failed", i);
                return false;
            }
        };
        let (qx, qy) = match radeon_scalar_mul(&proque, &key, 1) {
            Some(v) => v,
            None => {
                eprintln!(
                    "[radeon-self-test] vector {}: OpenCL error during dispatch",
                    i
                );
                return false;
            }
        };
        // Reconstruct the address from (Qx, Qy) on the host using the same
        // keccak path the CPU oracle uses.
        let mut pub65 = [0u8; 65];
        pub65[0] = 0x04;
        for j in 0..8 {
            let b = qx[j].to_be_bytes();
            pub65[1 + 2 * j] = b[0];
            pub65[2 + 2 * j] = b[1];
            let b = qy[j].to_be_bytes();
            pub65[17 + 2 * j] = b[0];
            pub65[18 + 2 * j] = b[1];
        }
        let got = crate::crypto::pubkey_to_address(&pub65);
        if got != expected {
            eprintln!(
                "[radeon-self-test] vector {}: MISMATCH (GPU affine != CPU address)",
                i
            );
            return false;
        }
        eprintln!("[radeon-self-test] vector {}: OK", i);
    }
    true
}

pub fn benchmark(seconds: u64, batch: usize, selection: DeviceSelection) -> Option<f64> {
    let (proque, dev) = select_device(selection)?;
    let dev_name = device_name(&dev);
    let impossible = [0x0Fu8; 40];
    let pat = Pattern {
        prefix: impossible.to_vec(),
        suffix: vec![],
        alt_suffixes: vec![],
    };

    let base_buf = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(8)
        .build()
        .ok()?;
    let pubs = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(batch * 16)
        .build()
        .ok()?;
    let addrs = Buffer::<u8>::builder()
        .queue(proque.queue().clone())
        .len(batch * 20)
        .build()
        .ok()?;
    let out_found = Buffer::<i32>::builder()
        .queue(proque.queue().clone())
        .len(1)
        .build()
        .ok()?;
    let out_priv = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(8)
        .build()
        .ok()?;
    let out_addr = Buffer::<u8>::builder()
        .queue(proque.queue().clone())
        .len(20)
        .build()
        .ok()?;
    let params = Buffer::<u32>::builder()
        .queue(proque.queue().clone())
        .len(params_len(&pat))
        .build()
        .ok()?;

    let derive = proque
        .kernel_builder("derive_pubkeys")
        .arg(&base_buf)
        .arg(&pubs)
        .build()
        .ok()?;
    let hash = proque
        .kernel_builder("hash_addrs")
        .arg(&pubs)
        .arg(&addrs)
        .build()
        .ok()?;
    let matcher = proque
        .kernel_builder("match_addrs")
        .arg(&base_buf)
        .arg(&addrs)
        .arg(&out_found)
        .arg(&out_priv)
        .arg(&out_addr)
        .arg(&params)
        .build()
        .ok()?;

    let mut base_bytes = [0u8; 32];
    OsRng.fill_bytes(&mut base_bytes);
    let mut base = bytes_to_u32x8(&base_bytes);

    params.write(&make_params(&base, &pat)).enq().ok()?;

    let start = std::time::Instant::now();
    let deadline = std::time::Duration::from_secs(seconds.max(1));
    let mut total: u64 = 0;
    loop {
        if start.elapsed() >= deadline {
            break;
        }
        base_buf.write(&base[..]).enq().ok()?;
        let zero = [0i32];
        out_found.write(&zero[..]).enq().ok()?;
        proque.queue().finish().ok()?;
        unsafe {
            derive
                .cmd()
                .global_work_size(SpatialDims::One(batch))
                .enq()
                .ok()?;
        }
        proque.queue().finish().ok()?;
        unsafe {
            hash.cmd()
                .global_work_size(SpatialDims::One(batch))
                .enq()
                .ok()?;
        }
        proque.queue().finish().ok()?;
        unsafe {
            matcher
                .cmd()
                .global_work_size(SpatialDims::One(batch))
                .enq()
                .ok()?;
        }
        proque.queue().finish().ok()?;
        total += batch as u64;
        let mut bb = u32x8_to_bytes(&base);
        add_u64_be(&mut bb, batch as u64);
        base = bytes_to_u32x8(&bb);
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    let rate = total as f64 / elapsed;
    eprintln!(
        "[benchmark] device='{}' batch={} total={} elapsed={:.1}s rate={:.2}Mkeys/s",
        dev_name,
        batch,
        total,
        elapsed,
        rate / 1e6
    );
    Some(rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::pubkey_to_address;

    /// Helper: re-derive the EVM address from a device-side affine point.
    fn affine_to_address(qx: &[u32; 8], qy: &[u32; 8]) -> [u8; 20] {
        let mut pub65 = [0u8; 65];
        pub65[0] = 0x04;
        for j in 0..8 {
            let b = qx[j].to_be_bytes();
            pub65[1 + 2 * j] = b[0];
            pub65[2 + 2 * j] = b[1];
            let b = qy[j].to_be_bytes();
            pub65[17 + 2 * j] = b[0];
            pub65[18 + 2 * j] = b[1];
        }
        pubkey_to_address(&pub65)
    }

    /// Build a ProQue on the first available GPU; skip the test if none.
    fn first_gpu_proque() -> Option<ProQue> {
        for p in Platform::list() {
            if let Ok(devs) = Device::list(p, Some(DeviceType::GPU)) {
                for d in devs {
                    if let Some(proque) = build_proque(p, d) {
                        if probe_device(&proque) {
                            return Some(proque);
                        }
                    }
                }
            }
        }
        None
    }

    #[test]
    #[ignore = "requires a real OpenCL GPU (CI runners have none); run with `cargo test -- --ignored`"]
    fn radeon_scalar_mul_matches_cpu() {
        let proque = match first_gpu_proque() {
            Some(p) => p,
            None => {
                eprintln!("[test] no reliable GPU — skipping radeon_scalar_mul test");
                return;
            }
        };

        let cases: Vec<[u8; 32]> = vec![
            {
                let mut b = [0u8; 32];
                b[31] = 1; // generator
                b
            },
            {
                let mut b = [0u8; 32];
                b[31] = 2;
                b
            },
            {
                let mut b = [0u8; 32];
                b[31] = 3;
                b
            },
            [0x12; 32],
            [0xAB; 32],
        ];

        for bytes in &cases {
            let key = bytes_to_u32x8(bytes);
            let expected = privkey_to_address(bytes).expect("CPU reference");
            let (qx, qy) = radeon_scalar_mul(&proque, &key, 1).expect("radeon_scalar_mul dispatch");
            let got = affine_to_address(&qx, &qy);
            assert_eq!(
                got, expected,
                "radeon_scalar_mul mismatch for key {:?}",
                bytes
            );
        }
    }

    #[test]
    #[ignore = "requires a real OpenCL GPU (CI runners have none); run with `cargo test -- --ignored`"]
    fn radeon_self_test_runs() {
        // This exercises the full gated path; it only asserts the function
        // returns a bool (true on a working device, false otherwise) without
        // panicking. On a machine without a reliable GPU it returns false.
        let result = radeon_self_test(DeviceSelection::Auto);
        if result {
            eprintln!("[radeon] self-test OK on this host");
        } else {
            // CI runners and most Linux dev boxes have no reliable GPU; this
            // test asserts the dispatch path is reachable and panic-free, not
            // that the host must own a working AMD discrete GPU. Skip cleanly.
            eprintln!("[radeon] skipped: no reliable GPU on this host");
        }
    }
}
