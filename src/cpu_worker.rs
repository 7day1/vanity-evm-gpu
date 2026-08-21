//! CPU search worker (used as fallback, or when --cpu is set).
//! Each thread pulls random keys; the FIRST match is recorded under a mutex.

use crate::config::Pattern;
use crate::crypto::{addr_matches, zeroize_key};
use crate::progress::{Progress, ProgressCb};
use rand::rngs::OsRng;
use rand::RngCore;
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use zeroize::ZeroizeOnDrop;

/// Result of a CPU match. Zeroized on drop (defense-in-depth).
#[derive(ZeroizeOnDrop)]
pub struct Match {
    pub priv32: [u8; 32],
    pub addr: [u8; 20],
}

pub fn run_cpu(
    pattern: &Pattern,
    max_seconds: Option<u64>,
    workers: usize,
    cb: Option<ProgressCb>,
) -> Option<Match> {
    let found = Arc::new(AtomicBool::new(false));
    let result = Arc::new(Mutex::new(None));
    // Shared aggregate attempt counter so the progress callback can report a
    // consistent total across all worker threads.
    let total_attempts = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let deadline = max_seconds.map(Duration::from_secs);
    let mut handles = Vec::new();

    if let Some(cb) = &cb {
        cb(&Progress {
            backend: "CPU",
            device: "host CPU (secp256k1 + Keccak)".to_string(),
            attempts: 0,
            rate: 0.0,
            elapsed_secs: 0.0,
            done: false,
        });
    }

    for _ in 0..workers.max(1) {
        let found = found.clone();
        let result = result.clone();
        let total_attempts = total_attempts.clone();
        let cb = cb.clone();
        let pat = Pattern {
            prefix: pattern.prefix.clone(),
            suffix: pattern.suffix.clone(),
        };
        let h = std::thread::spawn(move || {
            let secp = Secp256k1::new();
            let mut rng = OsRng;
            let mut buf = [0u8; 32];
            let mut attempts: u64 = 0;
            loop {
                if found.load(Ordering::Relaxed) {
                    break;
                }
                if let Some(d) = deadline {
                    if start.elapsed() >= d {
                        break;
                    }
                }
                attempts += 1;
                rng.fill_bytes(&mut buf);
                let sk = match SecretKey::from_slice(&buf) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let pk = PublicKey::from_secret_key(&secp, &sk);
                let ser = pk.serialize_uncompressed();
                let mut arr = [0u8; 65];
                arr.copy_from_slice(&ser);
                // derive address
                let hash = crate::crypto::keccak256(&arr[1..]);
                let mut addr = [0u8; 20];
                addr.copy_from_slice(&hash[12..]);
                if addr_matches(&addr, &pat.prefix, &pat.suffix) {
                    if !found.swap(true, Ordering::Relaxed) {
                        *result.lock().unwrap() = Some((buf, addr));
                    }
                    break;
                }
                let _ = total_attempts.fetch_add(1, Ordering::Relaxed);
                if attempts.is_multiple_of(2_000_000) {
                    let elapsed = start.elapsed().as_secs_f64();
                    let agg = total_attempts.load(Ordering::Relaxed);
                    let rate = agg as f64 / elapsed.max(1e-6);
                    eprintln!(
                        "[cpu] attempts={} rate={:.0}/s elapsed={:.0}s",
                        agg, rate, elapsed
                    );
                    if let Some(cb) = &cb {
                        let keep_going = cb(&Progress {
                            backend: "CPU",
                            device: "host CPU (secp256k1 + Keccak)".to_string(),
                            attempts: agg,
                            rate,
                            elapsed_secs: elapsed,
                            done: false,
                        });
                        // A `false` return is the front-end's cancellation
                        // request (e.g. the GUI "Stop" button). Stop this
                        // worker; the final `done = true` progress is emitted
                        // by the orchestrator once all workers have joined.
                        if !keep_going {
                            break;
                        }
                    }
                }
            }
            zeroize_key(&mut buf);
        });
        handles.push(h);
    }
    for h in handles {
        let _ = h.join();
    }
    let guard = result.lock().unwrap();
    if let Some(cb) = &cb {
        let elapsed = start.elapsed().as_secs_f64();
        let agg = total_attempts.load(Ordering::Relaxed);
        cb(&Progress {
            backend: "CPU",
            device: "host CPU (secp256k1 + Keccak)".to_string(),
            attempts: agg,
            rate: agg as f64 / elapsed.max(1e-6),
            elapsed_secs: elapsed,
            done: true,
        });
    }
    guard.as_ref().map(|(priv32, addr)| Match {
        priv32: *priv32,
        addr: *addr,
    })
}
