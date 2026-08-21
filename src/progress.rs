//! Shared live-progress type used to surface search statistics to front-ends
//! (CLI logs, GUI panel, etc.) without coupling the search loops to any
//! particular presentation.

/// A point-in-time snapshot of a running search, published by `run_gpu` /
/// `run_cpu` through a `ProgressCb`. All fields are cheap to copy so front-ends
/// can snapshot them freely.
#[derive(Clone, Debug, Default)]
pub struct Progress {
    /// Which backend produced this sample: `"CPU"` or `"GPU"`.
    pub backend: &'static str,
    /// Human-readable device label (e.g. "Intel(R) UHD Graphics 630" or
    /// "host CPU (secp256k1 + Keccak)").
    pub device: String,
    /// Total candidate private keys tested so far.
    pub attempts: u64,
    /// Throughput in candidate keys per second.
    pub rate: f64,
    /// Wall-clock time elapsed since the search started, in seconds.
    pub elapsed_secs: f64,
    /// True once the search has stopped (found a match, hit the deadline, or
    /// was cancelled). Front-ends use this to flip a "running" indicator off.
    pub done: bool,
}

/// Callback invoked periodically (and once with `done = true`) during a search.
/// Shareable across threads: `run_cpu` invokes it from worker threads, so it is
/// `Send + Sync` and only observes `&Progress` (it must not require mutation of
/// captured state beyond interior mutability like a `Mutex`).
///
/// Return value is used as a **cancellation signal**: returning `false` asks the
/// search loop to stop at the next check point (the loop checks after each
/// periodic sample). This lets a front-end implement a "Stop" button without
/// extra parameters. Returning `true` continues normally.
pub type ProgressCb = std::sync::Arc<dyn Fn(&Progress) -> bool + Send + Sync>;
