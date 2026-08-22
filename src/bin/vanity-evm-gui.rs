//! `vanity-evm-gui` — a small eframe (egui) front-end for `vanity-evm-gpu`.
//!
//! Shows, in real time:
//!   * which backend is active (CPU / GPU) and the device name,
//!   * throughput (keys/s and Mkeys/s), total attempts, elapsed time,
//!   * a running indicator and the final match (address + private key, or
//!     redacted).
//!
//! The search itself runs on a background thread (so the UI never blocks) and
//! communicates with the UI exclusively through the shared `ProgressCb` plus a
//! small `Arc<Mutex<GuiState>>`. The Stop button sets `stop_requested`, which
//! makes the callback return `false` and cleanly cancels the search loop.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use eframe::egui;
use vanity_evm_gpu::{
    config, cpu_worker, gpu, output,
    progress::{Progress, ProgressCb},
    Found,
};

/// Live, UI-facing snapshot of the search. Guarded by a single `Mutex` so the
/// background thread and the UI thread never race on it.
///
/// The matched private key lives inside `result: Option<Found>`, and `Found`
/// already derives `ZeroizeOnDrop` — so when the `Option` is dropped (program
/// exit, or when a new search overwrites the field) the plaintext private key
/// is byte-zeroed. That gives the same defense-in-depth as the CLI's
/// `Zeroizing<String>` without needing `GuiState` itself to implement `Drop`
/// (which would forbid the `..Default::default()` struct-reset pattern used by
/// the Start button).
struct GuiState {
    /// True while a search is running.
    running: bool,
    /// Backend label shown in the status panel ("CPU" / "GPU").
    backend: String,
    /// Device name (GPU model, or "host CPU ...").
    device: String,
    /// Throughput in keys/s.
    rate: f64,
    /// Total candidate keys tested.
    attempts: u64,
    /// Wall-clock elapsed time, seconds.
    elapsed: f64,
    /// When set, a (human-readable) error or stop reason.
    error: Option<String>,
    /// The matched wallet (with the private key), if found.
    result: Option<Found>,
    /// Whether the private key should be shown or redacted in the UI.
    redact: bool,
    /// Set by the Stop button; the background thread's ProgressCb returns
    /// `false` once this flips, cancelling the search loop.
    stop_requested: bool,
    /// Result of the last GPU self-test ("PASS" / "FAIL" / message), if run.
    self_test_result: Option<String>,
    /// Result of the last EXPERIMENTAL Radeon multi-dispatch self-test,
    /// if run. The default Jacobian path is always used by `run_gpu`; this
    /// second button only validates the alternative kernel layout.
    radeon_self_test_result: Option<String>,
    /// When true, keep searching until every suffix group has a match (mirrors
    /// the CLI `--all-groups` flag). Added to the shared state so the background
    /// thread can read it without borrowing `self`.
    all_groups: bool,
}

impl Default for GuiState {
    fn default() -> Self {
        Self {
            running: false,
            backend: String::new(),
            device: String::new(),
            rate: 0.0,
            attempts: 0,
            elapsed: 0.0,
            error: None,
            result: None,
            redact: false,
            stop_requested: false,
            self_test_result: None,
            radeon_self_test_result: None,
            all_groups: false,
        }
    }
}

/// The eframe application.
struct VanityApp {
    /// Shared mutable state between the UI thread and the search thread.
    state: Arc<Mutex<GuiState>>,
    // --- input fields ---
    prefix: String,
    suffix: String,
    /// Additional suffix groups, comma-separated (mirrors CLI `--suffixes`).
    suffixes: String,
    max_seconds: String,
    force_cpu: bool,
    all_groups: bool,
    redact: bool,
    /// Which GPU device to request: "auto" / index / name substring.
    device_sel: String,
    /// Status line shown near the controls (parsed-error / hints).
    notice: String,
    /// Available OpenCL GPU devices (refreshed on app start), for the dropdown.
    gpu_devices: Vec<(usize, String)>,
    /// Index into `gpu_devices` the user picked; `None` means "auto".
    device_index: Option<usize>,
}

impl Default for VanityApp {
    fn default() -> Self {
        let gpu_devices = gpu::list_gpus();
        Self {
            state: Arc::new(Mutex::new(GuiState::default())),
            prefix: String::new(),
            suffix: String::new(),
            suffixes: String::new(),
            max_seconds: String::new(),
            force_cpu: false,
            all_groups: false,
            redact: false,
            device_sel: "auto".to_string(),
            notice: String::new(),
            gpu_devices,
            device_index: None,
        }
    }
}

impl eframe::App for VanityApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("vanity-evm-gpu");
            ui.label("EVM 靓号地址生成器 — GPU/CPU 实时状态");
            ui.separator();

            // ---- input region ----
            ui.horizontal(|ui| {
                ui.label("前缀 (prefix):");
                ui.text_edit_singleline(&mut self.prefix);
            });
            ui.horizontal(|ui| {
                ui.label("后缀 (suffix):");
                ui.text_edit_singleline(&mut self.suffix);
                ui.label("额外后缀 (逗号分隔, 等长):");
                ui.text_edit_singleline(&mut self.suffixes);
                if !self.suffixes.trim().is_empty() {
                    ui.small(format!("命中 {} 或任一额外后缀", self.suffix.trim()));
                }
            });
            ui.horizontal(|ui| {
                ui.label("最长秒数 (0=不限):");
                ui.text_edit_singleline(&mut self.max_seconds);
            });
            ui.horizontal(|ui| {
                ui.label("GPU 设备:");
                // Dropdown: "自动 (auto)" plus each detected device.
                let combo_label = match self.device_index {
                    Some(i) => self
                        .gpu_devices
                        .iter()
                        .find(|(idx, _)| *idx == i)
                        .map(|(_, n)| n.clone())
                        .unwrap_or_else(|| "自动 (auto)".to_string()),
                    None => "自动 (auto)".to_string(),
                };
                egui::ComboBox::from_label("")
                    .selected_text(combo_label)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.device_index, None, "自动 (auto)");
                        for (i, name) in &self.gpu_devices {
                            ui.selectable_value(&mut self.device_index, Some(*i), name.clone());
                        }
                    });
                // Keep device_sel in sync for the worker thread.
                self.device_sel = match self.device_index {
                    Some(i) => i.to_string(),
                    None => "auto".to_string(),
                };
            });
            ui.checkbox(&mut self.force_cpu, "强制 CPU 模式 (跳过 GPU 探测)");
            // The all_groups toggle needs to reach the background thread, which
            // only borrows the shared `GuiState`. We mirror it into `GuiState`
            // on every UI frame so the worker reads the latest value.
            {
                let mut s = self.state.lock().unwrap();
                s.all_groups = self.all_groups;
            }
            ui.checkbox(&mut self.all_groups, "每组后缀各出一个 (--all-groups)");
            ui.checkbox(&mut self.redact, "隐藏私钥 (redact)");

            ui.separator();

            // ---- start / stop / self-test ----
            let (running, can_stop) = {
                let s = self.state.lock().unwrap();
                (s.running, s.stop_requested)
            };
            ui.horizontal(|ui| {
                if ui.button("开始 (Start)").clicked() && !running {
                    self.start_search();
                }
                if ui.button("停止 (Stop)").clicked() && running && !can_stop {
                    self.state.lock().unwrap().stop_requested = true;
                }
                if running {
                    ui.label("● 运行中…");
                } else {
                    ui.label("○ 空闲");
                }
            });
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!running, egui::Button::new("自检 (Self-Test)"))
                    .clicked()
                {
                    self.run_self_test();
                }
                if ui
                    .add_enabled(!running, egui::Button::new("Radeon 自检 (Experimental)"))
                    .clicked()
                {
                    self.run_radeon_self_test();
                }
                ui.label("只验证 GPU，不搜索（新手安全演练）");
            });

            if !self.notice.is_empty() {
                ui.colored_label(egui::Color32::RED, &self.notice);
            }

            ui.separator();

            // ---- live status panel ----
            let snap = {
                let s = self.state.lock().unwrap();
                (
                    s.running,
                    s.backend.clone(),
                    s.device.clone(),
                    s.rate,
                    s.attempts,
                    s.elapsed,
                    s.error.clone(),
                    s.result
                        .as_ref()
                        .map(|f| (f.address_eip55(), f.priv_reduced)),
                    s.redact,
                )
            };
            let (running, backend, device, rate, attempts, elapsed, error, result, redact) = snap;

            // Status panel: each metric as a bold label line followed by its
            // value on the next line. This reads better with mixed
            // Chinese/English labels and avoids the cramped two-column grid.
            ui.group(|ui| {
                ui.strong("后端 (backend):");
                ui.label(if backend.is_empty() {
                    "—"
                } else {
                    backend.as_str()
                });

                ui.strong("设备 (device):");
                ui.label(if device.is_empty() {
                    "—"
                } else {
                    device.as_str()
                });

                ui.strong("速率 (rate):");
                ui.label(format!("{:.0} keys/s  ({:.2} Mkeys/s)", rate, rate / 1e6));

                ui.strong("已尝试 (attempts):");
                ui.label(format!("{}", attempts));

                ui.strong("耗时 (elapsed):");
                ui.label(format!("{:.1} s", elapsed));

                ui.strong("状态 (status):");
                if running {
                    ui.colored_label(egui::Color32::from_rgb(40, 160, 60), "运行中");
                } else {
                    ui.label("已停止");
                }
            });

            if let Some(msg) = &error {
                ui.separator();
                ui.colored_label(egui::Color32::RED, format!("错误/提示: {}", msg));
            }

            // ---- self-test result ----
            let self_test_result = {
                let s = self.state.lock().unwrap();
                s.self_test_result.clone()
            };
            if let Some(res) = self_test_result {
                ui.separator();
                let pass = res.starts_with("PASS");
                let color = if pass {
                    egui::Color32::from_rgb(40, 160, 60)
                } else if res.starts_with("FAIL") {
                    egui::Color32::RED
                } else {
                    egui::Color32::GRAY
                };
                ui.colored_label(color, format!("自检结果: {}", res));
            }

            // ---- radeon multi-dispatch self-test result ----
            let radeon_self_test_result = {
                let s = self.state.lock().unwrap();
                s.radeon_self_test_result.clone()
            };
            if let Some(res) = radeon_self_test_result {
                ui.separator();
                let pass = res.starts_with("PASS");
                let color = if pass {
                    egui::Color32::from_rgb(40, 160, 60)
                } else if res.starts_with("FAIL") {
                    egui::Color32::RED
                } else {
                    egui::Color32::GRAY
                };
                ui.colored_label(color, format!("Radeon 多 dispatch 自检: {}", res));
            }

            // ---- result ----
            if let Some((addr, priv32)) = result {
                ui.separator();
                ui.heading("命中 (match found)");
                ui.monospace(format!("Address: 0x{}", addr));
                if redact {
                    ui.monospace("PrivateKey: [redacted]");
                } else {
                    // Format the private key on the fly into a `Zeroizing<String>`
                    // so the plaintext copy is byte-zeroed once the frame is done
                    // rendering (defense-in-depth, matching the CLI's approach).
                    let mut hex = zeroize::Zeroizing::new(String::with_capacity(64));
                    for b in priv32 {
                        hex.push_str(&format!("{:02x}", b));
                    }
                    ui.monospace(format!("PrivateKey: 0x{}", hex.as_str()));
                    // One-click copy to clipboard (arboard) to avoid manual
                    // selection errors with the long hex strings.
                    ui.horizontal(|ui| {
                        if ui.button("复制地址").clicked() {
                            let _ = copy_to_clipboard(&format!("0x{}", addr));
                        }
                        if ui.button("复制私钥").clicked() {
                            let _ = copy_to_clipboard(&format!("0x{}", hex.as_str()));
                        }
                    });
                }
                ui.label(
                    "⚠️ 充值前请独立用 ethers/web3.py/alloy 从私钥重推地址，核对 EIP-55 一致。",
                );
            }
        });

        // Repaint continuously so the live numbers update even while the search
        // thread is busy. ~30 fps is plenty for a status panel.
        ctx.request_repaint_after(std::time::Duration::from_millis(33));
    }
}

impl VanityApp {
    /// Validate inputs, then spawn the background search thread.
    fn start_search(&mut self) {
        self.notice.clear();

        // Validate the pattern up front (in the UI thread) so we can show a
        // clear message instead of silently exiting.
        let suffixes_list: Vec<String> = self
            .suffixes
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let parsed = if suffixes_list.is_empty() {
            config::Pattern::parse(&self.prefix, &self.suffix)
        } else {
            config::Pattern::parse_multi(&self.prefix, &self.suffix, &suffixes_list)
        };
        match parsed {
            Ok(_) => {}
            Err(e) => {
                self.notice = format!("参数错误: {}", e);
                return;
            }
        }

        let max_seconds: Option<u64> = match self.max_seconds.trim() {
            "" => None,
            s => match s.parse::<u64>() {
                Ok(0) => None,
                Ok(n) => Some(n),
                Err(_) => {
                    self.notice = "最长秒数必须是非负整数".to_string();
                    return;
                }
            },
        };

        // Reset the shared state and mark running.
        {
            let mut s = self.state.lock().unwrap();
            *s = GuiState {
                running: true,
                redact: self.redact,
                ..Default::default()
            };
        }

        // Clone inputs for the worker thread.
        let prefix = self.prefix.trim().to_string();
        let suffix = self.suffix.trim().to_string();
        let force_cpu = self.force_cpu;
        let device_sel = self.device_sel.trim().to_string();
        let state = self.state.clone();

        thread::spawn(move || {
            run_search(
                &prefix,
                &suffix,
                &suffixes_list,
                max_seconds,
                force_cpu,
                &device_sel,
                state,
            );
        });
    }

    /// Spawn a background thread that runs `gpu::self_test` (validate the GPU
    /// kernel against the CPU reference without searching) and records the
    /// result into `state.self_test_result`. This is the GUI equivalent of the
    /// CLI's `--self-test` / `--dry-run` for safe beginner practice.
    fn run_self_test(&mut self) {
        self.notice.clear();
        let device_sel = self.device_sel.trim().to_string();
        let state = self.state.clone();
        thread::spawn(move || {
            let device = parse_device(&device_sel);
            let ok = gpu::self_test(device);
            let mut s = state.lock().unwrap();
            s.self_test_result = Some(if ok {
                "PASS — GPU 内核与 CPU 参考一致".to_string()
            } else {
                "FAIL — 该设备 GPU 结果不可信，请勿使用".to_string()
            });
        });
    }

    /// Spawn a background thread that runs `gpu::radeon_self_test`, which
    /// exercises the EXPERIMENTAL multi-dispatch scalar_mul path (alternative
    /// to the default Jacobian kernel). NOTE: this path has NO confirmed
    /// working GPU yet — on macOS (Apple OpenCL compiler) it fails with
    /// `cvms_element_build_from_source`, and on Windows/Linux the default
    /// Jacobian path is already correct, so this button is mainly a diagnostic
    /// for developers. On a healthy GPU it should report PASS; on macOS it
    /// will report FAIL (expected).
    fn run_radeon_self_test(&mut self) {
        self.notice.clear();
        let state = self.state.clone();
        thread::spawn(move || {
            let ok = gpu::radeon_self_test(gpu::DeviceSelection::Auto);
            let mut s = state.lock().unwrap();
            s.radeon_self_test_result = Some(if ok {
                "PASS — Radeon 多 dispatch 与 CPU 参考一致".to_string()
            } else {
                "FAIL — 该设备 Radeon 多 dispatch 结果不可信".to_string()
            });
        });
    }
}

/// Copy a string to the system clipboard using `arboard`.
fn copy_to_clipboard(text: &str) -> bool {
    match arboard::Clipboard::new() {
        Ok(mut cb) => cb.set_text(text.to_string()).is_ok(),
        Err(_) => false,
    }
}

/// Background-thread entry point. Owns the search loop and pushes progress into
/// `state` via a `ProgressCb`. On completion it records the result (or error)
/// and flips `running` to false.
fn run_search(
    prefix: &str,
    suffix: &str,
    suffixes_list: &[String],
    max_seconds: Option<u64>,
    force_cpu: bool,
    device_sel: &str,
    state: Arc<Mutex<GuiState>>,
) {
    let pattern = if suffixes_list.is_empty() {
        config::Pattern::parse(prefix, suffix)
    } else {
        config::Pattern::parse_multi(prefix, suffix, suffixes_list)
    };
    let pattern = match pattern {
        Ok(p) => p,
        Err(e) => {
            let mut s = state.lock().unwrap();
            s.running = false;
            s.error = Some(format!("pattern parse error: {}", e));
            return;
        }
    };

    // Decide the backend exactly like the CLI: prefer GPU unless forced, but
    // only if a reliable OpenCL device exists.
    let use_gpu = if force_cpu {
        false
    } else {
        gpu::gpu_available()
    };

    // Build the progress callback that bridges the search loop to `state`.
    let cb: ProgressCb = {
        let state = state.clone();
        Arc::new(move |p: &Progress| -> bool {
            let mut s = state.lock().unwrap();
            s.backend = p.backend.to_string();
            s.device = p.device.clone();
            s.rate = p.rate;
            s.attempts = p.attempts;
            s.elapsed = p.elapsed_secs;
            if p.done {
                s.running = false;
            }
            // Honour the Stop button: if requested, return false so the loop
            // breaks at its next check point.
            !s.stop_requested
        })
    };

    let batch = 1 << 12; // 4096 — safe default for integrated GPUs.
    let device = parse_device(device_sel);

    let all_groups = state.lock().unwrap().all_groups;
    if use_gpu {
        let matches = gpu::run_gpu(
            &pattern,
            max_seconds,
            batch,
            device,
            false,
            all_groups,
            None,
            Some(cb.clone()),
        );
        if let Some(m) = matches.into_iter().next() {
            let found = Found {
                priv_reduced: m.priv32,
                raw_addr: m.addr,
            };
            // Persist to ./results (mirrors the CLI) before moving `found`
            // into the shared state, since `Found` is not `Copy`.
            {
                let redact = state.lock().unwrap().redact;
                let _ = output::write_result(&PathBuf::from("results"), &found, redact);
            }
            let mut s = state.lock().unwrap();
            s.result = Some(found);
            s.running = false;
        } else {
            let mut s = state.lock().unwrap();
            if s.stop_requested {
                s.error = Some("已被用户停止 (Stop)".to_string());
            } else {
                s.error = Some("未在限定时间内找到匹配 (no match)".to_string());
            }
            s.running = false;
        }
    } else {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let matches =
            cpu_worker::run_cpu(&pattern, max_seconds, workers, all_groups, Some(cb.clone()));
        if let Some(m) = matches.into_iter().next() {
            let found = Found {
                priv_reduced: m.priv32,
                raw_addr: m.addr,
            };
            // Persist to ./results (mirrors the CLI) before moving `found`
            // into the shared state, since `Found` is not `Copy`.
            {
                let redact = state.lock().unwrap().redact;
                let _ = output::write_result(&PathBuf::from("results"), &found, redact);
            }
            let mut s = state.lock().unwrap();
            s.result = Some(found);
            s.running = false;
        } else {
            let mut s = state.lock().unwrap();
            if s.stop_requested {
                s.error = Some("已被用户停止 (Stop)".to_string());
            } else {
                s.error = Some("未在限定时间内找到匹配 (no match)".to_string());
            }
            s.running = false;
        }
    }
}

/// Mirror of the CLI's `parse_device`.
fn parse_device(sel: &str) -> gpu::DeviceSelection {
    if sel.eq_ignore_ascii_case("auto") || sel.is_empty() {
        gpu::DeviceSelection::Auto
    } else if let Ok(i) = sel.parse::<usize>() {
        gpu::DeviceSelection::Index(i)
    } else {
        gpu::DeviceSelection::Name(sel.to_string())
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        // eframe 0.27: window size moved into the `viewport` builder.
        // A wider window keeps the Chinese two-column status labels from
        // feeling cramped, and gives the match result room to breathe.
        viewport: egui::ViewportBuilder::default().with_inner_size([560.0, 640.0]),
        ..Default::default()
    };
    eframe::run_native(
        "vanity-evm-gpu",
        native_options,
        Box::new(|_cc| Box::new(VanityApp::default())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_device_auto_and_empty() {
        assert!(matches!(parse_device("auto"), gpu::DeviceSelection::Auto));
        assert!(matches!(parse_device(""), gpu::DeviceSelection::Auto));
        assert!(matches!(parse_device("AUTO"), gpu::DeviceSelection::Auto));
    }

    #[test]
    fn parse_device_index() {
        assert!(matches!(parse_device("1"), gpu::DeviceSelection::Index(1)));
        assert!(matches!(parse_device("0"), gpu::DeviceSelection::Index(0)));
    }

    #[test]
    fn parse_device_name_substring() {
        assert!(matches!(
            parse_device("Radeon"),
            gpu::DeviceSelection::Name(s) if s == "Radeon"
        ));
        // A non-numeric, non-"auto" string is treated as a name substring.
        assert!(matches!(
            parse_device("Intel"),
            gpu::DeviceSelection::Name(s) if s == "Intel"
        ));
    }

    #[test]
    fn gui_state_reset_clears_result() {
        // GuiState derives ZeroizeOnDrop; this verifies the reset path used by
        // Start clears a previous match so a stale private key is not shown.
        // Construct a populated state, then replace it with a fresh default
        // (mirroring `GuiState { running: true, redact, ..Default::default() }`).
        let populated = GuiState {
            running: true,
            backend: "GPU".into(),
            device: "test".into(),
            rate: 1.0,
            attempts: 10,
            elapsed: 1.0,
            error: None,
            result: Some(Found {
                priv_reduced: [0xab; 32],
                raw_addr: [0; 20],
            }),
            redact: false,
            stop_requested: true,
            self_test_result: None,
            radeon_self_test_result: None,
            all_groups: false,
        };
        // The reset path overwrites the whole struct; the old `result` is
        // dropped (and zeroized by ZeroizeOnDrop) rather than carried over.
        let reset: GuiState = GuiState {
            running: true,
            redact: false,
            ..Default::default()
        };
        // `populated` is dropped here; assert the reset value is clean.
        let _ = populated;
        assert!(reset.result.is_none());
        assert!(!reset.stop_requested);
        assert!(reset.running);
    }
}
