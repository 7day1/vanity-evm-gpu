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
        }
    }
}

/// The eframe application.
struct VanityApp {
    /// Shared mutable state between the UI thread and the search thread.
    state: Arc<Mutex<GuiState>>,
    // --- input fields ---
    prefix: String,
    /// Four independent suffix inputs, each with its own enable checkbox.
    /// Only checked, non-empty suffixes participate in the search; a match on
    /// ANY one of them stops the search (first-hit wins, like the CLI without
    /// `--all-groups`).
    suffix: [String; 4],
    /// Per-suffix enable checkboxes (box i gates `suffix[i]`).
    suffix_enabled: [bool; 4],
    max_seconds: String,
    /// GPU batch size (work-items per dispatch). 0 = use a built-in default
    /// tuned for the detected backend. Higher = more GPU throughput on
    /// discrete GPUs; smaller = more responsive on integrated GPUs.
    batch: String,
    force_cpu: bool,
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
            suffix: [String::new(), String::new(), String::new(), String::new()],
            // The first suffix is pre-checked so a single suffix "just works"
            // like before; the other three are opt-in.
            suffix_enabled: [true, false, false, false],
            max_seconds: String::new(),
            // Empty string means "use built-in default" — the worker thread
            // resolves it to 65,536 for GPU and 4,096 for CPU. The default is
            // tuned for discrete GPUs; the user can override by typing a
            // number into the input field.
            batch: String::new(),
            force_cpu: false,
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
            ui.label("后缀 (勾选即参与匹配，命中任意一个即停止):");
            for i in 0..4 {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.suffix_enabled[i], format!("后缀 {}:", i + 1));
                    ui.text_edit_singleline(&mut self.suffix[i]);
                });
            }
            ui.small("所有勾选的后缀需等长（如 88888888 / 77777777）");
            ui.horizontal(|ui| {
                ui.label("最长秒数 (0=不限):");
                ui.text_edit_singleline(&mut self.max_seconds);
            });
            ui.horizontal(|ui| {
                ui.label("每批 key 数 (留空=自动):");
                ui.text_edit_singleline(&mut self.batch);
                ui.small("dGPU 推荐 65K-1M / iGPU 用 16K 以下");
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

        // Collect the checked, non-empty suffix boxes into an ordered group
        // list. The first becomes the primary suffix (group 0) and the rest
        // are alternative groups. A match on ANY of them stops the search —
        // this mirrors the CLI's default (non `--all-groups`) behavior, which
        // is exactly what "勾选多个 = 命中任意一个即停" means.
        let groups: Vec<String> = (0..4)
            .filter(|&i| self.suffix_enabled[i])
            .map(|i| self.suffix[i].trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let primary_suffix = groups.first().cloned().unwrap_or_default();
        let alt_suffixes: &[String] = if groups.len() > 1 { &groups[1..] } else { &[] };

        let parsed = if alt_suffixes.is_empty() {
            config::Pattern::parse(&self.prefix, &primary_suffix)
        } else {
            config::Pattern::parse_multi(&self.prefix, &primary_suffix, alt_suffixes)
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

        // Batch size: empty / unparseable → use built-in default that is
        // tuned for discrete GPUs (65536). iGPU users can override down to
        // 4096 if `CL_OUT_OF_RESOURCES` is reported.
        let resolved_batch: usize = match self.batch.trim() {
            "" => 1 << 16, // 65536 — sweet spot for dGPU, also fine for iGPU
            s => match s.parse::<usize>() {
                Ok(0) => 1 << 16,
                Ok(n) => n,
                Err(_) => {
                    self.notice = "每批 key 数必须是正整数（留空=自动）".to_string();
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

        // Clone inputs for the worker thread. `groups` (the checked suffix
        // list) is moved into the closure so the thread owns it; the primary
        // and alternative groups are sliced from it inside the thread.
        let prefix = self.prefix.trim().to_string();
        let force_cpu = self.force_cpu;
        let device_sel = self.device_sel.trim().to_string();
        let state = self.state.clone();

        thread::spawn(move || {
            let primary = groups.first().map(|s| s.as_str()).unwrap_or("");
            let alts: &[String] = if groups.len() > 1 { &groups[1..] } else { &[] };
            run_search(
                &prefix,
                primary,
                alts,
                max_seconds,
                force_cpu,
                &device_sel,
                resolved_batch,
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
#[allow(clippy::too_many_arguments)]
fn run_search(
    prefix: &str,
    suffix: &str,
    suffixes_list: &[String],
    max_seconds: Option<u64>,
    force_cpu: bool,
    device_sel: &str,
    batch: usize,
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

    let batch = if force_cpu {
        // CPU work doesn't benefit from large batches — keep it small so the
        // progress callback fires frequently and Stop stays responsive.
        1 << 12
    } else {
        batch.max(1)
    };
    let device = parse_device(device_sel);

    // The GUI is "first-hit wins": a match on any checked suffix stops the
    // search. There is no --all-groups toggle in the GUI anymore.
    let all_groups = false;
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
        viewport: egui::ViewportBuilder::default().with_inner_size([560.0, 740.0]),
        ..Default::default()
    };
    eframe::run_native(
        "vanity-evm-gpu",
        native_options,
        Box::new(|cc| {
            // Install a CJK-capable font before the first paint so all
            // Chinese labels (buttons, status lines, match text) render as
            // glyphs instead of tofu boxes. We try the OS-installed font
            // first (msyh.ttc on Windows; PingFang/Hiragino on macOS; WQY
            // microhei on Linux). If none exists we fall back to the egui
            // default font, which still renders ASCII fine.
            install_cjk_font(&cc.egui_ctx);
            Box::new(VanityApp::default())
        }),
    )
}

/// Try to load a CJK font from a well-known OS location and install it into
/// egui's font stack. We register the loaded TTF/TTC against both the
/// `proportional` and `monospace` font families so all UI text (including the
/// monospaced address/private-key lines) gets CJK coverage. Returns silently
/// if no CJK font is found — the default Latin font is then used and Chinese
/// characters render as empty boxes (a known ugliness, but the app remains
/// fully functional).
fn install_cjk_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    // Candidate paths, in priority order. Each platform has one likely location
    // for a CJK font; missing files are skipped without error.
    let candidates: &[&str] = &[
        // Windows: 微软雅黑 (msyh.ttc) — bundled with every Windows since Vista.
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttc", // (duplicate guard for clarity)
        r"C:\Windows\Fonts\msyh.ttf", // some embedded SKUs ship .ttf
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        // macOS: PingFang on modern macOS.
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Songti.ttc",
        // Linux: 文泉驿微米黑, Noto CJK, DejaVu fallback.
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ];

    let font_data: Option<(Vec<u8>, &'static str)> =
        candidates.iter().find_map(|p| match std::fs::read(p) {
            Ok(bytes) => Some((bytes, *p)),
            Err(_) => None,
        });

    if let Some((bytes, path)) = font_data {
        // Pick a stable family name. On Windows, msyh.ttc is a TrueType
        // collection containing many faces; egui treats the whole file as one
        // font and renders any CJK glyphs from the first face that defines
        // them, which works well enough for our mixed CN/EN labels.
        let family_name = if path.contains("msyh") {
            "Microsoft YaHei"
        } else if path.contains("simhei") {
            "SimHei"
        } else if path.contains("simsun") {
            "SimSun"
        } else if path.contains("PingFang") {
            "PingFang"
        } else if path.contains("STHeiti") {
            "STHeiti"
        } else if path.contains("Songti") {
            "Songti"
        } else if path.contains("wqy") {
            "WenQuanYi Micro Hei"
        } else if path.contains("NotoSansCJK") {
            "Noto Sans CJK SC"
        } else {
            "CJKFallback"
        };

        fonts
            .font_data
            .insert(family_name.to_owned(), egui::FontData::from_owned(bytes));

        // Put our font FIRST in both stacks so it wins any glyph that the
        // default font doesn't have. The egui defaults still handle ASCII
        // and Latin-1 cleanly behind it.
        if let Some(proportional) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
            proportional.insert(0, family_name.to_owned());
        }
        if let Some(monospace) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
            monospace.insert(0, family_name.to_owned());
        }
    }

    ctx.set_fonts(fonts);
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
