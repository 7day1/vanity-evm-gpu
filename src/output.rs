//! Result persistence: atomic local file write, with optional private-key
//! redaction. Private keys never leave the local machine.

use crate::crypto::eip55;
use std::fs;
use std::path::Path;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

/// Holds the generated wallet. The private key is zeroized when this struct is
/// dropped (defense-in-depth): by then it has already been printed/written, but
/// this prevents it lingering in process memory.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct Found {
    pub priv_reduced: [u8; 32], // canonical private key
    pub raw_addr: [u8; 20],
}

impl Found {
    pub fn address_eip55(&self) -> String {
        eip55(&self.raw_addr)
    }
    /// Format the private key as a `0x`-prefixed hex string. The returned
    /// `Zeroizing<String>` is byte-zeroed on drop, so the plaintext private key
    /// does not linger as a heap copy after the caller is done with it.
    pub fn private_key_hex(&self) -> Zeroizing<String> {
        let mut s = String::with_capacity(64);
        for b in &self.priv_reduced {
            s.push_str(&format!("{:02x}", b));
        }
        Zeroizing::new(s)
    }
}

pub fn write_result(dir: &Path, found: &Found, redact: bool) -> std::io::Result<()> {
    fs::create_dir_all(dir).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("failed to create result dir {}: {}", dir.display(), e),
        )
    })?;
    let priv_line = if redact {
        "[redacted by --redact-private-key]".to_string()
    } else {
        format!("0x{}", found.private_key_hex().as_str())
    };
    // `content` is a `Zeroizing<String>` so the plaintext private key is
    // byte-zeroed when this function returns (defense-in-depth on top of the
    // `Found` zeroize-on-drop).
    let content: Zeroizing<String> = Zeroizing::new(format!(
        "Address: 0x{}\nPrivateKey: {}\n",
        found.address_eip55(),
        priv_line
    ));

    let stamp = chrono_stamp();
    let latest = dir.join("matched-wallet-latest.txt");
    let stamped = dir.join(format!("matched-wallet-{}.txt", stamp));

    // Two atomic writes (temp file + atomic rename) so a crash mid-write never
    // leaves a partial/corrupt wallet file behind. Files are created owner-only
    // (0o600) so other local users cannot read the private key.
    let tmp = dir.join(".matched-wallet.tmp");
    write_atomic(tmp.as_path(), content.as_str(), &latest)?;
    write_atomic(tmp.as_path(), content.as_str(), &stamped)?;
    Ok(())
}

/// Write `content` to `tmp`, then atomically rename onto `dst`. Sets the file
/// mode to owner-read/write only (0o600) so a written private key is not
/// world/group readable. Best-effort: a mode change failure is logged but does
/// not abort the write (some platforms/FS ignore it).
fn write_atomic(tmp: &Path, content: &str, dst: &Path) -> std::io::Result<()> {
    fs::write(tmp, content)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(tmp, std::fs::Permissions::from_mode(0o600));
    }
    fs::rename(tmp, dst)?;
    Ok(())
}

fn chrono_stamp() -> String {
    // lightweight, no external time crate dependency on formatting
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{}", secs)
}
