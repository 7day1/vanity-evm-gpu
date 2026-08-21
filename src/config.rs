//! Pattern parsing/validation: hex prefix/suffix -> nibble-value arrays.

#[derive(Debug)]
pub enum ConfigError {
    EmptyPattern,
    InvalidHex(String),
    TooLong(usize),
    TooManySuffixes(usize),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::EmptyPattern => write!(f, "prefix and suffix cannot both be empty"),
            ConfigError::InvalidHex(s) => write!(f, "invalid hex character in pattern: {:?}", s),
            ConfigError::TooLong(n) => write!(f, "pattern too long ({} > 40 hex chars)", n),
            ConfigError::TooManySuffixes(n) => {
                write!(
                    f,
                    "too many suffixes ({} > 16) — keep --suffixes under 16 groups",
                    n
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub prefix: Vec<u8>, // nibble values 0-15
    pub suffix: Vec<u8>, // primary suffix (group 0); kept for --suffix compat
    /// Additional suffix groups. An address matches if its suffix equals
    /// `suffix` OR any one of `alt_suffixes`. Empty when only `--suffix` is used.
    pub alt_suffixes: Vec<Vec<u8>>,
}

impl Pattern {
    pub fn parse(prefix_hex: &str, suffix_hex: &str) -> Result<Self, ConfigError> {
        Self::parse_multi(prefix_hex, suffix_hex, &[])
    }

    /// Parse a pattern with multiple alternative suffixes (from `--suffixes`).
    /// `alt_hexes` is a list of hex suffix strings; `suffix_hex` is the primary
    /// group 0 (may be empty). When `alt_hexes` is empty this is identical to
    /// `parse`.
    pub fn parse_multi(
        prefix_hex: &str,
        suffix_hex: &str,
        alt_hexes: &[String],
    ) -> Result<Self, ConfigError> {
        let prefix = hex_to_nibbles(prefix_hex)?;
        let suffix = hex_to_nibbles(suffix_hex)?;
        let mut alt_suffixes = Vec::with_capacity(alt_hexes.len());
        for h in alt_hexes {
            alt_suffixes.push(hex_to_nibbles(h)?);
        }
        if prefix.is_empty() && suffix.is_empty() && alt_suffixes.is_empty() {
            return Err(ConfigError::EmptyPattern);
        }
        // All suffix groups must share the same length so the GPU kernel can
        // use one stride. We enforce equal length up front.
        let mut groups: Vec<&Vec<u8>> = vec![&suffix];
        groups.extend(alt_suffixes.iter());
        let primary_len = suffix.len();
        for g in &alt_suffixes {
            if g.len() != primary_len {
                return Err(ConfigError::InvalidHex(format!(
                    "all suffixes must have equal length (primary={}, alt={})",
                    primary_len,
                    g.len()
                )));
            }
        }
        let max = prefix
            .len()
            .max(groups.iter().map(|g| g.len()).max().unwrap_or(0));
        if max > 40 {
            return Err(ConfigError::TooLong(max));
        }
        let total_groups = groups.len();
        if total_groups > 16 {
            return Err(ConfigError::TooManySuffixes(total_groups));
        }
        Ok(Self {
            prefix,
            suffix,
            alt_suffixes,
        })
    }

    /// Total number of suffix groups (1 if only `--suffix` is used).
    pub fn suffix_group_count(&self) -> usize {
        1 + self.alt_suffixes.len()
    }

    /// All suffix groups (group 0 first, then alts).
    pub fn all_suffixes(&self) -> Vec<&Vec<u8>> {
        let mut v: Vec<&Vec<u8>> = vec![&self.suffix];
        v.extend(self.alt_suffixes.iter());
        v
    }

    /// Returns the 0-based index of the suffix group that `addr` matches
    /// (under this pattern's prefix), or `None` if it matches none. Used to
    /// tell the user *which* of several `--suffixes` actually hit.
    pub fn matched_suffix_group(&self, addr: &[u8; 20], prefix: &[u8]) -> Option<usize> {
        // Reuse the nibble extraction logic from crypto via a local check.
        let mut nib = [0u8; 40];
        for i in 0..20 {
            nib[2 * i] = (addr[i] >> 4) & 0xF;
            nib[2 * i + 1] = addr[i] & 0xF;
        }
        for (i, p) in prefix.iter().enumerate() {
            if nib[i] != *p {
                return None;
            }
        }
        for (gi, s) in self.all_suffixes().iter().enumerate() {
            let slen = s.len();
            if slen == 0 {
                return Some(gi);
            }
            let mut ok = true;
            for j in 0..slen {
                if nib[40 - slen + j] != s[j] {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Some(gi);
            }
        }
        None
    }

    /// Expected number of attempts (~1 / sum over groups of 16^-(len)).
    /// This is an approximation: it treats each group as an independent event
    /// and sums the per-group hit probabilities, which slightly overestimates
    /// when groups overlap (they do not here — distinct fixed suffixes).
    pub fn expected_attempts(&self) -> f64 {
        let pfx = 16f64.powf(self.prefix.len() as f64);
        let mut total_p = 0.0;
        for s in self.all_suffixes() {
            total_p += 1.0 / 16f64.powf(s.len() as f64);
        }
        if total_p <= 0.0 {
            return f64::INFINITY;
        }
        (pfx / total_p).max(1.0)
    }
}

pub fn hex_to_nibbles(s: &str) -> Result<Vec<u8>, ConfigError> {
    let s = s.trim();
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    let mut out = Vec::with_capacity(s.len());
    for ch in s.chars() {
        let v = match ch {
            '0'..='9' => ch as u8 - b'0',
            'a'..='f' => ch as u8 - b'a' + 10,
            'A'..='F' => ch as u8 - b'A' + 10,
            _ => return Err(ConfigError::InvalidHex(s.to_string())),
        };
        out.push(v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_prefix_only() {
        let p = Pattern::parse("cafe", "").unwrap();
        assert_eq!(p.prefix, vec![0xC, 0xA, 0xF, 0xE]);
        assert!(p.suffix.is_empty());
        assert_eq!(p.suffix_group_count(), 1);
    }

    #[test]
    fn parse_strips_0x_and_is_case_insensitive() {
        let p = Pattern::parse("0XCaFe", "1234").unwrap();
        assert_eq!(p.prefix, vec![0xC, 0xA, 0xF, 0xE]);
        assert_eq!(p.suffix, vec![0x1, 0x2, 0x3, 0x4]);
    }

    #[test]
    fn parse_empty_pattern_rejected() {
        assert!(matches!(
            Pattern::parse("", ""),
            Err(ConfigError::EmptyPattern)
        ));
    }

    #[test]
    fn parse_invalid_hex_rejected() {
        assert!(matches!(
            Pattern::parse("xyz", ""),
            Err(ConfigError::InvalidHex(_))
        ));
    }

    #[test]
    fn parse_too_long_rejected() {
        let long = "a".repeat(41);
        assert!(matches!(
            Pattern::parse(&long, ""),
            Err(ConfigError::TooLong(_))
        ));
    }

    #[test]
    fn expected_attempts_grows() {
        let p = Pattern::parse("cafe", "").unwrap();
        assert_eq!(p.expected_attempts(), 16f64.powf(4.0));
    }

    #[test]
    fn parse_multi_equal_length_ok() {
        let p = Pattern::parse_multi("", "88888888", &["77777777".to_string()]).unwrap();
        assert_eq!(p.suffix_group_count(), 2);
        assert_eq!(p.suffix, vec![0x8; 8]);
        assert_eq!(p.alt_suffixes[0], vec![0x7; 8]);
        // Two 8-nibble groups: expected ~ 16^8 / 2.
        let exp = p.expected_attempts();
        assert!((exp - 16f64.powf(8.0) / 2.0).abs() < 1.0);
    }

    #[test]
    fn parse_multi_mismatched_length_rejected() {
        assert!(matches!(
            Pattern::parse_multi("a", "8888", &["77777777".to_string()]),
            Err(ConfigError::InvalidHex(_))
        ));
    }

    #[test]
    fn parse_multi_too_many_rejected() {
        let many: Vec<String> = (0..16).map(|_| "88888888".to_string()).collect();
        assert!(matches!(
            Pattern::parse_multi("a", "88888888", &many),
            Err(ConfigError::TooManySuffixes(_))
        ));
    }
}
