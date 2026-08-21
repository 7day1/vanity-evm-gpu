//! Pattern parsing/validation: hex prefix/suffix -> nibble-value arrays.

#[derive(Debug)]
pub enum ConfigError {
    EmptyPattern,
    InvalidHex(String),
    TooLong(usize),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::EmptyPattern => write!(f, "prefix and suffix cannot both be empty"),
            ConfigError::InvalidHex(s) => write!(f, "invalid hex character in pattern: {:?}", s),
            ConfigError::TooLong(n) => write!(f, "pattern too long ({} > 40 hex chars)", n),
        }
    }
}

impl std::error::Error for ConfigError {}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub prefix: Vec<u8>, // nibble values 0-15
    pub suffix: Vec<u8>,
}

impl Pattern {
    pub fn parse(prefix_hex: &str, suffix_hex: &str) -> Result<Self, ConfigError> {
        let prefix = hex_to_nibbles(prefix_hex)?;
        let suffix = hex_to_nibbles(suffix_hex)?;
        if prefix.is_empty() && suffix.is_empty() {
            return Err(ConfigError::EmptyPattern);
        }
        let max = prefix.len().max(suffix.len());
        if max > 40 {
            return Err(ConfigError::TooLong(max));
        }
        Ok(Self { prefix, suffix })
    }

    /// Expected number of attempts (~16^(prefix_len + suffix_len)).
    pub fn expected_attempts(&self) -> f64 {
        let n = (self.prefix.len() + self.suffix.len()) as f64;
        16f64.powf(n)
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
}
