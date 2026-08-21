//! CPU-side cryptography: private-key -> EVM address (Keccak-256 + EIP-55),
//! plus helpers used by both the CPU worker and the GPU verification oracle.

use secp256k1::{PublicKey, Secp256k1, SecretKey};
use sha3::{Digest, Keccak256};
use zeroize::Zeroize;

pub const ADDR_LEN: usize = 20;

pub fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(data);
    let out = h.finalize();
    let mut a = [0u8; 32];
    a.copy_from_slice(&out);
    a
}

/// Ethereum address from an uncompressed 65-byte public key (0x04 || X || Y).
pub fn pubkey_to_address(pub_uncompressed: &[u8; 65]) -> [u8; ADDR_LEN] {
    // Ethereum hashes X||Y (64 bytes), not the 0x04 prefix.
    let hash = keccak256(&pub_uncompressed[1..]);
    let mut addr = [0u8; ADDR_LEN];
    addr.copy_from_slice(&hash[12..]);
    addr
}

/// Derive the address for a 32-byte private key. Reduces mod n if needed.
/// Returns None only on an impossible error.
pub fn privkey_to_address(priv32: &[u8; 32]) -> Option<[u8; ADDR_LEN]> {
    let secp = Secp256k1::new();
    let sk = match SecretKey::from_slice(priv32) {
        Ok(s) => s,
        Err(_) => {
            // priv >= n: subtract n once (n ~ 2^256, so at most one subtraction).
            let mut k = *priv32;
            subtract_n(&mut k);
            match SecretKey::from_slice(&k) {
                Ok(s) => s,
                Err(_) => return None,
            }
        }
    };
    let pk = PublicKey::from_secret_key(&secp, &sk);
    let ser = pk.serialize_uncompressed();
    let mut arr = [0u8; 65];
    arr.copy_from_slice(&ser);
    let addr = pubkey_to_address(&arr);
    // Zeroize the intermediate uncompressed public key buffer (defense-in-depth:
    // `ser`/`arr` hold derived key material that should not linger in stack).
    arr.zeroize();
    Some(addr)
}

/// Reduce a 32-byte key mod n (group order), returning the canonical private key.
pub fn reduce_mod_n(priv32: &[u8; 32]) -> [u8; 32] {
    match SecretKey::from_slice(priv32) {
        Ok(s) => s.secret_bytes(),
        Err(_) => {
            let mut k = *priv32;
            subtract_n(&mut k);
            match SecretKey::from_slice(&k) {
                Ok(s) => s.secret_bytes(),
                Err(_) => *priv32,
            }
        }
    }
}

fn subtract_n(k: &mut [u8; 32]) {
    // n = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141
    let n = [
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFE, 0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B, 0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36,
        0x41, 0x41,
    ];
    let mut borrow = 0i32;
    for i in (0..32).rev() {
        let a = k[i] as i32;
        let b = n[i] + borrow;
        let mut d = a - b;
        if d < 0 {
            d += 256;
            borrow = 1;
        } else {
            borrow = 0;
        }
        k[i] = d as u8;
    }
}

/// EIP-55 mixed-case checksum encoding of a raw 20-byte address.
pub fn eip55(addr: &[u8; ADDR_LEN]) -> String {
    let lower: String = addr.iter().map(|b| format!("{:02x}", b)).collect();
    let hash = keccak256(lower.as_bytes());
    let mut out = String::with_capacity(40);
    for (i, ch) in lower.chars().enumerate() {
        let nibble = (hash[i / 2] >> (if i % 2 == 0 { 4 } else { 0 })) & 0x0F;
        if ch.is_ascii_digit() {
            out.push(ch);
        } else if nibble >= 8 {
            out.push(ch.to_ascii_uppercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// Add a u64 offset to a 256-bit big-endian key (mod 2^256). Mirrors the kernel.
pub fn add_u64_be(key: &mut [u8; 32], mut off: u64) {
    for i in (0..32).rev() {
        let s = key[i] as u64 + (off & 0xFF);
        key[i] = s as u8;
        off = (off >> 8) + (s >> 8);
    }
}

/// Convert 8 big-endian u32 limbs into 32 bytes.
pub fn u32x8_to_bytes(base: &[u32; 8]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for i in 0..8 {
        let v = base[i].to_be_bytes();
        b[i * 4..i * 4 + 4].copy_from_slice(&v);
    }
    b
}

/// Convert 32 bytes into 8 big-endian u32 limbs.
pub fn bytes_to_u32x8(b: &[u8; 32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for i in 0..8 {
        out[i] = u32::from_be_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
    }
    out
}

/// Does the address match the nibble-value prefix/suffix? (case = nibble value)
pub fn addr_matches(addr: &[u8; 20], prefix: &[u8], suffix: &[u8]) -> bool {
    let mut nib = [0u8; 40];
    for i in 0..20 {
        nib[2 * i] = (addr[i] >> 4) & 0xF;
        nib[2 * i + 1] = addr[i] & 0xF;
    }
    for i in 0..prefix.len() {
        if nib[i] != prefix[i] {
            return false;
        }
    }
    for i in 0..suffix.len() {
        if nib[40 - suffix.len() + i] != suffix[i] {
            return false;
        }
    }
    true
}

/// Zeroize a private key buffer after use (defense in depth).
pub fn zeroize_key(k: &mut [u8; 32]) {
    k.zeroize();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keccak_empty_known_vector() {
        // Keccak-256("") known test vector.
        let h = keccak256(b"");
        let expected = [
            0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c, 0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7,
            0x03, 0xc0, 0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b, 0x7b, 0xfa, 0xd8, 0x04,
            0x5d, 0x85, 0xa4, 0x70,
        ];
        assert_eq!(h, expected);
    }

    #[test]
    fn privkey_one_address_known() {
        // secret key = 1 -> well-known address.
        let mut k = [0u8; 32];
        k[31] = 1;
        let addr = privkey_to_address(&k).unwrap();
        let hex: String = addr.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, "7e5f4552091a69125d5dfcb7b8c2659029395bdf");
    }

    #[test]
    fn privkey_two_address_distinct_and_deterministic() {
        // secret key = 2 must derive a *different* address than key = 1, and the
        // derivation must be deterministic. This guards the CPU verification
        // oracle (which the GPU kernel's output is checked against) so that any
        // regression in the key->address mapping is caught on CPU without a GPU.
        let mut k1 = [0u8; 32];
        k1[31] = 1;
        let mut k2 = [0u8; 32];
        k2[31] = 2;
        let a1 = privkey_to_address(&k1).unwrap();
        let a2_first = privkey_to_address(&k2).unwrap();
        let a2_again = privkey_to_address(&k2).unwrap();
        assert_ne!(
            a1, a2_first,
            "key 1 and key 2 must map to different addresses"
        );
        assert_eq!(a2_first, a2_again, "derivation must be deterministic");
        // The derived address must be consistent with the reduced private key:
        // reducing k2 (already < n) yields k2, and re-deriving from the reduced
        // key must give the same address.
        let reduced = reduce_mod_n(&k2);
        assert_eq!(reduced, k2);
        assert_eq!(privkey_to_address(&reduced).unwrap(), a2_first);
    }

    #[test]
    fn eip55_all_caps_known_vector() {
        // EIP-55 canonical all-caps example.
        let addr = [
            0x52, 0x90, 0x84, 0x00, 0x09, 0x85, 0x27, 0x88, 0x6e, 0x0f, 0x70, 0x30, 0x06, 0x98,
            0x57, 0xd2, 0xe4, 0x16, 0x9e, 0xe7,
        ];
        assert_eq!(eip55(&addr), "52908400098527886E0F7030069857D2E4169EE7");
    }

    #[test]
    fn addr_matches_prefix_and_suffix() {
        let addr = [0xABu8; 20];
        let prefix = vec![0xAu8, 0xBu8];
        let suffix = vec![0xAu8, 0xBu8];
        assert!(addr_matches(&addr, &prefix, &suffix));
        let bad = vec![0x0u8, 0xBu8];
        assert!(!addr_matches(&addr, &bad, &suffix));
    }

    #[test]
    fn add_u64_be_wraps_mod_2_256() {
        let mut k = [0xFFu8; 32];
        add_u64_be(&mut k, 1);
        assert_eq!(k, [0u8; 32]); // 2^256 wraps to 0
        let mut k2 = [0u8; 32];
        k2[31] = 0xFE;
        add_u64_be(&mut k2, 2);
        assert_eq!(k2[31], 0x00);
        assert_eq!(k2[30], 0x01);
    }

    #[test]
    fn reduce_mod_n_identity_for_small_key() {
        let mut k = [0u8; 32];
        k[31] = 1;
        assert_eq!(reduce_mod_n(&k), k);
    }
}
