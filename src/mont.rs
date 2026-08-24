//! Montgomery field arithmetic over the secp256k1 prime field `F_p`.
//!
//! This is the mathematical foundation for the OpenCL kernel rewrite. The
//! kernel currently uses schoolbook 8x8 multiply + a 40-pass carry-and-fold
//! reduction, then does a full modular inverse (`fe_pow`, ~384 field multiplies)
//! **twice per point** when converting Jacobian -> affine. That is the single
//! biggest reason our GPU throughput is ~1 Mkeys/s instead of the ~100-300
//! Mkeys/s that mature tools (profanity, VanitySearch) reach on comparable
//! hardware.
//!
//! Montgomery form makes every field multiply a constant-time CIOS loop with
//! **no division and no multi-pass reduction**, and (combined with
//! Montgomery's trick for batch inversion) removes the per-point inverse
//! bottleneck entirely.
//!
//! # Representation
//!
//! A field element is 8 little-endian 32-bit limbs (`Fe = [u32; 8]`). In
//! Montgomery form an element `a` is stored as `a * R mod p` where `R = 2^256`.
//! Montgomery form is *additive* (`aR + bR == (a+b)R`), so add/sub are plain
//! modular add/sub; only multiply differs (CIOS).
//!
//! # Verification
//!
//! Every operation is checked against `num-bigint` in the test module below, so
//! the math is proven correct on the CPU before it is ever ported to OpenCL.
//! The OpenCL translation must be a 1:1 literal port of `mont_mul` / `mont_add`
//! / `mont_sub` — see the comments in `kernel.cl`'s replacement.

/// secp256k1 field prime `p = 2^256 - 2^32 - 977`, little-endian 32-bit limbs.
pub const P: [u32; 8] = [
    0xffff_fc2f,
    0xffff_fffe,
    0xffff_ffff,
    0xffff_ffff,
    0xffff_ffff,
    0xffff_ffff,
    0xffff_ffff,
    0xffff_ffff,
];

/// `mPrime = -p^{-1} mod 2^32`. Used by CIOS to pick the reduction factor.
pub const MPRIME: u32 = 0xd225_3531;

/// `R^2 mod p = 2^512 mod p`, little-endian. `to_mont(a) = mont_mul(a, R2)`.
pub const R2: [u32; 8] = [0x0e9_0a1, 0x7a2, 0x1, 0, 0, 0, 0, 0];

/// `R mod p = 2^256 mod p = 2^32 + 977`, little-endian. This is the Montgomery
/// representation of the field element 1 (the multiplicative identity).
pub const R: [u32; 8] = [0x3d1, 0x1, 0, 0, 0, 0, 0, 0];

/// A field element as 8 little-endian 32-bit limbs.
pub type Fe = [u32; 8];

/// `a >= b` as unsigned 256-bit comparison.
fn ge(a: &Fe, b: &Fe) -> bool {
    for i in (0..8).rev() {
        if a[i] > b[i] {
            return true;
        }
        if a[i] < b[i] {
            return false;
        }
    }
    true // equal
}

/// Modular addition `(a + b) mod p`. Works in or out of Montgomery form.
///
/// The sum may exceed 2^256 (a+b < 2p < 2^257), so we keep a 9th limb for the
/// carry and subtract `p` once. The subtraction borrow (when the carry is set,
/// the low 8 limbs are < p) is absorbed by the 9th limb.
pub fn mont_add(a: &Fe, b: &Fe) -> Fe {
    let mut t = [0u64; 9];
    let mut carry = 0u64;
    for i in 0..8 {
        let v = a[i] as u64 + b[i] as u64 + carry;
        t[i] = v & 0xffff_ffff;
        carry = v >> 32;
    }
    t[8] = carry;

    let low: Fe = [
        t[0] as u32,
        t[1] as u32,
        t[2] as u32,
        t[3] as u32,
        t[4] as u32,
        t[5] as u32,
        t[6] as u32,
        t[7] as u32,
    ];
    if t[8] > 0 || ge(&low, &P) {
        let mut borrow = 0i64;
        for i in 0..8 {
            let cur = t[i] as i64 - P[i] as i64 - borrow;
            if cur < 0 {
                t[i] = (cur + (1i64 << 32)) as u64;
                borrow = 1;
            } else {
                t[i] = cur as u64;
                borrow = 0;
            }
        }
        // The borrow (0 or 1) is absorbed by t[8] (which is 1 when set).
        debug_assert!(t[8] as i64 - borrow >= 0, "mont_add: borrow exceeds carry");
        t[8] = (t[8] as i64 - borrow) as u64;
    }

    let mut r: Fe = [0; 8];
    for i in 0..8 {
        r[i] = t[i] as u32;
    }
    r
}

/// Modular subtraction `(a - b) mod p`. Works in or out of Montgomery form.
pub fn mont_sub(a: &Fe, b: &Fe) -> Fe {
    let mut r = [0u32; 8];
    let mut borrow = 0i64;
    for i in 0..8 {
        let cur = a[i] as i64 - b[i] as i64 - borrow;
        if cur < 0 {
            r[i] = (cur + (1i64 << 32)) as u32;
            borrow = 1;
        } else {
            r[i] = cur as u32;
            borrow = 0;
        }
    }
    if borrow > 0 {
        // r = a - b + p
        let mut carry = 0u64;
        for i in 0..8 {
            let v = r[i] as u64 + P[i] as u64 + carry;
            r[i] = (v & 0xffff_ffff) as u32;
            carry = v >> 32;
        }
    }
    r
}

/// Montgomery multiply (CIOS): `r = a * b * R^{-1} mod p`.
///
/// This is the exact loop that must be ported verbatim to OpenCL. It keeps a
/// 10-limb accumulator `t[0..10]` in `u64` (each < 2^32): limb 9 carries the
/// overflow of the `t += a[i]*b` step (the product is 9 limbs wide), which is
/// folded back down by the `(t + m*p) >> 32` reduction. The final 9-limb value
/// is `< 2p`, so a single conditional subtract of `p` (whose borrow consumes
/// the extra high bit) yields the canonical 8-limb result.
///
/// Verified against `num-bigint` over 50k random cases in the Python prototype
/// and the Rust tests below.
// `needless_range_loop` is intentionally allowed: this loop must stay a literal
// 1:1 match of the OpenCL translation (indexed limbs, not iterator sugar).
#[allow(clippy::needless_range_loop)]
pub fn mont_mul(a: &Fe, b: &Fe) -> Fe {
    let mut t = [0u64; 10];

    for i in 0..8 {
        // (1) t += a[i] * b   (a[i]*b is 9 limbs wide)
        let mut carry: u64 = 0;
        for j in 0..8 {
            let v = t[j] + (a[i] as u64) * (b[j] as u64) + carry;
            t[j] = v & 0xffff_ffff;
            carry = v >> 32;
        }
        let v = t[8] + carry;
        t[8] = v & 0xffff_ffff;
        t[9] = v >> 32;

        // (2) m = t[0] * mPrime mod 2^32
        let m = ((t[0] as u32).wrapping_mul(MPRIME)) as u64;

        // (3) t = (t + m * p) >> 32  (t[0] becomes zero by construction)
        let mut carry: u64 = 0;
        for j in 0..8 {
            let v = t[j] + m * (P[j] as u64) + carry;
            if j >= 1 {
                t[j - 1] = v & 0xffff_ffff;
            }
            carry = v >> 32;
        }
        let v = t[8] + carry;
        t[7] = v & 0xffff_ffff;
        let c2 = v >> 32;
        let v = t[9] + c2;
        t[8] = v & 0xffff_ffff;
        t[9] = v >> 32;
    }

    let mut r: Fe = [0; 8];
    for j in 0..8 {
        r[j] = t[j] as u32;
    }
    // Final reduction: the 9-limb t (t[8] is 0 or 1) is < 2p; subtract p once.
    // The borrow from `r -= p` (when t[8]==1, r < p) is absorbed by t[8],
    // so the wrapped 8-limb result is exactly `t - p`.
    if t[8] > 0 || ge(&r, &P) {
        let mut borrow = 0i64;
        for j in 0..8 {
            let cur = r[j] as i64 - P[j] as i64 - borrow;
            if cur < 0 {
                r[j] = (cur + (1i64 << 32)) as u32;
                borrow = 1;
            } else {
                r[j] = cur as u32;
                borrow = 0;
            }
        }
    }
    r
}

/// Montgomery square.
pub fn mont_sqr(a: &Fe) -> Fe {
    mont_mul(a, a)
}

/// Convert a canonical value `< p` into Montgomery form: `a -> a*R mod p`.
pub fn to_mont(a: &Fe) -> Fe {
    mont_mul(a, &R2)
}

/// Convert out of Montgomery form: `aR -> a` (i.e. `mont_mul(a, 1)`).
pub fn from_mont(a: &Fe) -> Fe {
    let one: Fe = [1, 0, 0, 0, 0, 0, 0, 0];
    mont_mul(a, &one)
}

/// Doubling in Montgomery form (== mont_add(a, a)). Reuses `mont_add` so the
/// 2^256 carry is handled correctly (the left-shift here can overflow).
pub fn mont_double(a: &Fe) -> Fe {
    mont_add(a, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;
    use num_traits::{One, Zero};

    fn to_big(a: &Fe) -> BigUint {
        let mut v = BigUint::zero();
        for i in (0..8).rev() {
            v = (v << 32) | BigUint::from(a[i]);
        }
        v
    }

    fn from_big(x: &BigUint) -> Fe {
        let mut r = [0u32; 8];
        for (i, slot) in r.iter_mut().enumerate() {
            let limb = (x >> (32 * i)) & BigUint::from(0xffff_ffffu32);
            *slot = limb.iter_u32_digits().next().unwrap_or(0);
        }
        r
    }

    fn p_big() -> BigUint {
        // Reconstruct p from its little-endian limbs (single source of truth).
        to_big(&P)
    }

    fn rand_fe() -> Fe {
        // Sample a full 256-bit value, then reduce mod p. Values near p are
        // what stress the add/sub carry and borrow paths (a+b can exceed 2^256).
        use rand::RngCore;
        let mut b = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut b);
        let mut r = [0u32; 8];
        for (i, slot) in r.iter_mut().enumerate() {
            *slot = u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        from_big(&(to_big(&r) % p_big()))
    }

    #[test]
    fn mprime_correct() {
        // p[0] * mPrime == -1 (mod 2^32) == 0xFFFFFFFF
        assert_eq!(P[0].wrapping_mul(MPRIME), 0xffff_ffff);
    }

    #[test]
    fn to_from_mont_roundtrip() {
        for _ in 0..100 {
            let a = rand_fe();
            let am = to_mont(&a);
            let back = from_mont(&am);
            assert_eq!(back, a, "to_mont/from_mont roundtrip failed");
        }
    }

    #[test]
    fn mont_mul_matches_reference() {
        let p = p_big();
        for _ in 0..2000 {
            let a = rand_fe();
            let b = rand_fe();
            let am = to_mont(&a);
            let bm = to_mont(&b);
            let got = from_mont(&mont_mul(&am, &bm));
            let want = (to_big(&a) * to_big(&b)) % &p;
            assert_eq!(to_big(&got), want, "mont_mul mismatch");
        }
    }

    #[test]
    fn mont_add_sub_match_reference() {
        let p = p_big();
        for _ in 0..1000 {
            let a = rand_fe();
            let b = rand_fe();
            let add = mont_add(&a, &b);
            assert_eq!(to_big(&add), (to_big(&a) + to_big(&b)) % &p);
            let sub = mont_sub(&a, &b);
            assert_eq!(to_big(&sub), ((to_big(&a) + &p) - to_big(&b)) % &p);
        }
    }

    #[test]
    fn mont_mul_additive_identity() {
        // a * 1 == a (in Montgomery form, 1 is represented by R).
        let r = from_big(&((BigUint::one() << 256u32) % p_big())); // R mod p
        for _ in 0..50 {
            let a = rand_fe();
            let am = to_mont(&a);
            assert_eq!(mont_mul(&am, &r), am, "multiplicative identity failed");
        }
    }

    #[test]
    fn edge_cases() {
        let p = p_big();
        let zero: Fe = [0; 8];
        let one: Fe = [1, 0, 0, 0, 0, 0, 0, 0];
        let pm1 = from_big(&(&p - BigUint::one()));
        // 0 * anything == 0
        assert_eq!(mont_mul(&zero, &one), zero);
        assert_eq!(mont_mul(&one, &zero), zero);
        // (p-1) * 1 == p-1 (in Montgomery form: multiply by R leaves it fixed)
        assert_eq!(mont_mul(&to_mont(&pm1), &to_mont(&one)), to_mont(&pm1));
        // (p-1) + 1 == 0
        assert_eq!(mont_add(&pm1, &one), zero);
        // 0 - 1 == p-1
        assert_eq!(mont_sub(&zero, &one), pm1);
        // to_mont(0) == 0, to_mont(1) == R
        assert_eq!(to_mont(&zero), zero);
        assert_eq!(
            to_mont(&one),
            from_big(&((BigUint::one() << 256u32) % p_big()))
        );
    }
}
