//! Elliptic-curve point arithmetic over secp256k1, in Montgomery field form.
//!
//! This is the mathematical core of the kernel rewrite. The GPU path will do
//! byte-windowed scalar multiplication with a precomputed table of
//! `(b * 256^pos) * G` affine points, accumulating in Jacobian coordinates
//! (no per-add inversion), then convert to affine with a single inversion —
//! the exact recipe mature tools (profanity) use to reach ~100+ Mkeys/s.
//!
//! Everything here runs on the CPU in Montgomery form (see `crate::mont`) so
//! it can be proven correct against the `secp256k1` crate before being ported
//! verbatim to OpenCL. `generate_precomp` is also what the host will use at
//! runtime to build the GPU precomputation table.

use crate::mont::{from_mont, mont_add, mont_double, mont_mul, mont_sqr, mont_sub, to_mont, Fe};

/// secp256k1 group order `n` (big-endian bytes), for reducing table scalars.
const N_BYTES: [u8; 32] = [
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe,
    0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b, 0xbf, 0xd2, 0x5e, 0x8c, 0xd0, 0x36, 0x41, 0x41,
];

/// Generator G's affine coordinates, little-endian 32-bit limbs (canonical,
/// not Montgomery). `Fe` is little-endian, so these are the reverse of the
/// big-endian order used in `kernel.cl`.
pub const GX: [u32; 8] = [
    0x16f8_1798,
    0x59f2_815b,
    0x2dce_28d9,
    0x029b_fcdb,
    0xce87_0b07,
    0x55a0_6295,
    0xf9dc_bbac,
    0x79be_667e,
];
pub const GY: [u32; 8] = [
    0xfb10_d4b8,
    0x9c47_d08f,
    0xa685_5419,
    0xfd17_b448,
    0x0e11_08a8,
    0x5da4_fbfc,
    0x26a3_c465,
    0x483a_da77,
];

/// An affine point `(x, y)` in Montgomery form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Affine {
    pub x: Fe,
    pub y: Fe,
}

/// A Jacobian point `(X, Y, Z)` in Montgomery form. `Z == 0` is the point at
/// infinity.
#[derive(Clone, Copy, Debug)]
pub struct Jacobian {
    pub x: Fe,
    pub y: Fe,
    pub z: Fe,
}

impl Jacobian {
    pub const INF: Self = Self {
        x: [0; 8],
        y: [0; 8],
        z: [0; 8],
    };

    fn is_inf(&self) -> bool {
        self.z.iter().all(|&w| w == 0)
    }
}

/// The generator G in Montgomery form.
pub fn g_affine_mont() -> Affine {
    Affine {
        x: to_mont(&GX),
        y: to_mont(&GY),
    }
}

/// Jacobian point doubling `R = 2*P` (Montgomery form, secp256k1 a=0).
/// Formulae mirror the OpenCL `jdouble` (minus the dead `Z^2` term).
pub fn j_double(p: &Jacobian) -> Jacobian {
    if p.is_inf() || p.y.iter().all(|&w| w == 0) {
        return Jacobian::INF;
    }

    let t = mont_sqr(&p.x); // t = X^2
    let mut alpha = mont_double(&t); // 2*X^2
    alpha = mont_add(&alpha, &t); // 3*X^2

    let gamma = mont_sqr(&p.y); // Y^2
    let beta = mont_mul(&p.x, &gamma); // X*Y^2

    // X3 = alpha^2 - 8*beta
    let mut t1 = mont_sqr(&alpha);
    let mut t2 = mont_double(&beta);
    t2 = mont_double(&t2);
    t2 = mont_double(&t2); // 8*beta
    let x3 = mont_sub(&t1, &t2);

    // Z3 = 2*Y*Z
    let mut z3 = mont_mul(&p.y, &p.z);
    z3 = mont_double(&z3);

    // Y3 = alpha*(4*beta - X3) - 8*gamma^2
    t1 = mont_double(&beta);
    t1 = mont_double(&t1); // 4*beta
    t2 = mont_sub(&t1, &x3);
    t1 = mont_mul(&alpha, &t2); // alpha*(4*beta - X3)
    t2 = mont_sqr(&gamma);
    t2 = mont_double(&t2);
    t2 = mont_double(&t2);
    t2 = mont_double(&t2); // 8*gamma^2
    let y3 = mont_sub(&t1, &t2);

    Jacobian {
        x: x3,
        y: y3,
        z: z3,
    }
}

/// Mixed Jacobian + affine addition `R = P(Jacobian) + Q(affine)`.
/// Handles infinity and `P == Q` / `P == -Q`. No inversion.
///
/// All of `x`/`y`/`z` are Montgomery form; `z == R` (not 1) is the Jacobian
/// unit, and `z == 0` is the point at infinity.
pub fn j_add_mixed(p: &Jacobian, q: &Affine) -> Jacobian {
    if p.is_inf() {
        return Jacobian {
            x: q.x,
            y: q.y,
            z: crate::mont::R, // Montgomery unit
        };
    }

    let z1z1 = mont_sqr(&p.z);
    let u2 = mont_mul(&q.x, &z1z1);
    let z1_cubed = mont_mul(&p.z, &z1z1);
    let s2 = mont_mul(&q.y, &z1_cubed);

    let h = mont_sub(&u2, &p.x);
    let mut r = mont_sub(&s2, &p.y);

    if h.iter().all(|&w| w == 0) {
        if r.iter().all(|&w| w == 0) {
            return j_double(p);
        }
        return Jacobian::INF;
    }

    let hh = mont_sqr(&h);
    let i = mont_double(&mont_double(&hh)); // 4*HH
    let j = mont_mul(&h, &i);
    r = mont_double(&r); // 2*(S2 - Y1)
    let v = mont_mul(&p.x, &i);

    // X3 = r^2 - J - 2*V
    let mut t1 = mont_sqr(&r);
    let t2 = mont_sub(&t1, &j);
    let t3 = mont_double(&v);
    let x3 = mont_sub(&t2, &t3);

    // Y3 = r*(V - X3) - 2*Y1*J
    t1 = mont_sub(&v, &x3);
    let t2 = mont_mul(&r, &t1);
    let mut t3 = mont_mul(&p.y, &j);
    t3 = mont_double(&t3);
    let y3 = mont_sub(&t2, &t3);

    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    let t1 = mont_add(&p.z, &h);
    let t2 = mont_sqr(&t1);
    let t3 = mont_sub(&t2, &z1z1);
    let z3 = mont_sub(&t3, &hh);

    Jacobian {
        x: x3,
        y: y3,
        z: z3,
    }
}

/// Modular inverse via Fermat's little theorem: `a^-1 = a^(p-2)`, computed in
/// Montgomery form with square-and-multiply. One inversion = ~384 multiplies,
/// so the kernel avoids calling this per-add and instead does one final
/// inversion per point (or batches it with Montgomery's trick).
pub fn fe_inv(a: &Fe) -> Fe {
    // p - 2 = 2^256 - 2^32 - 979, little-endian limbs:
    let e: [u32; 8] = [
        0xffff_fc2d,
        0xffff_fffe,
        0xffff_ffff,
        0xffff_ffff,
        0xffff_ffff,
        0xffff_ffff,
        0xffff_ffff,
        0xffff_ffff,
    ];
    // Montgomery unit = R = to_mont(1).
    let one: Fe = [1, 0, 0, 0, 0, 0, 0, 0];
    let mut res = to_mont(&one);
    // Square-and-multiply from the MOST significant bit (e[7] bit 31) down to
    // the least significant (e[0] bit 0).
    for i in (0..8).rev() {
        for bit in (0..32).rev() {
            res = mont_sqr(&res);
            if (e[i] >> bit) & 1 == 1 {
                res = mont_mul(&res, a);
            }
        }
    }
    res
}

/// Convert a Jacobian point to affine (Montgomery form): `x = X/Z^2`, `y = Y/Z^3`.
pub fn j_to_affine(p: &Jacobian) -> Affine {
    if p.is_inf() {
        return Affine {
            x: [0; 8],
            y: [0; 8],
        };
    }
    let zinv = fe_inv(&p.z);
    let zz = mont_sqr(&zinv);
    let zzz = mont_mul(&zz, &zinv);
    Affine {
        x: mont_mul(&p.x, &zz),
        y: mont_mul(&p.y, &zzz),
    }
}

/// Byte-windowed scalar multiplication `R = k * G` using a precomputed table.
///
/// `precomp[i][b-1]` holds `(b * 256^(31-i)) * G` in affine Montgomery form
/// (i is the big-endian byte index of `k`). Result is the affine public key in
/// **canonical** (non-Montgomery) form.
pub fn point_mul(k: &[u8; 32], precomp: &[Vec<Affine>]) -> Affine {
    let mut r = Jacobian::INF;
    for i in 0..32 {
        let byte = k[i] as usize;
        if byte != 0 {
            let q = precomp[i][byte - 1];
            r = j_add_mixed(&r, &q);
        }
    }
    let a = j_to_affine(&r);
    Affine {
        x: from_mont(&a.x),
        y: from_mont(&a.y),
    }
}

/// Montgomery's trick batch inversion: given `xs`, returns `1/xs[i]` for all i,
/// using one field inversion + `3(N-1)` multiplies. `xs[i]` are in Montgomery
/// form; results are Montgomery form too.
pub fn batch_invert(xs: &[Fe]) -> Vec<Fe> {
    let n = xs.len();
    if n == 0 {
        return Vec::new();
    }
    // prefix[i] = xs[0] * xs[1] * ... * xs[i]  (Montgomery multiply)
    let mut prefix = vec![[0u32; 8]; n];
    let mut acc = crate::mont::R; // Montgomery unit
    for (i, &x) in xs.iter().enumerate() {
        acc = mont_mul(&acc, &x);
        prefix[i] = acc;
    }
    // Single inversion of the total product, then back-substitute.
    let mut inv = fe_inv(&acc);
    let mut result = vec![[0u32; 8]; n];
    for i in (0..n).rev() {
        let prev = if i == 0 {
            crate::mont::R
        } else {
            prefix[i - 1]
        };
        result[i] = mont_mul(&inv, &prev);
        inv = mont_mul(&inv, &xs[i]);
    }
    result
}

/// Convert 32 big-endian bytes to a little-endian 8-limb field element.
fn bytes_to_fe(b: &[u8]) -> Fe {
    let mut r = [0u32; 8];
    for (i, slot) in r.iter_mut().enumerate() {
        let s = 28 - 4 * i; // word 0 = b[28..32], word 7 = b[0..4]
        *slot = u32::from_be_bytes([b[s], b[s + 1], b[s + 2], b[s + 3]]);
    }
    r
}

fn big_to_bytes(k: &num_bigint::BigUint) -> [u8; 32] {
    let bytes = k.to_bytes_be();
    let mut out = [0u8; 32];
    let start = 32 - bytes.len();
    out[start..].copy_from_slice(&bytes);
    out
}

/// Generate the byte-windowed precomputation table.
///
/// `table[i][b-1]` holds `(b * 256^(31-i)) * G` in affine Montgomery form, for
/// big-endian byte index `i` and byte value `b ∈ 1..=255`. This is what the
/// host builds (once, at runtime) and uploads to the GPU.
pub fn generate_precomp() -> Vec<Vec<Affine>> {
    use num_bigint::BigUint;
    use secp256k1::{PublicKey, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let n = BigUint::from_bytes_be(&N_BYTES);
    let mut table = Vec::with_capacity(32);
    for i in 0..32 {
        let factor = BigUint::from(256u32).pow((31 - i) as u32);
        let mut col = Vec::with_capacity(255);
        for b in 1u32..=255 {
            let k = (BigUint::from(b) * &factor) % &n;
            let kbytes = big_to_bytes(&k);
            let sk = SecretKey::from_slice(&kbytes).expect("b*256^pos mod n is never zero");
            let pk = PublicKey::from_secret_key(&secp, &sk);
            let ser = pk.serialize_uncompressed();
            col.push(Affine {
                x: to_mont(&bytes_to_fe(&ser[1..33])),
                y: to_mont(&bytes_to_fe(&ser[33..65])),
            });
        }
        table.push(col);
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mont::R;
    use rand::RngCore;
    use secp256k1::{PublicKey, Secp256k1, SecretKey};

    fn rand_fe() -> Fe {
        use num_bigint::BigUint;
        let mut b = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut b);
        let x = BigUint::from_bytes_le(&b);
        // p as little-endian bytes from crate::mont::P (single source of truth).
        let mut pbytes = [0u8; 32];
        for (i, w) in crate::mont::P.iter().enumerate() {
            pbytes[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        let p = BigUint::from_bytes_le(&pbytes);
        let reduced = x % p;
        let bytes = reduced.to_bytes_le();
        let mut r = [0u32; 8];
        for (i, slot) in r.iter_mut().enumerate() {
            let s = 4 * i;
            *slot = if s + 4 <= bytes.len() {
                u32::from_le_bytes([bytes[s], bytes[s + 1], bytes[s + 2], bytes[s + 3]])
            } else {
                0
            };
        }
        r
    }

    #[test]
    fn point_mul_matches_secp() {
        let precomp = generate_precomp();
        let secp = Secp256k1::new();
        let mut tested = 0;
        for _ in 0..300 {
            let mut k = [0u8; 32];
            rand::rngs::OsRng.fill_bytes(&mut k);
            if let Ok(sk) = SecretKey::from_slice(&k) {
                let pk = PublicKey::from_secret_key(&secp, &sk);
                let ser = pk.serialize_uncompressed();
                let want_x = bytes_to_fe(&ser[1..33]);
                let want_y = bytes_to_fe(&ser[33..65]);
                let got = point_mul(&k, &precomp);
                assert_eq!(got.x, want_x, "x mismatch for key {:02x?}", k);
                assert_eq!(got.y, want_y, "y mismatch for key {:02x?}", k);
                tested += 1;
            }
        }
        assert!(tested > 150, "too few valid keys tested ({tested})");
    }

    #[test]
    fn batch_invert_matches_reference() {
        let xs: Vec<Fe> = (0..64).map(|_| rand_fe()).collect();
        let invs = batch_invert(&xs);
        for (x, inv) in xs.iter().zip(invs.iter()) {
            // x * inv == 1 in Montgomery form (product == R, the Montgomery unit)
            let prod = mont_mul(x, inv);
            assert_eq!(prod, R, "batch inversion wrong");
        }
    }

    #[test]
    fn precomp_has_expected_shape() {
        let t = generate_precomp();
        assert_eq!(t.len(), 32);
        for col in &t {
            assert_eq!(col.len(), 255);
        }
        // table[31][0] == G (byte 1 at position 31 => 1 * 256^0 * G == G)
        let g = g_affine_mont();
        assert_eq!(t[31][0], g);
    }
}
