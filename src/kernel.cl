// kernel.cl — EVM vanity search (OpenCL, Montgomery form, little-endian)
//
// Rewritten for speed. The previous kernel used schoolbook multiply + a
// 40-pass reduction and bit-by-bit double-and-add (256 doubles + 128 adds),
// plus TWO full modular inverses (~384 multiplies each) per point. That
// capped throughput at ~1 Mkeys/s. This kernel uses the recipe mature tools
// use (profanity/VanitySearch):
//   * Montgomery CIOS multiply (no division, no multi-pass reduction)
//   * byte-windowed scalar multiplication over a 32x255 precomputed table of
//     (b * 256^pos) * G affine points — ~16 mixed adds, no double loop
//   * Jacobian accumulator + a single inversion per point
//
// Every field element is 8 little-endian 32-bit limbs in Montgomery form
// (a*R mod p). This is a verbatim port of src/mont.rs and src/ec.rs, which
// are proven correct on the CPU against num-bigint and the secp256k1 crate.
//
// Kernel pipeline (2 kernels):
//   derive_points : byte-windowed point-mul + affine conversion -> (x,y) bytes
//   hash_match    : keccak256(x||y) -> EVM address, prefix/suffix match

// ---- field constants (little-endian limbs) --------------------------------

constant uint P[8] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};
constant uint MPRIME = 0xD2253531u; // -P^-1 mod 2^32
constant uint R_MONT[8] = { 0x3D1u, 0x1u, 0, 0, 0, 0, 0, 0 }; // R = 2^256 mod p
constant uint R2_MONT[8] = { 0xE90A1u, 0x7A2u, 0x1u, 0, 0, 0, 0, 0 }; // R^2 = 2^512 mod p

// ---- field helpers --------------------------------------------------------

// All input pointer parameters use __generic so the helpers can be called
// with both __private (stack/local arrays) and __constant (P, R2_MONT, …)
// pointers. AMD's comgr strictly enforces pointer address space in static
// helpers — without __generic a call like `fe_ge(r, P)` fails with
// "changed address space of pointer". OpenCL 2.0+ is required; the kernel
// is rejected on devices that only speak 1.2 by the same probe that already
// skips Mac integrated GPUs.
static inline int fe_ge(const __generic uint* a, const __generic uint* b) {
    for (int i = 7; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1; // equal
}
static inline int fe_is_zero(const __generic uint* a) {
    for (int i = 0; i < 8; i++) if (a[i]) return 0;
    return 1;
}

// Montgomery multiply (CIOS): r = a * b * R^-1 mod p.
// Verbatim port of src/mont.rs::mont_mul (10-limb accumulator).
static void mont_mul(uint* r, const __generic uint* a, const __generic uint* b) {
    uint64_t t[10];
    for (int i = 0; i < 10; i++) t[i] = 0;

    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t v = t[j] + (uint64_t)a[i] * (uint64_t)b[j] + carry;
            t[j] = v & 0xFFFFFFFFULL;
            carry = v >> 32;
        }
        uint64_t v = t[8] + carry;
        t[8] = v & 0xFFFFFFFFULL;
        t[9] = v >> 32;

        uint m = (uint)((t[0] & 0xFFFFFFFFULL) * MPRIME);

        carry = 0;
        for (int j = 0; j < 8; j++) {
            v = t[j] + (uint64_t)m * (uint64_t)P[j] + carry;
            if (j >= 1) t[j - 1] = v & 0xFFFFFFFFULL;
            carry = v >> 32;
        }
        v = t[8] + carry;
        t[7] = v & 0xFFFFFFFFULL;
        uint64_t c2 = v >> 32;
        v = t[9] + c2;
        t[8] = v & 0xFFFFFFFFULL;
        t[9] = v >> 32;
    }

    for (int j = 0; j < 8; j++) r[j] = (uint)t[j];
    // Final reduction: t[8] is 0 or 1; subtract p once if >= p.
    if (t[8] > 0 || fe_ge(r, P)) {
        int64_t borrow = 0;
        for (int j = 0; j < 8; j++) {
            int64_t cur = (int64_t)r[j] - (int64_t)P[j] - borrow;
            if (cur < 0) { r[j] = (uint)(cur + (1LL << 32)); borrow = 1; }
            else { r[j] = (uint)cur; borrow = 0; }
        }
    }
}

static void mont_sqr(uint* r, const __generic uint* a) { mont_mul(r, a, a); }

// Montgomery addition (a + b) mod p, with a 9th carry limb.
static void mont_add(uint* r, const __generic uint* a, const __generic uint* b) {
    uint64_t t[9];
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t v = (uint64_t)a[i] + (uint64_t)b[i] + carry;
        t[i] = v & 0xFFFFFFFFULL;
        carry = v >> 32;
    }
    t[8] = carry;
    if (t[8] > 0 || fe_ge((uint*)t, P)) {
        int64_t borrow = 0;
        for (int i = 0; i < 8; i++) {
            int64_t cur = (int64_t)t[i] - (int64_t)P[i] - borrow;
            if (cur < 0) { t[i] = (uint64_t)(cur + (1LL << 32)); borrow = 1; }
            else { t[i] = (uint64_t)cur; borrow = 0; }
        }
    }
    for (int i = 0; i < 8; i++) r[i] = (uint)t[i];
}

// Montgomery subtraction (a - b) mod p.
static void mont_sub(uint* r, const __generic uint* a, const __generic uint* b) {
    int64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        int64_t cur = (int64_t)a[i] - (int64_t)b[i] - borrow;
        if (cur < 0) { r[i] = (uint)(cur + (1LL << 32)); borrow = 1; }
        else { r[i] = (uint)cur; borrow = 0; }
    }
    if (borrow > 0) {
        uint64_t carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t v = (uint64_t)r[i] + (uint64_t)P[i] + carry;
            r[i] = (uint)(v & 0xFFFFFFFFULL);
            carry = v >> 32;
        }
    }
}

// Montgomery doubling (== mont_add(a, a)).
static void mont_double(uint* r, const __generic uint* a) { mont_add(r, a, a); }

// Convert canonical -> Montgomery form (multiply by R^2).
static void to_mont(uint* r, const __generic uint* a) { mont_mul(r, a, R2_MONT); }

// Convert Montgomery -> canonical form (multiply by 1).
static void from_mont(uint* r, const __generic uint* a) {
    uint one[8] = { 1, 0, 0, 0, 0, 0, 0, 0 };
    mont_mul(r, a, one);
}

// Modular inverse via Fermat: a^-1 = a^(p-2), MSB-first square-and-multiply.
static void fe_inv(uint* r, const __generic uint* a) {
    uint e[8] = { 0xFFFFFC2Du, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                  0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu };
    uint res[8];
    for (int i = 0; i < 8; i++) res[i] = R_MONT[i];
    for (int i = 7; i >= 0; i--) {
        for (int bit = 31; bit >= 0; bit--) {
            uint t[8];
            mont_sqr(t, res);
            for (int k = 0; k < 8; k++) res[k] = t[k];
            if ((e[i] >> bit) & 1u) {
                uint u[8];
                mont_mul(u, res, a);
                for (int k = 0; k < 8; k++) res[k] = u[k];
            }
        }
    }
    for (int i = 0; i < 8; i++) r[i] = res[i];
}

// ---- elliptic curve (Jacobian, Montgomery form) ---------------------------

// Jacobian doubling R = 2*P (secp256k1 a=0).
static void j_double(uint* RX, uint* RY, uint* RZ,
                     const __generic uint* PX, const __generic uint* PY, const __generic uint* PZ) {
    if (fe_is_zero(PZ) || fe_is_zero(PY)) {
        for (int i = 0; i < 8; i++) { RX[i] = 0; RY[i] = 0; RZ[i] = 0; }
        return;
    }
    uint t[8], alpha[8], gamma[8], beta[8], t1[8], t2[8];

    mont_sqr(t, PX);          // t = X^2
    mont_double(alpha, t);    // 2*X^2
    mont_add(alpha, alpha, t); // 3*X^2
    mont_sqr(gamma, PY);      // Y^2
    mont_mul(beta, PX, gamma); // X*Y^2

    mont_sqr(t1, alpha);       // alpha^2
    mont_double(t2, beta);     // 2*beta
    mont_double(t2, t2);       // 4*beta
    mont_double(t2, t2);       // 8*beta
    mont_sub(RX, t1, t2);      // X3 = alpha^2 - 8*beta

    mont_mul(t1, PY, PZ);
    mont_double(RZ, t1);       // Z3 = 2*Y*Z

    mont_double(t1, beta);     // 2*beta
    mont_double(t1, t1);       // 4*beta
    mont_sub(t2, t1, RX);      // 4*beta - X3
    mont_mul(t1, alpha, t2);   // alpha*(4*beta - X3)
    mont_sqr(t2, gamma);       // gamma^2
    mont_double(t2, t2);
    mont_double(t2, t2);
    mont_double(t2, t2);       // 8*gamma^2
    mont_sub(RY, t1, t2);      // Y3
}

// Mixed Jacobian + affine addition R = P(Jacobian) + Q(affine), no inversion.
static void j_add_mixed(uint* RX, uint* RY, uint* RZ,
                        const __generic uint* PX, const __generic uint* PY, const __generic uint* PZ,
                        const __generic uint* qx, const __generic uint* qy) {
    if (fe_is_zero(PZ)) {
        for (int i = 0; i < 8; i++) { RX[i] = qx[i]; RY[i] = qy[i]; RZ[i] = R_MONT[i]; }
        return;
    }
    uint z1z1[8], u2[8], s2[8], h[8], r[8], z1cubed[8];
    mont_sqr(z1z1, PZ);
    mont_mul(u2, qx, z1z1);
    mont_mul(z1cubed, PZ, z1z1);
    mont_mul(s2, qy, z1cubed);

    mont_sub(h, u2, PX);
    mont_sub(r, s2, PY);

    if (fe_is_zero(h)) {
        if (fe_is_zero(r)) {
            j_double(RX, RY, RZ, PX, PY, PZ);
        } else {
            for (int i = 0; i < 8; i++) { RX[i] = 0; RY[i] = 0; RZ[i] = 0; }
        }
        return;
    }

    uint hh[8], i4[8], j[8], v[8], t1[8], t2[8], t3[8];
    mont_sqr(hh, h);
    mont_double(i4, hh);       // 2*HH
    mont_double(i4, i4);       // 4*HH
    mont_mul(j, h, i4);
    mont_double(r, r);         // 2*(S2 - Y1)
    mont_mul(v, PX, i4);

    mont_sqr(t1, r);           // r^2
    mont_sub(t2, t1, j);       // r^2 - J
    mont_double(t3, v);        // 2*V
    mont_sub(RX, t2, t3);      // X3 = r^2 - J - 2*V

    mont_sub(t1, v, RX);       // V - X3
    mont_mul(t2, r, t1);       // r*(V - X3)
    mont_mul(t3, PY, j);       // Y1*J
    mont_double(t3, t3);       // 2*Y1*J
    mont_sub(RY, t2, t3);      // Y3

    mont_add(t1, PZ, h);       // Z1 + H
    mont_sqr(t2, t1);          // (Z1+H)^2
    mont_sub(t3, t2, z1z1);    // - Z1Z1
    mont_sub(RZ, t3, hh);      // Z3 = (Z1+H)^2 - Z1Z1 - HH
}

// Convert Jacobian to affine (Montgomery form): x = X/Z^2, y = Y/Z^3.
static void j_to_affine(uint* Qx, uint* Qy,
                        const __generic uint* X, const __generic uint* Y, const __generic uint* Z) {
    if (fe_is_zero(Z)) {
        for (int i = 0; i < 8; i++) { Qx[i] = 0; Qy[i] = 0; }
        return;
    }
    uint z2[8], z3[8], invz[8], invz2[8], invz3[8];
    mont_sqr(z2, Z);
    mont_mul(z3, z2, Z);
    fe_inv(invz, Z);           // single inversion
    mont_sqr(invz2, invz);
    mont_mul(invz3, invz2, invz);
    mont_mul(Qx, X, invz2);
    mont_mul(Qy, Y, invz3);
}

// ---- keccak-256 -----------------------------------------------------------

static inline uint64_t rotl64(uint64_t x, uint64_t n) {
    return (x << n) | (x >> (64 - n));
}

static void keccak_f(uint64_t* st) {
    const uint64_t RC[24] = {
        0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
        0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
        0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
        0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
        0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
        0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL
    };
    const int ROTC[24] = {
        1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44
    };
    const int PILN[24] = {
        10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1
    };
    for (int round = 0; round < 24; round++) {
        uint64_t bc[5];
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20];
        for (int i = 0; i < 5; i++) {
            uint64_t t = bc[(i+4)%5] ^ rotl64(bc[(i+1)%5], (uint64_t)1);
            for (int j = 0; j < 5; j++) st[i + 5*j] ^= t;
        }
        uint64_t tmp = st[1];
        for (int i = 0; i < 24; i++) {
            int j = PILN[i];
            uint64_t t = st[j];
            st[j] = rotl64(tmp, (uint64_t)ROTC[i]);
            tmp = t;
        }
        uint64_t tc[5];
        for (int j = 0; j < 5; j++) {
            for (int i = 0; i < 5; i++) tc[i] = st[j*5 + i];
            for (int i = 0; i < 5; i++)
                st[j*5 + i] = tc[i] ^ ((~tc[(i+1)%5]) & tc[(i+2)%5]);
        }
        st[0] ^= RC[round];
    }
}

// keccak256 of the 64-byte (x || y) big-endian byte buffer -> 20-byte address.
static void keccak256_addr(const uint* xbytes, const uint* ybytes, uchar* out_addr) {
    // xbytes / ybytes are 8 u32 each holding 4 bytes, big-endian byte order
    // (xbytes[0] holds the most-significant 4 bytes of x).
    uint64_t st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;
    for (int lane = 0; lane < 8; lane++) {
        uint a = (lane < 4) ? xbytes[2 * lane]     : ybytes[2 * (lane - 4)];
        uint b = (lane < 4) ? xbytes[2 * lane + 1] : ybytes[2 * (lane - 4) + 1];
        uint64_t v = 0;
        v |= ((uint64_t)((a >> 24) & 0xFF));
        v |= ((uint64_t)((a >> 16) & 0xFF)) << 8;
        v |= ((uint64_t)((a >> 8)  & 0xFF)) << 16;
        v |= ((uint64_t)( a        & 0xFF)) << 24;
        v |= ((uint64_t)((b >> 24) & 0xFF)) << 32;
        v |= ((uint64_t)((b >> 16) & 0xFF)) << 40;
        v |= ((uint64_t)((b >> 8)  & 0xFF)) << 48;
        v |= ((uint64_t)( b        & 0xFF)) << 56;
        st[lane] = v;
    }
    st[8]  ^= (uint64_t)0x01;
    st[16] ^= (uint64_t)0x8000000000000000ULL;
    keccak_f(st);
    for (int i = 0; i < 20; i++) out_addr[i] = (uchar)(st[(12 + i) / 8] >> (8 * ((12 + i) % 8)));
}

// ---- key helpers ----------------------------------------------------------

// key = base + gid (little-endian add), gid added at the least-significant limb.
static void key_add_gid(uint* key, __global const uint* base, uint64_t gid) {
    for (int i = 0; i < 8; i++) key[i] = base[i];
    uint64_t carry = gid;
    for (int i = 0; i < 8 && carry; i++) {
        uint64_t s = (uint64_t)key[i] + carry;
        key[i] = (uint)s;
        carry = s >> 32;
    }
}

// Extract the i-th byte of a little-endian 8-limb key, in big-endian order
// (i == 0 is the most-significant byte, i == 31 the least-significant).
static uint key_byte(const uint* key, int i) {
    int word = 7 - i / 4;
    int shift = 24 - (i % 4) * 8;
    return (key[word] >> shift) & 0xFFu;
}

// ---- kernels --------------------------------------------------------------

// precomp layout: 32 columns x 255 points x 16 u32 (x[8] then y[8]), all in
// Montgomery form. column i holds (b * 256^(31-i)) * G for b = 1..255.
//
// derive_points: for each work item, compute (base+gid)*G and emit the affine
// public key as 16 big-endian u32 (x[8] then y[8], x[0] most significant).
__kernel void derive_points(__global uint* restrict base,
                            __global const uint* restrict precomp,
                            __global uint* restrict out) {
    size_t gid = get_global_id(0);

    uint key[8];
    key_add_gid(key, base, (uint64_t)gid);

    // Accumulator R in Jacobian form, starting at infinity (RZ == 0).
    uint RX[8], RY[8], RZ[8];
    for (int i = 0; i < 8; i++) { RX[i] = 0; RY[i] = 0; RZ[i] = 0; }

    for (int i = 0; i < 32; i++) {
        uint byte = key_byte(key, i);
        if (byte != 0) {
            uint qx[8], qy[8];
            uint off = (uint)i * 255u * 16u + (byte - 1u) * 16u;
            for (int k = 0; k < 8; k++) {
                qx[k] = precomp[off + k];
                qy[k] = precomp[off + 8 + k];
            }
            uint TX[8], TY[8], TZ[8];
            j_add_mixed(TX, TY, TZ, RX, RY, RZ, qx, qy);
            for (int k = 0; k < 8; k++) { RX[k] = TX[k]; RY[k] = TY[k]; RZ[k] = TZ[k]; }
        }
    }

    // Affine conversion, then de-Montgomery, then emit big-endian u32.
    uint ax[8], ay[8];
    j_to_affine(ax, ay, RX, RY, RZ);
    uint cx[8], cy[8];
    from_mont(cx, ax);
    from_mont(cy, ay);

    size_t off = gid * 16;
    for (int k = 0; k < 8; k++) {
        out[off + k]     = cx[7 - k]; // little-endian cx -> big-endian out
        out[off + 8 + k] = cy[7 - k];
    }
}

// hash_match: keccak256(x||y) -> EVM address, then prefix/suffix match.
// params layout (u32):
//   [0]      = prefix_len
//   [1]      = suffix_len (group 0)
//   [2..9]   = base[8] (little-endian, for private-key recovery)
//   [10..49] = prefix nibbles (40 slots)
//   [50..89] = group-0 suffix nibbles (40 slots)
//   [90]     = num_alt_suffixes
//   [91]     = alt suffix length
//   [92..]   = alt suffix nibbles, 40 slots per group
__kernel void hash_match(__global uint* restrict base,
                         __global const uint* restrict points,
                         __global int*  restrict out_found,
                         __global uint* restrict out_priv,
                         __global uchar* restrict out_addr,
                         __global uint* restrict params) {
    size_t gid = get_global_id(0);
    size_t off = gid * 16;

    uint xb[8], yb[8];
    for (int i = 0; i < 8; i++) {
        xb[i] = points[off + i];
        yb[i] = points[off + 8 + i];
    }

    uchar addr[20];
    keccak256_addr(xb, yb, addr);

    uint prefix_len = params[0];
    uint suffix_len = params[1];
    uint num_alt = params[90];
    uint alt_len = params[91];

    // Match prefix.
    int prefix_ok = 1;
    for (uint i = 0; i < prefix_len; i++) {
        uchar n = (i & 1u) ? (addr[i / 2] & 0xF) : ((addr[i / 2] >> 4) & 0xF);
        if (n != (uchar)(params[10 + i])) { prefix_ok = 0; break; }
    }
    if (!prefix_ok) return;

    // Match group 0.
    int match = 1;
    for (uint i = 0; i < suffix_len; i++) {
        uint idx = 40u - suffix_len + i;
        uchar n = (idx & 1u) ? (addr[idx / 2] & 0xF) : ((addr[idx / 2] >> 4) & 0xF);
        if (n != (uchar)(params[50 + i])) { match = 0; break; }
    }
    // Match alternative groups.
    for (uint g = 0; g < num_alt && !match; g++) {
        uint base_off = 92u + g * 40u;
        int matched_local = 1;
        for (uint i = 0; i < alt_len; i++) {
            uint idx = 40u - alt_len + i;
            uchar n = (idx & 1u) ? (addr[idx / 2] & 0xF) : ((addr[idx / 2] >> 4) & 0xF);
            if (n != (uchar)(params[base_off + i])) { matched_local = 0; break; }
        }
        if (matched_local) match = 1;
    }

    if (match) {
        int idx = atomic_inc(out_found);
        if (idx == 0) {
            uint key[8];
            key_add_gid(key, base, (uint64_t)gid);
            for (int i = 0; i < 8; i++) out_priv[i] = key[i];
            for (int i = 0; i < 20; i++) out_addr[i] = addr[i];
        }
    }
}
