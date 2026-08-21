// kernel.cl — EVM vanity search (OpenCL)
//
// Host-side note: this file is compiled together with a small stdint prelude
// (see gpu.rs) because Apple's OpenCL 1.2 compiler does not auto-inject
// <stdint.h>. All fixed-width 64-bit state uses int64_t/uint64_t.
//
// Portability notes for Apple OpenCL 1.2 (Intel UHD 630 / AMD Radeon Pro 560X):
//   * The compiler mis-handles dynamic indexing into file-scope `constant`
//     arrays, so keccak_f keeps its round constants as function-local `const`.
//   * The 64-bit rotate() builtin is mis-compiled on the Radeon driver, so we
//     use a manual rotl64() definition.
//   * Affine scalar multiplication needs one modular inverse per point
//     operation; on the Intel UHD 630 that exceeds the macOS display-GPU
//     watchdog for private keys with many 1-bits. We therefore use Jacobian
//     projective coordinates: doubling and addition need no inverses, and only
//     the final conversion back to affine performs a single inverse.
//   * scalar_mul() and keccak256_addr() were historically cross-optimized
//     incorrectly on Radeon, so we keep a three-pass pipeline connected by
//     global scratch buffers:
//       derive_pubkeys : key -> (Qx, Qy)
//       hash_addrs     : (Qx, Qy) -> EVM address bytes
//       match_addrs    : address -> prefix/suffix match
//
// The host re-derives every candidate address on the CPU, so a buggy kernel can
// never emit a mismatched private key/address pair.

#define NL 8

constant uint P[8] = {
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFEu, 0xFFFFFC2Fu
};
constant uint GX[8] = {
    0x79BE667Eu, 0xF9DCBBACu, 0x55A06295u, 0xCE870B07u,
    0x029BFCDBu, 0x2DCE28D9u, 0x59F2815Bu, 0x16F81798u
};
constant uint GY[8] = {
    0x483ADA77u, 0x26A3C465u, 0x5DA4FBFCu, 0x0E1108A8u,
    0xFD17B448u, 0xA6855419u, 0x9C47D08Fu, 0xFB10D4B8u
};
constant uint PM2[8] = {
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFEu, 0xFFFFFC2Du
};

// ---- field helpers -------------------------------------------------------

static inline int fe_cmp(const uint* a, const uint* b) {
    for (int i = 0; i < NL; i++) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}
static inline int fe_ge(const uint* a, const uint* b) { return fe_cmp(a, b) >= 0; }
static inline int fe_is_zero(const uint* a) {
    for (int i = 0; i < NL; i++) if (a[i]) return 0;
    return 1;
}

static inline void fe_sub(uint* r, const uint* a, const uint* b) {
    int64_t borrow = 0;
    for (int i = NL - 1; i >= 0; i--) {
        int64_t cur = (int64_t)a[i] - (int64_t)b[i] - borrow;
        if (cur < 0) { cur += (1LL << 32); borrow = 1; }
        else borrow = 0;
        r[i] = (uint)cur;
    }
}

static inline void fe_add_raw(uint* r, const uint* a, const uint* b) {
    uint64_t c = 0;
    for (int i = NL - 1; i >= 0; i--) {
        c = (uint64_t)a[i] + (uint64_t)b[i] + (c >> 32);
        r[i] = (uint)(c & 0xFFFFFFFFULL);
    }
}

static inline void fe_sub_mod(uint* r, const uint* a, const uint* b) {
    uint d[8];
    int64_t borrow = 0;
    for (int i = NL - 1; i >= 0; i--) {
        int64_t cur = (int64_t)a[i] - (int64_t)b[i] - borrow;
        if (cur < 0) { cur += (1LL << 32); borrow = 1; }
        else borrow = 0;
        d[i] = (uint)cur;
    }
    if (!borrow) {
        for (int i = 0; i < NL; i++) r[i] = d[i];
    } else {
        uint p[8];
        for (int i = 0; i < NL; i++) p[i] = P[i];
        fe_add_raw(r, d, p);
    }
}

static inline void fe_add(uint* r, const uint* a, const uint* b) {
    uint64_t c = 0;
    for (int i = NL - 1; i >= 0; i--) {
        c = (uint64_t)a[i] + (uint64_t)b[i] + (c >> 32);
        r[i] = (uint)(c & 0xFFFFFFFFULL);
    }
    uint p[8];
    for (int i = 0; i < NL; i++) p[i] = P[i];
    if (c >> 32) { fe_sub(r, r, p); }
    else if (fe_ge(r, p)) { fe_sub(r, r, p); }
}

static void fe_reduce_512(uint* r, uint64_t* acc) {
    // Carry-only reduction. Each pass normalizes the high limbs and folds any
    // overflow in limbs 0..7 back into limbs 7..15 using 2^256 ≡ 2^32 + 977
    // (mod P). Written with fully unrolled, constant-index statements (no
    // loop-variable array indexing) for portability on Apple OpenCL 1.2 / Radeon,
    // whose compiler mis-compiles dynamic `acc[7+k]`-style indexing.
    for (int pass = 0; pass < 40; pass++) {
        // carry normalize (constant indices)
        acc[14] += acc[15] >> 32; acc[15] &= 0xFFFFFFFFULL;
        acc[13] += acc[14] >> 32; acc[14] &= 0xFFFFFFFFULL;
        acc[12] += acc[13] >> 32; acc[13] &= 0xFFFFFFFFULL;
        acc[11] += acc[12] >> 32; acc[12] &= 0xFFFFFFFFULL;
        acc[10] += acc[11] >> 32; acc[11] &= 0xFFFFFFFFULL;
        acc[9]  += acc[10] >> 32; acc[10] &= 0xFFFFFFFFULL;
        acc[8]  += acc[9]  >> 32; acc[9]  &= 0xFFFFFFFFULL;
        acc[7]  += acc[8]  >> 32; acc[8]  &= 0xFFFFFFFFULL;
        acc[6]  += acc[7]  >> 32; acc[7]  &= 0xFFFFFFFFULL;
        acc[5]  += acc[6]  >> 32; acc[6]  &= 0xFFFFFFFFULL;
        acc[4]  += acc[5]  >> 32; acc[5]  &= 0xFFFFFFFFULL;
        acc[3]  += acc[4]  >> 32; acc[4]  &= 0xFFFFFFFFULL;
        acc[2]  += acc[3]  >> 32; acc[3]  &= 0xFFFFFFFFULL;
        acc[1]  += acc[2]  >> 32; acc[2]  &= 0xFFFFFFFFULL;
        acc[0]  += acc[1]  >> 32; acc[1]  &= 0xFFFFFFFFULL;

        uint64_t hv0 = acc[0], hv1 = acc[1], hv2 = acc[2], hv3 = acc[3];
        uint64_t hv4 = acc[4], hv5 = acc[5], hv6 = acc[6], hv7 = acc[7];
        uint64_t any = hv0 | hv1 | hv2 | hv3 | hv4 | hv5 | hv6 | hv7;
        if (!any) break;
        acc[0] = 0; acc[1] = 0; acc[2] = 0; acc[3] = 0;
        acc[4] = 0; acc[5] = 0; acc[6] = 0; acc[7] = 0;
        // fold high 256 bits back: acc[7+k] += hv[k]; acc[8+k] += hv[k]*977
        acc[7]  += hv0;          acc[8]  += hv0 * 977ULL;
        acc[8]  += hv1;          acc[9]  += hv1 * 977ULL;
        acc[9]  += hv2;          acc[10] += hv2 * 977ULL;
        acc[10] += hv3;          acc[11] += hv3 * 977ULL;
        acc[11] += hv4;          acc[12] += hv4 * 977ULL;
        acc[12] += hv5;          acc[13] += hv5 * 977ULL;
        acc[13] += hv6;          acc[14] += hv6 * 977ULL;
        acc[14] += hv7;          acc[15] += hv7 * 977ULL;
    }
    uint rr[8];
    for (int i = 0; i < 8; i++) rr[i] = (uint)acc[8 + i];
    uint p[8];
    for (int i = 0; i < NL; i++) p[i] = P[i];
    while (fe_ge(rr, p)) fe_sub(rr, rr, p);
    for (int i = 0; i < NL; i++) r[i] = rr[i];
}

static void fe_mul(uint* r, const uint* a, const uint* b) {
    uint64_t acc[16];
    for (int i = 0; i < 16; i++) acc[i] = 0;
    // Fully unrolled 8x8 schoolbook multiply. Constant indices only — no
    // loop-variable array indexing — for Apple OpenCL 1.2 / Radeon portability.
    // a[i]*b[j] contributes to acc[i+j] (high 32 bits) and acc[i+j+1] (low 32 bits).
    // i=0
    acc[0]  += ((uint64_t)a[0] * (uint64_t)b[0]) >> 32;  acc[1]  += ((uint64_t)a[0] * (uint64_t)b[0]) & 0xFFFFFFFFULL;
    acc[1]  += ((uint64_t)a[0] * (uint64_t)b[1]) >> 32;  acc[2]  += ((uint64_t)a[0] * (uint64_t)b[1]) & 0xFFFFFFFFULL;
    acc[2]  += ((uint64_t)a[0] * (uint64_t)b[2]) >> 32;  acc[3]  += ((uint64_t)a[0] * (uint64_t)b[2]) & 0xFFFFFFFFULL;
    acc[3]  += ((uint64_t)a[0] * (uint64_t)b[3]) >> 32;  acc[4]  += ((uint64_t)a[0] * (uint64_t)b[3]) & 0xFFFFFFFFULL;
    acc[4]  += ((uint64_t)a[0] * (uint64_t)b[4]) >> 32;  acc[5]  += ((uint64_t)a[0] * (uint64_t)b[4]) & 0xFFFFFFFFULL;
    acc[5]  += ((uint64_t)a[0] * (uint64_t)b[5]) >> 32;  acc[6]  += ((uint64_t)a[0] * (uint64_t)b[5]) & 0xFFFFFFFFULL;
    acc[6]  += ((uint64_t)a[0] * (uint64_t)b[6]) >> 32;  acc[7]  += ((uint64_t)a[0] * (uint64_t)b[6]) & 0xFFFFFFFFULL;
    acc[7]  += ((uint64_t)a[0] * (uint64_t)b[7]) >> 32;  acc[8]  += ((uint64_t)a[0] * (uint64_t)b[7]) & 0xFFFFFFFFULL;
    // i=1
    acc[1]  += ((uint64_t)a[1] * (uint64_t)b[0]) >> 32;  acc[2]  += ((uint64_t)a[1] * (uint64_t)b[0]) & 0xFFFFFFFFULL;
    acc[2]  += ((uint64_t)a[1] * (uint64_t)b[1]) >> 32;  acc[3]  += ((uint64_t)a[1] * (uint64_t)b[1]) & 0xFFFFFFFFULL;
    acc[3]  += ((uint64_t)a[1] * (uint64_t)b[2]) >> 32;  acc[4]  += ((uint64_t)a[1] * (uint64_t)b[2]) & 0xFFFFFFFFULL;
    acc[4]  += ((uint64_t)a[1] * (uint64_t)b[3]) >> 32;  acc[5]  += ((uint64_t)a[1] * (uint64_t)b[3]) & 0xFFFFFFFFULL;
    acc[5]  += ((uint64_t)a[1] * (uint64_t)b[4]) >> 32;  acc[6]  += ((uint64_t)a[1] * (uint64_t)b[4]) & 0xFFFFFFFFULL;
    acc[6]  += ((uint64_t)a[1] * (uint64_t)b[5]) >> 32;  acc[7]  += ((uint64_t)a[1] * (uint64_t)b[5]) & 0xFFFFFFFFULL;
    acc[7]  += ((uint64_t)a[1] * (uint64_t)b[6]) >> 32;  acc[8]  += ((uint64_t)a[1] * (uint64_t)b[6]) & 0xFFFFFFFFULL;
    acc[8]  += ((uint64_t)a[1] * (uint64_t)b[7]) >> 32;  acc[9]  += ((uint64_t)a[1] * (uint64_t)b[7]) & 0xFFFFFFFFULL;
    // i=2
    acc[2]  += ((uint64_t)a[2] * (uint64_t)b[0]) >> 32;  acc[3]  += ((uint64_t)a[2] * (uint64_t)b[0]) & 0xFFFFFFFFULL;
    acc[3]  += ((uint64_t)a[2] * (uint64_t)b[1]) >> 32;  acc[4]  += ((uint64_t)a[2] * (uint64_t)b[1]) & 0xFFFFFFFFULL;
    acc[4]  += ((uint64_t)a[2] * (uint64_t)b[2]) >> 32;  acc[5]  += ((uint64_t)a[2] * (uint64_t)b[2]) & 0xFFFFFFFFULL;
    acc[5]  += ((uint64_t)a[2] * (uint64_t)b[3]) >> 32;  acc[6]  += ((uint64_t)a[2] * (uint64_t)b[3]) & 0xFFFFFFFFULL;
    acc[6]  += ((uint64_t)a[2] * (uint64_t)b[4]) >> 32;  acc[7]  += ((uint64_t)a[2] * (uint64_t)b[4]) & 0xFFFFFFFFULL;
    acc[7]  += ((uint64_t)a[2] * (uint64_t)b[5]) >> 32;  acc[8]  += ((uint64_t)a[2] * (uint64_t)b[5]) & 0xFFFFFFFFULL;
    acc[8]  += ((uint64_t)a[2] * (uint64_t)b[6]) >> 32;  acc[9]  += ((uint64_t)a[2] * (uint64_t)b[6]) & 0xFFFFFFFFULL;
    acc[9]  += ((uint64_t)a[2] * (uint64_t)b[7]) >> 32;  acc[10] += ((uint64_t)a[2] * (uint64_t)b[7]) & 0xFFFFFFFFULL;
    // i=3
    acc[3]  += ((uint64_t)a[3] * (uint64_t)b[0]) >> 32;  acc[4]  += ((uint64_t)a[3] * (uint64_t)b[0]) & 0xFFFFFFFFULL;
    acc[4]  += ((uint64_t)a[3] * (uint64_t)b[1]) >> 32;  acc[5]  += ((uint64_t)a[3] * (uint64_t)b[1]) & 0xFFFFFFFFULL;
    acc[5]  += ((uint64_t)a[3] * (uint64_t)b[2]) >> 32;  acc[6]  += ((uint64_t)a[3] * (uint64_t)b[2]) & 0xFFFFFFFFULL;
    acc[6]  += ((uint64_t)a[3] * (uint64_t)b[3]) >> 32;  acc[7]  += ((uint64_t)a[3] * (uint64_t)b[3]) & 0xFFFFFFFFULL;
    acc[7]  += ((uint64_t)a[3] * (uint64_t)b[4]) >> 32;  acc[8]  += ((uint64_t)a[3] * (uint64_t)b[4]) & 0xFFFFFFFFULL;
    acc[8]  += ((uint64_t)a[3] * (uint64_t)b[5]) >> 32;  acc[9]  += ((uint64_t)a[3] * (uint64_t)b[5]) & 0xFFFFFFFFULL;
    acc[9]  += ((uint64_t)a[3] * (uint64_t)b[6]) >> 32;  acc[10] += ((uint64_t)a[3] * (uint64_t)b[6]) & 0xFFFFFFFFULL;
    acc[10] += ((uint64_t)a[3] * (uint64_t)b[7]) >> 32;  acc[11] += ((uint64_t)a[3] * (uint64_t)b[7]) & 0xFFFFFFFFULL;
    // i=4
    acc[4]  += ((uint64_t)a[4] * (uint64_t)b[0]) >> 32;  acc[5]  += ((uint64_t)a[4] * (uint64_t)b[0]) & 0xFFFFFFFFULL;
    acc[5]  += ((uint64_t)a[4] * (uint64_t)b[1]) >> 32;  acc[6]  += ((uint64_t)a[4] * (uint64_t)b[1]) & 0xFFFFFFFFULL;
    acc[6]  += ((uint64_t)a[4] * (uint64_t)b[2]) >> 32;  acc[7]  += ((uint64_t)a[4] * (uint64_t)b[2]) & 0xFFFFFFFFULL;
    acc[7]  += ((uint64_t)a[4] * (uint64_t)b[3]) >> 32;  acc[8]  += ((uint64_t)a[4] * (uint64_t)b[3]) & 0xFFFFFFFFULL;
    acc[8]  += ((uint64_t)a[4] * (uint64_t)b[4]) >> 32;  acc[9]  += ((uint64_t)a[4] * (uint64_t)b[4]) & 0xFFFFFFFFULL;
    acc[9]  += ((uint64_t)a[4] * (uint64_t)b[5]) >> 32;  acc[10] += ((uint64_t)a[4] * (uint64_t)b[5]) & 0xFFFFFFFFULL;
    acc[10] += ((uint64_t)a[4] * (uint64_t)b[6]) >> 32;  acc[11] += ((uint64_t)a[4] * (uint64_t)b[6]) & 0xFFFFFFFFULL;
    acc[11] += ((uint64_t)a[4] * (uint64_t)b[7]) >> 32;  acc[12] += ((uint64_t)a[4] * (uint64_t)b[7]) & 0xFFFFFFFFULL;
    // i=5
    acc[5]  += ((uint64_t)a[5] * (uint64_t)b[0]) >> 32;  acc[6]  += ((uint64_t)a[5] * (uint64_t)b[0]) & 0xFFFFFFFFULL;
    acc[6]  += ((uint64_t)a[5] * (uint64_t)b[1]) >> 32;  acc[7]  += ((uint64_t)a[5] * (uint64_t)b[1]) & 0xFFFFFFFFULL;
    acc[7]  += ((uint64_t)a[5] * (uint64_t)b[2]) >> 32;  acc[8]  += ((uint64_t)a[5] * (uint64_t)b[2]) & 0xFFFFFFFFULL;
    acc[8]  += ((uint64_t)a[5] * (uint64_t)b[3]) >> 32;  acc[9]  += ((uint64_t)a[5] * (uint64_t)b[3]) & 0xFFFFFFFFULL;
    acc[9]  += ((uint64_t)a[5] * (uint64_t)b[4]) >> 32;  acc[10] += ((uint64_t)a[5] * (uint64_t)b[4]) & 0xFFFFFFFFULL;
    acc[10] += ((uint64_t)a[5] * (uint64_t)b[5]) >> 32;  acc[11] += ((uint64_t)a[5] * (uint64_t)b[5]) & 0xFFFFFFFFULL;
    acc[11] += ((uint64_t)a[5] * (uint64_t)b[6]) >> 32;  acc[12] += ((uint64_t)a[5] * (uint64_t)b[6]) & 0xFFFFFFFFULL;
    acc[12] += ((uint64_t)a[5] * (uint64_t)b[7]) >> 32;  acc[13] += ((uint64_t)a[5] * (uint64_t)b[7]) & 0xFFFFFFFFULL;
    // i=6
    acc[6]  += ((uint64_t)a[6] * (uint64_t)b[0]) >> 32;  acc[7]  += ((uint64_t)a[6] * (uint64_t)b[0]) & 0xFFFFFFFFULL;
    acc[7]  += ((uint64_t)a[6] * (uint64_t)b[1]) >> 32;  acc[8]  += ((uint64_t)a[6] * (uint64_t)b[1]) & 0xFFFFFFFFULL;
    acc[8]  += ((uint64_t)a[6] * (uint64_t)b[2]) >> 32;  acc[9]  += ((uint64_t)a[6] * (uint64_t)b[2]) & 0xFFFFFFFFULL;
    acc[9]  += ((uint64_t)a[6] * (uint64_t)b[3]) >> 32;  acc[10] += ((uint64_t)a[6] * (uint64_t)b[3]) & 0xFFFFFFFFULL;
    acc[10] += ((uint64_t)a[6] * (uint64_t)b[4]) >> 32;  acc[11] += ((uint64_t)a[6] * (uint64_t)b[4]) & 0xFFFFFFFFULL;
    acc[11] += ((uint64_t)a[6] * (uint64_t)b[5]) >> 32;  acc[12] += ((uint64_t)a[6] * (uint64_t)b[5]) & 0xFFFFFFFFULL;
    acc[12] += ((uint64_t)a[6] * (uint64_t)b[6]) >> 32;  acc[13] += ((uint64_t)a[6] * (uint64_t)b[6]) & 0xFFFFFFFFULL;
    acc[13] += ((uint64_t)a[6] * (uint64_t)b[7]) >> 32;  acc[14] += ((uint64_t)a[6] * (uint64_t)b[7]) & 0xFFFFFFFFULL;
    // i=7
    acc[7]  += ((uint64_t)a[7] * (uint64_t)b[0]) >> 32;  acc[8]  += ((uint64_t)a[7] * (uint64_t)b[0]) & 0xFFFFFFFFULL;
    acc[8]  += ((uint64_t)a[7] * (uint64_t)b[1]) >> 32;  acc[9]  += ((uint64_t)a[7] * (uint64_t)b[1]) & 0xFFFFFFFFULL;
    acc[9]  += ((uint64_t)a[7] * (uint64_t)b[2]) >> 32;  acc[10] += ((uint64_t)a[7] * (uint64_t)b[2]) & 0xFFFFFFFFULL;
    acc[10] += ((uint64_t)a[7] * (uint64_t)b[3]) >> 32;  acc[11] += ((uint64_t)a[7] * (uint64_t)b[3]) & 0xFFFFFFFFULL;
    acc[11] += ((uint64_t)a[7] * (uint64_t)b[4]) >> 32;  acc[12] += ((uint64_t)a[7] * (uint64_t)b[4]) & 0xFFFFFFFFULL;
    acc[12] += ((uint64_t)a[7] * (uint64_t)b[5]) >> 32;  acc[13] += ((uint64_t)a[7] * (uint64_t)b[5]) & 0xFFFFFFFFULL;
    acc[13] += ((uint64_t)a[7] * (uint64_t)b[6]) >> 32;  acc[14] += ((uint64_t)a[7] * (uint64_t)b[6]) & 0xFFFFFFFFULL;
    acc[14] += ((uint64_t)a[7] * (uint64_t)b[7]) >> 32;  acc[15] += ((uint64_t)a[7] * (uint64_t)b[7]) & 0xFFFFFFFFULL;

    fe_reduce_512(r, acc);
}

static void fe_sqr(uint* r, const uint* a) {
    fe_mul(r, a, a);
}

static void fe_pow(uint* r, const uint* a, const uint* e) {
    uint res[8] = {0,0,0,0,0,0,0,1};
    uint base[8];
    for (int i = 0; i < NL; i++) base[i] = a[i];
    for (int i = 0; i < NL; i++) {
        for (int bit = 31; bit >= 0; bit--) {
            uint t[8];
            fe_mul(t, res, res);
            for (int k = 0; k < NL; k++) res[k] = t[k];
            if ((e[i] >> bit) & 1u) {
                uint u[8];
                fe_mul(u, res, base);
                for (int k = 0; k < NL; k++) res[k] = u[k];
            }
        }
    }
    for (int i = 0; i < NL; i++) r[i] = res[i];
}

static void fe_inv(uint* r, const uint* a) {
    uint pm2[8];
    for (int i = 0; i < NL; i++) pm2[i] = PM2[i];
    fe_pow(r, a, pm2);
}

// ---- elliptic curve (Jacobian projective) --------------------------------
//
// Point = (X, Y, Z); affine x = X/Z^2, y = Y/Z^3.
// Infinity = (1, 1, 0).

static inline void jset(uint* X, uint* Y, uint* Z,
                        const uint* Xs, const uint* Ys, const uint* Zs) {
    for (int i = 0; i < NL; i++) {
        X[i] = Xs[i];
        Y[i] = Ys[i];
        Z[i] = Zs[i];
    }
}

static inline void jset_inf(uint* X, uint* Y, uint* Z) {
    uint one[8] = {0,0,0,0,0,0,0,1};
    jset(X, Y, Z, one, one, one);
    Z[7] = 0; // Z = 0 -> infinity, but keep X=Y=1 to avoid div-by-zero
}

// Jacobian point doubling: R = 2*P. Handles infinity.
static void jdouble(uint* RX, uint* RY, uint* RZ,
                    const uint* PX, const uint* PY, const uint* PZ) {
    if (fe_is_zero(PZ)) {
        jset_inf(RX, RY, RZ);
        return;
    }
    if (fe_is_zero(PY)) {
        jset_inf(RX, RY, RZ);
        return;
    }

    uint delta[8];   // Z^2
    uint gamma[8];   // Y^2
    uint beta[8];    // X*gamma
    uint alpha[8];   // 3*X^2  (secp256k1 has a=0)
    uint t1[8], t2[8], t3[8];

    fe_sqr(delta, PZ);
    fe_sqr(gamma, PY);
    fe_mul(beta, PX, gamma);

    // alpha = 3 * X^2
    fe_sqr(t1, PX);
    fe_add(alpha, t1, t1);
    fe_add(alpha, alpha, t1);

    // X3 = alpha^2 - 8*beta  (mod P)
    fe_sqr(t1, alpha);
    fe_add(t2, beta, beta);
    fe_add(t2, t2, t2);
    fe_add(t2, t2, t2); // 8*beta
    fe_sub_mod(RX, t1, t2);

    // Z3 = 2*Y*Z  (mod P)
    fe_mul(t1, PY, PZ);
    fe_add(RZ, t1, t1);

    // Y3 = alpha*(4*beta - X3) - 8*gamma^2  (mod P)
    fe_add(t1, beta, beta);
    fe_add(t1, t1, t1); // 4*beta
    fe_sub_mod(t2, t1, RX);
    fe_mul(t3, alpha, t2);
    fe_sqr(t1, gamma);
    fe_add(t2, t1, t1);
    fe_add(t2, t2, t2);
    fe_add(t2, t2, t2); // 8*gamma^2
    fe_sub_mod(RY, t3, t2);
}

// Mixed Jacobian + affine addition: R = P(Jacobian) + Q(affine, constant mem).
// Handles infinity and P == Q / P == -Q.
static void jadd_mixed(uint* RX, uint* RY, uint* RZ,
                       const uint* PX, const uint* PY, const uint* PZ,
                       constant uint* QX, constant uint* QY) {
    // Copy constant Q into private arrays so all downstream field helpers
    // (which expect private address-space pointers) can be reused unchanged.
    uint qx[8], qy[8];
    for (int i = 0; i < NL; i++) { qx[i] = QX[i]; qy[i] = QY[i]; }

    if (fe_is_zero(PZ)) {
        for (int i = 0; i < NL; i++) {
            RX[i] = qx[i];
            RY[i] = qy[i];
        }
        uint one[8] = {0,0,0,0,0,0,0,1};
        for (int i = 0; i < NL; i++) RZ[i] = one[i];
        return;
    }

    uint z2[8], z3[8];
    fe_sqr(z2, PZ);
    fe_mul(z3, z2, PZ);

    uint u2[8], s2[8];
    fe_mul(u2, qx, z2);
    fe_mul(s2, qy, z3);

    uint h[8], rr[8];
    fe_sub_mod(h, u2, PX);
    fe_sub_mod(rr, s2, PY);

    if (fe_is_zero(h)) {
        if (fe_is_zero(rr)) {
            // P == Q (affine Q == Jacobian P)
            jdouble(RX, RY, RZ, PX, PY, PZ);
        } else {
            // P == -Q
            jset_inf(RX, RY, RZ);
        }
        return;
    }

    uint h2[8], h3[8], u1h2[8], t1[8], t2[8];
    fe_sqr(h2, h);
    fe_mul(h3, h2, h);
    fe_mul(u1h2, PX, h2);

    // X3 = r^2 - h^3 - 2*u1h2  (mod P)
    fe_sqr(t1, rr);
    fe_sub_mod(t2, t1, h3);
    fe_add(t1, u1h2, u1h2); // 2*u1h2, reduced
    fe_sub_mod(RX, t2, t1);

    // Y3 = r*(u1h2 - X3) - s1*h^3  (mod P)
    fe_sub_mod(t1, u1h2, RX);
    fe_mul(t2, rr, t1);
    fe_mul(t1, PY, h3);
    fe_sub_mod(RY, t2, t1);

    // Z3 = Z1 * h
    fe_mul(RZ, PZ, h);
}

// Convert Jacobian point to affine. If point is infinity, sets Qx=0, Qy=0.
static void jto_affine(uint* Qx, uint* Qy,
                       const uint* X, const uint* Y, const uint* Z) {
    if (fe_is_zero(Z)) {
        for (int i = 0; i < NL; i++) { Qx[i] = 0; Qy[i] = 0; }
        return;
    }
    uint z2[8], z3[8], invz2[8], invz3[8];
    fe_sqr(z2, Z);
    fe_mul(z3, z2, Z);
    fe_inv(invz2, z2);
    fe_inv(invz3, z3);
    fe_mul(Qx, X, invz2);
    fe_mul(Qy, Y, invz3);
}

static void scalar_mul(uint* Qx, uint* Qy, const uint* k) {
    uint RX[8], RY[8], RZ[8];
    jset_inf(RX, RY, RZ);
    int Rinf = 1;

    for (int i = 0; i < NL; i++) {
        for (int bit = 31; bit >= 0; bit--) {
            uint kb = (k[i] >> bit) & 1u;

            if (!Rinf) {
                uint TX[8], TY[8], TZ[8];
                jdouble(TX, TY, TZ, RX, RY, RZ);
                jset(RX, RY, RZ, TX, TY, TZ);
            }

            if (kb) {
                if (Rinf) {
                    for (int t = 0; t < NL; t++) {
                        RX[t] = GX[t];
                        RY[t] = GY[t];
                    }
                    uint one[8] = {0,0,0,0,0,0,0,1};
                    for (int t = 0; t < NL; t++) RZ[t] = one[t];
                    Rinf = 0;
                } else {
                    uint TX[8], TY[8], TZ[8];
                    jadd_mixed(TX, TY, TZ, RX, RY, RZ, GX, GY);
                    jset(RX, RY, RZ, TX, TY, TZ);
                }
            }
        }
    }

    jto_affine(Qx, Qy, RX, RY, RZ);
}

// ---- keccak-256 ----------------------------------------------------------

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
    volatile int rounds = 24;
    for (int round = 0; round < rounds; round++) {
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
        for (int j = 0; j < 5; j++) {
            uint64_t tc[5];
            for (int i = 0; i < 5; i++) tc[i] = st[j*5 + i];
            for (int i = 0; i < 5; i++)
                st[j*5 + i] = tc[i] ^ ((~tc[(i+1)%5]) & tc[(i+2)%5]);
        }
        st[0] ^= RC[round];
    }
}

static void keccak256_addr(const uint* x, const uint* y, __global uchar* out_addr) {
    uint64_t st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;
    for (int lane = 0; lane < 8; lane++) {
        uint a = (lane < 4) ? x[2 * lane]     : y[2 * (lane - 4)];
        uint b = (lane < 4) ? x[2 * lane + 1] : y[2 * (lane - 4) + 1];
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

// ---- matching ------------------------------------------------------------

static void key_add_gid(uint* key, __global uint* base, uint64_t gid) {
    for (int i = 0; i < 8; i++) key[i] = base[i];
    uint64_t carry = gid;
    for (int i = 7; i >= 0 && carry; i--) {
        uint64_t s = (uint64_t)key[i] + carry;
        key[i] = (uint)s;
        carry = s >> 32;
    }
}

__kernel void derive_pubkeys(__global uint* base, __global uint* pubs) {
    size_t gid = get_global_id(0);
    size_t off = gid * 16;

    uint key[8];
    key_add_gid(key, base, (uint64_t)gid);

    uint Qx[8], Qy[8];
    scalar_mul(Qx, Qy, key);

    for (int i = 0; i < 8; i++) {
        pubs[off + i]     = Qx[i];
        pubs[off + 8 + i] = Qy[i];
    }
}

__kernel void hash_addrs(__global uint* pubs, __global uchar* addrs) {
    size_t gid = get_global_id(0);
    size_t off = gid * 16;
    size_t addr_off = gid * 20;

    uint Qx[8], Qy[8];
    for (int i = 0; i < 8; i++) {
        Qx[i] = pubs[off + i];
        Qy[i] = pubs[off + 8 + i];
    }
    keccak256_addr(Qx, Qy, &addrs[addr_off]);
}

// params layout: [0]=prefix_len, [1]=suffix_len, [2..9]=base[8],
//                [10..49]=prefix nibbles(40), [50..89]=suffix nibbles(40)
__kernel void match_addrs(__global uint* base,
                          __global uchar* addrs,
                          __global int*  out_found,
                          __global uint* out_priv,
                          __global uchar* out_addr,
                          __global uint* params) {
    size_t gid = get_global_id(0);
    size_t addr_off = gid * 20;

    uint prefix_len = params[0];
    uint suffix_len = params[1];

    uint key[8];
    key_add_gid(key, base, (uint64_t)gid);

    uchar addr[20];
    for (int i = 0; i < 20; i++) addr[i] = addrs[addr_off + i];

    int match = 1;
    for (uint i = 0; i < prefix_len; i++) {
        uchar n = (i & 1u) ? (addr[i / 2] & 0xF) : ((addr[i / 2] >> 4) & 0xF);
        if (n != (uchar)(params[10 + i])) { match = 0; break; }
    }
    if (match) {
        for (uint i = 0; i < suffix_len; i++) {
            uint idx = 40u - suffix_len + i;
            uchar n = (idx & 1u) ? (addr[idx / 2] & 0xF) : ((addr[idx / 2] >> 4) & 0xF);
            if (n != (uchar)(params[50 + i])) { match = 0; break; }
        }
    }
    if (match) {
        int idx = atomic_inc(out_found);
        if (idx == 0) {
            for (int i = 0; i < 8; i++) out_priv[i] = key[i];
            for (int i = 0; i < 20; i++) out_addr[i] = addr[i];
        }
    }
}
