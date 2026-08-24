// Standalone test: simulate what the OpenCL kernel `derive_points` produces
// for a single key, on the CPU using the same Rust `mont`/`ec` primitives
// that the kernel was ported from. Print both the GPU-equivalent pubkey and
// the address, for keys 1, 2, 3, and a few random ones. Run with:
//
//   cargo run --release --bin sim_kernel
//
// This lets us compare against the GPU output without an AMD GPU on hand.

use vanity_evm_gpu::ec::{generate_precomp, point_mul};
use vanity_evm_gpu::mont::Fe;

fn fe_to_hex(f: &Fe) -> String {
    let mut s = String::new();
    for i in (0..8).rev() {
        s.push_str(&format!("{:08x}", f[i]));
    }
    s
}

fn fe_bytes_be(f: &Fe) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..8 {
        let limb = f[7 - i].to_be_bytes();
        out[4 * i..4 * i + 4].copy_from_slice(&limb);
    }
    out
}

fn print_key(label: &str, k: &[u8; 32]) {
    let precomp = generate_precomp();
    let p = point_mul(k, &precomp);
    let mut full = [0u8; 65];
    full[0] = 0x04;
    full[1..33].copy_from_slice(&fe_bytes_be(&p.x));
    full[33..65].copy_from_slice(&fe_bytes_be(&p.y));
    let addr = vanity_evm_gpu::crypto::pubkey_to_address(&full);
    eprintln!(
        "{}: privkey={}",
        label,
        k.iter()
            .rev()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    );
    eprintln!("  X = 0x{}", fe_to_hex(&p.x));
    eprintln!("  Y = 0x{}", fe_to_hex(&p.y));
    eprintln!(
        "  addr = 0x{}",
        addr.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    );
}

fn main() {
    // Edge case: host's `bytes_to_u32x8_le([0,..,0,2])` returns limbs
    // [0,..,0, 0x02000000] (limb 7 high byte = 2). The kernel's
    // `key_byte(key, i)` for i in 0..3 reads key[7]'s high bytes — so the
    // GPU ends up addressing column 0 entry 1 = (1 * 256^31) * G, NOT 1G.
    // Print both interpretations so we can see which one matches GPU output.
    let mut k_msb = [0u8; 32];
    k_msb[0] = 2;
    print_key("scalar = 2 * 256^31 (k_msb[0]=2)", &k_msb);

    let mut k1 = [0u8; 32];
    k1[31] = 1;
    print_key("key=1", &k1);
    let mut k2 = [0u8; 32];
    k2[31] = 2;
    print_key("key=2", &k2);
    let mut k3 = [0u8; 32];
    k3[31] = 3;
    print_key("key=3", &k3);

    use rand::RngCore;
    let mut rng = rand::rngs::OsRng;
    for v in 0..3 {
        let mut k = [0u8; 32];
        rng.fill_bytes(&mut k);
        // Reduce mod n
        if let Some(addr) = vanity_evm_gpu::crypto::privkey_to_address(&k) {
            eprintln!("random vector {} (privkey reduced mod n):", v);
            eprintln!(
                "  reduced key = {}",
                k.iter().map(|b| format!("{:02x}", b)).collect::<String>()
            );
            eprintln!(
                "  addr        = 0x{}",
                addr.iter()
                    .map(|b| format!("{:02x}", b))
                    .collect::<String>()
            );
            let precomp = generate_precomp();
            let p = point_mul(&k, &precomp);
            let mut full = [0u8; 65];
            full[0] = 0x04;
            full[1..33].copy_from_slice(&fe_bytes_be(&p.x));
            full[33..65].copy_from_slice(&fe_bytes_be(&p.y));
            let got_addr = vanity_evm_gpu::crypto::pubkey_to_address(&full);
            eprintln!("  point_mul x = 0x{}", fe_to_hex(&p.x));
            eprintln!("  point_mul y = 0x{}", fe_to_hex(&p.y));
            eprintln!(
                "  point_mul addr = 0x{}",
                got_addr
                    .iter()
                    .map(|b| format!("{:02x}", b))
                    .collect::<String>()
            );
        }
    }
}
