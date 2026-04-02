#include "4_2mulifma.h"
#include "4_2bench.h"

#include <gmp.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

namespace {

struct Mpz {
    mpz_t v;
    Mpz() { mpz_init(v); }
    ~Mpz() { mpz_clear(v); }
};

static void build_modulus(mpz_t p) {
    mpz_set_ui(p, 1);
    mpz_mul_2exp(p, p, 505);
    mpz_sub_ui(p, p, 1);
}

static void limbs10_to_mpz_base52_top37(mpz_t rop, const uint64_t limbs[10]) {
    mpz_set_ui(rop, limbs[9] & ((1ULL << 37) - 1));
    for (int i = 8; i >= 0; --i) {
        mpz_mul_2exp(rop, rop, 52);
        mpz_add_ui(rop, rop, static_cast<unsigned long>(limbs[i] & ((1ULL << 52) - 1)));
    }
}

static void pack_pair0(const uint64_t x[10], __m512i out[5]) {
    alignas(64) uint64_t lanes[8];
    for (int i = 0; i < 5; ++i) {
        lanes[0] = x[i];
        lanes[1] = x[i + 5];
        for (int lane = 2; lane < 8; ++lane) lanes[lane] = lanes[lane & 1];
        out[i] = _mm512_load_si512((const void*)lanes);
    }
}

static void unpack_pair0(const __m512i in[5], uint64_t x[10]) {
    alignas(64) uint64_t lanes[8];
    for (int i = 0; i < 5; ++i) {
        _mm512_store_si512((void*)lanes, in[i]);
        x[i] = lanes[0] & ((1ULL << 52) - 1);
        x[i + 5] = lanes[1] & ((1ULL << 52) - 1);
    }
    x[9] &= ((1ULL << 37) - 1);
}

static bool run_random(std::size_t iterations, uint64_t seed) {
    std::mt19937_64 rng(seed);
    Mpz p, R, A, B, Ref, Out, OutMod;
    build_modulus(p.v);
    mpz_set_ui(R.v, 1);
    mpz_mul_2exp(R.v, R.v, 520);
    mpz_mod(R.v, R.v, p.v);

    for (std::size_t it = 0; it < iterations; ++it) {
        uint64_t a[10], b[10];
        for (int i = 0; i < 9; ++i) {
            a[i] = rng() & ((1ULL << 52) - 1);
            b[i] = rng() & ((1ULL << 52) - 1);
        }
        a[9] = rng() & ((1ULL << 37) - 1);
        b[9] = rng() & ((1ULL << 37) - 1);

        alignas(64) __m512i va[5], vb[5], vo[5];
        pack_pair0(a, va);
        pack_pair0(b, vb);
        mulmodifma4_2(va, vb, vo, P);

        uint64_t out[10];
        unpack_pair0(vo, out);

        limbs10_to_mpz_base52_top37(A.v, a);
        limbs10_to_mpz_base52_top37(B.v, b);
        mpz_mul(Ref.v, A.v, B.v);
        mpz_mod(Ref.v, Ref.v, p.v);

        limbs10_to_mpz_base52_top37(Out.v, out);
        mpz_mod(OutMod.v, Out.v, p.v);
        mpz_mul(OutMod.v, OutMod.v, R.v);
        mpz_mod(OutMod.v, OutMod.v, p.v);

        if (mpz_cmp(Ref.v, OutMod.v) != 0) {
            std::cerr << "[FAIL][4_2 avx512ifma] mismatch at it=" << it << "\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::size_t iters = 200;
    uint64_t seed = 12345;
    if (argc >= 2) iters = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    if (argc >= 3) seed = static_cast<uint64_t>(std::strtoull(argv[2], nullptr, 10));

    if (!run_random(iters, seed)) return 1;
    std::cout << "[OK] 4_2/avx512ifma reduction vs GMP passed: iterations=" << iters << "\n";
    return 0;
}
