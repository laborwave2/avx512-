#include "8_1mul.h"

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
    Mpz(const Mpz&) = delete;
    Mpz& operator=(const Mpz&) = delete;
};

static void build_modulus(mpz_t p) {
    // p = 2^505 - 1
    mpz_set_ui(p, 1);
    mpz_mul_2exp(p, p, 505);
    mpz_sub_ui(p, p, 1);
}

static void limbs18_to_mpz_base29_top12_u64(mpz_t rop, const uint64_t limbs[18]) {
    mpz_set_ui(rop, static_cast<unsigned long>(limbs[17] & 0xFFFULL));
    for (int i = 16; i >= 0; --i) {
        mpz_mul_2exp(rop, rop, 29);
        mpz_add_ui(rop, rop, static_cast<unsigned long>(limbs[i] & 0x1FFFFFFFULL));
    }
}

static bool check_ranges_limbs18_base29_top12_u64(const uint64_t limbs[18]) {
    for (int i = 0; i < 17; ++i) {
        if ((limbs[i] >> 29) != 0) return false;
    }
    if ((limbs[17] >> 12) != 0) return false;
    return true;
}

static void make_random_elem(__m512i a[18], std::mt19937_64& rng) {
    alignas(64) uint64_t lanes[8];
    for (int i = 0; i < 17; ++i) {
        for (int lane = 0; lane < 8; ++lane) lanes[lane] = rng() & 0x1FFFFFFFULL;
        a[i] = _mm512_load_si512((const void*)lanes);
    }
    for (int lane = 0; lane < 8; ++lane) lanes[lane] = rng() & 0xFFFULL;
    a[17] = _mm512_load_si512((const void*)lanes);
}

static void extract_lane_limbs18(const __m512i vec[18], int lane, uint64_t limbs[18]) {
    alignas(64) uint64_t tmp[8];
    for (int i = 0; i < 18; ++i) {
        _mm512_store_si512((void*)tmp, vec[i]);
        limbs[i] = tmp[lane];
    }
}

static bool run_known_cases() {
    Mpz p, R, A, B, Ref, Out, OutMod;
    build_modulus(p.v);
    mpz_set_ui(R.v, 1);
    mpz_mul_2exp(R.v, R.v, 522);
    mpz_mod(R.v, R.v, p.v);

    auto pack_const = [](__m512i a[18], const uint64_t limbs[18]) {
        alignas(64) uint64_t lanes[8];
        for (int i = 0; i < 18; ++i) {
            for (int lane = 0; lane < 8; ++lane) lanes[lane] = limbs[i];
            a[i] = _mm512_load_si512((const void*)lanes);
        }
    };

    uint64_t zero[18] = {};
    uint64_t one[18] = {};
    one[0] = 1;
    uint64_t pm1[18];
    for (int i = 0; i < 17; ++i) pm1[i] = 0x1FFFFFFFULL;
    pm1[17] = 0xFFFULL;

    const struct {
        const char* name;
        const uint64_t* a;
        const uint64_t* b;
    } cases[] = {
        {"0*0", zero, zero},
        {"0*1", zero, one},
        {"1*1", one, one},
        {"(p-1)*(p-1)", pm1, pm1},
    };

    for (const auto& tc : cases) {
        alignas(64) __m512i a[18], b[18], out[18];
        pack_const(a, tc.a);
        pack_const(b, tc.b);

        mulmod8_1(a, b, out);

        for (int lane = 0; lane < 8; ++lane) {
            uint64_t out_limbs[18];
            extract_lane_limbs18(out, lane, out_limbs);
            if (!check_ranges_limbs18_base29_top12_u64(out_limbs)) {
                std::cerr << "[FAIL][8_1 avx512F] " << tc.name << " lane=" << lane
                          << " output limbs out of range\n";
                return false;
            }

            limbs18_to_mpz_base29_top12_u64(A.v, tc.a);
            limbs18_to_mpz_base29_top12_u64(B.v, tc.b);
            mpz_mul(Ref.v, A.v, B.v);
            mpz_mod(Ref.v, Ref.v, p.v);

            limbs18_to_mpz_base29_top12_u64(Out.v, out_limbs);
            mpz_mod(OutMod.v, Out.v, p.v);
            mpz_mul(OutMod.v, OutMod.v, R.v);
            mpz_mod(OutMod.v, OutMod.v, p.v);

            if (mpz_cmp(OutMod.v, Ref.v) != 0) {
                std::cerr << "[FAIL][8_1 avx512F] " << tc.name << " lane=" << lane << " mismatch\n";
                return false;
            }
        }
    }
    return true;
}

static bool run_random(std::size_t iterations, uint64_t seed) {
    std::mt19937_64 rng(seed);

    Mpz p, R, A, B, Ref, Out, OutMod;
    build_modulus(p.v);
    mpz_set_ui(R.v, 1);
    mpz_mul_2exp(R.v, R.v, 522);
    mpz_mod(R.v, R.v, p.v);

    for (std::size_t it = 0; it < iterations; ++it) {
        alignas(64) __m512i a[18], b[18], out[18];
        make_random_elem(a, rng);
        make_random_elem(b, rng);

        mulmod8_1(a, b, out);

        for (int lane = 0; lane < 8; ++lane) {
            uint64_t a_limbs[18], b_limbs[18], out_limbs[18];
            extract_lane_limbs18(a, lane, a_limbs);
            extract_lane_limbs18(b, lane, b_limbs);
            extract_lane_limbs18(out, lane, out_limbs);

            if (!check_ranges_limbs18_base29_top12_u64(out_limbs)) {
                std::cerr << "[FAIL][8_1 avx512F] it=" << it << " lane=" << lane
                          << " output limbs out of range\n";
                return false;
            }

            // For modulus p=2^505-1, using Montgomery REDC with R=2^505 gives R mod p = 1.
            // So the expected value matches (a*b) mod p.
            limbs18_to_mpz_base29_top12_u64(A.v, a_limbs);
            limbs18_to_mpz_base29_top12_u64(B.v, b_limbs);
            mpz_mul(Ref.v, A.v, B.v);
            mpz_mod(Ref.v, Ref.v, p.v);

            limbs18_to_mpz_base29_top12_u64(Out.v, out_limbs);
            mpz_mod(OutMod.v, Out.v, p.v);
            mpz_mul(OutMod.v, OutMod.v, R.v);
            mpz_mod(OutMod.v, OutMod.v, p.v);

            if (mpz_cmp(OutMod.v, Ref.v) != 0) {
                std::cerr << "[FAIL][8_1 avx512F] it=" << it << " lane=" << lane << " mismatch\n";
                return false;
            }
        }
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t iters = 200;
    uint64_t seed = 12345;
    if (argc >= 2) iters = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    if (argc >= 3) seed = static_cast<uint64_t>(std::strtoull(argv[2], nullptr, 10));

    if (!run_known_cases()) return 1;
    if (!run_random(iters, seed)) return 1;

    std::cout << "[OK] 8_1/avx512F mulmod8_1 passed: iterations=" << iters << " lanes/iter=8\n";
    return 0;
}
