#include "8_1mulifma.h"

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

static void limbs10_to_mpz_base52_top37(mpz_t rop, const uint64_t limbs[10]) {
    mpz_set_ui(rop, limbs[9] & ((1ULL << 37) - 1));
    for (int i = 8; i >= 0; --i) {
        mpz_mul_2exp(rop, rop, 52);
        mpz_add_ui(rop, rop, static_cast<unsigned long>(limbs[i] & ((1ULL << 52) - 1)));
    }
}

static bool check_ranges_limbs10_base52_top37(const uint64_t limbs[10]) {
    for (int i = 0; i < 9; ++i) {
        if ((limbs[i] >> 52) != 0) return false;
    }
    if ((limbs[9] >> 37) != 0) return false;
    return true;
}

static void make_random_elem(__m512i a[10], std::mt19937_64& rng) {
    const uint64_t mask52 = (1ULL << 52) - 1;
    alignas(64) uint64_t lanes[8];
    for (int i = 0; i < 9; ++i) {
        for (int lane = 0; lane < 8; ++lane) lanes[lane] = rng() & mask52;
        a[i] = _mm512_load_si512((const void*)lanes);
    }
    for (int lane = 0; lane < 8; ++lane) lanes[lane] = rng() & ((1ULL << 37) - 1);
    a[9] = _mm512_load_si512((const void*)lanes);
}

static void extract_lane_limbs10(const __m512i vec[10], int lane, uint64_t limbs[10]) {
    alignas(64) uint64_t tmp[8];
    for (int i = 0; i < 10; ++i) {
        _mm512_store_si512((void*)tmp, vec[i]);
        limbs[i] = tmp[lane];
    }
}

static bool run_known_cases() {
    Mpz p, A, B, Ref, Out, OutMod;
    build_modulus(p.v);

    auto pack_const = [](__m512i a[10], const uint64_t limbs[10]) {
        alignas(64) uint64_t lanes[8];
        for (int i = 0; i < 10; ++i) {
            for (int lane = 0; lane < 8; ++lane) lanes[lane] = limbs[i];
            a[i] = _mm512_load_si512((const void*)lanes);
        }
    };

    uint64_t zero[10] = {};
    uint64_t one[10] = {};
    one[0] = 1;
    uint64_t pm1[10];
    pm1[0] = ((1ULL << 52) - 2);
    for (int i = 1; i < 9; ++i) pm1[i] = (1ULL << 52) - 1;
    pm1[9] = (1ULL << 37) - 1;

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
        alignas(64) __m512i a[10], b[10], out[10];
        pack_const(a, tc.a);
        pack_const(b, tc.b);

        mulmod8_1(a, b, out);

        for (int lane = 0; lane < 8; ++lane) {
            uint64_t out_limbs[10];
            extract_lane_limbs10(out, lane, out_limbs);

            if (!check_ranges_limbs10_base52_top37(out_limbs)) {
                std::cerr << "[FAIL][8_1 avx512ifma] " << tc.name << " lane=" << lane
                          << " output limbs out of range\n";
                return false;
            }

            limbs10_to_mpz_base52_top37(A.v, tc.a);
            limbs10_to_mpz_base52_top37(B.v, tc.b);
            mpz_mul(Ref.v, A.v, B.v);
            mpz_mod(Ref.v, Ref.v, p.v);

            limbs10_to_mpz_base52_top37(Out.v, out_limbs);
            mpz_mod(OutMod.v, Out.v, p.v);

            if (mpz_cmp(OutMod.v, Ref.v) != 0) {
                std::cerr << "[FAIL][8_1 avx512ifma] " << tc.name << " lane=" << lane
                          << " mismatch\n";
                return false;
            }
        }
    }

    return true;
}

static bool run_random(std::size_t iterations, uint64_t seed) {
    std::mt19937_64 rng(seed);

    Mpz p, A, B, Ref, Out, OutMod;
    build_modulus(p.v);

    for (std::size_t it = 0; it < iterations; ++it) {
        alignas(64) __m512i a[10], b[10], out[10];
        make_random_elem(a, rng);
        make_random_elem(b, rng);

        mulmod8_1(a, b, out);

        for (int lane = 0; lane < 8; ++lane) {
            uint64_t a_limbs[10], b_limbs[10], out_limbs[10];
            extract_lane_limbs10(a, lane, a_limbs);
            extract_lane_limbs10(b, lane, b_limbs);
            extract_lane_limbs10(out, lane, out_limbs);

            if (!check_ranges_limbs10_base52_top37(out_limbs)) {
                std::cerr << "[FAIL][8_1 avx512ifma] it=" << it << " lane=" << lane
                          << " output limbs out of range\n";
                return false;
            }

            limbs10_to_mpz_base52_top37(A.v, a_limbs);
            limbs10_to_mpz_base52_top37(B.v, b_limbs);
            mpz_mul(Ref.v, A.v, B.v);
            mpz_mod(Ref.v, Ref.v, p.v);

            limbs10_to_mpz_base52_top37(Out.v, out_limbs);
            mpz_mod(OutMod.v, Out.v, p.v);

            if (mpz_cmp(OutMod.v, Ref.v) != 0) {
                std::cerr << "[FAIL][8_1 avx512ifma] it=" << it << " lane=" << lane
                          << " mismatch\n";
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

    std::cout << "[OK] 8_1/avx512ifma mulmod8_1 passed: iterations=" << iters
              << " lanes/iter=8\n";
    return 0;
}
