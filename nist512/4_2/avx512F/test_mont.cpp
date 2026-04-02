#include "4_2mul.h"

#include <gmp.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>

namespace {

// Modulus m = 2^505 - 1 in base 2^29 with a 12-bit top limb, packed as:
// vec[i] lane = limbs[i] (low32) | limbs[i+9] (high32)
static alignas(64) const __m512i MOD_P[9] = {
    _mm512_setr_epi64(0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL,
                      0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL),
    _mm512_setr_epi64(0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL,
                      0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL),
    _mm512_setr_epi64(0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL,
                      0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL),
    _mm512_setr_epi64(0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL,
                      0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL),
    _mm512_setr_epi64(0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL,
                      0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL),
    _mm512_setr_epi64(0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL,
                      0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL),
    _mm512_setr_epi64(0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL,
                      0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL),
    _mm512_setr_epi64(0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL,
                      0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL, 0x1FFFFFFFULL),
    _mm512_setr_epi64(0x1FFFFFFFULL, 0xFFFULL, 0x1FFFFFFFULL, 0xFFFULL,
                      0x1FFFFFFFULL, 0xFFFULL, 0x1FFFFFFFULL, 0xFFFULL),
};

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

static void limbs18_to_mpz_base29_top12(mpz_t rop, const uint32_t limbs[18]) {
    mpz_set_ui(rop, limbs[17] & 0xFFFu);
    for (int i = 16; i >= 0; --i) {
        mpz_mul_2exp(rop, rop, 29);
        mpz_add_ui(rop, rop, static_cast<unsigned long>(limbs[i] & 0x1FFFFFFFu));
    }
}

static bool check_ranges_limbs18_base29_top12(const uint32_t limbs[18]) {
    for (int i = 0; i < 17; ++i) {
        if ((limbs[i] >> 29) != 0) return false;
    }
    if ((limbs[17] >> 12) != 0) return false;
    return true;
}

static void pack_limbs18_to_vec9(const uint32_t limbs_lane[18][8], __m512i out[9]) {
    alignas(64) uint64_t lanes[8];
    for (int i = 0; i < 9; ++i) {
        for (int lane = 0; lane < 8; ++lane) {
            uint64_t lo = static_cast<uint64_t>(limbs_lane[i][lane]);
            uint64_t hi = static_cast<uint64_t>(limbs_lane[i + 9][lane]);
            lanes[lane] = lo | (hi << 32);
        }
        out[i] = _mm512_load_si512((const void*)lanes);
    }
}

static void unpack_vec9_lane_to_limbs18(const __m512i vec[9], int lane, uint32_t limbs[18]) {
    alignas(64) uint64_t tmp[8];
    for (int i = 0; i < 9; ++i) {
        _mm512_store_si512((void*)tmp, vec[i]);
        uint64_t v = tmp[lane];
        limbs[i] = static_cast<uint32_t>(v & 0xFFFFFFFFu);
        limbs[i + 9] = static_cast<uint32_t>((v >> 32) & 0xFFFFFFFFu);
    }
}

static void make_random_limbs18(uint32_t limbs_lane[18][8], std::mt19937_64& rng) {
    for (int i = 0; i < 17; ++i) {
        for (int lane = 0; lane < 8; ++lane) {
            limbs_lane[i][lane] = static_cast<uint32_t>(rng() & 0x1FFFFFFFULL);
        }
    }
    for (int lane = 0; lane < 8; ++lane) {
        limbs_lane[17][lane] = static_cast<uint32_t>(rng() & 0xFFFULL);
    }
}

static void make_const_limbs18(uint32_t limbs_lane[18][8], const uint32_t limbs[18]) {
    for (int i = 0; i < 18; ++i) {
        for (int lane = 0; lane < 8; ++lane) {
            limbs_lane[i][lane] = limbs[i];
        }
    }
}

static bool run_known_cases() {
    Mpz p, A, B, Ref, Out, OutMod;
    build_modulus(p.v);

    const uint32_t zero[18] = {};
    uint32_t one[18] = {};
    one[0] = 1;
    uint32_t pm1[18];
    for (int i = 0; i < 17; ++i) pm1[i] = 0x1FFFFFFF;
    pm1[17] = 0xFFF;

    const struct {
        const char* name;
        const uint32_t* a;
        const uint32_t* b;
    } cases[] = {
        {"0*0", zero, zero},
        {"0*1", zero, one},
        {"1*1", one, one},
        {"(p-1)*(p-1)", pm1, pm1},
    };

    for (const auto& tc : cases) {
        uint32_t a_lane[18][8], b_lane[18][8];
        make_const_limbs18(a_lane, tc.a);
        make_const_limbs18(b_lane, tc.b);

        alignas(64) __m512i a[9], b[9], out[9];
        pack_limbs18_to_vec9(a_lane, a);
        pack_limbs18_to_vec9(b_lane, b);

        mulmod4_2(a, b, out, MOD_P);

        for (int lane = 0; lane < 8; ++lane) {
            uint32_t out_limbs[18];
            unpack_vec9_lane_to_limbs18(out, lane, out_limbs);

            if (!check_ranges_limbs18_base29_top12(out_limbs)) {
                std::cerr << "[FAIL][4_2 avx512F] " << tc.name << " lane=" << lane
                          << " output limbs out of range\n";
                return false;
            }

            limbs18_to_mpz_base29_top12(A.v, tc.a);
            limbs18_to_mpz_base29_top12(B.v, tc.b);
            mpz_mul(Ref.v, A.v, B.v);
            mpz_mod(Ref.v, Ref.v, p.v);

            limbs18_to_mpz_base29_top12(Out.v, out_limbs);
            mpz_mod(OutMod.v, Out.v, p.v);

            if (mpz_cmp(OutMod.v, Ref.v) != 0) {
                char* ref_str = mpz_get_str(nullptr, 16, Ref.v);
                char* out_str = mpz_get_str(nullptr, 16, OutMod.v);
                std::cerr << "[FAIL][4_2 avx512F] " << tc.name << " lane=" << lane << " mismatch\n";
                std::cerr << "  ref(mod p)=0x" << ref_str << "\n";
                std::cerr << "  out(mod p)=0x" << out_str << "\n";
                void (*gmp_free_func)(void*, size_t);
                mp_get_memory_functions(nullptr, nullptr, &gmp_free_func);
                gmp_free_func(ref_str, 0);
                gmp_free_func(out_str, 0);
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
        uint32_t a_lane[18][8], b_lane[18][8];
        make_random_limbs18(a_lane, rng);
        make_random_limbs18(b_lane, rng);

        alignas(64) __m512i a[9], b[9], out[9];
        pack_limbs18_to_vec9(a_lane, a);
        pack_limbs18_to_vec9(b_lane, b);

        mulmod4_2(a, b, out, MOD_P);

        for (int lane = 0; lane < 8; ++lane) {
            uint32_t a_limbs[18], b_limbs[18], out_limbs[18];
            for (int i = 0; i < 18; ++i) {
                a_limbs[i] = a_lane[i][lane];
                b_limbs[i] = b_lane[i][lane];
            }
            unpack_vec9_lane_to_limbs18(out, lane, out_limbs);

            if (!check_ranges_limbs18_base29_top12(out_limbs)) {
                std::cerr << "[FAIL][4_2 avx512F] it=" << it << " lane=" << lane
                          << " output limbs out of range\n";
                return false;
            }

            // Both implementations in this repo target modulus p = 2^505 - 1.
            // If you view mulmod4_2 as Montgomery REDC with R = 2^505, then R mod p = 1
            // and the expected value is still (a*b) mod p.
            limbs18_to_mpz_base29_top12(A.v, a_limbs);
            limbs18_to_mpz_base29_top12(B.v, b_limbs);
            mpz_mul(Ref.v, A.v, B.v);
            mpz_mod(Ref.v, Ref.v, p.v);

            limbs18_to_mpz_base29_top12(Out.v, out_limbs);
            mpz_mod(OutMod.v, Out.v, p.v);

            if (mpz_cmp(OutMod.v, Ref.v) != 0) {
                std::cerr << "[FAIL][4_2 avx512F] it=" << it << " lane=" << lane << " mismatch\n";
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

    std::cout << "[OK] 4_2/avx512F mulmod4_2 passed: iterations=" << iters << " lanes/iter=8\n";
    return 0;
}
