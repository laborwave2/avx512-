#include "test.h"
#include "4_2mul.h"
#include "4_2bench.h"

#include <gmp.h>

#include <immintrin.h>
#include <cstdint>
#include <iostream>

static void build_p(mpz_t p) {
    // p = 2^505 - 1
    mpz_set_ui(p, 1);
    mpz_mul_2exp(p, p, 505);
    mpz_sub_ui(p, p, 1);
}

static void limbs18_to_mpz_base29_top12(mpz_t rop, const uint32_t limbs[18]) {
    mpz_set_ui(rop, limbs[17] & ((1u << 12) - 1));
    for (int i = 16; i >= 0; --i) {
        mpz_mul_2exp(rop, rop, 29);
        mpz_add_ui(rop, rop, static_cast<unsigned long>(limbs[i] & 0x1FFFFFFFu));
    }
}

static void pack_limbs18_all_lanes(const uint32_t limbs[18], __m512i out[9]) {
    for (int i = 0; i < 9; ++i) {
        uint64_t lane_val = static_cast<uint64_t>(limbs[i]) |
                            (static_cast<uint64_t>(limbs[i + 9]<< 32) );
        alignas(64) uint64_t lanes[8];
        for (int lane = 0; lane < 8; ++lane) lanes[lane] = lane_val;
        out[i] = _mm512_load_si512((const void*)lanes);
    }
}

static void unpack_lane_limbs18(const __m512i vec[9], int lane, uint32_t limbs[18]) {
    alignas(64) uint64_t tmp[8];
    for (int i = 0; i < 9; ++i) {
        _mm512_store_si512((void*)tmp, vec[i]);
        uint64_t v = tmp[lane];
        limbs[i] = static_cast<uint32_t>(v & 0xFFFFFFFFu);
        limbs[i + 9] = static_cast<uint32_t>((v >> 32) & 0xFFFFFFFFu);
    }
    // 规范化顶 limb
    limbs[17] &= (1u << 12) - 1;
}

bool run_test_4_2_avx512F() {
    // 构造 a=b=p=2^505-1
    uint32_t limbs[18];
    for (int i = 0; i < 17; ++i) limbs[i] = 0x1FFFFFFF;
    limbs[17] = 0xFFF;

    alignas(64) __m512i a[9];
    alignas(64) __m512i b[9];
    pack_limbs18_all_lanes(limbs, a);
    pack_limbs18_all_lanes(limbs, b);

    alignas(64) __m512i out[9];
    mulmod4_2(a, b, out, P);

    // GMP 参考
    mpz_t p, A, B, ref, impl;
    mpz_init(p);
    mpz_init(A);
    mpz_init(B);
    mpz_init(ref);
    mpz_init(impl);

    build_p(p);
    limbs18_to_mpz_base29_top12(A, limbs);
    limbs18_to_mpz_base29_top12(B, limbs);
    mpz_mul(ref, A, B);
    mpz_mod(ref, ref, p);

    uint32_t out_limbs[18];
    unpack_lane_limbs18(out, 0, out_limbs);
    limbs18_to_mpz_base29_top12(impl, out_limbs);

    bool ok = (mpz_cmp(impl, ref) == 0);

    if (!ok) {
        char* ref_str = mpz_get_str(nullptr, 16, ref);
        char* impl_str = mpz_get_str(nullptr, 16, impl);
        std::cout << "[4_2][case p*p] MISMATCH\n";
        std::cout << "  ref = 0x" << ref_str << "\n";
        std::cout << "  out = 0x" << impl_str << "\n";
        void (*gmp_free_func)(void*, size_t);
        mp_get_memory_functions(nullptr, nullptr, &gmp_free_func);
        gmp_free_func(ref_str, 0);
        gmp_free_func(impl_str, 0);
    } else {
        std::cout << "[4_2][case p*p] OK (ref == out)\n";
    }

    mpz_clear(impl);
    mpz_clear(ref);
    mpz_clear(B);
    mpz_clear(A);
    mpz_clear(p);

    return ok;
}
