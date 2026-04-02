#include <cstdint>
#include <array>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <immintrin.h>
#include <cstring>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include "4_2bench.h"
#include "4_2mulifma.h"
#include <gmp.h>

alignas(64) extern const __m512i P[5] = {
    _mm512_setr_epi64(0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL),
    _mm512_setr_epi64(0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL),
    _mm512_setr_epi64(0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL),
    _mm512_setr_epi64(0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL),
    _mm512_setr_epi64(0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL,0x000FFFFFFFFFFFFFULL)
};

uint64_t rdtscp() {
    unsigned aux, lo, hi;
    asm volatile ("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) ::"memory");
    return ((uint64_t)hi << 32) | lo;
}

void cpuid_serialize() {
    unsigned a, b, c, d;
    asm volatile ("cpuid" : "=a"(a),"=b"(b),"=c"(c),"=d"(d) : "a"(0) : "memory");
}

void pin_to_core(int core) {
    if (core < 0) return;

    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu <= 0 || core >= ncpu) {
        return;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);

    int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        std::cerr << "Failed to set CPU affinity: " << strerror(rc) << std::endl;
    }
}

std::vector<Pair512> make_random_pairs512(size_t N, uint64_t seed ) {
    std::mt19937_64 rng(seed);
    const uint64_t MASK52_VAL = 0x000FFFFFFFFFFFFFULL;

    std::vector<Pair512> vec(N);

    for (size_t n = 0; n < N; ++n) {
        for (int k = 0; k < 5; ++k) {
            alignas(64) uint64_t a_lanes[8], b_lanes[8];
            for (int lane = 0; lane < 8; ++lane) {
                a_lanes[lane] = rng() & MASK52_VAL;
                b_lanes[lane] = rng() & MASK52_VAL;
            }
            vec[n].a[k] = _mm512_load_si512((const void*)a_lanes);
            vec[n].b[k] = _mm512_load_si512((const void*)b_lanes);
        }
    }
    return vec;
}

void bench_vs_gmp_4_2_avx512ifma(size_t N, uint64_t seed) {
    auto data = make_random_pairs512(N, seed);
    alignas(64) __m512i out[5]{};

    mpz_t p, A, B, C;
    mpz_init(p); mpz_init(A); mpz_init(B); mpz_init(C);
    mpz_set_ui(p, 1);
    mpz_mul_2exp(p, p, 505);
    mpz_sub_ui(p, p, 1);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        mulmodifma4_2(const_cast<__m512i*>(data[i].a), const_cast<__m512i*>(data[i].b), out, P);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        alignas(64) uint64_t al[8], bl[8];
        uint64_t a_limb[10]{}, b_limb[10]{};
        for (int limb = 0; limb < 5; ++limb) {
            _mm512_store_si512((void*)al, data[i].a[limb]);
            _mm512_store_si512((void*)bl, data[i].b[limb]);
            a_limb[limb] = al[0] & 0xFFFFFFFFFFFFFULL;
            a_limb[limb + 5] = al[1] & 0xFFFFFFFFFFFFFULL;
            b_limb[limb] = bl[0] & 0xFFFFFFFFFFFFFULL;
            b_limb[limb + 5] = bl[1] & 0xFFFFFFFFFFFFFULL;
        }
        a_limb[9] &= 0x1FFFFFFFFFULL;
        b_limb[9] &= 0x1FFFFFFFFFULL;

        mpz_set_ui(A, static_cast<unsigned long>(a_limb[9]));
        mpz_set_ui(B, static_cast<unsigned long>(b_limb[9]));
        for (int limb = 8; limb >= 0; --limb) {
            mpz_mul_2exp(A, A, 52);
            mpz_mul_2exp(B, B, 52);
            mpz_add_ui(A, A, static_cast<unsigned long>(a_limb[limb]));
            mpz_add_ui(B, B, static_cast<unsigned long>(b_limb[limb]));
        }
        mpz_mul(C, A, B);
        mpz_mod(C, C, p);
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    double avx_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / double(N);
    double gmp_ns = std::chrono::duration<double, std::nano>(t3 - t2).count() / double(N);
    std::cout << "[bench-vs-gmp][4_2 avx512ifma] avx ns/op=" << avx_ns
              << "  gmp ns/op(lane-pair0)=" << gmp_ns << "\n";

    mpz_clear(p); mpz_clear(A); mpz_clear(B); mpz_clear(C);
}
