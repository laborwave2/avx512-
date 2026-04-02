#pragma once
#include <immintrin.h>  
#include <cstdint>
#include <array>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>

struct alignas(64) Pair512 {
    __m512i a[18];
    __m512i b[18];
};

uint64_t rdtscp() ;

void cpuid_serialize() ;

void pin_to_core(int core);

std::vector<Pair512> make_random_pairs512(size_t N, uint64_t seed = 42);
void bench_vs_gmp_8_1_avx512F(size_t N = 2000, uint64_t seed = 42);

template<typename F>
void bench_cycles_512(const char* name, F&& f, const std::vector<Pair512>& vec, int rounds = 100)
{
    pin_to_core(2);

    uint64_t sink = 0;
    alignas(64) __m512i out[18]={};
    

    for (int i = 0; i < 1000; ++i) {
        const auto& p = vec[static_cast<size_t>(i) % vec.size()];
        f(const_cast<__m512i*>(p.a), const_cast<__m512i*>(p.b), out);
 
        alignas(64) uint64_t tmp[8];
        _mm512_store_si512((void*)tmp, out[0]);
        sink ^= tmp[0];
    }

    const size_t N = vec.size();
    uint64_t best = ~0ull;

   
    for (int r = 0; r < rounds; ++r) {
        cpuid_serialize();
        uint64_t t0 = rdtscp();
        for (size_t i = 0; i < N; ++i) {
            f(const_cast<__m512i*>(vec[i].a), const_cast<__m512i*>(vec[i].b), out);
            
            alignas(64) uint64_t tmp[8];
            _mm512_store_si512((void*)tmp, out[0]);
            sink ^= tmp[0];
        }
        uint64_t t1 = rdtscp();
        cpuid_serialize();
        best = std::min(best, t1 - t0);
    }

  
    auto c0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        f(const_cast<__m512i*>(vec[i].a), const_cast<__m512i*>(vec[i].b), out);
        alignas(64) uint64_t tmp[8];
        _mm512_store_si512((void*)tmp, out[1]);
        sink ^= tmp[0];
    }
    auto c1 = std::chrono::high_resolution_clock::now();
    double ns_total = std::chrono::duration<double, std::nano>(c1 - c0).count();

    double cyc_op = static_cast<double>(best) / static_cast<double>(N);
    double ns_op  = ns_total / static_cast<double>(N);

    std::cout << std::left << std::setw(28) << name
              << "  cycles/op = " << std::setw(12) << static_cast<long long>(cyc_op)
              << "  ns/op ≈ "     << std::fixed << std::setprecision(2) << ns_op
              << "   (sink="      << sink << ")\n";
}
