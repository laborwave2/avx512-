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

#include "8_1mul.h"

struct alignas(64) Pair512 {
    __m512i a[18];
    __m512i b[18];
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


std::vector<Pair512> make_random_pairs512(size_t N, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    const uint64_t MASK29_VAL = 0x1FFFFFFF;
    
    std::vector<Pair512> vec(N);
    
    for (size_t n = 0; n < N; ++n) {
        for (int k = 0; k < 18; ++k) {
            alignas(64) uint64_t a_lanes[8], b_lanes[8];
            for (int lane = 0; lane < 8; ++lane) {
                a_lanes[lane] = rng() & MASK29_VAL;
                b_lanes[lane] = rng() & MASK29_VAL;
            }
            vec[n].a[k] = _mm512_load_si512((const void*)a_lanes);
            vec[n].b[k] = _mm512_load_si512((const void*)b_lanes);
        }
    }
    return vec;
}


