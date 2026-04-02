#include "8_1bench.h"
#include"8_1mul.h"

#include "test.h"

#include <cstdint>

int main() {
    if (!run_tests_8_1_avx512F(200, 12345)) return 1;

    const size_t N = 100000;
    auto data = make_random_pairs512(N, 42);


   
    bench_cycles_512("mulmod8_1", mulmod8_1, data, 100);
    bench_vs_gmp_8_1_avx512F(2000, 42);

 
    return 0;
}
