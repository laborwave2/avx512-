#include"4_2mul.h"
#include"4_2bench.h"
#include "test.h"
int main() {
   /* if (!run_test_4_2_avx512F()) {
        return 1;
    }*/
    const size_t N = 100000;
    auto data = make_random_pairs512(N, 42);

    bench_cycles_512("mulmod4_2", mulmod4_2, data, 100);
    bench_vs_gmp_4_2_avx512F(2000, 42);


    return 0;
}
