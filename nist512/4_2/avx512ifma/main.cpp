#include"4_2mulifma.h"
#include"4_2bench.h"
int main() {
    const size_t N = 100000;
    auto data = make_random_pairs512(N, 42);


    bench_cycles_512("mulmodifma4_2", mulmodifma4_2, data, 100);
    bench_vs_gmp_4_2_avx512ifma(2000, 42);

 
    return 0;
}
