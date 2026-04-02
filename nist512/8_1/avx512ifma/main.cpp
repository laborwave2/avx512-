#include "8_1mulifma.h"
#include "8_1bench.h"

#include "test.h"


int main() {
    if (!run_tests_8_1_avx512ifma(200, 12345, true)) return 1;
 
    const size_t N = 100000;
    auto data = make_random_pairs512(N, 42);

    bench_cycles_512("mulmod8_1ifma", mulmod8_1, data, 100);

    return 0;
}
