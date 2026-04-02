#include"4_2mulifma.h"
#include"4_2bench.h"
int main() {
    const size_t N = 100000;
    auto data = make_random_pairs512(N, 42);


    bench_cycles_512("mulmodifma4_2", mulmodifma4_2, data, 100);

 
    return 0;
}

