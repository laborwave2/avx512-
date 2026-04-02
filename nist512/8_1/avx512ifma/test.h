#pragma once
#include <cstddef>

bool run_tests_8_1_avx512ifma(std::size_t iterations = 2000, unsigned seed = 1, bool strict = true);
void bench_ref_8_1_avx512ifma(std::size_t iterations = 200, unsigned seed = 1);
