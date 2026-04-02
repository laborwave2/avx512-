#include<immintrin.h>
#include <immintrin.h>
#include <stdint.h>
#pragma once


inline const __m512i MASK29 = _mm512_set1_epi64(0x1fffffff);
void mul4_2(__m512i a[9], __m512i b[9], __m512i out[27]);
void mod4_2(__m512i in[27], __m512i out[9],const __m512i P[9]);
void mulmod4_2(__m512i a[18], __m512i b[18], __m512i out[9],const __m512i P[9]);