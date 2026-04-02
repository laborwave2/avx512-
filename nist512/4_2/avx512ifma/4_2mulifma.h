#include<immintrin.h>
#include <immintrin.h>
#include <stdint.h>
#pragma once


inline const __m512i MASK52 = _mm512_set1_epi64(0x000FFFFFFFFFFFFF);
void mul4_2(__m512i a[5], __m512i b[5], __m512i out[15]);
void mod4_2(__m512i in[15], __m512i out[5],const __m512i P[5]);
void mulmodifma4_2(__m512i a[5], __m512i b[5], __m512i out[5],const __m512i P[5]);