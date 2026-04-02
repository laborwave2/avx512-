#pragma once
#include <immintrin.h>


inline const __m512i MASK29 = _mm512_set1_epi64(0x1fffffff);
inline const __m512i MASK12 = _mm512_set1_epi64(0xFFF);

void mul8_1(__m512i a[18], __m512i b[18], __m512i out[36]);
void mod8_1(__m512i in[36], __m512i out[18]);
void mulmod8_1(__m512i a[18], __m512i b[18], __m512i out[18]);
