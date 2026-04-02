#pragma once
#include <immintrin.h>

inline const __m512i MASK52= _mm512_set1_epi64(0xFFFFFFFFFFFFFULL);  
inline const __m512i MASK37= _mm512_set1_epi64(0x1FFFFFFFFFULL);    
void mul8_1(__m512i a[10],__m512i b[10], __m512i out[20]);
void mod8_1(__m512i in[20], __m512i out[10]);
void mulmod8_1(__m512i a[10],__m512i b[10], __m512i out[10]);
