#include"4_2mulifma.h"
#include <immintrin.h>

void  mul4_2(__m512i a[5], __m512i b[5], __m512i out[15]){
   for (int i = 0; i < 15;i++){
      out[i] = _mm512_setzero_si512();
   }
      
   for (int i = 0; i < 5; i++)
   {
      __m512i M = _mm512_shuffle_epi32(b[i], static_cast<_MM_PERM_ENUM>(0x44));
      out[i]=_mm512_madd52lo_epu64(out[i], M, a[0]);
      for (int j = 0; j < 5; j++)
      {
        out[i+j]=_mm512_madd52lo_epu64(out[i+j], M, a[j]);
        out[i+j]=_mm512_madd52hi_epu64(out[i+j], M, a[j-1]);
      }
        out[i+5]=_mm512_madd52hi_epu64(out[i+5], M, a[4]);
   }
   for (int i = 0; i < 5;i++)
   {
      __m512i N = _mm512_shuffle_epi32(b[i], static_cast<_MM_PERM_ENUM>(0xEE));
        out[i+5]=_mm512_madd52lo_epu64(out[i+5], N, a[0]);
      for (int j = 0; j < 5;j++){
         out[i+j+5]=_mm512_madd52lo_epu64(out[i+j+5], N, a[j]);
        out[i+j+5]=_mm512_madd52hi_epu64(out[i+j+5], N, a[j-1]);
        
      }
       out[i+10]=_mm512_madd52hi_epu64(out[i+10], N, a[4]);
   }
}

void mod4_2(__m512i in[15], __m512i out[5], const __m512i P[5]){
   for (int i = 0; i < 10;i++){
      __m512i C = _mm512_srli_epi64(in[i], 52);
      in[i] = _mm512_and_epi64(in[i], MASK52);
      __m512i t = in[i + 1];
      in[i + 1] = _mm512_add_epi64(t, C);
      __m512i U = _mm512_shuffle_epi32(in[i], static_cast<_MM_PERM_ENUM>(0x44));
       in[i]=_mm512_madd52lo_epu64(in[i], U, P[0]);
      for (int j = 0; j < 5;j++){
        in[i+j]=_mm512_madd52lo_epu64(in[i+j], U, P[j]);
        in[i+j]=_mm512_madd52hi_epu64(in[i+j], U, P[j-1]);
      }
      in[i+5]=_mm512_madd52hi_epu64(in[i+5], U, P[4]);
      in[i + 5] = _mm512_mask_add_epi64(in[i + 5], 0x55, in[i + 5], _mm512_shuffle_epi32(in[i], static_cast<_MM_PERM_ENUM>(0x4E)));
   }
   for (int k = 0; k < 5;k++){
      out[k] = in[k + 10];
   }
   for (int i = 0; i < 2;i++){
      for (int j = 0; j < 4;j++){
         __m512i C = _mm512_srli_epi64(out[j], 52);
         out[j] = _mm512_and_epi64(out[j], MASK52);
         out[j + 1] = _mm512_add_epi64(out[j + 1], C);
      }
   }
   __m512i C = _mm512_srli_epi64(out[4], 52);
   out[4] = _mm512_and_epi64(out[4], MASK52);
   out[4] = _mm512_mask_add_epi64(out[0], 0xAA, out[0], _mm512_shuffle_epi32(C, static_cast<_MM_PERM_ENUM>(0x4E)));
}

void mulmodifma4_2(__m512i a[5], __m512i b[5], __m512i out[5],const __m512i P[5]){
   __m512i m[15];
   mul4_2(a, b, m);
   mod4_2(m, out, P);
}