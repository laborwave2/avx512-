#include"4_2mul.h"
#include <immintrin.h>

void  mul4_2(__m512i a[9], __m512i b[9], __m512i out[27]){
   for (int i = 0; i < 27;i++){
      out[i] = _mm512_setzero_si512();
   }
      
   for (int i = 0; i < 9; i++)
   {
      __m512i M = _mm512_shuffle_epi32(b[i], static_cast<_MM_PERM_ENUM>(0x44));
      for (int j = 0; j < 8; j++)
      {
         __m512i I = out[i + j];
         out[i + j] = _mm512_add_epi64(I, _mm512_mul_epu32(M, a[j]));
      }
      out[i + 8] = _mm512_mul_epu32(M, a[8]);
   }
   for (int i = 0; i < 9;i++)
   {
      __m512i N = _mm512_shuffle_epi32(b[i], static_cast<_MM_PERM_ENUM>(0xEE));
      for (int j = 0; j < 8;j++){
         __m512i L = out[i + j + 9];
         out[i + j + 9] = _mm512_add_epi64(L, _mm512_mul_epu32(N, a[j]));
        
      }
       out[i + 17] = _mm512_mul_epu32(N, a[8]);
   }
}

void mod4_2(__m512i in[27], __m512i out[9], const __m512i P[9]){
   for (int i = 0; i < 18;i++){
      __m512i C = _mm512_srli_epi64(in[i], 29);
      in[i] = _mm512_and_epi64(in[i], MASK29);
      __m512i t = in[i + 1];
      in[i + 1] = _mm512_add_epi64(t, C);
      __m512i U = _mm512_shuffle_epi32(in[i], static_cast<_MM_PERM_ENUM>(0x44));
      for (int j = 0; j < 9;j++){
         __m512i g = in[i + j];
         in[i + j] = _mm512_add_epi64(g, _mm512_mul_epu32(U,P[j]));
      }
      in[i + 9] = _mm512_mask_add_epi64(in[i + 9], 0x55, in[i + 9], _mm512_shuffle_epi32(in[i ], static_cast<_MM_PERM_ENUM>(0x4E)));
   }
   for (int k = 0; k < 9;k++){
      out[k] = in[k + 18];
   }
   for (int i = 0; i < 2;i++){
      for (int j = 0; j < 8;j++){
         __m512i C = _mm512_srli_epi64(out[j], 29);
         out[j] = _mm512_and_epi64(out[j], MASK29);
         out[j + 1] = _mm512_add_epi64(out[j + 1], C);
      }
   }
   __m512i C = _mm512_srli_epi64(out[8], 29);
   out[8] = _mm512_and_epi64(out[8], MASK29);
   out[0] = _mm512_mask_add_epi64(out[0], 0xAA, out[0], _mm512_shuffle_epi32(C, static_cast<_MM_PERM_ENUM>(0x4E)));

   // final normalization after folding carry into out[0]
   for (int j = 0; j < 8; j++) {
      __m512i C2 = _mm512_srli_epi64(out[j], 29);
      out[j] = _mm512_and_epi64(out[j], MASK29);
      out[j + 1] = _mm512_add_epi64(out[j + 1], C2);
   }
   __m512i C3 = _mm512_srli_epi64(out[8], 29);
   out[8] = _mm512_and_epi64(out[8], MASK29);
   out[0] = _mm512_mask_add_epi64(out[0], 0xAA, out[0], _mm512_shuffle_epi32(C3, static_cast<_MM_PERM_ENUM>(0x4E)));

   // one more carry sweep to ensure canonical limbs
   for (int j = 0; j < 8; j++) {
      __m512i C4 = _mm512_srli_epi64(out[j], 29);
      out[j] = _mm512_and_epi64(out[j], MASK29);
      out[j + 1] = _mm512_add_epi64(out[j + 1], C4);
   }
   __m512i C5 = _mm512_srli_epi64(out[8], 29);
   out[8] = _mm512_and_epi64(out[8], MASK29);
   out[0] = _mm512_mask_add_epi64(out[0], 0xAA, out[0], _mm512_shuffle_epi32(C5, static_cast<_MM_PERM_ENUM>(0x4E)));
}

void mulmod4_2(__m512i a[9], __m512i b[9], __m512i out[9],const __m512i P[9]){
   __m512i m[27];
   mul4_2(a, b, m);
   mod4_2(m, out, P);
}