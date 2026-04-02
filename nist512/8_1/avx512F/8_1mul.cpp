#include"8_1mul.h"
#include <immintrin.h>


void mul8_1(__m512i a[18],__m512i b[18], __m512i out[36]){
    
    alignas(64) __m512i t[18]={0};
        for(int i=0;i<18;i++){
            int k=0;
            for(int j=0;j<i+1;j++){
                k=i-j;
                t[i]=_mm512_add_epi64(t[i],_mm512_mul_epu32(a[j],b[k]));
            }
            
        }
    __m512i acc=_mm512_srli_epi64(t[17],12);
    t[17]=_mm512_and_epi64(t[17],MASK12);


    alignas(64) __m512i r[18]={0};
    for(int i=18;i<35;i++){
        for(int j=i-17;j<18;j++){
            int k=i-j;
            acc=_mm512_add_epi64(acc,_mm512_mul_epu32(a[j],b[k]));
            r[i-18]=_mm512_and_epi64(acc,MASK29);
            acc=_mm512_srli_epi64(acc,29);
        }
    }
    r[17]=acc;
    
    for(int i=0;i<18;i++){
        out[i]=t[i];
    }
    for(int i=18;i<36;i++){
        out[i]=r[i-18];
    }

}


void mod8_1(__m512i in[36], __m512i out[18]){

    alignas(64) __m512i acc=_mm512_setzero_si512();

for(int i=0;i<17;i++){
    in[i]+=in[i+18];
    in[i]+=acc;
    acc=_mm512_srli_epi64(in[i],29);
    in[i]=_mm512_and_epi64(in[i],MASK29);

}
    in[17]+=in[35];
    acc=_mm512_srli_epi64(in[17],12);
    in[17]=_mm512_and_epi64(in[17],MASK12);

for (int i=0;i<17;i++)
{
    in[i]+=acc;
    acc=_mm512_srli_epi64(in[i],29);
    in[i]=_mm512_and_epi64(in[i],MASK29);
}
   in[17]+=acc;
   for(int i=0;i<18;i++){
       out[i]=in[i];
}
}

void mulmod8_1(__m512i a[18],__m512i b[18], __m512i out[18]){
    alignas(64) __m512i t[36]={0};
    mul8_1(a,b,t);
    mod8_1(t,out);
}