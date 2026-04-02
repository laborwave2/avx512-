#include"8_1mulifma.h"
#include <immintrin.h>


void mul8_1(__m512i a[10],__m512i b[10], __m512i out[20]){
   
    out[0]=_mm512_madd52lo_epu64(out[0],a[0],b[0]);
    __m512i carry=_mm512_setzero_si512(); 
        for(int i=1;i<10;i++){
            int k=0;
            for(int j=0;j<i+1;j++){
                k=i-j;
                out[i]=_mm512_madd52lo_epu64( out[i],a[j],b[k]);
            }
             for(int j=0;j<i;j++){
                k=i-j-1;
                out[i]=_mm512_madd52hi_epu64( out[i],a[j],b[k]);
            }
            out[i]=_mm512_add_epi64( out[i],carry);
            carry=_mm512_srli_epi64( out[i],52);
        }
       
        for(int i=10;i<19;i++){
            int k=0;
            for(int j=i-9;j<10;j++){
                k=i-j;
                out[i]=_mm512_madd52lo_epu64(out[i],a[j],b[k]);
            }
             for(int j=i-10;j<10;j++){
                k=i-j-1;
                out[i]=_mm512_madd52hi_epu64(out[i],a[j],b[k]);
            }
            out[i]=_mm512_add_epi64(out[i],carry);
            carry=_mm512_srli_epi64(out[i],52);
        }
        out[19]=carry;        
}


void mod8_1(__m512i in[20], __m512i out[10]){

    alignas(64) __m512i acc=_mm512_setzero_si512();

for(int i=0;i<9;i++){
    in[i]+=in[i+10];
    in[i]+=acc;
    acc=_mm512_srli_epi64(in[i],52);
    in[i]=_mm512_and_epi64(in[i],MASK52);

}
    in[9]+=in[19];
    in[9]+=acc;
    acc=_mm512_srli_epi64(in[9],37);
    in[9]=_mm512_and_epi64(in[9],MASK37);

for (int i=0;i<9;i++)
{
    in[i]+=acc;
    acc=_mm512_srli_epi64(in[i],52);
    in[i]=_mm512_and_epi64(in[i],MASK52);
}
   in[9]+=acc;
   for(int i=0;i<10;i++){
       out[i]=in[i];
}

}

void mulmod8_1(__m512i a[10],__m512i b[10], __m512i out[10]){
    __m512i t[20]={0};
    mul8_1(a,b,t);
    mod8_1(t,out);
}