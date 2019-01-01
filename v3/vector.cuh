#ifndef VECTOR_CUH
#define VECTOR_CUH

typedef struct res_arr
{
    double *arr;
    int *pmt_list;
    int index;
    int id;
    // int begin;
    // int len;
} Res_Arr;

__device__ void append_res_arr(Res_Arr *p, double val);
__device__ void init_res_arr(Res_Arr *p,double *result,int *pmt_res_list,int pmtid,int size);

#endif //VECTOR_CUH