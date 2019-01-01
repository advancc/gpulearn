#include "vector.cuh"

__device__ void
append_res_arr(Res_Arr *p, double val)//追加，可能成功，可能失败
{
    p->arr[p->index+p->pmt_list[p->id]] = val;
    p->pmt_list[p->id] += 1;
    return;

}

__device__ void
init_res_arr(Res_Arr *p,double *result,int *pmt_res_list,int pmtid,int size){
    p->arr = result;//存储的内存空间
    p->pmt_list = pmt_res_list;//存储每个pmt内存空间使用量
    p->index = pmtid*pmt_mem;//存储该pmt在数组中的起始存取点
    p->id = pmtid;
    // p->begin = begin;
    // p->len = len;
    return;
}