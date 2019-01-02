#ifndef SIMU_MUON_CUH
#define SIMU_MUON_CUH

#include <cuda.h>
#include <stdio.h>
#include "vector.cuh"
#include "sample.cuh"
#include <cuda_runtime_api.h>
#include <math_constants.h>

__global__ void pmt_calculate(double r,double pos_x,double pos_y,double pos_z,double *pmt_x,double *pmt_y,double *pmt_z,double intPart,double fractionPart,double start_time,int numElements,double *hittime_histo,double *npe,int *seed,double *result,int *pmt_res_list,int size);
__device__ double calculateAngle(double x,double y,double z,double a,double b,double c);
__device__ void generateHits(double r,double theta, double ratio,double start_time,double *hittime_histo,double *npe,curandState *state,Res_Arr r_arr);
__device__ int get_hittime(double r, double theta, int mode, double *hittime_histo, curandState *state);
__device__ int get_hittime_bin(int binx, int biny, int mode, double *hittime_histo, curandState *state);
__device__ int get_hittime_all(int binx, int biny,double *hittime_histo, curandState *state);
__device__ int get_bin_x(double r);
__device__ int get_bin_y(double theta);
__device__ int r_findBin(double r);
__device__ int get_npe(double r,double theta,double *npe,curandState *state);
__device__ int r3_findBin(double r3);
__device__ int theta_findBin(double theta);
__device__ int get_npe_num(int binx,int biny,double *npe,curandState *state);

#define CHECK(call) \
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        printf("Error:%s:%d, ", __FILE__, __LINE__);\
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}
#define pmt_num 17746
#define pmt_mem 2000
#define CUDART_PI_F 3.141592654f

#endif