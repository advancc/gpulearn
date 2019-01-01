#ifndef SAMPLE_CUH
#define SAMPLE_CUH

#include <curand.h>
#include <curand_kernel.h>

__device__ double generateRandom(curandState *state);
__device__ void generateRandomInit(curandState *state,int seed);
__device__ int sampling(curandState *state,double *histo,int max,int id);
__device__ int binarySearch(double *histo,double target,int max,int id);
__device__ int generateRandomInt(curandState *state,int begin,int end);

#endif