#include "test.cuh"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
__global__ void kernelPrint(){
    printf("GPU run!\n");
}

void CudaRun(){
    printf("cpu run!\n");
    kernelPrint<<<1,5>>>();
    cudaDeviceSynchronize();
}
