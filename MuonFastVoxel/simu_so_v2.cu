/**
 * Author:易培淮
 * Mail:yiph@ihep.ac.cn
 * Function:Accelerate simulation with Single GPU
 * 2018/11/27
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>


__device__ double generateRandom(curandState *state);
__device__ void generateRandomInit(curandState *state,int seed);
__device__ int sampling(curandState state,double *histo,int max,int id);
float CDF_Sampling_Wrapping(double *h_pmt,double *h_hit,double *h_result, int *seed,int total_num, int nBytes,int max_n,int max_time);

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

__global__ void
CDF_Sampling(double *pmt, double *hittime, double *result, int *seed,int numElements,int max_n,int max_time)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    curandState state;
    generateRandomInit(&state,seed[id]);
    if (id < numElements)
    {
        int n = sampling(state,pmt,max_n,id);
		for (int item = 0;item < n;item++) 
		{
			result[id*max_n+item] = (double)sampling(state,hittime,max_time,id);
		}
    }
}
__device__ double
generateRandom(curandState *state)
{
    double result = abs(curand_uniform_double(state));
    return result;
}
__device__ void
generateRandomInit(curandState *state,int seed)
{
    curand_init(seed, 0, 0, state);
}
__device__ int 
sampling(curandState state,double *histo,int max,int id)
{
    double prob; 
	prob = generateRandom(&state);
	return binarySearch(histo,prob,max,id);
	// int result = 0;
	// return result;
	// double sum = 0;
    // for (int item = 0; item < max;item++)
    // {
    //     sum += histo[id*max+item];
    //     if (prob <= sum)
    //     {
    //         result = item;
    //         printf("thread %d: hit times:%d\n", id, result);
    //         break;
    //     }
    // }
    // return result;
}
__device__ int
binarySearch(double *histo,double target,int max,int id)
{
	int start = 0;
	int end = max-1; 
	int mid;

	while(start+1<end){
		mid = start+(end-start)/2;
		if (histo[id*max+mid]==target){
			end = mid;
		} 
		else if (histo[id*max+mid] < target){
			start = mid;
		}
		else if (histo[id*max+mid] > target){
			end = mid;
		}
	}
	if (histo[id*max+start] == target){
		return start;
	}
	if (histo[id*max+end] == target){
		return end;
	}
	return -1
}

extern "C" 
{
    float CDF_Sampling_wrapper(double *h_pmt,double *h_hit,double *h_result, int *seed, int total_num, int nBytes,int max_n,int max_time)
    {
		//GPU计时，设置开始和结束事件
		cudaEvent_t start, stop, gpu_start,gpu_stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_stop);
		cudaEventRecord(start);
        //申请GPU内存
		double *d_pmt, *d_hit,*d_result;
		int *d_seed;
	    CHECK(cudaMalloc((double**)&d_pmt,nBytes));
	    CHECK(cudaMalloc((double**)&d_hit, nBytes));
		CHECK(cudaMalloc((double**)&d_result, nBytes));
		CHECK(cudaMalloc((int**)&d_seed,nBytes/2));
        //将CPU内存拷贝到GPU
	    CHECK(cudaMemcpy(d_pmt, h_pmt, nBytes, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_hit, h_hit, nBytes, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_seed,seed,nBytes/2,cudaMemcpyHostToDevice));
        //设置使用编号为0的GPU
	    CHECK(cudaSetDevice(0));
		//设置线程数量
		int threadPerBlock,blocksPerGrid;
		if (total_num<128)
		{
			threadPerBlock = 128;
			blocksPerGrid =1;
		}
		else if(total_num<1024)
		{
			threadPerBlock = 128;
			blocksPerGrid =int(ceil(total_num/(double)threadPerBlock));
		}
		else
		{
			threadPerBlock = 1024;
			blocksPerGrid =int(ceil(total_num/(double)threadPerBlock));
		}
		
	    dim3 block(threadPerBlock);
	    //设置块数量
		dim3 grid(blocksPerGrid);//blocksPerGrid
		
		cudaEventRecord(gpu_start);
        //调用核函数
		CDF_Sampling <<<grid, block >>>(d_pmt, d_hit, d_result, d_seed,total_num,max_n,max_time);
		
		cudaEventRecord(gpu_stop);
		cudaEventSynchronize(gpu_stop);//同步，强制CPU等待GPU event被设定

        CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(h_result, d_result, nBytes, cudaMemcpyDeviceToHost));
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float time,total_time;
		//计算用时，精度0.5us
		cudaEventElapsedTime(&total_time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		printf("threadPerBlock:%d\n",threadPerBlock);
		printf("blocksPerGrid；%d\n",blocksPerGrid);
		printf("total use time %f ms\n", total_time);
		cudaEventElapsedTime(&time, gpu_start, gpu_stop);
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_stop);
		printf("gpu use time %f ms\n", time);
		printf("占用内存：%d B\n", nBytes);
		printf("占用内存：%d kB\n", nBytes / 1024);
        //释放GPU内存
	    CHECK(cudaFree(d_pmt));
	    CHECK(cudaFree(d_hit));
		CHECK(cudaFree(d_result));
		return total_time;
    }
}

