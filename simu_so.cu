#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

__device__ double generateRandom(curandState *state);
__device__ void generateRandomInit(curandState *state);
//错误处理宏
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
CDF_Sampling(double *pmt, double *hittime, double *result, int numElements,int max_n,int max_time)
{
    int id = threadIdx.x;
	curandState state;
	generateRandomInit(&state);
    if (id < numElements)
    {
		double prob; 
		prob = generateRandom(&state);
		double sum = 0;
		int n = 0;
		for (int item = 0; item < max_n;item++)
		{
			sum += pmt[id*max_n+item];
			if (prob <= sum)
			{
				n = item;
				printf("thread %d: hit times:%d\n", id, n);
				break;
			}
		}
		for (int item = 0;item < n;item++) 
		{
			double prob2;
			prob2 = generateRandom(&state);
			double sum = 0;
			for (int j = 0; j < max_time;j++)
			{
				sum += hittime[id*max_time+j];
				if (prob2 <= sum)
				{
					result[id*max_n+item] = (double)j;
					printf("thread %d: %dth hit time %d\n", id, item+1,j);
					break;
				}
			}

		}
    }
}
__device__ double
generateRandom(curandState *state)
{
	int id = threadIdx.x;
    double result = abs(curand_uniform_double(state));
	printf("thread:%d random double: %f \n",id,result);
	return result;
}

__device__ void
generateRandomInit(curandState *state)
{
	int id = threadIdx.x;
	long seed = (unsigned long long)clock();
	curand_init(seed, id, 0, state);
}

extern "C" 
{
    void CDF_Sampling_wrapper(double *h_pmt,double *h_hit,double *h_result, int total_num, int nBytes,int max_n,int max_time)
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
	    CHECK(cudaMalloc((double**)&d_pmt,nBytes));
	    CHECK(cudaMalloc((double**)&d_hit, nBytes));
	    CHECK(cudaMalloc((double**)&d_result, nBytes));
        //将CPU内存拷贝到GPU
	    CHECK(cudaMemcpy(d_pmt, h_pmt, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_hit, h_hit, nBytes, cudaMemcpyHostToDevice));
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
		CDF_Sampling <<<grid, block >>>(d_pmt, d_hit, d_result, total_num,max_n,max_time);
		
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
    }
}