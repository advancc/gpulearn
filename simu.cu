/**
 * Author:yipeihuai
 * Mail:yiph@ihep.ac.cn
 * Function:Accelerate simulation with GPU
 */

#include <stdio.h>
#include <stdlib.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <string.h>

__device__ double generateRandom();
//__global__ void generateRandom(long rand, double *result);
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
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
#define CHECK_CURAND(call) \
{\
	const cudaError_t error = call;\
	if (error != CURAND_STATUS_SUCCESS)\
	{\
		printf("Error:%s:%d, ", __FILE__, __LINE__);\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
		exit(2);\
	}\
}
long 
getFileSize(FILE* file)
{
	long size;
	fseek(file, 0, SEEK_END);
	size = ftell(file);
	rewind(file);
	return size;
}

void 
readFile(FILE* file, char* buffer, long size)
{
	size_t result;
	result = fread(buffer, 1, size, file);
	if (result != size)
	{
		printf("file read error:%d/%d", result, size);
		exit(3);
	}
}

__global__ void
vectorAdd(double *pmt, double *hittime, double *result, int numElements)
{
    int id = threadIdx.x;
	
    if (id < numElements)
    {
		double prob; 
		prob = generateRandom();
		double sum = 0;
		int n = 0;
		//抽样是否可用纹理内存，simon
		for (int item = 0; item < 10;item++)
		{
			sum += pmt[id*10+item];
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
			prob2 = generateRandom();
			double sum = 0;
			for (int j = 0; j < 10;j++)
			{
				sum += hittime[id*10+j];
				if (prob2 <= sum)
				{
					result[id*10+item] = (double)j;
					printf("thread %d: %dth hit time %d\n", id, item+1,j);
					break;
				}
			}

		}
    }
}
__device__ double
generateRandom()
{
	curandState state;
	int id = threadIdx.x;
	//种子要用固定的种子，改成种子列表存在内存中。
	//
	long seed = id+(unsigned long long)clock();
	//不要多次初始化。
	curand_init(seed, id, 0, &state);
    double result = abs(curand_uniform_double(&state));
	printf("thread:%d random double: %f \n",id,result);
	return result;
}

/**
 * Host main routine
 */
 //编写动态库，方便其他文件调用，CPU/GPU版本，共用接口
 //探索是否能用python和cuda和C++联合一起用
int
main(void)
{
	//获取真实数据，进行真正的模拟。
	int total_num = 10;
	int max_n = 10;
	int max_time = 10;
	size_t nBytes = total_num * max_n * sizeof(double);
	double *pmt;
	pmt = (double*)malloc(nBytes);
	//NumPy内存通信
	for (int i = 0;i < total_num;i++)
	{
		for (int j = 0;j < max_n;j++)
		{
			pmt[i*total_num +j] = 0.1;
		}
	}
	
	double *hittime;
	hittime = (double*)malloc(nBytes);
	for (int i = 0;i < total_num;i++)
	{
		for (int j = 0;j < max_time;j++)
		{
			hittime[i*total_num+j] = 0.1;
		}
	}
	
	
	double *h_res = (double*)malloc(nBytes);

	double *d_pmt, *d_hit,*d_result;
	//GPU内存分配，线程与块级的全局内存和局部内存的分配
	CHECK(cudaMalloc((double**)&d_pmt,nBytes));
	CHECK(cudaMalloc((double**)&d_hit, nBytes));
	CHECK(cudaMalloc((double**)&d_result, nBytes));
	CHECK(cudaMemcpy(d_pmt, pmt, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_hit, hittime, nBytes, cudaMemcpyHostToDevice));

	// srand((unsigned int)time(NULL));
	cudaSetDevice(0);
	//自动分配多GPU，网格、线程
	int blocksPerGrid = 1;
	int threadsPerBlock = 10;

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	vectorAdd<<<blocksPerGrid, threadsPerBlock >>>(d_pmt,d_hit,d_result,total_num);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("GPU use time %f ms（us）\n",time);
	//测试时间要画图，画图量化GPU对算法的提升
	//测量时间的精度
	// CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(h_res, d_result, nBytes, cudaMemcpyDeviceToHost));
	
	for (int i = 0;i < total_num;i++)
	{
		for (int j = 0;j < max_time;j++)
		{
			printf("%f ",h_res[i*10+j]);
		}
		printf("\n");
	}DeviceSynchronize());
	CHECK(cudaMemcpy(h_res, d_result, nBytes, cudaMemcpyDeviceToHost));
	
	for (int i = 0;i < total_num;i++)
	{
		for (int j = 0;j < max_time;j++)
		{
			printf("%f ",h_res[i*10+j]);
		}
		printf("\n");
	}

	CHECK(cudaFree(d_pmt));
	CHECK(cudaFree(d_hit));
	CHECK(cudaFree(d_result));
	free(pmt);
	free(hittime);
	free(h_res);
	cudaDeviceReset();
	return 0;
 