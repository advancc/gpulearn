/**
 * Author:���໴
 * Mail:yiph@ihep.ac.cn
 * Function:Accelerate simulation with GPU
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <string.h>

__device__ double generateRandom(curandState *state);
__device__ void generateRandomInit(curandState *state);

//�������
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
//�������
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

//�ں˺��� ��__global__ǰ׺���εĺ����ڱ�������GPU���룬��CPU���ã�����CPUȫ�ֿɼ�
__global__ void
CDF_Sampling(double *pmt, double *hittime, double *result, int numElements)
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
			prob2 = generateRandom(&state);
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
//GPU���õĺ��� ��__device__ǰ׺���εĺ�����GPU�����У���CPU���ɼ�
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

/**
 * Host main routine
 */
int
main(void)
{
	
	//���ɼ�����
	int total_num = 10;
	int max_n = 10;
	int max_time = 10;
	size_t nBytes = total_num * max_n * sizeof(double);
	double *pmt;
	pmt = (double*)malloc(nBytes);
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

	//GPU��ʱ�����ÿ�ʼ�ͽ����¼�
	cudaEvent_t start, stop, gpu_start,gpu_stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(start);

	//����GPU�ڴ�
	double *d_pmt, *d_hit,*d_result;
	CHECK(cudaMalloc((double**)&d_pmt,nBytes));
	CHECK(cudaMalloc((double**)&d_hit, nBytes));
	CHECK(cudaMalloc((double**)&d_result, nBytes));
	//��CPU�ڴ濽����GPU
	CHECK(cudaMemcpy(d_pmt, pmt, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_hit, hittime, nBytes, cudaMemcpyHostToDevice));

	//����ʹ�ñ��Ϊ0��GPU
	cudaSetDevice(0);


	//�����߳�����
	dim3 block(total_num);//threadsPerBlock
	//���ÿ�����
	dim3 grid(total_num / block.x);//blocksPerGrid

	cudaEventRecord(gpu_start);

	//���ú˺���
	CDF_Sampling <<<grid, block >>>(d_pmt, d_hit, d_result, total_num);

	cudaEventRecord(gpu_stop);
	cudaEventSynchronize(gpu_stop);//ͬ����ǿ��CPU�ȴ�GPU event���趨

	
	//��GPU�������ݵ�CPU
	CHECK(cudaMemcpy(h_res, d_result, nBytes, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);


	float time;
	//������ʱ������0.5us
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	printf("total use time %f ms\n", time);
	cudaEventElapsedTime(&time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);
	printf("gpu use time %f ms\n", time);

	for (int i = 0;i < total_num;i++)
	{
		for (int j = 0;j < max_time;j++)
		{
			printf("%f ",h_res[i*10+j]);
		}
		printf("\n");
	}
	//�ͷ�GPU�ڴ�
	CHECK(cudaFree(d_pmt));
	CHECK(cudaFree(d_hit));
	CHECK(cudaFree(d_result));
	free(pmt);
	free(hittime);
	free(h_res);
	//�����ռGPU��Դ
	cudaDeviceReset();
	return 0;
 
}

