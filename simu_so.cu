#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

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
    void CDF_Sampling_wrapper(double *h_pmt,double *h_hit,double *h_result,int nBytes)
    {
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
	    dim3 block(256);
	    //设置块数量
        dim3 grid(total_num / block.x);//blocksPerGrid
        //调用核函数
        CDF_Sampling <<<grid, block >>>(d_pmt, d_hit, d_result, total_num);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_result, d_result, nBytes, cudaMemcpyDeviceToHost));
        //释放GPU内存
	    CHECK(cudaFree(d_pmt));
	    CHECK(cudaFree(d_hit));
	    CHECK(cudaFree(d_result));
    }
}