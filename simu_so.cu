#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <string.h>

__device__ double generateRandom(curandState *state);
__device__ void generateRandomInit(curandState *state,int i );

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

typedef struct
{
	double *d_pmt, *d_hit, *d_result;
	// double *h_result_s;
	cudaStream_t stream;
	cudaEvent_t start, stop;
}GPU_data;


__global__ void
CDF_Sampling(double *pmt, double *hittime, double *result, int numElements,int max_n,int max_time,int gpu_id)
{
	//compute one-dimensional data index 
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	curandState state;
	generateRandomInit(&state,id);
    if (id < numElements)
    {
		double prob; 
		prob = generateRandom(&state);
		double sum = 0;
		int n = 0;
		for (int item = 0; item < max_n; item++)
		{
			sum += pmt[id*max_n+item];
			if (prob <= sum)
			{
				n = item;
				// printf("[gpu: %d] thread %d: hit times:%d\n",gpu_id, id, n);
				break;
			}
		}
		for (int item = 0;item < n;item++) 
		{
			// double prob2;
			prob = generateRandom(&state);
			sum = 0;
			for (int j = 0; j < max_time;j++)
			{
				sum += hittime[id*max_time+j];
				if (prob <= sum)
				{
					result[id*max_n+item] = (double)j;
					// printf("[gpu: %d] thread %d: %dth hit time %d\n",gpu_id, id, item+1,j);
					break;
				}
			}

		}
    }
} 
__device__ double
generateRandom(curandState *state)
{
	// int id = blockIdx.x*blockDim.x+threadIdx.x;
    double result = abs(curand_uniform_double(state));
	// printf("thread:%d random double: %f \n",id,result);
	return result;
}

__device__ void
generateRandomInit(curandState *state,int id)
{
	// long seed = (unsigned long long)clock();
	curand_init(id, 0, 0, state);
}
// __host__ void
// initData(double *h_pmt, double *h_hit, int Size)
// {

// }

extern "C" 
{
    float CDF_Sampling_wrapper(double *h_pmt,double *h_hit,double *h_result, int total_num, int nBytes,int max_n,int max_time)
    {
		//GPU计时，设置开始和结束事件
		// cudaEvent_t start, stop;
		// cudaEventCreate(&start);
		// cudaEventCreate(&stop);
		
		
		//获取GPU数量
		int GPU_num;
		CHECK(cudaGetDeviceCount(&GPU_num));
		// printf("GPU number:%d\n",GPU_num);
		if(GPU_num<1)
		{
			printf("no CUDA capable devices were detected\n");
			return -1;
		}
		GPU_data data[GPU_num];
		int single_size = total_num/GPU_num;
		int single_bytes = nBytes/GPU_num;
		// cudaEventRecord(start,0);
        //申请GPU内存
		// double *d_pmt, *d_hit,*d_result;
		for(int gpu_id =0; gpu_id < GPU_num; gpu_id++)
		{
			cudaSetDevice(gpu_id);
			cudaStreamCreate(&data[gpu_id].stream);
			cudaEventCreate(&data[gpu_id].start);
			cudaEventCreate(&data[gpu_id].stop);
			cudaEventRecord(data[gpu_id].start,data[gpu_id].stream);
			CHECK(cudaMalloc((double**)&(data[gpu_id].d_pmt),single_bytes));
	    	CHECK(cudaMalloc((double**)&(data[gpu_id].d_hit), single_bytes));
			CHECK(cudaMalloc((double**)&(data[gpu_id].d_result), single_bytes));
			// data[gpu_id].h_result_s = (double*)malloc(single_bytes);
			//
			CHECK(cudaMemcpyAsync(data[gpu_id].d_pmt, (double*)(h_pmt+gpu_id*single_bytes/8), single_bytes, cudaMemcpyHostToDevice, data[gpu_id].stream));
			CHECK(cudaMemcpyAsync(data[gpu_id].d_hit, (double*)(h_hit+gpu_id*single_bytes/8), single_bytes, cudaMemcpyHostToDevice, data[gpu_id].stream));
		}
	    
        //将CPU内存拷贝到GPU
	    // CHECK(cudaMemcpy(d_pmt, h_pmt, nBytes, cudaMemcpyHostToDevice));
		// CHECK(cudaMemcpy(d_hit, h_hit, nBytes, cudaMemcpyHostToDevice));
		
        //设置使用编号为0的GPU
		// CHECK(cudaSetDevice(0));
		
		//设置线程数量
		int threadPerBlock,blocksPerGrid;
		if (single_size<128)
		{
			threadPerBlock = 128;
			blocksPerGrid =1;
		}
		else if(single_size<1024)
		{
			threadPerBlock = 128;
			blocksPerGrid =int(ceil(single_size/(double)threadPerBlock));
		}
		else
		{
			threadPerBlock = 1024;
			blocksPerGrid =int(ceil(single_size/(double)threadPerBlock));
		}
		
	    dim3 block(threadPerBlock);
	    //设置块数量
		dim3 grid(blocksPerGrid);//blocksPerGrid
		
		// cudaEventRecord(gpu_start);
		//调用核函数
		for(int gpu_id = 0; gpu_id < GPU_num; gpu_id++)
		{
			cudaSetDevice(gpu_id);
			//第三个参数为0，表示每个block用到的共享内存大小为0
			CDF_Sampling <<<grid, block, 0,data[gpu_id].stream >>>(data[gpu_id].d_pmt, data[gpu_id].d_hit, data[gpu_id].d_result, single_size,max_n,max_time,gpu_id);
			// CHECK(cudaMemcpyAsync(data[gpu_id].h_result_s, data[gpu_id].d_result, single_bytes, cudaMemcpyDeviceToHost,data[gpu_id].stream));
			CHECK(cudaMemcpyAsync((double*)(h_result+gpu_id*single_bytes/8),data[gpu_id].d_result,single_bytes,cudaMemcpyDeviceToHost,data[gpu_id].stream));
		}
		
		
		
		// cudaEventRecord(gpu_stop);
		// cudaEventSynchronize(gpu_stop);//同步，强制CPU等待GPU event被设定

		CHECK(cudaDeviceSynchronize());
		for(int gpu_id = 0; gpu_id < GPU_num; gpu_id++)
		{
			cudaEventRecord(data[gpu_id].stop,data[gpu_id].stream);
			cudaEventSynchronize(data[gpu_id].stop);
		}
		
		// CHECK(cudaMemcpy(h_result, d_result, nBytes, cudaMemcpyDeviceToHost));
		//等待stream流执行完成
		// for(int gpu_id = 0; gpu_id < GPU_num; gpu_id++)
		// {
		// 	CHECK(cudaStreamSynchronize(data[gpu_id].stream));
		// 	cudaEventRecord(data[gpu_id].stop,data[gpu_id].stream);
		// 	cudaEventSynchronize(data[gpu_id].stop);
		// }
		// for(i  nt gpu_id = 0; gpu_id < GPU_num; gpu_id++)
		// {
		// 	memcpy(h_result+gpu_id*single_bytes,data[gpu_id].h_result_s,single_bytes);
		// }
		// cudaEventRecord(stop,0);
		// cudaEventSynchronize(stop);
		float total_time;
		//计算用时，精度0.5us
		cudaEventElapsedTime(&total_time, data[0].start, data[0].stop);
		// cudaEventDestroy(start);
		// cudaEventDestroy(stop);

		// printf("threadPerBlock:%d\n",threadPerBlock);
		// printf("blocksPerGrid；%d\n",blocksPerGrid);
		// printf("total use time %f ms\n", total_time);
		// cudaEventElapsedTime(&time, gpu_start, gpu_stop);
		// cudaEventDestroy(gpu_start);
		// cudaEventDestroy(gpu_stop);
		// printf("gpu use time %f ms\n", time);
		// printf("占用内存：%d B\n", nBytes);
		// printf("占用内存：%d kB\n", nBytes / 1024);

		//释放GPU内存
		for(int gpu_id = 0; gpu_id < GPU_num; gpu_id++)
		{  
			CHECK(cudaFree(data[gpu_id].d_pmt));
	    	CHECK(cudaFree(data[gpu_id].d_hit));
			CHECK(cudaFree(data[gpu_id].d_result));
			CHECK(cudaStreamDestroy(data[gpu_id].stream));
			CHECK(cudaEventDestroy(data[gpu_id].start));
			CHECK(cudaEventDestroy(data[gpu_id].stop));
		}
	  
		CHECK(cudaDeviceReset());
		return total_time;
    }
}