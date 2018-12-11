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
#include <math.h>
#include <math_constants.h>


__device__ double generateRandom(curandState *state);
__device__ void generateRandomInit(curandState *state,int seed);
__device__ int sampling(curandState state,double *histo,int max,int id);
__device__ int binarySearch(double *histo,double target,int max,int id);
__device__ double calculateAngle(double x,double y,double z,double a,double b,double c);
__global__ void pmt_calculate(double r,double pos_x,double pos_y,double pos_z,double *pmt_x,double *pmt_y,double *pmt_z,int intPart,double fractionPart,double start_time,int numElements);
// float CDF_Sampling_Wrapping(double *h_pmt,double *h_hit,double *h_result, int *seed,int total_num, int nBytes,int max_n,int max_time);

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

#define npe_file_path "./npe_cdf.npy"
#define hit_file_path "./hittime_cdf.npy"
#define pmt_num 17746


// __global__ void
// CDF_Sampling(double *pmt, double *hittime, double *result, int *seed,int numElements,int max_n,int max_time)
// {
//     int id = blockIdx.x*blockDim.x+threadIdx.x;
//     curandState state;
//     generateRandomInit(&state,seed[id]);
//     if (id < numElements)
//     {
//         int n = sampling(state,pmt,max_n,id);
// 		for (int item = 0;item < n;item++) 
// 		{
// 			result[id*max_n+item] = (double)sampling(state,hittime,max_time,id);
// 		}
//     }
// }
__global__ void
pmt_calculate(double r,double pos_x,double pos_y,double pos_z,double *pmt_x,double *pmt_y,double *pmt_z,int intPart,double fractionPart,double start_time,int numElements){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	//numElements = pmt numbers
	if (id < numElements){
		double theta = calculateAngle(pmt_x[id],pmt_y[id],pmt_z[id],pos_x,pos_y,pos_z);
		for(int j = 0; j < intPart; ++j){
			generateHits(r,theta,1,id,start_time);
		}
		generateHits(r,theta,fractionPart,id,start_time);
	}
}
__global__ void
step_calculate(double *r,double *pos_x,double *pos_y,double *pos_z,double *pmt_x,double *pmt_y,double *pmt_z,int *intPart,double *fractionPart,double *start_time){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	//设置线程数量
	int threadPerBlock=1024;
	int blocksPerGrid = ceil(pmt_num/threadPerBlock);
	dim3 block(threadPerBlock);
	//设置块数量
	dim3 grid(blocksPerGrid);//blocksPerGrid
	pmt_calculate<<<grid, block>>>(r[id],pos_x[id],pos_y[id],pos_z[id],pmt_x,pmt_y,pmt_z,intPart[id],fractionPart[id],start_time[id],pmt_num);
}



__device__ double
calculateAngle(double x,double y,double z,double a,double b,double c){
	double result = 0;
	result = acos((a*x+b*y+c*z)/(norm3d(x,y,z)*norm3d(a,b,c)));
	return result;
}

__device__ generateHits(double r,double theta, double ratio,int pmtid,double start_time)
{
	int npe_histo_id = get_npe(r,theta);
	if (npe_histo_id>0){
		//code
	}
}
__device__ int get_npe(double r,double theta){
	int binx = r_findBin(pow(r,3));
	int biny = theta_findBin(theta);
	return get_npe_num(binx,biny);
}
__device__ int r_findBin(double r3){
	const int binnum = 100;
	const double begin = 0;
	const double end = 5600;
	return (int)floor(r3/(end-begin)*binnum) ;
}
__device__ int theta_findBin(double theta){
	const int binnum = 180;
	const double begin = 0;
	const double end = 180.01*acos(1)/180.0;
	return (int)floor(theta/(end-begin)*binnum) ;
}
__device__ int get_npe_num(int binx,int biny){
	int npe_from_single = 0;
	// code?????????????????????????????????????
	if (1<=binx and binx<=100 and 1<=biny and biny<=180) {
		TH1F* th = ths_npe[binx-1][biny-1];
		npe_from_single = th->GetRandom();
	} else if (binx==1 and (biny<1 or biny>180)) {
		biny = gRandom->Uniform(1,180);
		TH1F* th = ths_npe[binx-1][biny-1];
		npe_from_single = th->GetRandom();
	} else if (binx>1 and (biny<1 or biny>180)) {
		// std::cerr << "npe maybe lost: " << binx << "/" << biny << std::endl;
		// FIXME how to handle such situation.
		// biny = gRandom->Uniform(1,100);
		if (biny>180) { biny = 180; }
		else if (biny<1){ biny = 1; }

		TH1F* th = ths_npe[binx-1][biny-1];
		npe_from_single = th->GetRandom();
	} else {
		static long warning = 0;
		++warning;
		if (warning < 10) {
			std::cerr << "npe lost: " << binx << "/" << biny << std::endl;
		} else if (warning == 10) {
			std::cerr << "too many npe lost complains." << std::endl;
		}
	}
	return npe_from_single;
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
/*
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
*/
extern "C"
{
	float CDF_Sampling_wrapper(double *h_npe,double *h_hit,double *h_pmt_pos,double *h_result, int *seed, int total_num, int nBytes,int max_n,int max_time)
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
		step_calculate <<<grid, block >>>(d_pmt, d_hit, d_result, d_seed,total_num,max_n,max_time);
		
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
