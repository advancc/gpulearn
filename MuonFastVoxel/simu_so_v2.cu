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
// #include <math.h>
// #include <math_constants.h>

// typedef struct arr
// {
//     double *pBase;//存储的是数组第一个元素的地址
//     int len;//数组能容纳的最大元素的个数
//     int cnt;//有效数组个数
//             //自动增长因子
// } Arr;

typedef struct res_arr
{
	double *arr;
	int *pmt_list;
	int index;
	int id;
	// int begin;
	// int len;
} Res_Arr;


__device__ double generateRandom(curandState *state);
__device__ void generateRandomInit(curandState *state,int seed);
__device__ int sampling(curandState state,double *histo,int max,int id);
__device__ int binarySearch(double *histo,double target,int max,int id);
__device__ double calculateAngle(double x,double y,double z,double a,double b,double c);
__device__ void generateHits(double r,double theta, double ratio,int pmtid,double start_time,double *hittime,double *npe,curandState state,Res_Arr r_arr);
__device__ void save_hits(Res_Arr *p,double val);
__device__ double get_hittime(double r, double theta, int mode, double *hittime, curandState state);
__device__ double get_hittime_bin(int binx, int biny, int mode, double *hittime, curandState state);
__device__ double get_hittime_all(int binx, int biny,double *hittime, curandState state);
__device__ int get_bin_x(double r);
__device__ int get_bin_y(double theta);
__device__ int r_findBin(double r);
__device__ int get_npe(double r,double theta,double *npe,curandState state);
__device__ int r3_findBin(double r3);
__device__ int theta_findBin(double theta);
__device__ int get_npe_num(int binx,int biny,double *npe,curandState state);
__device__ int generateRandomInt(curandState *state,int begin,int end);

__global__ void pmt_calculate(double r,double pos_x,double pos_y,double pos_z,double *pmt_x,double *pmt_y,double *pmt_z,double intPart,double fractionPart,double start_time,int numElements,double *hittime,double *npe,int *seed,double *result,int *pmt_res_list);
__host__ void step_calculate(double r,double pos_x,double pos_y,double pos_z,double intPart,double fractionPart,double start_time,double *pmt_x,double *pmt_y,double *pmt_z,double *hittime,double *npe,int *seed,double *result,int *pmt_list,int size);
// __device__ void init_arr(Arr *pArr, int length);//初始化数组
// __device__ bool append_arr(Arr *pArr, int val);//追加，可能成功，可能失败
// __device__ bool is_full_arr(Arr *pArr);//是否满了
__device__ void append_res_arr(Res_Arr *p, double val);
__device__ void init_res_arr(Res_Arr *p,double *result,int *pmt_res_list,int pmtid,int size);

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
#define pmt_num 17746


__global__ void
pmt_calculate(double r,double pos_x,double pos_y,double pos_z,double *pmt_x,double *pmt_y,double *pmt_z,double intPart,double fractionPart,double start_time,int numElements,double *hittime,double *npe,int *seed,double *result,int *pmt_res_list,int size){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	//result[id*50*]
	// double hittime_single;
	//numElements = pmt numbers
	if (id < numElements){
		curandState state;
		generateRandomInit(&state, seed[id]);
		Res_Arr pmt_arr;
		init_res_arr(&pmt_arr,result,pmt_res_list,id,size);
		double theta = calculateAngle(pmt_x[id],pmt_y[id],pmt_z[id],pos_x,pos_y,pos_z);
		// for(int j = 0; j < intPart; ++j){
		// 	generateHits(r,theta,1,id,start_time,hittime,npe,state,pmt_arr);
		// 	// save_hit(&pmt_arr,hittime_single);
		// 	// save_hits_simple(hittime_single);
		// }
		generateHits(r,theta,fractionPart,id,start_time,hittime,npe,state,pmt_arr);
		// save_hits_simple(&pmt_arr,hittime_single);
	}
}


__host__ void
step_calculate(double r,double pos_x,double pos_y,double pos_z,double intPart,double fractionPart,double start_time,double *pmt_x,double *pmt_y,double *pmt_z,double *hittime,double *npe,int *seed,double *result,int *pmt_list,int size){
	//设置线程数量
	printf("[GPU]启动核函数\n");
	CHECK(cudaDeviceSynchronize());
	int threadPerBlock=1024;
	int blocksPerGrid = ceil(pmt_num/threadPerBlock);
	dim3 block(threadPerBlock);
	//设置块数量
	dim3 grid(blocksPerGrid);//blocksPerGrid
	printf("[GPU]网格，线程(%d,%d)\n",blocksPerGrid,threadPerBlock);
	CHECK(cudaDeviceSynchronize());
	pmt_calculate<<<grid, block>>>(r,pos_x,pos_y,pos_z,pmt_x,pmt_y,pmt_z,intPart,fractionPart,start_time,pmt_num,hittime,npe,seed,result,pmt_list,size);
}



__device__ double
calculateAngle(double x,double y,double z,double a,double b,double c)
{
	double result = 0;
	result = acos((a*x+b*y+c*z)/(norm3d(x,y,z)*norm3d(a,b,c)));
	return result;
}

__device__ void 
generateHits(double r,double theta, double ratio,int pmtid,double start_time,double *hittime,double *npe,curandState state,Res_Arr r_arr)
{
	int npe_histo_id = get_npe(r,theta,npe,state);
	if (npe_histo_id>0)
	{
		for (int hitj = 0; hitj < npe_histo_id; ++hitj) 
		{
            // skip the photon according to the energy deposit
			if (ratio<1 and generateRandom(&state)>ratio) 
			{
                continue;
            }
            double hittime_single = start_time;
            // (m_flag_time) 
        	hittime_single += get_hittime(r, theta, 0, hittime, state);
            // generated hit
			// (m_flag_savehits) 
			append_res_arr(&r_arr,hittime_single);
            // save_hits(pmtid, hittime_single,result);
        }
	}
}

// __device__ void 
// save_hits(Res_Arr *p,double val){
// 	append_res_arr(p,val);
// }

__device__ double///////////////
get_hittime(double r, double theta, int mode, double *hittime, curandState state) {
	int binx = get_bin_x(r);
	int biny = get_bin_y(theta);
	return get_hittime_bin(binx, biny, mode, hittime, state);
}

__device__ double 
get_hittime_bin(int binx, int biny, int mode, double *hittime, curandState state) {
	// hit time = tmean + tres
	double hittime_single = 0.0;
	if (mode == 0) {
		hittime_single = get_hittime_all(binx,biny,hittime,state);
	}
	return hittime_single;
}

__device__ double 
get_hittime_all(int binx, int biny,double *hittime, curandState state) {
	// TH1F* h = get_hist(binx, biny);
	int xbinnum = 200;
	int ybinnum = 180;
	if (binx<1) { binx = 1; }
    else if (binx > xbinnum) { binx = xbinnum;}
    if (biny<1) { biny = 1; }
    else if (biny > ybinnum) { biny = ybinnum;}

    int idx = (binx-1)*ybinnum+(biny-1);
	double hittime_single = sampling(state,hittime,3000,idx);
	return hittime_single;
}


__device__ int 
get_bin_x(double r) 
{
	int binx = 1;
	int xmode = 2;
	if (xmode == 2) //KR
	{
		binx = r_findBin(r);
	} 
	return binx;
}

__device__ int 
get_bin_y(double theta) {
	int biny = 1;
	int ymode = 4;
	if (ymode == 4) {
		biny = theta_findBin(theta);
	}
	return biny;
}

__device__ int 
r_findBin(double r)
{
	const int binnum = 200;
	const double begin = 0;
	const double end = 17.7;
	return (int)floor(r/(end-begin)*binnum);
}

__device__ int 
get_npe(double r,double theta,double *npe,curandState state)
{
	int binx = r3_findBin(pow(r,3));
	int biny = theta_findBin(theta);
	return get_npe_num(binx,biny,npe,state);
}

__device__ int 
r3_findBin(double r3)
{
	const int binnum = 100;
	const double begin = 0;
	const double end = 5600;
	return (int)floor(r3/(end-begin)*binnum);
}


__device__ int 
theta_findBin(double theta)
{
	const int binnum = 180;
	const double begin = 0;
	const double end = 180.01*acos(1.0)/180.0;
	return (int)floor(theta/(end-begin)*binnum);
}

__device__ int 
get_npe_num(int binx,int biny,double *npe,curandState state)
{
	int npe_from_single = 0;
	if (1 <= binx and binx <= 100 and 1 <= biny and biny <= 180) {
		npe_from_single = sampling(state,npe,50,(binx-1)*180+(biny-1));	
	} else if (binx==1 and (biny<1 or biny>180)) {
		biny = generateRandomInt(&state,1,180);
		npe_from_single = sampling(state,npe,50,(binx-1)*180+(biny-1));	
	} else if (binx>1 and (biny<1 or biny>180)) {
		if (biny>180) { biny = 180; }
		else if (biny<1){ biny = 1; }
		npe_from_single = sampling(state,npe,50,(binx-1)*180+(biny-1));
	} else {
		static long warning = 0;
		++warning;
		if (warning < 10) {
			printf("npe lost: %d/%d", binx,biny);
		} else if (warning == 10) {
			printf("too many npe lost complains.");
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

__device__ int
generateRandomInt(curandState *state,int begin,int end)
{
    int result = begin+int(ceil(abs(curand_uniform_double(state))*(end-begin)));
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
	return -1;
}

// __device__ bool 
// is_full_arr(Arr *pArr)//是否满了
// {
//     if (pArr->cnt == pArr->len) {
//         return true;
//     }
//     else{
//         return false;
//     }
// }

// __device__ bool 
// append_arr(Arr *pArr, int val)//追加，可能成功，可能失败
// {
//     //满时返回false
//     if (is_full_arr(pArr))
//         return false;
//     //不满时追加
//     pArr->pBase[pArr->cnt] = val;
//     (pArr->cnt)++;
//     return true;
// }

__device__ void
append_res_arr(Res_Arr *p, double val)//追加，可能成功，可能失败
{
	p->arr[p->index+p->pmt_list[p->id]] = val;
	p->pmt_list[p->id] += 1;
	return;
	//满时返回false
	// int flag = 0;
	// for(int i = 0 ; i < p->num;i++){
	// 	if(!is_full_arr(&(p->arr[i]))){
	// 		flag = 1;
	// 		append_arr(&(p->arr[i]),val);
	// 	}
	// }
	// if (flag == 0){
	// 	init_arr(&(p->arr[p->num]),100);
	// 	p->num +=1;
	// }
}

// __device__ void 
// init_arr(Arr *pArr, int length) {
//     //分配初始化数组
//     pArr->pBase = (double*)malloc(sizeof(double)*length);
//     if (NULL == pArr->pBase)
//     {
//         printf("free error 动态内存分配失败！\n");
//         exit(-1);//终止整个程序
//     }
//     else
//     {
//         pArr->len = length;
//         pArr->cnt = 0;
//     }
//     return;
// }

__device__ void
init_res_arr(Res_Arr *p,double *result,int *pmt_res_list,int pmtid,int size){
	p->arr = result;
	p->pmt_list = pmt_res_list;
	p->index = pmtid*size*50;//存储该pmt在数组中的起始存取点
	p->id = pmtid;
	// p->begin = begin;
	// p->len = len;
	return;
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
	float GPU_Sampling_wrapper(double *r,double *pos_x,double *pos_y,double *pos_z, double *intPart, double *fractionPart,double *start_time,double *pmt_x,double *pmt_y,double *pmt_z,double *data_hit,double *data_npe,int *seed,int *size,double* h_result)
    {
		//GPU计时，设置开始和结束事件
		cudaEvent_t start, stop;
		// cudaEvent_t gpu_start,gpu_stop;
		CHECK(cudaEventCreate(&start));
		CHECK(cudaEventCreate(&stop));
		// cudaEventCreate(&gpu_start);
		// cudaEventCreate(&gpu_stop);
		cudaEventRecord(start);
		CHECK(cudaDeviceSynchronize());
		printf("[GPU]GPU时间已记录\n");
        //申请GPU内存
		double *d_r, *d_pos_x,*d_pos_y,*d_pos_z,*d_intPart,*d_fractionPart,*d_start_time;
		double *d_pmt_x,*d_pmt_y,*d_pmt_z,*d_data_hit,*d_data_npe;
		double *d_result;
		int *d_seed;
		int pmt_res_list[pmt_num] = {0};
		CHECK(cudaMalloc((double**)&d_r,size[0]));
		CHECK(cudaMalloc((double**)&d_pos_x,size[0]));
		CHECK(cudaMalloc((double**)&d_pos_y,size[0]));
		CHECK(cudaMalloc((double**)&d_pos_z,size[0]));
		CHECK(cudaMalloc((double**)&d_intPart,size[0]));
		CHECK(cudaMalloc((double**)&d_fractionPart,size[0]));
		CHECK(cudaMalloc((double**)&d_start_time,size[0]));
		CHECK(cudaMalloc((double**)&d_pmt_x,size[1]));
		CHECK(cudaMalloc((double**)&d_pmt_y,size[1]));
		CHECK(cudaMalloc((double**)&d_pmt_z,size[1]));
		CHECK(cudaMalloc((double**)&d_data_hit,size[2]));
		CHECK(cudaMalloc((double**)&d_data_npe,size[3]));
		CHECK(cudaMalloc((int**)&d_seed,size[4]));
		CHECK(cudaMalloc((double**)&d_result,pmt_num*size[0]*50));
        //将CPU内存拷贝到GPU
	    CHECK(cudaMemcpy(d_r, r, size[0], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_pos_x, pos_x, size[0], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_pos_y, pos_y, size[0], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_pos_z, pos_z, size[0], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_intPart, intPart, size[0], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_fractionPart, fractionPart, size[0], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_start_time, start_time, size[0], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_pmt_x, pmt_x, size[1], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_pmt_y, pmt_y, size[1], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_pmt_z, pmt_z, size[1], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_data_hit, data_hit, size[2], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_data_npe, data_npe, size[3], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_seed, seed, size[4], cudaMemcpyHostToDevice));
		
		printf("[GPU]GPU数据拷贝完成\n");
        //设置使用编号为0的GPU
	    CHECK(cudaSetDevice(0));
		// //设置线程数量
		// int threadPerBlock = 1024;
		// int blocksPerGrid =ceil(17746/1024);
		
	    // dim3 block(threadPerBlock);
	    // //设置块数量
		// dim3 grid(blocksPerGrid);//blocksPerGrid

		//调用核函数
		for(int i = 0;i<size[0]/8;i++) {
			printf("[GPU]开始运行核函数[%d]\n",i);
			CHECK(cudaDeviceSynchronize());
			step_calculate(d_r[i],d_pos_x[i],d_pos_y[i],d_pos_z[i],d_intPart[i],d_fractionPart[i],d_start_time[i],d_pmt_x,d_pmt_y,d_pmt_z,d_data_hit,d_data_npe,(int*)(d_seed+i*pmt_num),d_result,pmt_res_list,size[0]/8);
		}
		
		
		printf("[GPU]核函数运行完成\n");
		CHECK(cudaDeviceSynchronize());

		CHECK(cudaMemcpy(h_result, d_result, pmt_num*size[0]*50, cudaMemcpyDeviceToHost));
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float total_time;
		//计算用时，精度0.5us
		cudaEventElapsedTime(&total_time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("total use time %f ms\n", total_time);
		// printf("threadPerBlock:%d\n",threadPerBlock);
		// printf("blocksPerGrid；%d\n",blocksPerGrid);
		
		// cudaEventElapsedTime(&time, gpu_start, gpu_stop);
		// cudaEventDestroy(gpu_start);
		// cudaEventDestroy(gpu_stop);

        //释放GPU内存
	    CHECK(cudaFree(d_r));
		CHECK(cudaFree(d_pos_x));
		CHECK(cudaFree(d_pos_y));
		CHECK(cudaFree(d_pos_z));
		CHECK(cudaFree(d_intPart));
		CHECK(cudaFree(d_fractionPart));
		CHECK(cudaFree(d_data_hit));
		CHECK(cudaFree(d_data_npe));
		CHECK(cudaFree(d_start_time));
		CHECK(cudaFree(d_pmt_x));
		CHECK(cudaFree(d_pmt_y));
		CHECK(cudaFree(d_pmt_z));
		CHECK(cudaFree(d_seed));
		CHECK(cudaFree(d_result));
		printf("[GPU]GPU运行完成\n");
		return total_time;
		// return 0.0;
    }
}