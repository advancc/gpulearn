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
#include <math_constants.h>

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
__device__ int sampling(curandState *state,double *histo,int max,int id);
__device__ int binarySearch(double *histo,double target,int max,int id);
__device__ double calculateAngle(double x,double y,double z,double a,double b,double c);
__device__ void generateHits(double r,double theta, double ratio,int pmtid,double start_time,double *hittime,double *npe,curandState *state,Res_Arr r_arr);
__device__ void save_hits(Res_Arr *p,double val);
__device__ int get_hittime(double r, double theta, int mode, double *hittime, curandState *state);
__device__ int get_hittime_bin(int binx, int biny, int mode, double *hittime, curandState *state);
__device__ int get_hittime_all(int binx, int biny,double *hittime, curandState *state);
__device__ int get_bin_x(double r);
__device__ int get_bin_y(double theta);
__device__ int r_findBin(double r);
__device__ int get_npe(double r,double theta,double *npe,curandState *state);
__device__ int r3_findBin(double r3);
__device__ int theta_findBin(double theta);
__device__ int get_npe_num(int binx,int biny,double *npe,curandState *state);
__device__ int generateRandomInt(curandState *state,int begin,int end);

__global__ void pmt_calculate(double r,double pos_x,double pos_y,double pos_z,double *pmt_x,double *pmt_y,double *pmt_z,double intPart,double fractionPart,double start_time,int numElements,double *hittime,double *npe,int *seed,double *result,int *pmt_res_list,int size);
__global__ void init_random(curandState *state);
// __global__ void step_calculate(double r,double pos_x,double pos_y,double pos_z,double intPart,double fractionPart,double start_time,double *pmt_x,double *pmt_y,double *pmt_z,double *hittime,double *npe,int *seed,double *result,int *pmt_list,int size);
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
#define CUDART_PI_F 3.141592654f

__global__ void 
init_random(curandState *state)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // if (id < numElements){
    //     generateRandomInit((state+id), id);
    // }
}

__global__ void
pmt_calculate(double r,double pos_x,double pos_y,double pos_z,double *pmt_x,double *pmt_y,double *pmt_z,double intPart,double fractionPart,double start_time,int numElements,double *hittime,double *npe,int *seed,double *result,int *pmt_res_list,int size){
    int id = blockIdx.x*blockDim.x+threadIdx.x;   
    // printf("num= %d",numElements); 
    // double hittime_single;
    //numElements = pmt numbers
    if (id < numElements){
        curandState state;
        generateRandomInit(&state, seed[id]);
        Res_Arr pmt_arr;
        init_res_arr(&pmt_arr,result,pmt_res_list,id,size);
        double theta = calculateAngle(pmt_x[id],pmt_y[id],pmt_z[id],pos_x,pos_y,pos_z);
        // printf("theta = %lf\n",theta);
        for(int j = 0; j < intPart; ++j){
            //r 单位 米
        	generateHits(r,theta,1,id,start_time,hittime,npe,&state,pmt_arr);
        	// save_hit(&pmt_arr,hittime_single);
        	// save_hits_simple(hittime_single);
        }
        generateHits(r,theta,fractionPart,id,start_time,hittime,npe,&state,pmt_arr);
        // save_hits_simple(&pmt_arr,hittime_single);
    }
}


// __host__ void
// step_calculate(double r,double pos_x,double pos_y,double pos_z,double intPart,double fractionPart,double start_time,double *pmt_x,double *pmt_y,double *pmt_z,double *hittime,double *npe,int *seed,double *result,int *pmt_list,int size){
// 	//设置线程数量
// 	CHECK(cudaDeviceSynchronize());
// 	printf("[GPU]启动核函数\n");
    
// 	int threadPerBlock=1024;
// 	int blocksPerGrid = ceil(pmt_num/threadPerBlock);
// 	dim3 block(threadPerBlock);
// 	//设置块数量
// 	dim3 grid(blocksPerGrid);//blocksPerGrid
// 	printf("[GPU]网格，线程(%d,%d)\n",blocksPerGrid,threadPerBlock);
// 	CHECK(cudaDeviceSynchronize());
// 	pmt_calculate<<<grid, block>>>(r,pos_x,pos_y,pos_z,pmt_x,pmt_y,pmt_z,intPart,fractionPart,start_time,pmt_num,hittime,npe,seed,result,pmt_list,size);
// }



__device__ double
calculateAngle(double x,double y,double z,double a,double b,double c)
{
//  printf("x=%lf,y=%lf,z=%lf,a=%lf,b=%lf,c=%lf\n",x,y,z,a,b,c);
    double result = 0;
    if (a == 0 and b == 0 and c == 0){
        return result;
    }
    else{
        result = acos((a*x+b*y+c*z)/(norm3d(x,y,z)*norm3d(a,b,c)));
        //printf("result theta = %lf",result);
        return result;
    }
}

__device__ void 
generateHits(double r,double theta, double ratio,int pmtid,double start_time,double *hittime,double *npe,curandState *state,Res_Arr r_arr)
{
    
    int npe_histo_id = get_npe(r,theta,npe,state);
    //  printf("npe_histo_id = %d\n",npe_histo_id);
    if (npe_histo_id>0)
    {
        // printf("npe_histo_id = %d\n",npe_histo_id);
        for (int hitj = 0; hitj < npe_histo_id; ++hitj) 
        {
            printf("ratio=%lf\n",ratio);
            // skip the photon according to the energy deposit
            if (ratio<1 and generateRandom(state)>ratio) 
            {
                continue;
            }
            printf("start_time =%lf\n",start_time);
            double hittime_single = start_time;
            // (m_flag_time) 
            hittime_single += (double)get_hittime(r, theta, 0, hittime, state);
            printf("hittime = %lf\n",hittime_single);
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

__device__ int
get_hittime(double r, double theta, int mode, double *hittime, curandState *state) {
    int binx = get_bin_x(r);
    int biny = get_bin_y(theta);
    return get_hittime_bin(binx, biny, mode, hittime, state);
}

__device__ int 
get_hittime_bin(int binx, int biny, int mode, double *hittime, curandState *state) {
    // hit time = tmean + tres
    int hittime_single = 0;
    if (mode == 0) {
        hittime_single = get_hittime_all(binx,biny,hittime,state);
    }
    return hittime_single;
}

__device__ int 
get_hittime_all(int binx, int biny,double *hittime, curandState *state) {
    // TH1F* h = get_hist(binx, biny);
    const int xbinnum = 200;
    const int ybinnum = 180;
    if (binx<1) { binx = 1; }
    else if (binx > xbinnum) { binx = xbinnum;}
    if (biny<1) { biny = 1; }
    else if (biny > ybinnum) { biny = ybinnum;}

    int idx = (binx-1)*ybinnum+(biny-1);
    int hittime_single = sampling(state,hittime,3000,idx);
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
    if(r==0){
        return 1;
    }
    else{
        return (int)ceil((r-begin)/(end-begin)*binnum); 
    }
}

__device__ int 
get_npe(double r,double theta,double *npe,curandState *state)
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
    if(r3 == 0){
        return 1;
    }
    else{
        return (int)ceil((r3-begin)/(end-begin)*binnum);
    }
}


__device__ int 
theta_findBin(double theta)
{
    const int binnum = 180;
    const double begin = 0; 
    const double end = 180.01*CUDART_PI_F/180.0;
    //printf("theta = %lf\n",theta);
    if(theta == 0){
        // printf("theta = %lf,bin=%d\n",theta,1);
        return 1;
    }
    else{
        // printf("theta = %lf,bin=%d\n",theta,(int)ceil(theta/(end-begin)*binnum));
        return (int)ceil((theta-begin)/(end-begin)*binnum);
    }
}

__device__ int 
get_npe_num(int binx,int biny,double *npe,curandState *state)
{
    // printf("binx = %d,biny = %d\n",binx,biny);
    int npe_from_single = 0;
    if (1 <= binx and binx <= 100 and 1 <= biny and biny <= 180) {
        npe_from_single = sampling(state,npe,33,(binx-1)*180+(biny-1));	
    } else if (binx==1 and (biny<1 or biny>180)) {
        biny = generateRandomInt(state,1,180);
        npe_from_single = sampling(state,npe,33,(binx-1)*180+(biny-1));	
    } else if (binx>1 and (biny<1 or biny>180)) {
        if (biny>180) { biny = 180; }
        else if (biny<1){ biny = 1; }
        npe_from_single = sampling(state,npe,33,(binx-1)*180+(biny-1));
    } else {
        static long warning = 0;
        ++warning;
        if (warning < 10) {
            printf("npe lost: %d/%d\n", binx,biny);
        } else if (warning == 10) {
            printf("too many npe lost complains.\n");
        }
    }
    return npe_from_single;
}

__device__ double
generateRandom(curandState *state)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;   
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
    // printf("seed = %d\n",seed);
    int id = blockIdx.x*blockDim.x+threadIdx.x;   
    curand_init(seed, 0, 0, state);
}

__device__ int 
sampling(curandState *state,double *histo,int max,int id)
{
    double prob; 
    prob = generateRandom(state);
    return binarySearch(histo,prob,max,id);
}

__device__ int
binarySearch(double *histo,double target,int max,int id)
{
    
    // printf("prob=%lf \n",target);
    // if (max == 3000){
    //     for(int i = 0; i<max;i++){
    //         printf("%lf ",histo[id*max+i]);
    //     }
    // }
    
    // printf("]\n");
    // int result_for = -1;
    // int result_bin = 0;
    // for (int i=0;i<max;i++){
    //     if (target<=histo[id*max+i]){
    //         // printf("[debug]histo = %lf,%lf\n",histo[id*max],histo[id*max+i]);
    //         // printf("[debug]target=%lf,max=%d,id =%d,i=%d\n",target,max,id,i);
    //         return i;
    //     }
    // }
    // return result_for;
    // printf("error:histo = %lf,%lf\n",histo[id*max],histo[id*max+max-1]);
    // printf("error:target=%lf,max=%d,id =%d\n",target,max,id);
    // return -1;
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

        // if (max==3000){
        //     printf("[debug 3000] start = %d,end = %d,mid = %d\n",start,end,mid);
        // }
    }
    // if (end-start !=1 and max == 33){
    //     printf("error:end=%d, start=%d\n",end,start);
    // }
    if (target <= histo[id*max+start]){
        return start;
    }
    else if (histo[id*max+start] < target){
        return end;
    }
    // else if (histo[id*max+end] == target){
    //     result_bin = end;
    // }
    // if (result_bin != result_for){
    //     printf("error :result_bin =%d,result_for = %d\n",result_bin,result_for);
    // }
    // else{
    //     return result_bin;
    // }
    return -1;
}

__device__ void
append_res_arr(Res_Arr *p, double val)//追加，可能成功，可能失败
{
    p->arr[p->index+p->pmt_list[p->id]] = val;
    p->pmt_list[p->id] += 1;
    return;

}

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
    
        //申请GPU内存
        // double *d_r, *d_pos_x,*d_pos_y,*d_pos_z,*d_intPart,*d_fractionPart,*d_start_time;
        double *d_pmt_x,*d_pmt_y,*d_pmt_z,*d_data_hit,*d_data_npe;
        double *d_result;
        int *d_seed,*d_pmt_res_list;

        // CHECK(cudaMalloc((double**)&d_r,size[0]));
        // CHECK(cudaMalloc((double**)&d_pos_x,size[0]));
        // CHECK(cudaMalloc((double**)&d_pos_y,size[0]));
        // CHECK(cudaMalloc((double**)&d_pos_z,size[0]));
        // CHECK(cudaMalloc((double**)&d_intPart,size[0]));
        // CHECK(cudaMalloc((double**)&d_fractionPart,size[0]));
        // CHECK(cudaMalloc((double**)&d_start_time,size[0]));
        CHECK(cudaMalloc((double**)&d_pmt_x,size[1]));
        CHECK(cudaMalloc((double**)&d_pmt_y,size[1]));
        CHECK(cudaMalloc((double**)&d_pmt_z,size[1]));
        CHECK(cudaMalloc((double**)&d_data_hit,size[2]));
        CHECK(cudaMalloc((double**)&d_data_npe,size[3]));
        CHECK(cudaMalloc((int**)&d_seed,size[4]));
        CHECK(cudaMalloc((double**)&d_result,pmt_num*size[0]*50));
        CHECK(cudaMalloc((int**)&d_pmt_res_list,pmt_num*sizeof(int)));

        //设置内存
        CHECK(cudaMemset(d_pmt_res_list,0,pmt_num*sizeof(int)));
        CHECK(cudaMemset(d_result,0,pmt_num*size[0]*50));
        //将CPU内存拷贝到GPU
        // CHECK(cudaMemcpy(d_r, r, size[0], cudaMemcpyHostToDevice));
        // CHECK(cudaMemcpy(d_pos_x, pos_x, size[0], cudaMemcpyHostToDevice));
        // CHECK(cudaMemcpy(d_pos_y, pos_y, size[0], cudaMemcpyHostToDevice));
        // CHECK(cudaMemcpy(d_pos_z, pos_z, size[0], cudaMemcpyHostToDevice));
        // CHECK(cudaMemcpy(d_intPart, intPart, size[0], cudaMemcpyHostToDevice));
        // CHECK(cudaMemcpy(d_fractionPart, fractionPart, size[0], cudaMemcpyHostToDevice));
        // CHECK(cudaMemcpy(d_start_time, start_time, size[0], cudaMemcpyHostToDevice));
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
        int threadPerBlock= 1024;
        int blocksPerGrid = 18;
        dim3 block(threadPerBlock);
        //设置块数量
        dim3 grid(blocksPerGrid);//blocksPerGrid
        printf("[GPU]网格，线程(%d,%d)\n",blocksPerGrid,threadPerBlock);
        //调用核函数
        for(int i = 0;i<size[0]/8;i++) {
            CHECK(cudaDeviceSynchronize());
            printf("[GPU]核函数开始运行[%d]\n",i);
            pmt_calculate<<<grid, block>>>(r[i],pos_x[i],pos_y[i],pos_z[i],d_pmt_x,d_pmt_y,d_pmt_z,intPart[i],fractionPart[i],start_time[i],17746,d_data_hit,d_data_npe,(int*)(d_seed+i*pmt_num),d_result,d_pmt_res_list,(int)size[0]/8);
            // step_calculate(r[i],pos_x[i],pos_y[i],pos_z[i],intPart[i],fractionPart[i],start_time[i],d_pmt_x,d_pmt_y,d_pmt_z,d_data_hit,d_data_npe,(int*)(d_seed+i*pmt_num),d_result,d_pmt_res_list,(int)size[0]/8);
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
        // CHECK(cudaFree(d_r));
        // CHECK(cudaFree(d_pos_x));
        // CHECK(cudaFree(d_pos_y));
        // CHECK(cudaFree(d_pos_z));
        // CHECK(cudaFree(d_intPart));
        // CHECK(cudaFree(d_fractionPart));
        CHECK(cudaFree(d_data_hit));
        CHECK(cudaFree(d_data_npe));
        // CHECK(cudaFree(d_start_time));
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