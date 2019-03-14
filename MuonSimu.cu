#include "MuonSimu.cuh"

__global__ void
evt_calculate_add(int *evt_res_list,int *evt_res_back,int evtnum,int pmtnum)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;   
    // int evt_res_by_pmt = 0;
    if (id < evtnum) // The number of threads can't exceed the number of event
    {
        for(int i=0; i<pmtnum; i++) // i represent pmtid , id represent evtid
        {
            evt_res_back[id] += evt_res_list[id+i*evtnum];
           
        }
        printf("thread = %d,evt_res_back=%d\n",id,evt_res_back[id]);
        // evt_res_back[id] = evt_res_by_pmt;//return the result 

    }
}

__global__ void
step_calculate_every_pmt(double *r,double *pos_x,double *pos_y,double *pos_z,double *pmt_x,\
    double *pmt_y,double *pmt_z,double *intPart,double *fractionPart,double *start_time, int *evt_id,\
    int numElements,int evtnum,double *hittime_histo,double *npe,int *seed,double *result, \
    int *pmt_res_list,int *evt_res_list,int size)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    //numElements = pmt numbers
    if (id < numElements){
        curandState state;
        generateRandomInit(&state,seed[id]);
        // generateRandomInit(&state,seed);
        Res_Arr pmt_arr;
        init_res_arr(&pmt_arr,result,pmt_res_list,evt_res_list,id,size,evtnum);
        

        for(int i=0;i<size;i++)
        {
            // int index = i*numElements+id;
            calculate_by_step(r[i],pos_x[i],pos_y[i],pos_z[i],pmt_x[id],pmt_y[id],\
                pmt_z[id],intPart[i],fractionPart[i],start_time[i],evt_id[i],hittime_histo,\
                npe,&state,&pmt_arr,evt_res_list);
            // (int*)(evt_res_list+id*evtnum)
        }
        
    }
}

__device__ void
calculate_by_step(double r,double pos_x,double pos_y,double pos_z,double pmt_x,double pmt_y,\
    double pmt_z,double intPart,double fractionPart,double start_time,int evt_id,double *hittime_histo,\
    double *npe,curandState *state,Res_Arr *p_pmt_arr,int *evt_res_list)
{
    double theta = calculateAngle(pmt_x,pmt_y,pmt_z,pos_x,pos_y,pos_z);
    for(int j = 0;j<intPart; ++j){
        //r(m)
        generateHits(r,theta,1,start_time,evt_id,hittime_histo,npe,state,p_pmt_arr,evt_res_list);
    }
    generateHits(r,theta,fractionPart,start_time,evt_id,hittime_histo,npe,state,p_pmt_arr,evt_res_list);
}

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
generateHits(double r,double theta, double ratio,double start_time,int evt_id,double *hittime_histo,\
    double *npe,curandState *state,Res_Arr *p_r_arr,int *evt_res_list)
{
    
    int npe_histo_id = get_npe(r,theta,npe,state);
    if (npe_histo_id>0)
    {
        for (int hitj = 0; hitj < npe_histo_id; ++hitj) 
        {
            // skip the photon according to the energy deposit
            if (ratio<1 and generateRandom(state)>ratio)
            {
                continue;
            }
            double hittime_single = start_time;
            // (m_flag_time) 
            hittime_single += (double)get_hittime(r, theta, 0, hittime_histo, state);
            // printf("hittime = %lf\n",hittime_single);
            // generated hit
            // (m_flag_savehits) 
            append_res_arr(p_r_arr,hittime_single);
            add_evt_arr(p_r_arr,evt_id);
            // save_hits(pmtid, hittime_single,result);
        }
    }
}

__device__ int
get_hittime(double r, double theta, int mode, double *hittime_histo, curandState *state) {
    int binx = get_bin_x(r);
    int biny = get_bin_y(theta);
    
    return get_hittime_bin(binx, biny, mode, hittime_histo, state);
}

__device__ int 
get_hittime_bin(int binx, int biny, int mode, double *hittime_histo, curandState *state) {
    // hit time = tmean + tres
    int hittime_single = 0;
    if (mode == 0) {
        hittime_single = get_hittime_all(binx,biny,hittime_histo,state);
    }
    return hittime_single;
}

__device__ int 
get_hittime_all(int binx, int biny,double *hittime_histo, curandState *state) {
    // TH1F* h = get_hist(binx, biny);
    const int xbinnum = 200;
    const int ybinnum = 180;
    if (binx<1) { binx = 1; }
    else if (binx > xbinnum) { binx = xbinnum;}
    if (biny<1) { biny = 1; }
    else if (biny > ybinnum) { biny = ybinnum;}
    int idx = (binx-1)*ybinnum+(biny-1);
    int hittime_single = sampling(state,hittime_histo,3000,idx);
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
    // printf("[npe] r=%lf,theta=%lf,binx=%d,biny=%d\n",r,theta,binx,biny);
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
    if(theta == 0){
        
        return 1;
    }
    else{
        return (int)ceil((theta-begin)/(end-begin)*binnum);
    }
}

__device__ int 
get_npe_num(int binx,int biny,double *npe,curandState *state)
{
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
    // int id = blockIdx.x*blockDim.x+threadIdx.x;   
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
    curand_init(seed, id, 0, state);
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
    if (target <= histo[id*max+start]){
        return start;
    }
    else if (histo[id*max+start] < target){
        return end;
    }
    
    return -1;
}

__device__ void
append_res_arr(Res_Arr *p, double val)//追加，可能成功，可能失败
{
    p->arr[p->index+p->pmt_list[p->id]] = val;
    p->pmt_list[p->id] += 1;
    if(p->pmt_list[p->id]>p->max){
        printf("error! out of memory for result!\n");
        assert(p->pmt_list[p->id]<=p->max);
    }

    return;

}

__device__ void 
add_evt_arr(Res_Arr *p,int evtid)
{
    p->evt_res[evtid+p->id*p->evt_num] +=1;
    return;
}

__device__ void
init_res_arr(Res_Arr *p,double *result,int *pmt_res_list,int *evt_res_list,int pmtid,int size,int evt_num){
    p->arr = result;//存储的内存空间
    p->evt_res = evt_res_list;//totalPE by event
    p->pmt_list = pmt_res_list;//存储每个pmt内存空间使用量
    p->index = pmtid*pmt_mem;//存储该pmt在数组中的起始存取点
    p->id = pmtid;
    p->max = pmt_mem;
    p->evt_num = evt_num;
    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    // Catch errors from CUDA kernel calls
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    // err = cudaDeviceSynchronize();
    // if( cudaSuccess != err )
    // {
    //     fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
    //              file, line, cudaGetErrorString( err ) );
    //     exit( -1 );
    // }
#endif

    return;
}

float GPU_Sampling_wrapper(double *r,double *pos_x,double *pos_y,double *pos_z, \
    double *intPart, double *fractionPart,double *start_time,int *evt_id,int evtnum,double *pmt_x,double *pmt_y,\
    double *pmt_z,double *data_hit,double *data_npe,int *seed,int *size,double* h_result,int *h_evt_res_list)
{
    //debug
    // size_t psize = 0;
    // cudaDeviceGetLimit(&psize,cudaLimitPrintfFifoSize);
    
    //GPU计时，设置开始和结束事件
    cudaEvent_t start, stop;
    cudaEvent_t gpu_start,gpu_stop,data_start,data_stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventCreate(&gpu_start));
    CHECK(cudaEventCreate(&gpu_stop));
    CHECK(cudaEventCreate(&data_start));
    CHECK(cudaEventCreate(&data_stop));
    CHECK(cudaEventRecord(start));
    CHECK(cudaEventRecord(data_start));
    //申请GPU内存
    double *d_r, *d_pos_x,*d_pos_y,*d_pos_z,*d_intPart,*d_fractionPart,*d_start_time;
    double *d_pmt_x,*d_pmt_y,*d_pmt_z,*d_data_hit,*d_data_npe;
    double *d_result;
    int *d_seed,*d_pmt_res_list,*d_evt_id,*d_evt_res_list,*d_evt_res_back;

    int *h_pmt_res_list = (int*)malloc(pmt_num*sizeof(int));
    CHECK(cudaMalloc((double**)&d_r,size[0]));
    CHECK(cudaMalloc((double**)&d_pos_x,size[0]));
    CHECK(cudaMalloc((double**)&d_pos_y,size[0]));
    CHECK(cudaMalloc((double**)&d_pos_z,size[0]));
    CHECK(cudaMalloc((double**)&d_intPart,size[0]));
    CHECK(cudaMalloc((double**)&d_fractionPart,size[0]));
    CHECK(cudaMalloc((double**)&d_start_time,size[0]));
    CHECK(cudaMalloc((int**)&d_evt_id,size[0]/2));
    CHECK(cudaMalloc((double**)&d_pmt_x,size[1]));
    CHECK(cudaMalloc((double**)&d_pmt_y,size[1]));
    CHECK(cudaMalloc((double**)&d_pmt_z,size[1]));
    CHECK(cudaMalloc((double**)&d_data_hit,size[2]));
    CHECK(cudaMalloc((double**)&d_data_npe,size[3]));
    CHECK(cudaMalloc((int**)&d_seed,size[4]));
    CHECK(cudaMalloc((double**)&d_result,pmt_num*pmt_mem*sizeof(double)));
    CHECK(cudaMalloc((int**)&d_pmt_res_list,pmt_num*sizeof(int)));
    CHECK(cudaMalloc((int**)&d_evt_res_list,evtnum*pmt_num*sizeof(int)));
    CHECK(cudaMalloc((int**)&d_evt_res_back,evtnum*sizeof(int)));

    //设置内存
    CHECK(cudaMemset(d_pmt_res_list,0,pmt_num*sizeof(int)));
    CHECK(cudaMemset(d_evt_res_list,0,evtnum*pmt_num*sizeof(int)));
    CHECK(cudaMemset(d_evt_res_back,0,evtnum*sizeof(int)));
    CHECK(cudaMemset(d_result,0,pmt_num*pmt_mem*sizeof(double)));
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
    CHECK(cudaMemcpy(d_evt_id, evt_id, size[0]/2, cudaMemcpyHostToDevice));

    CHECK(cudaEventRecord(data_stop));
    CHECK(cudaEventSynchronize(data_stop));
    // printf("[GPU]GPU数据拷贝完成\n");
    //设置使用编号为0的GPU
    CHECK(cudaSetDevice(0));
    // //设置线程数量
    int threadPerBlock= 256;
    int blocksPerGrid = 70;
    dim3 block(threadPerBlock);
    //设置块数量
    dim3 grid(blocksPerGrid);//blocksPerGrid
    //调用核函数
    printf("[GPU]核函数开始运行\n");
    CHECK(cudaEventRecord(gpu_start));
   
    step_calculate_every_pmt<<<grid, block>>>(d_r,d_pos_x,d_pos_y,d_pos_z,d_pmt_x,d_pmt_y,\
        d_pmt_z,d_intPart,d_fractionPart,d_start_time,d_evt_id,pmt_num,evtnum,d_data_hit,d_data_npe,d_seed,\
        d_result,d_pmt_res_list,d_evt_res_list,(int)(size[0]/sizeof(double)));
    CHECK(cudaDeviceSynchronize());
    dim3 grid_reduce(evtnum/threadPerBlock+1);
    evt_calculate_add<<<grid_reduce,block>>>(d_evt_res_list,d_evt_res_back,evtnum,pmt_num);
    CHECK(cudaDeviceSynchronize());
    CudaCheckError();
    CHECK(cudaEventRecord(gpu_stop));
    CHECK(cudaEventSynchronize(gpu_stop));
    
    printf("[GPU]核函数运行完成\n");
    // CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_result, d_result, pmt_num*pmt_mem*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_pmt_res_list, d_pmt_res_list, pmt_num*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_evt_res_list, d_evt_res_back, evtnum*sizeof(int), cudaMemcpyDeviceToHost));
    

    //释放GPU内存
    CHECK(cudaFree(d_r));
    CHECK(cudaFree(d_pos_x));
    CHECK(cudaFree(d_pos_y));
    CHECK(cudaFree(d_pos_z));
    CHECK(cudaFree(d_intPart));
    CHECK(cudaFree(d_fractionPart));
    CHECK(cudaFree(d_start_time));
    CHECK(cudaFree(d_data_hit));
    CHECK(cudaFree(d_data_npe));
    CHECK(cudaFree(d_pmt_x));
    CHECK(cudaFree(d_pmt_y));
    CHECK(cudaFree(d_pmt_z));
    CHECK(cudaFree(d_seed));
    CHECK(cudaFree(d_result));
    CHECK(cudaFree(d_pmt_res_list));
    CHECK(cudaFree(d_evt_res_list));
    // printf("[GPU]GPU运行完成\n");

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float total_time,gputime,datatime;
    //计算用时，精度0.5us
    CHECK(cudaEventElapsedTime(&datatime, data_start, data_stop));
    CHECK(cudaEventElapsedTime(&gputime, gpu_start, gpu_stop));
    CHECK(cudaEventElapsedTime(&total_time, start, stop));
    CHECK(cudaEventDestroy(gpu_start));
    CHECK(cudaEventDestroy(gpu_stop));
    CHECK(cudaEventDestroy(data_start));
    CHECK(cudaEventDestroy(data_stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));        
    printf("total use time %f ms\n", total_time);
    printf("gpu use time %f ms\n",gputime);
    printf("data use time %f ms\n",datatime);
    printf("data transport back use time %f ms\n",total_time - datatime - gputime);
    CHECK(cudaDeviceReset());        
    return total_time;
    // return 0.0;
}