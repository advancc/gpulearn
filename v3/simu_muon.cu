#include "simu_muon.cuh"
__global__ void
pmt_calculate(double r,double pos_x,double pos_y,double pos_z,double *pmt_x,double *pmt_y,double *pmt_z,double intPart,double fractionPart,double start_time,int numElements,double *hittime_histo,double *npe,int *seed,double *result,int *pmt_res_list,int size){
    int id = blockIdx.x*blockDim.x+threadIdx.x;   
    //numElements = pmt numbers
    if (id < numElements){
        curandState state;
        generateRandomInit(&state, seed[id]);
        Res_Arr pmt_arr;
        init_res_arr(&pmt_arr,result,pmt_res_list,id,size);
        double theta = calculateAngle(pmt_x[id],pmt_y[id],pmt_z[id],pos_x,pos_y,pos_z);
        for(int j = 0; j < intPart; ++j){
            //r 单位 米
        	generateHits(r,theta,1,start_time,hittime_histo,npe,&state,pmt_arr);
        }
        generateHits(r,theta,fractionPart,start_time,hittime_histo,npe,&state,pmt_arr);
    }
}
__device__ double
calculateAngle(double x,double y,double z,double a,double b,double c)
{
    double result = 0;
    if (a == 0 and b == 0 and c == 0){
        return result;
    }
    else{
        result = acos((a*x+b*y+c*z)/(norm3d(x,y,z)*norm3d(a,b,c)));
        return result;
    }
}
__device__ void 
generateHits(double r,double theta, double ratio,double start_time,double *hittime_histo,double *npe,curandState *state,Res_Arr r_arr)
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
            printf("hittime = %lf\n",hittime_single);
            // save_hits
            append_res_arr(&r_arr,hittime_single);
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
