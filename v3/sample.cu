#include "sample.cuh"

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