#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "sample.cuh"
#include "vector.cuh"
#include "simu_muon.cuh"

extern "C"
{
    float GPU_Sampling_wrapper(double *r,double *pos_x,double *pos_y,double *pos_z, double *intPart, double *fractionPart,double *start_time,double *pmt_x,double *pmt_y,double *pmt_z,double *data_hit,double *data_npe,int *seed,int *size,double* h_result)
    {
        //GPU计时，设置开始和结束事件
        cudaEvent_t start, stop;
        cudaEvent_t gpu_start,gpu_stop,data_start,data_stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        cudaEventCreate(&gpu_start);
        cudaEventCreate(&gpu_stop);
        cudaEventCreate(&data_start);
        cudaEventCreate(&data_stop);
        cudaEventRecord(start);
        cudaEventRecord(data_start);
        //申请GPU内存
        // double *d_r, *d_pos_x,*d_pos_y,*d_pos_z,*d_intPart,*d_fractionPart,*d_start_time;
        double *d_pmt_x,*d_pmt_y,*d_pmt_z,*d_data_hit,*d_data_npe;
        double *d_result;
        int *d_seed,*d_pmt_res_list;

        CHECK(cudaMalloc((double**)&d_pmt_x,size[1]));
        CHECK(cudaMalloc((double**)&d_pmt_y,size[1]));
        CHECK(cudaMalloc((double**)&d_pmt_z,size[1]));
        CHECK(cudaMalloc((double**)&d_data_hit,size[2]));
        CHECK(cudaMalloc((double**)&d_data_npe,size[3]));
        CHECK(cudaMalloc((int**)&d_seed,size[4]));
        CHECK(cudaMalloc((double**)&d_result,pmt_num*pmt_mem*8));
        CHECK(cudaMalloc((int**)&d_pmt_res_list,pmt_num*sizeof(int)));

        //设置内存
        CHECK(cudaMemset(d_pmt_res_list,0,pmt_num*sizeof(int)));
        CHECK(cudaMemset(d_result,0,pmt_num*pmt_mem*8));
        //将CPU内存拷贝到GPU
        CHECK(cudaMemcpy(d_pmt_x, pmt_x, size[1], cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_pmt_y, pmt_y, size[1], cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_pmt_z, pmt_z, size[1], cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_data_hit, data_hit, size[2], cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_data_npe, data_npe, size[3], cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_seed, seed, size[4], cudaMemcpyHostToDevice));
        
        cudaEventRecord(data_stop);
        cudaEventSynchronize(data_stop);
        // printf("[GPU]GPU数据拷贝完成\n");
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
        // printf("[GPU]网格，线程(%d,%d)\n",blocksPerGrid,threadPerBlock);
        //调用核函数
        cudaEventRecord(gpu_start);
        for(int i = 0;i<size[0]/8;i++) {
            CHECK(cudaDeviceSynchronize());
            // printf("[GPU]核函数开始运行[%d]\n",i);
            pmt_calculate<<<grid, block>>>(r[i],pos_x[i],pos_y[i],pos_z[i],d_pmt_x,d_pmt_y,d_pmt_z,intPart[i],fractionPart[i],start_time[i],17746,d_data_hit,d_data_npe,(int*)(d_seed+i*pmt_num),d_result,d_pmt_res_list,(int)size[0]/8);
        }
        cudaEventRecord(gpu_stop);
        cudaEventSynchronize(gpu_stop);
        
        // printf("[GPU]核函数运行完成\n");
        // CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_result, d_result, pmt_num*pmt_mem*8, cudaMemcpyDeviceToHost));
        
        // printf("threadPerBlock:%d\n",threadPerBlock);
        // printf("blocksPerGrid；%d\n",blocksPerGrid);
        

        //释放GPU内存
        CHECK(cudaFree(d_data_hit));
        CHECK(cudaFree(d_data_npe));
        CHECK(cudaFree(d_pmt_x));
        CHECK(cudaFree(d_pmt_y));
        CHECK(cudaFree(d_pmt_z));
        CHECK(cudaFree(d_seed));
        CHECK(cudaFree(d_result));
        // printf("[GPU]GPU运行完成\n");

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float total_time,gputime,datatime;
        //计算用时，精度0.5us
        cudaEventElapsedTime(&datatime, data_start, data_stop);
        cudaEventElapsedTime(&gputime, gpu_start, gpu_stop);
        cudaEventElapsedTime(&total_time, start, stop);
        cudaEventDestroy(gpu_start);
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(data_start);
        cudaEventDestroy(data_stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);        
        printf("total use time %f ms\n", total_time);
        printf("gpu use time %f ms\n",gputime);
        printf("data use time %f ms\n",datatime);
        printf("data transport back use time %f ms\n",total_time - datatime - gputime);
        CHECK(cudaDeviceReset());        
        return total_time;
        // return 0.0;
    }
}