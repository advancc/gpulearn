#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

int main(int argc,char** argv)
{
    //生成假数据
	int total_num =100;
	if (argc>1)
    {
        total_num = atoi(argv[1]);
    }
	int max_n = 10;
	int max_time = 10;
	size_t nBytes = total_num * max_n * sizeof(double);

	double *pmt;
	pmt = (double*)malloc(nBytes);
	if (pmt == NULL)
	{
		printf("CPU内存分配失败\n");
		exit(0);
	}
	for (int i = 0;i < total_num;i++)
	{
		for (int j = 0;j < max_n;j++)
		{
			pmt[i*max_n +j] = 0.1;
		}
	}
	double *hittime;
	hittime = (double*)malloc(nBytes);
	for (int i = 0;i < total_num;i++)
	{
		for (int j = 0;j < max_time;j++)
		{
			hittime[i*max_time +j] = 0.1;
		}
	}
	double *h_res = (double*)malloc(nBytes);
	//
	//
	void *handle = dlopen("./simu_so.so", RTLD_LAZY);
	if(!handle)
	{
		printf("open lib error\n");
		cout<<dlerror()<<endl;
		return -1;
	}
	typedef float (*cdf_sampling_t)(double *h_pmt, double *h_hit, double *h_result,int total_num,int nBytes,int max_n,int max_time);
	cdf_sampling_t CDF_Sampling_Wrapper = (cdf_sampling_t) dlsym(handle, "CDF_Sampling_wrapper");
	if(!CDF_Sampling_Wrapper)
	{
		cout<<dlerror()<<endl;
		dlclose(handle);
		return -1;
	}

	float time_use = CDF_Sampling_Wrapper(pmt,hittime,h_res,total_num,nBytes,max_n,max_time);
	for (int i = 0;i < total_num;i++)
	{
		for (int j = 0;j < max_time;j++)
		{
			printf("%f ", h_res[i*max_time + j]);
		}
		printf("\n");
	}
	printf("total time = %f\n",time_use);

	dlclose(handle);
	return 0;
}
