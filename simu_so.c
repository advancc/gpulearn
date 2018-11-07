#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

void CDF_Sampling(double *pmt, double *hittime, double *result, int numElements, int max_n, int max_time);
double generateRandom();
void generateRandomInit(int i);



double generateRandom()
{
    return (double)rand()/RAND_MAX;
}
void generateRandomInit(int i)
{
	srand(i);
}
void CDF_Sampling(double *pmt, double *hittime, double *result, int numElements, int max_n, int max_time)
{

    for(int i=0;i < numElements;i++)
    {
        generateRandomInit(i);
		double prob = generateRandom();
		double sum = 0;
		int n = 0;
		for (int item = 0; item < max_n;item++)
		{
			sum += pmt[i*max_n+item];
			if (prob <= sum)
			{
				n = item;
				// printf("circle %d:hit times:%d\n",i, n);
				break;
			}
		}
		for (int item = 0;item < n;item++)
		{
			double prob2 = generateRandom();
			double sum = 0;
			for (int j = 0; j < max_time;j++)
			{
				sum += hittime[i*max_time+j];
				if (prob2 <= sum)
				{
					result[i*max_n+item] = (double)j;
					// printf("circle %d: %dth hit time %d\n", i, item+1,j);
					break;
				}
			}

		}
    }
}

extern double CDF_Sampling_wrapper(double *pmt,double *hittime,double *h_result, int total_num, int max_n,int max_time)
{
    struct timeval t1,t2;
    double timeuse;
    gettimeofday(&t1,NULL);
    //gcc -std=c99 simu_so.c -fPIC -shared -o simu_c.so
    //gcc test9.c -lrt运行
  
    CDF_Sampling(pmt, hittime, h_result, total_num, max_n, max_time);
  
    
    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec)*1000 + (t2.tv_usec - t1.tv_usec)/1000.0;
    // printf("Use Time: %f ms\n",timeuse);
    return timeuse;
}
