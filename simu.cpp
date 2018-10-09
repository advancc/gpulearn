#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <windows.h>
using namespace std;

void CDF_Sampling(double *pmt, double *hittime, double *result, int numElements);
double generateRandom();
void generateRandomInit();

int main()
{
	int total_num = 10;
	int max_n = 10;
	int max_time = 10;
	size_t nBytes = total_num * max_n * sizeof(double);
	double *pmt;
	pmt = (double*)malloc(nBytes);
	for (int i = 0;i < total_num;i++)
	{
		for (int j = 0;j < max_n;j++)
		{
			pmt[i*total_num +j] = 0.1;
		}
	}
	double *hittime;
	hittime = (double*)malloc(nBytes);
	for (int i = 0;i < total_num;i++)
	{
		for (int j = 0;j < max_time;j++)
		{
			hittime[i*total_num+j] = 0.1;
		}
	}
	double *h_res = (double*)malloc(nBytes);

	LARGE_INTEGER t1,t2,tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	CDF_Sampling(pmt, hittime, h_res, total_num);

	QueryPerformanceCounter(&t2);
	printf("CPU Use Time:%f us\n",1000000*(t2.QuadPart - t1.QuadPart)*1.0/tc.QuadPart);

	for (int i = 0;i < total_num;i++)
	{
		for (int j = 0;j < max_time;j++)
		{
			printf("%f ",h_res[i*10+j]);
		}
		printf("\n");
	}

	free(pmt);
	free(hittime);
	free(h_res);


	return 0;
}
double generateRandom()
{
	return (double)rand()/RAND_MAX;
}
void generateRandomInit()
{
	srand((unsigned int)time(NULL));
}
void CDF_Sampling(double *pmt, double *hittime, double *result, int numElements)
{
	generateRandomInit();
	for(int i=0;i < numElements;i++)
	{
		double prob = generateRandom();
		double sum = 0;
		int n = 0;
		for (int item = 0; item < 10;item++)
		{
			sum += pmt[i*10+item];
			if (prob <= sum)
			{
				n = item;
				printf("circle %d:hit times:%d\n",i, n);
				break;
			}
		}
		for (int item = 0;item < n;item++)
		{
			double prob2 = generateRandom();
			double sum = 0;
			for (int j = 0; j < 10;j++)
			{
				sum += hittime[i*10+j];
				if (prob2 <= sum)
				{
					result[i*10+item] = (double)j;
					printf("circle %d: %dth hit time %d\n", i, item+1,j);
					break;
				}
			}

		}
	}
}
