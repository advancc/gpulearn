NVCC=nvcc
CUDAFLAGS= -arch=sm_35
OPT= -g -G -dc -Xcompiler -fPIC 
RM=rm -f
all: simu_gpu.so

api.o: simu_muon.o api.cu
	${NVCC} ${OPT} ${CUDAFLAGS} -c api.cu
sample.o : sample.cu sample.cuh
	${NVCC} ${OPT} ${CUDAFLAGS} -c sample.cu
vector.o : vector.cu vector.cuh
	${NVCC} ${OPT} ${CUDAFLAGS} -c vector.cu
simu_muon.o : sample.o vector.o simu_muon.cu simu_muon.cuh
	${NVCC} ${OPT} ${CUDAFLAGS} -c simu_muon.cu
simu_gpu.so: api.o simu_muon.o vector.o sample.o
	${NVCC} -Xcompiler -fPIC -shared ${CUDAFLAGS} -o simu_gpu.so *.o

clean:
	${RM} *.o *.so