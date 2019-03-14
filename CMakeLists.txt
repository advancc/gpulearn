# CMakeLists.txt for G4CU project

project(muon_simu)

# required cmake version
cmake_minimum_required(VERSION 2.8)


# packages
find_package(CUDA REQUIRED)

# header file path 
include_directories ("${PROJECT_SOURCE_DIR}")

# library file path
LINK_DIRECTORIES(/usr/lib /usr/local/lib)

# nvcc flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CUDA_NVCC_FLAGS -O3;-G;-g)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35;)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=sm_70;)
#set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
#set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g)

#file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
#file(GLOB CURRENT_SOURCES  *.cpp *.cu)

#source_group("Include" FILES ${CURRENT_HEADERS}) 
#source_group("Source" FILES ${CURRENT_SOURCES}) 

#cuda_add_library(gpu SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
#cuda_add_library(gpu STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

#Output directory in which to build RUNTIME target files.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./)

CUDA_ADD_EXECUTABLE(muon_simu.out
    HelpMuonSimu.cpp
    MuonSimu.cu
)
