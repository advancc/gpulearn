#查找当前目录下全部源文件并保存在变量DIR_LIB_SRCS中
aux_source_directory(. DIR_LIB_SRCS)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

#生成cuda动态库
add_library(GPU SHARED 
    ${DIR_LIB_SRCS} 
)