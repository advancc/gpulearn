cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(name MFSAlg)
project(${name} LANGUAGES CXX CUDA)
message(STATUS "[info] This is project source dir :" ${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# ROOT
set(CMAKE_MODULE_PATH /afs/ihep.ac.cn/soft/juno/JUNO-ALL-SLC6/DEV/juno-dev-root6/ExternalLibs/ROOT.con.install/6.10.08/etc/cmake ${CMAKE_MODULE_PATH})
#set(CMAKE_MODULE_PATH /afs/ihep.ac.cn/soft/juno/JUNO-ALL-SLC7/Pre-Release/J18v2r1-Pre2/ExternalLibs/ROOT/6.12.06/cmake  ${CMAKE_MODULE_PATH})
find_package(ROOT MODULE)

if(ROOT_FOUND)
    include_directories(${ROOT_INCLUDE_DIR})
    link_directories(${ROOT_LIBRARY_DIR})
else(ROOT_FOUND)
    message(FATAL_ERROR "Could not find ROOT")
endif(ROOT_FOUND)
set(CMAKE_CXX_FLAGS "${ROOT_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
link_libraries(${ROOT_LIBRARIES})

## Python
include_directories($ENV{JUNO_EXTLIB_Python_HOME}/include/python2.7)
find_library(PYTHON_LIBRARIES python
    HINTS $ENV{JUNO_EXTLIB_Python_HOME}/lib
)
link_directories($ENV{JUNO_EXTLIB_Python_HOME}/lib)
    
## Boost
include_directories($ENV{JUNO_EXTLIB_Boost_HOME})
find_library(BOOST_LIBRARIES boost_python
    HINTS $ENV{JUNO_EXTLIB_Boost_HOME}/lib
)
link_directories($ENV{JUNO_EXTLIB_Boost_HOME}/lib)

## SNiPER
include_directories(
    $ENV{SNIPERKERNELROOT}
    $ENV{ROOTWRITERROOT}
    $ENV{DATABUFFERROOT}
)
find_library(SNIPERKERNEL_LIBRARIES SniperKernel SniperPython
    HINTS $ENV{SNIPERKERNELROOT}/$ENV{CMTCONFIG}
)
link_directories($ENV{SNIPERKERNELROOT}/$ENV{CMTCONFIG})
find_library(ROOTWRITER_LIBRARIES RootWriter
    HINTS $ENV{ROOTWRITERROOT}/$ENV{CMTCONFIG}
)
link_directories($ENV{ROOTWRITERROOT}/$ENV{CMTCONFIG})

include_directories ("${PROJECT_SOURCE_DIR}/cuda")
#添加子目录
add_subdirectory(cuda)

#指定输出库和可执行文件路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/amd64_linux26)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/amd64_linux26)
#指定源文件路径
aux_source_directory(./src DIR_LIB_SRCS)

# 指定生成目标
add_library(${name} SHARED 
    ${DIR_LIB_SRCS} 
)
# 添加链接库
target_link_libraries(${name} GPU)