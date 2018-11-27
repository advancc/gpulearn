"""
@version: 1.1
@author: yipeihuai
@mail: yiph@ihep.ac.cn
@file: simu_so.py
@time: 2018/10/15 10:58
@modify time: 2018/11/27
"""

import numpy as np
import ctypes
import random
import sys
import logging
from decimal import Decimal
from ctypes import *


# extract cdf_sampling_wrapper function pointer in the shared object simu_so.so
def get_cdf_sampling_wrapper():
    dll = ctypes.CDLL('./simu_so.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.CDF_Sampling_wrapper
    func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_int), c_int, c_int, c_int, c_int]
    func.restype = c_float
    return func

def get_cdf_sampling_wrapper_C():
    dll = ctypes.CDLL('./simu_c.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.CDF_Sampling_wrapper
    func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int]
    func.restype = c_double
    return func   

# create __cuda_cdf_sampling function with get_cdf_sampling_wrapper()
__cuda_cdf_sampling   = get_cdf_sampling_wrapper()
__cuda_cdf_sampling_C = get_cdf_sampling_wrapper_C()
# convenient python wrapper for __cuda_cdf_sampling
# it does all job with types convertation
# from python ones to C++ ones
def cuda_cdf_sampling(a, b, c, seed, total_num, size, n, time):
    a_p = a.ctypes.data_as(POINTER(c_double))
    b_p = b.ctypes.data_as(POINTER(c_double))
    c_p = c.ctypes.data_as(POINTER(c_double))
    seed_p = seed.ctypes.data_as(POINTER(c_int))

    return __cuda_cdf_sampling(a_p, b_p, c_p, seed_p, total_num, size, n, time)

def cuda_cdf_sampling_C(a, b, c, total_num, n, time):
    a_p = a.ctypes.data_as(POINTER(c_double))
    b_p = b.ctypes.data_as(POINTER(c_double))
    c_p = c.ctypes.data_as(POINTER(c_double))

    return __cuda_cdf_sampling_C(a_p, b_p, c_p, total_num, n, time)

def create_npe_histo(max_n):
    # max_n 最大光子数量
    _range = Decimal('100')
    prob = []
    for item in range(max_n-1):
        _prob = Decimal(random.randrange(0,_range,1))
        prob.append(Decimal(_prob/100))
        _range = _range - _prob
    prob.append(1-sum(prob))
    histo = [str(i) for i in prob]
    return histo

def create_hittime_histo(max_time):
    # max_time 最大击中时间
    _range = Decimal('100')
    prob_list = []
    prob = []
    for item in range(max_time-1):
        _prob = Decimal(random.randrange(0,_range,1))
        prob.append(Decimal(_prob/100))
        _range = _range - _prob
    prob.append(1-sum(prob))
    histo = [str(i) for i in prob]
    return histo

def create_seed():
    _seed =random.randint(0,100000000)
    return _seed

def run(pmt_num,loc_num):
    #数据生成
    # pmt_num = 500
    # loc_num = 100
    max_n = 10
    max_time = 10
    total = str(pmt_num * loc_num)
    count = 1
    temp_list_npe = []
    temp_list_hit = []
    temp_list_seed = []
    for i in range(loc_num):
        for j in range(pmt_num):
            temp_list_npe.append([0,0,0,0,0.5,0,0.2,0,0.3,0])
            temp_list_hit.append([0,0.2,0.2,0.1,0,0,0,0.2,0.2,0.1])
            temp_list_seed.append(create_seed())
            # print("当前进度" + str(count) + "/" + total)
            count += 1
    temp_np_npe = np.array(temp_list_npe).astype('double')
    temp_np_hit = np.array(temp_list_hit).astype('double')
    temp_np_seed = np.array(temp_list_seed).astype('int')
    # print("数据生成任务已完成")

    #假定两种表内存一致大小，即：max_n= max_time
    # 8 :sizeof(double) = 8
    size=int(pmt_num * loc_num *max_n*8 )

    result = np.zeros(size/8).astype('double')

    gputime = cuda_cdf_sampling(temp_np_npe, temp_np_hit, result, temp_np_seed, pmt_num*loc_num, size, max_n, max_time)
    # print("GPU执行结束")
    # logging.info(gputime)
    # print("CPU开始执行")
    cputime = cuda_cdf_sampling_C(temp_np_npe, temp_np_hit, result, pmt_num*loc_num,max_n,max_time)
    # print("CPU执行结束")
    # logging.info(cputime)

    return gputime,cputime
    #强制打印numpy数组全部元素
    # np.set_printoptions(threshold=np.nan)
    # for index in range(0,len(result),max_n):
    #     logging.info(result[index:index+max_n])


if __name__ == '__main__':
    logging.basicConfig(filename='logger.log', filemode='w', level=logging.INFO)
    if len(sys.argv)>3:
        begin = int(sys.argv[1])
        end = int(sys.argv[2])
        step = int(sys.argv[3])
        gpu_time_list = []
        cpu_time_list = []

        for i in range(begin,end+step,step):
            print("total num:"+str(i))
            logging.info("total num:"+str(i))
            gpu_time_list = []
            cpu_time_list = []
            for j in range(10):
                gpu,cpu = run(int(i/10),10)
                gpu_time_list.append(gpu)
                cpu_time_list.append(cpu)
            print("gpu time:"+str(sum(gpu_time_list)/10.0))
            logging.info("gpu time:"+str(sum(gpu_time_list)/10.0))
            print("cpu time:"+str(sum(cpu_time_list)/10.0))
            logging.info("cpu time:"+str(sum(cpu_time_list)/10.0))
            # print("gpu time:"+str(gpu))
            # logging.info("gpu time:"+str(gpu))
            # print("cpu time:"+str(cpu))
            # logging.info("cpu time:"+str(cpu))
    elif len(sys.argv)>2:
        pmt_num = int(sys.argv[1])
        loc_num = int(sys.argv[2])
        run(pmt_num,loc_num)
    else:
        pmt_num = 10
        loc_num = 2
        run(pmt_num,loc_num)
    