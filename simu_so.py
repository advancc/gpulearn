"""
@version: 1.0
@author: yipeihuai
@mail: yiph@ihep.ac.cn
@file: simu_so.py
@time: 2018/10/15 10:58
"""

import numpy as np
import ctypes
import random
from decimal import Decimal
from ctypes import *

# extract cdf_sampling_wrapper function pointer in the shared object simu_so.so
def get_cdf_sampling_wrapper():
    dll = ctypes.CDLL('./simu_so.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.CDF_Sampling_wrapper
    func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int, c_int]
    return func

# create __cuda_cdf_sampling function with get_cdf_sampling_wrapper()
__cuda_cdf_sampling = get_cdf_sampling_wrapper()

# convenient python wrapper for __cuda_cdf_sampling
# it does all job with types convertation
# from python ones to C++ ones
def cuda_cdf_sampling(a, b, c, total_num, size, n, time):
    a_p = a.ctypes.data_as(POINTER(c_double))
    b_p = b.ctypes.data_as(POINTER(c_double))
    c_p = c.ctypes.data_as(POINTER(c_double))

    __cuda_cdf_sampling(a_p, b_p, c_p, total_num, size, n, time)


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


if __name__ == '__main__':
    #数据生成
    pmt_num = 500
    loc_num = 100
    max_n = 10
    max_time = 10
    total = str(pmt_num * loc_num)
    count = 1
    temp_list_npe = []
    temp_list_hit = []
    for i in range(loc_num):
        for j in range(pmt_num):
            temp_list_npe.append(create_npe_histo(max_n))
            temp_list_hit.append(create_hittime_histo(max_time))

            print("当前进度" + str(count) + "/" + total)
            count += 1
    temp_np_npe = np.array(temp_list_npe).astype('double')
    temp_np_hit = np.array(temp_list_hit).astype('double')
    print("数据生成任务已完成")

    #假定两种表内存一致大小，即：max_n= max_time
    # 8 :sizeof(double) = 8
    size=int(pmt_num * loc_num *max_n*8 )

    result = np.zeros(size/8).astype('double')

    cuda_cdf_sampling(temp_np_npe, temp_np_hit, result, pmt_num*loc_num, size, max_n, max_time)
    print("GPU执行结束")
    #强制打印numpy数组全部元素
    # np.set_printoptions(threshold=np.nan)
    for index in range(0,len(result),max_n):
        print(result[index:index+max_n])