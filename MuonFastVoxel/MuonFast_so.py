import numpy as np
import ctypes
import random
import sys
import logging
from decimal import Decimal
from ctypes import *
# import matplotlib.pyplot as plt

class MuonFast:
    def __init__(self,path):
        self.data_step  = np.load(path+"0_data.npy")
        self.data_pmt_x = np.load(path+"pmt_x.npy")
        self.data_pmt_y = np.load(path+"pmt_y.npy")
        self.data_pmt_z = np.load(path+"pmt_z.npy")
        self.data_npe   = np.load(path+"npe_cdf.npy")
        self.data_hit   = np.load(path+"hittime_cdf.npy")
        self.seed       = self.create_seed(len(self.data_step[0])*17746)

    def create_seed(self,number):
        _seed = []
        for i in range(number):
            # _seed.append(random.randint(0,100000000))
            _seed.append(i)
        seed = np.array(_seed).astype('int') 
        return seed

    def cuda_wrapper(self):
        def get_cdf_sampling_wrapper():
            dll = ctypes.CDLL('./simu_so.so', mode=ctypes.RTLD_GLOBAL)
            func = dll.GPU_Sampling_wrapper
            func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double),POINTER(c_double),POINTER(c_double),POINTER(c_double),POINTER(c_double),POINTER(c_double),POINTER(c_double),POINTER(c_double), POINTER(c_double),POINTER(c_double), POINTER(c_int),POINTER(c_int),POINTER(c_double)]
            func.restype = c_float
            return func

        __cuda_cdf_sampling   = get_cdf_sampling_wrapper()

        def cuda_cdf_sampling(a, b, c, d, e, f, g, h, j, i, k, l, seed, size,result):
            a_p = a.ctypes.data_as(POINTER(c_double))
            b_p = b.ctypes.data_as(POINTER(c_double))
            c_p = c.ctypes.data_as(POINTER(c_double))
            d_p = d.ctypes.data_as(POINTER(c_double))
            e_p = e.ctypes.data_as(POINTER(c_double))
            f_p = f.ctypes.data_as(POINTER(c_double))
            g_p = g.ctypes.data_as(POINTER(c_double))
            h_p = h.ctypes.data_as(POINTER(c_double))
            j_p = j.ctypes.data_as(POINTER(c_double))
            i_p = i.ctypes.data_as(POINTER(c_double))
            k_p = k.ctypes.data_as(POINTER(c_double))
            l_p = l.ctypes.data_as(POINTER(c_double))
            seed_p = seed.ctypes.data_as(POINTER(c_int))
            size_p = size.ctypes.data_as(POINTER(c_int))
            res_p = result.ctypes.data_as(POINTER(c_double))
            return __cuda_cdf_sampling(a_p, b_p, c_p, d_p, e_p, f_p, g_p, h_p, j_p, i_p, k_p, l_p, seed_p, size_p,res_p)

        _size  = [len(self.data_step[0])*8,len(self.data_pmt_x)*8,len(self.data_hit)*8,len(self.data_npe)*8,len(self.seed)*4]
        size   = np.array(_size).astype('int')
        result = np.zeros(17746*len(self.data_step[0])*50).astype('double')
        print("GPU开始执行")
        #float GPU_Sampling_wrapper(double *r,double *pos_x,double *pos_y,double *pos_z, double *intPart, double *fractionPart,double *start_time,double *pmt_x,double *pmt_y,double *pmt_z,double *data_hit,double *data_npe,int *seed,int *size,double* h_result)
        cuda_cdf_sampling(self.data_step[0],self.data_step[1],self.data_step[2],self.data_step[3],self.data_step[4],self.data_step[5],self.data_step[6],self.data_pmt_x, self.data_pmt_y, self.data_pmt_z, self.data_hit, self.data_npe, self.seed, size,result)
        print("GPU执行结束，数据保存中")
        np.save("./result.npy",result)
        print("全部完成！")
    # def check(self):
    #     plt.figure(1)
    #     ax1 = plt.subplot(7,1,1)
    #     ax2 = plt.subplot(7,1,2)
    #     ax3 = plt.subplot(7,1,3)
    #     ax4 = plt.subplot(7,1,4)
    #     ax5 = plt.subplot(7,1,5)
    #     ax6 = plt.subplot(7,1,6)
    #     ax7 = plt.subplot(7,1,7)
    #     plt.sca(ax1)
    #     plt.bar(range(0,len(self.data_step[0])),self.data_step[0])
    #     plt.title("r")
    #     plt.sca(ax2)
    #     plt.bar(range(0,len(self.data_step[1])),self.data_step[1])
    #     plt.title("pos_x")
    #     plt.sca(ax3)
    #     plt.bar(range(0,len(self.data_step[2])),self.data_step[2])
    #     plt.title("pos_y")
    #     plt.sca(ax4)
    #     plt.bar(range(0,len(self.data_step[3])),self.data_step[3])
    #     plt.title("pos_z")
    #     plt.sca(ax5)
    #     plt.bar(range(0,len(self.data_step[4])),self.data_step[4])
    #     plt.title("intPart")
    #     plt.sca(ax6)
    #     plt.bar(range(0,len(self.data_step[5])),self.data_step[5])
    #     plt.title("fractionPart")
    #     plt.sca(ax7)
    #     plt.bar(range(0,len(self.data_step[6])),self.data_step[6])
    #     plt.title("startTime")
    #     plt.show()

    def run(self):
        self.cuda_wrapper()

def run():
    print("开始加载")
    muon = MuonFast("./data/")
    print("数据加载完成")
    muon.run()
    # muon.check()
    
if __name__ == '__main__':
    # logging.basicConfig(filename='logger.log', filemode='w', level=logging.INFO)
    run()