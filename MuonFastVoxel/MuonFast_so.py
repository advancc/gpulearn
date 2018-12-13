import numpy as np
import ctypes
import random
import sys
import logging
from decimal import Decimal
from ctypes import *

class MuonFast:
    def __init__(self,path):
        self.data_step  = np.load(path+"0_data.npy")
        self.date_pmt_x = np.load(path+"pmt_x.npy")
        self.date_pmt_y = np.load(path+"pmt_y.npy")
        self.date_pmt_z = np.load(path+"pmt_z.npy")
        self.data_npe   = np.load(path+"npe_cdf.npy")
        self.data_hit   = np.load(path+"hittime_cdf.npy")
    def create_seed(self):
        _seed =random.randint(0,100000000)
        return _seed
    def cuda_wrapper(self):
        def get_cdf_sampling_wrapper():
            dll = ctypes.CDLL('./simu_so.so', mode=ctypes.RTLD_GLOBAL)
            func = dll.CDF_Sampling_wrapper
            func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_int), c_int, c_int, c_int, c_int]
            func.restype = c_float
            return func
        __cuda_cdf_sampling   = get_cdf_sampling_wrapper()
        def cuda_cdf_sampling(a, b, c, seed, total_num, size, n, time):
            a_p = a.ctypes.data_as(POINTER(c_double))
            b_p = b.ctypes.data_as(POINTER(c_double))
            c_p = c.ctypes.data_as(POINTER(c_double))
            seed_p = seed.ctypes.data_as(POINTER(c_int))
            return __cuda_cdf_sampling(a_p, b_p, c_p, seed_p, total_num, size, n, time)
        cuda_cdf_sampling(self.data_hit,self.data_npe,date_pmt_z,date_pmt_y,date_pmt_x,data_step,?????)
    
    def run():
        self.cuda_wrapper()



def run():
    muon = MuonFast("./data/")
    muon.run()
    
if __name__ == '__main__':
    logging.basicConfig(filename='logger.log', filemode='w', level=logging.INFO)
    run()