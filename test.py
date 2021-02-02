import time

import cupy as cp
import numpy as np

import fileioput as fio

data0 = fio.readimage("./data/Lenna.png")

data=data0
nptime=time.time()
for i in range(10000):
    data=np.fft.fft2(data)
    data=np.fft.ifft2(data)
print("CPU:",time.time()-nptime)

data=cp.asarray(data0)
cptime=time.time()
for i in range(10000):
    data=cp.fft.fft2(data)
    data=cp.fft.ifft2(data)
print("GPU:",time.time()-cptime)

