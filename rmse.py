import cupy as cp
import numpy as np

import fileioput as fio

data0 = cp.array(fio.readimage("./data/Lenna_gray_180d.png"), dtype='int64')
data1 = cp.array(fio.readimage(
    "./data/OSS/Lenna_test_OSS_180d.png"), dtype='int64')
data2 = cp.array(fio.readimage(
    "./data/OSS/Lenna_test_OSS_0.5_re_0.5.png"), dtype='int64')

rmse=data0+127-data2
#fio.writeimage(cp.asnumpy(rmse), "./data/OSS/rms_OSS.png")
print(cp.sqrt(cp.mean((data0-data1)**2)))
print(cp.sqrt(cp.mean((data0-data2)**2)))
