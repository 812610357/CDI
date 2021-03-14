import cupy as cp
import numpy as np

import fileioput as fio

data0 = cp.array(fio.readimage("./data/Lenna_gray_180d.png"),dtype='int64')
data1 = cp.array(fio.readimage("./data/Lenna_test_RL_0.5_re.png"),dtype='int64')
data2 = cp.array(fio.readimage("./data/Lenna_test_1000_180d.png"),dtype='int64')

print(cp.sqrt(cp.mean((data0-data1)**2)))
print(cp.sqrt(cp.mean((data0-data2)**2)))