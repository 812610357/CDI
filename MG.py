import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import core
import fileioput as fio

path = "./data/Lenna_test.png"
oversamplingRatio = 3
data = cp.array(fio.readimage("./data/Lenna.png"),dtype='float64')

padding = np.array(np.floor(np.array(data.shape) *
                            (oversamplingRatio-1)*0.5), dtype='int64')

data = cp.pad(data, ((padding[0], padding[1]),
                     (padding[0], padding[1])), 'constant')


projection = np.abs(core.FFT(data))

for i in range(4):
    data=core.MultiGridRestriction(data)

'''
for i in range(4):
    data=core.MultiGridProlongation(data)
'''

fio.showimage(data,cmap='gray')
plt.show()
data=1