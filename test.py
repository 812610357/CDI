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

plt.figure(figsize=[8,8])
plt.subplot(2,2,1)
fio.showimage(data,cmap='gray')
plt.text(0,64,'(a)',color='w',fontsize=18)

projection = cp.abs(core.FFTs(data))

plt.subplot(2,2,2)
fio.showimage(projection,cmap='jet',vmax=1e4)
plt.text(0,64,'(b)',color='w',fontsize=18)

data = cp.pad(data, ((padding[0], padding[1]),
                     (padding[0], padding[1])), 'constant')

plt.subplot(2,2,3)
fio.showimage(data,cmap='gray')
plt.text(0,200,'(c)',color='w',fontsize=18)

projection = cp.abs(core.FFTs(data))

plt.subplot(2,2,4)
fio.showimage(projection,cmap='jet',vmax=1e4)
plt.text(0,200,'(d)',color='w',fontsize=18)

plt.savefig('./OS.png')
