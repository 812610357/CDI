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

plt.figure(figsize=[12,6])
plt.subplot(2,4,3)
fio.showimage(data,cmap='gray')
plt.text(0,64,'(a)',color='w',fontsize=18)

projection = cp.abs(core.FFTs(data))

plt.subplot(2,4,4)
fio.showimage(projection,cmap='jet',vmax=1e4)
plt.text(0,64,'(b)',color='w',fontsize=18)

data = cp.pad(data, ((padding[0], padding[1]),
                     (padding[0], padding[1])), 'constant')

plt.subplot(2,4,7)
fio.showimage(data,cmap='gray')
plt.text(0,200,'(c)',color='w',fontsize=18)

projection = cp.abs(core.FFTs(data))

plt.subplot(2,4,8)
fio.showimage(projection,cmap='jet',vmax=1e4)
plt.text(0,200,'(d)',color='w',fontsize=18)

plt.subplot(2,1,2)
oversamplingRatio = 3
data = cp.array(fio.readimage("./data/Lenna_gray.png"))

for i in range(1):
    data=core.MultiGridRestriction(data)

padding = np.array(np.floor(np.array(data.shape) *
                            (oversamplingRatio-1)*0.5), dtype='int64')
data = cp.pad(data, ((padding[0], padding[1]),
                     (padding[0], padding[1])), 'constant')
projection = cp.abs(core.FFT(data))

fourlieSpace = projection
realSpace = core.iFFT(projection)
t=200
error=cp.zeros(t)

for i in range(t):
    realSpace,fourlieSpace = core.ER(realSpace, projection, padding)
    error[i]=np.linalg.norm(data-realSpace)/(padding[0]*3*255)
plt.plot(cp.asnumpy(error[:]),label='ER')


realSpace = core.iFFT(projection)
for i in range(t):
    realSpace,fourlieSpace = core.HIO(realSpace, projection, padding, 0.5)
    error[i]=np.linalg.norm(data-realSpace)/(padding[0]*3*255)
plt.plot(cp.asnumpy(error[:]),label='HIO')


realSpace = core.iFFT(projection)
for i in range(t):
    if np.mod(i,50)<45:
        realSpace,fourlieSpace = core.HIO(realSpace, projection, padding, 0.5)
    else:
        realSpace,fourlieSpace = core.ER(realSpace, projection, padding)
    error[i]=np.linalg.norm(data-realSpace)/(padding[0]*3*255)
plt.plot(cp.asnumpy(error[:]),'--',label='ER-HIO')
plt.text(0,0.014,'(d)',fontsize=18)

plt.legend()
plt.xlabel('step')
plt.ylabel('RMSE')

plt.savefig('./OS.png')
