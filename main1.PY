import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import core
import fileioput as fio

oversamplingRatio = 3
data = cp.array(fio.readimage("./data/Lenna_gray.png"))
comp=cp.array(fio.readimage("./data/Lenna_gray_180d.png"))

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


plt.legend()
#plt.axis([0,100,0.325,0.345])
plt.xlabel('step')
plt.ylabel('RMSE')
plt.show()