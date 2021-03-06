import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import core
import fileioput as fio

oversamplingRatio = 3
data = cp.array(fio.readimage("./data/Lenna_gray.png"))
comp=cp.array(fio.readimage("./data/Lenna_gray_180d.png"))

for i in range(0):
    data=core.MultiGridRestriction(data)
    comp=core.MultiGridRestriction(comp)

padding = np.array(np.floor(np.array(data.shape) *
                            (oversamplingRatio-1)*0.5), dtype='int64')
data = cp.pad(data, ((padding[0], padding[1]),
                     (padding[0], padding[1])), 'constant')
comp = cp.pad(comp, ((padding[0], padding[1]),
                     (padding[0], padding[1])), 'constant')
projection = cp.abs(core.FFT(data))

#projection = core.H_RL(projection, 1, 0.5)
#fio.showimage(cp.fft.fftshift(projection))
fourlieSpace = projection
realSpace = core.iFFT(projection)
t=1000
error=cp.zeros(t)

for i in range(t):
    realSpace,fourlieSpace = core.ER(realSpace, projection, padding)
    error[i]=np.linalg.norm(comp-realSpace)/(padding[0]*3*255)
plt.plot(cp.asnumpy(error[:]),label='ER')
result = cp.asnumpy(
    cp.abs(realSpace[padding[0]:-padding[0], padding[1]:-padding[1]]))
fio.writeimage(result, "Lenna_test_ER.png")

realSpace = core.iFFT(projection)
for i in range(t):
    realSpace,fourlieSpace = core.HIO(realSpace, projection, padding, 0.5)
    error[i]=np.linalg.norm(comp-realSpace)/(padding[0]*3*255)
plt.plot(cp.asnumpy(error[:]),label='HIO')
result = cp.asnumpy(
    cp.abs(realSpace[padding[0]:-padding[0], padding[1]:-padding[1]]))
fio.writeimage(result, "Lenna_test_HIO.png")

realSpace = core.iFFT(projection)
for i in range(t):
    if np.mod(i,50)<45:
        realSpace,fourlieSpace = core.HIO(realSpace, projection, padding, 0.5)
    else:
        realSpace,fourlieSpace = core.ER(realSpace, projection, padding)
    error[i]=np.linalg.norm(comp-realSpace)/(padding[0]*3*255)
plt.plot(cp.asnumpy(error[:]),'--',label='ER-HIO')
result = cp.asnumpy(
    cp.abs(realSpace[padding[0]:-padding[0], padding[1]:-padding[1]]))
fio.writeimage(result, "Lenna_test_ER-HIO.png")

plt.legend()
plt.axis([0,t,0.33,0.345])
plt.xlabel('step')
plt.ylabel('RMSE')
plt.show()