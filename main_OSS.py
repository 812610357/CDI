import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import core
import fileioput as fio

pa分别th = "./data/Lenna_test.png"
oversamplingRatio = 3
data = cp.array(fio.readimage("./data/Lenna.png"))

padding = np.array(np.floor(np.array(data.shape) *
                            (oversamplingRatio-1)*0.5), dtype='int64')
data = cp.pad(data, ((padding[0], padding[1]),
                     (padding[0], padding[1])), 'constant')
projection = np.abs(core.FFT(data))

projection = core.H_RL(projection, 1, 0.5)
#fio.showimage(cp.fft.fftshift(projection))
fourlieSpace = projection
realSpace = core.iFFT(fourlieSpace)

for i in range(2):
    alpha = projection.shape[0]**(1-2*i/9)
    realSpace = core.OSS(realSpace, projection, padding, alpha, 0.8, 1000)

result = cp.asnumpy(
    np.abs(realSpace[padding[0]:-padding[0], padding[1]:-padding[1]]))
# fio.showimage(result)
fio.writeimage(result, "./data/OSS/Lenna_test_OSS_0.5_anti.png")
print("1")
