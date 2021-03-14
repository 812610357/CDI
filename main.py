import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import core
import fileioput as fio

path = "./data/Lenna_test.png"
oversamplingRatio = 3
data = cp.array(fio.readimage("./data/Lenna.png"))

padding = np.array(np.floor(np.array(data.shape) *
                            (oversamplingRatio-1)*0.5), dtype='int64')
data = cp.pad(data, ((padding[0], padding[1]),
                     (padding[0], padding[1])), 'constant')
projection = np.abs(core.FFT(data))

projection = core.H_RL(projection, 0.4, 1)
# fio.showimage(cp.fft.fftshift(projection))
fourlieSpace = projection
realSpace = core.iFFT(fourlieSpace)

for i in range(1000):
    realSpace = core.HIO(realSpace, projection, padding, 0.8, 20)
    realSpace = core.ER(realSpace, projection, padding, 5)

result = cp.asnumpy(
    np.abs(realSpace[padding[0]:-padding[0], padding[1]:-padding[1]]))
# fio.showimage(result)
fio.writeimage(result, "./data/linear/Lenna_test_0.4-1.png")
print("1")
