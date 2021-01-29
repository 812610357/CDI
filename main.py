import fileioput as fio
import core
import numpy as np
import matplotlib.pyplot as plt

path = "./data/Lenna_test.png"
oversamplingRatio = 3
data = fio.readimage("./data/Lenna.png")
padding = np.array(np.floor(np.array(data.shape) *
                            (oversamplingRatio-1)*0.5), dtype='int64')
data = np.pad(data, ((padding[0], padding[1]),
                     (padding[0], padding[1])), 'constant')
projection = np.abs(core.FFT(data))
# noise=np.random.poisson(projection.shape[0],)

fourlieSpace = projection
realSpace = core.iFFT(fourlieSpace)

for i in range(20):
    realSpace = core.HIO(realSpace, projection, padding, 0.5, 20)
    realSpace = core.ER(realSpace, projection, padding, 5)


projection = np.abs(realSpace[padding[0]:-padding[0], padding[1]:-padding[1]])
# fio.showimage(projection)
fio.writeimage(projection, "./data/Lenna_test.png")
