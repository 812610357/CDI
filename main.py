import math
import fileioput as fio
import core
import numpy as np
import matplotlib.pyplot as plt

oversamplingRatio=3
data=fio.readimage("./data/Lenna.png")
padding=np.array(np.floor(np.array(data.shape)*(oversamplingRatio-1)*0.5),dtype='int64')
data=np.pad(data,((padding[0],padding[1]),(padding[0],padding[1])),'constant')
projection=core.FFT(data)

fourlieSpace=projection
realSpace=np.zeros((projection.shape[0],projection.shape[1]))
for i in range(10):
    realSpace=core.iFFT(fourlieSpace)
    realSpace=core.ERcc(realSpace)
    fourlieSpace=core.FFT(realSpace)




projection=np.abs(projection)
fio.showimage(projection)
#io.writeimage(data,"./data/Lenna_test.png")