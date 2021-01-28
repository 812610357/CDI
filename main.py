import fileioput as fio
import core
import numpy as np
import matplotlib.pyplot as plt
path="./data/Lenna_test.png"
oversamplingRatio=3
data=fio.readimage("./data/Lenna.png")
padding=np.array(np.floor(np.array(data.shape)*(oversamplingRatio-1)*0.5),dtype='int64')
data=np.pad(data,((padding[0],padding[1]),(padding[0],padding[1])),'constant')
projection=np.abs(core.FFT(data))
#noise=np.random.poisson(projection.shape[0],)

fourlieSpace=projection
realSpace=np.zeros((projection.shape[0],projection.shape[1]))
for i in range(1000):
    realSpace=core.iFFT(fourlieSpace)
    realSpace=core.ERcc(realSpace,padding)
    fourlieSpace=core.FFT(realSpace)
    fourlieSpace=core.FScc(fourlieSpace,projection)




projection=np.abs(realSpace[padding[0]:-padding[0],padding[1]:-padding[1]])
fio.showimage(projection)
fio.writeimage(projection,"./data/Lenna_test.png")