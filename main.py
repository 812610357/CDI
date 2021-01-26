import math
import ioput as io
import numpy as np
import matplotlib.pyplot as plt

OversamplingRatio=3
data=io.readimage("./data/Lenna.png")
padding=np.array(np.floor(np.array(data.shape)*(OversamplingRatio-1)*0.5),dtype='int64')
data=np.pad(data,((padding[0],padding[1]),(padding[0],padding[1])),'constant')
projection=np.fft.fftshift(np.fft.fft2(data))


projection=np.abs(projection)
io.showimage(projection)
#io.writeimage(data,"./data/Lenna_test.png")