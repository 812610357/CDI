import cupy as cp
import numpy as np

import fileioput as fio

data0=fio.readimage("./data/Lenna_gray_180d.png")

data1 = fio.readimage("./data/OSS/Lenna_test_OSS_0.5.png")

data2=fio.readimage("./data/OSS/Lenna_test_OSS_0.5_anti.png")



for i in range(11):
    datarms=0.1*i*data1+0.1*(10-i)*data2
    fio.writeimage(datarms,"./data/OSS/Lenna_test_OSS_0.5_re_"+str(0.1*i)+".png")

    data = fio.readimage(
    "./data/OSS/Lenna_test_OSS_0.5_re_"+str(0.1*i)+".png")
    a=np.array(data,dtype='float64')
    b=np.array(data0,dtype='float64')
    print(np.sqrt(np.mean((b-a)**2)))

