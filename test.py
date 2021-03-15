import cupy as cp
import numpy as np

import fileioput as fio

data0=fio.readimage("./data/Lenna_gray_180d.png")

data1 = fio.readimage("./data/linear/Lenna_test_RL_0.5_re.png")

data2=fio.readimage("./data/linear/Lenna_test_0.3-1_anti.png")

datarms=data0+128-data1

fio.writeimage(datarms,"./data/linear/rms_0.5_re.png")