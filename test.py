import cupy as cp
import numpy as np

import fileioput as fio

data1 = fio.readimage("./data/Lenna_test_RL_0.png")

data2=fio.readimage("./data/Lenna_test_RL_0_anti.png")

datarms=data1//2+data2//2

fio.writeimage(datarms,"./data/Lenna_test_RL_0_re.png")