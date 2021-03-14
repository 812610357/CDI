import cupy as cp
import numpy as np

import fileioput as fio

data1 = fio.readimage("./data/linear/Lenna_test_0.3-1.png")

data2=fio.readimage("./data/linear/Lenna_test_0.3-1_anti.png")

datarms=data1//2+data2//2

fio.writeimage(datarms,"./data/linear/Lenna_test_0.3-1_re.png")