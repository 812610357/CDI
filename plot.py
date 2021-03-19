import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace

data = np.array([12.2181224613069,
                 11.620578540312538,
                 11.216215649663475,
                 10.942967383495667,
                 10.873013545170759,
                 11.022383198109509,
                 11.327591919030365,
                 12.02290240925748,
                 13.160756261366355,
                 16.011480146772143,
                 19.499825990340643])
d=np.array([14.0177,14.0177])

plt.plot(linspace(0,1,11),data,label="ER-HIO-FS")
plt.plot(linspace(0,1,2),d,label="ER-HIO")
plt.xlabel("Proportion of high-pass")  # X轴标签
plt.ylabel("RMSE")  # Y轴标签
plt.legend()
plt.show()

'''
11.57146671161051,
11.419583655682926,
11.708595535401379,
11.930548596235166,
11.065022843617864,
10.68742789974766,
11.283545939378474,
12.679756545127402,
14.614576227789538,
16.87979476062694,
19.014680912651702])
'''