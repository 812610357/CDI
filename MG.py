import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import core
import fileioput as fio

path = "./data/Lenna_test.png"
oversamplingRatio = 3
data0 = cp.array(fio.readimage("./data/Lenna.png"))

level0=core.FFTs(data0)
len0=level0.shape
plt.figure(figsize=(4, 4))
fio.showimage(cp.abs(level0),vmax=1e4)
plt.text(5,25,'$512\\times 512$',fontsize=8,color='w')
plt.plot([128,384,384,128,128],[128,128,384,384,128],'w',linewidth='1')
plt.text(128,128-5,'$256\\times 256$',fontsize=8,color='w')
plt.plot([192,320,320,192,192],[192,192,320,320,192],'w',linewidth='1')
plt.text(192,192-5,'$128\\times 128$',fontsize=8,color='w')
plt.plot([224,288,288,224,224],[224,224,288,288,224],'w',linewidth='1')
plt.text(224,224-5,'$64\\times 64$',fontsize=8,color='w')
plt.plot([240,272,272,240,240],[240,240,272,272,240],'w',linewidth='1')
plt.text(240,240-5,'$32\\times 32$',fontsize=4,color='w')
plt.savefig('./doc/0517/images/f.pdf')
'''
level1=level0[len0[0]//2-len0[0]//8:len0[0]//2+len0[0]//8,len0[1]//2-len0[1]//8:len0[1]//2+len0[1]//8]
data1=core.iFFTs(level1)/(4**2)
plt.figure(figsize=(4, 4))
fio.showimage(cp.abs(data1),cmap='gray')
plt.savefig('./doc/0517/images/128.pdf')
'''