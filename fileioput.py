import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from PIL import Image


def readimage(path):
    return(np.array(Image.open(path).convert('L')))


def showimage(data):
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('top')
    plt.pcolor(range(data.shape[0]+1), range(data.shape[1]+1), cp.asnumpy(data), cmap='jet',vmax=1000)
    plt.show()
    pass

def writeimage(data,path):
    data=np.array(np.floor(data*(255/np.max(data))),dtype='uint8')
    Image.fromarray(data).save(path)
    pass
