from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def readimage(path):
    return(np.array(Image.open(path).convert('L')))


def showimage(data):
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('top')
    plt.pcolor(range(data.shape[0]+1), range(data.shape[1]+1), data, cmap='jet',vmax=1e5)
    plt.show()
    pass

def writeimage(data,path):
    Image.fromarray(data).save(path)
    pass