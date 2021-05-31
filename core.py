import cupy as cp
import pycuda
import numpy as np
import scipy.interpolate as intp
from PIL import Image

import fileioput as fio



def FFT(data):
    '''
    傅里叶变换
    '''
    return(cp.fft.fft2(data))


def iFFT(data):
    '''
    傅里叶逆变换
    '''
    return(cp.fft.ifft2(data))


def FFTs(data):
    '''
    傅里叶变换
    '''
    return(cp.fft.fftshift(cp.fft.fft2(data)))


def iFFTs(data):
    '''
    傅里叶逆变换
    '''
    return(cp.fft.ifft2(cp.fft.fftshift(data)))


def FScc(data, measurement):
    data = cp.abs(measurement)*data/cp.abs(data)
    return(data)


def ERcc(data, padding):
    '''
    ER算法的实空间约束条件
    '''
    data = data[padding[0]:-padding[0], padding[1]:-padding[1]]
    data = cp.pad(data, ((padding[0], padding[1]),
                         (padding[0], padding[1])), 'constant')
    return(data)


def HIOcc(data, temp, padding, beta):
    '''
    HIO算法的实空间约束条件
    '''
    inside = temp[padding[0]:-padding[0], padding[1]:-padding[1]]
    outside = data-beta*temp
    outside[padding[0]:outside.shape[0]-padding[0],
            padding[1]:outside.shape[1]-padding[1]] = inside
    return(outside)


def NRcc(data, fourlie, padding, gamma):
    outside = data
    outside[padding[0]:outside.shape[0]-padding[0],
            padding[1]:outside.shape[1]-padding[1]] = cp.zeros(padding)
    nosiedata = FFT(outside)
    data = iFFT(fourlie-nosiedata*gamma)
    return(data)


def OSScc(data, padding, alpha):
    '''
    OSS算法的实空间约束条件
    '''
    inside = data[padding[0]:-padding[0], padding[1]:-padding[1]]
    outside = iFFT(FFT(data)*gaussion(alpha, data.shape))
    outside[padding[0]:outside.shape[0]-padding[0],
            padding[1]:outside.shape[1]-padding[1]] = inside
    return(outside)


def ER(realSpace, measurement, padding):
    fourlieSpace = FFT(realSpace)
    fourlieSpace = FScc(fourlieSpace, measurement)
    realSpace = iFFT(fourlieSpace)
    realSpace = ERcc(realSpace, padding)

    return realSpace, fourlieSpace


def HIO(realSpace, measurement, padding, beta):
    fourlieSpace = FFT(realSpace)
    fourlieSpace = FScc(fourlieSpace, measurement)
    realSpaceTemp = iFFT(fourlieSpace)
    realSpace = HIOcc(realSpace, realSpaceTemp, padding, beta)
    return realSpace, fourlieSpace


def NR(realSpace, measurement, padding, gamma, t):
    fourlieSpace = None
    for i in range(t):
        fourlieSpace = FFT(realSpace)
        fourlieSpace = FScc(fourlieSpace, measurement)
        realSpace = iFFT(fourlieSpace)
        realSpace = NRcc(realSpace, fourlieSpace, padding, gamma)
        realSpace = ERcc(realSpace, padding)
    return(realSpace)


def H_RL(data, low, high):
    data = cp.fft.fftshift(data)
    RLfilter = cp.array([[low, low], [low, low]], dtype='float64')
    RLfilter = cp.pad(RLfilter, ((data.shape[0]//2-1, data.shape[1]//2-1), (
        data.shape[0]//2-1, data.shape[1]//2-1)), 'linear_ramp', end_values=(high, high))
    data = RLfilter*data
    return(cp.fft.fftshift(data))


def gaussion(alpha, shape):
    data = cp.zeros((2, 2))
    data = cp.pad(data, ((shape[0]//2-1, shape[1]//2-1), (shape[0]//2-1, shape[1]//2-1)),
                  'linear_ramp', end_values=(shape[0]//2-1, shape[1]//2-1))
    return(cp.fft.fftshift(cp.exp(-(data/alpha)**2/2)))


def OSS(realSpace, measurement, padding, alpha, beta, t):
    fourlieSpace = None
    for i in range(t):
        fourlieSpace = FFT(realSpace)
        fourlieSpace = FScc(fourlieSpace, measurement)
        realSpaceTemp = iFFT(fourlieSpace)
        realSpace = HIOcc(realSpace, realSpaceTemp, padding, beta)
        realSpace = OSScc(realSpace, padding, alpha)
    return(realSpace)


def MultiGridFilter(data):
    data = cp.fft.fftshift(data)
    size = data.shape
    data = data[size[0]//2-size[0]//4:size[0]//2+size[0] //
                4, size[1]//2-size[1]//4:size[1]//2+size[1]//4]
    data = cp.fft.fftshift(data)
    return data

def MultiGridRestriction(data):
    size = data.shape
    '''
    data=cp.asnumpy(data.reshape((1,size[0]*size[1])))[0]
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    x, y = np.meshgrid(x, y)
    x=x.reshape((size[0]*size[1],1))
    y=y.reshape((size[0]*size[1],1))
    xy=np.column_stack((x,y))
    x_new = np.linspace(0, 1, size[0]//2)
    y_new = np.linspace(0, 1, size[1]//2)
    x_new, y_new = np.meshgrid(x_new, y_new)
    res=intp.griddata(xy,data,(x_new, y_new),method='cubic')
    res=cp.asarray(res)
    '''
    '''
    x = np.arange(0, size[0],2)
    y = np.arange(0, size[1],2)
    x, y = np.meshgrid(x, y)
    res=np.zeros((size[0]//2,size[1]//2))
    res=(data[y,x]+data[y+1,x]+data[y,x+1]+data[y+1,x+1])/4
    '''
    
    x = cp.arange(2, size[0],2)
    y = cp.arange(2, size[1],2)
    xi, yi = cp.meshgrid(x, y)
    res=cp.zeros((size[0]//2,size[1]//2))
    res[1:,1:]=(4*data[yi,xi]+2*(data[yi-1,xi]+data[yi+1,xi]+data[yi,xi-1]+data[yi,xi+1])+data[yi-1,xi-1]+data[yi-1,xi+1]+data[yi+1,xi+1]+data[yi+1,xi-1])/16
    res[0,1:]=(2*data[0,x]+data[0,x-1]+data[0,x+1])/4
    res[1:,0]=(2*data[y,0]+data[y-1,0]+data[y+1,0])/4
    res[0,0]=(2*data[0,0]+data[1,0]+data[0,1])/4
    
    '''
    x = cp.arange(1, size[0]-1,2)
    y = cp.arange(1, size[1]-1,2)
    xi, yi = cp.meshgrid(x, y)
    res=cp.zeros((size[0]//2,size[1]//2))
    res[:-1,:-1]=(4*data[yi,xi]+2*(data[yi-1,xi]+data[yi+1,xi]+data[yi,xi-1]+data[yi,xi+1])+data[yi-1,xi-1]+data[yi-1,xi+1]+data[yi+1,xi+1]+data[yi+1,xi-1])/16
    res[-1,:-1]=(2*data[-1,x]+data[-1,x-1]+data[-1,x+1])/4
    res[:-1,-1]=(2*data[y,-1]+data[y-1,-1]+data[y+1,-1])/4
    res[-1,-1]=(2*data[-1,-1]+data[-1,-2]+data[-2,-1])/4
    '''
    return res

def MultiGridProlongation(data):
    size = data.shape
    data=cp.asnumpy(data.reshape((1,size[0]*size[1])))[0]
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    x, y = np.meshgrid(x, y)
    x=x.reshape((size[0]*size[1],1))
    y=y.reshape((size[0]*size[1],1))
    xy=np.column_stack((x,y))
    x_new = np.linspace(0, 1, size[0]*2)
    y_new = np.linspace(0, 1, size[1]*2)
    x_new, y_new = np.meshgrid(x_new, y_new)
    res=intp.griddata(xy,data,(x_new,y_new),method='linear')
    return cp.asarray(res)