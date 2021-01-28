import numpy as np

def FFT(data):
    '''
    傅里叶变换
    '''
    return(np.fft.fftshift(np.fft.fft2(data)))

def iFFT(data):
    '''
    傅里叶逆变换
    '''
    return(np.fft.ifft2(np.fft.ifftshift(data)))

def ERcc(data,padding):
    '''
    ER算法的实空间约束条件
    '''
    data=data[padding[0]:-padding[0],padding[1]:padding[1]]
    data=np.pad(data,((padding[0],padding[1]),(padding[0],padding[1])),'constant')
    return(data)
