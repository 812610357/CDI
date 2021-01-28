import numpy as np

def FFT(data):
    '''
    傅里叶变换
    '''
    return(np.fft.fft2(data))

def iFFT(data):
    '''
    傅里叶逆变换
    '''
    return(np.fft.ifft2(data))

def ERcc(data,padding):
    '''
    ER算法的实空间约束条件
    '''
    data=data[padding[0]:-padding[0],padding[1]:-padding[1]]
    data=np.pad(data,((padding[0],padding[1]),(padding[0],padding[1])),'constant')
    return(data)

def HIOcc(data,temp,padding,beta):
    '''
    HIO算法的实空间约束条件
    '''
    inside=temp[padding[0]:-padding[0],padding[1]:-padding[1]]
    insaide=np.pad(inside,((padding[0],padding[1]),(padding[0],padding[1])),'constant')
    outside=data-beta*temp
    for i in range(padding[0],outside.shape[0]-padding[0]):
        for j in range(padding[1],outside.shape[1]-padding[1]):
            outside[i,j]=0
    return(insaide+outside) 

def FScc(data,measurement):
    data=np.abs(measurement)*data/np.abs(data)
    return(data)

def ER(realSpace,measurement,padding,t):
    fourlieSpace=None
    for i in range(t):
        fourlieSpace=FFT(realSpace)
        fourlieSpace=FScc(fourlieSpace,measurement)
        realSpace=iFFT(fourlieSpace)
        realSpace=ERcc(realSpace,padding)
    return(realSpace)


def HIO(realSpace,measurement,padding,beta,t):
    fourlieSpace=None
    for i in range(t):
        fourlieSpace=FFT(realSpace)
        fourlieSpace=FScc(fourlieSpace,measurement)
        realSpaceTemp=iFFT(fourlieSpace)
        realSpace=HIOcc(realSpace,realSpaceTemp,padding,beta)
    return(realSpace)