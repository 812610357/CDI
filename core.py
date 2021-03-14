import cupy as cp


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


def ER(realSpace, measurement, padding, t):
    fourlieSpace = None
    for i in range(t):
        fourlieSpace = FFT(realSpace)
        fourlieSpace = FScc(fourlieSpace, measurement)
        realSpace = iFFT(fourlieSpace)
        realSpace = ERcc(realSpace, padding)

    return(realSpace)


def HIO(realSpace, measurement, padding, beta, t):
    fourlieSpace = None
    for i in range(t):
        fourlieSpace = FFT(realSpace)
        fourlieSpace = FScc(fourlieSpace, measurement)
        realSpaceTemp = iFFT(fourlieSpace)
        realSpace = HIOcc(realSpace, realSpaceTemp, padding, beta)
    return(realSpace)


def NR(realSpace, measurement, padding, gamma, t):
    fourlieSpace = None
    for i in range(t):
        fourlieSpace = FFT(realSpace)
        fourlieSpace = FScc(fourlieSpace, measurement)
        realSpace = iFFT(fourlieSpace)
        realSpace = NRcc(realSpace, fourlieSpace, pdadding, gamma)
        realSpace = ERcc(realSpace, padding)
    return(realSpace)


def H_RL(data, low, high):
    data = cp.fft.fftshift(data)
    RLfilter = cp.array([[low, low], [low, low]], dtype='float64')
    RLfilter = cp.pad(RLfilter, ((data.shape[0]//2-1, data.shape[1]//2-1), (
        data.shape[0]//2-1, data.shape[1]//2-1)), 'linear_ramp', end_values=(high, high))
    data = RLfilter*data
    return(cp.fft.fftshift(data))


def H_(data):
    data = cp.fft.fftshift(data)
    RLfilter = cp.zeros((data.shape[0]//3, data.shape[1]//3))
    RLfilter = cp.pad(RLfilter, ((data.shape[0]//3, data.shape[1]//3),
                                 (data.shape[0]//3, data.shape[1]//3)),
                                 'constant', constant_values=(1, 1))
    data = RLfilter*data
    return(cp.fft.fftshift(data))
