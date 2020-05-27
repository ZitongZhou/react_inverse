#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:51:00 2020

@author: zitongzhou
"""

import numpy as np
import gstools as gs
import matplotlib.pyplot as plt

# x = np.arange(80)
# y = np.arange(40)
# model = gs.Exponential(dim=2, var=0.5, len_scale=[100, 50])
# srf = gs.SRF(model, seed=20170519)
# srf.structured([x, y])
# srf.plot()


def fftIndgen(n):
    a = range(0, n//2+1)
    b = range(1, n//2)
    b = b[::-1]
    b = [-i for i in b]
    return list(a) + b

def gaussian_random_field(Pk = lambda k : k**-3.0, size1 = 100, size2 = 100):
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))
    noise = np.fft.fft2(np.random.normal(size = (size1, size2)))
    amplitude = np.zeros((size1,size2))
    for i, kx in enumerate(fftIndgen(size1)):
        for j, ky in enumerate(fftIndgen(size2)):            
            amplitude[i, j] = Pk2(kx, ky)
    return np.fft.ifft2(15*noise * amplitude)

for alpha in [-3.5]:
    out = gaussian_random_field(Pk = lambda k: k**alpha, size1=81, size2 = 41)
    out = out.real+np.abs(np.min(out.real))
    out= out.transpose()
    # out = np.exp(out)
    plt.figure()
    plt.imshow(out, interpolation='none')
    plt.colorbar()
    
    
import pickle as pk
with open('log_k','wb') as file:
     pk.dump(out, file)