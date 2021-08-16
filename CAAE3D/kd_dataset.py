#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 17:27:34 2021

@author: zitongzhou
"""
import numpy as np
import h5py
import sys
import scipy.io

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt



def crop(input, depth, height, width, stride_d, stride_h, stride_w):
    D, H, W = input.shape # height and width of the original image
    print(input.shape)
    i_z   = (D-depth)//stride_d  # maximum z index
    i_row = (H-height)//stride_h # maximum row index
    i_col = (W-width)//stride_w  # maximum column index

    print(i_z,i_row,i_col)
    kds = []
    for i in range(i_z): # along z-axis first
        for j in range(i_row): # horizontally second
            for k in range(i_col): # then vertically
                # print(i,j)
                cond = input[
                    i*stride_d : i*stride_d+depth, 
                    j*stride_h:j*stride_h+height, 
                    k*stride_w : k*stride_w+width
                    ]
                kds.append(cond)
    return np.asarray(kds)

depth  = 6
height = 41
width  = 81
stride_d, stride_h, stride_w = 2, 6, 10

train_kds = []
for i in range(1,5):
    train_im = scipy.io.loadmat(
        '/Volumes/GoogleDrive/My Drive/react_inverse/CAAE/train_K/K{}.mat'.format(i)
        )
    train_im = train_im['K'] # image size 105 x 180 x 150
    train_kds.append(
        crop(train_im, depth, height, width, stride_d, stride_h, stride_w)
    )
train_kds = np.vstack(train_kds)
print(train_kds.shape)


test_kds = []
for i in range(1,5):
    test_im = scipy.io.loadmat(
        '/Volumes/GoogleDrive/My Drive/react_inverse/CAAE/test_K/K{}.mat'.format(i)
        )
    test_im = test_im['K'] # image size 15 x 180 x 150
    test_kds.append(
        crop(test_im, depth, height, width, stride_d, stride_h, stride_w)
    )
test_kds = np.vstack(test_kds)
print(test_kds.shape)