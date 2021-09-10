#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 17:27:34 2021

@author: zitongzhou
"""
import numpy as np
import pickle as pkl

import scipy.io
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors
import matplotlib.backends.backend_pdf


mpl.rcParams['figure.figsize'] = (8, 8)
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rcParams['text.usetex'] = True
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


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

kds = []
for i in range(1,5):
    train_im = scipy.io.loadmat(
        '/Volumes/GoogleDrive/My Drive/react_inverse/CAAE/train_K/K{}.mat'.format(i)
        )
    train_im = train_im['K'] # image size 105 x 180 x 150
    kds.append(
        crop(train_im, depth, height, width, stride_d, stride_h, stride_w)
    )
kds = np.vstack(kds)
print(kds.shape)
with open('kds.pkl','wb') as file:
    pkl.dump(kds, file)

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

with open('/Volumes/GoogleDrive/My Drive/react_inverse/CAAE/test_K/test_kds.pkl','wb') as file:
    pkl.dump(test_kds, file)
    
def plot_3d(data, title='', cut=None):
    data = np.transpose(data, (2, 1, 0))
    data = np.flip(data, axis=2)
    filled = np.ones(data.shape)
    if cut is not None:
        filled[cut[2]:, :cut[1], (6-cut[0]):] = 0
    x, y, z = np.indices(np.array(filled.shape) + 1)
    
    v1 = np.linspace(np.min(data),np.max(data), 8, endpoint=True)
    norm = matplotlib.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(x, y, z, filled, facecolors=plt.cm.jet(norm(data)), edgecolors=None)
    # ax.set_box_aspect([250, 125, 50])
    ax.set_box_aspect([180,150,120])
    
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    fig.colorbar(m, ax=ax, fraction=0.015, pad=0.04,ticks=v1,)
    ax.set_axis_off()
    plt.tight_layout()
    # ax.set_title(title)
    plt.savefig(title+'.pdf',bbox_inches='tight')
    return fig   
    
whole_img = np.vstack((train_im, test_im))
fig = plot_3d(whole_img, title='/Users/zitongzhou/Desktop/react_inverse/CAAE3D/logk_training', cut=None)    

    
    
    
    