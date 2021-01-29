import numpy as np
import h5py
import sys
import scipy.io
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import pickle as pk
plt.switch_backend('agg')


def plot(idx):
    x = scipy.io.loadmat('input/{}.mat'.format(idx))
    x = x['cond']
    nrow = x.shape[0]//2
    fig, axes = plt.subplots(nrow,2)
    for j, ax in enumerate(fig.axes):
        if j < x.shape[0]:
            ax.set_axis_off()
            ax.set_aspect('equal')

            cax = ax.imshow(x[j], cmap='jet', origin='lower')

            cbar = plt.colorbar(cax, ax=ax, fraction=0.040, pad=0.04,
                                            format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.update_ticks()

    plt.savefig("{}.png".format(idx),bbox_inches='tight',dpi=300)

def crop(input, depth, height, width, stride_d, stride_h, stride_w, n):
    D, H, W = input.shape # height and width of the original image

    i_z   = (D-depth)//stride_d  # maximum z index
    i_row = (H-height)//stride_h # maximum row index
    i_col = (W-width)//stride_w  # maximum column index

    kds = []
    for i in range(i_z): # along z-axis first
        for j in range(i_row): # horizontally second
            for k in range(i_col): # then vertically
                # print(i,j)
                cond = input[i*stride_d : i*stride_d+depth, j*stride_h:j*stride_h+height, k*stride_w : k*stride_w+width]
                kds.append([cond])
    with open('kds_'+str(n)+'.pkl', 'wb') as file:
        pk.dump(kds, file)
    return

    
os.chdir('/Volumes/GoogleDrive/My Drive/react_inverse/training data')
depth  = 6
height = 41
width  = 81
stride_d, stride_h, stride_w = 2, 6, 10

for n in range(1,5):
    im = scipy.io.loadmat('K{}.mat'.format(n))
    im = im['K'] # image size N x D x H x W
    crop(im, depth, height, width, stride_d, stride_h, stride_w, n)
    