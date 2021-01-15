#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:22:29 2020

@author: zitongzhou
"""
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
annots = loadmat("/Volumes/GoogleDrive/My Drive/inverse_modeling/reactive_transport/kd.mat")
annots.keys()

kle = annots['kle_terms']
K = annots['K']

MeanY = 2.
eig_vals = loadmat("/Volumes/GoogleDrive/My Drive/inverse_modeling/reactive_transport/eig_vals.mat")
eig_vals = eig_vals['eig_vals']
eig_vecs = loadmat("/Volumes/GoogleDrive/My Drive/inverse_modeling/reactive_transport/eig_vecs.mat")
eig_vecs = eig_vecs['eig_vecs']
log_K = MeanY + np.matmul( np.matmul(eig_vecs,  np.sqrt(eig_vals)), np.transpose(kle))
K_re = np.exp(log_K)

##np.transpose(np.reshape(K_re[:,i], (81, 41)))

fig, axs = plt.subplots(1,1)
c01map = axs.imshow(np.transpose(np.reshape(K_re[:,3], (81, 41))), interpolation='none')
fig.colorbar(c01map, ax=axs,shrink=0.62)
title = 'hydraulic conductivity'
name = title + '.pdf'
plt.title(title)
# fig.savefig(name, format='pdf',bbox_inches='tight')