from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle as pk
from smt.sampling_methods import LHS
from multiprocessing import Pool
import sys
import os
import numpy as np
import matplotlib as mpl
import pickle as pk
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
import shutil
from IPython.display import clear_output
from time import sleep
import h5py
# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join('..', '..'))
    sys.path.append(fpth)
    import flopy
from flopy.utils.util_array import read1d
mpl.rcParams['figure.figsize'] = (8, 8)
from TCP3d_model import mymf

def simu(input_file):
    f = h5py.File(input_file, "r")
    kd = np.array(f["kd"])
    kd[kd<0] = 0
    kd = np.exp(kd)
    spd = np.array(f["welspd"])
    spd = {
        i:[
           tuple(
                [int(spd[i,0]), int(spd[i,1]), int(spd[i,2]),
                spd[i,3],
                int(spd[i,4])]
           )
        ]
        for i in range(len(spd))
    }
    f.close()
    my_model = mymf(dirname=input_file[:-5])
    ## run model
    conc, heads = my_model.run_model(kd, spd)
    
    f_out = h5py.File('../simu_outputs/output_' + input_file[6:], "w")
    f_out.create_dataset('concentration', data = conc, dtype ='f', compression = 'gzip')
    f_out.create_dataset('head', data = heads, dtype ='f', compression = 'gzip')
    f_out.close()
#     my_model.plot_head(kd[3], title='conductivity field')
#     my_model.simple_plot(conc[-1][3], title='conc')
#     my_model.plot_head(heads, title='head')
#     return conc, heads
    return