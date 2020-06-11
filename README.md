# Source identification with HMC for reactive transport
control + shift + m to preview
## Setting up
Use flopy for the flow and transport simulation.
Build and install flopy, mt3dms, modflow 2005: follow this [tutorial](https://www2.hawaii.edu/~jonghyun/classes/S18/CEE696/files/05_1_flopy-pymake-compiling.pdf)

## Training samples 
The training samples are generated with gen_1500.ipynb in TCP folder. 
The location of the source is chosen from 20 candidates. The strength of the release is determined by the injection rate, the concentration in the injected water is fixed at 10000 m/V, the injection rate is varied for 10 periods, each period is 2 years' long. 

Therefore, the release happens for 20 years with 1 location parameter, 10 injection rate parameters.  During the release, maps are saved once every 2 years. After the release, the frequency of observation increases to 2 times a year, the observation is obtained for another 20 years, so the 50 times of observations are:
2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 20.5, 21, 21.5, ..., 40 th year.

The output samples are saved in google drive in this [link](https://drive.google.com/drive/folders/10jjWGuZxqcxH_TgAFH94WE4k_N_SLsZX?usp=sharing)
Use this command: [x_ind, x, q, mod_wels, con_rate, N_obs, N_maps] = pickle.load(file)
X_ind: 1500 *[well index(:0-19)] for well index
x: well locations, 1500* [y_wel(6, 13, 20, 27, 34), x_wel(1, 7, 13, 20)] for well location
q: 1500*[10 injection rate(:0-1000)]

