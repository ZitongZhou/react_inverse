# Source identification with HMC for reactive transport
control + shift + m to preview
## Setting up
Use flopy for the flow and transport simulation.
Build and install flopy, mt3dms, modflow 2005: follow this [tutorial](https://www2.hawaii.edu/~jonghyun/classes/S18/CEE696/files/05_1_flopy-pymake-compiling.pdf)

## Training samples 
depth  = 6
height = 41
width  = 81
Model: 
### Conductivity field data
K: conductivity field, use TCE_3d/KD_training_data/ gen_save_data.py & K1.mat, K2.mat, K3.mat, K4.mat to generate the 3D conductivity fields, .mat files and the saved .pkl files are all in [Google Drive](https://drive.google.com/file/d/1_mRIL1jbApaRTdbS8zTzjejqjnlDMSzy/view?usp=sharing).
### Contaminant release location
The release is from 1 of the 20 candidate wells as constant concentration release.
![](TCP_3d/images/Head_well.png?raw=true)


The training samples are generated with gen_1500.ipynb in TCP folder. 
The location of the source is chosen from 20 candidates. The strength of the release is determined by the injection rate, the concentration in the injected water is fixed at 10000 m/V, the injection rate is varied for 5 periods, each period is 5 years' long. 

Therefore, the release happens for 20 years with 1 location parameter, 10 injection rate parameters.  During the release, maps are saved once every 2 years. After the release, the frequency of observation increases to 2 times a year, the observation is obtained for another 20 years, so the 50 times of observations are:
5, 10, 15, ... th year.

The output samples are saved in google drive in this [link](https://drive.google.com/file/d/1vJr15qSvBSEtMvNi-GZ4UOnAoKmnJZTt/view?usp=sharing)
with open('AR_dataset.pkl', 'rb') as file:
    [AR_input, AR_output] = pk.load(file)
The intput and output here are autoregressive input and output, as it is a 3D simulation, C*H*D*W(C=10, H=6, D=41, W=81) would be a very big matrix, causing memory problem.

The autoregressive model input and output:
![](TCP_3d/images/AR_in_out.png?raw=true)
