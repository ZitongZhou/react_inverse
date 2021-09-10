# Hydraulic conductivity and comtaminant source identification with ESMDA for reactive transport

This repo includes the construction of two neural networks (CAAE and DenseED), and ESMDA inversion, aiming to solve the inverse problem for the hydraulic conductivity field and the contaminant release history in a three dimensional domain.
## Flow and transport simulation through PDE sovlers
MODFLOW+MT3DMS simulators are used to simulate the flow and transport processes, providing training/testing datasets for the surrogate DenseED model, and serve as the forward model in CAAE-ESMDA inversion framework. 
## CAAE parameterization
The training/testing data for CAAE of a three dimensional conductivity field can be found in the sub-folders in CAAE3D, CAAE3D/CAAE.ipynb is the GoogleColab notebook for CAAE model training.
![](images/CAAE_test.png?raw=true)
## DenseED surrogate forward model
Training/testing datasets and the training GoogleColab notebook, together with a model checkpoint can be found in CAAE-DenseED-ESMDA folder.
The autoregressive model input and output are illustrated in the following figure, together with the architecture of the DenseED convolutional neural network architecture:
![](images/denseED_arch.png?raw=true)

## ESMDA inversion
### CAAE-ESMDA

### CAAE-DenseED-ESMDA
