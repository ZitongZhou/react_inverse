# Hydraulic conductivity and comtaminant source identification with ESMDA for reactive transport

This repo includes the construction of two neural networks (CAAE and DenseED), and ESMDA inversion, aiming to solve the inverse problem for the hydraulic conductivity field and the contaminant release history in a three dimensional domain. 
 
The organization of the three folders are shown below: 
## CAAE
### parameterization of the hydraulic conductivity field
The training/testing data for CAAE of a three dimensional conductivity field can be found in the sub-folders in CAAE3D, CAAE3D/CAAE.ipynb is the GoogleColab notebook for CAAE model training.
![](images/CAAE_test.png?raw=true)

## CAAE-ESMDA
This folder includes the MODFLOW+MT3DMS simulation functions, and the CAAE-ESMDA inversion.
### Flow and transport simulation through PDE sovlers
MODFLOW+MT3DMS simulators are used to simulate the flow and transport processes, providing training/testing datasets for the surrogate DenseED model, and serve as the forward model in CAAE-ESMDA inversion framework.
### CAAE-ESMDA inversion
ESMDA-MT3D.ipynb includes the Jupyter notebook scripts to perform ESMDA inversion.

## CAAE-DenseED-ESMDA
This folder includes the DenseED convolutional neural network surrogate model construction, and the CAAE-DenseED-ESMDA inversion.
### DenseED surrogate model
Training/testing datasets and the training GoogleColab notebook, together with a model checkpoint can be found in CAAE-DenseED-ESMDA folder.
The autoregressive model input and output are illustrated in the following figure, together with the architecture of the DenseED convolutional neural network architecture:
![](images/denseED_arch.png?raw=true)
### CAAE-ESMDA inversion
CAAE-ESMDA/CAAE-DenseED-ESMDA.ipynb includes the Jupyter notebook scripts to perform ESMDA inversion.




