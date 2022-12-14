# Deep Learning for Simultaneous Inference of Hydraulic and Transport Properties

This repo includes the construction of two neural networks (CAAE and DenseED), and ESMDA inversion, aiming to solve the inverse problem for the hydraulic conductivity field and the contaminant release history in a three dimensional domain. 
 
The organization of the three folders are shown below: 
## CAAE
### parameterization of the hydraulic conductivity field
The training/testing data for CAAE of a three dimensional conductivity field can be found in CAAE, CAAE/CAAE.ipynb is the GoogleColab notebook for CAAE model training.
![](images/CAAE_test.png?raw=true)

## CAAE-ESMDA
This folder includes the MODFLOW+MT3DMS simulation functions, and the CAAE-ESMDA inversion.

### Flow and transport simulation through PDE solvers
MODFLOW+MT3DMS simulators are used to simulate the flow and transport processes, providing training/testing datasets for the surrogate DenseED model, and serve as the forward model in CAAE-ESMDA inversion framework.

### Flow and transport simulation with reconstructed conductivity field through PDE solvers
We passed a set of conductivity fields through the encoder and decoder in CAAE, and compared the simulated concentration and hydraulic head results simulated with the PDE solver.

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

### Citation
If you find this useful, please cite us:
```
@article{zhou2022deep,
  title={Deep Learning for Simultaneous Inference of Hydraulic and Transport Properties},
  author={Zhou, Zitong and Zabaras, Nicholas and Tartakovsky, Daniel M},
  journal={Water Resources Research},
  volume={58},
  number={10},
  pages={e2021WR031438},
  year={2022},
  publisher={Wiley Online Library}
}
```


