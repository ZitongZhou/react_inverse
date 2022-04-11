# CAAE-ESMDA
## MODFLOW + MT3DMS simulations
Flopy(python package for MODFLOW) with MT3DMS are used to simulate the flow and transport processes. TCP3d_model.py include the model settings except for the conductivity field and the contaminant release information. input_dataset.ipynb includes the scripts to generate input hdf5 files with the conductivity and release configurations, it is later used to read the output hdf5 files to generate dataset.
para_simu.py wraps a function to read the input hdf5 file and then simulate and save the model outputs to an output hdf5 file. gen_simus.ipynb is a Jupyter notebook to generate a large number of simulations in parallel utilizing multi-threads processing with python.
Simulation input files and simulation output files can be found in [link1](https://drive.google.com/drive/folders/1ImiSOEbkJxTXhWENAnVPQLeXvxRGNdpg?usp=sharing) and [link2]( https://drive.google.com/drive/folders/1a_v4yi5TGE2ilG9jD4g9Mu6uUTdUbeck?usp=sharing) in Google drive.

## MODFLOW + MT3DMS simulations with fake conductivity field

150 testing conductivity fields data are passed through the CAAE to obtain fake (reconstructed) conductivity fields. These fake conductivity fields are then used with the same release history setting with the corresponding testing dataset to run simulations with MODFLOW + MT3DMS. This is to test the ability of CAAE in the PDE-based forward model. Code is gathered in MT3D_fakeKD.ipynb.

Simulations are packed as autoregressive type of input and output, and can be found here(https://drive.google.com/file/d/1M9_ShQiXaJ6pwzqmXlTzNnjs6BnJYhhM/view?usp=sharing), the unpacking function can be found in MT3D_fakeKD.ipynb too.

## ESMDA inversion
ESMDA-MT3D.ipynb is the jupyter notebook for CAAE-ESMDA inversion with MODFLOW+MT3DMS simulators.
The resulting 11 ESMDA ensembles can be found in this [link](https://drive.google.com/drive/folders/1NQObllG025n1LN6PXVjYd5ULNU2xylRS?usp=sharing) .

The three reference datasets are attached as real_hk_data.pkl, set2_hk_data.pkl, set3_hk_data.pkl.

The inspection of the ESMDA results are done together with CAAE-DenseED-ESMDA results inspection in CAAE-DenseED-ESMDA/CAAE-DenseED-ESMDA.ipynb GoogleColab notebook.


