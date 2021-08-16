# Source identification with ES-MDA for reactive transport
control + shift + m to preview
This repo includes the construction of two neural networks (CAAE and DenseED), and ESMDA inversion, aiming to solve the inverse problem for the hydraulic conductivity field and the contaminant release history in a three dimensional domain.
##CAAE parameterization

##DenseED surrogate forward model
The training samples are saved in google drive in this [link](https://drive.google.com/file/d/1vJr15qSvBSEtMvNi-GZ4UOnAoKmnJZTt/view?usp=sharing)
with open('AR_dataset.pkl', 'rb') as file:
    [AR_input, AR_output] = pk.load(file)
The intput and output here are autoregressive input and output, the dimension of the input and output are C*H*D*W. C=3, H=6, D=41, W=81 for the input, C=2, H=6, D=41, W=81 for the output. 

The autoregressive model input and output:
![](images/AR_in_out.png?raw=true)

##ESMDA