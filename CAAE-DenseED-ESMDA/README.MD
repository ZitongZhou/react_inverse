#DenseED surrogate model, CAAE-DenseED-ESMDA inversion

The training/testing samples are saved in google drive in this [link](https://drive.google.com/file/d/15nr0DL2eVqdDI9rXWyIn0WZjGiFOT2V_/view?usp=sharing)
```
with open('AR_dataset.pkl', 'rb') as file:
    [AR_input, AR_output] = pk.load(file)
```
The intput and output here are autoregressive input and output, the dimension of the input and output are C*H*D*W. C=3, H=6, D=41, W=81 for the input, C=2, H=6, D=41, W=81 for the output. 

checkpoint for the model is in this [link](https://drive.google.com/file/d/108urmY1Aigk9PvqbzGZSCPDIibhJNKzm/view?usp=sharing)