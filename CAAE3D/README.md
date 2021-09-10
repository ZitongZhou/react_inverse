# Training and testing data for CAAE
train_K folder and test_K folder each includes four Kx.mat, corresponding to the original training and testing images and their rotated versions along each axis. With make_dataset.py, 6x41x81 images are cropped from four Kx.mat files in train_K folder and then saved as [kds.pkl](https://drive.google.com/file/d/1hdYAz3EAKlYCLvbrmUOK7FGRgtmmHu2D/view?usp=sharing) as the training data for CAAE, those in test_K folder form [test_kds.pkl](https://drive.google.com/file/d/1RpWeLDLDx6titiFM4UTKeWVB_-irYjqg/view?usp=sharing). Two pickle files can be downloaded from the links.

The combined training+testing image is shown here:
![](logk_training.png?raw=true)
# CAAE training
CAAE.ipynb is the GoogleColab notebook to train the CAAE models with the data given above.
