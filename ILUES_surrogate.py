import numpy as np
import time
import sys
import os
import torch
import scipy.io
from attrdict import AttrDict
from ILUES import ILUES
from dense_ed import DenseED
import argparse
from torch.utils.data import DataLoader
Par=AttrDict()

'''Load the CAAE model first'''
n_train = 23000
n_test = 100
batch_size = 64
n_epochs = 50
lr = 0.0002 ## adam learning rate
lw = 0.01 ## "adversarial loss weight"

current_dir = "/content/drive/MyDrive/react_inverse/CAAE/"
date = 'experiments/Feb_14_CAAE3D'
exp_dir = current_dir + date + "/N{}_Bts{}_Eps{}_lr{}_lw{}".\
    format(n_train, batch_size, n_epochs, lr, lw)

output_dir = exp_dir + "/predictions"
model_dir = exp_dir

nf, d, h, w = 2, 2, 11, 21

# Initialize generator and discriminator

decoder = Decoder(inchannels=nf)
decoder.load_state_dict(torch.load(model_dir + '/AAE_decoder_epoch{}.pth'.format(n_epochs)))
if cuda:
    decoder.cuda()

decoder.eval()


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor    


'''Forward surrogate model'''
exp_dir = "/content/drive/MyDrive/react_inverse/Bayesian_DenseED"
all_over_again = 'Apr_04'
n_samples = 20
n_train = 1000
batch_size = 40
lr = 0.005
lr_noise = 0.01
n_epochs = 300
ckpt_epoch = None
run_dir = exp_dir + '/' + all_over_again\
    + '/ns{}_ntr{}_bt{}_lr{}_lrn{}_ep{}'.format(
        n_samples, n_train, batch_size, lr,
        lr_noise, n_epochs)
pred_dir = run_dir + '/predictions'
ckpt_dir = run_dir + '/checkpoints'
nic = 3
noc = 2
blocks = [3,6,3]
growth_rate = 40
init_features = 48
drop_rate = 0.
bn_size = 8
bottleneck = False

dense_ed = DenseED(
    in_channels=nic, 
    out_channels=noc, 
    blocks=blocks,
    growth_rate=rowth_rate, 
    num_init_features=init_features,
    drop_rate=drop_rate,
    bn_size=bn_size,
    bottleneck=bottleneck,
)
# print(dense_ed)
# Bayesian NN
bayes_nn = BayesNN(dense_ed, n_samples=n_samples).to(device)
# load the pre-trained model
if ckpt_epoch is not None:
    checkpoint = ckpt_dir + '/model_epoch{}.pth'.format(ckpt_epoch)
else:
    checkpoint = ckpt_dir + '/model_epoch{}.pth'.format(epochs)
bayes_nn.load_state_dict(torch.load(checkpoint))
print('Loaded pre-trained model: {}'.format(checkpoint))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # training on GPU or CPU


def run_surrogate(X, Par, model):
    ya = numpy.zeros(Par.Ne, Par.Nobs)
    model.eval()
    log_K, source = gen_input4net(X, Par)
    for ind in Par.Ne:
        x = np.full((1,3,6,41,81), 0.0)           # three input channels: hydraulic conductivity field, source term, previous concentration field
        y = np.full( (Par.Nt,2,6,41,81), 0.0) # two output channles: concentration and head fields
        y_i_1 = np.full((6,41,81), 0.0)   # y_0 = 0

        for i in range(Par.Nt):
            x[0,0,:,:,:] = log_K           # hydraulic conductivity
            x[0,1,:,:,:] = source[i]      # source rate
            x[0,2,:,:,:] = y_i_1          # the i-1)^th predicted concentration field, which is treated as an input channel
            x_tensor = Tensor(x)
            with torch.no_grad():
                y_hat, y_var = model.predict(x_tensor)
            y_hat = y_hat.data.cpu().numpy()
            y[i] = y_hat
            y_i_1 = y_hat[0,0,:,:,:]      # the updated (i-1)^th predicted concentration field

        y_pred = np.full( (Par.Nt + 1,6,41,81), 0.0)
        y_pred[:Par.Nt] = y[:,0]   # the concentration fields at Nt time instances
        y_pred[Par.Nt]  = y[0,1]   # the hydraulic head field
        ya[ind, :] = get_obs(sensor, y_pred)  # get the simulated outputs at observation locations using interpolation

    return ya

def gen_input4net(X, Par):
    '''generate batch input'''
    ## log conductivity field
    latent_z = X[:Par.Nlat, :]
    latent_z = torch.reshape(
          Tensor(latent_z), (-1, nf, d, h, w)
          )
    log_K = decoder(latent_z)
    ## source loc
    y_wel_samp = X[Par.Nlat, :]
    x_wel_samp = X[Par.Nlat+1, :]
    Sy_id, Sx_id = step_loc(y_wel_samp, x_wel_samp)
    ## source rate
    source_rate = X[Par.Nlat+2:, :]
    source = np.zeros(Par.Ne, Par.Nt, 6, 41, 81,)
    for j in range(Par.Nt_re): #j'th timestep of release
        for i in range(Par.Ne): #i'th sample
            source[i, j, 3, Sy_id[i], Sx_id[i]] = source_rate[i, j]

    return log_K, source

def get_obs(sensor, y_pred):

    y_sim_obs = []
    for y in y_pred:
        y_sim_obs.append(y[Par.sensor])

    return y_sim_obs

def gen_init(Ne, Par, seed=888):
    x = np.zeros((Par.Npar, Ne))
    np.random.seed(seed)
    ## log_K
    x[:Par.Nlat, :] = np.random.randn(Par.Nlat, Ne)
    ## release locations
    y_wel = np.array([125, 125*3, 125*5, 125*7, 125*9])
    x_wel = np.array([125, 125*3, 125*5, 125*7])
    # wells = {i: [y_wel[i], x_wel[i]] for i in range(len(y_wel))}
    y_wel_samp = np.random.choice(y_wel, Ne)
    x_wel_samp = np.random.choice(x_wel, Ne)

    x[Par.Nlat,:] = y_wel_samp
    x[Par.Nlat+1,:] = x_wel_samp
    ## release concentration for 5 periods
    q = np.random.uniform(low=100, high=1000, size=(5,Ne)).astype(int)
    x[Par.Nlat+2:,:] = q
    return x

'''MAKE A FUNCTION TO STEPWISE THE WELL LOCATIONS'''
def step_loc(y_loc, x_loc):
    '''
    x_loc: (Ne, ), convert from meters to index,
    y_loc: (Ne, ), convert from meters to index.
    '''
    dy = 1250/40
    dx = 2500/80
    y_wel = np.array([125, 125*3, 125*5, 125*7, 125*9])
    x_wel = np.array([125, 125*3, 125*5, 125*7])
    N = len(x_loc)
    x_dist_wel = np.array(
        [
            [np.abs(x_loc[i] - x_wel[j]) for j in range(len(x_wel))] 
            for i in range(N)
        ]
    )
    y_dist_wel = np.array(
        [
            [np.abs(y_loc[i] - y_wel[j]) for j in range(len(y_wel))] 
            for i in range(N)
        ]
    )
    y_wel = (y_wel)//dy
    x_wel = (x_wel)//dx
    y_loc_ind = np.array([y_wel[np.argmin(y_dist_wel, axis=1)[i]] for i in range(N)])
    x_loc_ind = np.array([x_wel[np.argmin(x_dist_wel, axis=1)[i]] for i in range(N)])
    
    return y_loc.astype(int), x_loc.astype(int)


## load the measurements, meas[:,0]: the measurements, meas[:, 1]: sigma for meas error.
with open('obs_sd.pkl', 'rb') as file:
    meas = pkl.load(file)
Par=AttrDict()
Par.obs = meas[:,0]; # observations
Par.sd  = meas[:,2]; # standard deviations of the observation error
Par.Nobs = meas.shape[0]
Par.Nlat = nf*d*h*w
Par.Nt = 10
Par.Nt_re = 5
Par.Npar = Par.Nlat + 2 + Par.Nt_re 

Par.N_iter = 20
Par.alpha = 0.1  # a scalar within [0 1]
Par.Ne = 60000
Par.beta = np.sqrt(Par.N_Iter)
Par.para_range = np.asarray(
    [
        [0, 0] + [100 for i in range(Par.Nt_re)] + [-5 for i in range(Par.Nlat)],
        [1200, 812.5] + [1000 for i in range(Par.Nt_re)] + [5 for i in range(Par.Nlat)]
    ]
)








