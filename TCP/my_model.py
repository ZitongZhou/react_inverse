import sys
import os
import numpy as np
import matplotlib as mpl
import pickle as pk
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
import shutil
import time
from IPython.display import clear_output
from time import sleep

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join('..', '..'))
    sys.path.append(fpth)
    import flopy
    
from flopy.utils.util_array import read1d
mpl.rcParams['figure.figsize'] = (8, 8)

exe_name_mf = '/Users/zitongzhou/Downloads/pymake/examples/mf2005'
exe_name_mt = '/Users/zitongzhou/Downloads/pymake/examples/mt3dms'
datadir = os.path.join('..', 'mt3d_test', 'mt3dms')
workdir = os.path.join('.',)
    
print(sys.version)
print('numpy version: {}'.format(np.__version__))
print('matplotlib version: {}'.format(mpl.__version__))
print('flopy version: {}'.format(flopy.__version__))


class mymf:
    """
    modflow model for multiple well pumping
    every values are represented as 2D np array
    """
    def __init__(self,):
        self.dirname = 'binary_files'
        self.model_ws = os.path.join(workdir, self.dirname)
        
    def run_model(self, hk, welspd, c_spd):
        if os.path.isdir(self.dirname):
            shutil.rmtree(self.dirname, ignore_errors=True)

        mixelm = -1 # algorithm

        Lx = 2500.
        Ly = 1250.
        nlay = 1
        nrow = 41
        ncol = 81
        ztop = 0.
        zbot = -50*nlay
        delr = Lx / (ncol-1)  # spacings along a row, can be an array
        delc = Ly / (nrow-1)  # spacings along a column, can be an array
    #        delv = (ztop - zbot) / nlay
        delv = 50
        prsity = 0.3

        q0 = 100.
        c0 = 10000.

        perlen_mf = [365*2]*10 + [365/2]*40
        perlen_mt = [365*2]*10 + [365/2]*40
    #     nper = len(perlen_mf)
        laytyp = 0. 
        rhob = 1.587 #bulk density of porous media

        modelname_mf = self.dirname + '_mf'
        mf = flopy.modflow.Modflow(modelname=modelname_mf, model_ws= self.model_ws, exe_name=exe_name_mf)
        dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol,
                                       delr=delr, delc=delc, top=ztop, botm=zbot,nper=len(perlen_mf),
                                       perlen=perlen_mf)
        # Variables for the BAS package
        # active > 0, inactive = 0, or constant head < 0
        ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
        ibound[:, :, 0] = -1
        ibound[:, :, -1] = -1

        # initial head value also serves as boundary conditions
        strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
        h_grad = 0.0012
        l_head, r_head = h_grad*Lx, 0.
        strt[:, :, 0] = l_head
        strt[:, :, -1] = r_head
        # hk = np.dstack((hk,hk))
        # hk = np.dstack((hk,hk))
        # hk = np.swapaxes(hk, 0, 2)
        # hk = np.swapaxes(hk, 1, 2)
        bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
        lpf = flopy.modflow.ModflowLpf(mf, hk=hk, laytyp=laytyp)
        pcg = flopy.modflow.ModflowPcg(mf)
        lmt = flopy.modflow.ModflowLmt(mf, output_file_name='mt3d_link.ftl')

        wel = flopy.modflow.ModflowWel(mf, stress_period_data=welspd)

    #     spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
        spd = {(0, 0): ['save head', 'save budget']}
    #     oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)
        oc = flopy.modflow.ModflowOc(mf,compact=True)
        mf.write_input()
        mf.run_model(silent = True)

        modelname_mt = self.dirname + '_mt'
        mt = flopy.mt3d.Mt3dms(modelname=modelname_mt, model_ws=self.model_ws, 
                               exe_name=exe_name_mt, modflowmodel=mf, ftlfilename='mt3d_link.ftl')

        self.obs =   [[0, 31, 49],
                     [0, 18, 56],
                     [0, 20, 48],
                     [0, 27, 76],
                     [0, 24, 60],
                     [0, 12, 31],
                     [0, 19, 69],
                     [0, 30, 32],
                     [0, 33, 38],
                     [0, 18, 65],
                     [0, 22, 74],
                     [0, 21, 38],
                     [0, 34, 55],
                     [0, 17, 44],
                     [0, 5, 66],
                     [0, 30, 59],
                     [0, 8, 52],
                     [0, 21, 60],
                     [0, 11, 40],
                     [0, 31, 40]]
        btn = flopy.mt3d.Mt3dBtn(mt, icbund=1, prsity=prsity, sconc=0.,
                                 nper=len(perlen_mt), perlen=perlen_mt, nprs = -1, obs=self.obs)
        dceps = 1.e-5 # small Relative Cell Concentration Gradient below which advective transport is considered
        nplane = 1 #whether the random or fixed pattern is selected for initial placement of moving particles. If NPLANE = 0, the random pattern is selected for initial placement.
        npl = 0 #number of initial particles per cell to be placed at cells where the Relative Cell Concentration Gradient is less than or equal to DCEPS.
        nph = 16 #number of initial particles per cell to be placed at cells where the Relative Cell Concentration Gradient is greater than DCEPS. 
        npmin = 2
        npmax = 32
        nlsink = nplane #for sink cells
        npsink = nph

        adv = flopy.mt3d.Mt3dAdv(mt, mixelm=mixelm, dceps=dceps, nplane=nplane, 
                                 npl=npl, nph=nph, npmin=npmin, npmax=npmax,
                                 nlsink=nlsink, npsink=npsink, percel=0.5)

        al = 35.
        trpt = 0.3
        trpv = 0.03
        dsp = flopy.mt3d.Mt3dDsp(mt, al=al, trpt=trpt, trpv=trpv, dmcoef=1.e-9)#dmcoef: molecular diffusion
        lambda1 = 0.
        rct = flopy.mt3d.Mt3dRct(mt, isothm=2, ireact=1, igetsc=0, rhob=rhob, sp1=0.1,sp2 = 0.9, 
                             rc1=lambda1, rc2=lambda1)


        ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=c_spd)
        gcg = flopy.mt3d.Mt3dGcg(mt)
        # write mt3dms input
        while True:
            try:
                mt.write_input()
            except OSError as err:
                print("mt File writing error: %s" % (err))
            else:  # if we succeed, get out of the loop
                break

        fname = os.path.join(self.model_ws, 'MT3D001.UCN')
        if os.path.isfile(fname):
            os.remove(fname)

        mt.run_model(silent=True)

    #     fname = os.path.join(model_ws, 'MT3D001.UCN')
    #     ucnobj = flopy.utils.UcnFile(fname)
    #     times = ucnobj.get_times()
    #     conc = ucnobj.get_alldata()

        fname = os.path.join(self.model_ws, 'MT3D001.OBS')
        if os.path.isfile(fname):
            cvt = mt.load_obs(fname)
        else:
            cvt = None

        return cvt 
    
    def simple_plot(self, c_map, title):
        nx = 81
        ny = 41
        Lx = 2500
        Ly = 1250

        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X,Y = np.meshgrid(x, y)
        fig, axs = plt.subplots(1,1)
    #        axs.set_xlabel('x(m)')
    #        axs.set_ylabel('y(m)')
        axs.set_xlim(0,Lx)
        axs.set_ylim(0,Ly)
        c01map = axs.imshow(c_map, cmap='jet',
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  vmin=c_map.min(), vmax = c_map.max(),
                  origin='lower')
        fig.colorbar(c01map, ax=axs,shrink=0.62)
        name = title + '.pdf'
        plt.title(title)
#         fig.savefig('images/'+name, format='pdf',bbox_inches='tight')
        plt.show()
        return
    
    
    def head(self):
        hds = bf.HeadFile(os.path.join(self.model_ws, self.dirname + '_mf.hds'))
        times = hds.get_times()  # simulation time, steady state
        heads = hds.get_data(totim=times[-1])
        hds.close()  # close the file object for the next run

        head = heads[0]
        head = np.flip(head, 0)
        simple_plot(head, 'head')
        
    
    def take_obs(self, cvt):

        obss = [[self.obs[i][j]+1 for j in range(3)] for i in range(len(self.obs))]
        year = np.arange(2*365, 21*365, 2*365).tolist() + np.arange(20.5*365, 40*365+1, 365/2).tolist()
        def closest(lst, K): 
            lst = np.asarray(lst) 
            idx = (np.abs(lst - K)).argmin() 
            return idx
        inds = [closest(cvt['time'], year[i]) for i in range(len(year))]
        meas = [cvt[str(tuple(obss[i]))][inds] for i in range(len(obss))]
        return meas
    
    def make_movie(self):
        fname = os.path.join(self.model_ws, 'MT3D001.UCN')
        ucnobj = flopy.utils.UcnFile(fname)
        times = ucnobj.get_times()

        year = np.arange(2, 41)*365

        def close(lst, K): 
            lst = np.asarray(lst) 
            idx = (np.abs(lst - K)).argmin() 
            return lst[idx]

        time = [close(times, year[i]) for i in range(len(year))]
        nx = 81
        ny = 41
        Lx = 2500
        Ly = 1250

        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X,Y = np.meshgrid(x, y)

        fig, axs = plt.subplots()
        #        axs.set_xlabel('x(m)')
        #        axs.set_ylabel('y(m)')
        axs.set_xlim(0,Lx)
        axs.set_ylim(0,Ly)

        for i in range(len(year)):
            c_map = ucnobj.get_data(totim=time[i])[0,]
            c01_map = plt.imshow(c_map, cmap='jet',
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  vmin=c_map.min(), vmax = c_map.max(),
                  origin='lower')
#             fig.colorbar(c01_map, ax=axs, shrink=0.62)
            plt.title("year {}".format(i))
            plt.show()
            # Note that using time.sleep does *not* work here!
            sleep(0.1)
            clear_output(wait=True)
    
    def figures(self):
        fname = os.path.join(self.model_ws, 'MT3D001.UCN')
        ucnobj = flopy.utils.UcnFile(fname)
        times = ucnobj.get_times()
        year = np.arange(2*365, 21*365, 2*365).tolist() + np.arange(20.5*365, 40*365+1, 365/2).tolist()
        
        def close(lst, K): 
            lst = np.asarray(lst) 
            idx = (np.abs(lst - K)).argmin() 
            return lst[idx]

        time = [close(times, year[i]) for i in range(len(year))]
        maps = []
        for i in range(len(time)):
            maps.append(ucnobj.get_data(totim=time[i])[0,])
        return maps
        
if __name__ == '__main__':
    my_model = mymf()

    welspd = {}
    for i in range(10):
        welspd[i] = [0, 15, 25, 100]
    welspd[10] = [0,25, 25, 0]

    spd = {}
    for i in range(10):
        spd[i] = [0, 15, 25, 10000., 2]
    spd[10] = [0, 15, 25, 0., 2]

    with open('hk', 'rb') as file:
        hk = pk.load(file)

    cvt = my_model.run_model(hk, welspd, spd)
    meas = my_model.take_obs(cvt)
    maps = my_model.figures()
    my_model.make_movie()
    my_model.simple_plot(maps[1],'')