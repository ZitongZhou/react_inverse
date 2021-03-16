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

    
# print(sys.version)
# print('numpy version: {}'.format(np.__version__))
# print('matplotlib version: {}'.format(mpl.__version__))
# print('flopy version: {}'.format(flopy.__version__))


class mymf:
    """
    modflow model for multiple well pumping
    every values are represented as 2D np array
    """
    def __init__(self, dirname):
        self.dirname = dirname + '_files'
        workdir = os.path.join('.',)
        self.model_ws = os.path.join(workdir, self.dirname)
        
    def run_model(self, hk, c_spd,
                 exe_name_mf = '/Users/zitongzhou/Downloads/pymake/examples/mf2005',
                 exe_name_mt = '/Users/zitongzhou/Downloads/pymake/examples/mt3dms',
                 ):
        self.exe_name_mf = exe_name_mf
        self.exe_name_mt = exe_name_mt
        datadir = os.path.join('..', 'mt3d_test', 'mt3dms')
        workdir = os.path.join('.',)
    
        if os.path.isdir(self.dirname):
            shutil.rmtree(self.dirname, ignore_errors=True)

        Lx = 2500.  #meter
        Ly = 1250.
        nlay = 6
        nrow = 41
        ncol = 81
        ztop = 0.
        delr = Lx / (ncol-1)  # spacings along a row, can be an array
        delc = Ly / (nrow-1)  # spacings along a column, can be an array
    #        delv = (ztop - zbot) / nlay
        delv = 50
        prsity = 0.3

        perlen_mf = [365*4*10] #day
        perlen_mt = [365*4]*5 + [365*4]*5
    #     nper = len(perlen_mf)
        laytyp = 0. 
        rhob = 1.587 #bulk density of porous media g/m^3

        modelname_mf = self.dirname + '_mf'
        mf = flopy.modflow.Modflow(modelname=modelname_mf, model_ws=self.model_ws, exe_name=self.exe_name_mf)
        dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol,
                                       delr=delr, delc=delc, top=ztop, 
                                       botm=[-delv * k for k in range(1, nlay + 1)],
                                       nper=len(perlen_mf), perlen=perlen_mf)
        # Variables for the BAS package
        # active > 0, inactive = 0, or constant head < 0
        ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
        ibound[:, :, 0] = -1
        ibound[:, :, -1] = -1

        # initial head value also serves as boundary conditions
        strt = np.zeros((nlay, nrow, ncol), dtype=np.float32)
        h_grad = 0.012
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

        welspd = {}
        y_wel = np.array([6, 13, 20, 27, 34,
                6, 13, 20, 27, 34,
                6, 13, 20, 27, 34,
                6, 13, 20, 27, 34])# #np.random.randint(low = 5, high = 35, size = 20)
        x_wel = np.array([1, 1, 1, 1, 1, 
                 7, 7, 7, 7, 7,
                 13, 13, 13, 13, 13,
                 20, 20, 20, 20, 20])# #np.random.randint(low = 0, high = 25, size = 20)
        welspd[0] = [[3, y_wel[i], x_wel[i], 0.] for i in range(len(y_wel))]
                     
        wel = flopy.modflow.ModflowWel(mf, stress_period_data=welspd)
        spd = {(0, 0): ['save head', 'save budget']}
        oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)
        mf.write_input()
        mf.run_model(silent = True)

        modelname_mt = self.dirname + '_mt'
        mt = flopy.mt3d.Mt3dms(modelname=modelname_mt, model_ws=self.model_ws, 
                               exe_name=self.exe_name_mt, 
                               modflowmodel=mf, ftlfilename='mt3d_link.ftl')
        # nprs:, if 0, only save at the end of the stress period, 1 for head, 
        # 10 for concentration; if -1, save every timestep
        btn = flopy.mt3d.Mt3dBtn(mt, icbund=1, 
                                 prsity=prsity, sconc=0., 
                                 nper=len(perlen_mt), perlen=perlen_mt, nprs = 0)
        dceps = 1.e-9 # small Relative Cell Concentration Gradient below which advective transport is considered
        nplane = 1 #whether the random or fixed pattern is selected for initial placement of moving particles. If NPLANE = 0, the random pattern is selected for initial placement.
        npl = 0 #number of initial particles per cell to be placed at cells where the Relative Cell Concentration Gradient is less than or equal to DCEPS.
        nph = 16 #number of initial particles per cell to be placed at cells where the Relative Cell Concentration Gradient is greater than DCEPS. 
        npmin = 2
        npmax = 32
        nlsink = nplane #for sink cells
        npsink = nph
        mixelm = 0 # algorithm, -1: the third-order TVD scheme (ULTIMATE)
        adv = flopy.mt3d.Mt3dAdv(mt, mixelm=mixelm, dceps=dceps, nplane=nplane, 
                                 npl=npl, nph=nph, npmin=npmin, npmax=npmax,
                                 nlsink=nlsink, npsink=npsink, percel=0.5)

        al = 35. #meter
        trpt = 0.3
        trpv = 0.3
        #dmcoef: molecular diffusion, m2/d

        dsp = flopy.mt3d.Mt3dDsp(
            mt, al=al, trpt=trpt, trpv=trpv, 
            dmcoef=1.e-9, 
        )

        rct = flopy.mt3d.Mt3dRct(
            mt, isothm=2, ireact=0, igetsc=0, rhob=rhob, 
            sp1=0.1, sp2=0.9, 
        )

        ssm = flopy.mt3d.Mt3dSsm(mt, mxss=1500, stress_period_data=c_spd) #mxss: maximum number of all point sinks and sources
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
        
        ucnobj = flopy.utils.UcnFile(fname)
        
        hds = bf.HeadFile(os.path.join(self.model_ws, self.dirname + '_mf.hds'))
        times = hds.get_times()  # simulation time, steady state
        heads = hds.get_data(totim=times[-1])## steady state, save the last head map

        ##remove the binary files after running
        if os.path.isdir(self.dirname):
            shutil.rmtree(self.dirname, ignore_errors=True)
        conc = [ucnobj.get_data(totim=t) for t in ucnobj.get_times()]

        return conc, heads
    
    
    def plot_head(self, head, title = 'head'):
        head = np.flip(head, 0)
        self.simple_plot(head, title)
        
    
    def take_obs(self, cvt):
        self.obs =   [[4, 31, 49],
                     [4, 18, 56],
                     [4, 20, 48],
                     [4, 27, 76],
                     [4, 24, 60],
                     [4, 12, 31],
                     [4, 19, 69],
                     [4, 30, 32],
                     [4, 33, 38],
                     [4, 18, 65],
                     [4, 22, 74],
                     [4, 21, 38],
                     [4, 34, 55],
                     [4, 17, 44],
                     [4, 5, 66],
                     [4, 30, 59],
                     [4, 8, 52],
                     [4, 21, 60],
                     [4, 11, 40],
                     [4, 31, 40]]
        obss = [[self.obs[i][j]+1 for j in range(3)] for i in range(len(self.obs))]
        year = np.arange(2*365, 21*365, 2*365).tolist() + np.arange(20.5*365, 40*365+1, 365/2).tolist()
        def closest(lst, K): 
            lst = np.asarray(lst) 
            idx = (np.abs(lst - K)).argmin() 
            return idx
        inds = [closest(cvt['time'], year[i]) for i in range(len(year))]
        meas = [cvt[str(tuple(obss[i]))][inds] for i in range(len(obss))]
        return meas
    
    def make_movie(self, species='1', layer=0):
        fname = os.path.join(self.model_ws, 'MT3D00'+species+'.UCN')
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
            c_map = ucnobj.get_data(totim=time[i])[layer,]
            c01_map = plt.imshow(c_map, cmap='jet',
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  vmin=c_map.min(), vmax = c_map.max(),
                  origin='lower')
            plt.colorbar(c01_map, ax=axs, shrink=0.62)
            plt.title("year {}".format(i))
            plt.show()
            # Note that using time.sleep does *not* work here!
            sleep(0.1)
            if i < len(year)-1:
                clear_output(wait=True)
    
    def figures(self, species='1', layer = 0):
        fname = os.path.join(self.model_ws, 'MT3D00'+species+'.UCN')
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
            maps.append(ucnobj.get_data(totim=time[i])[layer,])
        return maps

def simple_plot(c_map, title=''):
    nx = 81
    ny = 41
    Lx = 2500
    Ly = 1250

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X,Y = np.meshgrid(x, y)
    if len(c_map) == 41:
        fig, axs = plt.subplots(1,1)
    #        axs.set_xlabel('x(m)')
    #        axs.set_ylabel('y(m)')
        # axs.set_xlim(0,Lx)
        # axs.set_ylim(0,Ly)
        c01map = axs.imshow(c_map, cmap='jet',
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  vmin=c_map.min(), vmax = c_map.max(),
                  origin='lower')
        fig.colorbar(c01map, ax=axs,shrink=0.62)
    else:
        fig, axs = plt.subplots(len(c_map)//3, 3, figsize=(7, 2.5))
        axs = axs.flat
        for i, ax in enumerate(axs):
            # ax.set_xlim(0,Lx)
            # ax.set_ylim(0,Ly)
            c01map = ax.imshow(c_map[i], cmap='jet', interpolation='nearest',
                      extent=[x.min(), x.max(), y.min(), y.max()],
                      vmin=c_map[i].min(), vmax = c_map[i].max(),
                      origin='lower')
            ax.set_axis_off()
            v1 = np.linspace(np.min(c_map[i]),np.max(c_map[i]), 5, endpoint=True)
            fig.colorbar(c01map, ax=ax, fraction=0.021, pad=0.04,ticks=v1,)

    plt.suptitle(title)
    name = title + '.pdf'
    plt.tight_layout()
#         fig.savefig('images/'+name, format='pdf',bbox_inches='tight')
    plt.show()
    return
        
if __name__ == '__main__':
    exe_name_mf = '/Users/zitongzhou/Downloads/pymake/examples/mf2005'
    exe_name_mt = '/Users/zitongzhou/Downloads/pymake/examples/mt3dms'
    os.chdir('/Users/zitongzhou/Desktop/react_inverse/TCP_3d')
    start = time.time()
    my_model = mymf('binary_files')

    spd = {}
    for i in range(5):
        spd[i] = [
            (3, 13, 20, 100., -1),
            ]
    spd[5] = [
        (3, 13, 20, 0., -1)
        ]

    with open('3dkd.pkl', 'rb') as file:
        hk = np.exp(pk.load(file))

    conc, heads = my_model.run_model(hk, spd)
    # my_model.plot_head()
    print(time.time() - start)
    # maps = my_model.figures()
    for i in range(len(conc)):
        title='conc time '+str(i)
        simple_plot(c_map=conc[i], title=title)
    title='head'
    simple_plot(c_map=heads, title=title)
    # my_model.simple_plot(maps[1],'')