class ILUES:

    def ilues_select(self,Par, x, y):
        self.Par = Par
        Cd = np.eye(Par.Nobs)
        for i in range(Par.Nobs):
            Cd[i, i] = Par.sd[i]**2 #covariance matrix for the error

        meanxf = np.tile(np.mean(x1, axis=1, keepdims=True), (1, Par.Ne))
        Cm = np.matmul((x1 - meanxf), (x1 - meanxf).T)/(Ne - 1) # auto-covariance of the prior parameters
        
        J1 = np.zeros((Par.Ne,1))
        for i in range(Par.Ne):
            J1[i,] = np.matmul((yf[:,i]-Par.obs).T/Cd, (yf[:,i]-Par.obs))[0,0]

        xa = np.zeros(x1.shape)  # define the updated ensemble   
        for j in range(Par.Ne):
            xa[:,j] = self.local_update(xf,yf,Cm,sd,obs,alpha,beta,J1,j)
        return xa


    def local_update(self,xf,yf,Cm,sd,obs,alpha,beta,J1,jj):
        # The local updating scheme used in ILUES
        Ne = xf.shape[1]
        xr = xf[:, jj]
        xr = np.tile(np.reshape(xr, (-1, 1)), (1, Ne))
        J = np.matmul((xf-xr).T/Cm, (xf-xr))
        J2 = np.diag(J)
        J3 = J1/np.max(J1) + J2/np.max(J2)
        M = np.ceil(Ne*alpha)

        J3min, index = np.min(J3), np.unravel_index(np.argmin(J3, axis=None), J3.shape)
        xl = xf[:, index]
        yl = yf[:, index]
        alpha_ = J3min / J3
        alpha_[index] = 0
        index1 = self.RouletteWheelSelection(alpha_, M-1)
        xl1 = xf[:, index1]
        yl1 = yf[:, index1]
        xl = np.asarray([xl, xl1])
        yl = np.asarray([yl, yl1])
        xu =  self.update_para(xl,yl,self.Par.para_range,sd*beta,obs)
        a = time.time()
        np.random.seed(int(a))
        xest = xu[:,np.random.permutation(M)]
        return xest

    def RouletteWheelSelection(self,V,m):
        '''
        Input:
              V           -----fitness criterion
              m           -----number of individuals to be chosen
        Output:
              index       -----index of the chosen individuals
        '''
        n = V.shape[1]
        if np.max(V)==0 and np.min(V)==0:
            index=np.ceil(np.random.uniform(size=(1,m))*n)
        else:
            temindex= np.nonzero(V)
            n=len(temindex)
            V=V[temindex]

            V=np.cumsum(V)/np.sum(V)

            pp=np.random.uniform(size=(1,m))
            index = []
            for i in range(m):

                while True:
                    flag = True
                    for j in range(n):
                        if pp[i] < V[j]:
                            index.append(j)
                            V[j] = 0
                            flag = False
                            break

                    if flag:
                        pp[i] = np.random.uniform()
                    else:
                        break
        return np.array(index)


    def update_para(self,xf,yf,para_range, sd,obs):
        #Update the model parameters via the ensemble smoother
        para_range = se
        Npar = xf.shape[0]
        Ne = xf.shape[1]
        Nobs = len(obs)

        Cd = np.eye(Nobs)
        for i in range(Nobs):
            Cd[i,i] = sd(i)**2
        meanxf = np.tile(np.mean(xf, axis=1, keepdims=True), (1, Ne))
        meanyf = np.tile(np.mean(yf, axis=1, keepdims=True), (1, Ne))	
        Cxy = np.matmul((xf - meanxf), (yf - meanyf).T)/(Ne - 1)
        Cyy = np.matmul((yf - meanyf), (yf - meanyf).T)/(Ne - 1)

        kgain = linalg.lstsq((Cyy + Cd).T, Cxy.T)[0].T ##Cxy/(...), A/B = (B'\A')', b/a: linalg.lstsq(a.T, b.T)
        obse = np.tile(np.reshape(obs,(-1,1)),(1,Ne)) +\
          np.random.normal(np.zeros((Nobs,Ne)),np.tile(np.reshape(sd, (-1, 1)),(1,Ne)))
        xa = xf + np.matmul(kgain, (obse - yf))

        ##when the updated parameters exceed the range	
        for i in range(Ne):
            for j in range(Npar):
                if xa[j,i] > para_range[1, j]:
                    xa[j,i] = (para_range[1, j] + xf[j,i])/2
                elif xa[j,i] < para_range[0, j]:
                    xa[j,i] = (para_range[0, j] + xf[j,i])/2
        
        return xa