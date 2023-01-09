import numpy as np
import numpy.random as R
from scipy import stats
from scipy.stats import qmc

### Calculate the matrix C
def dFunc0(x,Func,M2,shiftedSobol=True):
    chi,dim = x.shape[0],x.shape[1]
    temp = np.zeros((dim,chi*M2))

    if(shiftedSobol == True):
        rand_add = qmc.Sobol(dim, scramble=False).random(M2+1)[1:]        
        for i in range(M2):       
            xe = np.tile(x.transpose(), dim).transpose()*(1-np.repeat(np.array(np.identity(dim)), chi, axis=0))
            rand = (x+rand_add[i])%1
            xe += rand.T.reshape(dim*chi,1)*np.repeat(np.array(np.identity(dim)), chi, axis=0)
            temp[:,i*(chi):(i+1)*(chi)] = ((Func(xe)-np.tile(Func(x),dim))/np.sum(xe-np.tile(x.transpose(),dim).transpose(),axis=1)).reshape(dim,chi)
            
    else:
        for i in range(M2):       
            xe = np.tile(x.transpose(), dim).transpose()*(1-np.repeat(np.array(np.identity(dim)), chi, axis=0))
            rand = R.uniform(0,1,(chi,dim))
            xe += rand.T.reshape(dim*chi,1)*np.repeat(np.array(np.identity(dim)), chi, axis=0)
            temp[:,i*(chi):(i+1)*(chi)] = ((Func(xe)-np.tile(Func(x),dim))/np.sum(xe-np.tile(x.transpose(),dim).transpose(),axis=1)).reshape(dim,chi)
    
    return temp/M2


### Calculate eigenvalue decomposition
def GAS(Func,dim,chi,M1,M2=10,shiftedSobol):
    z = R.uniform(0, 1, (int(chi/M1), dim))
    deriv = dFunc0(z,Func,M2,shiftedSobol)
    deriv /= np.sqrt(chi)
    u, s, vh = np.linalg.svd(deriv.astype(float), full_matrices=True) 
    s = s**2
    return u, s