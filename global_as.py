import numpy as np
import numpy.random as R
from scipy.stats import qmc
import chaospy as chaospy
from scipy.stats import norm

# Calculate the matrix C
def dFunc0(x,Func,M2,shiftedSobol=True,distribution='normal'):
    chi,dim = x.shape[0],x.shape[1]
    temp = np.zeros((dim,chi*M2))

    if(shiftedSobol == True):
        rand_add = qmc.Sobol(dim, scramble=False).random(M2+1)[1:]        
        for i in range(M2):       
            xe = np.tile(x.transpose(), dim).transpose()*(1-np.repeat(np.array(np.identity(dim)), chi, axis=0))
            if distribution == 'normal':
                rand = norm.ppf((norm.cdf(x)+rand_add[i])%1)
            elif distribution == 'uniform':
                rand = (x+rand_add[i])%1
            xe += rand.T.reshape(dim*chi,1)*np.repeat(np.array(np.identity(dim)), chi, axis=0)
            temp[:,i*(chi):(i+1)*(chi)] = ((Func(xe)-np.tile(Func(x),dim))/np.sum(xe-np.tile(x.transpose(),dim).transpose(),axis=1)).reshape(dim,chi)
            
    else:
        for i in range(M2):       
            xe = np.tile(x.transpose(), dim).transpose()*(1-np.repeat(np.array(np.identity(dim)), chi, axis=0))
            if distribution == 'normal':
                rand = R.normal(0,1,(chi,dim))
            elif distribution == 'uniform':
                rand = R.uniform(0,1,(chi,dim))
            xe += rand.T.reshape(dim*chi,1)*np.repeat(np.array(np.identity(dim)), chi, axis=0)
            temp[:,i*(chi):(i+1)*(chi)] = ((Func(xe)-np.tile(Func(x),dim))/np.sum(xe-np.tile(x.transpose(),dim).transpose(),axis=1)).reshape(dim,chi)
    
    return temp

# Calculate eigenvalue decomposition
def GAS(Func,dim,chi,M1,M2,shiftedSobol=True,distribution='normal'):
    if distribution == 'normal':
        z = R.normal(0, 1, (M1, dim))
    elif distribution == 'uniform':
        z = R.uniform(0, 1, (M1, dim))
    deriv = dFunc0(z,Func,M2,shiftedSobol,distribution=distribution)
    deriv /= np.sqrt(chi)
    u, s, vh = np.linalg.svd(deriv.astype(float), full_matrices=True) 
    s = s**2
    return u, s

# Estimate \Gamma_i's
def compute_C_u_1(x,Func,u,M2):
    '''
    ### Explanation of finding the mean of conditional expectation
    # Find direction v1
    # v1*u1 + ...+ vn*un = 0
    # av1 = z1 + bu1
    # av2 = z2 + bu2
    # ...
    # avn = zn + bun
    # z1^2 + ... + zn^2 = b^2 + a^2
    # a, b, v1,...,vn are unknown
    '''
    expect = []
    f = Func(x)
    qmc_sequence = qmc.Sobol(1, scramble=False).random(M2+1)[1:]
    for j in range(x.shape[1]):
        expecttemp = 0
        for i in range(M2):
            dist = (norm.ppf((norm.cdf(np.dot(x,u[:,j:(j+1)]))+qmc_sequence[i])%1) - np.dot(x,u[:,j:(j+1)]))
            x_mod = x + dist*u[:,j:(j+1)].transpose()
            f_mod = Func(x_mod)
            expecttemp += np.sum(((f_mod - f)/(dist.flatten()))**2)
        expect.append(expecttemp/M2/x.shape[0])
    return expect


# Estimate the matrix Phi
def Phi_est(X, expo, coef, P):
    Num = X.shape[0]
    dim = X.shape[1]
    return np.dot(np.prod((np.tile(X, P)**expo.flatten()).reshape((Num*P, dim)),axis = 1).reshape((Num, P))
               ,np.array(coef)).T

def Funy(z,z1,u,Func):
    dim = u.shape[0]
    dim1 = z.shape[1]
    chi = z.shape[0]
    f1 = Func(np.dot(z,u[:,:dim1].transpose())+np.dot(z1,u[:,dim1:].transpose()))
    return f1

# Estimate k hat
def estimate_k_reg1(z,z1,u, exponents, coefficients, P, Func):   
    phi = Phi_est(z, exponents, coefficients,P)
    khat = np.linalg.inv(phi @ np.transpose(phi)) @ phi @ Funy(z,z1,u,Func)
    return khat

# Estimate the expectation from PCE
def GAS_PCE(Func, Num_exp, z1, dim1, u, exponents, coefficients, P): 
    PCE = np.zeros(Num_exp)
    for i in range(Num_exp):   
        kii = estimate_k_reg1(z1[i][:,:dim1], z1[i][:,dim1:], u, exponents, coefficients, P, Func)
        PCE[i] = kii[0]    
    return PCE