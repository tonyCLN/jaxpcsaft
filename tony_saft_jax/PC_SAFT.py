"""
Spyder Editor
# Author: Antonio Cavalcante de Lima Neto
# Github: https://github.com/tonyCLN
# Date: 14-10-2022
# Updated: 14-11-2022

References:
    
Gross, J., & Sadowski, G. (2001). 
Perturbed-Chain SAFT: An Equation of State 
Based on a Perturbation Theory for Chain Molecules
Joachim. Industrial and Engineering Chemistry Research, 40, 1244–1260.
 https://doi.org/10.1021/ie0003887
    
Gross, J., & Sadowski, G. (2019). Reply to Comment on 
“perturbed-Chain SAFT: An Equation of State Based on 
a Perturbation Theory for Chain Molecules.” Industrial 
and Engineering Chemistry Research, 58(14), 5744–
5745. https://doi.org/10.1021/acs.iecr.9b01515

Michelsen, M. L., & Hendriks, E. M. (2001). 
Physical properties from association models. 
Fluid Phase Equilibria, 180(1–2), 165–174. 
https://doi.org/10.1016/S0378-3812(01)00344-2

X association algorithm:
Tan, S. P., Adidharma, H., & Radosz, M. (2004).
Generalized Procedure for Estimating the Fractions of 
Nonbonded Associating Molecules and Their Derivatives in 
Thermodynamic Perturbation Theory. Industrial and Engineering 
Chemistry Research, 43(1), 203–208. https://doi.org/10.1021/ie034041q

Delta funtion:
Tan, S. P., Adidharma, H., & Radosz, M. (2004). Generalized Procedure 
for Estimating the Fractions of Nonbonded Associating Molecules and 
Their Derivatives in Thermodynamic Perturbation Theory. Industrial 
and Engineering Chemistry Research, 43(1), 203–208. 
https://doi.org/10.1021/ie034041q

"""
from jax.config import config
config.update("jax_enable_x64", True)
from functools import partial
from jax import jit as njit
from scipy.constants import k as kb  # constante de Boltzmann J /K
from scipy.constants import Avogadro as Navo  # numero de avogadro mol^-1
from scipy.constants import pi
from scipy import optimize
import numpy as nnp
# import numpy as np
from jax import numpy as np
# from jax import numpy as nnp
from jax import jacfwd
from jax import lax

class  PC_SAFT_jax():
    def __init__(self, m, sigma, epsilon_k, M=None, kbi=None, kAB_k=None, eAB_k=None, S=None,kbiasc=None):
        self.ncomp = len(m)
        self.M = M
        self.m = m
        self.sigma = sigma
        self.epsilon_k = epsilon_k
        
        if eAB_k is None:
            self.eAB_k = np.zeros([self.ncomp])
            self.kAB_k = np.zeros([self.ncomp])
            self.S = np.zeros([self.ncomp])
        
        else:
            self.eAB_k = eAB_k
            self.kAB_k = kAB_k
            self.S = S

        if kbi is None:
            self.kbi = np.zeros([self.ncomp, self.ncomp])
        else:
            self.kbi = kbi
            
        if kbiasc is None:
            self.kbiasc = np.zeros([self.ncomp, self.ncomp])
        else:
            self.kbiasc = kbiasc
            
            
        ap = np.zeros([7, 3])
        bp = np.zeros([7, 3])

        ap = np.array([[ 9.10563145e-01, -3.08401692e-01, -9.06148351e-02],
                [ 6.36128145e-01,  1.86053116e-01,  4.52784281e-01],
                [ 2.68613479e+00, -2.50300473e+00,  5.96270073e-01],
                [-2.65473625e+01,  2.14197936e+01, -1.72418291e+00],
                [ 9.77592088e+01, -6.52558853e+01, -4.13021125e+00],
                [-1.59591541e+02,  8.33186805e+01,  1.37766319e+01],
                [ 9.12977741e+01, -3.37469229e+01, -8.67284704e+00]])
        
        bp = np.array([[ 7.24094694e-01, -5.75549808e-01,  9.76883116e-02],
                [ 2.23827919e+00,  6.99509552e-01, -2.55757498e-01],
                [-4.00258495e+00,  3.89256734e+00, -9.15585615e+00],
                [-2.10035768e+01, -1.72154716e+01,  2.06420760e+01],
                [ 2.68556414e+01,  1.92672264e+02, -3.88044301e+01],
                [ 2.06551338e+02, -1.61826462e+02,  9.36267741e+01],
                [-50.8003365888685e0 * 7, -23.6010990650801e0 *7, -4.23812936930675e0 * 7]])
        
        self.ap = ap
        self.bp = bp

        return


    # EQ -- ok!
    def  dens(self, T, P, x, phase=None, opt=False, method=None,real=False):
        T = nnp.asarray(T)
        P = nnp.asarray(P)
        x = nnp.asarray(x)
        m = nnp.asarray(self.m)
        d_T =  nnp.asarray(   d_T(T, self.ncomp, self.sigma, self.epsilon_k))
        soma = 0
        # for liquid
        etaguessL = 0.5
        # for gas
        etaguessV = 1e-10
        for i in range(self.ncomp):
            soma += x[i]*m[i]*d_T[i]**3
            
        densL0 = 6/pi*etaguessL/soma*1e30/Navo
        densV0 = 6/pi*etaguessV/soma*1e30/Navo

        def residuo(dens):
            densi, = dens
            # res0 = 1 - (self. Pressure(densi, T, x))/P
            res0 = (((self. Pressure(densi, T, x))/P) -1)
            f = nnp.asarray([res0])
            return f
        # aqui otimiza
        if opt is True:
            def fobjL(dens):
                f = ((P - self. Pressure(dens, T, x)))**2 - min(0,1/(6/pi*(pi/(3/np.sqrt(2)))/soma*1e30/Navo)- dens)
                return f

            def fobjV(dens):
                f = ((P - self. Pressure(dens, T, x)))**2
                return f
            
            
            if phase is None or phase == 'liq':
                ans = optimize.minimize(fobjL, densL0, method=method )
                densL_1 = ans["x"][0]
                if phase == 'liq':
                    return densL_1

            if phase is None or phase == 'vap':
                ans = optimize.minimize(fobjV, densV0, method=method)
                densV_1 = ans["x"][0]
                if phase == 'vap':
                    return densV_1

        # aqui usa o solver
        else:
            def residuo_log(dens_ad):  # escalar

                densi = dens_ad[0]
                pcalc = self. Pressure(densi, T, x)
                res0 = np.log(pcalc / P)
                f = [res0]

                return f
            
            if method is None:
                method = 'hybr'

            if phase is None or phase == 'liq':
                ans = optimize.root(residuo, [densL0, ], method=method)
                densL_1 = ans["x"][0]

                if phase == 'liq':
                    return densL_1

            if phase is None or phase == 'vap':
                ans = optimize.root(residuo_log, [densV0, ], method=method,tol=None)
                densV_1_ad = ans["x"][0]
                densV_1 = densV_1_ad
                if phase == 'vap':
                    return densV_1
            
        return densL_1, densV_1

        if phase is None or phase == 'vap':
            ans = optimize.root(residuo, [densV0, ])
            densV_1_ad = ans["x"][0]
            densV_1 = densV_1_ad
            if phase == 'vap':
                return densV_1
            
        return densL_1, densV_1



    def  Psat(self, T, guessP):
        x = np.array([1])
        dens_L0, dens_V0 = self. dens(T, guessP, x)

        def residuo(var):
            Psat = var[0]
            dens_L = var[1]
            dens_V = var[2]

            Pl = self. Pressure(dens_L, T, x)
            Pv = self. Pressure(dens_V, T, x)
            phiL = self. phi(dens_L, T, x)
            phiV = self. phi(dens_V, T, x)

            res1 = (1-phiL/phiV)**2
            res2 = ((Pl-Psat)/Pl)**2
            res3 = ((Pv-Psat)/Pv)**2

            f = res1 + res2 + res3
            return f

        ans = optimize.minimize(
            residuo, [guessP, dens_L0, dens_V0], method='Nelder-Mead')
        Psat = ans["x"][0]
        dens_L = ans["x"][1]
        dens_V = ans["x"][2]

        return Psat, dens_L, dens_V

    def  Psat2(self, T, iguess_P):  # ,index):
        RES = 1
        TOL = 1e-7
        MAX = 100
        P = iguess_P
        i = 0
        while(RES > TOL and i < MAX):
            x = np.array([1.])
            dens_L, dens_V = self. dens(T, P, x)
            if np.abs(dens_L-dens_V) < 1e-9:
                print('solução trivial')
                return np.nan, -1
            phiL = self. phi(dens_L, T, x)
            phiV = self. phi(dens_V, T, x)
            P = P*(phiL/phiV)
            RES = np.abs(phiL/phiV-1.)
            i = i+10
        return P[0]

    # kg/m³
    def  massdens(self, T, P, x, phase=None, method=None, opt=False):

        M = self.M
        if phase is None:
            densl, densv = self. dens(T, P, x, method=method, opt=opt)
            massdens = np.zeros(2)
            for i in range(self.ncomp):
                massdens[0] += x[i]*M[i]*densl/1e+3
                massdens[1] += x[i]*M[i]*densv/1e+3
        else:
            dens = self. dens(
                T, P, x, phase=phase, method=method, opt=opt)
            massdens = 0
            for i in range(self.ncomp):
                massdens += x[i]*M[i]*dens/1e+3

        return massdens

    def  dens1phase(self,T,P,x):
        m = self.m
        d_T =    d_T(T, self.ncomp, self.sigma, self.epsilon_k)
        soma = 0
        etaguessL = 0.5
        for i in range(self.ncomp):
            soma += x[i]*m[i]*d_T[i]**3
        densL0 = 6/pi*etaguessL/soma*1e30/Navo
        
        def Q(var):
            dens = np.array([var[0]])
            dens0 =1
            nl=1
            
            Al = self. a_res(dens,T,x)*T*kb*Navo
            somx = 0
            for i in range(self.ncomp):
                somx += x[i]*np.log(x[i])
    
            f =  (Al*nl +kb*Navo*T*(somx)
                  -kb*Navo*T*(np.log(dens0/dens)*nl ) +  
                  P*(nl/dens) )
    
            return f
        
        def jacQ(var):
            jacQ = jacfwd(Q)(var)
            return jacQ[0]
        
        def hessQ(var):
            hessQ = jacfwd(jacQ)(var)
            return hessQ[0]
        
        densmax = densL0
        densmin = P/(kb*Navo*T)
        bnds = [((densmin),(densmax))]
        ans = optimize.minimize(Q, densL0,jac=jacQ,hess=hessQ,method='trust-constr',bounds=bnds)
        densL_1 = ans["x"][0]
        return densL_1
    
    def  XA(self,dens,T,x):
        XA=  X_tan(dens, T, x, self.ncomp, self.sigma, self.epsilon_k, self.m, self.kAB_k, self.eAB_k, self.S,self.kbiasc)
        return XA
    
    def  a_res(self,dens,T,x):
        ares =  a_res(dens,T,x, self.ap, self.bp, self.ncomp, self.sigma, self.epsilon_k, self.m, self.kbi, self.kAB_k, self.eAB_k, self.S,self.kbiasc)
        return ares
    
    def  mu_res_kT_autoeZ_res(self,dens,T,x):
        mu,Z_res =  mu_res_kT_autoeZ_res(dens,T,x, self.ap, self.bp, self.ncomp, self.sigma, self.epsilon_k, self.m, self.kbi, self.kAB_k, self.eAB_k, self.S,self.kbiasc)
        return mu,Z_res
    
    def  phi_auto(self,dens,T,x):
        phi =  phi(dens,T,x, self.ap, self.bp, self.ncomp, self.sigma, self.epsilon_k, self.m, self.kbi, self.kAB_k, self.eAB_k, self.S,self.kbiasc)
        return phi
    
    def  Pressure(self,dens,T,x):
        P =  Pressure(dens,T,x, self.ap, self.bp, self.ncomp, self.sigma, self.epsilon_k, self.m, self.kbi, self.kAB_k, self.eAB_k, self.S,self.kbiasc)
        return P
    
# EQ A.9 ok!
@partial(njit, static_argnames=['ncomp'] ) #https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
                                           #https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
def  d_T( T, ncomp, sigma, epsilon_k):
    
    d_T = sigma*(1.0-0.12*np.exp(-3.*epsilon_k/(T)))

    return d_T
    
# EQ A.8 ok!
@partial(njit, static_argnames=['ncomp'] )
def  csi(dens, T, x,ncomp,sigma,epsilon_k,m):
    
    d_T =  d_T(T, ncomp, sigma, epsilon_k)
    rho =  rho(dens)

    powd = np.stack([d_T**0,d_T,d_T**2,d_T**3],axis=1)
    csi = (x*m)@powd*pi*rho/6
    
    return csi

# EQ A.21 ok!
@partial(njit)
def  rho(dens):
    
    rho = dens*Navo/1.0e30
    
    return rho

@partial(njit, static_argnames=['ncomp'])
def  dij(T, ncomp, sigma, epsilon_k):
    d_T =   d_T(T, ncomp, sigma, epsilon_k)
    
    dij = d_T.reshape((ncomp,1))*d_T /(d_T.reshape((ncomp,1))+d_T)
    
    return dij
    
# EQ A.7 ok!
@partial(njit, static_argnames=['ncomp'])
def  ghs( dens, T, x, ncomp, sigma, epsilon_k,m):
    
    csi =    csi(dens, T, x, ncomp, sigma, epsilon_k,m)
    dij =      dij(T,ncomp, sigma, epsilon_k)
    
    ghs = 1/(1-csi[3]) + dij*3*csi[2]/( 1-csi[3])**2 + ((dij)**2)*2*csi[2]**2/(1-csi[3])**3

    return ghs

# EQ A.26 ok!
@partial(njit, static_argnames=['ncomp'])
def  Zhs(dens, T, x, ncomp, sigma, epsilon_k,m):
    
    csi =    csi(dens, T, x, ncomp, sigma, epsilon_k,m)

    P1 = csi[3]/(1-csi[3])
    P2 = 3*csi[1]*csi[2]/(csi[0]*(1-csi[3])**2)
    P3 = (3*csi[2]**3 - csi[3]*csi[2]**3)/(csi[0]*(1-csi[3])**3)
    Zhs = P1+P2+P3

    return Zhs

# EQ A.5 ok!
@partial(njit)
def  mmed(x,m):

    mmed = sum(x*m)

    return mmed

# EQ A.6 ok!
@partial(njit, static_argnames=['ncomp'])
def  a_hs( dens, T, x,ncomp,sigma,epsilon_k,m):
    
    csi =    csi(dens, T, x, ncomp, sigma, epsilon_k,m)

    a_hs = (3*csi[1]*csi[2]/(1-csi[3]) + csi[2]**3/(csi[3]*(1-csi[3])
            ** 2) + (csi[2]**3/csi[3]**2 - csi[0])*np.log(1-csi[3]))/csi[0]

    return a_hs

# EQ A.4 ok!
@partial(njit, static_argnames=['ncomp'])
def  a_hc( dens, T, x,ncomp,sigma,epsilon_k,m):
    
    mmed =  mmed(x,m)
    ghs =   ghs(dens, T, x, ncomp, sigma, epsilon_k,m)
    a_hs =   a_hs(dens, T, x, ncomp, sigma, epsilon_k,m)

    soma = sum(-x*(m-1)*np.log(ghs.diagonal()))
    a_hc = mmed*a_hs + soma

    return a_hc

# EQ A.18 AND A.19  ok!
@partial(njit)
def  a_e_b(x,ap,bp,m):

    mmed =  mmed(x,m)

    a = ap[:, 0] + (mmed-1)*ap[:, 1]/mmed + \
        (1-1/mmed)*(1-2/mmed)*ap[:, 2]
    b= bp[:, 0] + (mmed-1)*bp[:, 1]/mmed + \
        (1-1/mmed)*(1-2/mmed)*bp[:, 2]

    return a, b

# A.16 and A.17 ok!
@partial(njit, static_argnames=['ncomp'])
def  I1_e_I2( dens, T, x,ap,bp,ncomp,sigma,epsilon_k,m):
    
    a, b =   a_e_b(x,ap,bp,m)
    eta =    csi(dens, T, x, ncomp, sigma, epsilon_k,m)[3]
    i = nnp.arange(7)

    I1 = sum(a*eta**i)
    I2 = sum(b*eta**i)

    return I1, I2

# EQ A.11 ok! -> its not inverted in the paper
@partial(njit, static_argnames=['ncomp'])
def  C1( dens, T, x,ncomp,sigma,epsilon_k,m):
    
    mmed =  mmed(x,m)
    eta =    csi(dens, T, x, ncomp, sigma, epsilon_k,m)[3]

    C1 = (1 + mmed*(8*eta-2*eta**2)/(1-eta)**4 + (1-mmed)*(20*eta -
          27*eta**2 + 12*eta**3 - 2*eta**4)/((1-eta)*(2-eta))**2)**-1

    return C1

# EQ A.31 ok!
@partial(njit, static_argnames=['ncomp'])
def  C2( dens, T, x,ncomp,sigma,epsilon_k,m):
    
    C1 =   C1(dens, T, x, ncomp, sigma, epsilon_k,m)
    mmed =  mmed(x,m)
    eta =    csi(dens, T, x, ncomp, sigma, epsilon_k,m)[3]

    C2 = -C1**2*(mmed*(-4*eta**2 + 20*eta + 8)/(1-eta)**5 + (1-mmed)
                 * (2*eta**3+12*eta**2-48*eta+40)/((1-eta)*(2-eta))**3)

    return C2

# EQ A.14 ok!
@partial(njit, static_argnames=['ncomp'])
def  MAT_sigma(x,ncomp,sigma):

    MAT_sigma = (sigma.reshape((ncomp,1))+sigma)/2

    return MAT_sigma

# EQ A.15 ok!
@partial(njit, static_argnames=['ncomp'])
def  MAT_epsilon_k(x,ncomp,epsilon_k,kbi):

    MAT_epsilon_k = (epsilon_k.reshape((ncomp,1))*epsilon_k)**(1/2)*(1-kbi)

    return MAT_epsilon_k

# EQ A.12 and A.13 ok!
@partial(njit, static_argnames=['ncomp'])
def  m_2esig_3_e_m_2e_2sig_3(T, x,ncomp,sigma,epsilon_k,kbi,m):

    MAT_epsilon_k =   MAT_epsilon_k( x, ncomp, epsilon_k,kbi)
    MAT_sigma =   MAT_sigma( x, ncomp, sigma)
    
    m_2esig_3 = np.einsum('i,j,i,j,ij,ij-> ', x,x,m,m,(MAT_epsilon_k/T),MAT_sigma**3)
    m_2e_2sig_3 = np.einsum('i,j,i,j,ij,ij-> ', x,x,m,m,(MAT_epsilon_k/T)**2,MAT_sigma**3)

    return m_2esig_3, m_2e_2sig_3

# EQ A.10 ok!
@partial(njit, static_argnames=['ncomp'])
def  a_disp(dens, T, x,ap,bp,ncomp,sigma,epsilon_k,m ,kbi):
    
    I1, I2 =   I1_e_I2(dens, T, x,ap,bp,ncomp,sigma,epsilon_k,m)
    m_2esig_3, m_2e_2sig_3 =   m_2esig_3_e_m_2e_2sig_3(T, x, ncomp, sigma, epsilon_k, kbi,m)
    rho =    rho(dens)
    mmed =  mmed(x,m)
    C1 =   C1(dens, T, x, ncomp, sigma, epsilon_k,m)

    a_disp = -2*pi*rho*I1*m_2esig_3 - pi*rho*mmed*C1*I2*m_2e_2sig_3
    
    return a_disp

@partial(njit, static_argnames=['ncomp'])
def  eAiBj_k(ncomp,eAB_k):

    eAiBj_k = (eAB_k.reshape((ncomp,1))+eAB_k)/2

    return eAiBj_k

@partial(njit, static_argnames=['ncomp'])
def  kAiBj_k(x, ncomp, sigma, kAB_k,kbiasc):
    
    MAT_sigma =   MAT_sigma( x, ncomp, sigma)
    
    kAB_kij = np.sqrt(kAB_k.reshape((ncomp,1))*kAB_k)
    MAT_sigmaii = (np.sqrt(MAT_sigma.diagonal().reshape((ncomp,1))*MAT_sigma.diagonal())/((MAT_sigma.T+MAT_sigma)/2))**3*(1-kbiasc)
    kAiBj_k = kAB_kij*MAT_sigmaii
    
    return kAiBj_k

@partial(njit, static_argnames=['ncomp'])
def  delt(dens, T, x, ncomp, sigma, epsilon_k, m, kAB_k, eAB_k, S,kbiasc):
    nsite = len(S)
    MAT_sigma =   MAT_sigma( x, ncomp, sigma)
    eAiBj_k =   eAiBj_k( ncomp, eAB_k)
    kAiBj_k =   kAiBj_k(x, ncomp, sigma, kAB_k,kbiasc)
    ghs =   ghs(dens, T, x, ncomp, sigma, epsilon_k,m)
    invkro = nnp.ones((nsite, ncomp, nsite, ncomp))
    i = nnp.arange(nsite)
    invkro[i,:,i,:] = 0
    
    sum1 = (MAT_sigma**3*ghs*(np.exp(eAiBj_k/T) - 1 )*kAiBj_k)
    B = np.expand_dims(sum1, [0,2])
    delt =np.tile(B,(nsite,1,nsite,1))*invkro
    
    return delt

@partial(njit, static_argnames=['ncomp'])
def  X_tan(dens, T, x, ncomp, sigma, epsilon_k, m, kAB_k, eAB_k, S,kbiasc):
    delta =  delt(dens, T, x, ncomp, sigma, epsilon_k, m, kAB_k, eAB_k, S,kbiasc)
    nsite = len(S)
    rho =    rho(dens)
    X_A = nnp.ones([nsite, ncomp])*0.5
    X_A_old = nnp.ones([nsite, ncomp])*(-1)
    it = 0

    def itX_A(args): #body fun
        X_A = args[1]
        it=args[2]
        X_A_old = X_A*1
        sum1 = np.einsum('k,lk,lk,lkji-> ji', x,S,X_A_old,delta)
        X_A = 1./(1. + rho*sum1)
        it = it+1
        return [X_A_old,X_A,it]
    
    def cond_fun(args):
        X_A_old = args[0]
        X_A = args[1]
        it=args[2]
        resmin = 3.e-16
        itMAX = 1e+3
        dif = ((X_A-X_A_old)/X_A)**2
        res = nnp.max(dif)*1
        cond1 = res>resmin
        cond2 = it<itMAX
        return np.logical_and(cond1,cond2) 
    
    _,X_A,_ = lax.while_loop(cond_fun, itX_A, [X_A_old,X_A,it])
    return X_A

@partial(njit, static_argnames=['ncomp'])
def  a_asc( dens, T, x, ncomp, sigma, epsilon_k, m, kAB_k, eAB_k, S,kbiasc):

    X_A =   X_tan(dens, T, x, ncomp, sigma, epsilon_k, m, kAB_k, eAB_k, S,kbiasc)
    
    s1 = sum((np.log(X_A) - X_A/2 + 0.5 )*S)
    a_ass = sum(x*(s1))
    
    return a_ass

# EQ A.3 ok!
@partial(njit, static_argnames=['ncomp'])
def  a_res(dens, T, x, ap, bp, ncomp, sigma, epsilon_k, m, kbi, kAB_k, eAB_k, S,kbiasc):

    a_res =   a_hc(dens, T, x, ncomp, sigma, epsilon_k,m) +   a_disp(dens, T, x, ap,bp,ncomp,sigma,epsilon_k,m,kbi) +   a_asc(dens, T, x, ncomp, sigma, epsilon_k, m, kAB_k, eAB_k, S,kbiasc)
    
    return a_res

@partial(njit, static_argnames=['ncomp'])
def  mu_res_kT_autoeZ_res(dens, T, x, ap, bp, ncomp, sigma, epsilon_k, m, kbi, kAB_k, eAB_k, S,kbiasc):
    ares =  a_res(dens, T, x, ap, bp, ncomp, sigma, epsilon_k, m, kbi, kAB_k, eAB_k, S,kbiasc)
    jaca_res = jacfwd( a_res,argnums=(0,2))(dens, T, x, ap, bp, ncomp, sigma, epsilon_k, m, kbi, kAB_k, eAB_k, S,kbiasc)
    
    soma = np.einsum('j,j->', x,jaca_res[1])
    Z_res = dens*jaca_res[0]
    mu = jaca_res[1] -soma + Z_res + ares
    
    return mu,Z_res

# EQ A.32 ok!
@partial(njit, static_argnames=['ncomp'])
def  phi(dens, T, x, ap, bp, ncomp, sigma, epsilon_k, m, kbi, kAB_k, eAB_k, S,kbiasc):
    mu,Z_res =  mu_res_kT_autoeZ_res(dens, T, x, ap, bp, ncomp, sigma, epsilon_k, m, kbi, kAB_k, eAB_k, S,kbiasc)

    lnphi = mu - np.log(Z_res+1)

    phi = np.exp(lnphi)
    return phi

@partial(njit, static_argnames=['ncomp'])
def  Z(dens, T, x, ap, bp, ncomp, sigma, epsilon_k, m, kbi, kAB_k, eAB_k, S,kbiasc):

    da_drho = jacfwd( a_res,argnums=0)(dens,T,x, ap, bp, ncomp, sigma, epsilon_k, m, kbi, kAB_k, eAB_k, S,kbiasc)

    Z = 1+dens*da_drho

    return Z

@partial(njit, static_argnames=['ncomp'])
# EQ A.23 ok! -> there is no avogadro number in the paper, only volume convertion in intead
def  Pressure(dens, T, x, ap, bp, ncomp, sigma, epsilon_k, m, kbi, kAB_k, eAB_k, S,kbiasc):
    
    Z =  Z(dens, T, x, ap, bp, ncomp, sigma, epsilon_k, m, kbi, kAB_k, eAB_k, S,kbiasc)
    
    P = Z*kb*T*dens*Navo
    
    return P