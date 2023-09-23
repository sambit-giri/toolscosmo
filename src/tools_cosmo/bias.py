"""

CALCULATE HALO BIAS

"""

import numpy as np

from .cosmo import growth_factor, variance


def halo_bias(param):
    """
    bias function adopted from Cooray&Sheth Eq.68
    """

    window = param.mf.window
    cc     = param.mf.c
    Om     = param.cosmo.Om
    rhoc   = param.cosmo.rhoc
    dc     = param.mf.dc
    p      = param.mf.p
    q      = param.mf.q

    rbin, var, dlnvardlnr = variance(param)

    if (window == 'tophat' or window == 'gaussian'):
        mbin = 4*np.pi*Om*rhoc*rbin**3/3

    elif (window == 'sharpk' or window == 'smoothk'):
        mbin = 4*np.pi*Om*rhoc*(cc*rbin)**3/3


    if (param.cosmo.zbin=='lin'):
        zz = np.linspace(param.cosmo.zmin,param.cosmo.zmax,param.cosmo.Nz)
    elif (param.cosmo.zbin=='log'):
        zz = np.logspace(np.log(param.cosmo.zmin),np.log(param.cosmo.zmax),param.cosmo.Nz,base=np.e)
    else:
        print("ERROR: neither lin nor log for binning. Abort")
        exit()

    Dz = growth_factor(zz, param)

    bias = []
    for i in range(len(zz)):
        dcz = dc/Dz[i]
        nu  = dcz**2.0/var

        if(param.code.bias=='spherical'):
            #spherical collapse
            bias += [1.0 + (nu - 1.0)/dc]

        elif(param.code.bias=='ellipsoidal'):
            #cooray and sheth
            e1 = (q*nu - 1.0)/dc
            E1 = 2.0*p/dc/(1.0 + (q*nu)**p)
            bias += [1.0 + e1 + E1]
        
        elif(param.code.bias=='tinker'):
            #tinker
            y = np.log10(200)
            A = 1+0.24*y*np.exp(-(4/y)**4)
            a = 0.44*y - 0.88
            B = 0.183
            b = 1.5
            C = 0.019 + 0.107*y+0.19*np.exp(-(4/y)**4)
            c = 2.4
            bias += [1-A*nu**(a/2)/(nu**(a/2) + dc**a) + B*nu**(b/2) + C*nu**(c/2)]

        elif(param.code.bias=='jing'):
            #Jing98
            par1 = 1 + (nu-1.0)/dc
            par2 = (1 + 1/(2*nu**4 - 1.0))**(0.06-0.02*0.96)
            bias += [par1*par2]

        else:
            print("ERROR: bias method does not exist!")
            exit()

    bias = np.array(bias)

    return mbin, zz, bias
