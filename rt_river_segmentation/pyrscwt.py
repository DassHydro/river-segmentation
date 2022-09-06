import math
import numpy as np
from scipy.fft import fft, ifft

def cwt(x, dt, dj=None, s0=None, j1=None, mother="morlet", param="6"):
    """Continuous wavelet transform
       Farge, M., 1992: Wavelet transforms and their applications to turbulence.
       Annu. Rev. Fluid Mech., 24, 395-457

    Parameters
    ----------
    x : numpy.ndarray
        Original signal series of size N.
    dt : float
        Sample spacing.
    dj : float, optional
        Spacing between discrete scales as used in the pycwt.cwt function. Default value is 0.4875 if dj==None.
    s0 : float, optional
        Smallest wavelet scale. Default value is 2*dt if s0==None. 
    j1 : integer, optional
        Number of scales. Default value is (log2(N dt/s0))/dj if j1==None. 
    mother : str
        Mother wavelet name. Possible choices are 'morlet', 'paul' or 'dog'. Default is 'morlet'.
    param : integer or float
        Mother wavelet parameter.
    

    Returns
    -------
    waves : numpy.ndarray
        Array of complex values of the wavelet transform.
    period : 
    sj : numpy.ndarray
        Computed scales.
    coi : numpy.ndarray
        Cone of influence.
    dj : float
        Actual spacing between discrete scales.
    freqs : numpy.ndarray
        Frequencies of the transform (positive range only).
    """
    
    # Compute default parameters if not provided
    if dj is None:
        dj = 0.4875
    if s0 is None:
        s0 = 2.0 * dt
    if j1 is None:
        j1 = int(math.floor(math.log2(len(x) * dt / s0) / dj))
    
    # Compute and remove bias
    bias = np.mean(x)
    x = x - bias
    
    n = len(x)
    
    # Construct frequencies
    freqs = np.arange(1, int(np.round(n / 2))+1) * (2.0 * np.pi) / (n * dt)
    freqs = np.concatenate(([0.], freqs, -freqs[::-1]))
    
    # Compute FFT
    f = fft(x)
    freqs = freqs[0:f.size]
    
    # Construct scales
    sj = s0 * 2.0**(np.arange(0, j1) * dj)
    
    # Computes waves
    waves = np.zeros((j1, n), dtype=complex)
    
    for i in range(0, j1):
        
        daughter, fmult, coi, _ = _wave_bases(mother, param, freqs, sj[i])
        waves[i, :] = ifft(f * daughter)

    # Compute the cone of influence
    coi = coi * dt * np.concatenate(([1E-5], np.arange(1,(n+1)/2), np.arange(1, n/2)[::-1],[1E-5]))

    # Compute periods
    period = fmult * sj
        
    return waves, period, sj, coi, dj, freqs
    
    

def icwt(W, sj, freqs, dt, dj=1/12, mother='morlet', param=-1):
    """Inverse continuous wavelet transform (Correction of the buggy icwt function of pywct). 
       Farge, M., 1992: Wavelet transforms and their applications to turbulence.
       Annu. Rev. Fluid Mech., 24, 395-457

    Parameters
    ----------
    W : numpy.ndarray
        Wavelet transform, the result of the pycwt.cwt function.
    sj : numpy.ndarray
        Vector of scale indices as returned by the pycwt.cwt function.
    dt : float
        Sample spacing.
    freqs : numpy.ndarray
        Vector of frequencies as returned by the pycwt.cwt function.
    dj : float, optional
        Spacing between discrete scales as used in the pycwt.cwt function. Default value is 0.25.
    mother : identifier of the mother wavelet
        Mother wavelet class. Default is Morlet
    

    Returns
    -------
    iW : numpy.ndarray
        Inverse wavelet transform.
    freqs : numpy.ndarray
        Frequencies of the transform (full range).
    """
    
    # CHECK-UP
    if not isinstance(mother, str):
        raise ValueError("'wavelet' must be 'morlet', 'paul' or 'dog'")
    elif mother == "morlet":
        if not isinstance(param, float) and not isinstance(param, int):
            raise ValueError("'param' must be a float for morlet wavelet")
        if param <= 0.:
            raise ValueError("'param' must be positive")
    elif mother == "paul":
        if not isinstance(param, int):
            raise ValueError("'param' must be an integer for paul wavelet")
        if param <= 0:
            raise ValueError("'param' must be positive")
    elif mother == "dog":
        if not isinstance(param, int):
            print(param, type(param))
            raise ValueError("'param' must be an integer for dog wavelet")
        if param <= 0:
            raise ValueError("'param' must be positive")
    else:
        raise ValueError("'mother' must be 'morlet', 'paul' or 'dog'")
        
    a, b = W.shape
    c = sj.size
    if a == c:
        sj = (np.ones([b, 1]) * sj).transpose()
    elif b == c:
        sj = np.ones([a, 1]) * sj
    else:
        raise Warning('Input array dimensions do not match.')
    
    Wr = np.real(W)
    summand = (np.real(W) / np.sqrt(sj)).sum(axis=0)
    
    k = np.arange(0, b//2)
    freqs = 2.0 * np.pi * k / (b * dt)
    freqs = np.concatenate((freqs, -freqs[::-1]))
    
    # Compute the Fourier spectrum at each scale (Eq 12 in [Farge, 1992])
    Wdelta = np.zeros(len(sj), dtype=complex)
    #print(len(sj))
    for i in range(0, len(sj)):
        
        daughter, _, _, _ = _wave_bases(mother, param, freqs, sj[i, 0])
        Wdelta[i] = (1.0/b) * np.sum(daughter)
        
    C = np.sum(np.real(Wdelta) / np.sqrt(sj[:, 0]))
    
    iW = summand / C
    iW = (iW - np.mean(iW)) / _tabulated_multiplier(mother, param)

    return iW, freqs


def _wave_bases(mother, param, freqs, scales):
    
    n = freqs.size
    norm = math.sqrt(freqs[1]) * math.sqrt(n)
    
    if mother == "morlet":
        f0 = param
        mul = math.pi**(-0.25) * norm
        expnt = -(scales * freqs - f0)**2 * 0.5 * (freqs > 0.)
        daughter = mul * np.sqrt(scales) * np.exp(expnt)
        daughter = daughter * (freqs > 0.)
        fmult = 4 * math.pi / (f0 + math.sqrt(2 + f0**2))
        coi = fmult / math.sqrt(2)
        dofmin = 2
    elif mother == "paul":
        m = param
        mul = np.sqrt(scales) * (2**m / math.sqrt(m * np.prod(np.arange(2, 2*m)))) * norm
        expnt = -(scales * freqs) * (freqs > 0.)
        daughter = mul * (( scales * freqs)**m) * np.exp(expnt)
        daughter = daughter * (freqs > 0.)
        fmult = 4 * math.pi / (2.0 * m + 1)
        coi = fmult * math.sqrt(2)
        dofmin = 2
    elif mother == "dog":
        m = param
        mul = -((1j)**m / math.sqrt(math.gamma(m + 0.5))) * np.sqrt(scales) * norm
        expnt = -(scales * freqs)**2 * 0.5
        daughter = mul * (( scales * freqs)**m) * np.exp(expnt)
        fmult = 2 * math.pi * math.sqrt(2. / (2 * m + 1))
        coi = fmult / math.sqrt(2);
        dofmin = 1
    else:
        raise ValueError("Wrong type for 'mother' : %s" % mother)
        
    return daughter, fmult, coi, dofmin

def _tabulated_multiplier(mother, param):
    
    if mother == "morlet":
        
        tabulated_par = np.linspace(1.0, 10, 37, endpoint=True)
        tabulated_mul = [9.3414, 7.7459, 6.3586, 5.1910, 4.2406,
                         3.4910, 2.9152, 2.4802, 2.1528, 1.9062,
                         1.7165, 1.5668, 1.4467, 1.3499, 1.2730,
                         1.2116, 1.1611, 1.1163, 1.0762, 1.0362,
                         1.0014, 0.9820, 0.9799, 0.9971, 1.0332,
                         1.0748, 1.1006, 1.0956, 1.0550, 1.0016,
                         0.8997, 0.8060, 0.7218, 0.6961, 0.7392, 
                         0.8171, 0.9192, 1.0350, 1.1008, 1.1464]
        tabulated_mul = tabulated_mul[0:tabulated_par.size]
    elif mother == 'paul':
        tabulated_par = np.linspace(1.0, 10, 37, endpoint=True)
        tabulated_mul = [5.2631, 4.2772, 3.5891, 3.0903, 2.7147,
                         2.4250, 2.1950, 2.0092, 1.8565, 1.7293, 
                         1.6224, 1.5317, 1.4542, 1.3877, 1.3302, 
                         1.2804, 1.2370, 1.1990, 1.1658, 1.1365, 
                         1.1107, 1.0879, 1.0675, 1.0494, 1.0330,
                         1.0183, 1.0050, 0.9927, 0.9813, 0.9708,
                         0.9609, 0.9516, 0.9428, 0.9344, 0.9262, 
                         0.9182, 0.9104, 0.9027, 0.8950, 0.8873]
        tabulated_mul = tabulated_mul[0:tabulated_par.size]
        
    elif mother == "dog":
        
        tabulated_par = np.linspace(2.0, 20, 10, endpoint=True)
        tabulated_mul = [4.2708, 2.8440, 2.2734, 1.9469, 1.7318,
                         1.5811, 1.4637, 1.3581, 1.2555, 1.1720]
        
    return np.interp(param, tabulated_par, tabulated_mul)
