# -*- coding: utf-8 -*-
"""
Author: Jacob T. Carlin
Contact: jacob.carlin@noaa.gov 

This file contains the scattering functions used in the one-dimensional
downdraft model.
"""

# Import relevant packages
import numpy as np
import mh_namelist as nm
import src.mh_microphysics as mp

# Deprecated imports... retaining for posterity
#from pytmatrix.tmatrix import Scatterer
#from pytmatrix import orientation, radar, tmatrix_aux, refractive

# Import necessary variables and constants from namelist
wave = nm.wave
eps_0 = nm.eps_0
t0 = nm.t0
pi = nm.pi
ew0 = nm.ew0
deld = nm.deld


def dielectric_water(t):
    """
    Calculate dielectric constant for fresh water at temperature T based on
    Ray (1972).

    Input:
        Temperature [K]
    Output:
        Dielectric constant
    """

    ew_eps_s = (78.54 * (1.0 - 4.579e-3 * (t - t0 - 25) +
                         1.19e-5 * (t - t0 - 25)**2 -
                          2.8e-8 * (t - t0 - 25)**3))
    ew_eps_inf = 5.27137 + 2.16474e-2 * (t - t0) - 1.31198e-3 * (t - t0)**2
    ew_alpha = (-16.8129 / t) + 6.09265e-2
    ew_lambda = 3.3836e-6 * np.exp(2513.98 / t)
    ew_sigma = 1.1117e-4
    ew_real = ew_eps_inf + (((ew_eps_s - ew_eps_inf) * (1 + (ew_lambda / (0.001 * wave))**(1 - ew_alpha) * np.sin(ew_alpha * np.pi / 2))) /
                                        (1 + 2 * (ew_lambda / (0.001 * wave))**(1 - ew_alpha) * np.sin(ew_alpha * np.pi/2) + (ew_lambda / (0.001 * wave))**(2 * (1 - ew_alpha))))
    ew_imag = (((ew_eps_s - ew_eps_inf) * ((ew_lambda / (0.001 * wave))**(1 - ew_alpha) * np.cos(ew_alpha * np.pi/2))) /
                                        (1 + 2*(ew_lambda / (0.001 * wave))**(1 - ew_alpha) * np.sin(ew_alpha * np.pi/2) + (ew_lambda / (0.001 * wave))**(2 * (1 - ew_alpha)))
                                        + (ew_sigma * (0.001 * wave)) / (2 * np.pi * 3e8 * eps_0))
    ew = complex(ew_real, ew_imag)

    return ew


def shape_factors(arm):
    """
    Return shape factors La and Lb for Rayleigh scattering calculations.
    Source: Ryzhkov et al. (2011)

    Input:
        Aspect ratio
    Output:
        La (Shape parameter for minor axis of oblate spheroid)
        Lb (Shape parameter for major axis of oblate spheroid)
    """

    x = np.sqrt((1.0 / arm)**2 - 1.0)
    la = np.where(x > 0,
                  ((1 + x**2) / x**2) * (1 - (np.arctan(x) / x)),
                  1.0 / 3.0)
    lb = np.where(x > 0,
                  0.5 * (1 - la),
                  1.0 / 3.0)

    return la, lb



def generate_scattering_tables(nbin):
    """
    Generate scattering amplitude look up tables using PyTMatrix

    Parameters
    ----------
    nbin : int
        Number of particle size bins

    Returns
    -------
    fhh_180 : Complex
        Horizontal backscattering complex amplitudes
    fvv_180 : Complex
        Vertical backscattering complex amplitudes
    fhh_0 : Complex
        Horizontal forwardscattering complex amplitudes
    fvv_0 : Complex
        Vertical forwardscattering complex amplitudes

    """

    ar_vec = np.arange(0.56, 1.01, 0.01)
    d_vec = deld * np.arange(nbin)
    d_vec[0] = 1e-3
    eps_vec = np.arange(3.18, 83.18) + 1j * np.maximum((np.arange(3.18, 83.18)*0.29745118 - 0.98009048), 0.00854)
    
    fhh_180 = np.zeros((len(d_vec), len(eps_vec), len(ar_vec)), dtype='complex')
    fvv_180 = np.zeros((len(d_vec), len(eps_vec), len(ar_vec)), dtype='complex')
    fhh_0 = np.zeros((len(d_vec), len(eps_vec), len(ar_vec)), dtype='complex')
    fvv_0 = np.zeros((len(d_vec), len(eps_vec), len(ar_vec)), dtype='complex')
    
    for ii in range(len(d_vec)):
        print('Diameter:', d_vec[ii])
        if ii == 0:
            fhh_180[ii, :, :] = 0.0
            fvv_180[ii, :, :] = 0.0
            fhh_0[ii, :, :] = 0.0
            fvv_0[ii, :, :] = 0.0
            continue            

        for jj in range(len(eps_vec)):
            #print('Epsilon:', eps_vec[jj])
            for kk in range(len(ar_vec)):
                # NOTE: This should be done on the entire array and reduce looping time...
                rp = d_vec[ii] * np.sqrt(np.absolute(eps_vec[jj])) / wave
                if rp < 0.3:
                    la, lb = shape_factors(ar_vec[kk])
                    fvv_180[ii, jj, kk] = fvv_0[ii, jj, kk] = (((pi**2 * (d_vec[ii])**3) / (6 * wave**2)) * (1 / (la + (1 / (eps_vec[jj] -1)))))
                    fhh_180[ii, jj, kk] = fhh_0[ii, jj, kk] = (((pi**2 * (d_vec[ii])**3) / (6 * wave**2)) * (1 / (lb + (1 / (eps_vec[jj] -1)))))
                else:
                    scatterer = Scatterer(radius=0.5 * d_vec[ii],
                                      wavelength=wave,
                                      m=np.sqrt(eps_vec[jj]),
                                      axis_ratio = 1.0 / ar_vec[kk],
                                      ndgs = 2)
                    scatterer.set_geometry(tmatrix_aux.geom_horiz_back)
                    fvv_180[ii, jj, kk] = scatterer.get_S()[0, 0]
                    fhh_180[ii, jj, kk] = -1.0*scatterer.get_S()[1, 1]
                    scatterer.set_geometry(tmatrix_aux.geom_horiz_forw)
                    fvv_0[ii, jj, kk] = scatterer.get_S()[0, 0]
                    fhh_0[ii, jj, kk] = scatterer.get_S()[1, 1]
                    
    return fhh_180, fvv_180, fhh_0, fvv_0



def calc_zh(fhh_180, fvv_180, ang2, ang4):
    """ Return the reflectivity at horizontal polarization of a single
    particle following Ryzhkov et al. (2011).

    Input:
        Backscattering amplitude along major axis [mm]
        Backscattering amplitude along minor axis [mm]
        2nd Angular Moment [unitless]
        4th Angular Moment [unitless]
    Output:
        Horizontal reflectivity per unit particle [mm6]
    """

    kw = (abs((ew0 - 1) / (ew0 + 2)))**2
    cz = (4.0 * wave**4)/(pi**4 * kw)

    zh = cz * ((np.absolute(fhh_180))**2 -
               2.0 * ang2 * np.real(np.conjugate(fhh_180) * (fhh_180 - fvv_180)) +
               ang4 * (np.absolute(fhh_180 - fvv_180))**2)

    return zh


def calc_zv(fhh_180, fvv_180, ang1, ang3):
    """ Return the reflectivity at vertical polarization of a single
    particle following Ryzhkov et al. (2011).

    Input:
        Backscattering amplitude along major axis [mm]
        Backscattering amplitude along minor axis [mm]
        1st Angular Moment [unitless]
        3rd Angular Moment [unitless]
    Output:
        Vertical reflectivity per unit particle [mm6]
    """
    
    kw = (abs((ew0 - 1) / (ew0 + 2)))**2
    cz = (4.0 * wave**4)/(pi**4 * kw)

    zv = cz * ((np.absolute(fhh_180))**2 -
               2.0 * ang1 * np.real(np.conjugate(fhh_180) * (fhh_180 - fvv_180)) +
               ang3 * (np.absolute(fhh_180 - fvv_180))**2)

    return zv


def calc_kdp(fhh_0, fvv_0, ang7):
    """ Return the specific differential phase shift of a single
    particle following Ryzhkov et al. (2011).

    Input:
        Forward-scattering amplitude along major axis [mm]
        Forward-scattering amplitude along minor axis [mm]
        7th Angular moment [unitless]
    Output:
        Specific differential phase per unit particle [deg/km]
    """

    ckdp = (0.18 / pi) * wave

    kdp = ckdp * ang7 * np.real(fhh_0 - fvv_0)

    return kdp


def calc_ah(fhh_0, fvv_0, ang2):
    """ Return the specific attenuation of a single particle following
    Ryzhkov et al. (2011).

    Input:
        Forward-scattering amplitude along major axis [mm]
        Forward-scattering amplitude along minor axis [mm]
        2nd Angular moment
    Output:
        Specific attenuation per unit particle [dB/km]
    """

    ca = 8.686E-3 * wave

    ah = ca * np.imag(fhh_0 - ang2 * (fhh_0 - fvv_0))

    return ah


def calc_adp(fhh_0, fvv_0, ang7):
    """ Return the specific differential attenuation of a single particle
    following Ryzhkov et al. (2011).

    Input:
        Forward-scattering amplitude along major axis [mm]
        Forward-scattering amplitude along minor axis [mm]
        7th Angular moment
    Output:
        Specific differential attenuation per unit particle [dB/km]
    """
    
    ca = 8.686E-3 * wave

    adp = ca * ang7 * np.imag(fhh_0 - fvv_0)

    return adp


def calc_delta(fhh_180, fvv_180, ang1, ang2, ang5):
    """ Return the backscatter differential phase shift of a single
    particle following Tromel et al. (2013).

    Input:
        Backscattering amplitude along major axis [mm]
        Backscattering amplitude along minor axis [mm]
        1st Angular moment [unitless]
        2nd Angular moment [unitless]
        5th Angular moment [unitless]
    Output:
        Backscatter differential phase shift per unit particle (deg)
    """

    kw = (abs((ew0 - 1) / (ew0 + 2)))**2
    cz = (4.0 * wave**4)/(pi**4 * kw)

    delta = cz * ((np.absolute(fhh_180))**2 +
             ang5 * (abs(fhh_180 - fvv_180))**2 -
             ang1 * np.conjugate(fhh_180) * (fhh_180 - fvv_180) -
             ang2 * fhh_180 * np.conjugate(fhh_180 - fvv_180))

    return delta

def calc_ldr(fhh_180, fvv_180, ang5):
    """ Return the linear depolarization ratio of a single
    particle following Ryzhkov et al. (2011).

    Input:
        Backscattering amplitude along major axis [mm]
        Backscattering amplitude along minor axis [mm]
        5th Angular Moment [unitless]
    Output:
        Depolarized reflectivity [dB]
    """

    kw = (abs((ew0 - 1) / (ew0 + 2)))**2
    cz = (4.0 * wave**4)/(pi**4 * kw)

    ldr = cz * (ang5 * (np.absolute(fhh_180 - fvv_180))**2)

    return ldr

