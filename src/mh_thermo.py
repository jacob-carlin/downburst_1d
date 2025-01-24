#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 16:18:33 2021

@author: jacob.carlin
"""

import numpy as np
import mh_namelist as nm

# Constants
Rd = nm.rd


def convertCtoK(tc):
    """Convert Celsius to Kelvin
    
    Output units: K"""
    
    return tc + 273.15


def virtual_temperature(tk, q):
    """ Calculate virtual temperature
    
    Output units: K"""
    
    return tk * (1 + (q / 0.622)) / (1 + q)


def density_air(p, t):
    """ Calculate air density
    
    Output units: kg/m3"""
    
    return p / (Rd * t)


def dynamic_viscosity_air(tk):
    """ Calculate dynamic viscosity of air 
    
    Output units: kg/m/s"""
    
    return (0.379565 + 0.0049 * tk) * 1e-5  


def thermal_diffusivity_air(tk):
    """ Calculate thermal diffusivity of air
    
    Output units: m2/s"""
    
    return 9.1018e-11 * tk**2 + 8.8197e-8 * tk - 1.0654e-5 


def thermal_diffusivity_water(tk, p):
    """ Calculate thermal diffusivity of water vapor
    
    Output units: m2/s"""
    
    return 2.11e-5 * ((tk / 273.15)**(1.94)) * (101325 / p)


def thermal_conductivity_air(tk):
    
    """ Calculate thermal conductivity of air
    
    Output units: J/m/s/K"""
    
    return (0.441635 + 0.0071 * tk) * 1e-2  


def thermal_conductivity_water(tk):
    
    """ Calculate thermal conductivity of water
    
    Output units: J/m/s/K"""
    
    return 0.568 * np.exp(3.473e-3 * (tk-273.15) - 3.823e-5 * (tk-273.15)**2 + 1.087e-6 * (tk-273.15)**3)


def sat_vapor_p(tc, i_flag=0):
    """
    Return saturation vapor wrt water/ice following Buck et al. (1996).

    Input:
        T [C]
        i_flag: =0 (w.r.t water; default), =1 (w.r.t ice)

    Output:
        es [Pa]
    """

    if i_flag == 1:
        es = 611.15 * np.exp((23.036 - (tc / 333.7)) * (tc / (279.82 + tc)))
    else:
        es = 611.21 * np.exp((18.678 - (tc / 234.5)) * (tc / (257.14 + tc)))

    return es


def vapor_mixing_ratio(e, p):
    """Calculate water vapor mixing ratio (saturated if e = es)
    
    Output units: kg/kg"""
    
    return 0.622 * e / (p - e)
