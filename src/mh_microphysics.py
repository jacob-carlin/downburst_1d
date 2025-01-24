# -*- coding: utf-8 -*-
"""
Author: Jacob T. Carlin
Contact: jacob.carlin@noaa.gov 

This file contains the microphysical functions used in the one-dimensional
downdraft model.
"""

# Impact required packages
import numpy as np
from scipy import optimize
import src.mh_thermo as th

# Constants (copied from /scripts/mh_namelist)
t0 = 273.15                 # Celsius --> Kelvin
g = 9.81                    # Gravitational acceleration [m/s2]
rv = 461.5                  # Gas constant for water vapor [J/kg/K]
rd = 287.0                  # Gas constant for dry air [J/kg/K]
ras = 1.292                 # Air density at sea level [kg/m3]
es0 = 611.0                 # Saturation pressure of water vapor at 0C [kg/m/s2]
cp = 1005                   # Specific heat of air at constant pressure [J/kg/K]
ci = 2108                   # Specific heat of ice at constant pressure [J/kg/K]
cw = 4187                   # Specific heat of water at constant pressure [J/kg/K]
lf = 3.335e5                # Enthalpy of fusion [J/kg]
lv = 2.499e6                # Enthalpy of vaporization [J/kg]
ls = 2.85e6                 # Enthalpy of sublimation [J/kg]
rw = 1000.0                 # Density of water [kg/m3]
ri = 917.0                  # Density of solid ice [kg/m3]
p0 = 101325                 # Normalization air pressure [Pa]
pi = np.pi                  # Pi :)

def best_num(m, ra, eta):
    """
    Return Best number (RH87).

    Input:
        Total particle mass [kg]
        Air density [kg/m3]
        Dynamic viscosity of air [kg/m/s]
    Output:
        Best number [unitless]

    """
    
    best = (8 * m * g * ra) / (pi * eta**2)
    # Note: This is an alternative form of the more traditional
    # form: (2 * m * g * Dmax**2) / (ra * nu**2 * A_cross) valid
    # for oblate spheroids and spheres
    
    return best
    

def reynolds_num_hail(m, ra, eta):
    """
    Return Reynolds number as a function of Best numnber (RH87).

    Input:
        Total particle mass [kg]
        Air density [kg/m3]
        Dynamic viscosity of air [kg/m/s]
    Output:
        Reynolds number [unitless]
    """

    best = best_num(m, ra, eta)
    
    # Reynolds number from Rasmussen and Heymsfield (1987):
    # nre_out = np.where(best < 3.46e8,
    #                   0.4487*best**0.5536,
    #                   np.sqrt(best / 0.6))
    
    # Hail Reynolds number from Heymsfield et al. (2018):
    nre_out = 0.38*best**0.56

    return nre_out



def reynolds_num_graupel(m, ra, eta):
    """
    Return Reynolds number as a function of Best number (Theis et al. 2022).

    Input:
        Total particle mass [kg]
        Air density [kg/m3]
        Dynamic viscosity of air [kg/m/s]
    Output:
        Reynolds number [unitless]
    """

    best = best_num(m, ra, eta) 
    nre_out = 0.38*best**0.56 # H18

    return nre_out


def reynolds_num_theis(m, ra, eta):
    """
    Return Reynolds number as a function of Best number (Theis et al. 2022).
    Note this relation was derived for dry graupel with densities of ~400 kg/m3.

    Input:
        Total particle mass [kg]
        Air density [kg/m3]
        Dynamic viscosity of air [kg/m/s]
    Output:
        Reynolds number [unitless]
    """

    best = best_num(m, ra, eta) 
    f = -1.6065 + 1.193 * np.log10(best) - 0.094226 * np.log10(best)**2 + 0.00432673 * np.log10(best)**3
    nre_out = 10**(f)

    return nre_out


def reynolds_num_sphere(m, ra, eta):
    """
    Return Reynolds number as a function of Best number for a smooth, rigid
    sphere following Abraham (1970).

    Input:
        Total particle mass [kg]
        Air density [kg/m3]
        Dynamic viscosity of air [kg/m/s]
    Output:
        Reynolds number [unitless]
    """

    best = best_num(m, ra, eta)
    nre_out = 20.5209 * (np.sqrt(1 + 0.09018 * np.sqrt(best))-1)**2
    
    return nre_out
    

# def drag_coef_sphere(nre):
#     """
#     Return the drag coefficient Cd as a function of Reynolds number for a 
#     smooth sphere (Cheng et al. 2009).
    
#     Input:
#         Reynolds number
#     Output:
#         Drag coefficient
#     """
    
#     return (24/nre) * (1 + 0.27*nre)**0.43 + 0.47 * (1 - np.exp(-0.04*nre**0.38))


# !!! This function is deprecated as this parameterization was originally
# intended for melting snow aggregates that may differ from
# spheroid shapes, not rigid hailstones.
# Retaining for posterity.
# 
# def capacitance(ar, d, fm):
#     """
#     Return particle capacitance following Mitra et al. (1990).

#     Input:
#         Aspect ratio [unitless]
#         Particle diameter [m]
#         Meltwater fraction [unitless]
#     Output:
#         Particle capacitance [m]
#     """

#     x = np.sqrt(1 - ar**2)
#     shape = ar**(-1.0 / 3.0) * (x / np.arcsin(x))
#     shape[ar > 0.99] = 1.0
#     c0 = 0.5 * d * shape
#     c = c0 * (0.8 + 0.2 * fm)  # Account for deviations from spheroidal shape

#     return c


def aspect_ratio_rain(d):
    """
    Return particle aspect ratio (a/b) of raindrops 
    according to Brandes et al. (2005).

    Input:
        Particle diameter [mm]
    Output:
        Aspect ratio [unitless]
    """

    ar = (0.9951000 +
          0.0251000 * d -
          0.0364400 * d**2 +
          0.0053030 * d**3 -
          0.0002492 * d**4)
    # Limit to between [0.56, 1.0]
    ar = np.where(ar > 1.0,
                  1.0,
                  ar)
    ar = np.where(ar < 0.56,
                  0.56,
                  ar)

    return ar


def term_vel_hail_phillips(nre, d, di, dmax, ar_in, mi, mw_inside, mw_outside, va, va0, rs, nu, ra, eta, nu0, ra0,
                           d_justsoaked, m_justsoaked):
    
    """
    Return terminal velocity of melting hailstones based on the 
    parameterization of Phillips et al. (2007).

    Input:
        
    Output:
        Reynolds number of melting hail stone [unitless]
        Terminal velocity of melting hail stone [m/s]
    """
    
    best = best_num((mi + mw_inside + mw_outside), ra, eta)
    nre_dry = reynolds_num_theis((mi + mw_inside + mw_outside), ra, eta)
    nre_smooth = reynolds_num_sphere((mi + mw_inside + mw_outside), ra, eta)
    
    if (mw_inside + mw_outside) == 0.0: # Particle is dry ice
        u_dry = (nre_dry * nu) / dmax
        return nre_dry, u_dry
    
    elif mi == 0.0: # Particle is rain
        u_rain = (ras/ra)**(0.4) * term_vel_rain(mw_outside + mw_inside)
        nre_rain = u_rain * dmax / nu
        return nre_rain, u_rain
    
    elif va > 0: # Hailstone undergoing soaking
        if nre < 4000: # Should this be based on nre_dry rather than input Nre?
            nre_transition = nre_smooth + (nre_dry - nre_smooth) * (va / va0)
            # Note: Above is an adjustment from Phillips scheme by transitioning between
            # dry and smooth hailstones based on amount of air cavities left. 
            # The Phillips parameterization considers the surface of the hailstone wet regardless
            # of it soaking is still occurring and would therefore be using "Nre_smooth".
            u_transition = (nre_transition * nu) / dmax
            
            return nre_transition, u_transition
        else:
            d_fullyfrozen = ((6 / pi) * ((mi + mw_inside) / rs))**(1./3.)
            dmax_fullyfrozen = d_fullyfrozen * (ar_in)**(-1./3.)
            u_fullyfrozen = (nre_dry * nu) / dmax_fullyfrozen
            u_dry = u_fullyfrozen * (dmax_fullyfrozen / dmax)

            return nre_dry, u_dry

    else: # Meltwater accumulating on outside of hailstone
        # Maximum meltwater hailstone can hold
        mwmax = 2.68e-4 + 0.1389 * (mi + mw_inside)
        # Diameter of hailstone when mw_outside = mwmax
        d_crit = ((6 / pi) * (mi/ri + (mw_inside + mwmax)/rw))**(1./3.)
        # Reynolds number of hailstone when mw_outside = mwmax
        nre_crit = 4800 + 4.8315e6 * (mi + mw_inside)
        # Best number of hailstone when mw_outside = mwmax
        best_crit = (8 * (mi + mw_inside + mwmax) * g * ra) / (pi * eta**2)

        
        # Calculate equilibrium fallspeed for when mw_outside = mwmax
        if nre_crit < 5000:
            u_crit = (ras/ra)**(0.4) * term_vel_rain(mi + mw_inside + mwmax)
        else:
            b_core = (di**3. / ar_in)**(1./3.)
            a_core = (di**3. / b_core**2.)
            da = np.minimum(0.5e-3, 0.05 * di)
            db = np.sqrt((((6 * 2.68e-4) / (pi * rw)) + ((1.1275 * a_core**3) / ar_in**2)) * (1. / (da + a_core))) - (a_core / ar_in)
            ar_crit = (a_core + da) / (db + b_core)
            dmax_crit = d_crit * (ar_crit)**(-1./3.)
            if nre_crit > 25000:
                #best = best_num((mi + mw_inside + mwmax), ra, eta)
                nre_crit = (best_crit / 0.6)**(0.5)
                u_crit = (nre_crit * nu / dmax_crit)
            else:
                #u_crit = (nre_crit / dmax_crit) * (1.5e-5) * (ras/ra)**(0.5) 
                u_crit = (nre_crit * nu / dmax_crit) * (ras/ra)**(0.4) 
                # Approximtion from Phillips and RH87 for ratio of Cd and rho at
                # the pressure level of interest and SLP, which is where the NRE_crit
                # parameterization is valid for.
            
        # Calculate fallspeed when particle is initially soaked
        #m_justsoaked = (mi + mw_inside + mw_outside) / (1.0 + (rw/rs) - (rw/ri))
        #d_justsoaked = ((6/pi) * (m_justsoaked/ri))**(1./3.)
        # This is from Phillips et al. (2007) but is still technically wrong for
        # particles which have lost (or gained) water to evaporation. Need to 
        # come up with a way to feed function actual mw_inside and mi when va
        # first hits 0.0
        dmax_justsoaked = d_justsoaked * (ar_in)**(-1./3.)
        best_justsoaked = (8 * m_justsoaked * g * ra) / (pi * eta**2)
        
        if (nre_dry < 4000) or (rs <= 800): # Original conditions also had rho < 800 kg/m3 criteria...
            nre_smooth = reynolds_num_sphere(m_justsoaked, ra, eta)
            u_justsoaked = (nre_smooth * nu / dmax_justsoaked)
        else:
            d_dry = ((6 / pi) * (m_justsoaked / rs))**(1./3.)
            dmax_dry = d_dry * (ar_in)**(-1./3.)
            nre_justsoaked = reynolds_num_theis(m_justsoaked, ra, eta)
            u_dry = (nre_justsoaked * nu) / dmax_dry
            u_justsoaked = u_dry * (dmax_dry / dmax_justsoaked)

        u_melting = u_justsoaked + (u_crit - u_justsoaked) * (mw_outside / mwmax)
        
        # !!! New addition
        # Once mw_outside > (mw_inside + mi), begin smooth transition to the fall 
        # characteristics of raindrops
        u_rain = (ras/ra)**(0.4) * term_vel_rain(mi + mw_inside + mw_outside)
        if mw_outside >= (mi + mw_inside):
            u_melting = u_melting + (u_rain - u_melting) * ((mw_outside - mi - mw_inside) / (mw_outside + mi + mw_inside))
        
        nre_melting = u_melting * dmax / nu

        return nre_melting, u_melting

# !!! Note: This function is deprecated. Based on the original RH87 melting scheme
# used in Ryzhkov et al. (2013).
# Retaining for posterity.
# 
# def term_vel_hail(nre, d, dmax, nu, ra, mi, mw_inside, mw_outside):
#     """
#     Return terminal velocity of melting hail as a function of Reynolds number.

#     Input:
#         Reynolds number [unitless]
#         Particle diameter [m]
#         Kinematic viscosity [m2/s]
#         Air density [kg/m3]
#         Ice mass [kg]
#         Internal meltwater [kg]
#         External meltwater [kg]
#     Output:
#         Terminal velocity [m/s]
#     """

#     m = mi + mw_inside + mw_outside  # Total particle mass
#     mw_crit = 2.68e-4 + 0.1389 * (mi + mw_inside) # Critical meltwater
#     nre_crit = 4800 + 4.8315e6 * (mi + mw_inside) # Critical Reynolds number
#     #d_crit = (((mi/ri) + (mw_outside/rw)) * (6/pi))**(1./3.) # m
#     d_crit = (((mi/ri) + (mw_inside/rw)) * (6/pi))**(1./3.) # m

#     # Terminal velocity of dry stone:
#     ud = term_vel_dry(nre, nu, d)
#     #ud = term_vel_dry(nre, nu, dmax)
    
#     if mi == 0.0:
#         #uw = term_vel_rain(1e3 * d, ra, d_flag = 1)
#         uw = (ras/ra)**(0.4) * term_vel_rain(mw_inside + mw_outside, ra, d_flag = 0)
        
#         return uw
    
#     # Equilibrium (point at which shedding begins) fall speed for melting hail
#     if nre < 5000:
#         # For Nre < 5000, use equivalent raindrop fall speed
#         uw = (ras/ra)**(0.4)  * term_vel_rain(m, ra)
#     elif (nre >= 5000) and (nre < 25000):
#         # For 5000 < Nre < 25000 use RH87 eqn for "once shedding starts"
#         uw = (nu / d_crit) * nre_crit
#         # The above is what was originally in code. However, RH87 has the
#         # mass of ice and soaked water as the relevant mass, rather than the
#         # total mass as used in the above equation.
#     else:
#         # For the largest hailstones (most meltwater shed), equilibrium
#         # terminal velocity equal to that of a dry stone.
#         uw = nu * nre / d_crit
#         #uw = ud

#     # To calculate the instantaneous fallspeed, a linear function of
#     # critical meltwater fraction is used. This is our interpretation of
#     # the "equilibrium water mass" in RH87. However, for low-density
#     # stones, this seems to cause problems with unrealistic sudden changes
#     # in velocity. Thus, we have it based on the total meltwater fraction.
#     u = (mw_outside / mw_crit) * (uw - ud) + ud
#     # Note: Moved all separate density correction factors to this one final location

#     return u


# !!! NOTE: This function is not currently being used. Nevertheless retaining
# because the wind tunnel results are relevant.
def term_vel_graupel(rs, fm, d, dmax, eta, nu, ra, mi, mw_inside, mw_outside):
    """
    Return terminal velocity of melting graupel following wind tunnel observations
    of Theis et al. (2022).

    Input:
        Particle density [kg/m3]
        Meltwater fraction [unitless]
        Particle diameter [m]
        Particle maximum diameter [m]
        Dynamic viscosity of air [kg/m/s]
        Kinematic viscosity [m2/s]
        Air density [kg/m3]
        Ice mass [kg]
        Internal meltwater [kg]
        External meltwater [kg]
    Output:
        Terminal velocity [m/s]
    """

    # Terminal velocity of a raindrop of equivalent mass
    u_w = term_vel_rain(mi + mw_outside + mw_inside, ra, d_flag = 0)
    nre_graupel = reynolds_num_graupel(mi + mw_inside + mw_outside, ra, eta)
    # Terminal velocity of a dry graupel particle of equivalent mass
    u_d = term_vel_dry(nre_graupel, nu, d)
     
    # Parameterization of Theis et al. (2022)
    a2 = -1.205 * (1e-3 * rs) + 0.947
    a3 = 11.0 + 96.7 * (1e-3 * rs)**(3.8)
    u_norm = 1. / (1 + np.exp((a2 - fm) * a3)) 
    
    u_melt = u_norm * (u_w - u_d) + u_d
    
    return u_melt, u_norm
    


def term_vel_rain(x, ra=ras, d_flag=0):
    """
    Return terminal velocity of rain following Brandes et al. (2002).

    Input:
        Mass [kg; d_flag=0] or particle diameter [mm; d_flag=1]
        Air density [kg/m3; default = ras]
    Output:
        Terminal velocity [m/s]
    ra = density (ras used as default value)
    """

    if (d_flag == 0):  # Mass equation
        u = (np.maximum(-0.1021 +
                        6.116e2 * x**(1.0 / 3.0) -
                        1.469e4 * x**(2.0 / 3.0) +
                        1.513e5 * x**(3.0 / 3.0) -
                        5.584e5 * x**(4.0 / 3.0), 0.01))
    else:  # Diameter equation
        u = (np.maximum(-0.10210 +
                         4.932000 * x -
                         0.955100 * x**2 +
                         0.079340 * x**3 -
                         0.002362 * x**4, 0.01))

    return u


def term_vel_dry(nre, nu, d):
    """
    Return terminal velocity of dry hail following RH87.

    Input:
        Reynolds number [unitless]
        Kinematic viscosity [m2/s]
        Particle diameter [m]
    Output:
        Terminal velocity [m/s]
    """

    u = nu * nre / d
    # Density correction not needed bc it is incorporated into Best number
    # calculation when determining Nre. NOTE THAT IF WE SWITCH BACK TO THE
    # ABOVE FORMULATION WE NEED TO NOT ACCOUNT FOR DENSITY WHEN CALCULATING 
    # THE TERMINAL VELOCITY OF MELTING HAIL.

    return u


def tp_conv_subl(tp, e, t, kair, dv, vent_ratio):
    """
    Function used in optimizer routine (iterative solver) to calculate the
    equilibrium temperature of a sublimating particle.
    
    Input:
        Particle temperature [C]
        Environmental vapor pressure [Pa]
        Environmental temperature [C]
        Thermal conductivity of air [J/m/s/K]
        Diffusivity of water vapor in air [m2/s]
        Mean ratio of ventilation enhancement factors for vapor and heat [unitless]
    Output:
        Particle temperature [C]
    """
    
    estp = th.sat_vapor_p(tp, i_flag=1)
    coef = vent_ratio * (ls * dv) / (kair * rv)
    drho = (e / (t + t0)) - (estp / (tp + t0))

    return np.abs(t + (coef * drho) - tp)
    

def tp_conv_evap(tp, e, t, kair, dv, vent_ratio):
    """
    Function used in optimizer routine (iterative solver) to calculate the
    equilibrium temperature of an evaporating particle.
    
    Input:
        Particle temperature [C]
        Environmental vapor pressure [Pa]
        Environmental temperature [C]
        Thermal conductivity of air [J/m/s/K]
        Diffusivity of water vapor in air [m2/s]
        Mean ratio of ventilation enhancement factors for vapor and heat [unitless]
    Output:
        Particle temperature [C]
    """
    estp = th.sat_vapor_p(tp)
    coef = vent_ratio * (lv * dv) / (kair * rv)
    drho = (e / (t + t0)) - (estp / (tp + t0))

    return np.abs(t + (coef * drho) - tp)


def tp_conv_melt_graup(x, e, t, kair, fh, lv, dv, fv, di, kwa, rv, d, mw_inside, mi, rs, fw):
    """
    Function used in optimizer routine (iterative solver) to calculate the
    equilibrium temperature of a melting hailstone that has turbulent meltwater (750 < Nre < 6000)
    
    I *think* this was derived from Theis et al. (2022)? (See page 1081)
    
    Input:
        Particle temperature [C]
        Environmental vapor pressure [Pa]
        Environmental temperature [C]
        Thermal conductivity of air [J/m/s/K]
        Ventilation coefficient for heat [unitless]
        Enthalpy of vaporization [J/kg]
        Diffusivity of water vapor in air [m2/s]
        Ventilation coefficient for vapor [unitless]
        Particle inner diameter [m]
        Thermal conductivity of water [J/m/s/K]
        Gas constant for water vapor [J/kg/K]
        Particle diameter [m]
        Internal meltwater mass [kg]
        Ice mass [kg]
        Particle density [kg/m3]
        Meltwater fraction [unitless]
    Output:
        Particle temperature [C]
    """
    estp = th.sat_vapor_p(x)
    drho = (1 / rv) * ((e / (t + t0)) - (estp / (x + t0)))
    r = 0.5 * d
    rcore = 0.5 * di
    rsoak = ((3 * mi) / (4 * pi * rs))**(1./3.)
    
    term1 = (kair * fh * (t - x))
    term2 = (lv * dv * fv * drho)
    term3 = (kwa * fw * (0.0 - x)) * (rcore / (r - rcore))

    return np.abs(term1 + term2 + term3)
