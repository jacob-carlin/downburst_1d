"""
Original Author: Alexander V. Ryzhkov (2009-2012)
Python Translation & Additions: Jacob T. Carlin (2018-2024)

Description:    One-dimensional model of downdraft development based on Srivastava (1987) with
                melting hailstones of variable density, based on the studies of Rasmussen et al. (1984), 
                Rasmussen et al. (1987), Phillips et al. (2007), Ryzhkov et al. (2013a), and Theis et al. (2022) 
                and coupled to the polarimetric operator of Ryzhkov et al. (2011).

Contact: jacob.carlin@noaa.gov

Updates:
05/15/2018: Added iterative solver for Ta (particle temperature at ice/air
            interface) for melting in the 3000 < NRe < 6000 regime based on
            Eqn. (3) from RH87.

            Updated budgeting of meltwater for clarity and to incorporate
            possible mass losses due to sublimation/evaporation in future
            updates.n

            Added switches to turn shedding and drop breakup on/off.

            Updated calculation of es and esi to use Buck (1996) formulas.

            Added calculation of latent cooling due to melting.

05/18/2018: Added calculation of dielectric constant using Maxwell-Garnett
            mixing formulas for spongy (low-density) hail.

            Adjusted calculation of aspect ratio for spongy (low-density) hail
            to use the meltwater fraction of outside water rather than total
            meltwater fraction (which includes soaked water).

            Added option to use sighail = 60 and vary the sigma between f = 0.0
            and f = 0.5 following Dawson et al. (2014).

05/21/2018: Added backscatter differential phase calculation based on Eq. (2)
            of Tromel et al. (2013) (Note: In that Eq. Ang3 should be Ang5).

05/25/2018: Added function for iteratively calculating particle equilibrium
            surface temperatures.

            Added function for the sublimation of dry ice and its associated
            cooling rate.

            Added function for the evaporation of both melting ice and shed
            raindrops and their associated cooling rate.

            Restructured code to store microphysical and scattering
            computations in functions.

06/19/2018: Separated microphysical and scattering computations into separate
            imported modules.

            Fixed evaporation and sublimation subroutines to follow
            Pruppacher and Klett (1997) using Eqs. (13-28) and (13-76).

            Added model timer.

            Added option and flag for Weisman-Klemp environmental profile.

            Fixed calculation of radar variables for shed and breakup water and
            incorporated differing DSDs of shed drops due to differential
            evaporation depending on when drops were shed.

            Fixed bug in variable density meltwater partitioning.

06/28/2018: Fixed calculation of cooling rate due to melting according to new
            understanding of heat balance via Srivastava et al. (1987) and
            Szyrmer and Zawadzki (1999).

07/01/2018: Changed iterative solver for Tp to use scipy.optimize.

            Added non-equilibrium Tp calculation based on terminal velocity.
            
02/05/2019: Implemented dynamically-evolving downdraft & environment based on 
            the models of Srivastava (1985, 1987) and, to a lesser extent,
            Feingold (1991). Employs coupled equations for vertical velocity 
            (w), equivalent temperature (h), water vapor and cloud water mixing
            ratio (qstar), and concentration (dsdm), a saturation adjustment,
            and mixing by entrainment, but excludes breakup and shedding (for
            now) and perturbation pressures. Conditions at the top of the
            downdraft are fixed and is initially equal to that of the 
            environment, which is assumed to be constant. 
            
11/29/2021: Added breakup and shedding routines among existing liquid particles.

            Corrected implementation error of dqstar/dt.
            
            Corrected error in melting/evaporation/sublimation rates (previous
            version had only dt as residence time in each grid leading to 
            too-slow melting, etc.)
            
            Added rainrate and overall precipitation rate calculations with and
            without vertical velocity factor.
            
            Added surface windspeed parameterization based on Anabor et al.
            (2011) and Hawbecker et al. (2018).
            
            Added look-up tables for complex scattering amplitudes and
            interpolation routines for more efficient radar variable calculations
            (* Currently only for S-band *).
            
            Added mixing ratio-based moisture lapse rate to approximate well-
            mixed boundary layer (instead of constant RH).
            
            Added option for writing out netcdf files of model variables.
            
            Updated equilibrium temperature convergence calculations (but 
            removed non-equilibrium condition for the time being).
            
04/01/2021: Fixed air density calculation to use Tv for moist and Tk for dry.

# Note: These updates have (ironically) not been kept updated. Model has changed
# beyond what is described here.
"""

# Change to directory containing model Python modules
workdir = '/Users/jacob.carlin/Documents/Data/1D Downburst Model/'
print('Working directory: ', workdir)

###############################################################################
# Import necessary packages
###############################################################################
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.family'] = 'PT Sans'
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.colors import ColorConverter, LinearSegmentedColormap
from scipy.integrate import simpson
from scipy import optimize
import scipy as sp
import metpy.calc as mpcalc
from metpy.units import units
import os
from scipy import ndimage
import pandas as pd
from siphon.catalog import TDSCatalog
import pyart

# Import other internal functions
#cwd = os.getcwd()
#os.chdir(workdir)
import src.mh_scattering as rc
import src.mh_microphysics as mp
import mh_namelist as nm
import src.mh_thermo as th
from src.get_hrrr_sounding import *
from src.mh_thermo import *
#os.chdir(cwd)

###############################################################################
# Import variables and constants from namelist
###############################################################################

ntstp = nm.ntstp
delt = nm.delt
nbin = nm.nbin
deld = nm.deld
init_frozen_opt = nm.init_frozen_opt
dsd_norm = nm.dsd_norm
nr0 = nm.nr0
lamr = nm.lamr
mur = nm.mur
nrw = nm.nrw
dmr = nm.dmr
rs_opt = nm.rs_opt
ng0 = nm.ng0
lamg = nm.lamg
nh0 = nm.nh0
lamh = nm.lamh
dmax_limit = nm.dmax_limit
Fsub = nm.Fsub
wave = nm.wave
lamda = nm.lamda
ew0 = nm.ew0
ei = nm.ei
ea = nm.ea
shed_opt = nm.shed_opt
shed_dsd_opt = nm.shed_dsd_opt
break_opt = nm.break_opt
evap_opt = nm.evap_opt
subl_opt = nm.subl_opt
radar_opt = nm.radar_opt
generate_lut = nm.generate_lut
use_lut = nm.use_lut
use_2layer = nm.use_2layer
lut_path = nm.lut_path
twolayer_lut_path = nm.twolayer_lut_path
write_netcdf = nm.write_netcdf
netcdf_path = nm.netcdf_path
verbose = nm.verbose
ar_opt = nm.ar_opt
sigma_opt = nm.sigma_opt
profile_opt = nm.profile_opt
sounding_path = nm.sounding_path
sounding_time = nm.sounding_time
sounding_lat = nm.sounding_lat
sounding_lon = nm.sounding_lon
sounding_alt = nm.sounding_alt
h0 = nm.h0
dh = nm.dh
t_top = nm.t_top
gam = nm.gam
rh_top = nm.rh_top
gam_rh = nm.gam_rh
use_mixing_ratio = nm.use_mixing_ratio
qv_top = nm.qv_top
t0 = nm.t0
p0 = nm.p0
rw = nm.rw
ri = nm.ri
rg = nm.rg
ar_g = nm.ar_g
pi = nm.pi
g = nm.g
rv = nm.rv
rd = nm.rd
ras = nm.ras
es0 = nm.es0
cp = nm.cp
ci = nm.ci
cw = nm.cw
lf = nm.lf
lv = nm.lv
ls = nm.ls
sigrain = nm.sigrain
sighail = nm.sighail
eps_0 = nm.eps_0
mix_coef = nm.mix_coef
make_plots = nm.make_plots
save_plots = nm.save_plots

# Radar constants
kw = (abs((ew0 - 1) / (ew0 + 2)))**2
cz = (4.0 * wave**4)/(pi**4 * kw) # Z prefactor
ckdp = (0.18 / pi) * wave # KDP prefactor
ca = 8.686E-3 * wave # Attenuation prefactor

###############################################################################
# Vectorize functions to allow them to be used for full DSD simultaneously
###############################################################################

ew_vec = np.vectorize(rc.dielectric_water)
shape_factors_vec = np.vectorize(rc.shape_factors)
sat_vapor_p_vec = np.vectorize(th.sat_vapor_p)
term_vel_hail_phillips_vec = np.vectorize(mp.term_vel_hail_phillips)

###############################################################################
# Start timer
###############################################################################

start_time = datetime.now()

###############################################################################
# Stability check
###############################################################################

# Check Courant number (C = u * delt / delh) < 1 assuming maximum fallspeed 
# (u + w) assumed to max out at 80 m/s (just to be safe)

courant = 80.0 * delt / dh
if courant >= 1.0 :
    print('Courant–Friedrichs–Lewy condition violated. Model will be unstable. Exiting.')
    sys.exit()

if init_frozen_opt:
    print("Graupel Lamda: ", lamg, "mm-1")
    print("Graupel Intercept: ", ng0, "m-3 mm-1")
    print("Hail Lamda: ", lamh, "mm-1")
    print("Hail Intercept: ", nh0, "m-3 mm-1")
    print("Maximum Hail Size: ", dmax_limit, "mm")
    if rs_opt == 0:
        print("Solid ice")
    else:
        print("Variable-density ice")
else:
    if dsd_norm:
        print("Using normalized gamma raindrop distribution...")
        print("Normalized rain intercept: ", nrw, "m-3 mm-1")
        print("Normalized mean-mass diameter: ", dmr, "mm")
    else:
        print("Using gamma raindrop distribution...")
        print("Rain Intercept: ", nr0, "m-3 mm-1")
        print("Rain Lamda: ", lamr, "mm-1")
        print("Rain Mu: ", mur)
    print("Maximum raindrop size: ", dmax_limit, "mm")
    
###############################################################################
########################### Define environment ################################
###############################################################################

nlev = int(round((h0 / dh) + 1, 0))  # Number of height levels
heights = h0 - dh * np.arange(nlev)  # Array of heights for plotting purposes
nradlev = int((nlev / 2) + 1.0)  # Number of radar levels

# Idealized profile given by temperature and humidity lapse rates in namelist
if profile_opt == 0:
    print('Environment: Idealized')
    print('T0: ', t0, 'C')
    print('Temperature lapse rate: ', gam, 'C/km')
    print('RH0: ', rh_top, '%')
    print('RH lapse rate:', gam_rh, '%/km')
    tenv = t_top + (dh/1000) * gam * np.arange(nlev)  # Temperature profile in C
    tenv_k = tenv + t0  # Temperature profile in K
    penv = p0 * (t0/(t0+np.max(tenv)-tenv))**(g/(0.001*gam*287.0))  # Pressure profile [hPa]
    # Note: This approximates hydrostatic balance. Applicability inside downdraft?
    rhenv = rh_top + (dh/1000) * gam_rh * np.arange(nlev)
    if use_mixing_ratio: 
        # Use constant mixing ratio instead of RH to approximate well-mixed PBL
        # Note that qv_top must be chosen carefully considering the model top (h0)
        # in order to not result in RH > 100 or unrealistically low surface dewpoints
        rhenv = 1e2 * (qv_top / (0.622 * sat_vapor_p(tenv) / (penv - sat_vapor_p(tenv))))
        rhenv[rhenv > 100.] = 100.
        if (np.max(rhenv) > 100.) or (np.min(rhenv) < 0.):
            print('Mixing ratio choice resulting in unrealistic RH profile. Exiting.')
            sys.exit()
    
# Custom input sounding from the University of Wyoming sounding archive website
# Source: http://weather.uwyo.edu/upperair/sounding.html
if profile_opt == 1:
    print('Environment: Pre-determined sounding.')
    env_data = open(sounding_path, "r")
    lines = env_data.readlines()[:] # Skips University of Wyoming header
    env_data.close()

    p = []
    h = []
    t = []
    rh = []

    for line in lines:
        parts = [float(x) for x in line.split() if x]
        line_p = parts[1] * 100     # [Pa]
        line_h = parts[0]           # [m]
        line_t = parts[2]           # [C]
        line_rh = parts[4]          # [%]

        p.append(line_p)
        h.append(line_h)
        t.append(line_t)
        rh.append(line_rh)

    # Convert to numpy arrays
    p = np.asarray(p)
    h = np.asarray(h)
    t = np.asarray(t)
    rh = np.asarray(rh)

    # Find top of domain
    h0 = int(dh * round(float(h[-1]) / dh))
    # Set first layer to first data point (0C)
    h[-1] = h0
    nlev = int(round((h0 / dh) + 1, 0))  # Number of height levels
    heights = h0 - dh * np.arange(nlev)  # Array of heights for plotting purposes
    nradlev = int((nlev / 2) + 1.0)  # Number of radar levels

    p_interp = np.zeros((int(nlev)))
    f = sp.interpolate.interp1d(h, p)
    p_interp[:] = f(heights)

    t_interp = np.zeros((int(nlev)))
    f = sp.interpolate.interp1d(h, t)
    t_interp[:] = f(heights)

    rh_interp = np.zeros((int(nlev)))
    f = sp.interpolate.interp1d(h, rh)
    rh_interp[:] = f(heights)
    
    for lev in range(len(t_interp)):
        print('Height: ', heights[lev], '  Temp: ', t_interp[lev], '  RH: ', rh_interp[lev])

    # Optional: Apply spline fit to smooth out discontinuities
#    spl = splrep(heights[::-1], p_interp[::-1], s=10, k=2)
#    p_spline = splev(heights, spl)
#    spl = splrep(heights[::-1], t_interp[::-1], s=10, k=2)
#    t_spline = splev(heights, spl)
#    spl = splrep(heights[::-1], rh_interp[::-1], s=10, k=2)
#    rh_spline = splev(heights, spl)    
    
    rhenv = rh_interp
    penv = p_interp
    tenv = t_interp

if profile_opt == 2:
    print('Environment: Weisman-Klemp sounding')
    # Weisman-Klemp idealized sounding
    # !!! NOTE: This has not been thoroughly tested or proofed!!!
    wk_pt0 = 300  # Surface potential temp [K]
    wk_pttr = 343  # Tropopause potential temp [K]
    wk_ztr = 12000
    wk_pbl = 2000  # Depth of well-mixed PBL [m]
    pt_prof = (wk_pt0 + (wk_pttr - wk_pt0) * (heights/wk_ztr)**(5./4.))
    tenv = pt_prof * (penv / 100000)**(0.286) - t0
    tenv_k = tenv + t0
    rhenv = 100.0 * (1 - 0.75 * (heights/12000)**(5./4.))
    rh_pbl = 100.0 * (0.012) / (0.622 * 611.21 * np.exp((18.678 - tenv / 234.5) * (tenv / (257.14 + tenv))) / penv)
    rhenv[nlev-int(0.1*wk_pbl):] = np.minimum(rhenv[nlev-int(0.1*wk_pbl):], rh_pbl[nlev-int(0.1*wk_pbl):])

if profile_opt == 3:
    print('Environment: RAP sounding at ' + str(sounding_lat) + 'N, ' + str(sounding_lon) + 'W')
    hour = f"{sounding_time.hour:02}"
    day = f"{sounding_time.day:02}"
    month = f"{sounding_time.month:02}"
    year = f"{sounding_time.year:04}"
    
    if (int(year) >= 2020) or (int(year) == 2020 and int(month) >= 5):
        url = 'https://www.ncei.noaa.gov/thredds/catalog/model-rap130anl/' + year + month + '/' + year + month + day + '/catalog.xml'
    else:
        url = 'https://www.ncei.noaa.gov/thredds/catalog/model-rap130anl-old/' + year + month + '/' + year + month + day + '/catalog.xml'

    print('Downloading RAP sounding profile...')
    cat = TDSCatalog(url)
    ds = cat.datasets['rap_130_' + year + month + day + '_' + hour.zfill(2) + '00_000.grb2']
    ncss = ds.subset() 
    query = ncss.query()
    query.lonlat_point(sounding_lon, sounding_lat)
    query.variables('Temperature_isobaric', 'Relative_humidity_isobaric', 'Geopotential_height_isobaric')
    data = ncss.get_data(query)
    
    p = np.flip(data['vertCoord']).copy() # Pa
    t = np.flip(data['Temperature_isobaric'] - 273.15).copy() # degC
    rh = np.flip(data['Relative_humidity_isobaric']).copy() # 0-1
    h = np.flip(data['Geopotential_height_isobaric']).copy() # m
    print('Download complete.')
    
    # Make relative to ground
    #h -= h[0] 
    h -= sounding_alt

    idx = np.where(t < 0)[0][0]
    h0 = int(dh * round(float(h[idx]) / dh))
    nlev = int(round((h0 / dh) + 1, 0))  # Number of height levels
    heights = h0 - dh * np.arange(nlev)  # Array of heights for plotting purposes
    
    f = sp.interpolate.interp1d(h, p, bounds_error=False, fill_value="extrapolate")
    penv = f(heights)
    f = sp.interpolate.interp1d(h, t, bounds_error=False, fill_value="extrapolate")
    tenv = f(heights)
    f = sp.interpolate.interp1d(h, rh, bounds_error=False, fill_value="extrapolate")
    rhenv = f(heights)  

    for lev in range(len(tenv)):
        print('Height: ', heights[lev], '  Pres: ', penv[lev], '  Temp: ', tenv[lev], '  RH: ', rhenv[lev])
    
    
if profile_opt == 4: # HRRR sounding for requested lat/lon
    #hrrr_time = hrrr_time.replace(tzinfo=timezone.utc)
    print('Environment: HRRR sounding at ' + str(sounding_lat) + 'N, ' + str(sounding_lon) + 'W')
    print('Downloading HRRR sounding profile...')
    p, t =  get_hrrr_sounding(sounding_time, 'TMP',  fxx=0, field="prs", lats=[sounding_lat], lons=[sounding_lon])
    p, h = get_hrrr_sounding(sounding_time, 'HGT', fxx=0, field="prs", lats=[sounding_lat], lons=[sounding_lon])
    p, rh =  get_hrrr_sounding(sounding_time, 'RH',  fxx=0, field="prs", lats=[sounding_lat], lons=[sounding_lon])
    print('Download complete.')
    
    t = np.asarray(t)[0][0:-3]
    h = np.asarray(h)[0][0:-3]
    rh = np.asarray(rh)[0][0:-3]
    p = np.asarray(p)[0:-3] * 1e2
    
    t -= 273.15 # K --> C
    h -= sounding_alt
    
    idx = np.where(t < 0)[0][0]
    h0 = int(dh * round(float(h[idx]) / dh))
    # Set first layer to first data point (0C)
    nlev = int(round((h0 / dh) + 1, 0))  # Number of height levels
    heights = h0 - dh * np.arange(nlev)  # Array of heights for plotting purposes
    
    # Interpolate to model height grid
    f = sp.interpolate.interp1d(h, p)
    penv = f(heights)
    f = sp.interpolate.interp1d(h, t)
    tenv = f(heights)
    f = sp.interpolate.interp1d(h, rh)
    rhenv = f(heights)

    for lev in range(len(tenv)):
        print('Height: ', heights[lev], '  Pres: ', penv[lev], '  Temp: ', tenv[lev], '  RH: ', rhenv[lev])
    
    
# Calculate the rest of the environmental parameters
tenv_k = convertCtoK(tenv)                      # Temperature [K]
esenv = sat_vapor_p(tenv)                       # Saturation vapor pressure [Pa] (Bolton 1980) 
esienv = sat_vapor_p(tenv, i_flag=1)            # Saturation vapor pressure wrt ice [Pa]
eenv = (rhenv / 100.) * esenv                   # Vapor pressure [Pa]
qsenv = vapor_mixing_ratio(esenv, penv)         # Saturation mixing ratio [kg/kg]
qenv = (rhenv / 100.) * qsenv                   # Mixing ratio [kg/kg]
tvenv = virtual_temperature(tenv_k, qenv)       # Virtual temperature [K]
radenv = density_air(penv, tenv_k)              # Density of dry air [kg/m3]
raenv = density_air(penv, tvenv)                # Density of moist air [kg/m3]
eta = dynamic_viscosity_air(tenv_k)             # Dynamic viscosity of air [kg/m/s]
nu = eta / raenv                                # Kinematic viscosity [m2/s]
ka = thermal_diffusivity_air(tenv_k)            # Thermal diffusivity of air [m2/s]
kair = thermal_conductivity_air(tenv_k)         # Thermal conductivity of air [J/m/s/K]
kwa = thermal_conductivity_water(tenv_k)        # Thermal conductivity of water [J/m/s/K]
dv = thermal_diffusivity_water(tenv_k, penv)    # Diffusivity of water vapor in air [m2/s]
pr = nu / ka                                    # Prandtl number
sc = nu / dv                                    # Schmidt number
henv = (lv * qenv) / cp  + tenv_k               # Total Heat (K)

# Define (empty) downdraft arrays that evolve in time
# Environmental variables
t = np.zeros((nlev, ntstp))                     # Temperature [C]
tk = np.zeros((nlev, ntstp))                    # Temperature [K]
tv = np.zeros((nlev, ntstp))                    # Virtual temperature [C]
rh = np.zeros((nlev, ntstp))                    # Relative humidity [%]
h = np.zeros((nlev, ntstp))                     # Heat [K]
q = np.zeros((nlev, ntstp))                     # Water vapor mixing ratio [kg/kg]
qs = np.zeros((nlev, ntstp))                    # Saturation water vapor mixing ratio [kg/kg]
qc = np.zeros((nlev, ntstp))                    # Cloud mixing ratio [kg/kg]
qstar = np.zeros((nlev, ntstp))                 # Cloud water and vapor mixing ratio [kg/kg]
qp = np.zeros((nlev, ntstp))                    # Total precipitation mixing ratio [kg/kg]
e = np.zeros((nlev, ntstp))                     # Vapor pressure [Pa]
es = np.zeros((nlev, ntstp))                    # Saturation vapor pressure wrt water [Pa]
esi = np.zeros((nlev, ntstp))                   # Saturation vapor pressure wrt ice [Pa]
w = np.zeros((nlev, ntstp))                     # Vertical velocity [m/s]
p = np.zeros((nlev, ntstp))                     # Pressure [Pa]
ra = np.zeros((nlev, ntstp))                    # Density of air [kg/m3]
rad = np.zeros((nlev, ntstp))                   # Density of dry air [kg/m3]
eta = np.zeros((nlev, ntstp))                   # Dynamic viscosity [kg/m/s]
nu = np.zeros((nlev, ntstp))                    # Kinematic viscosity [m2/s]
ka = np.zeros((nlev, ntstp))                    # Thermal diffusivity of air [m2/s]
kair = np.zeros((nlev, ntstp))                  # Thermal conductivity of air [J/m/s/K]
kwa = np.zeros((nlev, ntstp))                   # Thermal conductivity of water [J/m/s/K]
dv = np.zeros((nlev, ntstp))                    # Diffusivity of water vapor in air [m2/s]
pr = np.zeros((nlev, ntstp))                    # Prandtl number
sc = np.zeros((nlev, ntstp))                    # Schmidt number
sfc_wind = np.zeros((ntstp))                    # Estimated horizontal wind speed at surface [mph]
sfc_t = np.zeros((ntstp))                       # Surface temperature [C]
sfc_td = np.zeros((ntstp))                      # Surface dewpoint [C]

# Particle variables
rs = np.zeros(nbin)                             # Initial particle density [kg/m3]
d = np.zeros((nlev, nbin, ntstp))               # Particle (total) equivolume diameter [m]
di = np.zeros((nlev, nbin, ntstp))              # Ice core equivolum diameter [m]
dmax = np.zeros((nlev, nbin, ntstp))            # Maximum diameter of particles assuming oblate spheroids [m]
v = np.zeros((nlev, nbin, ntstp))               # Volumne of total particle [m3]
vi = np.zeros((nlev, nbin, ntstp))              # Volume of ice [m3]
vw = np.zeros((nlev, nbin, ntstp))              # Volume of liquid water [m3]
va = np.zeros((nlev, nbin, ntstp))              # Volume of air [m3]
mi = np.zeros((nlev, nbin, ntstp))              # Mass of ice [kg]
mw = np.zeros((nlev, nbin, ntstp))              # Mass of meltwater [kg]
mw_outside = np.zeros((nlev, nbin, ntstp))      # Mass of water on outside of particle [kg]
mw_inside = np.zeros((nlev, nbin, ntstp))       # Mass of water inside particle [kg]
u = np.zeros((nlev, nbin, ntstp))               # Terminal velocity [m/s]
ud = np.zeros((nlev, nbin, ntstp))              # Terminal velocity of dry hailstones [m/s]
uw = np.zeros((nlev, nbin, ntstp))              # Equil. terminal velocity of melting hailstones [m/s]
u_weighted = np.zeros((nlev, ntstp))            # Mass-weighted terminal velocity [m/s]
nre = np.zeros((nlev, nbin, ntstp))             # Reynolds numbers
fh = np.zeros((nlev, nbin, ntstp))              # Ventilation coefficient for heat
fv = np.zeros((nlev, nbin, ntstp))              # Ventilation coefficient for vapor
dvi = np.zeros((nlev, nbin, ntstp))             # Total change in ice volume within grid box [m3]
dvw = np.zeros((nlev, nbin, ntstp))             # Total change in water volume within grid box [m3]
dvw_evap = np.zeros((nlev, nbin, ntstp))        # Change in water volume due to evaporation within grid box [m3]
dvi_subl = np.zeros((nlev, nbin, ntstp))        # Change in ice volume due to sublimation within grid box [m3]
dvi_melt = np.zeros((nlev, nbin, ntstp))        # Change in ice volume due to melting within grid box[m3]
dmw = np.zeros((nlev, nbin, ntstp))             # Change in water mass within grid box[kg]
tp = np.zeros((nlev, nbin, ntstp))              # Particle temperature [C]
tp_eq = np.zeros((nlev, nbin, ntstp))           # Equilibrium particle temperature [C] (Defunct?)
fm = np.zeros((nlev, nbin, ntstp))              # Mass water fraction
fm_ar = np.zeros((nlev, nbin, ntstp))           # Fm for computing AR for ar_opt = 0
ar = np.zeros((nlev, nbin, ntstp))              # Aspect ratio
ar_out = np.zeros((nlev, nbin, ntstp))          # Outer aspect ratio
ar_in = np.zeros((nlev, nbin, ntstp))           # Inner (core) aspect ratio
arend = np.zeros((nlev, nbin, ntstp))           # Aspect ratio of equivalent-mass raindrops
dsdm = np.zeros((nlev, nbin, ntstp))            # Particle PSD [m-3]
mitot = np.zeros((nlev, ntstp))                 # Total IWC [g/m3]
mwtot = np.zeros((nlev, ntstp))                 # Total LWC (retained on particles) [g/m3]
sensible_flux = np.zeros((nlev, nbin, ntstp))   # Sensible heat term via conduction
latent_flux = np.zeros((nlev, nbin, ntstp))     # Latent heat term from evaporation/sublimation
heat = np.zeros((nlev, nbin, ntstp))            # Total heat available for melting
preciprate = np.zeros((nlev, ntstp))            # Precipitation rate [mm/hr] (includes hail)
preciprate_w = np.zeros((nlev, ntstp))          # Precipitation rate taking w into account [mm/hr] (includes hail) - UNCERTAIN
rainrate = np.zeros((nlev, ntstp))              # Rainrate [mm/hr]
rainrate_w = np.zeros((nlev, ntstp))            # Rainrate taking w into account

# Shedding arrays
mstot = np.zeros((nlev, ntstp))                 # Total mass content of shed water [g/m3]
mw_shed = np.zeros((nlev, nbin, ntstp))         # Mass of shed water from each hailstone size bin [kg]
dsd_shed = np.zeros((nlev, nbin, ntstp))        # DSD of shed drops 
nsh = np.zeros((nlev, ntstp))                   # Intercept parameter for shed drops DSD

# Breakup arrays
mbtot = np.zeros((nlev, ntstp))                 # Total mass of raindrops formed by breakup of melted particles [g/m3]
breakup_prob = np.zeros((nlev, nbin, ntstp))    # Probability of breakup within grid box
nbd = np.zeros((nlev, ntstp))                   # Number of breakup drops
dsd_breakup = np.zeros((nlev, nbin, ntstp))     # PSD of breakup drops

# Tendency variables
dTdt_melt = np.zeros((nlev, nbin, ntstp))       # Rate of temperature change due to melting within each bin [K/s]
dTdt_subl = np.zeros((nlev, nbin, ntstp))       # Rate of temperature change due to sublimation within each bin [K/s]
dTdt_evap = np.zeros((nlev, nbin, ntstp))       # Rate of temperature change due to evaporation [K/s]
dTdt_melt_tot = np.zeros((nlev, ntstp))         # Total rate of temperature change due to melting [K/s]
dTdt_subl_tot = np.zeros((nlev, ntstp))         # Total rate of temperature change due to sublimation [K/s]
dTdt_evap_tot = np.zeros((nlev, ntstp))         # Total rate of temperature change due to evaporation [K/s]
dTdt_tot = np.zeros((nlev, ntstp))              # Total rate of temperature change [K/s]
dqdt = np.zeros((nlev, nbin, ntstp))            # Change in environmental water vapor mixing ratio within each bin 
                                                # due to microphysical processes [(kg/kg)]
dqdt_tot = np.zeros((nlev, ntstp))              # Total rate of change of environmental water vapor mixing ratio
                                                # due to microphysical processes [(kg/kg)/s]

# Eq. (19) of Srivastava (1987)
dh_term1 = np.zeros((nlev, ntstp))              # Adiabatic warming term [K/s]
dh_term2 = np.zeros((nlev, ntstp))              # Vertical advection term [K/s]
dh_term3 = np.zeros((nlev, ntstp))              # Mixing term [K/s]
dh_term4 = np.zeros((nlev, ntstp))              # Diabatic cooling/moistening term [K/s]

# Eq. (24) of Srivastava (1987)
dw_term1 = np.zeros((nlev, ntstp))              # Vertical advection term [m/s2]
dw_term2 = np.zeros((nlev, ntstp))              # Buoyancy term [m/s2]
dw_term3 = np.zeros((nlev, ntstp))              # Precip loading term [m/s2]
dw_term4 = np.zeros((nlev, ntstp))              # Mixing term [m/s2]

# Eq. (18) of Srivastava (1987)
dN_term1 = np.zeros((nlev, nbin, ntstp))        # Divergence term [s-1]
dN_term2 = np.zeros((nlev, nbin, ntstp))        # Adveciton term [s-1]
dN_term3 = np.zeros((nlev, nbin, ntstp))        # Mixing term [s-1]

# Eq. (21) of Srivastava (1987)
dqstar_term1 = np.zeros((nlev, ntstp))          # Vapor advection term [(kg/kg)/s]
dqstar_term2 = np.zeros((nlev, ntstp))          # Precipitation advection term [(kg/kg)/s]
dqstar_term3 = np.zeros((nlev, ntstp))          # Divergence term [(kg/kg)/s]
dqstar_term4 = np.zeros((nlev, ntstp))          # Mixing term [(kg/kg)/s]
dqstar_term5 = np.zeros((nlev, ntstp))          # Microphysical moistening term [(kg/kg)/s]

# Radar arrays
ew = np.zeros((nlev, nbin, ntstp), dtype='complex')      # Dielectric of water
eps_in = np.zeros((nlev, nbin, ntstp), dtype='complex')  # Dielectric of mixed-phase particles inner core
eps_out = np.zeros((nlev, nbin, ntstp), dtype='complex') # Dielectric of mixed-phase particles outer layer
sigma = np.zeros((nlev, nbin, ntstp))                    # Std. Dev. of canting angle distribution
fhh_180 = np.zeros((nlev, nbin, ntstp), dtype='complex') # Horizontal backscatter amplitude (complex)
fvv_180 = np.zeros((nlev, nbin, ntstp), dtype='complex') # Vertical backscatter amplitude (complex)
fhh_0 = np.zeros((nlev, nbin, ntstp), dtype='complex')   # Horizontal forward scatter amplitude (complex)
fvv_0 = np.zeros((nlev, nbin, ntstp), dtype='complex')   # Vertical forward scatter amplitude (complex)
zhni = np.zeros((nlev, nbin, ntstp))                     # Zh for individual melting particle [mm6/m3]
zvni = np.zeros((nlev, nbin, ntstp))                     # Zv for individual melting particle [mm6/m3]
kdpi = np.zeros((nlev, nbin, ntstp))                     # Kdp for individual melting particle [deg/km]
ahi = np.zeros((nlev, nbin, ntstp))                      # Ah for individual melting particle [dB/km]
adpi = np.zeros((nlev, nbin, ntstp))                     # Adp for individual melting particle [dB/km]
deli = np.zeros((nlev, nbin, ntstp), dtype='complex')    # Delta for individual melting particle [deg]
ldri = np.zeros((nlev, nbin, ntstp))                     # Ldr for individual melting particle [dB]
zp = np.zeros((nlev, ntstp))                             # Total Z [dBZ]
zdrp = np.zeros((nlev, ntstp))                           # Total ZDR [dB]
kdpp = np.zeros((nlev, ntstp))                           # Total Kdp [deg/km]
ahp = np.zeros((nlev, ntstp))                            # Total Ah [dB/km]
adpp = np.zeros((nlev, ntstp))                           # Total Adp [dB/km]
delp = np.zeros((nlev, ntstp))                           # Total Delta [deg]
ldrp = np.zeros((nlev, ntstp))                           # Total linear depolarization ratio
rhvp = np.zeros((nlev, ntstp))                           # Total cross correlation coefficient
twolayer_useflag = np.zeros((nlev, nbin, ntstp))         # Flag whether 2-layer tables were used

# Generate scattering LUT for radar calculations
if generate_lut == True:
    fhh_180_lut, fvv_180_lut, fhh_0_lut, fvv_0_lut = rc.generate_scattering_tables(nbin)
    print('Scattering look-up table generation completed in ', datetime.now() - time_start)
    time_start = datetime.now()

if use_lut == True:
    if use_2layer == True: 
        # Read in saved 2-layer T-matrix look up tables
        # Note that I need to revisit this because the interpolation between the last valid
        # hailstone and the first invalid one is causing problems and discontinuities. 

        print('Reading in 2-layer T-matrix look-up tables')
        print('These tables are of the format:')
        print('Dout x Din x ARout x ARin x fvol x Temperature')
        ncdata = Dataset(twolayer_lut_path)
        
        # Read in scattering amplitudes from LUT
        fhh_180_lut = ncdata.variables['fhh_180_real'][:] + 1j * ncdata.variables['fhh_180_imag'][:]
        fvv_180_lut = ncdata.variables['fvv_180_real'][:] + 1j * ncdata.variables['fvv_180_imag'][:]
        fhh_0_lut = ncdata.variables['fhh_0_real'][:] + 1j * ncdata.variables['fhh_0_imag'][:]
        fvv_0_lut = ncdata.variables['fvv_0_real'][:] + 1j * ncdata.variables['fvv_0_imag'][:]
        
        d_vec = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5,
         8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,
         13.5, 14.0, 14.5, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
         23.0, 24.0, 26.0, 28.0, 30.0, 35.0, 40.0, 45.0]
        ar_vec = np.arange(0.5, 1.01, 0.1)
        fvol_vec = np.arange(0, 1.1, 0.1)
        t_vec = [0.0, 10.0, 20.0, 30.0, 40.0]
        
        # Put into Dout x Din x ARout x ARin x fvol x Temperature array format
        for ii in range(len(d_vec)): # Outer diameter
            for jj in range(len(d_vec)): # Inner diameter
                if jj == ii:
                    fhh_180_lut[ii, jj:, :, :, :, :] = np.conj(fhh_180_lut[ii, ii, :, :, :, :])
                    fvv_180_lut[ii, jj:, :, :, :, :] = np.conj(fvv_180_lut[ii, ii, :, :, :, :])
                    fhh_0_lut[ii, jj:, :, :, :, :] = np.conj(fhh_0_lut[ii, ii, :, :, :, :])
                    fvv_0_lut[ii, jj:, :, :, :, :] = np.conj(fvv_0_lut[ii, ii, :, :, :, :])
                    continue
        
        # Check for array grid points where parameters were selected such that
        # the inner particle protrudes from the outer particle. This will cause the T-matrix
        # calculations to be faulty because the T-matrix assumes concentric shells. 
        minor_axis_ratio = np.zeros((len(d_vec), len(d_vec), len(ar_vec), len(ar_vec)))
        major_axis_ratio = np.zeros((len(d_vec), len(d_vec), len(ar_vec), len(ar_vec)))
        
        for ii in range(len(d_vec)): # Outer diameter
            for jj in range(len(d_vec)): # Inner diameter
                for kk in range(len(ar_vec)): # Outer aspect ratio
                    for mm in range(len(ar_vec)): # Inner aspect ratio
    
                        minor_axis_outer = d_vec[ii] * ar_vec[kk]**(2./3.)
                        minor_axis_inner = d_vec[jj] * ar_vec[mm]**(2./3.)
                        
                        minor_axis_ratio[ii, jj, kk, mm] = minor_axis_inner / minor_axis_outer
                        
                        major_axis_outer = d_vec[ii] * ar_vec[kk]**(-1./3.)
                        major_axis_inner = d_vec[jj] * ar_vec[mm]**(-1./3.)
                        
                        major_axis_ratio[ii, jj, kk, mm] = major_axis_inner / major_axis_outer
                        
        bad_minor_axis = np.asarray(np.where(minor_axis_ratio > 1.0))
        bad_major_axis = np.asarray(np.where(major_axis_ratio > 1.0))
        
        # Attempt to address this by substituting in the "last good" grid point
        # I think this needs more work.
        for ii in range(np.shape(bad_minor_axis)[1]):
            idxs =  bad_minor_axis[:, ii]
            fhh_180_lut[idxs[0], idxs[1], idxs[2], idxs[3], :, :] = fhh_180_lut[idxs[0], idxs[1]-1, idxs[2], idxs[3], :, :]
            fvv_180_lut[idxs[0], idxs[1], idxs[2], idxs[3], :, :] = fvv_180_lut[idxs[0], idxs[1]-1, idxs[2], idxs[3], :, :]
            fhh_0_lut[idxs[0], idxs[1], idxs[2], idxs[3], :, :] = fhh_0_lut[idxs[0], idxs[1]-1, idxs[2], idxs[3], :, :]
            fvv_0_lut[idxs[0], idxs[1], idxs[2], idxs[3], :, :] = fvv_0_lut[idxs[0], idxs[1]-1, idxs[2], idxs[3], :, :]
            
        for ii in range(np.shape(bad_major_axis)[1]):
            idxs =  bad_major_axis[:, ii]
            fhh_180_lut[idxs[0], idxs[1], idxs[2], idxs[3], :, :] = fhh_180_lut[idxs[0], idxs[1], np.minimum(idxs[2]+1, 5), idxs[3], :, :]
            fvv_180_lut[idxs[0], idxs[1], idxs[2], idxs[3], :, :] = fvv_180_lut[idxs[0], idxs[1], np.minimum(idxs[2]+1, 5), idxs[3], :, :]
            fhh_0_lut[idxs[0], idxs[1], idxs[2], idxs[3], :, :] = fhh_0_lut[idxs[0], idxs[1], np.minimum(idxs[2]+1, 5), idxs[3], :, :]
            fvv_0_lut[idxs[0], idxs[1], idxs[2], idxs[3], :, :] = fvv_0_lut[idxs[0], idxs[1], np.minimum(idxs[2]+1, 5), idxs[3], :, :]           

        # Hack for dealing with NaNs due to bad conditions
        ii = np.where(np.isnan(fhh_180_lut))
        fhh_180_lut[ii] = 0.0 + 1j * 0.0

        ii = np.where(np.isnan(fvv_180_lut))
        fvv_180_lut[ii] = 0.0 + 1j * 0.0        
        
        ii = np.where(np.isnan(fhh_0_lut))
        fhh_0_lut[ii] = 0.0 + 1j * 0.0    
        
        ii = np.where(np.isnan(fvv_0_lut))
        fvv_0_lut[ii] = 0.0 + 1j * 0.0
        
    else:
        # Use single-layer scattering look-up tables for computing scattering amplitudes
        # NOTE: This is somewhat deprecated in favor of 2-layer T-matrix calculations
        print('Reading in 1-layer T-matrix look-up tables')
        lut = Dataset(lut_path)
        ar_vec = np.arange(0.56, 1.01, 0.01)
        d_vec = deld * np.arange(nbin)
        d_vec[0] = 1e-3

        fhh_180_lut = lut.variables['fhh_180_real'][:, :, :] + 1j * lut.variables['fhh_180_imag'][:, :, :]
        fvv_180_lut = lut.variables['fvv_180_real'][:, :, :] + 1j * lut.variables['fvv_180_imag'][:, :, :]
        fhh_0_lut = lut.variables['fhh_0_real'][:, :, :] + 1j * lut.variables['fhh_0_imag'][:, :, :]
        fvv_0_lut = lut.variables['fvv_0_real'][:, :, :] + 1j * lut.variables['fvv_0_imag'][:, :, :]
        lut.close()

# Specify initial downdraft conditions to be the same as the environment.
t[:, 0] = tenv
tk[:, 0] = t[:, 0] + t0
p[:, 0] = penv
rh[:, 0] = rhenv

# Constrain top of downdraft to be unchanging for duration of run with 
# conditions for cloud base from Srivastava et al. (1987) 
# (namely w = -1 m/s and RH = 100%)
rh[0, :] = 100.0
w[0, :] = -1.0

# Calculate the rest of downdraft environmental variables
es[:, 0] = sat_vapor_p(t[:, 0])
esi[:, 0] = sat_vapor_p(t[:, 0], i_flag=1)
qs[:, 0] = vapor_mixing_ratio(es[:, 0], p[:, 0])
q[:, 0] = (rh[:, 0] / 100.) * qs[:, 0]
e[:, 0] = (rh[:, 0] / 100.) * es[:, 0]
qc[:, 0] = 0.0
qstar[:, 0] = qc[:, 0] + q[:, 0]
h[:, 0] = (lv * q[:, 0] / cp) + tk[:, 0]
tv[:, 0] = virtual_temperature(tk[:, 0], q[:, 0])
rad[:, 0] = density_air(p[:, 0], tk[:, 0])
ra[:, 0] = density_air(p[:, 0], tv[:, 0])
eta[:, 0] = dynamic_viscosity_air(tk[:, 0])
nu[:, 0] = eta[:, 0] / ra[:, 0]
ka[:, 0] = thermal_diffusivity_air(tk[:, 0])
kair[:, 0] = thermal_conductivity_air(tk[:, 0])
kwa[:, 0] = thermal_conductivity_water(tk[:, 0])
dv[:, 0] = thermal_diffusivity_water(tk[:, 0], p[:, 0])
pr[:, 0] = nu[:, 0] / ka[:, 0]
sc[:, 0] = nu[:, 0] / dv[:, 0]

###############################################################################
############## Initialization of particles at freezing level ##################
###############################################################################

# Set particle density
if init_frozen_opt == False: # Start particles as rain
    rs[:] = rw
    graup_frac = np.zeros_like(rs) + 1.0 # Placeholder vector not used because fm == 1.0 everywhere
else: # Start particles as frozen
    if rs_opt == 0:
        rs[:] = ri
    elif rs_opt == 1: # Graupel density parameterization from Fig. 1 of Ryzhkov et al. (2013a).
        dcm = 0.01* np.arange(nbin)
        rs = 600 + 176*dcm - 24.7*dcm**2  # (Relation takes D in cm)
        rg = rs[0]
    elif rs_opt == 2: # Assume linearly increasing density from 600 to 917 kg/m3 from 0-6 mm particles
        rs_idx = np.where((0.5*deld + deld * np.arange(nbin)) <= 6.0)[0][-1]
        rs[0:rs_idx] = np.linspace(600, 917, rs_idx)
        rs[rs_idx:] = ri
        rg = rs[0]
    elif rs_opt == 3: # Assume constant graupel density (default 600 kg/m3) up to some size (default 5 mm),
                      # then linearly increasing up to solid ice at some size (default 1 cm)
        rs_idx_soft = np.where((0.5*deld + deld * np.arange(nbin)) <= 5.0)[0][-1]
        rs[0:rs_idx_soft] = rg
        rs_idx_solid = np.where((0.5*deld + deld * np.arange(nbin)) <= 10.0)[0][-1]
        rs[rs_idx_soft:rs_idx_solid] = np.linspace(rg, ri, rs_idx_solid-rs_idx_soft)
        rs[rs_idx_solid:] = ri
        
    # Define "graupel fraction" parameter for interpolating between graupel 
    # parameterization (Theis et al. 2022) and hail parameterization (RH87) 
    # based on density. 
    graup_frac = (ri - rs) / (ri - rg)

###############################################################################
################## Begin looping through model time steps #####################
###############################################################################
for tstp in range(ntstp-1):
    
    # Print current maximum downdraft velocity for tracking
    if verbose == True:
        print('Time: ', tstp * delt, 's | ', 'Minimum w: ', np.round(np.nanmin(w[:, tstp]), 2), 'm/s')

    # Create array of equally spaced particle size bins
    d[0, :, tstp] = 1e-3 * (0.5*deld + deld * np.arange(nbin))
    
    if init_frozen_opt == False: # Start as rain
        v[0, :, tstp] = (pi/6) * (d[0, :, tstp])**(3.)          # Particle total volume
        vw[0, :, tstp] = v[0, :, tstp]                          # Particle water volume
        mw[0, :, tstp] = rs * vw[0, :, tstp]                    # Particle water mass
        fm[0, :, tstp] = 1.0                                    # Particle meltwater fraction
        u[0, :, tstp] = (ras/ra[0, tstp])**(0.4) * mp.term_vel_rain(mw[0, :, tstp])
        ar_out[0, :, tstp] = ar_in[0, :, tstp] = ar[0, :, tstp] = mp.aspect_ratio_rain(d[0, :, tstp])
        dmax[0, :, tstp] =  d[0, :, tstp] * (ar[0, :, tstp])**(-1./3.)
        nre[0, :, tstp] = u[0, :, tstp] * dmax[0, :, tstp] / nu[0, tstp]
        fh[0, :, tstp] = 0.78 + 0.308 * (pr[0, tstp])**(1.0/3.0) * np.sqrt(nre[0, :, tstp])
        fv[0, :, tstp] = 0.78 + 0.308 * (sc[0, tstp])**(1.0/3.0) * np.sqrt(nre[0, :, tstp])
        sigma[0, :] = sigrain
                
        if dsd_norm: # Normalized rain DSD
            f_mu = (6 * (4 + mur)**(4 + mur)) / (4**4 * sp.special.gamma(4 + mur))
            dsdm[0, :, tstp] = (np.where(d[0, :, tstp] <= (1e-3 * dmax_limit),
                                         deld * (nrw * f_mu * ((d[0, :, tstp] * 1000.) / dmr)**mur * np.exp(-(4 + mur) * (d[0, :, tstp] * 1000.)/dmr)),
                                         0.0))    
        else: # Gamma rain DSD
            dsdm[0, :, tstp] = (np.where(d[0, :, tstp] <= (1e-3 * dmax_limit),
                                         deld * (nr0 * (d[0, :, tstp] * 1000.0)**(mur) * np.exp(-lamr * d[0, :, tstp] * 1000.0)),
                                         0.0))        
        
        mwtot[0, tstp] = np.nansum(mw[0, :, tstp] * dsdm[0, :, tstp]) * 1000.0
        qp[0, tstp] = 1e-3 * mwtot[0, tstp] / rad[0, tstp]
        vent_ratio = np.nanmean(fv[0, :, tstp] / fh[0, :, tstp]) # Use mean ratio for computational efficiency
        tp[0, :, tstp] = (optimize.minimize_scalar(mp.tp_conv_evap, 
                                                   bounds=(tenv[0]-20, tenv[0]+20), 
                                                   args=(eenv[0], tenv[0], kair[0, tstp], dv[0, tstp], vent_ratio)).x)
    
        if tstp == 0:
            print('Initial LWC: ', mwtot[0, 0], ' g/m3')
    
    else: # Starting as graupel/hail
        di[0, :, tstp] = d[0, :, tstp]                      # Inner (ice) diameter initially == D
        v[0, :, tstp] = (pi/6) * (d[0, :, tstp])**(3.)      # Total volume
        vi[0, :, tstp] = (rs/ri) * v[0, :, tstp]            # Ice volume
        mi[0, :, tstp] = rs * v[0, :, tstp]                 # Ice mass
        va[0, :, tstp] = (1 - (rs/ri)) * v[0, :, tstp]      # Air volume
        fm[0, :, tstp] = 0.0                                # Meltwater fraction

        nre[0, :, tstp] = mp.reynolds_num_theis(mi[0, :, tstp],
                                                ra[0, tstp],
                                                eta[0, tstp]) # Reynolds number (uncertain!)
    
        # Calculate ventilation coefficients (Pruppacher and Klett)
        fh[0, :, tstp] = 0.78 + 0.308 * (pr[0, tstp])**(1.0/3.0) * np.sqrt(nre[0, :, tstp])
        fv[0, :, tstp] = 0.78 + 0.308 * (sc[0, tstp])**(1.0/3.0) * np.sqrt(nre[0, :, tstp])
    
    
        # Calculate initial aspect ratio of particles
        # !!! NOTE: this is highly uncertain and deserves further interrogation.
        # Retaining other explored options for posterity.
        
        # Aspect ratio parameterization from Ryzhkov et al. (2011) of AR decreasing
        # linearly from 1.0 to 0.8 up to D = 1.0 cm and 0.8 thereafter.
        #ar[0, :, tstp] = np.maximum(1.0 - 0.2 * (d[0, :, tstp] / 0.01), 0.8) 
        
        # Newly implemented parameterization from Shedd (2021) using manually measured
        # hailstone relation to convert Deq to Dmax. Used scipy.optimize.curve_fit with
        # 95% confidence interval estimates as weights (didn't affect much). 
        #ar[0, :, tstp] = np.clip(0.886 * (1e3*d[0, :, tstp])**(-0.116), 0.5, 0.9)
    
        # New parameterization from  Table 1 of Lin et al. (2024) converted to Deq
        ar_idx_1 = np.where((1e3 * d[0, :, tstp]) <= 14.0)[0][-1]
        ar[0, 0:ar_idx_1, tstp] = np.linspace(ar_g, 0.7, ar_idx_1)
        #ar[0, ar_idx_1:, tstp] = 2.08e-5*(1e3 * d[0, :, tstp])**2 - 5.66e-3*(1e3 * d[0, :, tstp]) + 6.794e-1
        ar[0, ar_idx_1:, tstp] = 0.75 - (0.00357143 * 1e3 * d[0, ar_idx_1:, tstp])
        ar_out[0, :, tstp] = ar_in[0, :, tstp] = ar[0, :, tstp]
        
        # Current parameterization: Increasingly oblate hailstones with graupel aspect 
        # ratio below 1 cm, and linearly decreasing to an aspect ratio of 0.7 by D = 5 cm.
        # Note this is more spherical than a lot of observational studies indicate. 
        #ar_idx_1 = np.where((1e3 * d[0, :, tstp]) <= 10.0)[0][-1]
        #ar[0, 0:ar_idx_1, tstp] = ar_g
        #ar_idx_2 = np.where((1e3 * d[0, :, tstp]) <= 50.0)[0][-1]
        #ar[0, ar_idx_2:, tstp] = 0.7
        #ar[0, ar_idx_1:ar_idx_2, tstp] = np.linspace(ar_g, 0.7, ar_idx_2-ar_idx_1)
        #ar_out[0, :, tstp] = ar[0, :, tstp]
        #ar_in[0, :, tstp] = ar[0, :, tstp]

        # Calculate particle Dmax assuming oblate spheroid
        # Note that Shedd et al. (2021) showed that hailstones are likely
        # better represented by triaxial ellipspoids
        dmax[0, :, tstp] =  d[0, :, tstp] * (ar[0, :, tstp])**(-1./3.)
    
        # Calculate particle terminal velocity as a density-weighted function between
        # graupel and hail parameteriations
        u[0, :, tstp] = mp.term_vel_dry(nre[0, :, tstp],
                                        nu[0, tstp],
                                        dmax[0, :, tstp])
        
        # Set initial canting angle width to that of dry hail
        sigma[0, :] = sighail

        ##########################################################################
        # If using time-varying PSDs, manually code them in for each timestep here
        ##########################################################################
        # if (tstp < 1000) or (tstp > 2000):
        #     ng0 = 5000
        #     lamg = 1.5
        #     dmax_limit = 6.0
        # else:
        #     ng0 = 10000
        #     lamg = 1.2
        #     dmax_limit = 14.0
        
        # Bi-exponential particle size distribution in m-3
        dsdm[0, :, tstp] = (np.where(d[0, :, tstp] <= (1e-3 * dmax_limit),
                                     deld * (nh0 * np.exp(-lamh * d[0, :, tstp] * 1000.0) +
                                             ng0 * np.exp(-lamg * d[0, :, tstp] * 1000.0)),
                                     0.0))       
            
        mitot[0, tstp] = np.nansum(mi[0, :, tstp] * dsdm[0, :, tstp]) * 1000.0 # [g]
        qp[0, tstp] = 1e-3 * mitot[0, tstp] / rad[0, tstp]

        # Calculate initial particle temperature (Theis et al. 2022, Eq. 14)
        # Note: Because of the assumption of 100% RH at cloud base, this technically results in a 
        # marginally >0C temperature to start, but microphysics do not begin until the next level
        vent_ratio = np.nanmean(fv[0, :, tstp] / fh[0, :, tstp]) # Use mean of ratio for speed
        tp[0, :, tstp] = (optimize.minimize_scalar(mp.tp_conv_subl, 
                                                   bounds=(tenv[0]-20, tenv[0]+20), 
                                                   args=(eenv[0], tenv[0], kair[0, tstp], dv[0, tstp], vent_ratio)).x)

    if tstp == 0:
        print('Initial IWC: ', mitot[0, 0], ' g/m3')

###############################################################################
#################### Begin looping through model levels #######################
###############################################################################
    for hgt in range(1, nlev):

        # Find empty bins where particles have disappeared and put in dummy values.
        empty_bins = ((mi[hgt-1, :, tstp] + mw[hgt-1,:,tstp]) <= 0.0).nonzero()[0]
        if len(empty_bins) > 0:
            d[hgt, empty_bins, tstp] = 0.0
            di[hgt, empty_bins, tstp] = 0.0
            u[hgt, empty_bins, tstp] = 0.0
            mi[hgt, empty_bins, tstp] = mi[hgt-1, empty_bins, tstp]
            mw[hgt, empty_bins, tstp] = mw[hgt-1, empty_bins, tstp]
            fm[hgt, empty_bins, tstp] = 0.0
            fm_ar[hgt, empty_bins, tstp] = 0.0
            ar[hgt, empty_bins, tstp] = 1.0
            arend[hgt, empty_bins, tstp] = 0.0
            ar_in[hgt, empty_bins, tstp] = 1.0
            ar_out[hgt, empty_bins, tstp] = 1.0
            tp[hgt, empty_bins, tstp] = 0.0
            dsdm[hgt, empty_bins, tstp] = 0.0

        # Find full bins
        full_bins = ((mi[hgt-1, :, tstp] + mw[hgt-1,:,tstp]) > 0.0).nonzero()[0]

        if len(full_bins) > 0:
            # Calculate Reynolds numbers
            # !!! NOTE: This is still highly uncertain and there are inconsistencies with how
            # particles are initialized vs. treated here. 
            nre_hail = mp.reynolds_num_hail((mi[hgt-1, full_bins, tstp] + mw[hgt-1, full_bins, tstp]),
                                            ra[hgt, tstp],
                                            eta[hgt, tstp])
            nre_graupel = mp.reynolds_num_graupel((mi[hgt-1, full_bins, tstp] + mw[hgt-1, full_bins, tstp]),
                                                   ra[hgt, tstp],
                                                   eta[hgt, tstp])
            # Calculate raindrop terminal velocity of equivalent mass. 
            u_rain = mp.term_vel_rain(mi[hgt-1, full_bins, tstp] + mw[hgt-1, full_bins, tstp]) # Density correction needed here?

            if init_frozen_opt == False:
                u[hgt, full_bins, tstp] = (ras/ra[hgt, tstp])**(0.4) * u_rain
                nre[hgt, full_bins, tstp] = u_rain * dmax[hgt-1, full_bins, tstp] / nu[hgt, tstp]

            # Deprecated code for how to linearly weight Nre for rain, graupel, and hail.
            # d_idx = np.where(d[hgt-1, full_bins, tstp] > 0.008)[0]
            #nre[hgt, full_bins, tstp] = (fm[hgt-1, full_bins, tstp] * nre_rain) + (1 - fm[hgt-1, full_bins, tstp]) * (graup_frac[full_bins] * nre_graupel + (1 - graup_frac[full_bins]) * nre_hail)

            else: # Calculate Nre and terminal velocit for graupel/hail
                # Find indices where meltwater soaking is first complete.
                # This helps alleviate some assumptions in the Phillips melting model
                soaked_idxs = [-1] * nbin
                for col in range(nbin):
                    for row in range(0, hgt+1):
                        if va[row][col][tstp] == 0:
                            soaked_idxs[col] = row
                            break
                        
                # Particle characteristics for the point where soaking is just completed.
                d_justsoaked = d[soaked_idxs, np.arange(nbin), tstp]
                m_justsoaked = mi[soaked_idxs, np.arange(nbin), tstp] + mw_inside[soaked_idxs, np.arange(nbin), tstp]
                
                nre[hgt, full_bins, tstp], u[hgt, full_bins, tstp] = term_vel_hail_phillips_vec(nre_hail, 
                                                                                                d[hgt-1, full_bins, tstp], 
                                                                                                di[hgt-1, full_bins, tstp], 
                                                                                                dmax[hgt-1, full_bins, tstp],
                                                                                                ar_in[hgt-1, full_bins, tstp],
                                                                                                mi[hgt-1, full_bins, tstp], 
                                                                                                mw_inside[hgt-1, full_bins, tstp], 
                                                                                                mw_outside[hgt-1, full_bins, tstp], 
                                                                                                va[hgt-1, full_bins, tstp], 
                                                                                                va[0, full_bins, tstp],
                                                                                                rs[full_bins], 
                                                                                                nu[hgt, tstp], 
                                                                                                ra[hgt, tstp], 
                                                                                                eta[hgt, tstp],
                                                                                                nu[-1, tstp],
                                                                                                ra[-1, tstp],
                                                                                                d_justsoaked[full_bins],
                                                                                                m_justsoaked[full_bins])

            # Calculate ventilation coefficients
            fh[hgt, full_bins, tstp] = 0.78 + 0.308 * (pr[hgt, tstp])**(1.0/3.0) * np.sqrt(nre[hgt, full_bins, tstp])
            fv[hgt, full_bins, tstp] = 0.78 + 0.308 * (sc[hgt, tstp])**(1.0/3.0) * np.sqrt(nre[hgt, full_bins, tstp])

            # Calculate particle temperatures....
            vent_ratio = np.nanmean(fv[hgt, full_bins, tstp] / fh[hgt, full_bins, tstp])
            
            # Temperature if particle is dry and undergoing sublimation
            tp_subl = np.minimum((optimize.minimize_scalar(mp.tp_conv_subl, 
                                                bounds=(t[hgt, tstp]-20, t[hgt, tstp]+20), 
                                                args=(e[hgt, tstp], t[hgt, tstp], kair[hgt, tstp], dv[hgt, tstp], vent_ratio)).x), 0.0)
            
            # Temperature if particle is fully melted and undergoing evaporation
            tp_evap = (optimize.minimize_scalar(mp.tp_conv_evap, 
                                                bounds=(t[hgt, tstp]-20, t[hgt, tstp]+20), 
                                                args=(e[hgt, tstp], t[hgt, tstp], kair[hgt, tstp], dv[hgt, tstp], vent_ratio)).x)
    
            tp_where = np.where(fm[hgt-1, :, tstp] == 0.0)[0]
            if len(tp_where) > 0:
                tp[hgt, tp_where, tstp] = tp_subl
                
            tp_where = np.where(fm[hgt-1, :, tstp] == 1.0)[0]
            if len(tp_where) > 0:
                tp[hgt, tp_where, tstp] = tp_evap
                
            # If meltwater soaking is occurring, particle remains at 0C during melting (Theis et al. 2022)
            tp_where = np.where((fm[hgt-1, :, tstp] > 0.0) & (fm[hgt-1, :, tstp] < 1.0) & ((va[hgt-1, :, tstp] > 0) | ((nre[hgt, :, tstp] < 3e3) | (nre[hgt, :, tstp] > 6e3))))[0]
            if len(tp_where) > 0:
                tp[hgt, tp_where, tstp] = 0.0
                
            tp[hgt, empty_bins, tstp] = np.nan
            
            # Find surface temperature with meltwater layer for 3000 < Nre < 6000 where meltwater doesn't circulate
            # and distribute temperature gradient (RH87)
            # !!! It is unclear if this should be applied to all graupel with surface meltwater 
            # or only between these Nre ranges
            # This has been modified to start at Nre of 750 based on the results fro graupel from Theis et al. (2022)
            # but remains highly uncertain and is deserving of further exploration.
            tp_where = np.where((fm[hgt-1, :, tstp] > 0.0) & (fm[hgt-1, :, tstp] < 1.0) & (va[hgt-1, :, tstp] == 0) &
                                (nre[hgt, :, tstp] >= 750) & (nre[hgt, :, tstp] <= 6e3))[0]
            if len(tp_where) > 0:
                for particle in tp_where:
                    if di[hgt-1, particle, tstp] == d[hgt-1, particle, tstp]:
                        # Retain Tp = 0.0 for point where melting begins and 
                        # there is no meltwater layer
                        tp[hgt, particle, tstp] = 0.0
                    else:
                        tp[hgt, particle, tstp] = (np.maximum(optimize.minimize_scalar(mp.tp_conv_melt_graup, 
                                                                            bounds=(t[hgt, tstp]-20, t[hgt, tstp]+20), 
                                                                            args=(e[hgt, tstp], t[hgt, tstp], kair[hgt, tstp], 
                                                                                  fh[hgt, particle, tstp], lv, dv[hgt, tstp],
                                                                                  fv[hgt, particle, tstp], di[hgt-1, particle, tstp],
                                                                                  kwa[hgt, tstp], rv, d[hgt-1, particle, tstp],
                                                                                  mw_inside[hgt-1, particle, tstp], mi[hgt-1, particle, tstp], 
                                                                                  rs[particle], 1.2)).x, 0.0)) 
                        
###############################################################################
############################ Begin microphysics ###############################
###############################################################################

        # Sublimation should only occur when there is no meltwater coating and the 
        # particle's surface temperature is below freezing
        subl_condition_1 = np.less(tp[hgt, :, tstp], 0.0)
        subl_condition_2 = np.equal(mw_outside[hgt-1, :, tstp], 0.0)
        subl_condition_3 = np.greater(mi[hgt-1, :, tstp], 0.0)
        
        if np.any(subl_condition_1 * subl_condition_2 * subl_condition_3):
            subl_where = np.logical_and(np.logical_and(subl_condition_1, subl_condition_2), subl_condition_3)

            # Enhancement factor from Theis et al. (2022) to approximate effects
            # from the melting of lumpy graupel, interpolated as a function of density
            # !!! Note: This value is uncertain and hasn't been used in past literature.
            Fsub_vec = graup_frac * Fsub + (1 - graup_frac) 
            
            estp = th.sat_vapor_p(tp[hgt, subl_where, tstp], i_flag=1)
            drho = (1 / rv) * ((e[hgt, tstp] / (t[hgt, tstp] + t0)) - (estp / (tp[hgt, subl_where, tstp] + t0)))
            dmidt = 4 * pi * (0.5 * d[hgt-1, subl_where, tstp]) * dv[hgt, tstp] * fv[hgt, subl_where, tstp] * drho * Fsub_vec[subl_where] # kg/s
            if subl_opt == False:
                dmidt = 0.0
            dTdt_subl[hgt, subl_where, tstp] = (ls / (cp * rad[hgt, tstp]) * dmidt) # (K m3) / s (Cooling rate per particle)
            dqdt[hgt, subl_where, tstp] = -1.0 * dmidt / rad[hgt, tstp] # ((kg/kg) m3) / s (Moistening rate per particle)
                
            
            # Use chain rule to determine total mass loss in grid box (i.e., dm/dt = dm/dz * dz/dt)
            # Actual mass evolution is governed by timestep-driven evolution of N
            # This follows e.g. Kumjian et al. (2010) and agrees with Srivastava's model output
            dmi = (dmidt * dh) / (u[hgt, subl_where, tstp] - w[hgt, tstp]) # kg
            dvi[hgt, subl_where, tstp] = np.maximum(dmi / ri, -vi[hgt-1, subl_where, tstp]) # m3 (Per particle)            
            vi[hgt, subl_where, tstp] = np.maximum(vi[hgt-1, subl_where, tstp] + dvi[hgt, subl_where, tstp], 0.0)  # New volume of ice
            mi[hgt, subl_where, tstp] = ri * vi[hgt, subl_where, tstp]
            # Assume constant density during sublimation (Validity? -- fine for solid ice)
            va[hgt, subl_where, tstp] = np.maximum(va[hgt-1, subl_where, tstp] + (1 - (rs[subl_where]/ri)) * dvi[hgt, subl_where, tstp], 0.0)
            vw[hgt, subl_where, tstp] = vw[hgt-1, subl_where, tstp]
            v[hgt, subl_where, tstp] = (vi[hgt, subl_where, tstp] +
                                        va[hgt, subl_where, tstp] +
                                        vw[hgt, subl_where, tstp])
            d[hgt, subl_where, tstp] = ((6/pi) * v[hgt, subl_where, tstp])**(1.0 / 3.0)
            di[hgt, subl_where, tstp] = ((6 / pi) * (vi[hgt, subl_where, tstp] + va[hgt, subl_where, tstp] + (mw_inside[hgt, subl_where, tstp] / rw)))**(1./3.)    

        # Evaporation should only occur when there is no ice left in the particle
        # (Note: Evaporation during melting is handled within melting routine)
        evap_condition_1 = np.equal(fm[hgt-1, :, tstp], 1.0)
        evap_condition_2 = np.greater(mw[hgt-1, :, tstp], 0.0)
        if np.any(evap_condition_1 * evap_condition_2):
            evap_where = np.logical_and(evap_condition_1, evap_condition_2)
            
            estp = sat_vapor_p(tp[hgt, evap_where, tstp])
            drho = (1 / rv) * ((e[hgt, tstp] / (t[hgt, tstp] + t0)) - (estp / (tp[hgt, evap_where, tstp] + t0)))            
            dmwdt = 4 * pi * (0.5 * d[hgt-1, evap_where, tstp]) * dv[hgt, tstp] * fv[hgt, evap_where, tstp] * drho # kg/s
            if evap_opt == False:
                dmwdt = 0.0
            dTdt_evap[hgt, evap_where, tstp] = (lv / (cp * rad[hgt, tstp]) * dmwdt) # (K m3) / s (Cooling rate per particle)
            dqdt[hgt, evap_where, tstp] = -1.0 * dmwdt / rad[hgt, tstp] # ((kg/kg) m3) / s (Moistening rate per particle)
                       
            # Use chain rule to determine total mass loss in grid box (i.e., dm/dt = dm/dz * dz/dt)
            # Actual mass evolution is governed by timestep-driven evolution of N
            # This follows e.g. Kumjian et al. (2010) and agrees with Srivastava's model output
            dmw = (dmwdt * dh) / (u[hgt, evap_where, tstp] - w[hgt, tstp]) # kg
            dvw[hgt, evap_where, tstp] = np.maximum(dmw / rw, -vw[hgt-1, evap_where, tstp]) # m3 (Per particle)
            vw[hgt, evap_where, tstp] = np.maximum(vw[hgt-1, evap_where, tstp] + dvw[hgt, evap_where, tstp], 0.0)  # New volume of water
            va[hgt, evap_where, tstp] = 0.0
            vi[hgt, evap_where, tstp] = 0.0
            mw[hgt, evap_where, tstp] = rw * vw[hgt, evap_where, tstp]
            mw_outside[hgt, evap_where, tstp] = mw[hgt, evap_where, tstp]
            v[hgt, evap_where, tstp] = vw[hgt, evap_where, tstp]
            d[hgt, evap_where, tstp] = ((6/pi) * v[hgt, evap_where, tstp])**(1.0 / 3.0)   
            di[hgt, evap_where, tstp] = 0.0

        # Melt conditions:
        # Here, melting (and potentially evaporation) of melting particles
        # is considered.
        #
        # The equation for heat balance of a melting ice particle is
        # (Srivastava et al. 1987, Eq. 5):
        #
        # dQenv     dmw   dQp      dmi
        # ----- + Lv--- = --- = -Lf---
        #  dt       dt    dt       dt
        #
        #  (1)      (2)   (3)     (4)
        #
        # where
        #
        # dQenv/dt is the rate of energy input from the environment through
        # conduction ~= 4*pi*r*fh*kair*(Tenv-Tp)
        #
        # Lv*dmw/dt is the rate of energy input from cooling/warming due to
        # evaporation/condensation ~= 4*pi*r*fv*Lv*Dv*(rhoenv - rhop)
        #
        # dQp/dt is the total heat available for melting ice in the particle
        #
        # Current understanding:
        # The change of environmental T is due solely to the conduction of
        # heat (dQenv/dt). (The corrolary for evaporation and sublimation exists,
        # it is just that dQp/dt = 0 (i.e., there is no net flow through from
        # the surface layer to the ice layer because there is no coating and 
        # the particle surface temp is in equilibrium), so that the sensible heat 
        # flux and the latent heat flux are balanced and the environmental 
        # temperature change can equivalently be found via either).... I think.
        # Evaporation (condensation) has to occur if the
        # vapor density at the particle surface exceeds (is smaller than) the
        # environmental vapor density. This uses (provides) heat from (to)
        # inside the particle, which consumes (adds to) the heat supplied from
        # the conduction. Szyrmer and Zawadzki (1999) defined the rate of
        # change as
        #
        # dTenv        Lf             kair*(Tenv-Tp)           dmi
        # ----- = - ------- ---------------------------------  ---
        #  dt       cp*rhoa kair*(Tenv-Tp)+Lv*Dv*(rhoenv-rhop) dt
        #
        # which can be re-arranged to show that
        #
        #         -4*pi*r*fh*kair*(Tenv-Tp)
        # dTenv = -------------------------
        #                 cp*rhoa
        #
        # The evaporation equations used below are modified according to those
        # reported in Rasmussen et al. (1987a) for hailstones.

        # Melting occurs if there is at least some ice left and the particle temperature 
        # is greater than or equal to 0C
        melt_condition_1 = np.less(fm[hgt-1, :, tstp], 1.0)
        melt_condition_2 = np.greater_equal(tp[hgt, :, tstp], 0.0)
        if np.any(np.logical_and(melt_condition_1, melt_condition_2)):
            
            melt_where = np.where(np.logical_and(melt_condition_1, melt_condition_2))[0]
            estp = th.sat_vapor_p(tp[hgt, melt_where, tstp])
            drho = (1 / rv) * ((e[hgt, tstp] / (t[hgt, tstp] + t0)) - (estp / (tp[hgt, melt_where, tstp] + t0)))               

            # Modified ventilation coefficient and relevant capacitance based on RH87
            # See: Table 1 of RH87 or Eqs. (14)-(17) of Phillips et al. (2007)
            fv_tmp = np.zeros((nbin)) 
            fh_tmp = np.zeros((nbin))
            c = np.zeros((nbin))

            # Where NRe < 250
            nre_cond = (nre[hgt, melt_where, tstp] < 250)
            nrebins = melt_where[nre_cond]
            fv_tmp[nrebins] = 2.0 * fv[hgt, nrebins, tstp]
            fh_tmp[nrebins] = 2.0 * fh[hgt, nrebins, tstp]
            c[nrebins] = 0.5 * d[hgt-1, nrebins, tstp]
            
            # Where 250 <= NRe < 3000
            nre_cond = np.logical_and((nre[hgt, melt_where, tstp] >= 250), (nre[hgt, melt_where, tstp] < 3000))
            nrebins = melt_where[nre_cond]
            fv_tmp[nrebins] = fv[hgt, nrebins, tstp]
            fh_tmp[nrebins] = fh[hgt, nrebins, tstp]            
            c[nrebins] = 0.5 * d[hgt-1, nrebins, tstp]
            
            # Where 3000 <= NRe < 6000
            nre_cond = np.logical_and((nre[hgt, melt_where, tstp] >= 3000), (nre[hgt, melt_where, tstp] < 6000))
            nrebins = melt_where[nre_cond]
            fv_tmp[nrebins] = fv[hgt, nrebins, tstp]
            fh_tmp[nrebins] = fh[hgt, nrebins, tstp]            
            c[nrebins] = 0.5 * d[hgt-1, nrebins, tstp]

            # Where 6000 <= NRe < 20000
            nre_cond = np.logical_and((nre[hgt, melt_where, tstp] >= 6000), (nre[hgt, melt_where, tstp] < 20000))
            nrebins = melt_where[nre_cond]
            ksi = 0.76
            fv_tmp[nrebins] = 0.5 * ksi * np.sqrt(nre[hgt, nrebins, tstp]) * (sc[hgt, tstp])**(1./3.)
            fh_tmp[nrebins] = 0.5 * ksi * np.sqrt(nre[hgt, nrebins, tstp]) * (pr[hgt, tstp])**(1./3.)
            c[nrebins] = 0.5 * di[hgt-1, nrebins, tstp]
            
            # Where NRe > 20000
            nre_cond = (nre[hgt, melt_where, tstp] > 20000)
            nrebins = melt_where[nre_cond]
            ksi = 0.57 + 9.0E-6*nre[hgt, nrebins, tstp]
            fv_tmp[nrebins] = 0.5 * ksi * np.sqrt(nre[hgt, nrebins, tstp]) * (sc[hgt, tstp])**(1./3.)
            fh_tmp[nrebins] = 0.5 * ksi * np.sqrt(nre[hgt, nrebins, tstp]) * (pr[hgt, tstp])**(1./3.)            
            c[nrebins] = 0.5 * di[hgt-1, nrebins, tstp]
            
            # Mass loss due to evaporation
            dmwdt_evap = 4.0 * pi * c[melt_where] * dv[hgt, tstp] * fv_tmp[melt_where] * drho
            dqdt[hgt, melt_where, tstp] = -1.0 * dmwdt_evap / rad[hgt, tstp] # ((kg/kg) m3) / s (Moistening rate per particle)
                   
            # Total heat leftover for melting (i.e., conduction heat minus that used up by evaporation)
            latent_flux[hgt, melt_where, tstp] = dmwdt_evap * lv
            sensible_flux[hgt, melt_where, tstp] = 4.0 * pi * c[melt_where] * kair[hgt, tstp] * fh_tmp[melt_where] * (t[hgt, tstp] - tp[hgt, melt_where, tstp])
            heat[hgt, melt_where, tstp] = np.maximum(sensible_flux[hgt, melt_where, tstp] + latent_flux[hgt, melt_where, tstp], 0.0) # Enforces no negative heat for melting
            
            # Where is net heating available for melting?
            net = np.where(heat[hgt, melt_where, tstp] > 0.0)
            
            if np.count_nonzero(net) > 0: # Melting occurs!
                # Mass loss due to melting
                dmidt = -1.0 * (heat[hgt, melt_where[net], tstp] / lf)
                # Total heat exchange during melting
                dTdt_melt[hgt, melt_where[net], tstp] = -1.0 * sensible_flux[hgt, melt_where[net], tstp] / (rad[hgt, tstp] * cp)
                # Reconcile total mass changes
                dmw_evap = (dmwdt_evap[net] * dh) / (u[hgt, melt_where[net], tstp] - w[hgt, tstp]) # kg  
                dmi_melt = (dmidt * dh) / (u[hgt, melt_where[net], tstp] - w[hgt, tstp]) # kg
                
                # Enforce that we can only evaporate as much meltwater as we had to start
                dvi[hgt, melt_where[net], tstp] = np.maximum(dmi_melt / ri, -vi[hgt-1, melt_where[net], tstp])
                vi[hgt, melt_where[net], tstp] = vi[hgt-1, melt_where[net], tstp] + dvi[hgt, melt_where[net], tstp]  # New volume of ice
                mi[hgt, melt_where[net], tstp] = ri * vi[hgt, melt_where[net], tstp]
                dvw_evap[hgt, melt_where[net], tstp] = np.maximum(dmw_evap / rw, -vw[hgt-1, melt_where[net], tstp]) # m3 (Individual particle)
                dvw[hgt, melt_where[net], tstp] = dvw_evap[hgt, melt_where[net], tstp] + (-(dvi[hgt, melt_where[net], tstp] * ri) / rw)
                
                ##############################################################
                #################### Distribute meltwater ####################
                ##############################################################
                
                # Assume we lose air volume to melting proportional to the fraction 
                # of ice lost to melting (i.e., that density is constant throughout particle).
                # Without this, ice sphere would shrink but the total volume of air would
                # only decrease by filling up with meltwater, not to volume of inner
                # core shrinking overall. 
                
                unfilled = np.where(va[hgt-1, melt_where[net], tstp] > 0.0)
                
                if len(unfilled[0]) > 0:
                
                    va_melting_loss = (dvi[hgt, melt_where[net][unfilled], tstp] / vi[hgt-1, melt_where[net][unfilled], tstp]) * va[hgt-1, melt_where[net][unfilled], tstp]
                    va_melting_loss[va[hgt-1, melt_where[net][unfilled], tstp] == 0.0] = 0.0
        
                    # Meltwater able to be soaked into remaining air volume
                    vw_retained = np.minimum(dvw[hgt, melt_where[net][unfilled], tstp], (va[hgt-1, melt_where[net][unfilled], tstp] + va_melting_loss))
                    vw_retained[va[hgt-1, melt_where[net][unfilled], tstp] == 0.0] = 0.0
                    
                    # New air volume with losses due to melting and soaking
                    va[hgt, melt_where[net][unfilled], tstp] = np.maximum(va[hgt-1, melt_where[net][unfilled], tstp] + va_melting_loss - vw_retained, 0.0)
                    mw_inside[hgt, melt_where[net][unfilled], tstp] = mw_inside[hgt-1, melt_where[net][unfilled], tstp] + rw * vw_retained
                    mw_outside[hgt, melt_where[net][unfilled], tstp] = mw_outside[hgt-1, melt_where[net][unfilled], tstp] + rw * (dvw[hgt, melt_where[net][unfilled], tstp] - vw_retained)
                    
                # For the case where no more air remains in the particle, 
                # we should be melting mi and transferring a proportional amount 
                # of mw_in to mw_out as the water is "leaked". Previously, the
                # meltwater that was initially internal remained internal, which
                # drastically underestimated the amount of water available for shedding....
                # Assumes a homogeneous distribution of mw_inside
                
                filled = np.where(va[hgt-1, melt_where[net], tstp] == 0.0)
                
                if len(filled[0]) > 0:
                    va[hgt, melt_where[net][filled], tstp] = va[hgt-1, melt_where[net][filled], tstp]
                    
                    mw_inside_melting_loss = (dvi[hgt, melt_where[net][filled], tstp] / vi[hgt-1, melt_where[net][filled], tstp]) * (mw_inside[hgt-1, melt_where[net][filled], tstp])
                    mw_inside[hgt, melt_where[net][filled], tstp] = mw_inside[hgt-1, melt_where[net][filled], tstp] + mw_inside_melting_loss
                    mw_outside[hgt, melt_where[net][filled], tstp] = mw_outside[hgt-1, melt_where[net][filled], tstp] + (dvw[hgt, melt_where[net][filled], tstp] * rw) - mw_inside_melting_loss

            # Case for which evaporation consumes all the heat that would've been used for melting
            # Either vaporation or sublimation occurs depending on whether the particle surface is wet
            nonet = np.where(heat[hgt, melt_where, tstp] <= 0.0)[0]
            
            if np.count_nonzero(nonet) > 0:
                evap_where = np.where(mw_outside[hgt-1, melt_where[nonet], tstp] > 0.)[0]
                if np.count_nonzero(evap_where) > 0:
                    estp = sat_vapor_p(tp[hgt, melt_where[nonet][evap_where], tstp])
                    drho = (1 / rv) * ((e[hgt, tstp] / (t[hgt, tstp] + t0)) - (estp / (tp[hgt, melt_where[nonet][evap_where], tstp] + t0)))            
                    dmwdt = 4 * pi * (0.5 * d[hgt-1, melt_where[nonet][evap_where], tstp]) * dv[hgt, tstp] * fv_tmp[melt_where[nonet][evap_where]] * drho # kg/s
                    if evap_opt == False:
                        dmwdt[:] = 0.0
                    dTdt_evap[hgt, melt_where[nonet][evap_where], tstp] = (lv / (cp * rad[hgt, tstp]) * dmwdt) # (K m3) / s (Cooling rate per particle)
                    dqdt[hgt, melt_where[nonet][evap_where], tstp] = -1.0 * dmwdt / rad[hgt, tstp] # ((kg/kg) m3) / s (Moistening rate per particle)
                    
                    dmw = (dmwdt * dh) / (u[hgt,  melt_where[nonet][evap_where], tstp] - w[hgt, tstp]) # kg
                    dvw[hgt, melt_where[nonet][evap_where], tstp] = np.maximum(dmw, -mw_outside[hgt-1, melt_where[nonet][evap_where], tstp]) / rw # m3 (Per particle)
                    mw_outside[hgt, melt_where[nonet][evap_where], tstp] = mw_outside[hgt-1, melt_where[nonet][evap_where], tstp] + dvw[hgt, melt_where[nonet][evap_where], tstp] * rw
                    vw[hgt, melt_where[nonet][evap_where], tstp] = vw[hgt-1, melt_where[nonet][evap_where], tstp] + dvw[hgt, melt_where[nonet][evap_where], tstp]  # New volume of water
                    vi[hgt, melt_where[nonet][evap_where], tstp] = vi[hgt-1, melt_where[nonet][evap_where], tstp]
                    va[hgt, melt_where[nonet][evap_where], tstp] = va[hgt-1, melt_where[nonet][evap_where], tstp]
                    mi[hgt, melt_where[nonet][evap_where], tstp] = ri * vi[hgt, melt_where[nonet][evap_where], tstp]
                    mw_inside[hgt, melt_where[nonet][evap_where], tstp] = mw_inside[hgt-1, melt_where[nonet][evap_where], tstp]
                    mw[hgt, melt_where[nonet][evap_where], tstp] = mw_inside[hgt, melt_where[nonet][evap_where], tstp] + mw_outside[hgt, melt_where[nonet][evap_where], tstp]
                    v[hgt, melt_where[nonet][evap_where], tstp] = vi[hgt, melt_where[nonet][evap_where], tstp] + va[hgt, melt_where[nonet][evap_where], tstp] + vw[hgt, melt_where[nonet][evap_where], tstp]
                    d[hgt, melt_where[nonet][evap_where], tstp] = ((6/pi) * v[hgt, melt_where[nonet][evap_where], tstp])**(1.0 / 3.0)   
                    di[hgt, melt_where[nonet][evap_where], tstp] = di[hgt-1, melt_where[nonet][evap_where], tstp]   
                        
                subl_where = np.where(mw_outside[hgt-1, melt_where[nonet], tstp] <= 0.)[0]
                if np.count_nonzero(subl_where) > 0:
                    estp = sat_vapor_p(tp[hgt, melt_where[nonet][subl_where], tstp], i_flag=1)
                    drho = (1 / rv) * ((e[hgt, tstp] / (t[hgt, tstp] + t0)) - (estp / (tp[hgt, melt_where[nonet][subl_where], tstp] + t0)))            
                    dmidt = 4 * pi * (0.5 * d[hgt-1, melt_where[nonet][subl_where], tstp]) * dv[hgt, tstp] * fv_tmp[melt_where[nonet][subl_where]] * drho # kg/s
                    if subl_opt == False:
                        dmidt[:] = 0.0
                    dTdt_subl[hgt, melt_where[nonet][subl_where], tstp] = (ls / (cp * rad[hgt, tstp]) * dmidt) # (K m3) / s (Cooling rate per particle)
                    dqdt[hgt, melt_where[nonet][subl_where], tstp] = -1.0 * dmidt / rad[hgt, tstp] # ((kg/kg) m3) / s (Moistening rate per particle)
                    
                    dmi = (dmidt * dh) / (u[hgt, melt_where[nonet][subl_where], tstp] - w[hgt, tstp]) # kg
                    dvi[hgt, melt_where[nonet][subl_where], tstp] = np.maximum(dmi / ri, -vi[hgt-1, melt_where[nonet][subl_where], tstp]) # m3 (Per particle)
                    vi[hgt, melt_where[nonet][subl_where], tstp] = vi[hgt-1, melt_where[nonet][subl_where], tstp] + dvi[hgt, melt_where[nonet][subl_where], tstp]  # New volume of ice
                    mi[hgt, melt_where[nonet][subl_where], tstp] = ri * vi[hgt, melt_where[nonet][subl_where], tstp]
                    # Assume constant density during sublimation (Validity? -- fine for solid ice)
                    va[hgt, melt_where[nonet][subl_where], tstp] = np.maximum(va[hgt-1, melt_where[nonet][subl_where], tstp] + (1 - (rs[melt_where[nonet][subl_where]]/ri)) * dvi[hgt, melt_where[nonet][subl_where], tstp], 0.0)
                    vw[hgt, melt_where[nonet][subl_where], tstp] = vw[hgt-1, melt_where[nonet][subl_where], tstp]
                    v[hgt, melt_where[nonet][subl_where], tstp] = (vi[hgt, melt_where[nonet][subl_where], tstp] +
                                                                   va[hgt, melt_where[nonet][subl_where], tstp] +
                                                                   vw[hgt, melt_where[nonet][subl_where], tstp])
                    d[hgt, melt_where[nonet][subl_where], tstp] = ((6/pi) * v[hgt, melt_where[nonet][subl_where], tstp])**(1.0 / 3.0)
                    di[hgt, melt_where[nonet][subl_where], tstp] = ((6 / pi) * (vi[hgt, melt_where[nonet][subl_where], tstp] + va[hgt, melt_where[nonet][subl_where], tstp] + (mw_inside[hgt, melt_where[nonet][subl_where], tstp] / rw)))**(1./3.)        

            # Meltwater Shedding
            mwmax = 2.68e-4 + 0.1389 * (mi[hgt, melt_where, tstp] + mw_inside[hgt, melt_where, tstp])
            mw_shed[hgt, melt_where, tstp] = (mw_outside[hgt, melt_where, tstp] - mwmax).clip(min=0.0)
            # Limit shedding to only partially melted particles
            mw_shed[hgt, np.where(mi[hgt, melt_where, tstp] == 0.0)[0], tstp] = 0.0
            mw_outside[hgt, melt_where, tstp] -= mw_shed[hgt, melt_where, tstp]            
            mw[hgt, melt_where, tstp] = np.maximum(mw_inside[hgt, melt_where, tstp] + mw_outside[hgt, melt_where, tstp], 0.0)
            vw[hgt, melt_where, tstp] = mw[hgt, melt_where, tstp] / rw
            v[hgt, melt_where, tstp] = vi[hgt, melt_where, tstp] + vw[hgt, melt_where, tstp] + va[hgt, melt_where, tstp]
            d[hgt, melt_where, tstp] = (6 * v[hgt, melt_where, tstp] / pi)**(1.0/3.0)
            di[hgt, melt_where, tstp] = (6 * (mi[hgt, melt_where, tstp]/ri + mw_inside[hgt, melt_where, tstp]/rw + va[hgt, melt_where, tstp])/pi)**(1.0/3.0)

        # Clean up any minute calculation errors
        d[hgt, :, tstp] = d[hgt, :, tstp].clip(min=0.0)
        mi_tmp = mi[hgt, :, tstp]
        mi[hgt, np.where(mi_tmp < 1e-16)[0], tstp] = 0.0 # Floating point mess
        mw_tmp = mw[hgt, :, tstp]
        mw[hgt, np.where(mw_tmp < 1e-16)[0], tstp] = 0.0 # Floating point mess
        mw_inside_tmp = mw_inside[hgt, :, tstp]
        mw_inside[hgt, np.where(mw_inside_tmp < 1e-16)[0], tstp] = 0.0 # Floating point mess
        mw_outside_tmp = mw_outside[hgt, :, tstp]
        mw_outside[hgt, np.where(mw_outside_tmp < 1e-16)[0], tstp] = 0.0 # Floating point mess

        # Calculate mass water fraction
        fm[hgt, :, tstp] = np.where((mi[hgt, :, tstp] + mw[hgt, :, tstp]) > 0.0,
                                    mw[hgt, :, tstp] / (mi[hgt, :, tstp] + mw[hgt, :, tstp]),
                                    0.0)
        fm[hgt, :, tstp] = fm[hgt, :, tstp].clip(min=0.0, max=1.0)

        # Calculate correspondnig aspect ratio of particle if fully melted
        de =  1e3 * (6 / pi * (mi[hgt, :, tstp] + mw[hgt, :, tstp]) / rw)**(1.0/3.0)
        arend[hgt, :, tstp] = np.where(de < 8.0,
                                       np.minimum(0.9951000 +
                                                  0.0251000 * de -
                                                  0.0364400 * de**2 +
                                                  0.0053030 * de**3 -
                                                  0.0002492 * de**4,
                                                  1.0),
                                       0.56)
        
        if ar_opt == 0:  
            # R11 parameterization (linear interpolation of AR
            # between that of dry hail and that of raindrop)
            solid_where = np.where(va[hgt-1, :, tstp] <= 0.0)[0]
            fm_ar[hgt, solid_where, tstp] = fm[hgt, solid_where, tstp]
            spongy_where = np.where(va[hgt-1, :, tstp] > 0.0)[0]
            fm_ar[hgt, spongy_where, tstp] = np.where((mi[hgt, spongy_where, tstp] + mw[hgt, spongy_where, tstp]) > 0,
                                                      mw_outside[hgt, spongy_where, tstp] / (mi[hgt, spongy_where, tstp] + mw_outside[hgt, spongy_where, tstp]),
                                                      fm_ar[hgt-1, spongy_where, tstp])
            ar_conds = [fm_ar[hgt, :, tstp] < 0.2,
                        (fm_ar[hgt, :, tstp] >= 0.2) & (fm_ar[hgt, :, tstp] < 0.8),
                        fm_ar[hgt, :, tstp] >= 0.8]
            ar_choices = [ar[0, :, tstp] - 5.0 * (ar[0, :, tstp] - 0.8) * fm_ar[hgt, :, tstp],
                          0.88 - 0.4 * fm_ar[hgt, :, tstp],
                          2.8 - 4.0 * arend[hgt, :, tstp] + 5.0 * (arend[hgt, :, tstp] - 0.56) * fm_ar[hgt, :, tstp]]
            # AR parameterization of melting graupel from Ryzhkov et al. (2011)
            ar_out[hgt, :, tstp] = np.select(ar_conds, ar_choices)
            ar_out[hgt, :, tstp] = np.where((d[hgt, :, tstp] * 1e3) > 6.0,
                                            np.max([ar_out[hgt, :, tstp], arend[hgt, :, tstp]], axis = 0),
                                            ar_out[hgt, :, tstp])
            ar[hgt, :, tstp] = ar_out[hgt, :, tstp] # Set "overall" aspect ratio to outer value
            ar_in[hgt, :, tstp] = ar[hgt, :, tstp]
        
        # Combined Kumjian et al. (2018) + Theis et al. (2022) parameterization
        if ar_opt == 1: 
            ar_initial = ar[0, :, tstp]
            
            # If particle is gone, retain AR from grid above as dummy  value
            gone_where = np.where((mi[hgt-1, :, tstp] + mw[hgt-1, :, tstp] == 0.0) | (d[hgt, :, tstp] == 0.0))[0]
            ar_out[hgt, gone_where, tstp] = 1.0
            ar_in[hgt, gone_where, tstp] = 1.0
            
            # If particle is completely frozen, allow particle to become increasingly
            # oblate during sublimation (following Theis et al. 2022) by assuming that
            # the decrease in size along the minor axis is 2x that along the major axis
            # (which is nothing more than an approximate guess)
            frozen_where = np.where((fm[hgt, :, tstp] == 0.0) & (di[hgt, :, tstp] > 0))[0]
            
            # Previous assumption: aspect ratio doesn't change during sublimation
            #ar_out[hgt, frozen_where, tstp] = ar_initial[frozen_where]
            #ar_in[hgt, frozen_where, tstp] = ar_initial[frozen_where]
            b_core = (di[hgt-1, frozen_where, tstp]**3. / ar_in[hgt-1, frozen_where, tstp])**(1./3.)
            a_core = (di[hgt-1, frozen_where, tstp]**3. / b_core**2.)
            
            #for mm in range(len(frozen_where)):
            #    print('Di:', di[hgt, mm, tstp])
            #    coeff = [0.25, 
            #             0.25*a_core[mm] + b_core[mm], 
            #             a_core[mm] * b_core[mm] + b_core[mm]**2, 
            #             a_core[mm] * b_core[mm]**2 - di[hgt, frozen_where[mm], tstp]**3]
            #    roots = np.roots(coeff)
            #    roots = roots.real[abs(roots.imag) < 1e-20]
            #    print(roots)
            #    if len(roots) == 1:
            #        delta_a = roots[0]
            #        print('Singular root found!')
            #    else:
            #        delta_a = roots[np.argmin(np.abs(roots))]
            #        print('Warning: There are multiple real roots occurring!')
            #        print('Roots: ', roots)
            #    delta_b = 0.5 * delta_a
            #    a_core[mm] = a_core[mm] + delta_a
            #    b_core[mm] = b_core[mm] + delta_b
            #    ar_in[hgt, frozen_where[mm], tstp] = ar_out[hgt, frozen_where[mm], tstp] = np.maximum(a_core[mm] / b_core[mm], 0.5)
            #    print('New ar_in: ', ar_in[hgt, frozen_where[mm], tstp])
            
            
            
            delta_a = (4.0 * (di[hgt-1, frozen_where, tstp] - di[hgt, frozen_where, tstp])**3)**(1./3.)
            a_core -= delta_a
            b_core -= 0.5 * delta_a
            ar_in[hgt, frozen_where, tstp] = a_core / b_core
            ar_out[hgt, frozen_where, tstp] = a_core / b_core
            
            # Allow particles to also become more oblate during first half of meltwater soaking
            filling_where = np.where((mw_inside[hgt, :, tstp] > 0) & (va[hgt, :, tstp] >= (0.5 * va[0, :, tstp])))[0]
            b_core = (di[hgt-1, filling_where, tstp]**3. / ar_in[hgt-1, filling_where, tstp])**(1./3.)
            a_core = (di[hgt-1, filling_where, tstp]**3. / b_core**2.)
            
            #for mm in range(len(filling_where)):
            #    print('Di:', di[hgt, mm, tstp])
            #    coeff = [0.25, 
            #             0.25*a_core[mm] + b_core[mm], 
            #             a_core[mm] * b_core[mm] + b_core[mm]**2, 
            #             a_core[mm] * b_core[mm]**2 - di[hgt, filling_where[mm], tstp]**3]
            #    roots = np.roots(coeff)
            #    roots = roots.real[abs(roots.imag) < 1e-20]
            #    print(roots)
            #    if len(roots) == 1:
            #        delta_a = roots[0]
            #        print('Singular root found!')
            #    else:
            #        delta_a = roots[np.argmin(np.abs(roots))]
            #        print('Warning: There are multiple real roots occurring!')
            #        print('Roots: ', roots)
            #    delta_b = 0.5 * delta_a
            #    a_core[mm] = a_core[mm] + delta_a
            #    b_core[mm] = b_core[mm] + delta_b
            #    ar_in[hgt, filling_where[mm], tstp] = ar_out[hgt, filling_where[mm], tstp] = np.maximum(a_core[mm] / b_core[mm], 0.5)
            #    print('New ar_in: ', ar_in[hgt, filling_where[mm], tstp])
            
            delta_a = (4.0 * (di[hgt-1, filling_where, tstp] - di[hgt, filling_where, tstp])**3)**(1./3.)
            a_core -= delta_a
            b_core -= 0.5 * delta_a # !!!
            ar_in[hgt, filling_where, tstp] = a_core / b_core
            ar_out[hgt, filling_where, tstp] = a_core / b_core
            
            # Once half the air volume has been soaked, keep aspect ratios constant during the rest of soaking
            spongy_where = np.where((va[hgt, :, tstp] < (0.5 * va[0, :, tstp])) & (va[hgt, :, tstp] > 0.0) & (mw_inside[hgt, :, tstp] > 0.0))[0]
            ar_in[hgt, spongy_where, tstp] = ar_in[hgt-1, spongy_where, tstp]
            ar_out[hgt, spongy_where, tstp] = ar_out[hgt-1, spongy_where, tstp]

            # One meltwater volume exceeds ice volume for a soaked particle,
            # begin linearly transitioning to AR of raindrop.
            deform_where = np.where((vw[hgt, :, tstp] > vi[hgt, :, tstp]) & (va[hgt, :, tstp] == 0.0))[0]
            ar_in[hgt, deform_where, tstp] = ar_in[hgt-1, deform_where, tstp]
            ar_out[hgt, deform_where, tstp] = (ar_out[hgt-1, deform_where, tstp] + 
                                               (arend[hgt, deform_where, tstp] - ar_out[hgt-1, deform_where, tstp]) * 
                                               ((vw[hgt, deform_where, tstp] - vi[hgt, deform_where, tstp]) / v[hgt, deform_where, tstp]))

         
            # Otherwise compute meltwater torus aspect ratio based on Kumjian (2018)
            torus_where = np.where((vw[hgt, :, tstp] <= vi[hgt, :, tstp]) & (va[hgt, :, tstp] == 0.0) & (mw_outside[hgt, :, tstp] > 0.0))[0]
            
            b_core = (di[hgt, torus_where, tstp]**3. / ar_in[hgt-1, torus_where, tstp])**(1./3.)
            a_core = (di[hgt, torus_where, tstp]**3. / b_core**2.)
            
            b_total = (d[hgt, torus_where, tstp]**3. / ar_in[hgt-1, torus_where, tstp])**(1./3.)
            a_total = (d[hgt, torus_where, tstp]**3. / b_total**2.)
            
            #min_layer = np.minimum(0.5e-3, 0.05 * di[hgt, torus_where, tstp])
            min_layer = np.zeros_like(di[hgt, torus_where, tstp]) + 0.5e-3
            thinshell_where = np.where((a_total - a_core) < min_layer)[0]
            ar_in[hgt, torus_where[thinshell_where], tstp] = ar_in[hgt-1, torus_where[thinshell_where], tstp]   
            ar_out[hgt, torus_where[thinshell_where], tstp] = ar_in[hgt-1, torus_where[thinshell_where], tstp]
            thickshell_where = np.where((a_total - a_core) >= min_layer)[0]
            if len(thickshell_where) > 0:
                ar_in[hgt, torus_where[thickshell_where], tstp] = ar_in[hgt-1, torus_where[thickshell_where], tstp] # Fix inner aspect ratio during melting (accuracy??)
                da = min_layer[thickshell_where]
                a_total = a_core[thickshell_where] + da
                db = np.sqrt((((6 * mw_outside[hgt, torus_where[thickshell_where], tstp]) / (pi * rw)) + di[hgt, torus_where[thickshell_where], tstp]**3.) * (1. / (a_total))) - (a_core[thickshell_where] / ar_in[hgt, torus_where[thickshell_where], tstp])
                b_total = b_core[thickshell_where] + db
                ar_out[hgt, torus_where[thickshell_where], tstp] = np.maximum(a_total / b_total, arend[hgt, torus_where[thickshell_where], tstp]) # Sets min to 0.56 
                
            # If particle is mostly melted, set OUTER aspect ratio to that of rain
            # and retain inner aspect ratio
            melted_where = np.where((fm[hgt, :, tstp] > 0.90))[0]
            ar_out[hgt, melted_where, tstp] = arend[hgt, melted_where, tstp]
            ar_in[hgt, melted_where, tstp] = ar_in[hgt-1, melted_where, tstp]

            # Finally, apply some sanity checks
            ar_out[hgt, :, tstp] = ar_out[hgt, :, tstp].clip(max=1.0, min=0.40) # Min of 0.40 or 0.56?    
            ar_in[hgt, :, tstp] = ar_in[hgt, :, tstp].clip(max=1.0, min=0.40)
            ar[hgt, :, tstp] = ar_out[hgt, :, tstp] # Set "overall" aspect ratio to outer value
            
        dmax[hgt, :, tstp] =  d[hgt, :, tstp] * (ar_out[hgt, :, tstp])**(-1./3.)

    ###############################################################################
    # Calculate fractional volumes of constituents for canting angle calculations
    ###############################################################################
    fvol_wat = np.divide(vw[:, :, tstp], v[:, :, tstp], out=np.zeros_like(v[:, :, tstp]), where=v[:, :, tstp]!=0)
    fvol_wat[fvol_wat < 0.0] = 0.0
    fvol_wat[np.isnan(fvol_wat)] = 1.0  # QC

    fvol_ice = np.divide(vi[:, :, tstp], v[:, :, tstp], out=np.zeros_like(v[:, :, tstp]), where=v[:, :, tstp]!=0)
    fvol_ice[fvol_ice < 0.0] = 0.0
    fvol_ice[np.isnan(fvol_ice)] = 0.0  # QC

    fvol_air = np.divide(va[:, :, tstp], v[:, :, tstp], out=np.zeros_like(v[:, :, tstp]), where=v[:, :, tstp]!=0)
    fvol_air[fvol_air < 0.0] = 0.0
    fvol_air[np.isnan(fvol_air)] = 0.0  # QC

    ###############################################################################
    # Shedding routine
    ###############################################################################

    if shed_opt == True:
        # Values originally included in code (not sure from who... Matt?):
        #lam_shed = 5  # Slope parameter for shed DSD [1/mm]
        #mu_shed = 3  # Shape parameter for shed DSD
        # Note: This puts drop mode at < 1 mm per observations)
        
        lam_shed = 2.0  # [1/mm]
        mu_shed = 2.0
        # Above values taken from Ryzhkov et al. (2013a)
        # How uncertain are these parameters?

        # !!! Experiment: Just redistribute drops into fully-melted original bins.
        # That way advection, evaporation, etc. are just taken into account already
        # and don't require separate array/treatment and teh only thing modified
        # is the number of drops in each bin. Previous version had a completely
        # separate distribution of shed drops, which would've required a second
        # treatment of advection, evaporation, etc. and really wasn't compatible
        # with the new model version.
        
        for hgt in range(0, int(nlev)):
            # Total mass of excess/shed drops
            mstot[hgt, tstp] = np.nansum(mw_shed[hgt, :, tstp] * dsdm[hgt, :, tstp]) * 1000.0 # g
            if mstot[hgt, tstp] > 0.0:
                shed_where = np.where((mi[hgt, :, tstp] == 0.0) & (d[hgt, :, tstp] < 6e-3))[0]
                # shed_where are the bins to redistribute shed drops INTO
                if len(shed_where) > 1:
                    # This is a hackish way to "bring back" bins that were totally empty to hold shed drops
                    new_drops = np.where(d[hgt, shed_where, tstp] == 0.0)[0]
                    if len(new_drops) > 0:
                        d[hgt, new_drops, tstp] = np.linspace(0.00001, d[hgt, new_drops[-1]+1, tstp], len(new_drops))
                    d_shed = 1e3 * d[hgt, shed_where, tstp] # mm
                    
                    ##########################################################
                    ######################### OPTION 1 #######################
                    ##########################################################
                    if shed_dsd_opt == 0:                        
                        dd_shed = np.ediff1d(d_shed)
                        dd_shed = np.pad(dd_shed, (0, 1), 'constant', constant_values=dd_shed[-1])
                   
                        # Work in log space
                        d_shed_log = np.log10(d_shed)
                        dd_shed_log = np.ediff1d(d_shed_log)
                        dd_shed_log = np.pad(dd_shed_log, (0, 1), 'constant', constant_values=dd_shed_log[-1])
                                   
                        # Parameters from Theis et al. (2021)
                        mu2 = np.log10(0.89)
                        sigma2 = np.log10(2.21)
                        conc2 = 1.0
                        mu1 = np.log10(0.1)
                        sigma1 = np.log10(1.61)
                        conc1 = 0.86 * conc2
                        
                        dist1 = conc1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp( - 0.5 * ((d_shed_log - mu1) / sigma1)**2) 
                        dist2 = conc2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp( - 0.5 * ((d_shed_log - mu2) / sigma2)**2) 
                        dist = (dist1 + dist2) / np.max(dist1+dist2) # Normalize to make peak of distribution = 1.0 for normalization
    
                        # Solve for intercept parameter of shed drop distribution by 
                        # integrating the distribution and solving for which N0 equals Mstot
                        nsh[hgt, tstp] = 1e-3 * mstot[hgt, tstp] / ((pi/6) * rw * np.nansum(dist * (1e-3*d_shed)**3))
                        
                        dsd_shed[hgt, shed_where, tstp] = np.maximum(nsh[hgt, tstp] * dist, 0.0)
                    
                    ##########################################################
                    ######################### OPTION 2 #######################
                    ##########################################################
                    if shed_dsd_opt == 1:
                        dd_shed = np.ediff1d(d_shed)
                        # !!! Having some issues with non-monotonicity of d (?!?!?)
                        # This should be resolved but this is a workaround for now
                        mono_where = np.where(dd_shed > 0)[0]
                        shed_where = shed_where[mono_where]
                        d_shed = d_shed[mono_where]
                        dd_shed = dd_shed[mono_where]
                        
                        nsh[hgt, tstp] = ((1e6 * 6.0 * mstot[hgt, tstp] / (pi * rw)) / 
                                    simpson((d_shed**(mu_shed + 3.0) *
                                            np.exp(-lam_shed * d_shed)), x=d_shed, dx=dd_shed, even='avg'))
                        dsd_shed[hgt, shed_where, tstp] = nsh[hgt, tstp] * (d_shed**mu_shed) * np.exp(-lam_shed * d_shed) * dd_shed
                    
                    ##########################################################
                    
                    # Add drops into existing distribution
                    dsdm[hgt, shed_where, tstp] += dsd_shed[hgt, shed_where, tstp]
                    vw[hgt, shed_where, tstp] = (pi/6) * d[hgt, shed_where, tstp]**3
                    mw[hgt, shed_where, tstp] = rw * vw[hgt, shed_where, tstp]
                    mw_outside[hgt, shed_where, tstp] = mw[hgt, shed_where, tstp]
                    mw_inside[hgt, shed_where, tstp] = 0.0
                    mi[hgt, shed_where, tstp] = 0.0
                    vi[hgt, shed_where, tstp] = 0.0
                    va[hgt, shed_where, tstp] = 0.0
                    fm[hgt, shed_where, tstp] = 1.0
                    ar_in[hgt, shed_where, tstp] = mp.aspect_ratio_rain(d[hgt, shed_where, tstp]*1e3)
                    ar_out[hgt, shed_where, tstp] = ar_in[hgt, shed_where, tstp]
                    ar[hgt, shed_where, tstp] = ar_out[hgt, shed_where, tstp]
                    dmax[hgt, shed_where, tstp] =  d[hgt, shed_where, tstp] * (ar_out[hgt, shed_where, tstp])**(-1./3.)
                    tp[hgt, shed_where, tstp] = np.nanmax(tp[hgt, shed_where, tstp])
                    u[hgt, shed_where, tstp] = (ras/ra[hgt, tstp])**(0.4) * mp.term_vel_rain(mw[hgt, shed_where, tstp])
                    
                elif len(shed_where) == 1: # All shed drops go to single size bin
                    print('Warning: Only one size bin available for shed drops.')
                    dsd_shed[hgt, shed_where, tstp] = (1e-3 * mstot[hgt, tstp] / mw[hgt, shed_where, tstp])
                    dsdm[hgt, shed_where, tstp] += dsd_shed[hgt, shed_where, tstp]
                    
                    
    ###############################################################################
    # Drop breakup (spontaneous) routine
    ###############################################################################
    if break_opt == True:

        bd_die = 400  # Fall distance (m) for 50% breakup [m] - from original RH13 code
        lam_bu = 0.453  # Slope for breakup drops [1/mm] (Kamra et al. 1991)

        for size in range(0, nbin):
            breakup_where = np.where((fm[:, size, tstp] == 1) & ((1e3 * d[:, size, tstp]) > 7.95))[0]
            # Note this is the height-vector probability of breakup of bin 'size'
            if len(breakup_where) > 0:
                m0 = np.min(breakup_where)
                mm = np.arange(nlev-m0)
                breakup_prob[m0:, size, tstp] = (1 - np.exp(-(mm * dh / (1.2 * bd_die))**2))
        
        for hgt in range(1, nlev):
            # Follow analogous procedure as shed drops weighted by the probability of breakup
            mbtot[hgt, tstp] = 1e3 * np.nansum(mw[hgt, :, tstp] * dsdm[hgt, :, tstp] * breakup_prob[hgt, :, tstp]) # g/m3 -- Total breakup (including from above)
            if mbtot[hgt, tstp] > 0:
                dsdm[hgt, :, tstp] = dsdm[hgt, :, tstp] * (1 - breakup_prob[hgt, :, tstp]) # Reduce droplets that have actually broken up
                breakup_where = np.where((fm[hgt, :, tstp] == 1.0) & ((1e3 * d[hgt, :, tstp]) < 4.0) & ((1e3 * d[hgt, :, tstp]) > 0.0))[0]
                # Note these are the bins to put the broken up drops IN
                d_breakup = 1e3 * d[hgt, breakup_where, tstp] # mm
                dd_breakup = np.ediff1d(d_breakup)
                dd_breakup = np.pad(dd_breakup, (0, 1), 'constant', constant_values=dd_breakup[-1])
                nbd[hgt, tstp] = ((1e6 * 6.0 * mbtot[hgt, tstp] / (pi * rw)) / 
                                    simpson((d_breakup**3. * np.exp(-lam_bu * d_breakup)), x=d_breakup, dx=dd_breakup, even='avg'))            
                dsd_breakup[hgt, breakup_where, tstp] = nbd[hgt, tstp] * np.exp(-lam_bu * d_breakup) * dd_breakup
                
                dsdm[hgt, breakup_where, tstp] += dsd_breakup[hgt, breakup_where, tstp]
                
    ###########################################################################
    # Calculate total mass contents
    ###########################################################################

    # Total IWC and LWC [g/m3]
    mitot[:, tstp] = np.nansum(mi[:, :, tstp] * dsdm[:, :, tstp], axis=1) * 1000.0  # To convert to g
    mwtot[:, tstp] = np.nansum(mw[:, :, tstp] * dsdm[:, :, tstp], axis=1) * 1000.0

    # Total precipitation mixing ratio [kg/kg]
    qp[:, tstp] = 1e-3 * (mitot[:, tstp] + mwtot[:, tstp]) / rad[:, tstp] # kg/kg
    # Note: mstot not included here because the water is already redistributed into mwtot

    # Note this is the total precipitation rate, rather than the RAINRATE. (!!! Not tested)
    preciprate[:, tstp] = 3.6 * 1e-3 * (pi/6) * np.nansum(dsdm[:, :, tstp] * u[:, :, tstp] * (1e3*d[:, :, tstp])**3, axis=1)
    preciprate_w[:, tstp] = np.nansum((v[:, :, tstp] * (u[:, :, tstp] - w[:, None, tstp]) * dsdm[:, :, tstp]), axis=1) * 3600 * 1000 # mm/hr
    for hgt in range(0, int(nlev)):
        melted_where = np.where(fm[hgt, :, tstp] == 1.0)[0]
        # Only included *fully-melted bins* in the rainrate calculation
        rainrate[hgt, tstp] = np.nansum((v[hgt, melted_where, tstp] * u[hgt, melted_where, tstp] * dsdm[hgt, melted_where, tstp])) * 3600 * 1000 # mm/hr
        rainrate_w[hgt, tstp] = np.nansum((v[hgt, melted_where, tstp] * (u[hgt, melted_where, tstp] - w[hgt, None, tstp]) * dsdm[hgt, melted_where, tstp])) * 3600 * 1000 # mm/hr

    ###############################################################################
    # Latent heating and cooling rates:
    ###############################################################################
    
    # Integrate heating/cooling rate along entire PSD
    for hgt in range(0, int(nlev)):
        dTdt_melt_tot[hgt, tstp] = np.nansum(dTdt_melt[hgt, :, tstp] * dsdm[hgt, :, tstp])  # [K/s]
        if subl_opt:
            dTdt_subl_tot[hgt, tstp] = np.nansum(dTdt_subl[hgt, :, tstp] * dsdm[hgt, :, tstp])  # [K/s]
        if evap_opt:
            dTdt_evap_tot[hgt, tstp] = np.nansum(dTdt_evap[hgt, :, tstp] * dsdm[hgt, :, tstp])
        dTdt_tot[hgt, tstp] = dTdt_melt_tot[hgt, tstp] + dTdt_subl_tot[hgt, tstp] + dTdt_evap_tot[hgt, tstp]  # K s-1
        dqdt_tot[hgt, tstp] = np.nansum(dqdt[hgt, :, tstp] * dsdm[hgt, :, tstp])  # (kg kg-1) s-1

    ###############################################################################
    # Update downdraft environment for tstp + 1
    ###############################################################################
    
    # Fixing parameters at the top of the domain to be constant:
    if init_frozen_opt:
        dsdm[0, :, tstp+1] = (np.where(d[0, :, tstp] <= 1e-3*dmax_limit,
                                       deld * (nh0 * np.exp(-lamh * d[0, :, tstp] * 1000.0) +
                                               ng0 * np.exp(-lamg * d[0, :, tstp] * 1000.0)),
                                       0.0))
    else:
        if dsd_norm:
            f_mu = (6 * (4 + mur)**(4 + mur)) / (4**4 * sp.special.gamma(4 + mur))
            dsdm[0, :, tstp+1] = (np.where(d[0, :, tstp] <= 1e-3*dmax_limit,
                                         deld * (nrw * f_mu * ((d[0, :, tstp] * 1000.) / dmr)**mur * np.exp(-(4 + mur) * (d[0, :, tstp] * 1000.)/dmr)),
                                         0.0))    
        else:
            dsdm[0, :, tstp+1] = (np.where(d[0, :, tstp] <= 1e-3*dmax_limit,
                                         deld * (nr0 * (d[0, :, tstp] * 1000.0)**(mur) * np.exp(-lamr * d[0, :, tstp] * 1000.0)),
                                         0.0)) 
            
    h[0, tstp+1] = h[0, 0]
    qstar[0, tstp+1] = q[0, 0]
    u_weighted[0, tstp+1] = (np.average(u[0, :, tstp],
                                        weights  = ((mi[0, :, tstp] + mw[0, :, tstp]) * dsdm[0, :, tstp+1])))
    w[0, tstp+1] = -1.0
    p[0, tstp+1] = p[0, tstp]
    t[0, tstp+1] = t[0, tstp]
    rh[0, tstp+1] = rh[0, tstp]
    q[0, tstp+1] = q[0, tstp]
    
    # Note: this can probably all be updated to be filled in at the very beginning
    tk[0, tstp+1] = t[0, tstp+1] + t0
    tv[0, tstp+1] = virtual_temperature(tk[0, tstp+1], q[0, tstp+1])
    es[0, tstp+1] = sat_vapor_p(t[0, tstp+1])
    esi[0, tstp+1] = sat_vapor_p(t[0, tstp+1], i_flag=1)
    qs[0, tstp+1] = vapor_mixing_ratio(es[0, tstp+1], p[0, tstp+1])
    e[0, tstp+1] = (rh[0, tstp+1] / 100.) * es[0, tstp+1]
    ra[0, tstp+1] = density_air(p[0, tstp+1], tv[0, tstp+1])
    rad[0, tstp+1] = density_air(p[0, tstp+1], tk[0, tstp+1])
    eta[0, tstp+1] = dynamic_viscosity_air(tk[0, tstp+1])
    nu[0, tstp+1] = eta[0, tstp+1] / ra[0, tstp+1]
    ka[0, tstp+1] = thermal_diffusivity_air(tk[0, tstp+1])
    kair[0, tstp+1] = thermal_conductivity_air(tk[0, tstp+1])
    kwa[0, tstp+1] = thermal_conductivity_water(tk[0, tstp+1])
    pr[0, tstp+1] = nu[0, tstp+1] / ka[0, tstp+1]
    sc[0, tstp+1] = nu[0, tstp+1] / dv[0, tstp+1]

    # Updating particle concentrations (Eq. 18 of S87)
    for ii in range(nbin):
        dN_term1[1:, ii, tstp] = dsdm[1:, ii, tstp] * ((u[1:, ii, tstp] / ra[1:, tstp]) * (-np.ediff1d(ra[:, tstp]) / dh) + (-np.ediff1d(u[:, ii, tstp]) / dh))
        #dN_term1[1:, ii, tstp] = dsdm[1:, ii, tstp] * (((w[1:, tstp] - u[1:, ii, tstp]) / ra[1:, tstp]) * (-np.ediff1d(ra[:, tstp]) / dh) + (-np.ediff1d((w[:, tstp] - u[:, ii, tstp])) / dh))
        idx = np.where(np.abs(dN_term1[:, ii, tstp]) < 1e-10) # QC
        dN_term1[idx, ii, tstp] = 0.0

        dN_term2[1:, ii, tstp] = -(w[1:, tstp] - u[1:, ii, tstp]) * (-np.ediff1d(dsdm[:, ii, tstp]) / dh)
        idx = np.where(np.abs(dN_term2[:, ii, tstp]) < 1e-10) # QC
        dN_term2[idx, ii, tstp] = 0.0
        
        # This and other mixing terms were in S85 but not S87. However, they are 
        # very small in comparison with the other terms.
        dN_term3[1:, ii, tstp] = -mix_coef * np.abs(w[1:, tstp]) * dsdm[1:, ii, tstp]
        idx = np.where(np.abs(dN_term3[:, ii, tstp]) < 1e-10) #QC
        dN_term3[idx, ii, tstp] = 0.0

        dsdm[1:, ii, tstp+1] = np.maximum(dsdm[1:, ii, tstp] + delt * (dN_term1[1:, ii, tstp] + dN_term2[1:, ii, tstp] + dN_term3[1:, ii, tstp]), 0.0)
        #dsdm[1:, ii, tstp+1] = dsdm[0, ii, tstp] * u[0, ii, tstp] / u[1:, ii, tstp]

    # Update equivalent temperature (Eq. 19 of S87)
    # Adiabatic warming
    dh_term1[1:, tstp] = -w[1:, tstp] * (0.0098)
    idx = np.where(np.abs(dh_term1[:, tstp]) < 1e-10)
    dh_term1[idx, tstp] = 0.0

    # Vertical advection
    dh_term2[1:, tstp] = -w[1:, tstp] * (-np.ediff1d(h[:, tstp]) / dh)
    idx = np.where(np.abs(dh_term2[:, tstp]) < 1e-10)
    dh_term2[idx, tstp] = 0.0

    # Entrainment/mixing term
    dh_term3[1:, tstp] = -mix_coef * np.abs(w[1:, tstp]) * ((t[1:, tstp] - tenv[1:]) + (lv/cp) * (q[1:, tstp] - qenv[1:]))
    idx = np.where(np.abs(dh_term3[:, tstp]) < 1e-10)
    dh_term3[idx, tstp] = 0.0
    
    # Microphysical term (cooling rate and change in dqdt, which should in theory offset each other?)
    # Note: Cooling rate is no longer reflected in previous adjustment of temperature 
    # that adjusts existing H field
    dh_term4[1:, tstp] = dTdt_tot[1:, tstp] + (lv * dqdt_tot[1:, tstp] / cp)
    idx = np.where(np.abs(dh_term4[:, tstp]) < 1e-10)
    dh_term4[idx, tstp] = 0.0
    
    h[1:, tstp+1] = h[1:, tstp] + delt * (dh_term1[1:, tstp] + dh_term2[1:, tstp] + dh_term3[1:, tstp] + dh_term4[1:, tstp])

    # Updating q* (qv + qc) (Eq. 21 of S87)
    # Note: This equation has been modified to be only for the q* portion of Eq. 21
    # with the dqp/dt calculated from the updated mitot and mwtot distributions
    # during the microphysical processes.
    # !!! I am still not sure this is completely correct but believe it is
    # dq/dt is now included as a term as calculated from the microphysics (and
    # as the inverse of dqp/dt) and the convergence term should (?) be taken into
    # account via N and the microphysical changes to the masses...

    # Calculate mass-weighted terminal velocity
    for hgt in range(nlev):
        #if (len(np.nonzero(dsdm[hgt, :, tstp])[0]) > 0) and (np.nansum(mi[hgt, :, tstp] + mw[hgt, :, tstp]) > 0):
        if np.nansum(dsdm[hgt, :, tstp] * (mi[hgt, :, tstp] + mw[hgt, :, tstp])) > 0.0:
            u_weighted[hgt, tstp] = np.average(u[hgt, :, tstp], weights  = ((mi[hgt, :, tstp] + mw[hgt, :, tstp]) * dsdm[hgt, :, tstp]))
        else:
            u_weighted[hgt, tstp] = 0.0
            
    # Precipitation advection
    #dqstar_term1[1:, tstp] = -(w[1:, tstp] - u_weighted[1:, tstp]) * (-np.ediff1d(qp[:, tstp]) / dh)
    #idx = np.where(np.abs(dqstar_term1[:, tstp]) < 1e-10)
    #dqstar_term1[idx, tstp] = 0.0
    
    # Precipitation flux convergence
    #dqstar_term2[1:, tstp] = qp[1:, tstp] * (u_weighted[1:, tstp] / ra[1:, tstp] * (-np.ediff1d(ra[:, tstp]) / dh) + (-np.ediff1d(u_weighted[:, tstp]) / dh))
    # idx = np.where(np.abs(dqstar_term2[:, tstp]) < 1e-10)
    # dqstar_term2[idx, tstp] = 0.0 
    
    # Moisture advection
    dqstar_term3[1:, tstp] = -w[1:, tstp] * (-np.ediff1d(qstar[:, tstp]) / dh)
    idx = np.where(np.abs(dqstar_term3[:, tstp]) < 1e-10)
    dqstar_term3[idx, tstp] = 0.0    

    # Entrainment/mixing
    dqstar_term4[1:, tstp] = -mix_coef * np.abs(w[1:, tstp]) * (q[1:, tstp] + qc[1:, tstp] + qp[1:, tstp] - qenv[1:])
    idx = np.where(np.abs(dqstar_term4[:, tstp]) < 1e-10)
    dqstar_term4[idx, tstp] = 0.0    
    
    # Local tendency due to microphysics
    dqstar_term5[1:, tstp] = dqdt_tot[1:, tstp]
    idx = np.where(np.abs(dqstar_term5[:, tstp]) < 1e-10)
    dqstar_term5[idx, tstp] = 0.0  
    
    qstar[1:, tstp+1] = qstar[1:, tstp] + delt * (dqstar_term1[1:, tstp] + dqstar_term2[1:, tstp] + dqstar_term3[1:, tstp] + dqstar_term4[1:, tstp] + dqstar_term5[1:, tstp])
                      
    # Saturation adjustment:
    # This accounts for areas where RH becomes >100% and produces cloud water
    # while iteratively balancing the temperature and moisture. 
    # Follows procedure of S85/S87.
    # This can probably be optimized further using an scipy.optimize routine
    for hgt in range(1, nlev):
        t_guess = h[hgt, tstp+1] - ((lv * qstar[hgt, tstp+1]) / cp) - t0  # degC; Assumes qc = 0
        es_guess = 611.21 * np.exp((18.678 - (t_guess / 234.5)) * (t_guess / (257.14 + t_guess)))
        qs_guess = 0.622 * es_guess / (p[hgt, tstp] - es_guess)
        if qs_guess >= qstar[hgt, tstp+1]:
            t[hgt, tstp+1] = t_guess
            q[hgt, tstp+1] = qstar[hgt, tstp+1]
            qc[hgt, tstp+1] = 0.0
        else:
            x = qs_guess
            t_guess = h[hgt, tstp+1] - ((lv * x) / cp) - t0
            es_guess = 611.21 * np.exp((18.678 - (t_guess / 234.5)) * (t_guess / (257.14 + t_guess)))
            q_guess = 0.622 * es_guess / (p[hgt, tstp] - es_guess)
            while (np.abs(q_guess - x) > 1e-6):
                if (q_guess - x) > 0:
                    x = q_guess - 0.5 * (q_guess - x)
                else:
                    x = q_guess + 0.5 * (x - q_guess)
                t_guess = h[hgt, tstp+1] - ((lv * x) / cp) - t0
                es_guess =  611.21 * np.exp((18.678 - (t_guess / 234.5)) * (t_guess / (257.14 + t_guess)))
                q_guess = 0.622 * es_guess / (p[hgt, tstp] - es_guess) 
            t[hgt, tstp+1] = t_guess
            q[hgt, tstp+1] = q_guess
            qc[hgt, tstp+1] = qstar[hgt, tstp+1] - q[hgt, tstp+1]



    # Update other downdraft thermodynamic variables based on updated fields.
    # Note: We are currently ignoring the dynamic pressure perturbation
    # within the downdraft, as done in Srivastava (1985, 1987) and
    # Feingold (1991). See Feingold (1991) for a discussion on the 
    # presumed impacts of this assumption. 
    tk[1:, tstp+1] = t[1:, tstp+1] + t0
    p[1:, tstp+1] = p[1:, tstp]
    tv[1:, tstp+1] = virtual_temperature(tk[1:, tstp+1], q[1:, tstp+1])
    es[1:, tstp+1] = sat_vapor_p(t[1:, tstp+1])
    esi[1:, tstp+1] = sat_vapor_p(t[1:, tstp+1], i_flag=1)
    qs[1:, tstp+1] = vapor_mixing_ratio(es[1:, tstp+1], p[1:, tstp+1])
    rh[1:, tstp+1] = 100.0 * (q[1:, tstp+1] / qs[1:, tstp+1])
    e[1:, tstp+1] = (rh[1:, tstp+1] / 100.) * es[1:, tstp+1]
    ra[1:, tstp+1] = density_air(p[1:, tstp+1], tv[1:, tstp+1])
    rad[1:, tstp+1] = density_air(p[1:, tstp+1], tk[1:, tstp+1])
    eta[1:, tstp+1] = dynamic_viscosity_air(tk[1:, tstp+1])
    nu[1:, tstp+1] = eta[1:, tstp+1] / ra[1:, tstp+1]
    ka[1:, tstp+1] = thermal_diffusivity_air(tk[1:, tstp+1])
    kair[1:, tstp+1] = thermal_conductivity_air(tk[1:, tstp+1])
    kwa[1:, tstp+1] = thermal_conductivity_water(tk[1:, tstp+1])
    pr[1:, tstp+1] = nu[1:, tstp+1] / ka[1:, tstp+1]
    dv[1:, tstp+1] = thermal_diffusivity_water(tk[1:, tstp+1], p[1:, tstp+1])
    sc[1:, tstp+1] = nu[1:, tstp+1] / dv[1:, tstp+1]
        
    # Update vertical velocity field (Eq. 24 of S87)
    # Vertical advection
    dw_term1[1:, tstp] = -w[1:, tstp] * (-np.ediff1d(w[:, tstp]) / dh)
    idx = np.where(np.abs(dw_term1[:, tstp]) < 1e-10)
    dw_term1[idx, tstp] = 0.0
    
    # Acceleration due to thermal buoyancy
    dw_term2[1:, tstp] = g * (((tv[1:, tstp] - tvenv[1:]) / tvenv[1:]))
    idx = np.where(np.abs(dw_term2[:, tstp]) < 1e-10)
    dw_term2[idx, tstp] = 0.0

    # Acceleration due to precipitation loading
    dw_term3[1:, tstp] = -g * (qp[1:, tstp] + qc[1:, tstp])
    idx = np.where(np.abs(dw_term3[:, tstp]) < 1e-10)
    dw_term3[idx, tstp] = 0.0

    # Entrainment/mixing
    dw_term4[1:, tstp] = -mix_coef * np.abs(w[1:, tstp]) * w[1:, tstp]
    idx = np.where(np.abs(dw_term4[:, tstp]) < 1e-10)
    dw_term4[idx, tstp] = 0.0

    w[1:, tstp+1] = w[1:, tstp] + delt * (dw_term1[1:, tstp] + dw_term2[1:, tstp] + dw_term3[1:, tstp] + dw_term4[1:, tstp])

    # !!!! Test to fix w = 0 at surface and decrease the w starting at 500 m AGL
    # min_w_height = 500.
    # min_w_idxs = np.where(heights <= min_w_height)[0]
    # w[min_w_idxs, tstp+1] = (np.maximum(w[min_w_idxs[0], tstp+1] * sp.special.erf(heights[min_w_idxs] / (0.5 * min_w_height)),
    #                                     w[min_w_idxs, tstp+1]))
    
    # Calculate surface wind, temperature, and dewpoint for simulated meteogram
    sfc_wind[tstp] = np.round(-1.5 * w[-1, tstp] * 2.23694, 3) # MPH
    sfc_t[tstp] = t[-1, tstp] # degC
    sfc_td[tstp] = mpcalc.dewpoint_from_relative_humidity(t[-1, tstp] * units.degC, 1e-2*rh[-1, tstp]).magnitude # degC
    # Converted from m/s to mph
    # Approximated as U/Wmin ~= 1.5 from Anabor (2011) and references therein
    # Other studies (e.g., Hawbecker et al. (2018) have found ratio closer to 1.3) 
    # so this parameter is fairly uncertain

    if np.max(w[:, tstp+1]) > 0.01:
        print(''''Uh oh! There's an updraft!''')
    if verbose:
        if w[-1, tstp+1] < -5: 
            # Start printing parameterized horizontal wind speed once downdraft reaches surface
            print('Horizontal wind speed: ', sfc_wind[tstp], 'mph!')

###############################################################################
# Perform radar scattering calculations
###############################################################################
for tstp in range(ntstp-1): #!!!
    print('Calculating radar variables for tstp ', tstp)
    if radar_opt == True:
        time_start = datetime.now()
        
        #######################################################################
        ###################### CANTING ANGLE DISTRIBUTION #####################
        #######################################################################

        # Canting angle distribution width linearly interpolated between rain and hail
        sigma = np.zeros_like(fvol_wat)
        
        if sigma_opt == 0: # Ryzhkov et al. (2013)
            sigma[:, :] = sighail - (sighail - sigrain) * fvol_wat
            
        elif sigma_opt == 1:  # Kumjian et al. (2018)
            sig_idx = (fm[:, :, tstp] >= 0.95)
            sigma[sig_idx] = sigrain
            
            sig_idx = (fm[:, :, tstp] == 0.0)
            sigma[sig_idx] = sighail 
            
            sig_idx = (sigma == 0.0)
            ar_out_tmp = ar_out[:, :, tstp]
            sigma[sig_idx] = sighail + ((ar_in_tmp[sig_idx] - ar_out_tmp[sig_idx]) / (ar_in_tmp[sig_idx] - 0.56)) * (sigrain - sighail)
        
        elif sigma_opt == 2: # Dawson et al. (2014)
            sigma = np.where(fm[:, :, tstp] >= 0.5,
                             sigrain,
                             sighail + (fm[:, :, tstp] / 0.5) * (sigrain - sighail))

        # Compute angular moments from Ryzhkov et al. (2011)
        sig = (pi/180) * sigma
        uu = np.exp(-2.0 * sig**2)
        ang1 = 0.25 * (1 + uu)**2
        ang2 = 0.25 * (1 - uu**2)
        ang3 = (0.375 + 0.5 * uu + 0.125 * uu**4)**2
        ang4 = ((0.375 - 0.5 * uu + 0.125 * uu**4) *
                (0.375 + 0.5 * uu + 0.125 * uu**4))
        ang5 = 0.125 * (0.375 + 0.5 * uu + 0.125 * uu**4) * (1 - uu**4)
        ang7 = 0.5 * uu * (1 + uu)
        
        #######################################################################
        ######################## SCATTERING AMPLITUDES ########################
        #######################################################################        
        
        # Calculate temperature-dependent dielectric factor of water
        ew[:, :, tstp] = ew_vec(tp[:, :, tstp] + t0)
        
        # Fully removed particles
        gone_where = np.equal(d[:, :, tstp], 0.0)
        if np.count_nonzero(gone_where) > 0:
            fvv_180[gone_where, tstp] = fvv_0[gone_where, tstp] = 0.0
            fhh_180[gone_where, tstp] = fhh_0[gone_where, tstp] = 0.0

        # Fully melted particles:
        # Assume Rayleigh scattering with water dielectric constant
        melted_where = np.greater_equal(fm[:, :, tstp], 0.95)
        if np.count_nonzero(melted_where) > 0:
            eps_in[melted_where, tstp] = eps_out[melted_where, tstp] = ew[melted_where, tstp]
            la, lb = shape_factors_vec(ar_out[melted_where, tstp])    
            fvv_180[melted_where, tstp] = fvv_0[melted_where, tstp] = (((pi**2 * (1e3 * d[melted_where, tstp])**3) / (6 * wave**2)) * (1 / (la + (1 / (eps_out[melted_where, tstp] -1)))))
            fhh_180[melted_where, tstp] = fhh_0[melted_where, tstp] = (((pi**2 * (1e3 * d[melted_where, tstp])**3) / (6 * wave**2)) * (1 / (lb + (1 / (eps_out[melted_where, tstp] -1)))))  
            
        # Fully frozen particles
        # Assume Rayleigh scattering with Maxwell-Garnett-derived dielectric with ice matrix and air inclusions
        frozen_where = np.equal(fm[:, :, tstp], 0.0)
        if np.count_nonzero(frozen_where) > 0:
            y = (ea - ei) / (ea + 2 * ei)
            fvol = va[frozen_where, tstp] / (vi[frozen_where, tstp] + va[frozen_where, tstp])
            eps = np.where(va[frozen_where, tstp] > 0.0,
                           ei * (1 + ((3 * fvol * y) / (1 - fvol * y))),
                           ei)
            eps_in[frozen_where, tstp] = eps_out[frozen_where, tstp] = eps
            la, lb = shape_factors_vec(ar_out[frozen_where, tstp])    
            fvv_180[frozen_where, tstp] = fvv_0[frozen_where, tstp] = (((pi**2 * (1e3 * d[frozen_where, tstp])**3) / (6 * wave**2)) * (1 / (la + (1 / (eps_out[frozen_where, tstp] -1)))))
            fhh_180[frozen_where, tstp] = fhh_0[frozen_where, tstp] = (((pi**2 * (1e3 * d[frozen_where, tstp])**3) / (6 * wave**2)) * (1 / (lb + (1 / (eps_out[frozen_where, tstp] -1)))))  
              
        # Melting particles that are soaking and have no outer layer of water
        # Assume Rayleigh scattering using Maxwell-Garnett-derived dielectric
        # Assume: Matrix of spongy ice (ice matrix + air inclusions) with inclusions of water
        soaking_where = np.logical_and.reduce((np.greater(fm[:, :, tstp], 0.0),
                                               np.less(fm[:, :, tstp], 0.95),
                                               np.greater(va[:, :, tstp], 0.0)))
        
        if np.count_nonzero(soaking_where) > 0:
            # First calculate eps of dry spongy ice (mat: ice, inc: air):
            y = (ea - ei) / (ea + 2 * ei)
            fvol = va[soaking_where, tstp] / (vi[soaking_where, tstp] + va[soaking_where, tstp])
            eps1 = ei * (1 + ((3 * fvol * y) / (1 - fvol * y)))
    
            # Then calculate eps of water-soaked spongy ice (mat: spongy ice, inc: water):
            y = (ew[soaking_where, tstp] - eps1) / (ew[soaking_where, tstp] + 2 * eps1)
            fvol = (mw_inside[soaking_where, tstp] / rw) / ((mw_inside[soaking_where, tstp] / rw) + vi[soaking_where, tstp] + va[soaking_where, tstp])
            eps2 = eps1 * (1 + (3 * fvol * y)/(1 - fvol * y))
            eps_in[soaking_where, tstp] = eps_out[soaking_where, tstp] = eps2
     
            la, lb = shape_factors_vec(ar_out[soaking_where, tstp])    
            fvv_180[soaking_where, tstp] = fvv_0[soaking_where, tstp] = (((pi**2 * (1e3 * d[soaking_where, tstp])**3) / (6 * wave**2)) * (1 / (la + (1 / (eps_out[soaking_where, tstp] -1)))))
            fhh_180[soaking_where, tstp] = fhh_0[soaking_where, tstp] = (((pi**2 * (1e3 * d[soaking_where, tstp])**3) / (6 * wave**2)) * (1 / (lb + (1 / (eps_out[soaking_where, tstp] -1)))))  
                    
        # Melting particles that have an outer layer of water
        # Use Bohren and Huffman two-layer Rayleigh scattering equations (Ryzhkov et al. 2011; Eq. 28)
        # Assume dielectric of water on outer layer, and spongy ice (ice matrix + air inclusions) for core
        twolayer_where = np.logical_and.reduce((np.greater(fm[:, :, tstp], 0.0),
                                                np.less(fm[:, :, tstp], 0.95),
                                                np.equal(va[:, :, tstp], 0.0)))   
        
        if np.count_nonzero(twolayer_where) > 0:
            y = (ew0 - ei) / (ew0 + 2 * ei)
            fvol = (mw_inside[twolayer_where, tstp]/rw) / (vi[twolayer_where, tstp] + (mw_inside[twolayer_where, tstp]/rw))
            eps1 = ei * (1 + ((3 * fvol * y) / (1 - fvol * y)))
            eps_in[twolayer_where, tstp] = eps1
            eps_out[twolayer_where, tstp] = ew[twolayer_where, tstp]
            
            la_in, lb_in = shape_factors_vec(ar_in[twolayer_where, tstp])    
            la_out, lb_out = shape_factors_vec(ar_out[twolayer_where, tstp])    
            
            zeta = di[twolayer_where, tstp]**3 / d[twolayer_where, tstp]**3
            coef = ((pi**2 * (1e3 * d[twolayer_where, tstp])**3) / (6 * wave**2))
            
            num = (eps_out[twolayer_where, tstp] - 1) * (eps_out[twolayer_where, tstp] + (eps_in[twolayer_where, tstp] - eps_out[twolayer_where, tstp])*(la_in - zeta * la_out)) + zeta * eps_out[twolayer_where, tstp] * (eps_in[twolayer_where, tstp] - eps_out[twolayer_where, tstp])
            den = (eps_out[twolayer_where, tstp] + (eps_in[twolayer_where, tstp] - eps_out[twolayer_where, tstp])*(la_in - zeta * la_out)) * (1 + (eps_out[twolayer_where, tstp] - 1)*la_out) + zeta * la_out * eps_out[twolayer_where, tstp] * (eps_in[twolayer_where, tstp] - eps_out[twolayer_where, tstp])
            fvv_180[twolayer_where, tstp] = fvv_0[twolayer_where, tstp] = coef * (num / den)
            
            num = (eps_out[twolayer_where, tstp] - 1) * (eps_out[twolayer_where, tstp] + (eps_in[twolayer_where, tstp] - eps_out[twolayer_where, tstp])*(lb_in - zeta * lb_out)) + zeta * eps_out[twolayer_where, tstp] * (eps_in[twolayer_where, tstp] - eps_out[twolayer_where, tstp])
            den = (eps_out[twolayer_where, tstp] + (eps_in[twolayer_where, tstp] - eps_out[twolayer_where, tstp])*(lb_in - zeta * lb_out)) * (1 + (eps_out[twolayer_where, tstp] - 1)*lb_out) + zeta * lb_out * eps_out[twolayer_where, tstp] * (eps_in[twolayer_where, tstp] - eps_out[twolayer_where, tstp])
            fhh_180[twolayer_where, tstp] = fhh_0[twolayer_where, tstp] = coef * (num / den)    
            
        if use_lut and use_2layer:
            y = (eps_in[:, :, tstp] - eps_out[:, :, tstp]) / (eps_in[:, :, tstp] + 2 * eps_out[:, :, tstp])
            fvol = di[:, :, tstp]**3 / d[:, :, tstp]**3
            eps_comb = eps_out[:, :, tstp] * (1 + (3 * fvol * y)/(1 - fvol * y))
            
            # Resonance parameter
            resonance_param = (1e3 * d[:, :, tstp]) * np.sqrt(np.real(eps_comb)) / wave
            resonance_where = resonance_param > 20.0 # Arbitrary choice
            
            if np.count_nonzero(resonance_where) > 0:
                dout_idx = np.interp(1e3 * d[resonance_where, tstp], d_vec, np.arange(len(d_vec)))
                din_idx = np.interp(1e3 * di[resonance_where, tstp], d_vec, np.arange(len(d_vec)))
                arout_idx = np.maximum(np.interp(ar_out[resonance_where, tstp], ar_vec, np.arange(len(ar_vec))), 1)
                arin_idx = np.interp(ar_in[resonance_where, tstp], ar_vec, np.arange(len(ar_vec)))
                fvol_idx = np.interp(((mw_inside[resonance_where, tstp]/rw) / (vi[resonance_where, tstp] + (mw_inside[resonance_where, tstp]/rw))), fvol_vec, np.arange(len(fvol_vec)))
                melted_where = np.where(np.isnan(fvol_idx))
                if len(melted_where) > 0:
                    fvol_idx[melted_where] = 10 # Pure water for when mw_inside and vi == 0.0 (because all external)
                t_idx = np.interp(tp[resonance_where, tstp], t_vec, np.arange(len(t_vec)))


                coords = np.dstack((dout_idx, din_idx, arout_idx, arin_idx, fvol_idx, t_idx))
                coords = np.moveaxis(coords, 2, 0)

                # Interpolate LUT scattering amplitudes (in log-space) to particle characteristics
                # Note that LUT currently assume an inner aspect ratio of 0.8!!!
                # This is a BIG assumption that should ideally be alleviated in the future!
                fhh_180[resonance_where, tstp] = 10**(ndimage.map_coordinates(np.log10(fhh_180_lut), coords, order=1, mode='nearest'))
                fvv_180[resonance_where, tstp] = 10**(ndimage.map_coordinates(np.log10(fvv_180_lut), coords, order=1, mode='nearest'))
                fhh_0[resonance_where, tstp] = 10**(ndimage.map_coordinates(np.log10(fhh_0_lut), coords, order=1, mode='nearest'))
                fvv_0[resonance_where, tstp] = 10**(ndimage.map_coordinates(np.log10(fvv_0_lut), coords, order=1, mode='nearest'))    

        # Calculate radar variable values of individual particles following Eq. 29 of R11
        zhni[:, :, tstp] = [rc.calc_zh(x, y, a2, a4) for x, y, a2, a4 in zip(fhh_180[:, :, tstp], fvv_180[:, :, tstp], ang2, ang4)]
        zvni[:, :, tstp] = [rc.calc_zv(x, y, a1, a3) for x, y, a1, a3 in zip(fhh_180[:, :, tstp], fvv_180[:, :, tstp], ang1, ang3)]
        kdpi[:, :, tstp] = [rc.calc_kdp(x, y, a7) for x, y, a7 in zip(fhh_0[:, :, tstp], fvv_0[:, :, tstp], ang7)]
        ahi[:, :, tstp] = [rc.calc_ah(x, y, a2) for x, y, a2 in zip(fhh_0[:, :, tstp], fvv_0[:, :, tstp], ang2)]
        adpi[:, :, tstp] = [rc.calc_adp(x, y, a7) for x, y, a7 in zip(fhh_0[:, :, tstp], fvv_0[:, :, tstp], ang7)]
        deli[:, :, tstp] = [rc.calc_delta(x, y, a1, a2, a5) for x, y, a1, a2, a5 in zip(fhh_180[:, :, tstp], fvv_180[:, :, tstp], ang1, ang2, ang5)]
        ldri[:, :, tstp] = [rc.calc_ldr(x, y, a5) for x, y, a5 in zip(fhh_180[:, :, tstp], fvv_180[:, :, tstp], ang5)]

        # Sum over PSD and convert to standard units, etc.
        zp[:, tstp] = 10 * np.log10(np.nansum(zhni[:, :, tstp] * dsdm[:, :, tstp], axis=1)) # [dBZ]
        zdrp[:, tstp] = (10 * np.log10((np.nansum(zhni[:, :, tstp] * dsdm[:, :, tstp], axis=1)) /
                                 (np.nansum(zvni[:, :, tstp] * dsdm[:, :, tstp], axis=1)))) # [dB]
        kdpp[:, tstp] = np.nansum(kdpi[:, :, tstp] * dsdm[:, :, tstp], axis=1)  # [deg/km]
        ahp[:, tstp] = np.nansum(ahi[:, :, tstp] * dsdm[:, :, tstp], axis=1)  # [dB/km]
        adpp[:, tstp] = np.nansum(adpi[:, :, tstp] * dsdm[:, :, tstp], axis=1)  # [dB/km]
        delp[:, tstp] = ((180 / pi) * np.arctan2(np.nansum(-np.imag(deli[:, :, tstp]) * dsdm[:, :, tstp], axis=1),
                                                  np.nansum(np.real(deli[:, :, tstp]) * dsdm[:, :, tstp], axis=1))) # [deg] # Not sure if correct?
        ldrp[:, tstp] = 10 * np.log10(np.nansum(ldri[:, :, tstp] * dsdm[:, :, tstp], axis=1) / (10 ** (0.1 * zp[:, tstp])))
        rhvp[:, tstp] = abs(np.nansum(deli[:, :, tstp] * dsdm[:, :, tstp], axis=1)) / np.sqrt((np.nansum(zhni[:, :, tstp] * dsdm[:, :, tstp], axis=1) * np.nansum(zvni[:, :, tstp] * dsdm[:, :, tstp], axis=1)))

###############################################################################
# Substitute in hgt = 1 radar variables at hgt = 0 for plotting purposes
###############################################################################
zp[0, :] = zp[1, :]
zdrp[0, :] = zdrp[1, :]
kdpp[0, :] = kdpp[1, :]
ahp[0, :] = ahp[1, :]
adpp[0, :] = adpp[1, :]
delp[0, :] = delp[1, :]
ldrp[0, :] = ldrp[1, :]

###############################################################################
# End timer
###############################################################################

end_time = datetime.now()
print('Total runtime: ', end_time - start_time)


###############################################################################
# Write run data to netcdf file
###############################################################################

if write_netcdf:
    output_file = Dataset(netcdf_path, 'w', clobber=True, format='NETCDF3_64BIT')
              
    # Create file attributes
    output_file.title = 'Polarimetric downburst generation model'
    output_file.author = 'Jacob Carlin'
    output_file.contact = 'jacob.carlin@noaa.gov'
    
    output_file.createDimension('time', ntstp)
    output_file.createDimension('height', nlev)
    output_file.createDimension('bin', nbin)
    
    # Write variables
    nc_height = output_file.createVariable('height', 'f8', ('height'))
    nc_height.units = 'meters'
    nc_height.long_name = 'Vertical grid heights'
    nc_height[:] = heights
    
    nc_time = output_file.createVariable('time', 'f8', ('time'))
    nc_time.units = 'seconds'
    nc_time.long_name = 'Model time from t=0'
    nc_time[:] = np.arange(ntstp) * delt
    
    nc_ref = output_file.createVariable('ref', 'f8', ('height', 'time'))
    nc_ref.units = 'dBZ'
    nc_ref.long_name = 'Radar reflectivity factor at horizontal polarization'
    nc_ref[:, :] = zp[:, :]
    
    nc_zdr = output_file.createVariable('zdr', 'f8', ('height', 'time'))
    nc_zdr.units = 'dB'
    nc_zdr.long_name = 'Differential reflectivity'
    nc_zdr[:, :] = zdrp[:, :]
    
    nc_kdp = output_file.createVariable('kdp', 'f8', ('height', 'time'))
    nc_kdp.units = 'deg/km'
    nc_kdp.long_name = 'Specific differential phase'
    nc_kdp[:, :] = kdpp[:, :]
    
    nc_kdpi = output_file.createVariable('kdpi', 'f8', ('height', 'bin', 'time'))
    nc_kdpi.units = 'deg/km'
    nc_kdpi.long_name = 'Specific differential phase for particle'
    nc_kdpi[:, :, :] = kdpi[:, :, :]

    nc_zhni = output_file.createVariable('zhni', 'f8', ('height', 'bin', 'time'))
    nc_zhni.units = 'mm6/m3'
    nc_zhni.long_name = 'Horizontal reflectivity for particle'
    nc_zhni[:, :, :] = zhni[:, :, :]

    nc_zvni = output_file.createVariable('zvni', 'f8', ('height', 'bin', 'time'))
    nc_zvni.units = 'mm6/m3'
    nc_zvni.long_name = 'Vertical reflectivity for particle'
    nc_zvni[:, :, :] = zvni[:, :, :]

    nc_deli = output_file.createVariable('deli', 'f8', ('height', 'bin', 'time'))
    nc_deli.units = 'deg'
    nc_deli.long_name = 'Delta per particle'
    nc_deli[:, :, :] = deli[:, :, :]
    
    nc_delta = output_file.createVariable('delta', 'f8', ('height', 'time'))
    nc_delta.units = 'deg'
    nc_delta.long_name = 'Backscatter differential phase'
    nc_delta[:, :] = delp[:, :]
    
    nc_ah = output_file.createVariable('ahp', 'f8', ('height', 'time'))
    nc_ah.units = 'dB/km'
    nc_ah.long_name = 'Specific attenuation at horizontal polarization'
    nc_ah[:, :] = ahp[:, :]
    
    nc_adp = output_file.createVariable('adp', 'f8', ('height', 'time'))
    nc_adp.units = 'dB/km'
    nc_adp.long_name = 'Differential specific attenuation'
    nc_adp[:, :] = adpp[:, :]

    nc_ldrp = output_file.createVariable('ldrp', 'f8', ('height', 'time'))
    nc_ldrp.units = 'dB'
    nc_ldrp.long_name = 'Linear depolarization ratio'
    nc_ldrp[:, :] = ldrp[:, :]
    
    nc_dsdm = output_file.createVariable('dsdm', 'f8', ('height', 'bin', 'time'))
    nc_dsdm.units = 'm^-3'
    nc_dsdm.long_name = 'Particle number concentration'
    nc_dsdm[:, :, :] = dsdm[:, :, :]
    
    nc_mi = output_file.createVariable('mi', 'f8', ('height', 'bin', 'time'))
    nc_mi.units = 'kg'
    nc_mi.long_name = 'Particle ice mass'
    nc_mi[:, :, :] = mi[:, :, :]
    
    nc_mw = output_file.createVariable('mw', 'f8', ('height', 'bin', 'time'))
    nc_mw.units = 'kg'
    nc_mw.long_name = 'Particle water mass'
    nc_mw[:, :, :] = mw[:, :, :]
    
    nc_d = output_file.createVariable('d', 'f8', ('height', 'bin', 'time'))
    nc_d.units = 'm'
    nc_d.long_name = 'Particle equivolume diameter'
    nc_d[:, :, :] = d[:, :, :]

    nc_di = output_file.createVariable('di', 'f8', ('height', 'bin', 'time'))
    nc_di.units = 'm'
    nc_di.long_name = 'Particle inner core equivolume diameter'
    nc_di[:, :, :] = di[:, :, :]

    nc_nre = output_file.createVariable('nre', 'f8', ('height', 'bin', 'time'))
    nc_nre.units = 'None'
    nc_nre.long_name = 'Particle Reynolds number'
    nc_nre[:, :, :] = nre[:, :, :]

    nc_tp = output_file.createVariable('tp', 'f8', ('height', 'bin', 'time'))
    nc_tp.units = 'degC'
    nc_tp.long_name = 'Particle temperature'
    nc_tp[:, :, :] = tp[:, :, :]
    
    nc_ar = output_file.createVariable('ar', 'f8', ('height', 'bin', 'time'))
    nc_ar.units = 'None'
    nc_ar.long_name = 'Particle axis ratio'
    nc_ar[:, :, :] = ar[:, :, :]

    nc_arin = output_file.createVariable('arin', 'f8', ('height', 'bin', 'time'))
    nc_arin.units = 'None'
    nc_arin.long_name = 'Inner particle axis ratio'
    nc_arin[:, :, :] = ar_in[:, :, :]

    nc_arout = output_file.createVariable('arout', 'f8', ('height', 'bin', 'time'))
    nc_arout.units = 'None'
    nc_arout.long_name = 'Outer particle axis ratio'
    nc_arout[:, :, :] = ar_out[:, :, :]

    nc_va = output_file.createVariable('va', 'f8', ('height', 'bin', 'time'))
    nc_va.units = 'm3'
    nc_va.long_name = 'Particle air volume'
    nc_va[:, :, :] = va[:, :, :]

    nc_mwin = output_file.createVariable('mwin', 'f8', ('height', 'bin', 'time'))
    nc_mwin.units = 'kg'
    nc_mwin.long_name = 'Particle internal meltwater mass'
    nc_mwin[:, :, :] = mw_inside[:, :, :]

    nc_u = output_file.createVariable('u', 'f8', ('height', 'bin', 'time'))
    nc_u.units = 'm/s'
    nc_u.long_name = 'Particle fall speed'
    nc_u[:, :, :] = u[:, :, :]
    
    nc_dqdt = output_file.createVariable('dqdt', 'f8', ('height', 'bin', 'time'))
    nc_dqdt.units = 'kg/s'
    nc_dqdt.long_name = 'Particle rate of moisture transfer'
    nc_dqdt[:, :, :] = dqdt[:, :, :]

    nc_dTdt = output_file.createVariable('dTdt', 'f8', ('height', 'bin', 'time'))
    nc_dTdt.units = 'K/s'
    nc_dTdt.long_name = 'Particle rate of temperature change'
    nc_dTdt[:, :, :] = dTdt_evap[:, :, :] + dTdt_subl[:, :, :] + dTdt_melt[:, :, :]
    
    nc_mitot = output_file.createVariable('mitot', 'f8', ('height', 'time'))
    nc_mitot.units = 'g/m3'
    nc_mitot.long_name = 'Total ice mass in primary particles'
    nc_mitot[:, :] = mitot[:, :]

    nc_mwtot = output_file.createVariable('mwtot', 'f8', ('height', 'time'))
    nc_mwtot.units = 'g/m3'
    nc_mwtot.long_name = 'Total water mass in primary particles'
    nc_mwtot[:, :] = mwtot[:, :]

    nc_mstot = output_file.createVariable('mstot', 'f8', ('height', 'time'))
    nc_mstot.units = 'g/m3'
    nc_mstot.long_name = 'Total water mass shed at each height'
    nc_mstot[:, :] = mstot[:, :]

    nc_w = output_file.createVariable('w', 'f8', ('height', 'time'))
    nc_w.units = 'm/s'
    nc_w.long_name = 'Vertical velocity'
    nc_w[:, :] = w[:, :]

    nc_dwterm1 = output_file.createVariable('dw_term1', 'f8', ('height', 'time'))
    nc_dwterm1.units = 'm/s2'
    nc_dwterm1.long_name = 'Vertical velocity Tendency Term 1'
    nc_dwterm1[:, :] = dw_term1[:, :]

    nc_dwterm2 = output_file.createVariable('dw_term2', 'f8', ('height', 'time'))
    nc_dwterm2.units = 'm/s2'
    nc_dwterm2.long_name = 'Vertical velocity Tendency Term 2'
    nc_dwterm2[:, :] = dw_term2[:, :]

    nc_dwterm3 = output_file.createVariable('dw_term3', 'f8', ('height', 'time'))
    nc_dwterm3.units = 'm/s2'
    nc_dwterm3.long_name = 'Vertical velocity Tendency Term 3'
    nc_dwterm3[:, :] = dw_term3[:, :]

    nc_dwterm4 = output_file.createVariable('dw_term4', 'f8', ('height', 'time'))
    nc_dwterm4.units = 'm/s2'
    nc_dwterm4.long_name = 'Vertical velocity Tendency Term 4'
    nc_dwterm4[:, :] = dw_term4[:, :]

    nc_t = output_file.createVariable('t', 'f8', ('height', 'time'))
    nc_t.units = 'C'
    nc_t.long_name = 'Downdraft temperature'
    nc_t[:, :] = t[:, :]
    
    nc_rh = output_file.createVariable('rh', 'f8', ('height', 'time'))
    nc_rh.units = '%'
    nc_rh.long_name = 'Relative Humidity'
    nc_rh[:, :] = rh[:, :]

    nc_q = output_file.createVariable('q', 'f8', ('height', 'time'))
    nc_q.units = 'kg/kg'
    nc_q.long_name = 'Water vapor mixing ratio'
    nc_q[:, :] = q[:, :]

    nc_qp = output_file.createVariable('qp', 'f8', ('height', 'time'))
    nc_qp.units = 'kg/kg'
    nc_qp.long_name = 'Precipitation mixing ratio'
    nc_qp[:, :] = qp[:, :]
    
    output_file.close()
    
    if verbose:
        print('netCDF output file written!')

###############################################################################
# Plotting routines
###############################################################################

if make_plots:
    
    def make_colormap(colors):
        """
    
        Parameters
        ----------
        colors : dict
            Dictionary of values between 0 and 1 (colormap endpoints) with associated
            color values (usually RGB) at discrete points along this range.
    
        Returns
        -------
        mymap : Linear Segmented Colormap
            Output colormap
    
        """
    
        z  = np.array(sorted(colors.keys()))
        n  = len(z)
        z1 = min(z)
        zn = max(z)
        x0 = (z - z1) / (zn - z1)
    
        CC = ColorConverter()
        R = []
        G = []
        B = []
        for i in range(n):
            Ci = colors[z[i]]
            if type(Ci) == str:
                RGB = CC.to_rgb(Ci)
            else:
                RGB = Ci
            R.append(RGB[0])
            G.append(RGB[1])
            B.append(RGB[2])
    
        cmap_dict = {}
        cmap_dict['red']   = [(x0[i],R[i],R[i]) for i in range(len(R))]
        cmap_dict['green'] = [(x0[i],G[i],G[i]) for i in range(len(G))]
        cmap_dict['blue']  = [(x0[i],B[i],B[i]) for i in range(len(B))]
        mymap = LinearSegmentedColormap('mymap',cmap_dict)
    
        return mymap

    colors = ({0.000: (0.00, 0.00, 0.00),
               0.330: (0.86, 0.86, 0.86),
               0.335: (0.55, 0.47, 0.71),
               0.354: (0.04, 0.04, 0.61),
               0.416: (0.26, 0.97, 0.83),
               0.458: (0.35, 0.86, 0.38),
               0.500: (1.00, 1.00, 0.40),
               0.580: (0.86, 0.04, 0.02),
               0.666: (0.69, 0.00, 0.00),
               0.750: (0.94, 0.47, 0.71),
               0.833: (1.00, 1.00, 1.00),
               1.000: (0.57, 0.18, 0.58)})
    zdr_cmap = make_colormap(colors) # Create colormap for ZDR similar to operational cmap

    colors = ({0.000: (0.55, 0.47, 0.71),
               0.031: (0.04, 0.04, 0.61),
               0.124: (0.26, 0.97, 0.83),
               0.184: (0.35, 0.86, 0.38),
               0.250: (1.00, 1.00, 0.40),
               0.370: (0.86, 0.04, 0.02),
               0.500: (0.69, 0.00, 0.00),
               0.625: (0.94, 0.47, 0.71),
               0.750: (1.00, 1.00, 1.00),
               1.000: (0.57, 0.18, 0.58)})
    zdr_cmap_new = make_colormap(colors)


    colors = ({0.000: (0.00, 0.00, 0.00),
               0.083: (0.75, 0.75, 0.75),
               0.166: (0.60, 0.00, 0.00),
               0.283: (1.00, 0.40, 1.00),
               0.350: (0.40, 0.00, 0.80),
               0.400: (0.00, 1.00, 1.00),
               0.433: (0.00, 0.80, 0.00),
               0.558: (1.00, 1.00, 0.00),
               0.650: (1.00, 0.50, 0.00),
               1.000: (1.00, 1.00, 1.00)})
    kdp_cmap = make_colormap(colors) # Create colormap for KDP similar to operational cmap
    
    colors = ({0.00: (0.00, 0.00, 1.00),
               0.20: (0.33, 0.78, 0.43),
               0.40: (0.43, 0.82, 0.10),
               0.50: (1.00, 1.00, 0.00),
               0.80: (1.00, 0.00, 0.00),
               0.90: (0.12, 0.08, 0.08),
               0.95: (0.20, 0.00, 0.00),
               1.00: (0.69, 0.69, 0.69)})
    
    rhv_cmap = make_colormap(colors)
    
    time_del_idx = int(100 / delt) # Every 100 s
    
    
    colors = plt.cm.PuBuGn(np.linspace(0,1,7))
    

    # Ice mixing ratio
    fig = plt.figure(figsize=(3.5, 5))
    plt.plot(mitot[:, time_del_idx]/ra[:, time_del_idx], heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(mitot[:, time_del_idx*2]/ra[:, time_del_idx], heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(mitot[:, time_del_idx*3]/ra[:, time_del_idx], heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(mitot[:, time_del_idx*4]/ra[:, time_del_idx], heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(mitot[:, time_del_idx*5]/ra[:, time_del_idx], heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(mitot[:, time_del_idx*6]/ra[:, time_del_idx], heights*1e-3, label='600 s', lw=3, color='k');
    plt.legend()
    plt.title(r'$q_{\mathrm{i}}$')
    plt.xlim(0, 6)
    plt.xticks([0, 1, 2, 3, 4, 5, 6])
    plt.xlabel('g kg$^{-1}$')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(0.35, 3.7, 'a', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_qi_profiles_newAR.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, mitot/ra, vmin=0, vmax=5, cmap='BuPu')
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'$q_{\mathrm{i}}$')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label=r'g kg$^{-1}$')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.text(25, 0.2, 'b', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_qi_TH_newAR.png', dpi=300, bbox_inches='tight')

    # Ice IWC
    fig = plt.figure(figsize=(3.5, 5))
    plt.plot(mitot[:, time_del_idx], heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(mitot[:, time_del_idx*2], heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(mitot[:, time_del_idx*3], heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(mitot[:, time_del_idx*4], heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(mitot[:, time_del_idx*5], heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(mitot[:, time_del_idx*6], heights*1e-3, label='600 s', lw=3, color='k');
    plt.legend()
    plt.title(r'Ice Mass Content')
    plt.xlim(0, 5)
    plt.xticks([0, 1, 2, 3, 4, 5])
    plt.xlabel('g m$^{-3}$')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(0.35, 3.7, 'a', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_iwc_profiles_newAR.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, mitot, vmin=0, vmax=5, cmap='BuPu')
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'Ice Mass Content')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label=r'g m$^{-3}$')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.text(25, 0.2, 'b', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_iwc_TH_newAR.png', dpi=300, bbox_inches='tight')

    # Water mixing ratio
    fig = plt.figure(figsize=(3.5, 5))
    plt.plot(mwtot[:, time_del_idx]/ra[:, time_del_idx], heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(mwtot[:, time_del_idx*2]/ra[:, time_del_idx], heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(mwtot[:, time_del_idx*3]/ra[:, time_del_idx], heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(mwtot[:, time_del_idx*4]/ra[:, time_del_idx], heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(mwtot[:, time_del_idx*5]/ra[:, time_del_idx], heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(mwtot[:, time_del_idx*6]/ra[:, time_del_idx], heights*1e-3, label='600 s', lw=3, color='k');
    #plt.legend()
    plt.title(r'$q_{\mathrm{r}}$')
    plt.xlim(0, 6)
    plt.xticks([0, 1, 2, 3, 4, 5, 6])
    plt.xlabel('g kg$^{-1}$')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(5.35, 3.7, 'c', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_qr_profiles_newAR.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, mwtot/ra, vmin=0, vmax=5, cmap='BuPu')
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'$q_{\mathrm{r}}$')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label=r'g kg$^{-1}$')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.text(25, 0.2, 'd', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_qr_TH_newAR.png', dpi=300, bbox_inches='tight')

# Water LWC
    fig = plt.figure(figsize=(3.5, 5))
    plt.plot(mwtot[:, time_del_idx], heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(mwtot[:, time_del_idx*2], heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(mwtot[:, time_del_idx*3], heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(mwtot[:, time_del_idx*4], heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(mwtot[:, time_del_idx*5], heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(mwtot[:, time_del_idx*6], heights*1e-3, label='600 s', lw=3, color='k');
    #plt.legend()
    plt.title(r'Liquid Mass Content')
    plt.xlim(0, 5)
    plt.xticks([0, 1, 2, 3, 4, 5])
    plt.xlabel('g m$^{-3}$')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(4.45, 3.7, 'c', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_lwc_profiles_newAR.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, mwtot, vmin=0, vmax=5, cmap='BuPu')
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'Liquid Mass Content')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label=r'g m$^{-3}$')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.text(25, 0.2, 'd', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_lwc_TH_newAR.png', dpi=300, bbox_inches='tight')



    # Relative Humidity
    fig = plt.figure(figsize=(3.5, 5))
    plt.plot(rh[:, time_del_idx], heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(rh[:, time_del_idx*2], heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(rh[:, time_del_idx*3], heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(rh[:, time_del_idx*4], heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(rh[:, time_del_idx*5], heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(rh[:, time_del_idx*6], heights*1e-3, label='600 s', lw=3, color='k');
    #plt.legend()
    plt.title(r'RH$_{\mathrm{w}}$')
    plt.xlim(0, 100)
    plt.xlabel('%')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(5, 3.7, 'e', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_RH_profiles_newAR.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, rh, vmin=0, vmax=100, cmap='BrBG')
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'RH$_{\mathrm{w}}$')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label=r'%')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.text(25, 0.2, 'f', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_RH_TH_newAR.png', dpi=300, bbox_inches='tight')

    
    # Vertical velocity
    fig = plt.figure(figsize=(3.5,5))
    plt.plot(w[:, time_del_idx], heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(w[:, time_del_idx*2], heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(w[:, time_del_idx*3], heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(w[:, time_del_idx*4], heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(w[:, time_del_idx*5], heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(w[:, time_del_idx*6], heights*1e-3, label='600 s', lw=3, color='k');
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=6)
    #plt.legend()
    plt.title(r'$w$')
    plt.xlim(-20, 0)
    plt.xlabel(r'm s$^{-1}$')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(-19, 3.7, 'g', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_w_profiles_newAR.png', dpi=300, bbox_inches='tight')
    
    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, w, vmin=-20, vmax=20, cmap='RdBu_r')
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'$w$')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label=r'm s$^{-1}$')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.text(25, 0.2, 'h', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_w_TH_newAR.png', dpi=300, bbox_inches='tight')



    # Reflectivity
    fig = plt.figure(figsize=(3.5, 5))
    plt.plot(zp[:, time_del_idx], heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(zp[:, time_del_idx*2], heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(zp[:, time_del_idx*3], heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(zp[:, time_del_idx*4], heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(zp[:, time_del_idx*5], heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(zp[:, time_del_idx*6], heights*1e-3, label='600 s', lw=3, color='k');
    #plt.legend()
    plt.title(r'$Z_{\mathrm{H}}$')
    plt.xlim(30, 70)
    plt.xticks([30, 40, 50, 60, 70])
    plt.xlabel('dBZ')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(32, 3.7, 'a', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_Z_profiles_newAR.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, zp, vmin=0, vmax=70, cmap='pyart_ChaseSpectral')
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'$Z_{\mathrm{H}}$')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label='dBZ')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.text(25, 0.2, 'b', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_Z_TH_newAR.png', dpi=300, bbox_inches='tight')

    # ZDR
    zdrp_tmp = zdrp.copy()
    zdrp[zp < 0] = np.nan
    
    fig = plt.figure(figsize=(3.5, 5))
    plt.plot(zdrp[:, time_del_idx], heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(zdrp[:, time_del_idx*2], heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(zdrp[:, time_del_idx*3], heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(zdrp[:, time_del_idx*4], heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(zdrp[:, time_del_idx*5], heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(zdrp[:, time_del_idx*6], heights*1e-3, label='600 s', lw=3, color='k');
    plt.legend()
    plt.title(r'$Z_{\mathrm{DR}}$')
    plt.xlim(0, 4)
    plt.xticks([0, 1, 2, 3, 4])
    plt.xlabel('dB')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(0.2, 0.2, 'c', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_ZDR_profiles_newAR.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, zdrp, vmin=0, vmax=4, cmap=zdr_cmap_new)
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'$Z_{\mathrm{DR}}$')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label='dB')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.text(25, 0.2, 'd', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_ZDR_TH_newAR.png', dpi=300, bbox_inches='tight')

    # KDP
    kdpp_tmp = kdpp.copy()
    kdpp[zp < 0] = np.nan
    
    fig = plt.figure(figsize=(3.5, 5))
    plt.plot(kdpp[:, time_del_idx], heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(kdpp[:, time_del_idx*2], heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(kdpp[:, time_del_idx*3], heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(kdpp[:, time_del_idx*4], heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(kdpp[:, time_del_idx*5], heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(kdpp[:, time_del_idx*6], heights*1e-3, label='600 s', lw=3, color='k');
    #plt.legend()
    plt.title(r'$K_{\mathrm{dp}}$')
    plt.xlim(0, 4)
    plt.xlabel(r'$^\circ$ km$^{-1}$')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(3.6, 3.7, 'e', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_KDP_profiles_newAR.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, kdpp, vmin=0, vmax=8, cmap=zdr_cmap_new)
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'$K_{\mathrm{dp}}$')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label=r'$^\circ$ km$^{-1}$')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.text(25, 0.2, 'f', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_KDP_TH_newAR.png', dpi=300, bbox_inches='tight')

    
    # AH
    ahp_tmp = ahp.copy()
    ahp[zp < 0] = np.nan
    
    fig = plt.figure(figsize=(3.5,5))
    plt.plot(ahp[:, time_del_idx], heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(ahp[:, time_del_idx*2], heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(ahp[:, time_del_idx*3], heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(ahp[:, time_del_idx*4], heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(ahp[:, time_del_idx*5], heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(ahp[:, time_del_idx*6], heights*1e-3, label='600 s', lw=3, color='k');
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=6)
    #plt.legend()
    plt.title(r'$A_{\mathrm{H}}$')
    plt.xlim(0, 0.4)
    plt.xlabel(r'dB km$^{-1}$')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(0.36, 3.7, 'g', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_AH_profiles_newAR.png', dpi=300, bbox_inches='tight')
    
    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, ahp, vmin=0, vmax=0.4, cmap='pyart_ChaseSpectral')
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'$A_{\mathrm{H}}$')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label=r'dB km$^{-1}$')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='w')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='w')
    plt.text(25, 0.2, 'h', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_AH_TH_newAR.png', dpi=300, bbox_inches='tight')


    # RhoHV
    rhvp_tmp = rhvp.copy()
    rhvp[zp < 0] = np.nan
    
    fig = plt.figure(figsize=(3.5,5))
    plt.plot(rhvp[:, time_del_idx], heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(rhvp[:, time_del_idx*2], heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(rhvp[:, time_del_idx*3], heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(rhvp[:, time_del_idx*4], heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(rhvp[:, time_del_idx*5], heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(rhvp[:, time_del_idx*6], heights*1e-3, label='600 s', lw=3, color='k');
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=6)
    #plt.legend()
    plt.title(r'$\rho_{\mathrm{hv}}$')
    plt.xlim(0.9, 1.0)
    plt.xlabel(r'')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(0.91, 3.7, 'g', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_RhoHV_profiles_newAR.png', dpi=300, bbox_inches='tight')
    
    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, rhvp, vmin=0.90, vmax=1.0, cmap=rhv_cmap)
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'$\rho_{\mathrm{hv}}$')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label=r'   ', ticks=[0.90, 0.95, 0.98, 0.99, 1.0])
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='w')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='w')
    plt.text(25, 0.2, 'h', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_RhoHV_TH_newAR.png', dpi=300, bbox_inches='tight')




    # Thermal Buoyancy Acceleration

    fig = plt.figure(figsize=(3.5, 5))
    plt.axvline(0.0, lw=0.5, color='grey')
    plt.plot(g*(tv[:, time_del_idx] - tvenv)/(tvenv) * 1e2, heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(g*(tv[:, time_del_idx*2] - tvenv)/(tvenv) * 1e2, heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(g*(tv[:, time_del_idx*3] - tvenv)/(tvenv) * 1e2, heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(g*(tv[:, time_del_idx*4] - tvenv)/(tvenv) * 1e2, heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(g*(tv[:, time_del_idx*5] - tvenv)/(tvenv) * 1e2, heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(g*(tv[:, time_del_idx*6] - tvenv)/(tvenv) * 1e2, heights*1e-3, label='600 s', lw=3, color='k');
    #plt.legend()
    plt.title(r'Buoyancy Acceleration')
    plt.xlim(-8, 8)
    plt.xticks([-8, -4, 0, 4, 8])
    plt.xlabel('10$^2$ m s$^{-2}$')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(-7, 3.7, 'a', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_thermal_buoyancy_profiles_newAR.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, 1e2*g * (tv - tvenv[:, np.newaxis])/tvenv[:, np.newaxis], vmin=-5, vmax=5, cmap='RdBu_r')
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'Buoyancy Acceleration')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label='10$^2$ m s$^{-2}$', ticks=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.text(25, 0.2, 'b', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_thermal_buoyancy_TH_newAR.png', dpi=300, bbox_inches='tight')

    # Precipitation Loading
    fig = plt.figure(figsize=(3.5, 5))
    plt.axvline(0.0, lw=0.5, color='grey')
    plt.plot(g * -qp[:, time_del_idx] * 1e2, heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(g * -qp[:, time_del_idx*2] * 1e2, heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(g * -qp[:, time_del_idx*3] * 1e2, heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(g * -qp[:, time_del_idx*4] * 1e2, heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(g * -qp[:, time_del_idx*5] * 1e2, heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(g * -qp[:, time_del_idx*6] * 1e2, heights*1e-3, label='600 s', lw=3, color='k');
    #plt.legend()
    plt.title(r'Precip. Loading Acceleration')
    plt.xlim(-8, 8)
    plt.xticks([-8, -4, 0, 4, 8])
    plt.xlabel('10$^2$ m s$^{-2}$')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(-7, 3.7, 'c', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_precip_loading_profiles_newAR.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, 1e2 * g * -qp, vmin=-5, vmax=5, cmap='RdBu_r')
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'Precip. Loading Acceleration')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label='10$^2$ m s$^{-2}$', ticks=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.text(25, 0.2, 'd', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_precip_loading_TH_newAR.png', dpi=300, bbox_inches='tight')

    # dTdt due to melting
    fig = plt.figure(figsize=(3.5, 5))
    plt.plot((dTdt_melt_tot[:, time_del_idx]+dTdt_subl_tot[:, time_del_idx])*3600, heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot((dTdt_melt_tot[:, time_del_idx*2]+dTdt_subl_tot[:, time_del_idx*2])*3600, heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot((dTdt_melt_tot[:, time_del_idx*3]+dTdt_subl_tot[:, time_del_idx*3])*3600, heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot((dTdt_melt_tot[:, time_del_idx*4]+dTdt_subl_tot[:, time_del_idx*4])*3600, heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot((dTdt_melt_tot[:, time_del_idx*5]+dTdt_subl_tot[:, time_del_idx*5])*3600, heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot((dTdt_melt_tot[:, time_del_idx*6]+dTdt_subl_tot[:, time_del_idx*6])*3600, heights*1e-3, label='600 s', lw=3, color='k');
    #plt.legend()
    plt.title(r'$\partial T / \partial t_{\mathrm{melt+subl}}$')
    plt.xlim(-80, 0)
    plt.xlabel(r'K h$^{-1}$')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(-76, 3.7, 'e', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_dTdt_melt_profiles_newAR.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, (dTdt_melt_tot+dTdt_subl_tot)*3600, vmin=-60, vmax=0, cmap='Blues_r')
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'$\partial T / \partial t_{\mathrm{melt+subl}}$')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label=r'K h$^{-1}$')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, dTdt_subl_tot*3600, [], linewidths=2, colors='r')
    plt.text(25, 0.2, 'f', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_dTdt_melt_TH_newAR.png', dpi=300, bbox_inches='tight')

    
    # dTdt due to evaporation
    fig = plt.figure(figsize=(3.5, 5))
    plt.plot(dTdt_evap_tot[:, time_del_idx]*3600, heights*1e-3, label='100 s', lw=3, color=colors[2]);
    plt.plot(dTdt_evap_tot[:, time_del_idx*2]*3600, heights*1e-3, label='200 s', lw=3, color=colors[3]);
    plt.plot(dTdt_evap_tot[:, time_del_idx*3]*3600, heights*1e-3, label='300 s', lw=3, color=colors[4]);
    plt.plot(dTdt_evap_tot[:, time_del_idx*4]*3600, heights*1e-3, label='400 s', lw=3, color=colors[5]);
    plt.plot(dTdt_evap_tot[:, time_del_idx*5]*3600, heights*1e-3, label='500 s', lw=3, color=colors[6]);
    plt.plot(dTdt_evap_tot[:, time_del_idx*6]*3600, heights*1e-3, label='600 s', lw=3, color='k');
    #plt.legend()
    plt.title(r'$\partial T / \partial t_{\mathrm{evap}}$')
    plt.xlim(-80, 0)
    plt.xlabel(r'K h$^{-1}$')
    plt.ylim(0, 4)
    plt.ylabel(r'km AGL')
    plt.text(-76, 3.7, 'g', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_dTdt_evap_profiles_newAR.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.pcolor(np.arange(0, ntstp) * delt, heights*1e-3, dTdt_evap_tot*3600, vmin=-60, vmax=0, cmap='Blues_r')
    plt.xlim(0, 600)
    plt.ylim(0, 4)
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks(np.arange(0, 4.5, 0.5), labels=['', '', '', '', '', '', '', '', ''])
    plt.title(r'$\partial T / \partial t_{\mathrm{evap}}$')
    plt.xlabel('s')
    plt.ylabel('')
    plt.colorbar(label=r'K h$^{-1}$')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-40, 0, 10), linewidths=2, colors='k')
    plt.contour(np.arange(0, ntstp) * delt, heights*1e-3, w, np.arange(-35, 0, 10), linewidths=1, colors='k')
    plt.text(25, 0.2, 'h', bbox=dict(boxstyle='round', facecolor='white'))
    if save_plots:
        plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/casestudy_dTdt_evap_TH_newAR.png', dpi=300, bbox_inches='tight')



















    sys.exit()




    # Diabatic cooling rate
    fig = plt.figure(figsize=(4,6))
    plt.plot(dTdt_tot[:, time_del_idx]*3600, heights, label='100 s', lw=2)
    plt.plot(dTdt_tot[:, time_del_idx*2]*3600, heights, label='200 s', lw=2)
    plt.plot(dTdt_tot[:, time_del_idx*3]*3600, heights, label='300 s', lw=2)
    plt.plot(dTdt_tot[:, time_del_idx*4]*3600, heights, label='400 s', lw=2)
    plt.plot(dTdt_tot[:, time_del_idx*5]*3600, heights, label='500 s', lw=2)
    plt.plot(dTdt_tot[:, time_del_idx*6]*3600, heights, label='600 s', lw=2)
    plt.legend()
    plt.title('Latent Heating Rate')
    plt.xlim(-300, 0)
    plt.xlabel('K/h')
    
    # Temperature perturbation
    fig = plt.figure(figsize=(4,6))
    plt.plot(t[:, time_del_idx] - tenv, heights, label='100 s', lw=2)
    plt.plot(t[:, time_del_idx*2] - tenv, heights, label='200 s', lw=2)
    plt.plot(t[:, time_del_idx*3] - tenv, heights, label='300 s', lw=2)
    plt.plot(t[:, time_del_idx*4] - tenv, heights, label='400 s', lw=2)
    plt.plot(t[:, time_del_idx*5] - tenv, heights, label='500 s', lw=2)
    plt.plot(t[:, time_del_idx*6] - tenv, heights, label='600 s', lw=2)
    plt.legend()
    plt.title('Temperature Perturbation')
    plt.xlim(-5, 5)
    plt.xlabel('K')
    

    
    # Thermal component of buoyancy
    fig = plt.figure(figsize=(4,6))
    plt.plot((tv[:, time_del_idx] - tvenv)/(tvenv) * 1e3, heights, label='100 s', lw=2)
    plt.plot((tv[:, time_del_idx*2] - tvenv)/(tvenv) * 1e3, heights, label='200 s', lw=2)
    plt.plot((tv[:, time_del_idx*3] - tvenv)/(tvenv) * 1e3, heights, label='300 s', lw=2)
    plt.plot((tv[:, time_del_idx*4] - tvenv)/(tvenv) * 1e3, heights, label='400 s', lw=2)
    plt.plot((tv[:, time_del_idx*5] - tvenv)/(tvenv) * 1e3, heights, label='500 s', lw=2)
    plt.plot((tv[:, time_del_idx*6] - tvenv)/(tvenv) * 1e3, heights, label='600 s', lw=2)
    plt.legend()
    plt.xlim(-8, 8)
    plt.title('Thermal Buoyancy Acceleration (x1000)')
    plt.xlabel('m/s2')
    
    # Precipitation loading component of buoyancy
    fig = plt.figure(figsize=(4,6))
    plt.plot(-qp[:, time_del_idx] * 1e3, heights, label='100 s', lw=2)
    plt.plot(-qp[:, time_del_idx*2] * 1e3, heights, label='200 s', lw=2)
    plt.plot(-qp[:, time_del_idx*3] * 1e3, heights, label='300 s', lw=2)
    plt.plot(-qp[:, time_del_idx*4] * 1e3, heights, label='400 s', lw=2)
    plt.plot(-qp[:, time_del_idx*5] * 1e3, heights, label='500 s', lw=2)
    plt.plot(-qp[:, time_del_idx*6] * 1e3, heights, label='600 s', lw=2)
    plt.legend()
    plt.xlim(-8, 8)
    plt.title('Precipitation Loading Acceleration (x1000)')
    plt.xlabel('m/s2')
    
    # Total LWC
    fig = plt.figure(figsize=(4,6))
    plt.plot(mwtot[:, time_del_idx], heights, label='100 s', lw=2)
    plt.plot(mwtot[:, time_del_idx*2], heights, label='200 s', lw=2)
    plt.plot(mwtot[:, time_del_idx*3], heights, label='300 s', lw=2)
    plt.plot(mwtot[:, time_del_idx*4], heights, label='400 s', lw=2)
    plt.plot(mwtot[:, time_del_idx*5], heights, label='500 s', lw=2)
    plt.plot(mwtot[:, time_del_idx*6], heights, label='600 s', lw=2)
    plt.legend()
    plt.title('Liquid Water Content')
    plt.xlabel(r'g m$^{-3}$')
    plt.xlim(0, 6)
    
    # Total IWC 
    fig = plt.figure(figsize=(4,6))
    plt.plot(mitot[:, time_del_idx], heights, label='100 s', lw=2)
    plt.plot(mitot[:, time_del_idx*2], heights, label='200 s', lw=2)
    plt.plot(mitot[:, time_del_idx*3], heights, label='300 s', lw=2)
    plt.plot(mitot[:, time_del_idx*4], heights, label='400 s', lw=2)
    plt.plot(mitot[:, time_del_idx*5], heights, label='500 s', lw=2)
    plt.plot(mitot[:, time_del_idx*6], heights, label='600 s', lw=2)
    plt.legend()
    plt.title('Ice Water Content')
    plt.xlabel(r'g m$^{-3}$')
    plt.xlim(0, 6)
    
    # Reflectivity
    fig = plt.figure(figsize=(4,6))
    plt.plot(zp[:, time_del_idx], heights, label='100 s', lw=2)
    plt.plot(zp[:, time_del_idx*2], heights, label='200 s', lw=2)
    plt.plot(zp[:, time_del_idx*3], heights, label='300 s', lw=2)
    plt.plot(zp[:, time_del_idx*4], heights, label='400 s', lw=2)
    plt.plot(zp[:, time_del_idx*5], heights, label='500 s', lw=2)
    plt.plot(zp[:, time_del_idx*6], heights, label='600 s', lw=2)
    plt.legend()
    plt.title(r'$Z$')
    plt.xlim(0, 70)
    plt.xlabel('dBZ')
    
    # Differential reflectivity
    fig = plt.figure(figsize=(4,6))
    plt.plot(zdrp[:, time_del_idx], heights, label='100 s', lw=2)
    plt.plot(zdrp[:, time_del_idx*2], heights, label='200 s', lw=2)
    plt.plot(zdrp[:, time_del_idx*3], heights, label='300 s', lw=2)
    plt.plot(zdrp[:, time_del_idx*4], heights, label='400 s', lw=2)
    plt.plot(zdrp[:, time_del_idx*5], heights, label='500 s', lw=2)
    plt.plot(zdrp[:, time_del_idx*6], heights, label='600 s', lw=2)
    plt.legend()
    plt.title(r'$Z_{\mathrm{DR}}$')
    plt.xlim(0, 4)
    plt.xlabel('dB')
    
    # Specific differential phase
    fig = plt.figure(figsize=(4,6))
    plt.plot(kdpp[:, time_del_idx], heights, label='100 s', lw=2)
    plt.plot(kdpp[:, time_del_idx*2], heights, label='200 s', lw=2)
    plt.plot(kdpp[:, time_del_idx*3], heights, label='300 s', lw=2)
    plt.plot(kdpp[:, time_del_idx*4], heights, label='400 s', lw=2)
    plt.plot(kdpp[:, time_del_idx*5], heights, label='500 s', lw=2)
    plt.plot(kdpp[:, time_del_idx*6], heights, label='600 s', lw=2)
    plt.legend()
    plt.title(r'$K_{\mathrm{dp}}$')
    plt.xlim(0, 5)
    plt.xlabel(r'$^\circ$ km$^{-1}$')
    
    # Surface (estimated horizontal) wind speed
    fig = plt.figure(figsize=(6,4))
    plt.plot(np.arange(ntstp)*delt, sfc_wind, lw=2, color='k')
    plt.title('Maximum Estimated Surface Wind Speed')
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [mph]')
    plt.xlim(0, delt*ntstp)
    plt.ylim(0, 80)
    
    # Surface temperature and dewpoint
    fig = plt.figure(figsize=(6,4))
    plt.plot(np.arange(ntstp)*delt, sfc_t, lw=2, label='Temperature', color='red')
    plt.plot(np.arange(ntstp)*delt, sfc_td, lw=2, label='Dewpoint', color='green', linestyle='dashed')
    plt.legend()
    plt.title('Surface meteogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [C]')
    plt.xlim(0, delt*ntstp)
    plt.ylim(0, 40)
    
    
    colors = plt.cm.YlGnBu_r(np.linspace(0, 1, 7))
    fig = plt.figure(figsize=(6,5))
    plt.plot(1e3*d[0, :, 0], ar[0, :, 0], lw=3, label='4 km AGL', color=colors[0])
    plt.plot(1e3*d[20, :, 0], ar[20, :, 0], lw=3, label='3 km AGL', color=colors[1])
    plt.plot(1e3*d[40, :, 0], ar[40, :, 0], lw=3, label='2 km AGL', color=colors[2])
    plt.plot(1e3*d[60, :, 0], ar[60, :, 0], lw=3, label='1 km AGL', color=colors[3])
    plt.plot(1e3*d[80, :, 0], ar[80, :, 0], lw=3, label='Surface', color=colors[4]) 
    plt.ylim(0.5, 1.0)
    plt.xlim(0, 20)
    plt.legend(loc = 1, fontsize=12)
    plt.ylabel('Aspect Ratio')
    plt.xlabel('Equivolume Diameter [mm]')
    plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/example_d_vs_ar_tmp.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(6,5))
    plt.plot(1e3*d[0, :, 0], u[0, :, 0], lw=3, label='4 km AGL', color=colors[0])
    plt.plot(1e3*d[20, :, 0], u[20, :, 0], lw=3, label='3 km AGL', color=colors[1])
    plt.plot(1e3*d[40, :, 0], u[40, :, 0], lw=3, label='2 km AGL', color=colors[2])
    plt.plot(1e3*d[60, :, 0], u[60, :, 0], lw=3, label='1 km AGL', color=colors[3])
    plt.plot(1e3*d[80, :, 0], u[80, :, 0], lw=3, label='Surface', color=colors[4])
    plt.ylim(0, 30)
    plt.xlim(0, 20)
    plt.legend(loc = 1, fontsize=12)
    plt.ylabel('Fall Speed [m s$^{-1}$]')
    plt.xlabel('Equivolume Diameter [mm]')
    plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/example_d_vs_u_newAR.png', dpi=300, bbox_inches='tight')

    
    # Plot examining where along DSD various forcing terms come from
    t_idx = 1598


    plt.savefig('/Users/jacob.carlin/Documents/Data/1D Downburst Model/Figures/variable_contribution_by_size_new.png', dpi=300, bbox_inches='tight')
    
