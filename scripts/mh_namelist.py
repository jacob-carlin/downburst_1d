# -*- coding: utf-8 -*-
"""
Author: Jacob T. Carlin
Contact: jacob.carlin@noaa.gov 

This file contains the input run parameters used in the one-dimensional
downdraft model.
"""

# Import necessary packages
import numpy as np
from datetime import datetime

###############################################################################
# Model parameters
###############################################################################
delt = 0.5                   # Model time step [s]
dh = 50.0                    # Vertical grid spacing [m]
total_t = 700                # Total model-time length [s]
ntstp = int(total_t / delt)  # Number of timestamps

###############################################################################
# Particle size distributions
###############################################################################
deld = 0.1                  # dD: Particle bin size interval [mm]

# Select whether to initialize model with ice or rain
init_frozen_opt = True      # True = Hail/Graupel
                            # False = Rain
dsd_norm = False            # True = Normalized gamma distribution used (for init_frozen_opt = False)
                            # False = Regular gamma distribution used (for init_frozen_opt = False)

# Rain parameters for init_frozen_opt = False
nr0 = 8000                  # Intercept parameter [1/m3/(mm+u)]
lamr = 2.7                  # Slope parameter [1/mm]
mur = 3                     # Shape parameter
nrw = 10**5                 # Normalized intercept parameter (for normalized gamma dist)
dmr = 1.50                  # Mass-weighted mean diameter (for normalized gamma dist) [mm]

# Ice parameters for init_frozen_opt = True
rg = 600.0                  # Density of pure graupel [kg/m3]


# Select whether density should be constant or vary across the spectrum
# 0: Solid ice
# 1: Variable density polynomial parameterization from Ryzhkov et al. (2013)
# 2: Variable density linear parameterization from rg at D=0 to ri at D=1 cm
# 3: Variable density with rg from D=0 to D=5 mm then linearly interpolated to ri by D=1 cm (recommended)
rs_opt = 3

# Graupel parameters
ng0 = 8000                  # Graupel intercept parameter [1/m3/mm]
lamg = 1.4                  # Graupel slope parameter [1/mm]
Fsub = 1.9                  # Graupel sublimation enhancement parameter from Theis et al. (2022). 
                            # They suggest 1.9 as a representative value for 0.4 kg/m3 graupel between 3 and 5 mm
                            # but this remains highly uncertain!

# Custom hail parameters (for hail_dist_opt == 5)
nh0 = 1.5                   # Hail intercept parameter [1/m3/mm]
lamh = 0.3                  # Hail slope parameter [mm-1]
dmax_limit = 20             # Maximum hail size [mm]

# Preset hail size distributions (see Ryzhkov et al. 2013a)
# 0: None
# 1: Small
# 2: Medium
# 3: Large
# 4: Giant
# 5: Custom (use above nh0 and lamh)
hail_dist_opt = 5

if hail_dist_opt == 0:
    lamh = 0
    nh0 = 0
    dmax_limit = 8.0
elif hail_dist_opt == 1:
    lamh = 0.99
    nh0 = 200*lamh**4.11
    dmax_limit = 20.0
elif hail_dist_opt == 2:
    lamh = 0.42
    nh0 = 400*lamh**4.11
    dmax_limit = 20.0 #20
elif hail_dist_opt == 3:
    lamh = 0.27
    nh0 = 800*lamh**4.11
    dmax_limit = 35.0
elif hail_dist_opt == 4:
    lamh = 0.19
    nh0 = 800*lamh**4.11
    dmax_limit = 20.0
else:
    lamh = lamh
    nh0 = nh0
    dmax_limit = dmax_limit

if init_frozen_opt == False:
    dmax_limit = 8.0            # Limit raindrops to no larger than 8 mm
nbin = int(dmax_limit / deld)   # Total number of particle size bins

# Option for calculation aspect ratio of *melting* hail/graupel
# 0 --> Ryzhkov et al. (2011) parameterization
# 1 --> Kumjian et al. (2018) + Theis et al. (2022) parameterization (recommended)
ar_opt = 1

# Option for calculation sigma of melting hail/graupel
# 0 --> Ryzhkov et al. (2011) parameterization
# 1 --> Kumjian et al. (2018) parameterization
# 2 --> Dawson et al. (2014) parameterization (recommended)
sigma_opt = 2
    
###############################################################################
# Electromagnetic calculations
###############################################################################

# Select waveflag
# 0: S-band (10.97 cm)
# 1: C-band (5.45 cm)
# 2: X-band (3.2 cm)
waveflag = 0

# Note: For use with PyTMatrix package, both the real and imaginary parts of the 
# dielectric constant should be positive (i.e., complex(81.0, 23.2)).
if waveflag == 0:
    wave = 109.7  # Radar wavelength [mm]
    lamda = 10.97  # Radar wavelength [cm]
    ew0 = complex(81.0, 23.2)  # Dielectric constant of water at 0C
elif waveflag == 1:
    wave = 54.5
    lamda = 5.45
    ew0 = complex(65.2, 37.2)
elif waveflag == 2:
    wave = 32.0
    lamda = 3.2
    ew0 = complex(44.5, 41.4)
ei = complex(3.18, 8.54e-3)  # Dielectric constant of ice
ea = complex(1.0006, 5.0E-7)  # Dielectric constant of air
eps_0 = 8.854187e-12 # Permittivity of free space (F/m)

# Canting angle distribution standard deviations
sigrain = 10.0              # Standard deviation of rain canting angle distribution [deg]
sighail = 60.0              # Standard deviation of hail canting angle distribution [deg]
                            # Ryzhkov (2011), Borowska (2011) - 40 | Dawson (2014) - 60

# Graupel aspect ratio
ar_g = 0.9                  # Graupel aspect ratio (Theis et al. 2022)

###############################################################################
# Process flags
###############################################################################
# True = On, False = Off
verbose = True              # Detailed print statements
shed_opt = True             # Turn meltwater shedding on/off
shed_dsd_opt = 0            # 0 = New parameterization based on Theis et al. (2021) (recommended)
                            # 1 = Original parameterization from Ryzhkov et al. (2013) with mu = 2, lam = 2
                            #     Note: mu = 3 and lam = 5 also proposed in literature
break_opt = True            # Turn drop breakup on/off
evap_opt = True             # Turn evaporation on/off
subl_opt = True             # Turn sublimation on/off
radar_opt = True           # Turn radar variable calculation on/off
generate_lut = False        # Flag to generate LUT (only applies if radar_opt = True) -- DEFAULT FALSE
use_lut = True              # Flag to use LUT (only applies if radar_opt = True) -- DEFAULT TRUE
use_2layer = True           # Flag to use 2-layer LUT (only applies if radar_opt = True) -- DEFAULT TRUE
lut_path = '../data/scattering_lut_sband.nc'
twolayer_lut_path = '../data/Tmatrix_2layer_LUT_variableT.nc'
#tp_eq_adj_opt = True       # Adjustment to particle temperature due to nonequilibrium
                            # owing to terminal velocity and small grid size (e.g.,
                            # Tardis and Rasmussen 2010)
write_netcdf = True
netcdf_path = '../results/control.nc'    
make_plots = False
save_plots = False

###############################################################################
# Initial environment for idealized runs
###############################################################################
# Input sounding
# 0: Idealized (see subsequent parameters)
# 1: Observed (use sounding_path)
# 2: Weisman-Klemp with qv0 = 12 g/kg and T0 = 300 K
# 3: RAP sounding for requested lat/lon
# 4: HRRR sounding for requested lat/lon
profile_opt = 0
h0 = 4000                   # Model top [m]
t_top = 0.0                 # Temperature at model top [C]
gam = 9.0                   # Temperature lapse rate [C/km]
rh_top = 50.0               # Relative humidity at model top [%]
gam_rh = 0.0                # Relative humidity lapse rate [%/km]
use_mixing_ratio = False    # Flag to use constant mixing ratio to emulate well-mixed PBL
qv_top = 0.010              # PBL Mixing Ratio [kg/kg] (if use_mixing_ratio = True)
                            # NOTE: qv_top must be chosen carefully to not result in deep layer 
                            # of RH > 100% depending in temperature profile
                            # For h0 = 4000, t_top = 0.0, and gam = 7.0, 6 g/kg is a good value for well-mixed, High-Plains downburst
sounding_path = ' ' # Copied text file from U. of Wyoming archive for profile_opt = 1
sounding_time = datetime(2017, 7, 27, 22, 0) # Datetime object for requested HRRR sounding (for profile_opt = 3, 4) in UTC YYYY, MM, DD, HH, MM
sounding_lat = 35.48        # Requested HRRR latitude (for profile_opt = 3, 4)
sounding_lon = -97.70       # Requested HRRR longitude (for profile_opt = 3, 4)
sounding_alt = 396          # m
mix_coef = 0.0001           # Mixing/entrainment coefficient (= 0.2 / radius) [1/m]
                            # Default = 0.0001 (2-km radius downdraft)
                            # Set to = 0 to turn off mixing

###############################################################################
# Other constants
###############################################################################
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
