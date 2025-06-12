<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!--
[![Issues][issues-shield]][issues-url]
-->

<!-- PROJECT LOGO -->
<!-- 
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
-->

<h2 align="center">1D Downburst Model</h2>
  <p align="center">
    <br />
    
  </p>

</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-model">About The Model</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#running">Running</a></li>
        <li><a href="#usage-example">Usage Example</a></li>
      </ul>
    </li>
    <li><a href="#issue-reporting">Issue Reporting</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#attribution">Attribution</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE MODEL -->
## About The Model

This model is a Python-based one-dimensional model of downburst development. The physical model is based on the seminal 1D downburst modeling study of [Srivastava (1987)](https://journals.ametsoc.org/view/journals/atsc/44/13/1520-0469_1987_044_1752_amoidd_2_0_co_2.xml). The microphysical processes of graupel and hail melting are based on [Rasmussen and Heymsfield (1987)](https://doi.org/10.1175/1520-0469(1987)973044<2754:MASOGA>2.0.CO;2) and [Ryzhkov et al. (2013)](https://doi.org/10.1175/JAMC-D-13-073.1), while drop shedding is based on [Jost et al. (2019)](https://meetingorganizer.copernicus.org/EGU2019/EGU2019-5125.pdf) and drop breakup is based on [Ryzhkov et al. (2013)](https://doi.org/10.1175/JAMC-D-13-073.1). The coupled polarimetric radar forward operator is based on [Ryzhkov et al. (2011)](https://journals.ametsoc.org/view/journals/apme/50/4/2010jamc2363.1.xml) with modifications based on [Dawson et al. (2014)](https://doi.org/10.1175/JAS-D-13-0118.1), [Kumjian et al. (2018)](https://doi.org/10.1175/JAMC-D-17-0362.1), [Theis et al. (2022)](https://doi.org/10.1175/JAS-D-21-0162.1), and [Lin et al. (2024)](https://doi.org/10.1175/JAS-D-23-0231.1). 

For a detailed physical description of the model and operator, please see Carlin and Ryzhkov (2025). 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To download a copy of the code and run the model locally, follow the below instructions.

### Prerequisites

The following packages are required:
```
arm_pyart==1.18.3
Cartopy==0.22.0
matplotlib==3.8.4
MetPy==1.6.2
netCDF4==1.6.5
numpy==2.3.0
pandas==2.3.0
pygrib==2.1.5
Requests==2.32.4
scipy==1.15.3
setuptools==69.5.1
siphon==0.9
xarray==2023.6.0
```

For ease, the use of Anaconda with the included conda environment is recommended.

### Installation

1. Clone the repo
   ```
   git clone https://github.com/jacob-carlin/downburst_1d.git
   ```
2. If using Anaconda, navigate to the repository director and load the conda environment
   ```
   conda env create -f ./environment.yml
   ```
3. Activate the conda environment
   ```
   conda activate 1d_downburst_model
   ```
4. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin jacob-carlin/downburst_1d
   git remote -v # confirm the changes
   ```

Note: Options exist in the namelist for using 2-layer T-matrix scattering calculations from a look-up table (LUT), which is the default recommended process. This file is very large (>200 MB) and is unable to be hosted on Github. If T-matrix (i.e., non-Rayleigh) scattering calculations are desired, please contact the creator. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Running

1. Modify the ``scripts/mh_namelist.py`` file with the requested options and parameters (detailed in the mh_namelist) and save the file.

2. Open a Terminal, navigate to the directory containing the repository, and run
   ```
   python ./scripts/downburst_model_cleaned.py
   ```

3. If ``write_netcdf == True`` in ``mh_namelist.py``, the model output will be saved in ``netcdf_path``.

<!-- USAGE EXAMPLES -->
### Usage Example

To conduct an example simulation, the model can be run with these default settings in mh_namelist. 

```
delt = 2.0                  # Model time step [s]
dh = 200.0                  # Vertical grid spacing [m]
total_t = 500               # Total model-time length [s]
deld = 0.1                  # dD: Particle bin size interval [mm]
init_frozen_opt = True      # True = Hail/Graupel
rg = 600.0                  # Density of pure graupel [kg/m3]
rs_opt = 3                  # Variable ice density with rg from D=0 to D=5 mm then linearly interpolated to ri by D=1 cm
ng0 = 8000                  # Graupel intercept parameter [1/m3/mm]
lamg = 1.4                  # Graupel slope parameter [1/mm]
Fsub = 1.9                  # Graupel sublimation enhancement parameter from Theis et al. (2022). 
nh0 = 1.5                   # Hail intercept parameter [1/m3/mm]
lamh = 0.3                  # Hail slope parameter [mm-1]
dmax_limit = 20             # Maximum hail size [mm]
hail_dist_opt = 5           # Custom hail size distribution
ar_opt = 1                  # Aspect ratio parameterization of Kumjian et al. (2018) + Theis et al. (2022)
sigma_opt = 2               # Melting hail canting angle parameterization of Dawson et al. (2014)
waveflag = 0                # Radar wavelength -- S band (10.97 cm)
sigrain = 10.0              # Standard deviation of rain canting angle distribution [deg]
sighail = 60.0              # Standard deviation of hail canting angle distribution [deg]
ar_g = 0.9                  # Graupel aspect ratio (Theis et al. 2022)
verbose = True              # Detailed print statements
shed_opt = True             # Turn meltwater shedding on/off
shed_dsd_opt = 0            # Shed drop DSD parameterization based on Theis et al. (2021) 
break_opt = True            # Turn drop breakup on/off
evap_opt = True             # Turn evaporation on/off
subl_opt = True             # Turn sublimation on/off
radar_opt = True            # Turn radar variable calculation on/off
generate_lut = False        # Flag to generate LUT (only applies if radar_opt = True) -- DEFAULT FALSE
use_lut = True              # Flag to use LUT (only applies if radar_opt = True) -- DEFAULT TRUE
use_2layer = True           # Flag to use 2-layer LUT (only applies if radar_opt = True) -- DEFAULT TRUE
lut_path = '../data/scattering_lut_sband.nc'
twolayer_lut_path = '../data/Tmatrix_2layer_LUT_variableT.nc'
write_netcdf = True
netcdf_path = '../results/control.nc'   
profile_opt = 0             # Idealized environment
h0 = 4000                   # Model top [m]
t_top = 0.0                 # Temperature at model top [C]
gam = 9.0                   # Temperature lapse rate [C/km]
rh_top = 50.0               # Relative humidity at model top [%]
gam_rh = 0.0                # Relative humidity lapse rate [%/km] 
```

```
python ./scripts/downdraft_model_cleaned.py
```

```
Working directory:  /Users/jacob.carlin/Documents/Data/downburst_1d/

## You are using the Python ARM Radar Toolkit (Py-ART), an open source
## library for working with weather radar data. Py-ART is partly
## supported by the U.S. Department of Energy as part of the Atmospheric
## Radiation Measurement (ARM) Climate Research Facility, an Office of
## Science user facility.
##
## If you use this software to prepare a publication, please cite:
##
##     JJ Helmus and SM Collis, JORS 2016, doi: 10.5334/jors.119

Graupel Lamda:  1.4 mm-1
Graupel Intercept:  8000 m-3 mm-1
Hail Lamda:  0.3 mm-1
Hail Intercept:  1.5 m-3 mm-1
Maximum Hail Size:  20 mm
Variable-density ice
Environment: Idealized
T0:  273.15 C
Temperature lapse rate:  9.0 C/km
RH0:  50.0 %
RH lapse rate: 0.0 %/km
Initial air density: 0.8047105807842989
Time:  0.0 s |  Minimum w:  -1.0 m/s
Initial IWC:  4.380679167502768  g/m3
Time:  2.0 s |  Minimum w:  -1.0 m/s
Time:  4.0 s |  Minimum w:  -1.0 m/s
Time:  6.0 s |  Minimum w:  -1.0 m/s
Time:  8.0 s |  Minimum w:  -1.0 m/s
Time:  10.0 s |  Minimum w:  -1.0 m/s
Time:  12.0 s |  Minimum w:  -1.0 m/s
Time:  14.0 s |  Minimum w:  -1.0 m/s
Time:  16.0 s |  Minimum w:  -1.0 m/s
Time:  18.0 s |  Minimum w:  -1.0 m/s
.
.
.
Calculating radar variables for tstp  246
Calculating radar variables for tstp  247
Calculating radar variables for tstp  248
Total runtime:  0:01:37.066282
/Users/jacob.carlin/Documents/Data/downburst_1d/scripts/downdraft_model_cleaned.py:2329: ComplexWarning: Casting complex values to real discards the imaginary part
  nc_deli[:, :, :] = deli[:, :, :]
netCDF output file written!
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
<!--
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->

## Issue Reporting

If you notice any bugs in the code or unexpected behavior, please report them by filing a report <a href="https://github.com/jacob-carlin/downburst_1d/issues/new?labels=bug&template=bug-report---.md">here</a>.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

While _active_ development of this code base by the lead developer is expected to cease in the summer of 2025, further contributions are always welcome (especially as our physical understanding of microphysics and downburst development evolves!) and are **greatly appreciated**.

If you have a code suggestion that would further improve this model, please either submit an issue report with the tag "enhancement" or, alternatively, fork the repo and create a pull request:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Attribution

If you use this code in any academic or scientific context, we kindly request attribution be given by citing Carlin and Ryzhkov (2025).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Jacob Carlin

Email: jacob.carlin@noaa.gov

[![LinkedIn][linkedin-shield]][linkedin-url]

Project Link: [https://github.com/jacob-carlin/downburst_1d](https://github.com/jacob-carlin/downburst_1d)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Support for this work was provided by National Science Foundation (NSF) Grant [#AGS2110709](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2110709&HistoricalAwards=false) and the NOAA/Office of Oceanic and Atmospheric Research under NOAA-University of Oklahoma Cooperative Agreement [#NA21OAR4320204](https://www.highergov.com/grant/NA21OAR4320204/).

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/jacob-carlin-4b205862/
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
