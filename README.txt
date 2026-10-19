================================================================================
  Broadband Antireflection in Ag-Assisted MACE Porous GaAs:
  Etching-Time Dependence and Effective-Medium Optical Modeling
================================================================================

Authors : S. Daboussi et al.
Journal : Royal Society Open Science 
Manuscript ID : RSOS-262506

--------------------------------------------------------------------------------
OVERVIEW
--------------------------------------------------------------------------------

This repository contains all experimental data, optical constants, and Python
simulation code supporting the results reported in the manuscript. The optical
model combines the Transfer Matrix Method (TMM) with the Bruggeman Effective
Medium Approximation (EMA) to fit UV-Vis reflectance spectra of porous GaAs
layers produced by Ag-assisted Metal-Assisted Chemical Etching (MACE) at five
etching durations (non-etched, 15, 25, 35, 55 min).

--------------------------------------------------------------------------------
REPOSITORY STRUCTURE
--------------------------------------------------------------------------------

  Data_results_RSOS-262506/
  |
  |-- README.txt                          This file
  |
  |-- Experimental_Data_results/
  |     |-- UV-Vis/                       UV-Vis reflectance and Kubelka Munk
  |     |     |-- R_bulk.Sample.csv
  |     |     |-- R 15.Sample.csv
  |     |     |-- R 25-V0.Sample.csv
  |     |     |-- R 35.Sample.csv
  |     |     |-- R 55.Sample.csv
  |     |     `-- Reflectance_ KubelkaMunk_all_samples.csv
  |     |
  |     |-- FTIR/                         FTIR spectra
  |     |     |-- FTIR_GaAs_non-etched.dpt
  |     |     |-- FTIR_GaAs_15min.dpt
  |     |     |-- FTIR_GaAs_25min.dpt
  |     |     |-- FTIR_GaAs_35min.dpt
  |     |     |-- FTIR_GaAs_55min.dpt
  |     |     `-- FTIR_all_samples.csv
  |     |
  |     |-- DRX/                          XRD patterns (Bruker RAW4 format)
  |     |     |-- DRX_GaAs_non-etched.txt
  |     |     |-- DRX_GaAs_15min.txt
  |     |     |-- DRX_GaAs_25min.txt
  |     |     |-- DRX_GaAs_35min.txt
  |     |     `-- DRX_GaAs_55min.txt
  |     |
  |     `-- MEB/                          SEM images
  |           |-- SEM_GaAs_25min.jpg
  |           |-- SEM_GaAs_25min_Cross-section.jpg
  |           `-- SEM_GaAs_35min.jpg
  |
  `-- Simulation_Data_code_results/
        |-- simulation_GaAs_non_etched.py
        |-- simulation_GaAs_15min.py
        |-- simulation_GaAs_25min.py
        |-- simulation_GaAs_35min.py
        |-- simulation_GaAs_55min.py
        |-- Gaas041t.nk
        |-- GaasOx.nk
        |-- Void.nk
        |-- result_simulation_non-etched_GaAs.txt
        |-- result_simulation_GaAs_15min.txt
        |-- result_simulation_GaAs_25min.txt
        |-- result_simulation_GaAs_35min.txt
        `-- result_simulation_GaAs_55min.txt

--------------------------------------------------------------------------------
EXPERIMENTAL DATA (Experimental_Data_results/)
--------------------------------------------------------------------------------

UV-Vis reflectance spectra  (UV-Vis/)
---------------------------------------
Semicolon-delimited files exported from the spectrophotometer software.
Decimal separator: comma. Wavelength range: 200-1200 nm.

  Column   Unit   Description
  ------   ----   ----------------------------
  nm       nm     Wavelength
  %R       %      Specular reflectance

Files:
  R_bulk.Sample.csv      Reference GaAs (non-etched)
  R 15.Sample.csv        Porous GaAs, 15 min MACE
  R 25-V0.Sample.csv     Porous GaAs, 25 min MACE
  R 35.Sample.csv        Porous GaAs, 35 min MACE
  R 55.Sample.csv        Porous GaAs, 55 min MACE

Fitting range used in the simulations: 277-800 nm.
Note: each file contains a one-line header (skipped automatically by the scripts).


Reflectance  KubelkaMunk  (UV-Vis/)
--------------------------------------
Diffuse reflectance spectra processed via the  KubelkaMunk function (all
samples compiled in a single CSV file).

File:
  Reflectance_ KubelkaMunk_all_samples.csv


FTIR spectra  (FTIR/)
-----------------------
Raw spectra in OPUS/Bruker binary format (.dpt) and compiled CSV.

Raw files (one per sample):
  FTIR_GaAs_non-etched.dpt
  FTIR_GaAs_15min.dpt
  FTIR_GaAs_25min.dpt
  FTIR_GaAs_35min.dpt
  FTIR_GaAs_55min.dpt

Compiled file (all samples):
  FTIR_all_samples.csv

  Column        Unit    Description
  ----------    -----   ----------------------------------------
  Wavenumber    cm-1    Wavenumber
  Signal        a.u.    Absorbance or transmittance


XRD patterns  (DRX/)
----------------------
Files are in Bruker RAW4 format (binary-text hybrid). The first line is the
format identifier:

  ;RAW4.00

Followed by metadata sections ([RawHeader], [VarInfo], [Drive], [PSD], etc.)
containing instrument configuration and acquisition parameters. The measured
data begin at the section marker:

  [Data]
  Angle, PSD,

Each subsequent line is a comma-separated pair: diffraction angle (2theta,
degrees) and intensity (PSD counts). Files can be read with Bruker EVA/DIFFRAC
software or parsed in Python/MATLAB by skipping all lines before [Data].

Files:
  DRX_GaAs_non-etched.txt
  DRX_GaAs_15min.txt
  DRX_GaAs_25min.txt
  DRX_GaAs_35min.txt
  DRX_GaAs_55min.txt


SEM images  (MEB/)
--------------------
Cross-sectional and plan-view images in JPEG format.
Scale bars are embedded in each image.

Files:
  SEM_GaAs_25min.jpg               Plan-view, 25 min MACE
  SEM_GaAs_25min_Cross-section.jpg Cross-section, 25 min MACE
  SEM_GaAs_35min.jpg               Plan-view, 35 min MACE

--------------------------------------------------------------------------------
SIMULATION CODE (Simulation_Data_code_results/)
--------------------------------------------------------------------------------

Optical constant files (.nk)
------------------------------
Tabulated complex refractive index in WinElli format.
Each file: one header line, then three columns:

  Column   Unit   Description
  ------   ----   -------------------------------------------
  E        eV     Photon energy
  n        -      Real part of refractive index
  k        -      Imaginary part (extinction coefficient)

Files:
  Gaas041t.nk   Bulk GaAs (Adachi model); E range 1.24-4.495 eV, 219 points
  GaasOx.nk     GaAs native oxide;        E range 1.6-6.0 eV,   44 points
  Void.nk       Air/void (n=1, k=0);      E range 0.01-200 eV,  70 points

Note: simulation_GaAs_15min.py references GaasOx.nk with lowercase 'ox'
(Gaasox.nk). On Windows the file system is case-insensitive; the file
GaasOx.nk is used in both cases.


Python simulation scripts
--------------------------
Five independent scripts, one per sample. Common features:
  - Optimization: Differential Evolution (DE) followed by L-BFGS-B
  - Model selection: RMSE, AIC, BIC
  - Output: two PNG figures + one numerical results file (.txt)

  simulation_GaAs_non_etched.py
    Structure        : Air / GaAsOx (free thickness) / Bulk GaAs
    Free parameters  : 1 (oxide thickness, 0.5-10 nm)
    Input NK files   : Gaas041t.nk, GaasOx.nk
    Input data       : R_bulk.Sample.csv
    Output directory : results_bulk_gaasox/

  simulation_GaAs_15min.py
    Structure        : Air / GaAsOx (free) / N porous GaAs sublayers / Bulk GaAs
    N tested         : 1 to 8 (BIC selection)
    Free parameters  : 2N + 1 (N thicknesses, N void fractions, oxide thickness)
    UV weighting     : factor 4 applied below 430 nm
    Input NK files   : Gaas041t.nk, GaasOx.nk
    Input data       : R 15.Sample.csv
    Output directory : results_15min/

  simulation_GaAs_25min.py
    Structure        : Air / N porous GaAs sublayers / Bulk GaAs
    N tested         : 2 to 7 (BIC selection)
    Free parameters  : 2N (N thicknesses, N void fractions)
    Warm start       : initialised from 35 min N=6 optimal parameters
    Input NK files   : Gaas041t.nk, Void.nk
    Input data       : R 25-V0.Sample.csv
    Output directory : results_english_porosity_neff/25min/

  simulation_GaAs_35min.py
    Structure        : Air / N porous GaAs sublayers / Bulk GaAs
    N tested         : 2 to 7 (BIC selection)
    Free parameters  : 2N
    Input NK files   : Gaas041t.nk, Void.nk
    Input data       : R 35.Sample.csv
    Output directory : results_english_porosity_neff/35min/

  simulation_GaAs_55min.py
    Structure        : Air / N porous GaAs sublayers / Bulk GaAs
    N tested         : 2 to 8 (BIC selection)
    Free parameters  : 2N
    Thickness bounds : 5-300 nm (wider than 15/25/35 min scripts: 5-150 nm)
    Warm starts      : N=3 to 8 (biased toward thick/porous solutions)
    Input NK files   : Gaas041t.nk, Void.nk
    Input data       : R 55.Sample.csv
    Output directory : results_english_porosity_neff/55min/


Dependencies
-------------
  Package      Tested version   Purpose
  ----------   --------------   ------------------------------------
  Python       3.10+            -
  NumPy        1.24+            Numerical arrays
  SciPy        1.10+            Optimization (DE, L-BFGS-B), interpolation
  Matplotlib   3.7+             Figures

Install:
  pip install numpy scipy matplotlib


How to run
-----------
Copy all files from Simulation_Data_code_results/ into the same working
directory (scripts, .nk files, and the relevant input .csv from UV-Vis/),
then:

  python simulation_GaAs_25min.py

Each script produces three output files in a subdirectory of the working
directory (see "Output directory" above):
  figure1_fit_<label>.png        Experimental vs. fitted reflectance curve
  figure2_analysis_<label>.png   RMSE/AIC/BIC vs N, porosity profile, n_eff/k_eff
  result_simulation_<label>.txt  Full numerical results (see below)

--------------------------------------------------------------------------------
NUMERICAL RESULTS (Simulation_Data_code_results/)
--------------------------------------------------------------------------------

Each result_simulation_<label>.txt file contains:
  - Model comparison table: N layers, RMSE, AIC, BIC, average porosity, n_eff
  - Best-model parameters: layer thickness d (nm), void fraction, porosity (%),
    n_eff and k_eff at 550 nm
  - Full simulated and experimental spectrum (wavelength nm, R_fit %, R_exp %)

Files:
  result_simulation_non-etched_GaAs.txt
  result_simulation_GaAs_15min.txt
  result_simulation_GaAs_25min.txt
  result_simulation_GaAs_35min.txt
  result_simulation_GaAs_55min.txt

--------------------------------------------------------------------------------
CITATION
--------------------------------------------------------------------------------

If you use these data or code, please cite:

  Daboussi A. et al. (2026). Data and code for: Broadband Antireflection in
  Ag-Assisted MACE Porous GaAs: Etching-Time Dependence and Effective-Medium
  Optical Modeling. Zenodo. https://doi.org/10.5281/zenodo.23477775

--------------------------------------------------------------------------------
LICENSE
--------------------------------------------------------------------------------

All data and code in this repository are released under the
Creative Commons Attribution 4.0 International (CC-BY 4.0) licence.
You are free to share and adapt the material for any purpose,
provided appropriate credit is given.

================================================================================

