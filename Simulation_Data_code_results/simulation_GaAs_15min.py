"""
================================================================================
  15 min MACA GaAs REFLECTANCE FITTING  [fit_15min_3.py]  — v4
  -----------------------------------------------------------------------
  TMM + EMA Bruggeman
  Structure : Air / GaAsOx(d_ox libre) / [C1 / ... / CN] / GaAs
  N couches poreuses independantes, N selectionne par BIC

  Ameliorations vs v3 pour le fit UV :
    1. d_ox LIBRE (2 nm - 40 nm) : physiquement justifie — a 15 min
       l'oxyde peut etre plus epais que la valeur bulk (5.168 nm)
       et il absorbe fortement en UV (k>0 dans GaAsox.nk des ~590 nm).
    2. RMSE pondere : poids UV_WEIGHT dans [FIT_MIN, UV_SPLIT_NM]
       pour forcer un meilleur fit UV sans degrader le visible.
    3. Rapport detaille : RMSE_UV et RMSE_vis separes + graphique annote.

  Optimiseur rapide : DE maxiter=300, popsize=15
================================================================================
"""

import os
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

NK_GAAS  = "Gaas041t.nk"
NK_OXIDE = "Gaasox.nk"
CSV_15   = "R 15.Sample.csv"

FIT_MIN  = 277.0
FIT_MAX  = 800.0

# UV / visible boundary
UV_SPLIT_NM = 430.0     # below: UV ; above: visible
UV_WEIGHT   = 4.0       # extra weight on UV region (increased from 2.5)

# Reference wavelength for effective-index interpretation
NEFF_REFERENCE_WL = 550.0  # nm

# Free oxide layer
D_OX_MIN = 2.0
D_OX_MAX = 40.0

# Porous GaAs sublayers
D_LAYER_MIN =   2.0
D_LAYER_MAX = 200.0
F_LAYER_MIN =   0.01
F_LAYER_MAX =   0.95

N_MIN = 1
N_MAX = 8               # extended from 6

RESULTS_DIR   = "results_15min"
OUTPUT_PNG1   = os.path.join(RESULTS_DIR, "figure1_fit_15min.png")
OUTPUT_PNG2   = os.path.join(RESULTS_DIR, "figure2_analysis_15min.png")
OUTPUT_TXT    = os.path.join(RESULTS_DIR, "results_15min.txt")

# =============================================================================
# FILE I/O
# =============================================================================

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def load_nk_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NK file not found: {filepath}")
    with open(filepath, encoding="latin-1") as f:
        lines = f.readlines()
    header = lines[0].split()
    E_min, E_max = float(header[1]), float(header[2])
    nk_data = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                nk_data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    nk_data  = np.array(nk_data)
    energies = np.linspace(E_min, E_max, len(nk_data))
    return energies, nk_data[:, 0], nk_data[:, 1]

def load_csv_reflectance(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    wavelengths, reflectances = [], []
    with open(filepath, encoding="latin-1") as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.strip().split(";")
        if len(parts) >= 2:
            try:
                wl = float(parts[0].replace(",", ".").strip())
                r  = float(parts[1].replace(",", ".").strip()) / 100.0
                wavelengths.append(wl)
                reflectances.append(r)
            except ValueError:
                continue
    wl  = np.array(wavelengths)
    R   = np.array(reflectances)
    idx = np.argsort(wl)
    return wl[idx], R[idx]

# =============================================================================
# TMM vectorise — toutes les longueurs d'onde en parallele (numpy)
# =============================================================================

def tmm_reflectance(wavelengths_nm, layers):
    """TMM entierement vectorise : pas de boucle Python sur les longueurs d'onde."""
    lam = wavelengths_nm                        # (N_wl,)
    N_wl = len(lam)

    # Matrice de transfert identite pour chaque longueur d'onde
    M = np.zeros((N_wl, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0

    for j in range(1, len(layers) - 1):
        nj    = layers[j]["n"] + 1j * layers[j]["k"]   # (N_wl,)
        delta = 2.0 * np.pi * nj * layers[j]["d"] / lam
        cd    = np.cos(delta)                           # (N_wl,)
        sd    = np.sin(delta)

        Mj = np.empty((N_wl, 2, 2), dtype=complex)
        Mj[:, 0, 0] =  cd
        Mj[:, 0, 1] = -1j * sd / nj
        Mj[:, 1, 0] = -1j * nj * sd
        Mj[:, 1, 1] =  cd

        # M = M @ Mj  pour chaque lambda simultanement
        M = np.einsum("nij,njk->nik", M, Mj)

    n0  = layers[0]["n"]  + 1j * layers[0]["k"]
    ns  = layers[-1]["n"] + 1j * layers[-1]["k"]
    m11, m12 = M[:, 0, 0], M[:, 0, 1]
    m21, m22 = M[:, 1, 0], M[:, 1, 1]
    num = (m11 + m12*ns)*n0 - (m21 + m22*ns)
    den = (m11 + m12*ns)*n0 + (m21 + m22*ns)
    return np.abs(num / den)**2

# =============================================================================
# EMA Bruggeman
# =============================================================================

def bruggeman_ema(n1, k1, n2, k2, f2):
    f2   = float(np.clip(f2, 1e-6, 1.0 - 1e-6))
    f1   = 1.0 - f2
    eps1 = (n1 + 1j*k1)**2
    eps2 = (n2 + 1j*k2)**2
    B    = -(f1*(2*eps1 - eps2) + f2*(2*eps2 - eps1))
    C    = -eps1 * eps2
    sq   = np.sqrt((B**2 - 8.0*C).astype(complex))
    s1, s2 = (-B + sq)/4.0, (-B - sq)/4.0
    eps  = np.where(np.imag(s1) >= 0, s1, s2)
    nc   = np.sqrt(eps)
    nc   = np.where(np.real(nc) < 0, -nc, nc)
    return np.real(nc), np.imag(nc)

# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    def __init__(self):
        print("[1/2] Chargement des constantes optiques...")
        gaas_e,  gaas_n,  gaas_k  = load_nk_file(NK_GAAS)
        oxide_e, oxide_n, oxide_k = load_nk_file(NK_OXIDE)

        print("[2/2] Chargement de la reflectance experimentale 15min...")
        wl_exp, R_exp = load_csv_reflectance(CSV_15)

        lam_gaas  = 1239.84 / gaas_e
        lam_oxide = 1239.84 / oxide_e

        idx_g = np.argsort(lam_gaas)
        lam_gaas = lam_gaas[idx_g]; gaas_n = gaas_n[idx_g]; gaas_k = gaas_k[idx_g]

        idx_o = np.argsort(lam_oxide)
        lam_oxide = lam_oxide[idx_o]; oxide_n = oxide_n[idx_o]; oxide_k = oxide_k[idx_o]

        lam_min = max(wl_exp.min(), lam_gaas.min(), FIT_MIN)
        lam_max = min(wl_exp.max(), lam_gaas.max(), FIT_MAX)
        mask    = (wl_exp >= lam_min) & (wl_exp <= lam_max)

        self.wl    = wl_exp[mask]
        self.R_exp = R_exp[mask]
        self.N     = len(self.wl)

        # Masques UV / visible
        self.mask_uv  = self.wl <  UV_SPLIT_NM
        self.mask_vis = self.wl >= UV_SPLIT_NM

        # Poids spectraux : UV_WEIGHT dans l'UV, 1.0 dans le visible
        self.weights = np.where(self.mask_uv, UV_WEIGHT, 1.0)

        print(f"  Plage de fit : {self.wl.min():.1f} - {self.wl.max():.1f} nm")
        print(f"  UV  (<{UV_SPLIT_NM:.0f} nm) : {self.mask_uv.sum()} pts  (poids x{UV_WEIGHT})")
        print(f"  Vis (>={UV_SPLIT_NM:.0f} nm) : {self.mask_vis.sum()} pts  (poids x1)")

        interp = lambda lam, y: interp1d(
            lam, y, kind="linear", fill_value="extrapolate")(self.wl)

        self.n_gaas  = interp(lam_gaas,  gaas_n)
        self.k_gaas  = interp(lam_gaas,  gaas_k)
        self.n_oxide = interp(lam_oxide, oxide_n)
        self.k_oxide = interp(lam_oxide, oxide_k)
        self.n_air   = np.ones(self.N)
        self.k_air   = np.zeros(self.N)
        self.n_void  = np.ones(self.N)
        self.k_void  = np.zeros(self.N)

# =============================================================================
# METRIQUES
# =============================================================================

def rmse_total(R_fit, R_exp):
    return float(np.sqrt(np.mean((R_fit - R_exp)**2)))

def rmse_weighted(R_fit, R_exp, weights):
    """RMSE pondere : force un meilleur fit dans les regions a fort poids."""
    return float(np.sqrt(np.average((R_fit - R_exp)**2, weights=weights)))

def rmse_region(R_fit, R_exp, mask):
    if mask.sum() == 0:
        return 0.0
    return float(np.sqrt(np.mean((R_fit[mask] - R_exp[mask])**2)))

def r2_fn(R_fit, R_exp):
    ss_res = np.sum((R_fit - R_exp)**2)
    ss_tot = np.sum((R_exp - R_exp.mean())**2)
    return float(1.0 - ss_res / max(ss_tot, 1e-30))

def bic_fn(R_fit, R_exp, k):
    n  = len(R_exp)
    ss = max(float(np.sum((R_fit - R_exp)**2)), 1e-30)
    return k * np.log(n) + n * np.log(ss / n)

def aic_fn(R_fit, R_exp, k):
    n  = len(R_exp)
    ss = max(float(np.sum((R_fit - R_exp)**2)), 1e-30)
    return 2*k + n * np.log(ss / n)

# =============================================================================
# MODELE N COUCHES
# Air / GaAsOx(d_ox libre) / [C1 / ... / CN] / GaAs
# params = [d_ox, d1, f1, d2, f2, ..., dN, fN]   (2N+1 params libres)
# =============================================================================

class ModelN:
    def __init__(self, data, N):
        self.d        = data
        self.N        = N
        self.n_params = 2 * N + 1   # +1 pour d_ox libre
        self.name     = f"N={N} couches poreuses"
        self.label    = f"N{N}"

    def _unpack(self, p):
        d_ox = float(np.clip(p[0], D_OX_MIN, D_OX_MAX))
        layers_p = []
        for i in range(self.N):
            d_i = float(np.clip(p[1 + 2*i],     D_LAYER_MIN, D_LAYER_MAX))
            f_i = float(np.clip(p[1 + 2*i + 1], F_LAYER_MIN, F_LAYER_MAX))
            layers_p.append((d_i, f_i))
        return d_ox, layers_p

    def simulate(self, p):
        d_ox, layers_p = self._unpack(p)
        layers = [
            {"n": self.d.n_air,   "k": self.d.k_air,   "d": 0.0},
            {"n": self.d.n_oxide, "k": self.d.k_oxide, "d": d_ox},
        ]
        for d_i, f_i in layers_p:
            ne, ke = bruggeman_ema(self.d.n_gaas, self.d.k_gaas,
                                   self.d.n_void,  self.d.k_void, f_i)
            layers.append({"n": ne, "k": ke, "d": d_i})
        layers.append({"n": self.d.n_gaas, "k": self.d.k_gaas, "d": 0.0})
        return tmm_reflectance(self.d.wl, layers)

    def bounds(self):
        return ([(D_OX_MIN, D_OX_MAX)] +
                [(D_LAYER_MIN, D_LAYER_MAX),
                 (F_LAYER_MIN, F_LAYER_MAX)] * self.N)

    def init_guess(self):
        # Physically motivated: thin oxide + porosity gradient high→low
        # Based on 25min/35min results scaled down for 15min (short etching)
        x0 = [10.0]   # d_ox initial (nm)
        f_top, f_bot = 0.65, 0.05
        d_total = 80.0
        for i in range(self.N):
            d_i = d_total / self.N
            f_i = f_top + (f_bot - f_top) * i / max(self.N - 1, 1)
            x0 += [d_i, f_i]
        return x0

    def objective(self, p):
        try:
            R = self.simulate(p)
            return rmse_weighted(R, self.d.R_exp, self.d.weights)
        except:
            return 1e6

    def porosity_metrics(self, layers_p):
        d_arr = np.array([lp[0] for lp in layers_p])
        f_arr = np.array([lp[1] for lp in layers_p])
        d_total = float(np.sum(d_arr))
        avg_por = float(np.sum(d_arr * f_arr) / d_total) if d_total > 0 else float("nan")
        return {
            "d_total":      d_total,
            "f_surface":    float(f_arr[0]),
            "f_bottom":     float(f_arr[-1]),
            "avg_porosity": avg_por,
        }

    def effective_index_metrics(self, layers_p, d_ox):
        idx_ref   = int(np.argmin(np.abs(self.d.wl - NEFF_REFERENCE_WL)))
        wl_actual = float(self.d.wl[idx_ref])

        n_gaas_ref = float(self.d.n_gaas[idx_ref])
        k_gaas_ref = float(self.d.k_gaas[idx_ref])
        n_opt_ar   = float(np.sqrt(n_gaas_ref))   # sqrt(n_air * n_GaAs)

        n_ox_ref = float(self.d.n_oxide[idx_ref])
        k_ox_ref = float(self.d.k_oxide[idx_ref])

        n_eff_arr, k_eff_arr = [], []
        for _, f_i in layers_p:
            n_e, k_e = bruggeman_ema(self.d.n_gaas, self.d.k_gaas,
                                     self.d.n_void,  self.d.k_void, f_i)
            n_eff_arr.append(float(n_e[idx_ref]))
            k_eff_arr.append(float(k_e[idx_ref]))

        n_eff_arr = np.array(n_eff_arr)
        k_eff_arr = np.array(k_eff_arr)
        d_arr     = np.array([lp[0] for lp in layers_p])
        d_total   = float(np.sum(d_arr))
        n_eff_avg = float(np.sum(d_arr * n_eff_arr) / d_total) if d_total > 0 else float("nan")
        k_eff_avg = float(np.sum(d_arr * k_eff_arr) / d_total) if d_total > 0 else float("nan")

        return {
            "wl_ref_actual":            wl_actual,
            "n_gaas_ref":               n_gaas_ref,
            "k_gaas_ref":               k_gaas_ref,
            "n_opt_ar":                 n_opt_ar,
            "d_ox":                     d_ox,
            "n_oxide_ref":              n_ox_ref,
            "k_oxide_ref":              k_ox_ref,
            "n_eff_surface":            float(n_eff_arr[0]),
            "k_eff_surface":            float(k_eff_arr[0]),
            "n_eff_bottom":             float(n_eff_arr[-1]),
            "k_eff_bottom":             float(k_eff_arr[-1]),
            "n_eff_avg":                n_eff_avg,
            "k_eff_avg":                k_eff_avg,
            "delta_surface_to_optimum": float(abs(n_eff_arr[0] - n_opt_ar)),
            "delta_avg_to_optimum":     float(abs(n_eff_avg - n_opt_ar)),
            "n_eff_layers":             n_eff_arr,
            "k_eff_layers":             k_eff_arr,
        }

# =============================================================================
# FITTING ENGINE
# =============================================================================

class FittingEngine:
    def __init__(self, model):
        self.model   = model
        self.history = []
        self._iter   = 0

    def _callback_de(self, xk, convergence):
        val = self.model.objective(xk) * 100
        self._iter += 1
        self.history.append((self._iter, val))
        if self._iter % 30 == 0:
            print(f"   DE {self._iter:4d} | RMSEw={val:.4f}%R | conv={convergence:.5f}")

    def run(self):
        bounds = self.model.bounds()
        print(f"\n[Fitting] {self.model.name}  ({self.model.n_params} params libres)")
        t0 = time.time()
        self._iter = 0; self.history = []

        res_de = differential_evolution(
            self.model.objective,
            bounds        = bounds,
            seed          = 42,
            maxiter       = 1500,
            tol           = 1e-12,
            popsize       = 20,
            mutation      = (0.4, 1.8),
            recombination = 0.85,
            polish        = False,
            workers       = -1,
            updating      = "deferred",
            callback      = self._callback_de)

        best_x   = res_de.x
        best_val = res_de.fun

        # Multistart L-BFGS-B — 12 random starts + DE result + init_guess
        rng = np.random.default_rng(0)
        starts = [res_de.x, self.model.init_guess()]
        for _ in range(12):
            starts.append([rng.uniform(lo, hi) for lo, hi in bounds])

        for x0 in starts:
            try:
                res_loc = minimize(
                    self.model.objective, x0=x0,
                    method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 10000, "ftol": 1e-15,
                             "gtol": 1e-11, "disp": False})
                if res_loc.fun < best_val:
                    best_val = res_loc.fun; best_x = res_loc.x
            except Exception:
                pass

        p_opt         = best_x
        R_opt         = self.model.simulate(p_opt)
        d_ox, layers_p = self.model._unpack(p_opt)

        rmse_tot = rmse_total(R_opt, self.model.d.R_exp)
        rmse_uv  = rmse_region(R_opt, self.model.d.R_exp, self.model.d.mask_uv)
        rmse_vis = rmse_region(R_opt, self.model.d.R_exp, self.model.d.mask_vis)
        r2       = r2_fn(R_opt, self.model.d.R_exp)
        bic      = bic_fn(R_opt, self.model.d.R_exp, self.model.n_params)
        aic      = aic_fn(R_opt, self.model.d.R_exp, self.model.n_params)

        print(f"   -> RMSE_tot={rmse_tot*100:.4f}%R | "
              f"RMSE_UV={rmse_uv*100:.4f}%R | "
              f"RMSE_vis={rmse_vis*100:.4f}%R | "
              f"R2={r2:.5f} | BIC={bic:.2f} | t={time.time()-t0:.1f}s")

        return p_opt, R_opt, d_ox, layers_p, rmse_tot, rmse_uv, rmse_vis, r2, aic, bic

# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(data, all_results, best):
    model = best["_model"]
    poro  = model.porosity_metrics(best["layers"])
    neff  = model.effective_index_metrics(best["layers"], best["d_ox"])

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("MULTILAYER MODEL COMPARISON - POROUS GaAs\n")
        f.write("="*75 + "\n\n")
        f.write(f"Sample                             : 15min\n")
        f.write(f"Experimental file                  : {CSV_15}\n")
        f.write(f"Fitting range                      : {FIT_MIN:.1f} - {FIT_MAX:.1f} nm\n")
        f.write(f"UV/vis boundary                    : {UV_SPLIT_NM:.1f} nm\n")
        f.write(f"UV weight                          : x{UV_WEIGHT}\n")
        f.write(f"GaAs NK                            : {NK_GAAS}\n")
        f.write(f"GaAsOx NK                          : {NK_OXIDE}\n")
        f.write(f"Oxide thickness bounds             : {D_OX_MIN} - {D_OX_MAX} nm (free)\n")
        f.write(f"Effective-index reference wl       : {NEFF_REFERENCE_WL:.1f} nm\n\n")

        f.write("MODEL COMPARISON\n")
        f.write("-"*130 + "\n")
        f.write(f"{'N layers':>10} {'Parameters':>12} {'RMSE (%R)':>12} "
                f"{'RMSE_UV':>10} {'RMSE_vis':>10} "
                f"{'AIC':>15} {'BIC':>15} "
                f"{'d_ox (nm)':>12} {'Avg porosity (%)':>18} "
                f"{'n_eff surf':>14} {'n_eff avg':>12}\n")
        f.write("-"*130 + "\n")
        for r in all_results:
            m    = r["_model"]
            p    = m.porosity_metrics(r["layers"])
            ne   = m.effective_index_metrics(r["layers"], r["d_ox"])
            mk   = " <- BIC" if r["N"] == best["N"] else ""
            f.write(f"{r['N']:>10d} {r['n_params']:>12d} "
                    f"{r['rmse_tot']*100:>12.4f} "
                    f"{r['rmse_uv']*100:>10.4f} "
                    f"{r['rmse_vis']*100:>10.4f} "
                    f"{r['aic']:>15.4f} {r['bic']:>15.4f} "
                    f"{r['d_ox']:>12.4f} "
                    f"{p['avg_porosity']*100:>18.4f} "
                    f"{ne['n_eff_surface']:>14.4f} "
                    f"{ne['n_eff_avg']:>12.4f}{mk}\n")

        f.write(f"\n\nSELECTED MODEL ACCORDING TO BIC\n")
        f.write("-"*50 + "\n")
        f.write(f"Selected model          : N = {best['N']} effective sublayers\n")
        f.write(f"Number of parameters    : {best['n_params']}\n")
        f.write(f"RMSE total              : {best['rmse_tot']*100:.4f} %R\n")
        f.write(f"RMSE UV                 : {best['rmse_uv']*100:.4f} %R\n")
        f.write(f"RMSE visible            : {best['rmse_vis']*100:.4f} %R\n")
        f.write(f"R2                      : {best['r2']:.5f}\n")
        f.write(f"AIC                     : {best['aic']:.4f}\n")
        f.write(f"BIC                     : {best['bic']:.4f}\n\n")

        f.write("OXIDE LAYER\n")
        f.write("-"*50 + "\n")
        f.write(f"GaAsOx thickness (free) : {neff['d_ox']:.4f} nm\n")
        f.write(f"n_GaAsOx @ {neff['wl_ref_actual']:.0f} nm      : {neff['n_oxide_ref']:.4f}\n")
        f.write(f"k_GaAsOx @ {neff['wl_ref_actual']:.0f} nm      : {neff['k_oxide_ref']:.4f}\n\n")

        f.write("POROSITY METRICS\n")
        f.write("-"*50 + "\n")
        f.write(f"Total porous thickness              : {poro['d_total']:.4f} nm\n")
        f.write(f"Surface void fraction               : {poro['f_surface']*100:.4f} %\n")
        f.write(f"Bottom void fraction                : {poro['f_bottom']*100:.4f} %\n")
        f.write(f"Thickness-weighted average porosity : {poro['avg_porosity']*100:.4f} %\n\n")

        f.write("EFFECTIVE REFRACTIVE INDEX METRICS\n")
        f.write("-"*50 + "\n")
        f.write(f"Reference wavelength                : {neff['wl_ref_actual']:.2f} nm\n")
        f.write(f"n_GaAs(ref)                         : {neff['n_gaas_ref']:.4f}\n")
        f.write(f"k_GaAs(ref)                         : {neff['k_gaas_ref']:.4f}\n")
        f.write(f"Optimal AR index sqrt(n_GaAs)        : {neff['n_opt_ar']:.4f}\n")
        f.write(f"Surface n_eff                       : {neff['n_eff_surface']:.4f}\n")
        f.write(f"Surface k_eff                       : {neff['k_eff_surface']:.4f}\n")
        f.write(f"Bottom n_eff                        : {neff['n_eff_bottom']:.4f}\n")
        f.write(f"Bottom k_eff                        : {neff['k_eff_bottom']:.4f}\n")
        f.write(f"Thickness-weighted average n_eff    : {neff['n_eff_avg']:.4f}\n")
        f.write(f"Thickness-weighted average k_eff    : {neff['k_eff_avg']:.4f}\n")
        f.write(f"|n_eff(surface)-n_opt|              : {neff['delta_surface_to_optimum']:.4f}\n")
        f.write(f"|n_eff(avg)-n_opt|                  : {neff['delta_avg_to_optimum']:.4f}\n\n")

        f.write("OPTIMIZED PARAMETERS OF THE SELECTED MODEL\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Layer':>8} {'d (nm)':>12} {'void fraction':>16} "
                f"{'porosity (%)':>14} {'n_eff(ref)':>12} {'k_eff(ref)':>12}\n")
        f.write("-"*80 + "\n")
        for i, (d_i, f_i) in enumerate(best["layers"]):
            f.write(f"{i+1:>8d} {d_i:>12.4f} {f_i:>16.4f} "
                    f"{f_i*100:>14.4f} "
                    f"{neff['n_eff_layers'][i]:>12.4f} "
                    f"{neff['k_eff_layers'][i]:>12.4f}\n")

        f.write("\n\nSPECTRUM OF THE SELECTED MODEL\n")
        f.write("lambda(nm)    R_fit(%R)   R_exp(%R)   resid(%R)   region\n")
        f.write("-"*60 + "\n")
        for wl, rf, re in zip(data.wl, best["R"], data.R_exp):
            region = "UV" if wl < UV_SPLIT_NM else "vis"
            f.write(f"{wl:10.1f} {rf*100:12.4f} {re*100:10.4f} "
                    f"{(rf-re)*100:10.4f} {region:>8}\n")

    print(f"-> Results saved: {OUTPUT_TXT}")

# =============================================================================
# PLOTTING
# =============================================================================

_RC = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#111111",
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "grid.color":        "#dddddd",
    "text.color":        "#111111",
    "font.family":       "sans-serif",
    "font.size":         10,
    "legend.framealpha": 0.9,
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#aaaaaa",
}

C_EXP   = "#2166ac"
C_FIT   = "#d6604d"
C_RMSE  = "#2166ac"
C_AIC   = "#4dac26"
C_BIC   = "#d6604d"
C_SEL   = "#f4a582"
C_THICK = "#4393c3"
C_PORO  = "#74add1"
C_AVG   = "#d73027"
C_PROF  = "#4dac26"
C_RES_P = "#E65100"
C_RES_N = "#6A1B9A"


def plot_fit_curve(data, best):
    """Figure 1 -- reflectance fit only (publication-ready)."""
    plt.rcParams.update(_RC)

    neff = best["_model"].effective_index_metrics(best["layers"], best["d_ox"])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    ax.plot(data.wl, data.R_exp * 100,
            "o", color=C_EXP, markersize=3, alpha=0.7, label="Experimental 15min")
    ax.plot(data.wl, best["R"] * 100,
            "-", color=C_FIT, linewidth=2.2,
            label=(f"TMM fit  --  N = {best['N']} sublayers"
                   f"  |  d_ox = {neff['d_ox']:.2f} nm"
                   f"  |  RMSE = {best['rmse_tot']*100:.4f} %R"
                   f"  |  BIC = {best['bic']:.2f}"))

    axr = ax.twinx()
    resid = (best["R"] - data.R_exp) * 100
    axr.fill_between(data.wl, resid, 0, where=resid >= 0, color=C_RES_P, alpha=0.2)
    axr.fill_between(data.wl, resid, 0, where=resid <  0, color=C_RES_N, alpha=0.2)
    axr.axhline(0, color="gray", lw=0.5, ls="--")
    axr.axvline(UV_SPLIT_NM, color="gray", lw=0.8, ls=":", alpha=0.6)
    axr.set_ylabel("Residual (%R)", color="gray", fontsize=9)
    axr.set_ylim(-6, 6)

    ax.axvline(UV_SPLIT_NM, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Reflectance (%R)", fontsize=12)
    ax.set_title(
        "Porous GaAs -- 15min etching  |  Best fit selected by BIC",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linewidth=0.6)
    ax.set_xlim([FIT_MIN, FIT_MAX])

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG1, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"-> Figure 1 (fit curve) saved: {OUTPUT_PNG1}")
    plt.close(fig)


def plot_model_analysis(data, all_results, best):
    """Figure 2 -- RMSE/AIC/BIC, layer parameters, n_eff, KPI box."""
    plt.rcParams.update(_RC)

    n_values    = [r["N"] for r in all_results]
    rmse_values = [r["rmse_tot"] * 100 for r in all_results]
    aic_values  = [r["aic"] for r in all_results]
    bic_values  = [r["bic"] for r in all_results]

    model = best["_model"]
    poro  = model.porosity_metrics(best["layers"])
    neff  = model.effective_index_metrics(best["layers"], best["d_ox"])

    d_opt = np.array([lp[0] for lp in best["layers"]])
    f_opt = np.array([lp[1] for lp in best["layers"]])

    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.40, wspace=0.32,
        top=0.91, bottom=0.06,
        left=0.07, right=0.97,
    )

    mk = dict(marker="o", linewidth=1.8, markersize=6)

    # -- ROW 0 : criteria vs N
    ax_rmse = fig.add_subplot(gs[0, 0])
    ax_rmse.plot(n_values, rmse_values, color=C_RMSE, **mk)
    ax_rmse.axvline(best["N"], color=C_SEL, linestyle="--", linewidth=1.4,
                    label=f"Selected N = {best['N']}")
    ax_rmse.set_xlabel("Number of sublayers  N", fontsize=10)
    ax_rmse.set_ylabel("RMSE (%R)", fontsize=10)
    ax_rmse.set_title("Fit quality  (RMSE vs N)", fontsize=10, fontweight="bold")
    ax_rmse.legend(fontsize=8)
    ax_rmse.grid(True, linewidth=0.5)
    ax_rmse.set_xticks(n_values)

    ax_aic = fig.add_subplot(gs[0, 1])
    ax_aic.plot(n_values, aic_values, color=C_AIC, **mk)
    ax_aic.axvline(best["N"], color=C_SEL, linestyle="--", linewidth=1.4,
                   label=f"Selected N = {best['N']}")
    ax_aic.set_xlabel("Number of sublayers  N", fontsize=10)
    ax_aic.set_ylabel("AIC", fontsize=10)
    ax_aic.set_title("Model complexity  (AIC vs N)", fontsize=10, fontweight="bold")
    ax_aic.legend(fontsize=8)
    ax_aic.grid(True, linewidth=0.5)
    ax_aic.set_xticks(n_values)

    ax_bic = fig.add_subplot(gs[0, 2])
    ax_bic.plot(n_values, bic_values, color=C_BIC, **mk)
    ax_bic.axvline(best["N"], color=C_SEL, linestyle="--", linewidth=1.4,
                   label=f"Selected N = {best['N']}")
    idx_best = int(np.argmin(bic_values))
    ax_bic.plot(n_values[idx_best], bic_values[idx_best],
                "s", color=C_SEL, markersize=10, zorder=5,
                label=f"Min BIC = {bic_values[idx_best]:.2f}")
    ax_bic.set_xlabel("Number of sublayers  N", fontsize=10)
    ax_bic.set_ylabel("BIC", fontsize=10)
    ax_bic.set_title("Selection criterion  (BIC vs N)", fontsize=10, fontweight="bold")
    ax_bic.legend(fontsize=8)
    ax_bic.grid(True, linewidth=0.5)
    ax_bic.set_xticks(n_values)

    # -- ROW 1 : per-layer parameters
    x       = np.arange(best["N"])
    xlabels = [f"L{i+1}" for i in range(best["N"])]

    ax_d = fig.add_subplot(gs[1, 0])
    bars_d = ax_d.bar(x, d_opt, color=C_THICK, edgecolor="white", linewidth=0.8)
    ax_d.bar_label(bars_d, fmt="%.1f", fontsize=8, padding=2)
    ax_d.set_xticks(x); ax_d.set_xticklabels(xlabels)
    ax_d.set_ylabel("Thickness (nm)", fontsize=10)
    ax_d.set_title(f"Layer thicknesses  (total = {poro['d_total']:.1f} nm)",
                   fontsize=10, fontweight="bold")
    ax_d.set_ylim(0, d_opt.max() * 1.22)
    ax_d.grid(True, axis="y", linewidth=0.5)

    ax_p = fig.add_subplot(gs[1, 1])
    bars_p = ax_p.bar(x, f_opt * 100, color=C_PORO, edgecolor="white", linewidth=0.8)
    ax_p.bar_label(bars_p, fmt="%.1f", fontsize=8, padding=2)
    ax_p.axhline(poro["avg_porosity"] * 100, color=C_AVG, linestyle="--", linewidth=1.5,
                 label=f"Weighted avg = {poro['avg_porosity']*100:.2f} %")
    ax_p.set_xticks(x); ax_p.set_xticklabels(xlabels)
    ax_p.set_ylabel("Effective void fraction (%)", fontsize=10)
    ax_p.set_ylim(0, min(110, float(f_opt.max()) * 100 * 1.25))
    ax_p.set_title("Effective void fractions", fontsize=10, fontweight="bold")
    ax_p.legend(fontsize=8)
    ax_p.grid(True, axis="y", linewidth=0.5)

    ax_prof = fig.add_subplot(gs[1, 2])
    depth   = np.cumsum(np.concatenate([[0], d_opt]))
    step_x, step_y = [], []
    for i in range(best["N"]):
        step_x += [f_opt[i] * 100, f_opt[i] * 100]
        step_y += [depth[i], depth[i + 1]]
    ax_prof.plot(step_x, step_y, color=C_PROF, linewidth=2.0)
    ax_prof.fill_betweenx(step_y, step_x, alpha=0.15, color=C_PROF)
    ax_prof.axvline(poro["avg_porosity"] * 100, color=C_AVG, linestyle="--",
                    linewidth=1.4, label=f"Avg = {poro['avg_porosity']*100:.2f} %")
    ax_prof.set_xlabel("Effective void fraction (%)", fontsize=10)
    ax_prof.set_ylabel("Depth (nm)", fontsize=10)
    ax_prof.invert_yaxis()
    ax_prof.set_title("Effective void-fraction profile", fontsize=10, fontweight="bold")
    ax_prof.legend(fontsize=8)
    ax_prof.grid(True, linewidth=0.5)

    # -- ROW 2 : n_eff / k_eff per layer + KPI summary box
    ax_neff = fig.add_subplot(gs[2, 0])
    ax_neff.bar(x, neff["n_eff_layers"], color="#8da0cb", edgecolor="white", linewidth=0.8)
    ax_neff.axhline(neff["n_opt_ar"], color=C_AVG, linestyle="--", linewidth=1.4,
                    label=f"n_opt = {neff['n_opt_ar']:.3f}")
    ax_neff.set_xticks(x); ax_neff.set_xticklabels(xlabels)
    ax_neff.set_ylabel(f"n_eff @ {neff['wl_ref_actual']:.0f} nm", fontsize=10)
    ax_neff.set_title("Effective refractive index", fontsize=10, fontweight="bold")
    ax_neff.legend(fontsize=8)
    ax_neff.grid(True, axis="y", linewidth=0.5)

    ax_keff = fig.add_subplot(gs[2, 1])
    ax_keff.bar(x, neff["k_eff_layers"], color="#fc8d62", edgecolor="white", linewidth=0.8)
    ax_keff.set_xticks(x); ax_keff.set_xticklabels(xlabels)
    ax_keff.set_ylabel(f"k_eff @ {neff['wl_ref_actual']:.0f} nm", fontsize=10)
    ax_keff.set_title("Effective extinction coefficient", fontsize=10, fontweight="bold")
    ax_keff.grid(True, axis="y", linewidth=0.5)

    # KPI box — publication-style summary (2 key metrics)
    ax_sum = fig.add_subplot(gs[2, 2])
    ax_sum.axis("off")

    box_color  = "#fff0f0"
    edge_color = "#c0392b"

    from matplotlib.patches import FancyBboxPatch
    fancy = FancyBboxPatch(
        (0.05, 0.12), 0.90, 0.76,
        boxstyle="round,pad=0.04",
        linewidth=2.5, edgecolor=edge_color, facecolor=box_color,
        transform=ax_sum.transAxes, zorder=1,
    )
    ax_sum.add_patch(fancy)

    # Separator line
    ax_sum.plot([0.07, 0.93], [0.50, 0.50],
                color=edge_color, linewidth=1.0, transform=ax_sum.transAxes)

    kpi_items = [
        ("AVERAGE EFFECTIVE VOID FRACTION",
         f"{poro['avg_porosity']*100:.2f} %",
         0.75),
        ("TOTAL THICKNESS",
         f"{poro['d_total']:.1f} nm",
         0.25),
    ]

    for label, value, yc in kpi_items:
        ax_sum.text(0.50, yc + 0.14, label,
                    ha="center", va="center",
                    fontsize=9.5, fontweight="bold",
                    color="#7f0000",
                    transform=ax_sum.transAxes, zorder=2)
        ax_sum.text(0.50, yc - 0.05, value,
                    ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="#c0392b",
                    transform=ax_sum.transAxes, zorder=2)

    fig.suptitle(
        f"Porous GaAs -- 15min etching  |  "
        "TMM + Bruggeman EMA  |  "
        "Air / GaAsOx / N porous GaAs / GaAs  |  Criterion: BIC",
        fontsize=12, fontweight="bold", y=0.97, color="#1a1a1a",
    )

    plt.savefig(OUTPUT_PNG2, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"-> Figure 2 (analysis) saved: {OUTPUT_PNG2}")
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("="*70)
    print("  15 min Porous GaAs — REFLECTANCE FITTING  [fit_15min_4.py  v5]")
    print(f"  Range     : {FIT_MIN}-{FIT_MAX} nm")
    print(f"  UV/vis    : < / >= {UV_SPLIT_NM} nm  (UV weight = x{UV_WEIGHT})")
    print(f"  Structure : Air / GaAsOx(free) / L1/.../LN / GaAs")
    print(f"  N tested  : {N_MIN} to {N_MAX}  (BIC selection)")
    print(f"  DE maxiter: 1500 | popsize: 20 | multi-start LBFGS: 14")
    print("="*70)
    print()

    safe_mkdir(RESULTS_DIR)
    data = DataLoader()

    all_results = []

    for N in range(N_MIN, N_MAX + 1):
        model  = ModelN(data, N)
        engine = FittingEngine(model)
        (p_opt, R_opt, d_ox, layers_p,
         rmse_tot, rmse_uv, rmse_vis, r2, aic, bic) = engine.run()

        all_results.append({
            "N":        N,
            "n_params": model.n_params,
            "d_ox":     d_ox,
            "layers":   layers_p,
            "R":        R_opt,
            "rmse_tot": rmse_tot,
            "rmse_uv":  rmse_uv,
            "rmse_vis": rmse_vis,
            "r2":       r2,
            "aic":      aic,
            "bic":      bic,
            "_model":   model,
        })

    best = min(all_results, key=lambda r: r["bic"])

    print()
    print("="*70)
    for r in all_results:
        mk = " <- BIC" if r["N"] == best["N"] else ""
        print(f"  N={r['N']} | "
              f"RMSE_tot={r['rmse_tot']*100:.4f}%R | "
              f"RMSE_UV={r['rmse_uv']*100:.4f}%R | "
              f"RMSE_vis={r['rmse_vis']*100:.4f}%R | "
              f"BIC={r['bic']:.2f}{mk}")
    print("="*70)

    save_results(data, all_results, best)
    plot_fit_curve(data, best)
    plot_model_analysis(data, all_results, best)

    print(f"\nAll outputs -> {RESULTS_DIR}")
    print()

    # Ouverture automatique des figures
    os.startfile(os.path.abspath(OUTPUT_PNG1))
    os.startfile(os.path.abspath(OUTPUT_PNG2))


if __name__ == "__main__":
    main()
