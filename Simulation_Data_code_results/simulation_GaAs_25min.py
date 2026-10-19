"""
================================================================================
  REFLECTANCE FITTING AND MODEL SELECTION - POROUS GaAs (25 min)
  -------------------------------------------------------------------
  Optical method : Transfer Matrix Method (TMM)
  Mixing model   : Bruggeman Effective Medium Approximation (EMA)
  Optimisation   : Differential Evolution -> L-BFGS-B with physical bounds

  Fit range      : 277-800 nm

  Strategy:
  - Tests N = 2 to 7 effective sublayers
  - Optimizes layer thicknesses d_i and void fractions f_i
  - Computes RMSE, AIC, BIC and selects the best model by BIC
  - For N = 6: warm start from the 35min best-fit parameters
    (similar physics, same etching process, shorter duration)
  - Saves results in results_english_porosity_neff/25min/
================================================================================
"""

import os
import time
import warnings
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

NK_GAAS = "Gaas041t.nk"
NK_VOID = "Void.nk"

# Experimental file for the 25 min etching sample
DATASET_LABEL = "25min"
DATASET_FILE  = "R 25-V0.Sample.csv"

FIT_MIN = 277.0
FIT_MAX = 800.0

# Reference wavelength for effective-index interpretation
NEFF_REFERENCE_WL = 550.0  # nm

# Tested number of effective sublayers
N_LAYERS_LIST = [2, 3, 4, 5, 6, 7]

# Initial total thickness for non-warm-start N values
# 25min etches less than 35min (479 nm) → estimated ~350 nm
D_INIT_TOTAL = 350.0  # nm

# Physical bounds
BOUNDS_D_SINGLE = (5.0, 150.0)    # layer thickness, nm
BOUNDS_F_SINGLE = (0.01, 0.99)    # void fraction

# Output directory
RESULTS_DIR = "results_english_porosity_neff"

# Optimisation settings
FAST_MODE = False
DE_MAXITER = 300 if not FAST_MODE else 30
DE_POPSIZE = 12  if not FAST_MODE else 6

# =============================================================================
# WARM START FROM 35 min BEST FIT (N = 6)
# Optimized params: [d1..d6, f1..f6]
# Source: results_english_porosity_neff/35min/model_selection_35min.txt
# =============================================================================
WARM_START = {
    6: [
        # thicknesses (nm)
        120.6740, 117.3886, 34.9727, 74.8725, 88.2916, 43.0749,
        # void fractions
        0.8722,   0.6159,  0.4064,  0.1518,  0.2508,  0.1019,
    ],
}


# =============================================================================
# UTILITIES
# =============================================================================

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_nk_file(filepath: str):
    """
    Reads a WinElli-like .nk file.
    First line: type E_min(eV) E_max(eV) N_points
    Next lines: n k
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, encoding="latin-1") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty file: {filepath}")

    header = lines[0].split()
    if len(header) < 3:
        raise ValueError(f"Invalid .nk header in: {filepath}")

    E_min, E_max = float(header[1]), float(header[2])

    nk_data = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                nk_data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    nk_data = np.array(nk_data)

    if len(nk_data) == 0:
        raise ValueError(f"No valid n/k data in: {filepath}")

    energies = np.linspace(E_min, E_max, len(nk_data))
    return energies, nk_data[:, 0], nk_data[:, 1]


def load_csv_reflectance(filepath: str):
    """
    Reads experimental reflectance CSV.
    Expected format: ';' separator, comma decimal, 2 header lines.
    Columns: wavelength nm ; R%
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    wavelengths = []
    reflectances = []

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

    wl = np.array(wavelengths)
    R  = np.array(reflectances)

    if len(wl) == 0:
        raise ValueError(f"No valid experimental data in: {filepath}")

    idx = np.argsort(wl)
    return wl[idx], R[idx]


# =============================================================================
# BRUGGEMAN EMA
# =============================================================================

def bruggeman_ema(n1, k1, n2, k2, f2):
    """
    Bruggeman EMA for a two-component medium:
    component 1: GaAs
    component 2: void/air
    f2: void fraction
    """
    f2 = np.clip(f2, BOUNDS_F_SINGLE[0], BOUNDS_F_SINGLE[1])
    f1 = 1.0 - f2

    eps1 = (n1 + 1j * k1) ** 2
    eps2 = (n2 + 1j * k2) ** 2

    B = -(f1 * (2 * eps1 - eps2) + f2 * (2 * eps2 - eps1))
    C = -eps1 * eps2

    disc   = B ** 2 - 8.0 * C
    sqrt_d = np.sqrt(disc.astype(complex))

    sol1 = (-B + sqrt_d) / 4.0
    sol2 = (-B - sqrt_d) / 4.0

    eps_eff = np.where(np.imag(sol1) >= 0, sol1, sol2)

    n_eff_complex = np.sqrt(eps_eff)
    n_eff_complex = np.where(
        np.real(n_eff_complex) < 0,
        -n_eff_complex,
        n_eff_complex
    )

    return np.real(n_eff_complex), np.imag(n_eff_complex)


# =============================================================================
# TMM AT NORMAL INCIDENCE
# =============================================================================

def tmm_reflectance(wavelengths_nm, layers):
    """Transfer matrix method at normal incidence — vectorised over wavelengths."""
    N = len(wavelengths_nm)

    M = np.zeros((N, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0

    for j in range(1, len(layers) - 1):
        nj    = layers[j]["n"] + 1j * layers[j]["k"]
        dj    = layers[j]["d"]
        delta = 2.0 * np.pi * nj * dj / wavelengths_nm

        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        M_j = np.empty((N, 2, 2), dtype=complex)
        M_j[:, 0, 0] =  cos_d
        M_j[:, 0, 1] = -1j * sin_d / nj
        M_j[:, 1, 0] = -1j * nj * sin_d
        M_j[:, 1, 1] =  cos_d

        M = M @ M_j

    n0  = layers[0]["n"]  + 1j * layers[0]["k"]
    ns  = layers[-1]["n"] + 1j * layers[-1]["k"]

    m11, m12 = M[:, 0, 0], M[:, 0, 1]
    m21, m22 = M[:, 1, 0], M[:, 1, 1]

    num = (m11 + m12 * ns) * n0 - (m21 + m22 * ns)
    den = (m11 + m12 * ns) * n0 + (m21 + m22 * ns)

    return np.abs(num / den) ** 2


# =============================================================================
# MULTILAYER MODEL
# =============================================================================

class ThinFilmModel:
    def __init__(self, n_layers: int, csv_exp: str, label: str):
        self.n_layers = int(n_layers)
        self.csv_exp  = csv_exp
        self.label    = label

        # Warm start for this N (if available)
        self.warm_start = WARM_START.get(self.n_layers, None)

        print(f"[1/4] Loading optical files for {label}, N = {self.n_layers} layers...")
        gaas_e, gaas_n, gaas_k = load_nk_file(NK_GAAS)
        load_nk_file(NK_VOID)   # existence check only

        print(f"[2/4] Loading experimental spectrum: {csv_exp}")
        wl_exp, R_exp = load_csv_reflectance(csv_exp)

        lam_gaas = 1239.84 / gaas_e
        idx      = np.argsort(lam_gaas)
        lam_gaas, gaas_n, gaas_k = lam_gaas[idx], gaas_n[idx], gaas_k[idx]

        lam_min = max(wl_exp.min(), lam_gaas.min(), FIT_MIN)
        lam_max = min(wl_exp.max(), lam_gaas.max(), FIT_MAX)
        mask    = (wl_exp >= lam_min) & (wl_exp <= lam_max)

        self.wl    = wl_exp[mask]
        self.R_exp = R_exp[mask]
        self.N     = len(self.wl)

        if self.N == 0:
            raise ValueError(
                f"No experimental points in the {FIT_MIN}-{FIT_MAX} nm range for {csv_exp}."
            )

        print(f"Fitting range: {self.wl.min():.1f} - {self.wl.max():.1f} nm  "
              f"({self.N} points)")

        interp_kw = dict(kind="linear", fill_value="extrapolate")
        self.n_gaas = interp1d(lam_gaas, gaas_n, **interp_kw)(self.wl)
        self.k_gaas = interp1d(lam_gaas, gaas_k, **interp_kw)(self.wl)
        self.n_void = np.ones(self.N)
        self.k_void = np.zeros(self.N)

        if self.warm_start is not None:
            rmse_ws = self.objective(self.warm_start) * 100
            print(f"   Warm-start (35min N={self.n_layers}) RMSE on 25min data: "
                  f"{rmse_ws:.4f} %R")

    def initial_params(self):
        """
        For N with a warm start: return warm-start values as starting point.
        Otherwise: uniform thickness + decreasing porosity gradient.
        """
        if self.warm_start is not None:
            return list(self.warm_start)
        d_init = [D_INIT_TOTAL / self.n_layers] * self.n_layers
        f_init = np.linspace(0.75, 0.15, self.n_layers).tolist()
        return d_init + f_init

    def bounds(self):
        return [BOUNDS_D_SINGLE] * self.n_layers + [BOUNDS_F_SINGLE] * self.n_layers

    def clip_params(self, params):
        params = np.array(params, dtype=float).copy()
        params[:self.n_layers] = np.clip(
            params[:self.n_layers], BOUNDS_D_SINGLE[0], BOUNDS_D_SINGLE[1]
        )
        params[self.n_layers:] = np.clip(
            params[self.n_layers:], BOUNDS_F_SINGLE[0], BOUNDS_F_SINGLE[1]
        )
        return params

    def simulate(self, params):
        params = self.clip_params(params)
        thicknesses_nm = params[:self.n_layers]
        void_fractions = params[self.n_layers:]

        layers = [{"n": self.n_void, "k": self.k_void, "d": 0.0}]  # air

        for d, fv in zip(thicknesses_nm, void_fractions):
            n_eff, k_eff = bruggeman_ema(
                self.n_gaas, self.k_gaas, self.n_void, self.k_void, fv
            )
            layers.append({"n": n_eff, "k": k_eff, "d": float(d)})

        layers.append({"n": self.n_gaas, "k": self.k_gaas, "d": 0.0})  # substrate

        return tmm_reflectance(self.wl, layers)

    def rmse(self, R_sim):
        return float(np.sqrt(np.mean((R_sim - self.R_exp) ** 2)))

    def rss(self, R_sim):
        return float(np.sum((R_sim - self.R_exp) ** 2))

    def aic_bic(self, R_sim):
        n   = len(self.R_exp)
        k   = 2 * self.n_layers
        rss = max(self.rss(R_sim), 1e-30)
        aic = 2 * k + n * np.log(rss / n)
        bic = k * np.log(n) + n * np.log(rss / n)
        return float(aic), float(bic)

    def objective(self, params):
        try:
            return self.rmse(self.simulate(self.clip_params(params)))
        except Exception:
            return 1e6

    def porosity_metrics(self, params):
        params  = self.clip_params(params)
        d_opt   = params[:self.n_layers]
        f_opt   = params[self.n_layers:]
        d_total = float(np.sum(d_opt))
        return {
            "d_total":      d_total,
            "f_surface":    float(f_opt[0]),
            "f_bottom":     float(f_opt[-1]),
            "avg_porosity": float(np.sum(d_opt * f_opt) / d_total) if d_total > 0 else np.nan,
        }

    def effective_index_metrics(self, params, ref_wl=NEFF_REFERENCE_WL):
        params    = self.clip_params(params)
        d_opt     = params[:self.n_layers]
        f_opt     = params[self.n_layers:]
        idx_ref   = int(np.argmin(np.abs(self.wl - ref_wl)))
        wl_actual = float(self.wl[idx_ref])

        n_gaas_ref = float(self.n_gaas[idx_ref])
        k_gaas_ref = float(self.k_gaas[idx_ref])
        n_opt_ar   = float(np.sqrt(n_gaas_ref))

        n_eff_layers, k_eff_layers = [], []
        for fv in f_opt:
            na, ka = bruggeman_ema(
                self.n_gaas, self.k_gaas, self.n_void, self.k_void, fv
            )
            n_eff_layers.append(float(na[idx_ref]))
            k_eff_layers.append(float(ka[idx_ref]))

        n_eff_layers = np.array(n_eff_layers)
        k_eff_layers = np.array(k_eff_layers)
        d_total      = float(np.sum(d_opt))
        n_eff_avg    = float(np.sum(d_opt * n_eff_layers) / d_total) if d_total > 0 else np.nan
        k_eff_avg    = float(np.sum(d_opt * k_eff_layers) / d_total) if d_total > 0 else np.nan

        return {
            "wl_ref_actual":          wl_actual,
            "n_gaas_ref":             n_gaas_ref,
            "k_gaas_ref":             k_gaas_ref,
            "n_opt_ar":               n_opt_ar,
            "n_eff_surface":          float(n_eff_layers[0]),
            "k_eff_surface":          float(k_eff_layers[0]),
            "n_eff_bottom":           float(n_eff_layers[-1]),
            "k_eff_bottom":           float(k_eff_layers[-1]),
            "n_eff_avg":              n_eff_avg,
            "k_eff_avg":              k_eff_avg,
            "delta_surface_to_optimum": float(abs(n_eff_layers[0] - n_opt_ar)),
            "delta_avg_to_optimum":     float(abs(n_eff_avg    - n_opt_ar)),
            "n_eff_layers":           n_eff_layers,
            "k_eff_layers":           k_eff_layers,
        }


# =============================================================================
# OPTIMISATION
# =============================================================================

def _lbfgsb(model, x0):
    """Shared L-BFGS-B call."""
    return minimize(
        model.objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=model.bounds(),
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8, "disp": False},
    )


class FittingEngine:
    def __init__(self, model: ThinFilmModel):
        self.model   = model
        self.history = []
        self._iter   = 0

    def _callback_de(self, xk, convergence):
        rmse = self.model.objective(xk) * 100
        self._iter += 1
        self.history.append((self._iter, rmse))
        if self._iter % 10 == 0:
            print(f"   DE iter {self._iter:4d} | RMSE = {rmse:.4f} %R | "
                  f"convergence = {convergence:.4f}")

    def run(self):
        p0     = self.model.initial_params()
        bounds = self.model.bounds()

        print("[3/4] Global optimisation — Differential Evolution")
        print(f"      Sample  : {self.model.label}")
        print(f"      N layers: {self.model.n_layers}")
        print(f"      DOF     : {2 * self.model.n_layers}")
        print(f"      p0 RMSE : {self.model.objective(p0) * 100:.4f} %R\n")

        t0         = time.time()
        self._iter = 0
        self.history = []

        result_de = differential_evolution(
            self.model.objective,
            bounds=bounds,
            seed=42,
            maxiter=DE_MAXITER,
            tol=1e-5,
            popsize=DE_POPSIZE,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=False,
            workers=1,
            callback=self._callback_de,
        )

        print(f"\n   -> DE done in {time.time() - t0:.1f}s "
              f"| RMSE = {result_de.fun * 100:.4f} %R")

        # ── Local refinement from DE result ───────────────────────────────────
        print("\n[4/4] Local refinement — L-BFGS-B")
        t1 = time.time()
        res_de_local = _lbfgsb(self.model, result_de.x)
        params_de    = self.model.clip_params(res_de_local.x)
        rmse_de      = self.model.objective(params_de)
        print(f"   -> DE path   RMSE = {rmse_de * 100:.4f} %R  "
              f"({time.time() - t1:.1f}s)")

        # ── Warm-start refinement (35min params → 25min data) ─────────────────
        params_opt = params_de
        rmse_final = rmse_de

        if self.model.warm_start is not None:
            t2 = time.time()
            res_ws   = _lbfgsb(self.model, self.model.warm_start)
            params_ws = self.model.clip_params(res_ws.x)
            rmse_ws   = self.model.objective(params_ws)
            print(f"   -> Warm-start RMSE = {rmse_ws * 100:.4f} %R  "
                  f"({time.time() - t2:.1f}s)")

            if rmse_ws < rmse_de:
                params_opt = params_ws
                rmse_final = rmse_ws
                print("   -> Warm-start selected (lower RMSE)")
            else:
                print("   -> DE path selected (lower RMSE)")

        print(f"\n   => Final RMSE = {rmse_final * 100:.4f} %R")
        return params_opt, rmse_final


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_model_selection_results(all_results, best_result, output_txt):
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("MULTILAYER MODEL COMPARISON - POROUS GaAs\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Sample: {best_result['label']}\n")
        f.write("Selection criteria: RMSE, AIC, BIC\n")
        f.write("Model: Air / N effective GaAs+Void sublayers / bulk GaAs\n")
        f.write(f"Fitting range: {FIT_MIN:.1f} - {FIT_MAX:.1f} nm\n")
        f.write(f"Effective-index reference wavelength: {NEFF_REFERENCE_WL:.1f} nm\n\n")

        f.write(
            f"{'N layers':>10} {'Parameters':>12} {'RMSE (%R)':>12} "
            f"{'AIC':>15} {'BIC':>15} {'Avg porosity (%)':>18} "
            f"{'n_eff surf':>14} {'n_eff avg':>12}\n"
        )
        f.write("-" * 120 + "\n")

        for res in all_results:
            poro = res["model"].porosity_metrics(res["params"])
            neff = res["model"].effective_index_metrics(res["params"])
            f.write(
                f"{res['n_layers']:>10d} {res['n_params']:>12d} "
                f"{res['rmse'] * 100:>12.4f} {res['aic']:>15.4f} {res['bic']:>15.4f} "
                f"{poro['avg_porosity'] * 100:>18.4f} "
                f"{neff['n_eff_surface']:>14.4f} {neff['n_eff_avg']:>12.4f}\n"
            )

        f.write("\n\nSELECTED MODEL ACCORDING TO BIC\n")
        f.write("-" * 50 + "\n")
        f.write(f"Selected model          : N = {best_result['n_layers']} effective sublayers\n")
        f.write(f"Number of parameters    : {best_result['n_params']}\n")
        f.write(f"RMSE                    : {best_result['rmse'] * 100:.4f} %R\n")
        f.write(f"AIC                     : {best_result['aic']:.4f}\n")
        f.write(f"BIC                     : {best_result['bic']:.4f}\n\n")

        model  = best_result["model"]
        params = best_result["params"]
        d_opt  = params[:model.n_layers]
        f_opt  = params[model.n_layers:]
        poro   = model.porosity_metrics(params)
        neff   = model.effective_index_metrics(params)

        f.write("POROSITY METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total porous thickness              : {poro['d_total']:.4f} nm\n")
        f.write(f"Surface void fraction               : {poro['f_surface'] * 100:.4f} %\n")
        f.write(f"Bottom void fraction                : {poro['f_bottom'] * 100:.4f} %\n")
        f.write(f"Thickness-weighted average porosity : {poro['avg_porosity'] * 100:.4f} %\n\n")

        f.write("EFFECTIVE REFRACTIVE INDEX METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Reference wavelength                : {neff['wl_ref_actual']:.2f} nm\n")
        f.write(f"n_GaAs(ref)                         : {neff['n_gaas_ref']:.4f}\n")
        f.write(f"k_GaAs(ref)                         : {neff['k_gaas_ref']:.4f}\n")
        f.write(f"Optimal AR index sqrt(n_air*n_GaAs) : {neff['n_opt_ar']:.4f}\n")
        f.write(f"Surface n_eff                       : {neff['n_eff_surface']:.4f}\n")
        f.write(f"Surface k_eff                       : {neff['k_eff_surface']:.4f}\n")
        f.write(f"Bottom n_eff                        : {neff['n_eff_bottom']:.4f}\n")
        f.write(f"Bottom k_eff                        : {neff['k_eff_bottom']:.4f}\n")
        f.write(f"Thickness-weighted average n_eff    : {neff['n_eff_avg']:.4f}\n")
        f.write(f"Thickness-weighted average k_eff    : {neff['k_eff_avg']:.4f}\n")
        f.write(f"|n_eff(surface)-n_opt|              : {neff['delta_surface_to_optimum']:.4f}\n")
        f.write(f"|n_eff(avg)-n_opt|                  : {neff['delta_avg_to_optimum']:.4f}\n\n")

        f.write("OPTIMIZED PARAMETERS OF THE SELECTED MODEL\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"{'Layer':>8} {'d (nm)':>12} {'void fraction':>16} "
            f"{'porosity (%)':>14} {'n_eff(ref)':>12} {'k_eff(ref)':>12}\n"
        )
        f.write("-" * 80 + "\n")
        for i in range(model.n_layers):
            f.write(
                f"{i + 1:>8d} {d_opt[i]:>12.4f} {f_opt[i]:>16.4f} "
                f"{f_opt[i] * 100:>14.4f} "
                f"{neff['n_eff_layers'][i]:>12.4f} "
                f"{neff['k_eff_layers'][i]:>12.4f}\n"
            )

        f.write("\n\nSPECTRUM OF THE SELECTED MODEL\n")
        f.write("lambda(nm)    R_fit(%R)   R_exp(%R)\n")
        f.write("-" * 45 + "\n")
        for wl, rf, re in zip(model.wl, best_result["R_final"], model.R_exp):
            f.write(f"{wl:10.1f} {rf * 100:12.4f} {re * 100:10.4f}\n")

    print(f"-> Results saved: {output_txt}")


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


def plot_fit_curve(best_result, output_png):
    """Figure 1 — reflectance fit only (publication-ready)."""
    plt.rcParams.update(_RC)

    best_model = best_result["model"]
    best_R     = best_result["R_final"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    ax.plot(best_model.wl, best_model.R_exp * 100,
            "o", color=C_EXP, markersize=3, alpha=0.7, label="Experimental")
    ax.plot(best_model.wl, best_R * 100,
            "-", color=C_FIT, linewidth=2.2,
            label=(f"TMM fit  —  N = {best_model.n_layers} sublayers"
                   f"   |   RMSE = {best_result['rmse'] * 100:.4f} %R"
                   f"   |   BIC = {best_result['bic']:.2f}"))
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Reflectance (%R)", fontsize=12)
    ax.set_title(
        f"Porous GaAs — {best_result['label']} etching  |  Best fit selected by BIC",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linewidth=0.6)
    ax.set_xlim([FIT_MIN, FIT_MAX])

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"-> Figure 1 (fit curve) saved: {output_png}")
    plt.close(fig)


def plot_model_analysis(all_results, best_result, output_png):
    """Figure 2 — RMSE/AIC/BIC, layer parameters, depth profile, summary table."""
    plt.rcParams.update(_RC)

    n_values    = [res["n_layers"] for res in all_results]
    rmse_values = [res["rmse"] * 100 for res in all_results]
    aic_values  = [res["aic"] for res in all_results]
    bic_values  = [res["bic"] for res in all_results]

    best_model  = best_result["model"]
    best_params = best_result["params"]

    d_opt = best_params[:best_model.n_layers]
    f_opt = best_params[best_model.n_layers:]
    poro  = best_model.porosity_metrics(best_params)
    neff  = best_model.effective_index_metrics(best_params)

    # Square-ish layout: 3 rows × 3 cols + table row at bottom
    fig = plt.figure(figsize=(13, 12))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        height_ratios=[1.5, 1.5, 1.5, 1.2],
        hspace=0.38, wspace=0.32,
        top=0.92, bottom=0.04,
        left=0.07, right=0.97,
    )

    mk = dict(marker="o", linewidth=1.8, markersize=6)

    # ── ROW 0 : criteria vs N ────────────────────────────────────────────────
    ax_rmse = fig.add_subplot(gs[0, 0])
    ax_rmse.plot(n_values, rmse_values, color=C_RMSE, **mk)
    ax_rmse.axvline(best_model.n_layers, color=C_SEL, linestyle="--", linewidth=1.4,
                    label=f"Selected N = {best_model.n_layers}")
    ax_rmse.set_xlabel("Number of sublayers  N", fontsize=10)
    ax_rmse.set_ylabel("RMSE (%R)", fontsize=10)
    ax_rmse.set_title("Fit quality  (RMSE vs N)", fontsize=10, fontweight="bold")
    ax_rmse.legend(fontsize=8)
    ax_rmse.grid(True, linewidth=0.5)
    ax_rmse.set_xticks(n_values)

    ax_aic = fig.add_subplot(gs[0, 1])
    ax_aic.plot(n_values, aic_values, color=C_AIC, **mk)
    ax_aic.axvline(best_model.n_layers, color=C_SEL, linestyle="--", linewidth=1.4,
                   label=f"Selected N = {best_model.n_layers}")
    ax_aic.set_xlabel("Number of sublayers  N", fontsize=10)
    ax_aic.set_ylabel("AIC", fontsize=10)
    ax_aic.set_title("Model complexity  (AIC vs N)", fontsize=10, fontweight="bold")
    ax_aic.legend(fontsize=8)
    ax_aic.grid(True, linewidth=0.5)
    ax_aic.set_xticks(n_values)

    ax_bic = fig.add_subplot(gs[0, 2])
    ax_bic.plot(n_values, bic_values, color=C_BIC, **mk)
    ax_bic.axvline(best_model.n_layers, color=C_SEL, linestyle="--", linewidth=1.4,
                   label=f"Selected N = {best_model.n_layers}")
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

    # ── ROW 1 : per-layer parameters ─────────────────────────────────────────
    x       = np.arange(best_model.n_layers)
    xlabels = [f"L{i + 1}" for i in range(best_model.n_layers)]

    ax_d = fig.add_subplot(gs[1, 0])
    bars_d = ax_d.bar(x, d_opt, color=C_THICK, edgecolor="white", linewidth=0.8)
    ax_d.bar_label(bars_d, fmt="%.1f", fontsize=8, padding=2)
    ax_d.set_xticks(x);  ax_d.set_xticklabels(xlabels)
    ax_d.set_ylabel("Thickness (nm)", fontsize=10)
    ax_d.set_title(f"Layer thicknesses  (total = {poro['d_total']:.1f} nm)",
                   fontsize=10, fontweight="bold")
    ax_d.grid(True, axis="y", linewidth=0.5)

    ax_p = fig.add_subplot(gs[1, 1])
    bars_p = ax_p.bar(x, np.array(f_opt) * 100, color=C_PORO, edgecolor="white", linewidth=0.8)
    ax_p.bar_label(bars_p, fmt="%.1f", fontsize=8, padding=2)
    ax_p.axhline(poro["avg_porosity"] * 100, color=C_AVG, linestyle="--", linewidth=1.5,
                 label=f"Weighted avg = {poro['avg_porosity'] * 100:.2f} %")
    ax_p.set_xticks(x);  ax_p.set_xticklabels(xlabels)
    ax_p.set_ylabel("Porosity (%)", fontsize=10)
    ax_p.set_ylim(0, min(105, max(np.array(f_opt) * 100) * 1.25))
    ax_p.set_title("Layer porosities", fontsize=10, fontweight="bold")
    ax_p.legend(fontsize=8)
    ax_p.grid(True, axis="y", linewidth=0.5)

    ax_prof = fig.add_subplot(gs[1, 2])
    depth   = np.cumsum([0] + list(d_opt))
    step_x, step_y = [], []
    for i in range(best_model.n_layers):
        step_x += [f_opt[i] * 100, f_opt[i] * 100]
        step_y += [depth[i], depth[i + 1]]
    ax_prof.plot(step_x, step_y, color=C_PROF, linewidth=2.0)
    ax_prof.fill_betweenx(step_y, step_x, alpha=0.15, color=C_PROF)
    ax_prof.axvline(poro["avg_porosity"] * 100, color=C_AVG, linestyle="--",
                    linewidth=1.4, label=f"Avg = {poro['avg_porosity'] * 100:.2f} %")
    ax_prof.set_xlabel("Porosity (%)", fontsize=10)
    ax_prof.set_ylabel("Depth (nm)", fontsize=10)
    ax_prof.invert_yaxis()
    ax_prof.set_title("Porosity depth profile", fontsize=10, fontweight="bold")
    ax_prof.legend(fontsize=8)
    ax_prof.grid(True, linewidth=0.5)

    # ── ROW 2 : n_eff per layer ───────────────────────────────────────────────
    ax_neff = fig.add_subplot(gs[2, 0])
    ax_neff.bar(x, neff["n_eff_layers"], color="#8da0cb", edgecolor="white", linewidth=0.8)
    ax_neff.axhline(neff["n_opt_ar"], color=C_AVG, linestyle="--", linewidth=1.4,
                    label=f"n_opt = {neff['n_opt_ar']:.3f}")
    ax_neff.set_xticks(x);  ax_neff.set_xticklabels(xlabels)
    ax_neff.set_ylabel(f"n_eff @ {neff['wl_ref_actual']:.0f} nm", fontsize=10)
    ax_neff.set_title("Effective refractive index", fontsize=10, fontweight="bold")
    ax_neff.legend(fontsize=8)
    ax_neff.grid(True, axis="y", linewidth=0.5)

    ax_keff = fig.add_subplot(gs[2, 1])
    ax_keff.bar(x, neff["k_eff_layers"], color="#fc8d62", edgecolor="white", linewidth=0.8)
    ax_keff.set_xticks(x);  ax_keff.set_xticklabels(xlabels)
    ax_keff.set_ylabel(f"k_eff @ {neff['wl_ref_actual']:.0f} nm", fontsize=10)
    ax_keff.set_title("Effective extinction coefficient", fontsize=10, fontweight="bold")
    ax_keff.grid(True, axis="y", linewidth=0.5)

    # ── ROW 2 col 2 : global summary text ────────────────────────────────────
    ax_sum = fig.add_subplot(gs[2, 2])
    ax_sum.axis("off")
    summary_lines = [
        ("Selected model",           f"N = {best_model.n_layers} sublayers"),
        ("RMSE",                     f"{best_result['rmse'] * 100:.4f} %R"),
        ("AIC",                      f"{best_result['aic']:.2f}"),
        ("BIC",                      f"{best_result['bic']:.2f}"),
        ("Total thickness",          f"{poro['d_total']:.2f} nm"),
        ("Avg porosity",             f"{poro['avg_porosity'] * 100:.2f} %"),
        ("Surface porosity",         f"{poro['f_surface'] * 100:.2f} %"),
        ("Bottom porosity",          f"{poro['f_bottom'] * 100:.2f} %"),
        ("n_opt (AR target)",        f"{neff['n_opt_ar']:.4f}"),
        ("n_eff surface",            f"{neff['n_eff_surface']:.4f}"),
        ("n_eff average",            f"{neff['n_eff_avg']:.4f}"),
        ("|n_eff(surf) − n_opt|",    f"{neff['delta_surface_to_optimum']:.4f}"),
        ("|n_eff(avg)  − n_opt|",    f"{neff['delta_avg_to_optimum']:.4f}"),
    ]
    dy = 1.0 / (len(summary_lines) + 1.5)
    ax_sum.text(0.05, 0.98, "Summary", fontsize=10, fontweight="bold",
                va="top", transform=ax_sum.transAxes, color="#1a1a1a")
    for k, (key, val) in enumerate(summary_lines):
        ypos = 0.98 - (k + 1.2) * dy
        ax_sum.text(0.05, ypos, key + " :", fontsize=8.5, va="top",
                    transform=ax_sum.transAxes, color="#333333")
        ax_sum.text(0.62, ypos, val, fontsize=8.5, va="top", fontweight="bold",
                    transform=ax_sum.transAxes, color="#1a5276")

    # ── ROW 3 : per-layer table ───────────────────────────────────────────────
    ax_tab = fig.add_subplot(gs[3, :])
    ax_tab.axis("off")

    col_labels = [
        "Layer", "Thickness (nm)", "Porosity (%)",
        f"n_eff @ {neff['wl_ref_actual']:.0f} nm",
        f"k_eff @ {neff['wl_ref_actual']:.0f} nm",
    ]
    table_data = [
        [
            f"L{i + 1}",
            f"{d_opt[i]:.2f}",
            f"{f_opt[i] * 100:.2f}",
            f"{neff['n_eff_layers'][i]:.4f}",
            f"{neff['k_eff_layers'][i]:.4f}",
        ]
        for i in range(best_model.n_layers)
    ]
    tbl = ax_tab.table(
        cellText=table_data, colLabels=col_labels,
        loc="center", bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#aaaaaa")
        if row == 0:
            cell.set_facecolor("#d0e4f7")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#f7fbff" if row % 2 == 0 else "white")

    fig.suptitle(
        f"Porous GaAs — {best_result['label']} etching  |  "
        "TMM + Bruggeman EMA  |  "
        f"Model selection: N = 2 … {max(n_values)}  |  Criterion: BIC",
        fontsize=12, fontweight="bold", y=0.96, color="#1a1a1a",
    )

    plt.savefig(output_png, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"-> Figure 2 (analysis) saved: {output_png}")
    plt.close(fig)


def plot_model_selection(all_results, best_result, output_png):
    """Wrapper kept for backward compatibility — delegates to the two split figures."""
    base = output_png.replace(".png", "")
    plot_fit_curve(best_result,           base + "_fit.png")
    plot_model_analysis(all_results, best_result, base + "_analysis.png")


# =============================================================================
# FIT ONE LAYER COUNT (parallel worker)
# =============================================================================

def fit_one_layer(args):
    """Parallel worker — must be at module level for multiprocessing (Windows spawn)."""
    label, csv_file, n_layers = args
    model  = ThinFilmModel(n_layers=n_layers, csv_exp=csv_file, label=label)
    engine = FittingEngine(model)
    params_opt, _ = engine.run()
    R_final    = model.simulate(params_opt)
    rmse_final = model.rmse(R_final)
    aic, bic   = model.aic_bic(R_final)
    return {
        "label":    label,
        "csv_file": csv_file,
        "n_layers": n_layers,
        "n_params": 2 * n_layers,
        "model":    model,
        "params":   params_opt,
        "R_final":  R_final,
        "rmse":     rmse_final,
        "aic":      aic,
        "bic":      bic,
        "history":  engine.history,
    }


def fit_one_dataset(label, csv_file, all_results):
    print("\n" + "#" * 90)
    print(f"# RESULTS — {label} | file: {csv_file}")
    print("#" * 90)

    all_results = sorted(all_results, key=lambda r: r["n_layers"])
    output_dir  = os.path.join(RESULTS_DIR, label)
    safe_mkdir(output_dir)

    best_bic, best_result = np.inf, None

    for res in all_results:
        poro = res["model"].porosity_metrics(res["params"])
        neff = res["model"].effective_index_metrics(res["params"])
        print(f"\n{'=' * 80}")
        print(f"{label} — N = {res['n_layers']} layers")
        print(f"  RMSE = {res['rmse'] * 100:.4f} %R  |  "
              f"AIC = {res['aic']:.4f}  |  BIC = {res['bic']:.4f}")
        print(f"  Avg porosity = {poro['avg_porosity'] * 100:.2f} %  |  "
              f"n_eff surface = {neff['n_eff_surface']:.4f}  |  "
              f"n_eff avg = {neff['n_eff_avg']:.4f}")
        if res["bic"] < best_bic:
            best_bic, best_result = res["bic"], res

    best_model = best_result["model"]
    best_poro  = best_model.porosity_metrics(best_result["params"])
    best_neff  = best_model.effective_index_metrics(best_result["params"])

    print(f"\n{'#' * 80}")
    print("SELECTED MODEL BY BIC")
    print(f"  N = {best_result['n_layers']} sublayers")
    print(f"  RMSE = {best_result['rmse'] * 100:.4f} %R  |  "
          f"AIC = {best_result['aic']:.4f}  |  BIC = {best_result['bic']:.4f}")
    print(f"  Total thickness = {best_poro['d_total']:.2f} nm")
    print(f"  Avg porosity    = {best_poro['avg_porosity'] * 100:.2f} %")
    print(f"  n_eff surface   = {best_neff['n_eff_surface']:.4f} @ {best_neff['wl_ref_actual']:.1f} nm")
    print(f"  n_eff average   = {best_neff['n_eff_avg']:.4f} @ {best_neff['wl_ref_actual']:.1f} nm")
    print(f"  Optimal AR n    = {best_neff['n_opt_ar']:.4f}")

    out_txt      = os.path.join(output_dir, f"model_selection_{label}.txt")
    out_png_fit  = os.path.join(output_dir, f"figure1_fit_{label}.png")
    out_png_ana  = os.path.join(output_dir, f"figure2_analysis_{label}.png")
    save_model_selection_results(all_results, best_result, out_txt)
    plot_fit_curve(best_result, out_png_fit)
    plot_model_analysis(all_results, best_result, out_png_ana)

    return best_result


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║ 25 min FIT - MODEL SELECTION AND EFFECTIVE INDEX             ║")
    print("║ TMM + Bruggeman EMA | DE + L-BFGS-B | Warm start from 35min ║")
    print("║ Criteria: RMSE + AIC + BIC                                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    safe_mkdir(RESULTS_DIR)

    tasks    = [(DATASET_LABEL, DATASET_FILE, n) for n in N_LAYERS_LIST]
    n_workers = min(len(tasks), os.cpu_count() or 4)
    print(f"Launching {len(tasks)} tasks on {n_workers} parallel processes...")
    print()

    collected = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(fit_one_layer, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                collected.append(result)
                print(
                    f"[DONE] {result['label']:>8} | "
                    f"N = {result['n_layers']} | "
                    f"RMSE = {result['rmse'] * 100:.4f} %R"
                )
            except Exception as exc:
                task = futures[future]
                print(f"[ERROR] {task}: {exc}")

    print()
    fit_one_dataset(DATASET_LABEL, DATASET_FILE, collected)

    print("\n✓ 25 min fitting completed successfully.\n")


if __name__ == "__main__":
    main()
