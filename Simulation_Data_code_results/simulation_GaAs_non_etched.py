"""
================================================================================
  BULK GaAs REFLECTANCE FITTING
  -------------------------------------------------------------------
  Optical method : Transfer Matrix Method (TMM)
  Structure      : Air / GaAsOx / Bulk GaAs
  Free parameter : GaAsOx thickness only

  Notes:
    - No porous layer is used for the bulk sample.
    - The initial simulation is used internally only.
    - The final figure displays only experimental data and optimized fit.
    - All outputs are written in English.
================================================================================
"""

import os
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

NK_GAAS = "Gaas041t.nk"
NK_OXIDE = "GaasOx.nk"          # Change this name if your oxide nk file is different
CSV_BULK = "R_bulk.Sample.csv" # Change this name if your bulk CSV file is different

FIT_MIN = 277.0
FIT_MAX = 800.0

# Oxide thickness bounds, in nm
D_OXIDE_INIT = 3.0
D_OXIDE_BOUNDS = (0.5, 10.0)

RESULTS_DIR = "results_bulk_gaasox"
OUTPUT_PNG_FIT      = os.path.join(RESULTS_DIR, "fit_bulk_gaasox_fit.png")
OUTPUT_PNG_ANALYSIS = os.path.join(RESULTS_DIR, "fit_bulk_gaasox_analysis.png")
OUTPUT_TXT = os.path.join(RESULTS_DIR, "results_bulk_gaasox.txt")


# =============================================================================
# UTILITIES
# =============================================================================

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_nk_file(filepath: str):
    """
    Read a WinElli-like .nk file.
    Expected first line: type E_min(eV) E_max(eV) N_points
    Following lines: n k
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, encoding="latin-1") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty file: {filepath}")

    header = lines[0].split()
    if len(header) < 3:
        raise ValueError(f"Invalid .nk header in file: {filepath}")

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
        raise ValueError(f"No valid n/k data found in file: {filepath}")

    energies = np.linspace(E_min, E_max, len(nk_data))
    return energies, nk_data[:, 0], nk_data[:, 1]


def load_csv_reflectance(filepath: str):
    """
    Read experimental reflectance CSV.
    Expected format:
      - separator ';'
      - decimal comma
      - two header lines
      - columns: wavelength (nm) ; reflectance (%)
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
                r = float(parts[1].replace(",", ".").strip()) / 100.0
                wavelengths.append(wl)
                reflectances.append(r)
            except ValueError:
                continue

    wl = np.array(wavelengths)
    R = np.array(reflectances)

    if len(wl) == 0:
        raise ValueError(f"No valid experimental data found in file: {filepath}")

    idx = np.argsort(wl)
    return wl[idx], R[idx]


# =============================================================================
# TMM AT NORMAL INCIDENCE
# =============================================================================

def tmm_reflectance(wavelengths_nm, layers):
    """
    Compute reflectance using Transfer Matrix Method at normal incidence.
    layers:
      first layer  = incident medium, semi-infinite
      middle       = finite layers
      last layer   = substrate, semi-infinite
    """
    R = np.empty(len(wavelengths_nm))

    for i, lam in enumerate(wavelengths_nm):
        M = np.eye(2, dtype=complex)

        # finite internal layers only
        for j in range(1, len(layers) - 1):
            nj = layers[j]["n"][i] + 1j * layers[j]["k"][i]
            dj = layers[j]["d"]

            delta = 2.0 * np.pi * nj * dj / lam

            cos_d = np.cos(delta)
            sin_d = np.sin(delta)

            M_j = np.array([
                [cos_d, -1j * sin_d / nj],
                [-1j * nj * sin_d, cos_d]
            ])

            M = M @ M_j

        n0 = layers[0]["n"][i] + 1j * layers[0]["k"][i]
        ns = layers[-1]["n"][i] + 1j * layers[-1]["k"][i]

        m11, m12, m21, m22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

        num = (m11 + m12 * ns) * n0 - (m21 + m22 * ns)
        den = (m11 + m12 * ns) * n0 + (m21 + m22 * ns)

        R[i] = np.abs(num / den) ** 2

    return R


# =============================================================================
# BULK MODEL: AIR / GaAsOx / GaAs
# =============================================================================

class BulkOxideModel:
    def __init__(self):
        print("[1/3] Loading optical constants...")

        gaas_e, gaas_n, gaas_k = load_nk_file(NK_GAAS)
        oxide_e, oxide_n, oxide_k = load_nk_file(NK_OXIDE)

        print("[2/3] Loading experimental bulk reflectance...")

        wl_exp, R_exp = load_csv_reflectance(CSV_BULK)

        lam_gaas = 1239.84 / gaas_e
        lam_oxide = 1239.84 / oxide_e

        idx_g = np.argsort(lam_gaas)
        lam_gaas = lam_gaas[idx_g]
        gaas_n = gaas_n[idx_g]
        gaas_k = gaas_k[idx_g]

        idx_o = np.argsort(lam_oxide)
        lam_oxide = lam_oxide[idx_o]
        oxide_n = oxide_n[idx_o]
        oxide_k = oxide_k[idx_o]

        lam_min = max(wl_exp.min(), lam_gaas.min(), lam_oxide.min(), FIT_MIN)
        lam_max = min(wl_exp.max(), lam_gaas.max(), lam_oxide.max(), FIT_MAX)

        mask = (wl_exp >= lam_min) & (wl_exp <= lam_max)

        self.wl = wl_exp[mask]
        self.R_exp = R_exp[mask]
        self.N = len(self.wl)

        if self.N == 0:
            raise ValueError(
                f"No experimental point found in the fitting range "
                f"{FIT_MIN:.1f}–{FIT_MAX:.1f} nm."
            )

        print(f"Fitting wavelength range: {self.wl.min():.1f} - {self.wl.max():.1f} nm")
        print(f"Number of fitted points: {self.N}")

        self.n_gaas = interp1d(
            lam_gaas,
            gaas_n,
            kind="linear",
            fill_value="extrapolate"
        )(self.wl)

        self.k_gaas = interp1d(
            lam_gaas,
            gaas_k,
            kind="linear",
            fill_value="extrapolate"
        )(self.wl)

        self.n_oxide = interp1d(
            lam_oxide,
            oxide_n,
            kind="linear",
            fill_value="extrapolate"
        )(self.wl)

        self.k_oxide = interp1d(
            lam_oxide,
            oxide_k,
            kind="linear",
            fill_value="extrapolate"
        )(self.wl)

        self.n_air = np.ones(self.N)
        self.k_air = np.zeros(self.N)

    def clip_param(self, d_oxide):
        return float(np.clip(d_oxide, D_OXIDE_BOUNDS[0], D_OXIDE_BOUNDS[1]))

    def simulate(self, d_oxide):
        d_oxide = self.clip_param(d_oxide)

        layers = [
            {"n": self.n_air, "k": self.k_air, "d": 0.0},          # Air / Void
            {"n": self.n_oxide, "k": self.k_oxide, "d": d_oxide}, # GaAsOx
            {"n": self.n_gaas, "k": self.k_gaas, "d": 0.0},       # Bulk GaAs
        ]

        return tmm_reflectance(self.wl, layers)

    def rmse(self, R_sim):
        return float(np.sqrt(np.mean((R_sim - self.R_exp) ** 2)))

    def rss(self, R_sim):
        return float(np.sum((R_sim - self.R_exp) ** 2))

    def aic_bic(self, R_sim):
        n = len(self.R_exp)
        k = 1  # only one free parameter: oxide thickness
        rss = max(self.rss(R_sim), 1e-30)

        aic = 2 * k + n * np.log(rss / n)
        bic = k * np.log(n) + n * np.log(rss / n)

        return float(aic), float(bic)

    def objective(self, x):
        try:
            d_oxide = self.clip_param(float(np.atleast_1d(x)[0]))
            R_sim = self.simulate(d_oxide)
            return self.rmse(R_sim)
        except Exception:
            return 1e6


# =============================================================================
# FITTING
# =============================================================================

class FittingEngine:
    def __init__(self, model: BulkOxideModel):
        self.model = model
        self.history = []
        self._iter = 0

    def _callback_de(self, xk, convergence):
        rmse = self.model.objective(xk) * 100
        self._iter += 1
        self.history.append((self._iter, rmse))

    def _callback_lbfgsb(self, xk):
        rmse = self.model.objective(xk) * 100
        self._iter += 1
        self.history.append((self._iter, rmse))

    def run(self):
        print("[3/3] Fitting GaAsOx thickness...")

        R_start = self.model.simulate(D_OXIDE_INIT)
        rmse_start = self.model.rmse(R_start)

        print(f"Initial oxide thickness: {D_OXIDE_INIT:.3f} nm")
        print(f"Initial RMSE: {rmse_start * 100:.4f} %R")

        t0 = time.time()
        self._iter = 0
        self.history = []

        result_de = differential_evolution(
            self.model.objective,
            bounds=[D_OXIDE_BOUNDS],
            seed=42,
            maxiter=120,
            tol=1e-8,
            popsize=20,
            polish=False,
            workers=1,
            callback=self._callback_de
        )

        result_local = minimize(
            self.model.objective,
            x0=result_de.x,
            method="L-BFGS-B",
            bounds=[D_OXIDE_BOUNDS],
            callback=self._callback_lbfgsb,
            options={
                "maxiter": 2000,
                "ftol": 1e-14,
                "gtol": 1e-10,
                "disp": False
            }
        )

        d_opt = self.model.clip_param(result_local.x[0])
        R_opt = self.model.simulate(d_opt)
        rmse_opt = self.model.rmse(R_opt)

        print(f"Optimized oxide thickness: {d_opt:.4f} nm")
        print(f"Final RMSE: {rmse_opt * 100:.4f} %R")
        print(f"Elapsed time: {time.time() - t0:.1f} s")

        return d_opt, R_start, R_opt, rmse_start, rmse_opt


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(model, d_opt, R_opt, rmse_opt):
    safe_mkdir(RESULTS_DIR)

    aic, bic = model.aic_bic(R_opt)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("BULK GaAs REFLECTANCE FITTING\n")
        f.write("=" * 60 + "\n\n")
        f.write("Structure: Air / GaAsOx / Bulk GaAs\n")
        f.write(f"Experimental file: {CSV_BULK}\n")
        f.write(f"GaAs nk file: {NK_GAAS}\n")
        f.write(f"GaAsOx nk file: {NK_OXIDE}\n")
        f.write(f"Fitting range: {model.wl.min():.1f} - {model.wl.max():.1f} nm\n\n")

        f.write("OPTIMIZED PARAMETER\n")
        f.write("-" * 60 + "\n")
        f.write(f"GaAsOx thickness = {d_opt:.6f} nm\n\n")

        f.write("FIT QUALITY\n")
        f.write("-" * 60 + "\n")
        f.write(f"RMSE final = {rmse_opt * 100:.6f} %R\n")
        f.write(f"AIC = {aic:.6f}\n")
        f.write(f"BIC = {bic:.6f}\n")
        f.write(f"Number of free parameters = 1\n\n")

        f.write("SPECTRUM\n")
        f.write("wavelength_nm    R_fit_percent    R_exp_percent    residual_percent\n")
        f.write("-" * 75 + "\n")

        residuals = (R_opt - model.R_exp) * 100
        for wl, rf, re, rr in zip(model.wl, R_opt, model.R_exp, residuals):
            f.write(f"{wl:12.4f} {rf * 100:16.6f} {re * 100:16.6f} {rr:18.6f}\n")

    print(f"Results saved: {OUTPUT_TXT}")


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
C_EXP  = "#2166ac"
C_FIT  = "#d6604d"
C_CONV = "#4dac26"
C_NK1  = "#1b7837"
C_NK2  = "#d73027"


def plot_fit(model, d_opt, R_opt, rmse_opt):
    """Figure 1 — reflectance fit only."""
    safe_mkdir(RESULTS_DIR)
    plt.rcParams.update(_RC)

    aic, bic = model.aic_bic(R_opt)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    ax.plot(model.wl, model.R_exp * 100,
            "o", color=C_EXP, markersize=3, alpha=0.7, label="Experimental bulk")
    ax.plot(model.wl, R_opt * 100,
            "-", color=C_FIT, linewidth=2.2,
            label=(f"Optimized fit  |  d_GaAsOx = {d_opt:.4f} nm"
                   f"   |   RMSE = {rmse_opt * 100:.4f} %R"
                   f"   |   AIC = {aic:.2f}   |   BIC = {bic:.2f}"))
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Reflectance (%R)", fontsize=11)
    ax.set_title(
        "Bulk GaAs  |  TMM  |  Air / GaAsOx / Bulk GaAs",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linewidth=0.6)
    ax.set_xlim([FIT_MIN, FIT_MAX])

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_FIT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Figure 1 (fit) saved: {OUTPUT_PNG_FIT}")
    plt.show()


def plot_analysis(model, d_opt, R_opt, rmse_opt, history):
    """Figure 2 — optical constants, convergence, table and summary."""
    safe_mkdir(RESULTS_DIR)
    plt.rcParams.update(_RC)

    aic, bic = model.aic_bic(R_opt)

    REF_WL      = 550.0
    idx_ref     = int(np.argmin(np.abs(model.wl - REF_WL)))
    wl_actual   = float(model.wl[idx_ref])
    n_gaas_ref  = float(model.n_gaas[idx_ref])
    k_gaas_ref  = float(model.k_gaas[idx_ref])
    n_oxide_ref = float(model.n_oxide[idx_ref])
    k_oxide_ref = float(model.k_oxide[idx_ref])
    n_opt_ar    = float(np.sqrt(n_gaas_ref))

    fig = plt.figure(figsize=(17, 9))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[2.2, 1.2],
        hspace=0.45, wspace=0.32,
        top=0.91, bottom=0.05,
        left=0.06, right=0.97,
    )

    # ── ROW 0 : GaAsOx n&k | convergence | GaAs n&k ──────────────────────────
    ax_nk_ox = fig.add_subplot(gs[0, 0])
    ax_nk_ox.plot(model.wl, model.n_oxide, color=C_NK1, linewidth=1.8, label="n (GaAsOx)")
    ax_nk_ox.plot(model.wl, model.k_oxide, color=C_NK2, linewidth=1.8,
                  linestyle="--", label="k (GaAsOx)")
    ax_nk_ox.set_xlabel("Wavelength (nm)", fontsize=10)
    ax_nk_ox.set_ylabel("Optical constants", fontsize=10)
    ax_nk_ox.set_title("GaAsOx optical constants  n & k",
                       fontsize=10, fontweight="bold")
    ax_nk_ox.legend(fontsize=8)
    ax_nk_ox.grid(True, linewidth=0.5)
    ax_nk_ox.set_xlim([FIT_MIN, FIT_MAX])

    ax_conv = fig.add_subplot(gs[0, 1])
    if history:
        iters, rmses = zip(*history)
        ax_conv.semilogy(iters, rmses, "-", color=C_CONV, linewidth=1.6)
        ax_conv.axhline(rmse_opt * 100, color=C_FIT, linewidth=1.2,
                        linestyle="--", label=f"Final RMSE = {rmse_opt * 100:.4f} %R")
        ax_conv.legend(fontsize=8)
    ax_conv.set_xlabel("Iteration", fontsize=10)
    ax_conv.set_ylabel("RMSE (%R)  [log]", fontsize=10)
    ax_conv.set_title("Fitting convergence", fontsize=10, fontweight="bold")
    ax_conv.grid(True, linewidth=0.5, which="both")

    ax_nk_gaas = fig.add_subplot(gs[0, 2])
    ax_nk_gaas.plot(model.wl, model.n_gaas, color=C_NK1, linewidth=1.8, label="n (GaAs)")
    ax_nk_gaas.plot(model.wl, model.k_gaas, color=C_NK2, linewidth=1.8,
                    linestyle="--", label="k (GaAs)")
    ax_nk_gaas.set_xlabel("Wavelength (nm)", fontsize=10)
    ax_nk_gaas.set_ylabel("Optical constants", fontsize=10)
    ax_nk_gaas.set_title("GaAs optical constants  n & k",
                          fontsize=10, fontweight="bold")
    ax_nk_gaas.legend(fontsize=8)
    ax_nk_gaas.grid(True, linewidth=0.5)
    ax_nk_gaas.set_xlim([FIT_MIN, FIT_MAX])

    # ── ROW 1 : layer table (left) | summary (right) ─────────────────────────
    ax_tab = fig.add_subplot(gs[1, :])
    ax_tab.axis("off")

    col_labels = [
        "Layer", "Material", "Thickness (nm)",
        f"n @ {wl_actual:.0f} nm",
        f"k @ {wl_actual:.0f} nm",
    ]
    table_data = [
        ["0 (incident)",  "Air",       "∞ (semi-inf.)", "1.0000",               "0.0000"],
        ["1 (finite)",    "GaAsOx",    f"{d_opt:.4f}",  f"{n_oxide_ref:.4f}",   f"{k_oxide_ref:.4f}"],
        ["2 (substrate)", "Bulk GaAs", "∞ (semi-inf.)", f"{n_gaas_ref:.4f}",    f"{k_gaas_ref:.4f}"],
    ]
    tbl = ax_tab.table(
        cellText=table_data, colLabels=col_labels,
        loc="left", bbox=[0.0, 0.0, 0.50, 1.0],
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

    summary_lines = [
        ("Structure",                        "Air / GaAsOx / Bulk GaAs"),
        ("RMSE total",                       f"{rmse_opt * 100:.4f} %R"),
        ("AIC",                              f"{aic:.2f}"),
        ("BIC",                              f"{bic:.2f}"),
        ("d_GaAsOx (free)",                  f"{d_opt:.4f} nm"),
        ("n_opt (AR target)",                f"{n_opt_ar:.4f}"),
        (f"n_GaAsOx @ {wl_actual:.0f} nm",  f"{n_oxide_ref:.4f}"),
        (f"k_GaAsOx @ {wl_actual:.0f} nm",  f"{k_oxide_ref:.4f}"),
        (f"n_GaAs @ {wl_actual:.0f} nm",    f"{n_gaas_ref:.4f}"),
        (f"k_GaAs @ {wl_actual:.0f} nm",    f"{k_gaas_ref:.4f}"),
        ("|n_GaAsOx − n_opt|",              f"{abs(n_oxide_ref - n_opt_ar):.4f}"),
        ("Fitting points",                   f"{model.N}"),
    ]
    x0 = 0.54
    dy = 1.0 / (len(summary_lines) + 1.5)
    ax_tab.text(x0, 0.98, "Summary", fontsize=10, fontweight="bold",
                va="top", transform=ax_tab.transAxes, color="#1a1a1a")
    for k, (key, val) in enumerate(summary_lines):
        ypos = 0.98 - (k + 1.2) * dy
        ax_tab.text(x0,        ypos, key + " :", fontsize=8.5, va="top",
                    transform=ax_tab.transAxes, color="#333333")
        ax_tab.text(x0 + 0.28, ypos, val,        fontsize=8.5, va="top",
                    fontweight="bold", transform=ax_tab.transAxes, color="#1a5276")

    fig.suptitle(
        "Bulk GaAs  |  TMM  |  Air / GaAsOx / Bulk GaAs  |  Optical analysis",
        fontsize=12, fontweight="bold", y=0.96, color="#1a1a1a",
    )

    plt.savefig(OUTPUT_PNG_ANALYSIS, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Figure 2 (analysis) saved: {OUTPUT_PNG_ANALYSIS}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║ BULK GaAs FITTING — Air / GaAsOx / GaAs                     ║")
    print("║ Free parameter: GaAsOx thickness                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    safe_mkdir(RESULTS_DIR)

    model = BulkOxideModel()
    engine = FittingEngine(model)

    d_opt, R_start, R_opt, rmse_start, rmse_opt = engine.run()

    save_results(model, d_opt, R_opt, rmse_opt)
    plot_fit(model, d_opt, R_opt, rmse_opt)
    plot_analysis(model, d_opt, R_opt, rmse_opt, engine.history)

    print("\n✓ Bulk fit completed successfully!\n")


if __name__ == "__main__":
    main()
