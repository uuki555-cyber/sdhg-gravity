"""
Generate figures for the README and for quick visual inspection.

Produces:
  figures/fig1_rar_comparison.png  — RAR with McGaugh vs p(M) overlaid
  figures/fig2_p_vs_mass.png      — p as a function of baryonic mass
  figures/fig3_global_fit.png     — Global fit model comparison bar chart

Usage:
    pip install matplotlib
    python make_figures.py
"""
import numpy as np
import os
from scipy.optimize import minimize_scalar, minimize
from sdhg import load_sparc, load_clusters, A0, G, MSUN, KPC

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required for figures: pip install matplotlib")
    raise SystemExit(1)

os.makedirs("figures", exist_ok=True)

# ============================================================
# Load data
# ============================================================
galaxies = load_sparc()
clusters = load_clusters()

# Collect all (x, mu, M) points
all_x, all_mu, all_M = [], [], []
for gid, pts in galaxies.items():
    if len(pts) < 5:
        continue
    V_last = pts[-1][1] * 1e3
    R_last = pts[-1][0] * KPC
    M_est = 0.5 * V_last ** 2 * R_last / G / MSUN
    for R_kpc, Vo, eV, Vd, Vg, Vb in pts:
        R = R_kpc * KPC
        go = (Vo * 1e3) ** 2 / R
        gb = (0.5 * (Vd * 1e3) ** 2 + np.sign(Vg) * (Vg * 1e3) ** 2
              + 0.7 * (Vb * 1e3) ** 2) / R
        if gb > 0 and go > 0:
            all_x.append(gb / A0)
            all_mu.append(gb / go)
            all_M.append(M_est)

all_x = np.array(all_x)
all_mu = np.array(all_mu)
all_M = np.array(all_M)

# ============================================================
# Figure 1: RAR — data + McGaugh + p(M) curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Data points colored by mass
logM = np.log10(all_M)
sc = ax.scatter(np.log10(all_x), np.log10(all_mu),
                c=logM, cmap="viridis", s=1, alpha=0.3, rasterized=True)
cbar = plt.colorbar(sc, ax=ax, label="log$_{10}$(M$_{bar}$ / M$_\\odot$)")

# McGaugh curve
x_line = np.logspace(-3, 2, 500)
mu_mcg = 1 - np.exp(-x_line ** 0.5)
ax.plot(np.log10(x_line), np.log10(mu_mcg), "r-", lw=2, label="McGaugh (p=0.5)")

# p(M) curves for different masses
M0 = 10 ** 10.17  # from global fit
alpha = 0.312
for M_label, color in [(1e8, "cyan"), (1e10, "orange"), (1e14, "magenta")]:
    u = (M_label / M0) ** alpha
    p = 2 * u / (1 + 3 * u)
    mu_p = 1 - np.exp(-x_line ** p)
    ax.plot(np.log10(x_line), np.log10(mu_p), "--", color=color, lw=1.5,
            label=f"p(M=10$^{{{int(np.log10(M_label))}}}$) = {p:.2f}")

# Cluster points
for cl in clusters:
    gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
    x_cl = gb / A0
    mu_cl = cl["M_bar_sun"] / cl["M500_sun"]
    ax.plot(np.log10(x_cl), np.log10(mu_cl), "r^", ms=8, mew=1.5,
            mfc="none", zorder=5)

ax.plot([], [], "r^", ms=8, mew=1.5, mfc="none", label="Galaxy clusters")

ax.set_xlabel("log$_{10}$(g$_{bar}$ / a$_0$)", fontsize=13)
ax.set_ylabel("log$_{10}$($\\mu$)", fontsize=13)
ax.set_xlim(-3, 2)
ax.set_ylim(-1.8, 0.3)
ax.legend(loc="lower right", fontsize=9)
ax.set_title("Radial Acceleration Relation", fontsize=14)
fig.tight_layout()
fig.savefig("figures/fig1_rar_comparison.png", dpi=150)
print("Saved figures/fig1_rar_comparison.png")
plt.close()

# ============================================================
# Figure 2: p vs baryonic mass
# ============================================================
# Fit p per galaxy
gal_p, gal_M = [], []
for gid, pts in galaxies.items():
    if len(pts) < 8:
        continue
    xs, mus = [], []
    for R_kpc, Vo, eV, Vd, Vg, Vb in pts:
        R = R_kpc * KPC
        go = (Vo * 1e3) ** 2 / R
        gb = (0.5 * (Vd * 1e3) ** 2 + np.sign(Vg) * (Vg * 1e3) ** 2
              + 0.7 * (Vb * 1e3) ** 2) / R
        if gb > 0 and go > 0:
            xs.append(gb / A0)
            mus.append(gb / go)
    if len(xs) < 8:
        continue
    xs = np.array(xs)
    mus = np.array(mus)

    def rms_p(p):
        pred = 1 - np.exp(-np.maximum(xs, 1e-20) ** p)
        return np.sqrt(np.mean(
            (np.log10(np.maximum(mus, 1e-20)) -
             np.log10(np.maximum(pred, 1e-20))) ** 2))

    r = minimize_scalar(rms_p, bounds=(0.01, 0.95), method="bounded")
    V_last = pts[-1][1] * 1e3
    R_last = pts[-1][0] * KPC
    M = 0.5 * V_last ** 2 * R_last / G / MSUN
    gal_p.append(r.x)
    gal_M.append(M)

gal_p = np.array(gal_p)
gal_M = np.array(gal_M)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(np.log10(gal_M), gal_p, s=20, alpha=0.5, c="steelblue", label="SPARC galaxies")

# Cluster points
for cl in clusters:
    gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
    x_cl = gb / A0
    u = (cl["M500_sun"] / M0) ** alpha
    p_cl = 2 * u / (1 + 3 * u)
    # "observed" p for cluster: solve mu_need = 1-exp(-x^p) for p
    mu_need = cl["M_bar_sun"] / cl["M500_sun"]
    if 0 < mu_need < 1 and x_cl > 0:
        p_obs = np.log(-np.log(1 - mu_need)) / np.log(x_cl)
        ax.plot(np.log10(cl["M500_sun"]), p_obs, "r^", ms=10, mew=1.5, mfc="none", zorder=5)

ax.plot([], [], "r^", ms=10, mew=1.5, mfc="none", label="Galaxy clusters")

# Theory curve
M_line = np.logspace(7, 16, 200)
u_line = (M_line / M0) ** alpha
p_line = 2 * u_line / (1 + 3 * u_line)
ax.plot(np.log10(M_line), p_line, "k-", lw=2, label=f"p(M), M$_0$=10$^{{{np.log10(M0):.1f}}}$")

ax.axhline(0.5, color="red", ls=":", lw=1, alpha=0.7, label="McGaugh (p=0.5)")
ax.axhline(2 / 3, color="gray", ls=":", lw=1, alpha=0.5, label="p = 2/3 limit")

ax.set_xlabel("log$_{10}$(M / M$_\\odot$)", fontsize=13)
ax.set_ylabel("Fitted p", fontsize=13)
ax.set_ylim(-0.05, 1.0)
ax.set_xlim(7, 16)
ax.legend(loc="lower right", fontsize=9)
ax.set_title("RAR exponent vs. baryonic mass", fontsize=14)
fig.tight_layout()
fig.savefig("figures/fig2_p_vs_mass.png", dpi=150)
print("Saved figures/fig2_p_vs_mass.png")
plt.close()

# ============================================================
# Figure 3: Global fit comparison (bar chart)
# ============================================================
# Run global fit
gal_data = []
for gid, pts in galaxies.items():
    if len(pts) < 5:
        continue
    V_last = pts[-1][1] * 1e3
    R_last = pts[-1][0] * KPC
    M_est = 0.5 * V_last ** 2 * R_last / G / MSUN
    for R_kpc, Vo, eV, Vd, Vg, Vb in pts:
        gal_data.append(((Vd * 1e3), (Vg * 1e3), (Vb * 1e3),
                         R_kpc * KPC, Vo * 1e3, M_est))


def score_pM(Y, p_fn):
    s, n = 0.0, 0
    for Vd, Vg, Vb, R, Vo, M in gal_data:
        go = Vo ** 2 / R
        gb = (Y * Vd ** 2 + np.sign(Vg) * Vg ** 2 + 0.7 * Vb ** 2) / R
        if gb > 0 and go > 0:
            p = p_fn(M)
            x = gb / A0
            mu = max(1 - np.exp(-max(x, 1e-20) ** p), 1e-20)
            s += (np.log10(go) - np.log10(gb / mu)) ** 2
            n += 1
    for cl in clusters:
        gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
        p = p_fn(cl["M500_sun"])
        x = gb / A0
        mu = max(1 - np.exp(-max(x, 1e-20) ** p), 1e-20)
        mu_need = cl["M_bar_sun"] / cl["M500_sun"]
        s += 50 * (np.log10(mu_need) - np.log10(mu)) ** 2
        n += 50
    return np.sqrt(s / n)


rC = minimize(lambda p: score_pM(p[0], lambda M: 0.5),
              [0.5], bounds=[(0.1, 3)], method="L-BFGS-B")

rA = minimize(lambda p: score_pM(p[0], lambda M, M0=10 ** p[1], a=p[2]:
              2 * (max(M, 1) / M0) ** a / (1 + 3 * (max(M, 1) / M0) ** a)),
              [0.5, 10.5, 0.33], bounds=[(0.1, 3), (8, 13), (0.1, 0.6)],
              method="L-BFGS-B")


def score_varG(params):
    Y, logM0, beta = params
    M0v = 10 ** logM0
    s, n = 0.0, 0
    for Vd, Vg, Vb, R, Vo, M in gal_data:
        go = Vo ** 2 / R
        gb = (Y * Vd ** 2 + np.sign(Vg) * Vg ** 2 + 0.7 * Vb ** 2) / R
        gbe = gb * (max(M, 1) / M0v) ** beta
        if gbe > 0 and go > 0:
            x = gbe / A0
            mu = max(1 - np.exp(-x ** 0.5), 1e-20)
            s += (np.log10(go) - np.log10(gbe / mu)) ** 2
            n += 1
    for cl in clusters:
        gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
        gbe = gb * (cl["M500_sun"] / M0v) ** beta
        x = gbe / A0
        mu = max(1 - np.exp(-max(x, 1e-20) ** 0.5), 1e-20)
        mn = cl["M_bar_sun"] / cl["M500_sun"]
        s += 50 * (np.log10(mn) - np.log10(mu)) ** 2
        n += 50
    return np.sqrt(s / n)


rB = minimize(score_varG, [0.5, 10.5, 0.05],
              bounds=[(0.1, 3), (8, 13), (-0.3, 0.3)], method="L-BFGS-B")

models = ["McGaugh\n(p=0.5)", "Variable G(M)\n[control]", "p(M)\nmodel"]
rms_vals = [rC.fun, rB.fun, rA.fun]
improvements = [0, (rC.fun - rB.fun) / rC.fun * 100,
                (rC.fun - rA.fun) / rC.fun * 100]
colors = ["#cc4444", "#888888", "#2266aa"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(models, rms_vals, color=colors, width=0.5, edgecolor="black", lw=0.8)

for bar, imp in zip(bars, improvements):
    if imp != 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{imp:+.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Unified RMS (galaxies + clusters)", fontsize=12)
ax.set_title("Global fit comparison\n(no per-galaxy fitting, same #params for G(M) and p(M))",
             fontsize=12)
ax.set_ylim(0, max(rms_vals) * 1.15)
fig.tight_layout()
fig.savefig("figures/fig3_global_fit.png", dpi=150)
print("Saved figures/fig3_global_fit.png")
plt.close()

print("\nAll figures generated.")
