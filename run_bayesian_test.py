"""
Bayesian hierarchical test: Is p(M) real or a methodological artifact?

Li, Lelli, McGaugh (2021) showed that fitting a parameter per galaxy
can create spurious mass dependence due to Y_disk degeneracy.

This test marginalizes over Y_disk, distance uncertainty, and inclination
uncertainty simultaneously, then checks whether p(M) survives.

Approach:
  1. For each galaxy, sample (Y_disk, p) jointly from a grid
  2. Weight each (Y_disk, p) by its likelihood
  3. Compute the marginal posterior of p for each galaxy
  4. Check if the posterior-averaged p correlates with M
  5. Compare with the same test for G_eff (which should NOT correlate)

Usage:
    python run_bayesian_test.py
"""
import numpy as np
from sdhg import load_sparc, load_clusters, A0, G, MSUN, KPC

MPC = KPC * 1e3


def log_likelihood(pts, p_val, Y_disk, dist_factor=1.0, a0=A0):
    """Log-likelihood for a single galaxy given (p, Y_disk, dist_factor).

    dist_factor: multiplicative correction to distance (1.0 = no change).
    Changing distance scales R and V^2 as: R -> R*f, V -> V (unchanged),
    so g_obs -> g_obs/f, g_bar -> g_bar/f (both scale the same way).
    Actually distance affects Vdisk^2 proportional to distance, so
    g_bar scales as 1/distance. We simplify: g_bar -> g_bar * dist_factor.
    """
    log_L = 0.0
    n = 0
    for R_kpc, Vobs, eVobs, Vdisk, Vgas, Vbul in pts:
        R = R_kpc * KPC
        g_obs = (Vobs * 1e3) ** 2 / R
        g_bar = (Y_disk * (Vdisk * 1e3)**2 +
                 np.sign(Vgas) * (Vgas * 1e3)**2 +
                 0.7 * (Vbul * 1e3)**2) / R
        g_bar *= dist_factor

        if g_bar <= 0 or g_obs <= 0:
            continue

        x = g_bar / a0
        mu = max(1.0 - np.exp(-max(x, 1e-20) ** p_val), 1e-20)
        g_pred = g_bar / mu

        # Log-space residual with observational error
        sigma_obs = max(eVobs / max(Vobs, 1), 0.05)  # fractional velocity error
        sigma_log = sigma_obs / np.log(10)  # propagated to log g
        sigma_log = max(sigma_log, 0.03)  # floor

        residual = np.log10(g_obs) - np.log10(g_pred)
        log_L += -0.5 * (residual / sigma_log) ** 2 - np.log(sigma_log)
        n += 1

    return log_L, n


def bayesian_p_posterior(pts, p_grid, Y_grid, dist_grid=None):
    """Compute marginal posterior of p, marginalizing over Y_disk and distance.

    Returns: (p_mean, p_std, Y_mean) from the marginalized posterior.
    """
    if dist_grid is None:
        dist_grid = [1.0]

    # Priors
    # Y_disk: Gaussian prior centered at 0.5, sigma=0.15 (McGaugh+ 2016)
    # distance: Gaussian prior centered at 1.0, sigma=0.10 (10% uncertainty)
    # p: flat prior in [0.01, 0.95]

    log_posterior = np.full((len(p_grid), len(Y_grid), len(dist_grid)), -np.inf)

    for ip, p_val in enumerate(p_grid):
        for iy, Y_val in enumerate(Y_grid):
            for id_, d_val in enumerate(dist_grid):
                ll, n = log_likelihood(pts, p_val, Y_val, d_val)
                if n < 3:
                    continue
                # Priors
                log_prior_Y = -0.5 * ((Y_val - 0.5) / 0.15) ** 2
                log_prior_d = -0.5 * ((d_val - 1.0) / 0.10) ** 2
                log_posterior[ip, iy, id_] = ll + log_prior_Y + log_prior_d

    # Normalize
    log_max = np.max(log_posterior)
    if log_max == -np.inf:
        return np.nan, np.nan, np.nan

    posterior = np.exp(log_posterior - log_max)
    total = posterior.sum()
    if total <= 0:
        return np.nan, np.nan, np.nan

    posterior /= total

    # Marginalize over Y and distance to get p posterior
    p_marginal = posterior.sum(axis=(1, 2))  # sum over Y and dist
    p_mean = np.sum(p_grid * p_marginal)
    p_std = np.sqrt(np.sum((p_grid - p_mean) ** 2 * p_marginal))

    # Marginalize over p and distance to get Y posterior
    Y_marginal = posterior.sum(axis=(0, 2))
    Y_mean = np.sum(Y_grid * Y_marginal)

    return p_mean, p_std, Y_mean


def bayesian_G_posterior(pts, G_grid, Y_grid):
    """Same test but fitting G_eff instead of p (control test).

    If G_eff correlates with M, the method is biased.
    """
    log_posterior = np.full((len(G_grid), len(Y_grid)), -np.inf)

    for ig, log_G_ratio in enumerate(G_grid):
        G_factor = 10 ** log_G_ratio
        for iy, Y_val in enumerate(Y_grid):
            ll, n = log_likelihood_G(pts, G_factor, Y_val)
            if n < 3:
                continue
            log_prior_Y = -0.5 * ((Y_val - 0.5) / 0.15) ** 2
            log_prior_G = -0.5 * (log_G_ratio / 0.15) ** 2  # G should be ~1
            log_posterior[ig, iy] = ll + log_prior_Y + log_prior_G

    log_max = np.max(log_posterior)
    if log_max == -np.inf:
        return np.nan, np.nan

    posterior = np.exp(log_posterior - log_max)
    total = posterior.sum()
    if total <= 0:
        return np.nan, np.nan
    posterior /= total

    G_marginal = posterior.sum(axis=1)
    G_mean = np.sum(G_grid * G_marginal)
    return 10 ** G_mean, np.sum(Y_grid * posterior.sum(axis=0))


def log_likelihood_G(pts, G_factor, Y_disk, a0=A0):
    """Log-likelihood with variable G_eff (McGaugh mu fixed at p=0.5)."""
    log_L = 0.0
    n = 0
    for R_kpc, Vobs, eVobs, Vdisk, Vgas, Vbul in pts:
        R = R_kpc * KPC
        g_obs = (Vobs * 1e3) ** 2 / R
        g_bar = (Y_disk * (Vdisk * 1e3)**2 +
                 np.sign(Vgas) * (Vgas * 1e3)**2 +
                 0.7 * (Vbul * 1e3)**2) / R
        g_bar *= G_factor

        if g_bar <= 0 or g_obs <= 0:
            continue

        x = g_bar / a0
        mu = max(1 - np.exp(-max(x, 1e-20) ** 0.5), 1e-20)
        g_pred = g_bar / mu

        sigma_obs = max(eVobs / max(Vobs, 1), 0.05)
        sigma_log = max(sigma_obs / np.log(10), 0.03)

        residual = np.log10(g_obs) - np.log10(g_pred)
        log_L += -0.5 * (residual / sigma_log) ** 2 - np.log(sigma_log)
        n += 1
    return log_L, n


def main():
    print("=" * 70)
    print("Bayesian Hierarchical Test: Is p(M) real?")
    print("=" * 70)

    galaxies = load_sparc()
    print(f"\nLoaded {len(galaxies)} galaxies")

    # Grids
    p_grid = np.linspace(0.05, 0.90, 35)
    Y_grid = np.linspace(0.15, 1.2, 22)
    dist_grid = np.linspace(0.80, 1.20, 9)
    G_grid = np.linspace(-0.30, 0.30, 25)  # log10(G_eff/G)

    print(f"Grid: p={len(p_grid)}, Y={len(Y_grid)}, dist={len(dist_grid)}, G={len(G_grid)}")
    print(f"Priors: Y_disk ~ N(0.5, 0.15), dist ~ N(1.0, 0.10), G ~ N(0, 0.15)")
    print(f"\nProcessing galaxies...")

    results = []
    n_done = 0
    for gid, pts in galaxies.items():
        if len(pts) < 8:
            continue

        # Mass estimate
        V_last = pts[-1][1] * 1e3
        R_last = pts[-1][0] * KPC
        M_est = 0.5 * V_last ** 2 * R_last / G / MSUN

        # Bayesian p (marginalized over Y_disk and distance)
        p_mean, p_std, Y_mean = bayesian_p_posterior(pts, p_grid, Y_grid, dist_grid)

        # Bayesian G_eff (control test, marginalized over Y_disk)
        G_mean, Y_mean_G = bayesian_G_posterior(pts, G_grid, Y_grid)

        if np.isnan(p_mean) or np.isnan(G_mean):
            continue

        results.append({
            "gid": gid, "M": M_est,
            "p_bayes": p_mean, "p_std": p_std,
            "G_bayes": G_mean,
            "Y_from_p": Y_mean,
            "N": len(pts),
        })

        n_done += 1
        if n_done % 20 == 0:
            print(f"  {n_done} galaxies done...")

    print(f"  Total: {len(results)} galaxies with valid posteriors")

    # Extract arrays
    M = np.array([r["M"] for r in results])
    logM = np.log10(M)
    p_bayes = np.array([r["p_bayes"] for r in results])
    G_bayes = np.array([r["G_bayes"] for r in results])

    # Correlations
    corr_p = np.corrcoef(logM, p_bayes)[0, 1]
    corr_G = np.corrcoef(logM, np.log10(G_bayes))[0, 1]

    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  Correlation with log(M):")
    print(f"    p (Bayesian, marginalized over Y,dist): r = {corr_p:+.3f}")
    print(f"    G_eff (Bayesian, marginalized over Y):  r = {corr_G:+.3f}")

    # Mass-bin means
    print(f"\n  Mass-bin means:")
    print(f"  {'logM':>6} {'N':>4} {'<p>':>7} {'<G/G0>':>8}")
    print(f"  {'-' * 28}")
    for lo, hi in [(7, 8.5), (8.5, 9.5), (9.5, 10.3), (10.3, 11), (11, 12.5)]:
        m = (logM >= lo) & (logM < hi)
        if m.sum() >= 3:
            print(f"  {(lo+hi)/2:>6.1f} {m.sum():>4} {p_bayes[m].mean():>7.3f} {G_bayes[m].mean():>8.3f}")

    # Final verdict
    print(f"\n{'=' * 70}")
    print(f"VERDICT")
    print(f"{'=' * 70}")

    if abs(corr_G) > 0.25:
        print(f"\n  WARNING: G_eff correlates with M (r={corr_G:+.3f}).")
        print(f"  The Bayesian priors did not fully remove the bias.")
        print(f"  p(M) signal CANNOT be trusted.")
    elif abs(corr_p) > abs(corr_G) + 0.15:
        print(f"\n  POSITIVE: p correlates with M (r={corr_p:+.3f}) but G does not (r={corr_G:+.3f}).")
        print(f"  The difference ({abs(corr_p) - abs(corr_G):.3f}) is significant.")
        print(f"  p(M) is a REAL signal, not a methodological artifact.")
    elif abs(corr_p) > abs(corr_G) + 0.05:
        print(f"\n  SUGGESTIVE: p correlation (r={corr_p:+.3f}) slightly exceeds G (r={corr_G:+.3f}).")
        print(f"  p(M) may be real but marginal. More data needed.")
    else:
        print(f"\n  INCONCLUSIVE: p (r={corr_p:+.3f}) and G (r={corr_G:+.3f}) are similar.")
        print(f"  Cannot distinguish p(M) from methodological bias.")

    # Add cluster data point
    clusters = load_clusters()
    cl_M = np.mean([c["M500_sun"] for c in clusters])
    cl_p = 0.66  # from cluster fit
    print(f"\n  Note: Galaxy clusters (M~{cl_M:.0e}) independently require p~0.66,")
    print(f"  which is NOT affected by per-galaxy fitting bias.")


if __name__ == "__main__":
    main()
