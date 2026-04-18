"""
Leave-one-out cross-validation of p(M) within SPARC.

For each galaxy i, fit p(M) parameters (M0, alpha) on all OTHER galaxies,
then predict galaxy i's rotation curve. Compare total RMS with McGaugh.

This tests whether p(M) generalizes to unseen data or overfits.

Usage:
    python run_loo_cv.py
"""
import numpy as np
from scipy.optimize import minimize
from sdhg import load_sparc, load_clusters, A0, G, MSUN, KPC


def main():
    print("=" * 70)
    print("Leave-One-Out Cross-Validation: p(M) within SPARC")
    print("=" * 70)

    galaxies = load_sparc()
    clusters = load_clusters()

    # Prepare galaxy data
    gal_ids = []
    gal_data = {}
    gal_mass = {}
    for gid, pts in galaxies.items():
        if len(pts) < 5:
            continue
        V_last = pts[-1][1] * 1e3
        R_last = pts[-1][0] * KPC
        M_est = 0.5 * V_last ** 2 * R_last / G / MSUN
        flat_pts = []
        for R_kpc, Vo, eV, Vd, Vg, Vb in pts:
            flat_pts.append((Vd * 1e3, Vg * 1e3, Vb * 1e3,
                             R_kpc * KPC, Vo * 1e3))
        gal_ids.append(gid)
        gal_data[gid] = flat_pts
        gal_mass[gid] = M_est

    N = len(gal_ids)
    print(f"\nGalaxies with >= 5 points: {N}")

    def score_train(Y, logM0, alpha, exclude_gid):
        """RMS on training set (all galaxies except exclude_gid)."""
        M0 = 10 ** logM0
        s, n = 0.0, 0
        for gid in gal_ids:
            if gid == exclude_gid:
                continue
            M = gal_mass[gid]
            u = (max(M, 1) / M0) ** alpha
            p = 2 * u / (1 + 3 * u)
            for Vd, Vg, Vb, R, Vo in gal_data[gid]:
                go = Vo ** 2 / R
                gb = (Y * Vd ** 2 + np.sign(Vg) * Vg ** 2 + 0.7 * Vb ** 2) / R
                if gb > 0 and go > 0:
                    x = gb / A0
                    mu = max(1 - np.exp(-max(x, 1e-20) ** p), 1e-20)
                    s += (np.log10(go) - np.log10(gb / mu)) ** 2
                    n += 1
        # Include clusters (always in training)
        for cl in clusters:
            gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
            u = (cl["M500_sun"] / M0) ** alpha
            p = 2 * u / (1 + 3 * u)
            x = gb / A0
            mu = max(1 - np.exp(-max(x, 1e-20) ** p), 1e-20)
            mu_need = cl["M_bar_sun"] / cl["M500_sun"]
            s += 50 * (np.log10(mu_need) - np.log10(mu)) ** 2
            n += 50
        return np.sqrt(s / max(n, 1))

    def predict_galaxy(gid, Y, logM0, alpha):
        """Predict one galaxy's RMS using trained parameters."""
        M0 = 10 ** logM0
        M = gal_mass[gid]
        u = (max(M, 1) / M0) ** alpha
        p = 2 * u / (1 + 3 * u)
        s, n = 0.0, 0
        for Vd, Vg, Vb, R, Vo in gal_data[gid]:
            go = Vo ** 2 / R
            gb = (Y * Vd ** 2 + np.sign(Vg) * Vg ** 2 + 0.7 * Vb ** 2) / R
            if gb > 0 and go > 0:
                x = gb / A0
                mu = max(1 - np.exp(-max(x, 1e-20) ** p), 1e-20)
                s += (np.log10(go) - np.log10(gb / mu)) ** 2
                n += 1
        return s, n

    def predict_mcgaugh(gid, Y):
        """Predict one galaxy's RMS using McGaugh."""
        s, n = 0.0, 0
        for Vd, Vg, Vb, R, Vo in gal_data[gid]:
            go = Vo ** 2 / R
            gb = (Y * Vd ** 2 + np.sign(Vg) * Vg ** 2 + 0.7 * Vb ** 2) / R
            if gb > 0 and go > 0:
                x = gb / A0
                mu = max(1 - np.exp(-x ** 0.5), 1e-20)
                s += (np.log10(go) - np.log10(gb / mu)) ** 2
                n += 1
        return s, n

    # First: fit McGaugh on all data for Y_global
    def mcg_obj(Y):
        s, n = 0.0, 0
        for gid in gal_ids:
            for Vd, Vg, Vb, R, Vo in gal_data[gid]:
                go = Vo ** 2 / R
                gb = (Y * Vd ** 2 + np.sign(Vg) * Vg ** 2 + 0.7 * Vb ** 2) / R
                if gb > 0 and go > 0:
                    x = gb / A0
                    mu = max(1 - np.exp(-x ** 0.5), 1e-20)
                    s += (np.log10(go) - np.log10(gb / mu)) ** 2
                    n += 1
        for cl in clusters:
            gb = G * cl["M_bar_sun"] * MSUN / cl["R500_m"] ** 2
            x = gb / A0
            mu = max(1 - np.exp(-max(x, 1e-20) ** 0.5), 1e-20)
            mu_need = cl["M_bar_sun"] / cl["M500_sun"]
            s += 50 * (np.log10(mu_need) - np.log10(mu)) ** 2
            n += 50
        return np.sqrt(s / max(n, 1))

    from scipy.optimize import minimize_scalar
    rM = minimize_scalar(mcg_obj, bounds=(0.1, 3.0), method="bounded")
    Y_mcg = rM.x

    # LOO
    print(f"\nRunning LOO (this may take a few minutes)...")
    total_s_sdhg, total_s_mcg, total_n = 0.0, 0.0, 0
    mass_bins = {"<9": [0, 0, 0], "9-10": [0, 0, 0], ">10": [0, 0, 0]}

    for i, gid in enumerate(gal_ids):
        # Fit p(M) on training set (exclude gid)
        def train_obj(params):
            return score_train(params[0], params[1], params[2], gid)

        r = minimize(train_obj, [0.5, 10.5, 0.33],
                     bounds=[(0.1, 3), (8, 13), (0.1, 0.6)],
                     method="L-BFGS-B")

        # Predict held-out galaxy
        s_s, n_s = predict_galaxy(gid, r.x[0], r.x[1], r.x[2])
        s_m, n_m = predict_mcgaugh(gid, Y_mcg)

        total_s_sdhg += s_s
        total_s_mcg += s_m
        total_n += n_s

        # Mass bin tracking
        logM = np.log10(gal_mass[gid])
        if logM < 9:
            key = "<9"
        elif logM < 10:
            key = "9-10"
        else:
            key = ">10"
        mass_bins[key][0] += s_s
        mass_bins[key][1] += s_m
        mass_bins[key][2] += n_s

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N} done...")

    rms_sdhg = np.sqrt(total_s_sdhg / total_n)
    rms_mcg = np.sqrt(total_s_mcg / total_n)
    improvement = (rms_mcg - rms_sdhg) / rms_mcg * 100

    print(f"\n{'Method':<20} {'RMS':>8} {'vs McGaugh':>12}")
    print("-" * 42)
    print(f"{'McGaugh (p=0.5)':<20} {rms_mcg:>8.3f} {'baseline':>12}")
    print(f"{'p(M) LOO':<20} {rms_sdhg:>8.3f} {improvement:>+11.1f}%")

    print(f"\nMass-bin breakdown:")
    print(f"  {'Bin':<8} {'N_pts':>6} {'SDHG':>8} {'McGaugh':>8} {'Improvement':>12}")
    print(f"  {'-' * 44}")
    for key in ["<9", "9-10", ">10"]:
        ss, sm, nn = mass_bins[key]
        if nn > 0:
            rs = np.sqrt(ss / nn)
            rm = np.sqrt(sm / nn)
            imp = (rm - rs) / rm * 100
            print(f"  {'logM' + key:<8} {nn:>6} {rs:>8.3f} {rm:>8.3f} {imp:>+11.1f}%")

    print(f"\n{'=' * 70}")
    if improvement > 3:
        print(f"  p(M) generalizes to unseen data (+{improvement:.1f}%).")
        print(f"  Not overfitting.")
    elif improvement > -3:
        print(f"  Marginal improvement ({improvement:+.1f}%). Within noise.")
    else:
        print(f"  Negative result ({improvement:+.1f}%). Possible overfitting.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
