"""
LITTLE THINGSの質量推定を徹底検証。
OH2015の質量 vs データからの再計算 vs 動的質量
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sdhg import load_little_things, p_of_M, A0, G, MSUN, KPC

OH2015_MASSES = {
    "cvidwa": 2.6e7, "ddo47": 3.5e8, "ddo50": 6.0e8,
    "ddo52": 3.3e8, "ddo53": 6.5e7, "ddo87": 4.0e8,
    "ddo101": 1.5e8, "ddo126": 2.0e8, "ddo133": 2.5e8,
    "ddo154": 5.0e8, "ddo168": 4.5e8, "ddo210": 2.0e7,
    "ddo216": 3.0e7, "ngc1569": 5.0e8, "ngc2366": 8.0e8,
    "ugc8508": 5.0e7, "wlm": 1.0e8,
}

# Iorio+2017 Table 3 distances (Mpc)
DISTANCES = {
    "cvidwa": 3.6, "ddo47": 5.2, "ddo50": 3.4,
    "ddo52": 10.3, "ddo53": 3.6, "ddo87": 7.7,
    "ddo101": 6.4, "ddo126": 4.9, "ddo133": 3.5,
    "ddo154": 3.7, "ddo168": 4.3, "ddo210": 0.9,
    "ddo216": 1.1, "ngc1569": 3.4, "ngc2366": 3.4,
    "ugc8508": 2.6, "wlm": 1.0,
}


def compute_gas_mass(pts):
    """HI面密度プロファイルからガス質量を積分。"""
    R_arr = np.array([p[0] for p in pts])  # kpc
    S_arr = np.array([p[3] for p in pts])  # Msun/pc^2
    M_gas = 0
    for i in range(len(R_arr)):
        R_pc = R_arr[i] * 1e3
        Sigma = S_arr[i]
        if i == 0:
            dR = R_arr[0] * 1e3
        else:
            dR = (R_arr[i] - R_arr[i-1]) * 1e3
        M_gas += 1.33 * Sigma * 2 * np.pi * R_pc * dR
    return M_gas


def compute_dyn_mass(pts):
    """動的質量 M_dyn = V_last^2 * R_last / G"""
    R_last = pts[-1][0] * KPC  # m
    V_last = pts[-1][1] * 1e3  # m/s
    return V_last**2 * R_last / G / MSUN


def rms_for_p(pts, M_bar, p_val):
    """Given mass and p, compute RMS."""
    R_last = max(p[0] for p in pts)
    xs, mus = [], []
    for R_kpc, Vc, eVc, Sigma in pts:
        R = R_kpc * KPC
        g_obs = (Vc * 1e3)**2 / R
        M_enc = M_bar * MSUN * (R_kpc / R_last)**1.5
        g_bar = G * M_enc / R**2
        if g_bar > 0 and g_obs > 0:
            xs.append(g_bar / A0)
            mus.append(g_bar / g_obs)
    if len(xs) < 4:
        return 9999
    xs = np.array(xs)
    mus = np.array(mus)
    pred = 1 - np.exp(-np.maximum(xs, 1e-20)**p_val)
    return np.sqrt(np.mean(
        (np.log10(np.maximum(mus, 1e-20)) -
         np.log10(np.maximum(pred, 1e-20)))**2
    ))


def rms_for_p_direct_gbar(pts, p_val):
    """Use Sigma directly to compute g_bar (no M_bar assumption)."""
    xs, mus = [], []
    R_arr = np.array([p[0] for p in pts])
    V_arr = np.array([p[1] for p in pts])
    S_arr = np.array([p[3] for p in pts])  # Msun/pc^2

    # Cumulative gas mass at each radius
    M_gas_cumul = np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        R_pc = R_arr[i] * 1e3
        Sigma = S_arr[i]
        if i == 0:
            dR = R_arr[0] * 1e3
        else:
            dR = (R_arr[i] - R_arr[i-1]) * 1e3
        ring_mass = 1.33 * Sigma * 2 * np.pi * R_pc * dR
        M_gas_cumul[i] = (M_gas_cumul[i-1] if i > 0 else 0) + ring_mass

    for i in range(len(R_arr)):
        R = R_arr[i] * KPC
        g_obs = (V_arr[i] * 1e3)**2 / R
        # g_bar from enclosed gas mass only (no stellar mass)
        g_bar = G * M_gas_cumul[i] * MSUN / R**2
        if g_bar > 0 and g_obs > 0:
            xs.append(g_bar / A0)
            mus.append(g_bar / g_obs)

    if len(xs) < 4:
        return 9999, np.array([]), np.array([])
    xs = np.array(xs)
    mus = np.array(mus)
    pred = 1 - np.exp(-np.maximum(xs, 1e-20)**p_val)
    rms = np.sqrt(np.mean(
        (np.log10(np.maximum(mus, 1e-20)) -
         np.log10(np.maximum(pred, 1e-20)))**2
    ))
    return rms, xs, mus


def main():
    lt = load_little_things()

    print("=" * 80)
    print("LITTLE THINGS 質量推定の徹底検証")
    print("=" * 80)

    # ================================================================
    # Part 1: 質量比較（OH2015 vs ガス積分 vs 動的質量）
    # ================================================================
    print("\n--- Part 1: 質量推定の比較 ---")
    print(f"{'Galaxy':>10} {'M_OH15':>10} {'M_gas':>10} {'M_dyn':>10} "
          f"{'OH/gas':>7} {'gas/dyn':>7}")
    print("-" * 65)

    mass_data = {}
    for gname in sorted(lt.keys()):
        pts = lt[gname]
        M_oh = OH2015_MASSES.get(gname, 0)
        M_gas = compute_gas_mass(pts)
        M_dyn = compute_dyn_mass(pts)
        mass_data[gname] = (M_oh, M_gas, M_dyn)

        oh_gas = M_oh / M_gas if M_gas > 0 else 0
        gas_dyn = M_gas / M_dyn if M_dyn > 0 else 0
        print(f"{gname:>10} {M_oh:>10.2e} {M_gas:>10.2e} {M_dyn:>10.2e} "
              f"{oh_gas:>7.2f} {gas_dyn:>7.3f}")

    # ================================================================
    # Part 2: M_enc ∝ R^1.5 の妥当性
    # ================================================================
    print("\n--- Part 2: M_enc ∝ R^1.5 近似の検証 ---")
    print("実際のガス面密度プロファイルから累積質量を計算し、R^1.5近似と比較")

    for gname in ['ddo154', 'ddo101', 'wlm', 'ngc2366']:
        pts = lt[gname]
        R_arr = np.array([p[0] for p in pts])
        S_arr = np.array([p[3] for p in pts])
        M_oh = OH2015_MASSES.get(gname, 0)

        # Actual cumulative mass
        M_cumul = np.zeros(len(R_arr))
        for i in range(len(R_arr)):
            R_pc = R_arr[i] * 1e3
            Sigma = S_arr[i]
            dR = (R_arr[i] - R_arr[i-1]) * 1e3 if i > 0 else R_arr[0] * 1e3
            ring_mass = 1.33 * Sigma * 2 * np.pi * R_pc * dR
            M_cumul[i] = (M_cumul[i-1] if i > 0 else 0) + ring_mass

        # R^1.5 approximation
        R_last = R_arr[-1]
        M_approx = M_oh * (R_arr / R_last)**1.5

        # Fit power law to actual profile
        valid = (M_cumul > 0) & (R_arr > 0)
        if valid.sum() >= 3:
            log_r = np.log10(R_arr[valid] / R_last)
            log_m = np.log10(M_cumul[valid] / M_cumul[valid][-1])
            slope = np.polyfit(log_r, log_m, 1)[0]
        else:
            slope = 0

        ratio_range = M_cumul / (M_approx + 1e-10)
        valid_ratio = ratio_range[(M_approx > 0) & (M_cumul > 0)]

        print(f"\n  {gname}: actual power law slope = {slope:.2f} "
              f"(assumed 1.5), ratio range = {valid_ratio.min():.2f}-{valid_ratio.max():.2f}")

    # ================================================================
    # Part 3: 直接g_bar計算（M_barを使わない）
    # ================================================================
    print("\n\n--- Part 3: g_barをHI面密度から直接計算（M_bar仮定なし） ---")
    print(f"{'Galaxy':>10} {'M_gas':>10} {'p_pred':>7} {'RMS_McG':>8} {'RMS_pred':>9} "
          f"{'RMS_gas_McG':>11} {'RMS_gas_pred':>12}")
    print("-" * 80)

    rms_mcg_old, rms_pred_old = [], []
    rms_mcg_new, rms_pred_new = [], []

    for gname in sorted(lt.keys()):
        pts = lt[gname]
        M_oh = OH2015_MASSES.get(gname, 0)
        M_gas = compute_gas_mass(pts)

        if M_oh == 0:
            continue

        # Old method (OH2015 mass + R^1.5)
        rms_old_mcg = rms_for_p(pts, M_oh, 0.5)
        p_pred_oh = p_of_M(M_oh)
        rms_old_pred = rms_for_p(pts, M_oh, p_pred_oh)

        # New method (direct g_bar from Sigma)
        rms_new_mcg, _, _ = rms_for_p_direct_gbar(pts, 0.5)
        p_pred_gas = p_of_M(M_gas)
        rms_new_pred, _, _ = rms_for_p_direct_gbar(pts, p_pred_gas)

        rms_mcg_old.append(rms_old_mcg)
        rms_pred_old.append(rms_old_pred)
        rms_mcg_new.append(rms_new_mcg)
        rms_pred_new.append(rms_new_pred)

        print(f"{gname:>10} {M_gas:>10.2e} {p_pred_gas:>7.3f} {rms_old_mcg:>8.4f} "
              f"{rms_old_pred:>9.4f} {rms_new_mcg:>11.4f} {rms_new_pred:>12.4f}")

    print(f"\n{'MEAN':>10} {'':>10} {'':>7} {np.mean(rms_mcg_old):>8.4f} "
          f"{np.mean(rms_pred_old):>9.4f} {np.mean(rms_mcg_new):>11.4f} "
          f"{np.mean(rms_pred_new):>12.4f}")

    imp_old = (np.mean(rms_mcg_old) - np.mean(rms_pred_old)) / np.mean(rms_mcg_old) * 100
    imp_new = (np.mean(rms_mcg_new) - np.mean(rms_pred_new)) / np.mean(rms_mcg_new) * 100

    print(f"\n旧方式 (OH2015 mass + R^1.5): SDHG vs McGaugh = {imp_old:+.1f}%")
    print(f"新方式 (HI直接積分):           SDHG vs McGaugh = {imp_new:+.1f}%")

    # ================================================================
    # Part 4: 動的質量を使う
    # ================================================================
    print("\n\n--- Part 4: 動的質量 M_dyn = V_flat^2 * R / G を使う ---")
    rms_mcg_dyn, rms_pred_dyn = [], []

    for gname in sorted(lt.keys()):
        pts = lt[gname]
        M_dyn = compute_dyn_mass(pts)

        p_pred = p_of_M(M_dyn)
        rms_mcg = rms_for_p(pts, M_dyn, 0.5)
        rms_pred = rms_for_p(pts, M_dyn, p_pred)

        rms_mcg_dyn.append(rms_mcg)
        rms_pred_dyn.append(rms_pred)

        better = 'O' if rms_pred < rms_mcg else 'X'
        print(f"  {gname:>10} M_dyn={M_dyn:.2e} p={p_pred:.3f} "
              f"McG={rms_mcg:.4f} SDHG={rms_pred:.4f} {better}")

    imp_dyn = (np.mean(rms_mcg_dyn) - np.mean(rms_pred_dyn)) / np.mean(rms_mcg_dyn) * 100
    print(f"\n動的質量方式: SDHG vs McGaugh = {imp_dyn:+.1f}%")
    n_win = sum(1 for a, b in zip(rms_pred_dyn, rms_mcg_dyn) if a < b)
    print(f"  勝率: {n_win}/{len(rms_pred_dyn)}")

    # ================================================================
    # Part 5: SPARCと同じ質量定義を使う
    # ================================================================
    print("\n\n--- Part 5: SPARCと同じ質量定義 M ~ 0.5*V_flat^2*R_last/G ---")
    rms_mcg_sparc, rms_pred_sparc = [], []

    for gname in sorted(lt.keys()):
        pts = lt[gname]
        R_arr = np.array([p[0] for p in pts])
        V_arr = np.array([p[1] for p in pts])

        # V_flat: outer 3 points average
        V_flat = V_arr[-3:].mean() if len(V_arr) >= 3 else V_arr[-1]
        R_last = R_arr[-1]
        M_sparc = 0.5 * (V_flat * 1e3)**2 * (R_last * KPC) / G / MSUN

        p_pred = p_of_M(M_sparc)
        rms_mcg = rms_for_p(pts, M_sparc, 0.5)
        rms_pred = rms_for_p(pts, M_sparc, p_pred)

        rms_mcg_sparc.append(rms_mcg)
        rms_pred_sparc.append(rms_pred)

        better = 'O' if rms_pred < rms_mcg else 'X'
        print(f"  {gname:>10} M={M_sparc:.2e} p={p_pred:.3f} "
              f"McG={rms_mcg:.4f} SDHG={rms_pred:.4f} {better}")

    imp_sparc = (np.mean(rms_mcg_sparc) - np.mean(rms_pred_sparc)) / np.mean(rms_mcg_sparc) * 100
    print(f"\nSPARC式質量: SDHG vs McGaugh = {imp_sparc:+.1f}%")
    n_win = sum(1 for a, b in zip(rms_pred_sparc, rms_mcg_sparc) if a < b)
    print(f"  勝率: {n_win}/{len(rms_pred_sparc)}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"方式                              SDHG vs McGaugh  ")
    print(f"-" * 50)
    print(f"旧 (OH2015 mass + R^1.5近似):     {imp_old:+.1f}%")
    print(f"HI直接積分 (g_bar from Sigma):    {imp_new:+.1f}%")
    print(f"動的質量 (V^2*R/G):               {imp_dyn:+.1f}%")
    print(f"SPARC式 (0.5*V_flat^2*R/G):       {imp_sparc:+.1f}%")


if __name__ == "__main__":
    main()
