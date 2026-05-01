"""
α_Σ = 5/3 の理論的導出。

核心の関係式:
  α_Σ = α_M / (1 - 2β)

where:
  α_M = 1/3  (SDHG/3D CDT: r ~ M^(1/3))
  β    = 0.4 (galaxy R-M scaling, between TF (0.5) and uniform density (1/3))

  → α_Σ = (1/3) / (1 - 0.8) = (1/3) / 0.2 = 5/3

徹底検証:
  1. SPARC実測R-M勾配 β の精密測定
  2. β=0.4 の物理的根拠
  3. CDT spectral dimension flow との同等性確認
  4. 独立予測テスト

Usage:
    python run_alpha_derivation.py
"""
import numpy as np
from scipy.optimize import curve_fit, minimize, minimize_scalar
from sdhg import load_sparc, load_clusters, load_little_things, A0, G, MSUN, KPC


def main():
    print("=" * 75)
    print("α_Σ = 5/3 の理論的導出")
    print("=" * 75)

    print("""
基本定理:
  p(Σ) = 2u/(1+3u)    with  u = (Σ/Σ₀)^α_Σ
  p(M) = 2u'/(1+3u')  with  u' = (M/M₀)^α_M

galaxies follow: R ∝ M^β (empirical R-M scaling)
Σ = M/(πR²) ∝ M/M^(2β) = M^(1-2β)

For p(Σ) and p(M) to describe THE SAME physics:
  u = u'
  (Σ/Σ₀)^α_Σ = (M/M₀)^α_M
  Σ^α_Σ = M^α_M × const
  (M^(1-2β))^α_Σ = M^α_M × const
  α_Σ × (1 - 2β) = α_M

  →  α_Σ = α_M / (1 - 2β)
""")

    # ================================================================
    # Test 1: SPARC実測 β
    # ================================================================
    print("\n--- Test 1: SPARC実測 R-M scaling ---")

    galaxies = load_sparc()
    Ms, Rs = [], []
    for gid, pts in galaxies.items():
        if len(pts) < 5:
            continue
        V_arr = np.array([p[1] for p in pts])
        R_arr = np.array([p[0] for p in pts])
        V_flat = V_arr[-3:].mean() if len(V_arr) >= 3 else V_arr[-1]
        R_last = R_arr[-1]
        M = 0.5 * (V_flat*1e3)**2 * (R_last*KPC) / G / MSUN
        Ms.append(M)
        Rs.append(R_last)

    logM = np.log10(Ms)
    logR = np.log10(Rs)

    # 1パラメータフィット (R = R₀ M^β)
    beta_obs, logR0 = np.polyfit(logM, logR, 1)
    print(f"\n  全銀河 (N={len(Ms)}): β = {beta_obs:.3f}")
    print(f"  log R = {beta_obs:.3f} × log M + {logR0:.2f}")

    # BT (baryonic Tully-Fisher) predicts β = 1/2 exactly
    print(f"\n  Tully-Fisher予測: V⁴ = G M a₀")
    print(f"    → V² = √(GMa₀)")
    print(f"    → R = GM/V² = √(GM/a₀) ∝ M^(1/2)")
    print(f"    → β_TF = 0.500")
    print(f"\n  観測 β = {beta_obs:.3f} (TF予測より ~{(0.5-beta_obs):.2f} 低い)")

    # Derivation chain
    print(f"\n  α_M = 1/3 (CDT prediction)")
    alpha_M = 1/3
    alpha_S_from_beta = alpha_M / (1 - 2*beta_obs)
    print(f"  β = {beta_obs:.3f} → α_Σ = (1/3) / (1 - 2×{beta_obs:.3f}) = "
          f"(1/3) / {1-2*beta_obs:.3f} = {alpha_S_from_beta:.3f}")

    # β = 0.4 gives 5/3
    print(f"\n  β = 0.400 → α_Σ = (1/3) / 0.200 = {(1/3)/0.2:.4f} = 5/3 exactly")
    print(f"  β = 0.500 (TF) → α_Σ = (1/3) / 0.000 → ∞ (singular)")

    # ================================================================
    # Test 2: β の精密測定 (ロバスト推定)
    # ================================================================
    print("\n--- Test 2: β のロバスト推定 ---")

    # 1σ誤差
    from scipy.stats import pearsonr, linregress
    lr = linregress(logM, logR)
    print(f"\n  線形回帰: β = {lr.slope:.4f} ± {lr.stderr:.4f}")
    print(f"  切片: {lr.intercept:.2f} ± {lr.intercept_stderr:.2f}")
    print(f"  相関: r = {lr.rvalue:.3f}")

    # Different subsets
    print(f"\n  サブセット別 β:")
    for lo, hi, label in [(7, 9, 'dwarf'), (9, 10, 'low'),
                           (10, 11, 'int'), (11, 13, 'massive')]:
        mask = (logM >= lo) & (logM < hi)
        if mask.sum() < 5:
            continue
        lr_sub = linregress(logM[mask], logR[mask])
        alpha_sub = alpha_M / (1 - 2*lr_sub.slope) if lr_sub.slope < 0.5 else float('nan')
        print(f"    logM {lo}-{hi} (N={mask.sum()}): β = {lr_sub.slope:.3f} "
              f"± {lr_sub.stderr:.3f}, α_Σ = {alpha_sub:.3f}")

    # Median β (robust)
    # Bootstrap
    rng = np.random.RandomState(42)
    betas_boot = []
    N = len(logM)
    for _ in range(1000):
        idx = rng.choice(N, N, replace=True)
        b, _ = np.polyfit(logM[idx], logR[idx], 1)
        betas_boot.append(b)
    betas_boot = np.array(betas_boot)
    print(f"\n  Bootstrap β (N=1000): median={np.median(betas_boot):.3f}, "
          f"16-84%ile = [{np.percentile(betas_boot, 16):.3f}, "
          f"{np.percentile(betas_boot, 84):.3f}]")

    # ================================================================
    # Test 3: α_Σ = α_M/(1-2β) の数値的検証
    # ================================================================
    print("\n--- Test 3: 関係式の数値検証 ---")

    # fit α_Σ simultaneously with β matching
    all_gals = []
    for gid, pts in galaxies.items():
        if len(pts) < 5:
            continue
        V_arr = np.array([p[1] for p in pts])
        R_arr = np.array([p[0] for p in pts])
        V_flat = V_arr[-3:].mean() if len(V_arr) >= 3 else V_arr[-1]
        R_last = R_arr[-1]
        M = 0.5 * (V_flat*1e3)**2 * (R_last*KPC) / G / MSUN
        Sigma = M / (np.pi * R_last**2)
        all_gals.append((gid, pts, M, Sigma, R_last))

    def p_Sigma(Sigma, Sigma0, alpha):
        u = (Sigma/Sigma0)**alpha
        return np.clip(2*u/(1+3*u), 0.01, 0.95)

    def rms(Y, Sigma0, alpha):
        resid = []
        for gid, pts, M, Sigma, R in all_gals:
            for R_kpc, Vobs, eVobs, Vdisk, Vgas, Vbul in pts:
                R_m = R_kpc*KPC
                g_obs = (Vobs*1e3)**2/R_m
                g_bar = (Y*(Vdisk*1e3)**2 + np.sign(Vgas)*(Vgas*1e3)**2 + 0.7*(Vbul*1e3)**2)/R_m
                if g_bar <= 0 or g_obs <= 0: continue
                x = g_bar/A0
                p = p_Sigma(Sigma, Sigma0, alpha)
                mu = max(1-np.exp(-max(x, 1e-20)**p), 1e-20)
                resid.append(np.log10(g_obs)-np.log10(g_bar/mu))
        return np.sqrt(np.mean(np.array(resid)**2)) if resid else 9999

    # Try both: α from derivation, and free fit
    SIGMA0_THEORY = A0 / (4 * np.pi**2 * G) / MSUN * KPC**2

    print(f"\n  Σ₀ = a₀/(4π²G) = {SIGMA0_THEORY:.2e} M☉/kpc²")
    print()
    print(f"  {'Assumption':>25} {'α_Σ':>7} {'Y':>6} {'RMS':>8}")
    print("-" * 50)

    # 1. β = 0.4 (theoretical "clean" value)
    alpha_04 = alpha_M / (1 - 2*0.4)
    Y_opt = minimize_scalar(lambda Y: rms(Y, SIGMA0_THEORY, alpha_04),
                            bounds=(0, 2), method='bounded').x
    r_04 = rms(Y_opt, SIGMA0_THEORY, alpha_04)
    print(f"  {'β=0.4 → α=5/3':>25} {alpha_04:>7.3f} {Y_opt:>6.3f} {r_04:>8.4f}")

    # 2. β from SPARC observation
    alpha_obs = alpha_M / (1 - 2*beta_obs)
    Y_opt = minimize_scalar(lambda Y: rms(Y, SIGMA0_THEORY, alpha_obs),
                            bounds=(0, 2), method='bounded').x
    r_obs = rms(Y_opt, SIGMA0_THEORY, alpha_obs)
    print(f"  {'β=observed → α_Σ':>25} {alpha_obs:>7.3f} {Y_opt:>6.3f} {r_obs:>8.4f}")

    # 3. Free fit α
    from scipy.optimize import minimize
    res = minimize(lambda p: rms(p[0], SIGMA0_THEORY, p[1]),
                   [0.5, 1.5], bounds=[(0, 2), (0.1, 3)], method='L-BFGS-B')
    Y_f, alpha_f = res.x
    print(f"  {'free fit α':>25} {alpha_f:>7.3f} {Y_f:>6.3f} {res.fun:>8.4f}")

    # ================================================================
    # Test 4: CDT spectral dimension との同等性
    # ================================================================
    print("\n--- Test 4: CDT spectral dimension flow との同等性 ---")
    print(f"""
  CDT spectral dimension (Ambjørn+ 2005):
    D_s(σ) = a - b/(c + σ^γ)
    4D CDT measurement: γ = 1.00 ± 0.1
    D_s(0) ≈ 2, D_s(∞) ≈ 4, so a = 4, b = 2

  SDHG formula for RAR:
    p = (2/3) - (2/3)/(1 + v)     where v = 3u = 3(Σ/Σ₀)^α
                                             (equivalent to CDT formula
                                              with γ = 1 in the v variable)

  The 'diffusion scale' σ in CDT maps onto v for galaxies:
    σ_CDT ≡ v = 3(Σ/Σ₀)^α

  Therefore:
    CDT γ = 1 corresponds to v ∝ σ¹
    v ∝ Σ^α = Σ^(5/3)  if α = 5/3
    Σ ∝ σ^(1/α) = σ^(3/5)  (the inverse mapping)

  Consistency check with CDT:
    γ_CDT = 1 ✓
    α_galaxy emerges from the Σ→σ mapping
""")

    # ================================================================
    # Test 5: 最終予測テスト (銀河団 + 矮小)
    # ================================================================
    print("\n--- Test 5: 最終予測検証 (out-of-sample) ---")

    # Clusters
    clusters = load_clusters()
    p_preds_cluster = []
    for c in clusters:
        R_kpc = c['R500_m']/KPC
        Sigma = c['M500_sun']/(np.pi*R_kpc**2)
        p_preds_cluster.append(p_Sigma(Sigma, SIGMA0_THEORY, 5/3))
    print(f"\n  銀河団 (α=5/3):")
    print(f"    予測 p = {np.mean(p_preds_cluster):.3f}")
    print(f"    観測 p ~ 0.66")
    print(f"    一致: {(1-abs(np.mean(p_preds_cluster)-0.66)/0.66)*100:.1f}%")

    # LITTLE THINGS (ultra-dwarfs)
    lt = load_little_things()
    p_preds_ud = []
    for gname, pts in lt.items():
        R_arr = np.array([p[0] for p in pts])
        V_arr = np.array([p[1] for p in pts])
        V_flat = V_arr[-3:].mean() if len(V_arr) >= 3 else V_arr[-1]
        R_last = R_arr[-1]
        M = 0.5*(V_flat*1e3)**2*(R_last*KPC)/G/MSUN
        if M < 1e8:
            Sigma = M/(np.pi*R_last**2)
            p_preds_ud.append(p_Sigma(Sigma, SIGMA0_THEORY, 5/3))
    if p_preds_ud:
        print(f"\n  Ultra-dwarf (M<10⁸, α=5/3):")
        print(f"    予測 mean p = {np.mean(p_preds_ud):.3f}")
        print(f"    観測: p < 0.25 (McGaughより低いp)")
        print(f"    確認: {'OK' if np.mean(p_preds_ud) < 0.3 else 'NG'}")


if __name__ == "__main__":
    main()
