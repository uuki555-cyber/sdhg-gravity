"""
α_Σ = 5/3 の理論的起源を探る。

1. 幾何学的導出: Σ₀×πG vs a₀ の関係
2. 次元解析: Σ → x = g_bar/a₀ 変換下でのαの意味
3. 隣接形式との同等性: p(x) = 2v/(1+3v), v = x^β
4. CDT接続: α_Σ と α_M の理論的関係
5. 物理量スケール: 5/3 vs 他の自然数

Usage:
    python run_alpha_theory.py
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sdhg import load_sparc, p_of_M, A0, G, MSUN, KPC

PC = 3.086e16  # m


def p_of_Sigma(Sigma, Sigma0, alpha):
    u = (Sigma / Sigma0)**alpha
    return np.clip(2*u/(1+3*u), 0.01, 0.95)


def p_of_x(x, x0, beta):
    """x (= g_bar/a0) の直接関数として p(x) = 2v/(1+3v), v = (x/x0)^β."""
    v = (x / x0)**beta
    return np.clip(2*v/(1+3*v), 0.01, 0.95)


def global_rms_px(gal_list, Y, x0, beta):
    """p(x) model RMS."""
    resid = []
    for gid, pts, M, Sigma in gal_list:
        for R_kpc, Vobs, eVobs, Vdisk, Vgas, Vbul in pts:
            R = R_kpc*KPC
            g_obs = (Vobs*1e3)**2/R
            g_bar = (Y*(Vdisk*1e3)**2 + np.sign(Vgas)*(Vgas*1e3)**2 +
                     0.7*(Vbul*1e3)**2)/R
            if g_bar <= 0 or g_obs <= 0:
                continue
            x = g_bar/A0
            p = p_of_x(x, x0, beta)
            mu = max(1-np.exp(-max(x, 1e-20)**p), 1e-20)
            resid.append(np.log10(g_obs)-np.log10(g_bar/mu))
    return np.sqrt(np.mean(np.array(resid)**2)) if resid else 9999


def global_rms_pSigma(gal_list, Y, Sigma0, alpha):
    resid = []
    for gid, pts, M, Sigma in gal_list:
        for R_kpc, Vobs, eVobs, Vdisk, Vgas, Vbul in pts:
            R = R_kpc*KPC
            g_obs = (Vobs*1e3)**2/R
            g_bar = (Y*(Vdisk*1e3)**2 + np.sign(Vgas)*(Vgas*1e3)**2 +
                     0.7*(Vbul*1e3)**2)/R
            if g_bar <= 0 or g_obs <= 0:
                continue
            x = g_bar/A0
            p = p_of_Sigma(Sigma, Sigma0, alpha)
            mu = max(1-np.exp(-max(x, 1e-20)**p), 1e-20)
            resid.append(np.log10(g_obs)-np.log10(g_bar/mu))
    return np.sqrt(np.mean(np.array(resid)**2)) if resid else 9999


def main():
    print("="*75)
    print("α_Σ = 5/3 の理論的起源探求")
    print("="*75)

    galaxies = load_sparc()
    all_gals = []
    for gid, pts in galaxies.items():
        if len(pts) < 5:
            continue
        V_last = pts[-1][1]*1e3
        R_kpc = pts[-1][0]
        R_last = R_kpc*KPC
        M = 0.5*V_last**2*R_last/G/MSUN
        Sigma = M / (np.pi * R_kpc**2)
        all_gals.append((gid, pts, M, Sigma))

    # ================================================================
    # Test 1: p(Σ) と p(x) の同等性
    # 単一ビーム近似: g_bar ~ πGΣ → x = πGΣ/a₀
    # もしこれが正確なら p(Σ) と p(x) は同じ
    # ================================================================
    print("\n--- Test 1: p(Σ) と p(x = g_bar/a₀) の同等性 ---")
    print()
    print("  もし g_bar = πGΣ なら:")
    print("    u = (Σ/Σ₀)^α = (πGΣ/a₀ × a₀/(πGΣ₀))^α = (x × a₀/(πGΣ₀))^α")
    print("    = (x / x₀)^α where x₀ = πGΣ₀/a₀")
    print()

    # Σ₀ = a₀/(4π²G) なら x₀ = πG × a₀/(4π²G) / a₀ = 1/(4π)
    Sigma0_theory_SI = A0 / (4 * np.pi**2 * G)
    x0_theory = 1 / (4 * np.pi)

    print(f"  理論予測: Σ₀ = a₀/(4π²G) → x₀ = πGΣ₀/a₀ = 1/(4π) = {x0_theory:.4f}")
    print()

    # p(x) で直接フィット
    def opt_px(params):
        Y, logx0, beta = params
        return global_rms_px(all_gals, Y, 10**logx0, beta)

    print("  p(x) model direct fit:")
    res_px = minimize(opt_px, [0.5, -1.0, 1.0],
                      bounds=[(0.0, 2.0), (-3.0, 2.0), (0.05, 3.0)],
                      method='L-BFGS-B')
    Y_x, logx0_x, beta_x = res_px.x
    print(f"    Y={Y_x:.3f}, x₀={10**logx0_x:.4f}, β={beta_x:.3f}, RMS={res_px.fun:.4f}")
    print(f"    (理論x₀ = 1/(4π) = {x0_theory:.4f})")
    print()

    # p(Σ) best fit
    def opt_pS(params):
        Y, logS0, alpha = params
        return global_rms_pSigma(all_gals, Y, 10**logS0, alpha)

    print("  p(Σ) model direct fit:")
    res_pS = minimize(opt_pS, [0.5, 7.5, 1.5],
                      bounds=[(0.0, 2.0), (4.0, 12.0), (0.05, 3.0)],
                      method='L-BFGS-B')
    Y_S, logS0_S, alpha_S = res_pS.x
    print(f"    Y={Y_S:.3f}, logΣ₀={logS0_S:.2f}, α={alpha_S:.3f}, RMS={res_pS.fun:.4f}")
    print()

    if abs(beta_x - alpha_S) < 0.2:
        print("  → β ≈ α ! p(x) と p(Σ) は同じ理論的内容")
    else:
        print(f"  → β={beta_x:.3f} ≠ α={alpha_S:.3f} ! Σとxのマッピングが単純でない")

    # ================================================================
    # Test 2: 5/3 は何から来るか？物理的に自然な候補
    # ================================================================
    print("\n--- Test 2: 5/3 の物理的候補 ---")

    candidates = [
        (5/3, "5/3: 単原子理想気体 γ, Jeans質量指数"),
        (2/3 + 1, "5/3 = 1 + 2/3 (McGaugh transition + holographic)"),
        (1 + 2/3, "同上"),
        ((np.pi/2)**(1/3), f"(π/2)^(1/3) = {(np.pi/2)**(1/3):.3f}"),
        ((2*np.pi)**(1/4), f"(2π)^(1/4) = {(2*np.pi)**(1/4):.3f}"),
        (np.log(3*np.pi), f"log(3π) = {np.log(3*np.pi):.3f}"),
        (np.sqrt(2*np.pi)/np.e, f"√(2π)/e = {np.sqrt(2*np.pi)/np.e:.3f}"),
        (np.e**(1/2), f"√e = {np.exp(0.5):.3f}"),
        (5/3, "確認"),
    ]

    print()
    # α_S を固定してΣ₀をフィット
    for alpha_candidate, label in candidates:
        res_fixed = minimize(
            lambda params: global_rms_pSigma(all_gals, params[0], 10**params[1], alpha_candidate),
            [0.5, 7.5],
            bounds=[(0.0, 2.0), (4.0, 12.0)],
            method='L-BFGS-B')
        Y_f, logS0_f = res_fixed.x
        delta = alpha_candidate - alpha_S
        print(f"  α = {alpha_candidate:.4f}: RMS = {res_fixed.fun:.4f}, "
              f"Δα = {delta:+.3f}  ({label})")

    # ================================================================
    # Test 3: α_Σ と α_M の関係
    # p(M) は u_M = (M/M₀)^α_M で α_M = 1/3 (SDHG予測)
    # 仮に M ~ ρR³ (一様密度), Σ ~ R² → Σ ~ M^(2/3)
    # すると (M/M₀)^α_M = (Σ/Σ_M)^(α_M × 3/2)
    # SDHG α_M = 1/3 → α_Σ = 1/2 (virial接続)
    # しかし実測 α_Σ = 5/3
    # 差は 5/3 - 1/2 = 7/6
    # ================================================================
    print("\n--- Test 3: α_Σ vs α_M の幾何学的マッピング ---")
    print()
    print("  仮定: 銀河系がvirialize → M ~ ρR³, R ~ M^(1/3)")
    print("  Σ = M/(πR²) ~ M/M^(2/3) = M^(1/3)")
    print("  → M = (Σ/Σ_M)^(1/α_M) なら Σ = M^(1/3) → α_Σ = 3 α_M")
    print()
    print("  α_M = 1/3 (SDHG prediction)")
    print(f"  予測 α_Σ = 3 × 1/3 = 1.0")
    print(f"  観測 α_Σ = {alpha_S:.3f}")
    print(f"  差: {alpha_S - 1:+.3f}")
    print()

    # fixed rho (uniform density)では合わない
    # 代わりに Σ ∝ M/(R²) で R ∝ M^(1/2) (thin disk with fixed surface density slope)
    print("  別の仮定: Σの半径分布が M^(1/2) スケール")
    print("  Σ_mean = M/(πR²) → R ~ M^(β), ならば Σ ~ M^(1-2β)")
    print("  α_Σ/α_M = 1/(1-2β) とする。")
    print(f"  α_M = 1/3, α_Σ = 5/3 → 比 = {5/3 / (1/3):.2f} = 5")
    print(f"  → 1-2β = 1/5 → β = 2/5 = 0.4")
    print()
    print("  意味: R ∝ M^(0.4). 経験的に銀河のTully-Fisher近傍では R ~ M^(0.3-0.5)")

    # 実測 R-M 関係を SPARC で計算
    Ms = np.array([g[2] for g in all_gals])
    Rs = np.array([g[1][-1][0] for g in all_gals])  # R_last

    logM = np.log10(Ms)
    logR = np.log10(Rs)
    beta_fit, logR0 = np.polyfit(logM, logR, 1)
    print(f"\n  SPARCでのR-M関係: log R = {beta_fit:.3f} × log M + {logR0:.2f}")
    print(f"  β = {beta_fit:.3f}")
    print(f"  予測 α_Σ/α_M = 1/(1-2β) = {1/(1-2*beta_fit):.3f}")
    print(f"  → α_Σ = α_M × 1/(1-2β) = {1/3 * 1/(1-2*beta_fit):.3f}")
    print(f"  (観測 α_Σ = {alpha_S:.3f})")

    # ================================================================
    # Test 4: CDT的解釈
    # ================================================================
    print("\n--- Test 4: CDT的解釈 ---")
    print()
    print("  CDT Ambjørn+2005 データに α_Σ = 5/3 を当てはめると？")
    print("  CDT spectral dim は σ (diffusion time) の関数")
    print("  銀河のΣと対応させるにはσ ↔ Σの対応が必要")
    print()
    print("  単純化: σ ~ R² (diffusionは平方距離に比例)")
    print("  Σ = M/R² = M/σ")
    print("  α_Σ = 5/3 → u ~ Σ^(5/3) = (M/σ)^(5/3)")
    print()
    print("  M固定でσ変化 → u ~ σ^(-5/3)")
    print("  CDTの元公式 u ~ σ^γ で γ = -5/3?")
    print("  Ambjørnは γ = 1 を見つけた → 符号が逆")
    print()
    print("  → 単純なσ↔Σ対応では CDT との整合性はつかない")

    # ================================================================
    # Test 5: BIC比較
    # ================================================================
    print("\n--- Test 5: パラメータ数の比較 (BIC) ---")

    N_data = sum(len(pts) for _, pts, _, _ in all_gals)
    print(f"  データ点数: {N_data}")

    # McGaugh: 1 param (Y_disk)
    # p(M) α=1/3 fixed: 2 params (Y_disk, M0)
    # p(Σ) α=5/3 fixed: 2 params (Y_disk, Σ0 with theoretical relation to a0? or free)
    # p(Σ) with Σ0 = a0/(4π²G) fixed: 1 param (Y_disk, α_Σ=5/3)

    rms_mcg = minimize_scalar(lambda Y: global_rms_pSigma(all_gals, Y, 1e99, 1.0),
                              bounds=(0.0, 2.0), method='bounded').fun
    # Note: Σ0 → ∞ makes u → 0, p → 0, but that's not McGaugh's p=0.5

    # Use actual McGaugh
    def mcg_rms(gal_list, Y):
        resid = []
        for gid, pts, M, S in gal_list:
            for R_kpc, Vobs, eVobs, Vdisk, Vgas, Vbul in pts:
                R = R_kpc*KPC
                g_obs = (Vobs*1e3)**2/R
                g_bar = (Y*(Vdisk*1e3)**2 + np.sign(Vgas)*(Vgas*1e3)**2 +
                         0.7*(Vbul*1e3)**2)/R
                if g_bar <= 0 or g_obs <= 0: continue
                x = g_bar/A0
                mu = max(1-np.exp(-max(x, 1e-20)**0.5), 1e-20)
                resid.append(np.log10(g_obs)-np.log10(g_bar/mu))
        return np.sqrt(np.mean(np.array(resid)**2)) if resid else 9999

    rms_mcg = minimize_scalar(lambda Y: mcg_rms(all_gals, Y),
                              bounds=(0.0, 2.0), method='bounded').fun

    # p(M) with α=1/3
    def pM_rms_fixed(params):
        Y, logM0 = params
        return global_rms_pSigma(all_gals, Y, 1.0, 1.0)  # dummy - needs rewrite

    # Simpler: compute BIC from chi² approximation
    def chi2_from_rms(rms, N):
        return (rms * np.log(10))**2 * N

    bic_mcg = N_data * np.log10(rms_mcg)**2 * np.log(10)**2 + 1 * np.log(N_data)

    # p(Σ) with Σ₀ theoretical, α=5/3 fixed
    Sigma0_theory_Msun_kpc2 = Sigma0_theory_SI / MSUN * KPC**2
    rms_pS_theory = minimize_scalar(
        lambda Y: global_rms_pSigma(all_gals, Y, Sigma0_theory_Msun_kpc2, 5/3),
        bounds=(0.0, 2.0), method='bounded').fun
    print(f"  p(Σ), Σ₀=a₀/(4π²G), α=5/3 (theory, 1 param): RMS = {rms_pS_theory:.4f}")
    print(f"  McGaugh (1 param):                              RMS = {rms_mcg:.4f}")

    # Δ dof = 0. Same number of free params (just Y_disk).
    # Improvement:
    imp_theory = (rms_mcg - rms_pS_theory)/rms_mcg*100
    print(f"  → 同数パラメータで p(Σ) が {imp_theory:+.1f}% 改善")

    # p(Σ) with both Σ₀ and α free (3 params)
    res_free = minimize(opt_pS, [0.5, 7.5, 1.5],
                        bounds=[(0.0, 2.0), (4.0, 12.0), (0.05, 3.0)],
                        method='L-BFGS-B')
    rms_pS_free = res_free.fun
    imp_free = (rms_mcg - rms_pS_free)/rms_mcg*100
    print(f"  p(Σ) fully free (3 params): RMS = {rms_pS_free:.4f} ({imp_free:+.1f}%)")

    # BIC calculation
    N_approx_chi2_mcg = N_data * (rms_mcg * np.log(10))**2
    N_approx_chi2_pS_theory = N_data * (rms_pS_theory * np.log(10))**2
    N_approx_chi2_pS_free = N_data * (rms_pS_free * np.log(10))**2

    bic_mcg = N_approx_chi2_mcg + 1 * np.log(N_data)
    bic_pS_theory = N_approx_chi2_pS_theory + 1 * np.log(N_data)  # same 1 param (Y only)
    bic_pS_free = N_approx_chi2_pS_free + 3 * np.log(N_data)

    print(f"\n  BIC (lower is better):")
    print(f"    McGaugh:      {bic_mcg:.1f}")
    print(f"    p(Σ) theory:  {bic_pS_theory:.1f}  ΔBIC = {bic_mcg - bic_pS_theory:.1f}")
    print(f"    p(Σ) free:    {bic_pS_free:.1f}  ΔBIC = {bic_mcg - bic_pS_free:.1f}")


if __name__ == "__main__":
    main()
