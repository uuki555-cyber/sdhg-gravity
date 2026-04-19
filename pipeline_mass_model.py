"""
Mass model pipeline: HI data + WISE photometry -> g_bar, g_obs -> RAR test.

Step 1: HI moment maps -> radial HI surface density -> V_gas(R)
Step 2: WISE 3.4um image -> radial stellar surface density -> V_disk(R)
Step 3: g_bar = (Y_disk * V_disk^2 + V_gas^2) / R
Step 4: g_obs = V_obs^2 / R (from velocity field or literature)

Usage:
    python pipeline_mass_model.py
"""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import os

G_CGS = 6.674e-8      # cm^3 g^-1 s^-2
MSUN_G = 1.989e33      # grams
PC_CM = 3.086e18       # cm
KPC_CM = 3.086e21


def load_moment_map(filepath):
    """Load FITS moment map, return data + WCS."""
    hdu = fits.open(filepath)
    data = hdu[0].data
    header = hdu[0].header
    # Handle 3D/4D cubes (squeeze extra dimensions)
    while data.ndim > 2:
        data = data[0]
    wcs = WCS(header, naxis=2)
    hdu.close()
    return data, header, wcs


def hi_surface_density(mom0_data, header):
    """Convert HI column density map to surface density in Msun/pc^2.

    MOM0 is in Jy/beam * m/s. Convert to column density:
    N_HI [cm^-2] = 1.823e18 * integral(T_b dv) [K km/s]
    Sigma_HI [Msun/pc^2] = 1.25e-2 * N_HI / 1e20

    For THINGS: MOM0 in Jy/beam * km/s.
    N_HI = 1.823e18 * MOM0 / (beam_area * 606) [approx]
    """
    # Get beam size
    bmaj = header.get('BMAJ', 0.001) * 3600  # arcsec
    bmin = header.get('BMIN', 0.001) * 3600
    cdelt = abs(header.get('CDELT1', header.get('CD1_1', 1e-4))) * 3600  # arcsec/pixel

    beam_area_pix = np.pi * bmaj * bmin / (4 * np.log(2)) / cdelt**2

    # Convert Jy/beam*m/s to K*km/s
    # T_b = 1.36e3 * S / (bmaj * bmin) for 21cm [K per Jy/beam]
    # Then N_HI = 1.823e18 * T_b * dv [cm^-2]

    # Simpler: N_HI [cm^-2] = 1.823e18 * 1.36e3 / (bmaj * bmin) * MOM0 [Jy/beam * km/s]
    # But MOM0 is already in Jy/beam * m/s in THINGS
    bunit = header.get('BUNIT', '')
    if 'JY/B*M/S' in bunit.upper() or 'JY/BEAM*M/S' in bunit.upper():
        mom0_kms = mom0_data / 1000  # m/s to km/s
    elif 'JY/B*KM/S' in bunit.upper():
        mom0_kms = mom0_data
    else:
        # Assume m/s
        mom0_kms = mom0_data / 1000

    # Column density: N_HI = 1.823e18 * MOM0_Kkms
    # where MOM0_Kkms = 605.7 * MOM0_Jybeam_kms / (bmaj * bmin) [K km/s]
    T_b_dv = 605.7 * mom0_kms / (bmaj * bmin)  # K km/s
    N_HI = 1.823e18 * T_b_dv  # cm^-2

    # Surface density including helium factor (1.33)
    sigma_HI = 1.33 * N_HI * 1.67e-24 / (MSUN_G / (PC_CM**2))  # Msun/pc^2

    return sigma_HI


def azimuthal_profile(data, header, center_pix, pa_deg, inc_deg, dist_mpc, rmax_kpc=20, nbins=30):
    """Compute azimuthally averaged radial profile.

    Returns R_kpc, mean_value arrays.
    """
    ny, nx = data.shape
    cdelt = abs(header.get('CDELT1', header.get('CD1_1', 1e-4)))  # deg/pixel
    arcsec_per_pix = cdelt * 3600
    kpc_per_arcsec = dist_mpc * 1e3 * np.tan(np.radians(1/3600))  # small angle
    kpc_per_pix = arcsec_per_pix * kpc_per_arcsec

    cy, cx = center_pix
    y, x = np.mgrid[0:ny, 0:nx]

    # Deproject
    dx = (x - cx) * kpc_per_pix
    dy = (y - cy) * kpc_per_pix
    pa_rad = np.radians(pa_deg)
    cos_i = np.cos(np.radians(inc_deg))

    x_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
    y_rot = (-dx * np.sin(pa_rad) + dy * np.cos(pa_rad)) / max(cos_i, 0.1)
    R = np.sqrt(x_rot**2 + y_rot**2)

    # Bin
    r_edges = np.linspace(0, rmax_kpc, nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    profile = np.zeros(nbins)

    valid = np.isfinite(data) & (data != 0)
    for i in range(nbins):
        mask = valid & (R >= r_edges[i]) & (R < r_edges[i+1])
        if mask.sum() > 0:
            profile[i] = np.mean(data[mask])

    return r_centers, profile


def v_gas_from_sigma(R_kpc, sigma_gas_msun_pc2):
    """Compute V_gas(R) from gas surface density profile.

    Uses thin disk approximation: V^2(R) = 4*pi*G*Sigma_0*R * I(y)
    where I(y) involves Bessel functions.
    Simplified: V^2 ~ pi*G*Sigma*R for each annulus contribution.
    """
    R_cm = R_kpc * KPC_CM
    sigma_cgs = sigma_gas_msun_pc2 * MSUN_G / PC_CM**2  # g/cm^2

    V2 = np.zeros(len(R_kpc))
    dR = R_cm[1] - R_cm[0] if len(R_cm) > 1 else R_cm[0]

    for i in range(len(R_kpc)):
        if R_cm[i] <= 0:
            continue
        # Sum contributions from all annuli (Freeman 1970 approximation)
        M_enc = 0
        for j in range(i + 1):
            M_enc += 2 * np.pi * R_cm[j] * sigma_cgs[j] * dR
        V2[i] = G_CGS * M_enc / R_cm[i]

    V_kms = np.sqrt(np.maximum(V2, 0)) / 1e5  # cm/s to km/s
    return V_kms


def main():
    print("=" * 60)
    print("Mass Model Pipeline: THINGS HI data")
    print("=" * 60)

    # NGC 925 parameters (from de Blok+2008)
    gal = {
        'name': 'NGC925',
        'dist_mpc': 9.16,
        'pa_deg': 287,
        'inc_deg': 66,
        'vsys_kms': 553,
        'mom0_file': 'data/things/NGC_925_NA_MOM0_THINGS.FITS',
        'mom1_file': 'data/things/NGC_925_NA_MOM1_THINGS.FITS',
    }

    print(f"\nProcessing {gal['name']} (D={gal['dist_mpc']} Mpc)")

    # Load moment maps
    mom0, hdr0, wcs0 = load_moment_map(gal['mom0_file'])
    mom1, hdr1, wcs1 = load_moment_map(gal['mom1_file'])

    print(f"  MOM0 shape: {mom0.shape}, BUNIT: {hdr0.get('BUNIT','?')}")
    print(f"  MOM1 shape: {mom1.shape}")

    # Find center (peak of MOM0)
    valid = np.isfinite(mom0) & (mom0 > 0)
    if valid.sum() == 0:
        print("  No valid MOM0 data!")
        return
    cy, cx = np.unravel_index(np.argmax(mom0 * valid), mom0.shape)
    print(f"  Center pixel: ({cx}, {cy})")

    # HI surface density
    sigma_HI = hi_surface_density(mom0, hdr0)
    print(f"  Sigma_HI range: {np.nanmin(sigma_HI[valid]):.2f} - {np.nanmax(sigma_HI[valid]):.2f} Msun/pc^2")

    # Radial profile
    R_kpc, sigma_prof = azimuthal_profile(
        sigma_HI, hdr0, (cy, cx), gal['pa_deg'], gal['inc_deg'],
        gal['dist_mpc'], rmax_kpc=15, nbins=20)

    # V_gas
    V_gas = v_gas_from_sigma(R_kpc, sigma_prof)

    # V_obs from velocity field (MOM1)
    # Simple: take azimuthal average of |V - V_sys| / sin(i)
    vel_field = mom1.copy()
    vel_valid = np.isfinite(vel_field) & (vel_field != 0)
    vel_deproj = np.abs(vel_field - gal['vsys_kms'] * 1000) / np.sin(np.radians(gal['inc_deg']))

    R_kpc_v, V_obs_raw = azimuthal_profile(
        vel_deproj / 1000,  # m/s to km/s
        hdr1, (cy, cx), gal['pa_deg'], gal['inc_deg'],
        gal['dist_mpc'], rmax_kpc=15, nbins=20)

    # Results
    print(f"\n{'R(kpc)':>8} {'V_obs':>8} {'V_gas':>8} {'g_obs/a0':>10} {'g_bar/a0':>10}")
    print("-" * 48)

    a0 = 1.2e-10
    rar_points = []
    for i in range(len(R_kpc)):
        if R_kpc[i] < 0.5 or V_obs_raw[i] < 5:
            continue
        R_m = R_kpc[i] * KPC_CM / 100  # meters
        g_obs = (V_obs_raw[i] * 1e3)**2 / R_m
        # g_bar from gas only (no stellar disk yet)
        g_gas = (V_gas[i] * 1e3)**2 / R_m if R_m > 0 else 0

        if g_obs > 0 and g_gas > 0:
            print(f"{R_kpc[i]:>8.1f} {V_obs_raw[i]:>8.1f} {V_gas[i]:>8.1f} "
                  f"{g_obs/a0:>10.4f} {g_gas/a0:>10.4f}")
            rar_points.append((g_obs, g_gas))

    if rar_points:
        print(f"\n{len(rar_points)} RAR points extracted (gas-only g_bar)")
        print("Note: V_disk from WISE needed for full g_bar")
    else:
        print("\nNo valid RAR points extracted")


if __name__ == "__main__":
    main()
