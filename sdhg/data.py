"""
Data loading utilities for SPARC and cluster datasets.
"""
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
MSUN = 1.989e30
KPC = 3.086e19
MPC = KPC * 1e3


def load_sparc(filepath=None):
    """Load SPARC mass model data (Lelli+ 2016).

    Returns
    -------
    galaxies : dict
        {galaxy_id: [(R_kpc, Vobs_kms, eVobs_kms, Vdisk_kms, Vgas_kms, Vbul_kms), ...]}
    """
    if filepath is None:
        filepath = DATA_DIR / "sparc_data.mrt"

    galaxies = {}
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(("Title", "Auth", "Table", "=", "-", " ", "Byte", "Note")):
                continue
            if len(line.strip()) < 40:
                continue
            try:
                gid = line[0:11].strip()
                R = float(line[19:25])      # kpc
                Vobs = float(line[26:32])    # km/s
                eVobs = float(line[33:38])   # km/s
                Vgas = float(line[39:45])    # km/s
                Vdisk = float(line[46:52])   # km/s
                Vbul = float(line[53:59])    # km/s
                if gid not in galaxies:
                    galaxies[gid] = []
                galaxies[gid].append((R, Vobs, eVobs, Vdisk, Vgas, Vbul))
            except (ValueError, IndexError):
                continue
    return galaxies


def load_clusters():
    """Load galaxy cluster data (Vikhlinin+ 2006, X-COP/Ettori+ 2019).

    Returns
    -------
    clusters : list of dict
        Each dict has: name, M_bar_sun, R500_m, M500_sun, source.
    """
    # M_gas in 10^13 Msun, R500 in Mpc, M500 in 10^14 Msun
    raw = [
        ("A133",    2.64, 0.998,  3.14, "V06"),
        ("A383",    3.78, 0.956,  3.10, "V06"),
        ("A478",    9.23, 1.359,  7.83, "V06"),
        ("A907",    5.70, 1.117,  4.71, "V06"),
        ("A1413",   8.09, 1.339,  7.78, "V06"),
        ("A1795",   6.44, 1.283,  6.57, "V06"),
        ("A2029",  10.03, 1.380,  8.29, "V06"),
        ("A2390",  15.01, 1.448, 10.88, "V06"),
        ("A85x",    8.48, 1.235,  5.65, "X19"),
        ("A644",    7.47, 1.230,  5.66, "X19"),
        ("A1795x",  6.44, 1.153,  4.63, "X19"),
        ("A2029x", 12.20, 1.414,  8.65, "X19"),
        ("A2142",  14.14, 1.424,  8.95, "X19"),
        ("A2255",   8.05, 1.196,  5.26, "X19"),
        ("A3158",   6.18, 1.123,  4.26, "X19"),
        ("A3266",  11.62, 1.430,  8.80, "X19"),
        ("RXC1825", 5.43, 1.105,  4.08, "X19"),
    ]
    clusters = []
    for name, Mg13, R5_mpc, M14, src in raw:
        clusters.append({
            "name": name,
            "M_bar_sun": Mg13 * 1e13 * 1.15,  # gas + stars
            "R500_m": R5_mpc * MPC,
            "M500_sun": M14 * 1e14,
            "source": src,
        })
    return clusters


def load_little_things(dirpath=None):
    """Load LITTLE THINGS rotation curves (Iorio+ 2017).

    Returns
    -------
    galaxies : dict
        {galaxy_id: [(R_kpc, Vc_kms, eVc_kms, Sigma_HI), ...]}
    """
    if dirpath is None:
        dirpath = DATA_DIR / "little_things" / "finalrot"

    galaxies = {}
    for fpath in sorted(dirpath.glob("*_onlinetab.txt")):
        gname = fpath.stem.replace("_onlinetab", "")
        if gname.endswith("b"):
            continue
        pts = []
        with open(fpath) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 12:
                    try:
                        R_kpc = float(parts[1])
                        Vc = float(parts[6])
                        eVc = float(parts[7])
                        Sigma = float(parts[10])
                        if R_kpc > 0 and Vc > 0:
                            pts.append((R_kpc, Vc, eVc, Sigma))
                    except (ValueError, IndexError):
                        continue
        if len(pts) >= 4:
            galaxies[gname] = pts
    return galaxies
