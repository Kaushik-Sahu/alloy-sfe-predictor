"""
Shared utility functions for the SFE project.
Owned jointly by: Person 1 (Data & Physics Lead) & Person 3 (App & Integration Lead).

CONTRACTS.md § 3 locks these signatures — do NOT rename or change types.
"""


# ─── VEC Map (Valence Electron Concentration) ───────────────────────
VEC_MAP: dict[str, float] = {
    "Al": 3,
    "Cr": 6,
    "Mn": 7,
    "Fe": 8,
    "Co": 9,
    "Ni": 10,
}


def normalize_composition(comp: dict[str, float]) -> dict[str, float]:
    """
    Drop values <= 0, then scale remaining mole fractions to sum to 1.0.

    Parameters
    ----------
    comp : dict[str, float]
        Element -> mole fraction mapping (may be un-normalized).

    Returns
    -------
    dict[str, float]
        Normalized composition with positive fractions summing to 1.0.

    Raises
    ------
    ValueError
        If no positive fractions remain after filtering.
    """
    positive = {k: v for k, v in comp.items() if v > 0}
    if not positive:
        raise ValueError("No positive mole fractions in composition")
    total = sum(positive.values())
    return {k: v / total for k, v in positive.items()}


def compute_vec(comp: dict[str, float]) -> float:
    """
    Return composition-weighted Valence Electron Concentration.

    Parameters
    ----------
    comp : dict[str, float]
        Element -> mole fraction mapping (should sum to ~1.0).

    Returns
    -------
    float
        Weighted VEC value.
    """
    return sum(VEC_MAP[el] * frac for el, frac in comp.items() if el in VEC_MAP)


def make_canonical_formula(comp: dict[str, float]) -> str:
    """
    Return alphabetical formula string with 4-decimal mole fractions,
    omitting elements with fraction < 0.02 (zero or trace elements).

    Example: {"Fe": 0.72, "Cr": 0.18, "Ni": 0.10} -> "Cr0.1800Fe0.7200Ni0.1000"

    Parameters
    ----------
    comp : dict[str, float]
        Element -> mole fraction mapping.

    Returns
    -------
    str
        Canonical pymatgen-compatible formula string.
    """
    parts = []
    for el in sorted(comp.keys()):
        frac = round(comp[el], 4)
        if frac >= 0.02:
            parts.append(f"{el}{frac:.4f}")
    return "".join(parts)


def classify_sfe_regime(sfe: float) -> str:
    """
    Classify SFE into deformation regime.

    Parameters
    ----------
    sfe : float
        Stacking fault energy in mJ/m².

    Returns
    -------
    str
        "TRIP" if sfe < 20.0,
        "TWIP" if 20.0 <= sfe <= 40.0,
        "Glide" if sfe > 40.0.
    """
    if sfe < 20.0:
        return "TRIP"
    elif sfe <= 40.0:
        return "TWIP"
    else:
        return "Glide"
