"""
Davis-model calibrated-BOLD formulas and the concordant/discordant voxel label.

Reimplemented from, and verified line-by-line against, the authors' own code in
https://github.com/NeuroenergeticsLab/two_modes_of_hemodynamics (Epp et al.,
"Two distinct modes of hemodynamic responses across the human cortex",
preprint: https://www.biorxiv.org/content/10.1101/2023.12.08.570806).

Source functions cross-checked (see combined_pipeline.py produced by
merge_notebooks.py from that repo):
  - DavisBOLD, DavisCMRO2RelChange, DavisCBFRelChange  (Davis model, ~lines 498-521)
  - the concordant/discordant sign rule                 (~lines 10122-10124)

Note: link.springer.com, pubmed/ncbi, and biorxiv.org are all blocked by this
session's network egress policy, so the published methods text could not be
fetched directly. Verification here relies on (a) the authors' own reference
implementation above, and (b) the standard Davis et al. (1998) calibrated-BOLD
equation, which is algebraically self-consistent (see test_labeling.py's
round-trip check).
"""

import numpy as np


def davis_bold_relchange(a, b, m, cbf_relchange, cmro2_relchange):
    """
    Reproduces DavisBOLD() in the source repo EXACTLY, bug included:
        m * (1 - (CBF_task/CBF_base)**(a-b) * (CMRO2_task/CMRO2_base)**b) - 1

    KNOWN DISCREPANCY (found while verifying the labeling, see test_labeling.py):
    the canonical Davis et al. (1998) equation is
        dS/S0 = M * (1 - CBF_ratio**(a-b) * CMRO2_ratio**b)
    with NO trailing "-1". The source repo's DavisBOLD has one anyway, which
    breaks round-trip consistency with its own sibling DavisCMRO2RelChange /
    DavisCBFRelChange (those two correctly invert the equation WITHOUT the
    "-1" and are each other's exact inverse, see test_davis_round_trip).
    DavisBOLD is also the only one of the three not called anywhere in the
    merged notebooks (grep confirms zero call sites) - it is unused dead code
    in the original pipeline, so this bug does not affect any published
    result or this project's concordant/discordant labels: those are built
    from empirically measured BOLD_percchange and Fick's-principle CMRO2
    (CBF*OEF*CaO2), never from this function. Kept here only for a faithful,
    documented reproduction; use davis_bold_relchange_standard() below if you
    actually need the forward model.

    a: Grubb exponent (CBV = k * CBF**a)
    b: beta exponent (deoxyhemoglobin sensitivity, ~1.5 at 3T)
    m: calibration constant (max BOLD signal at full deoxyhemoglobin washout)
    cbf_relchange, cmro2_relchange: relative change (task/base - 1), NOT percent
    """
    cbf_ratio = 1 + cbf_relchange
    cmro2_ratio = 1 + cmro2_relchange
    return m * (1 - cbf_ratio ** (a - b) * cmro2_ratio ** b) - 1


def davis_bold_relchange_standard(a, b, m, cbf_relchange, cmro2_relchange):
    """
    Canonical Davis et al. (1998) forward model, without the source repo's
    extra "-1" (see davis_bold_relchange docstring). This is the version that
    is the exact algebraic inverse of davis_cmro2_relchange /
    davis_cbf_relchange below.
    """
    cbf_ratio = 1 + cbf_relchange
    cmro2_ratio = 1 + cmro2_relchange
    return m * (1 - cbf_ratio ** (a - b) * cmro2_ratio ** b)


def davis_cmro2_relchange(a, b, m, bold_relchange, cbf_relchange):
    """
    Relative CMRO2 change, solving the Davis model for CMRO2 given BOLD and CBF.

    Matches DavisCMRO2RelChange() in the source repo:
        product = 1 - BOLD/m
        rCMRO2  = (product / (CBF_task/CBF_base)**(a-b)) ** (1/b)
        return rCMRO2 - 1
    """
    cbf_ratio = 1 + cbf_relchange
    product = 1 - bold_relchange / m
    cmro2_ratio = (product / cbf_ratio ** (a - b)) ** (1 / b)
    return cmro2_ratio - 1


def davis_cbf_relchange(a, b, m, bold_relchange, cmro2_relchange):
    """
    Relative CBF change, solving the Davis model for CBF given BOLD and CMRO2.
    Matches DavisCBFRelChange() in the source repo.
    """
    cmro2_ratio = 1 + cmro2_relchange
    product = 1 - bold_relchange / m
    cbf_ratio = (product / cmro2_ratio ** b) ** (1 / (a - b))
    return cbf_ratio - 1


def concordance_label(bold_percchange, cmro2_percchange):
    """
    Per-voxel concordant/discordant label, exactly as computed in the source repo:

        conc_disc = sign(CMRO2_percchange) * sign(BOLD_percchange)
        +1 : both same sign  -> concordant ("red")
        -1 : opposite signs  -> discordant ("blue")
         0 : either value is exactly zero (undefined / excluded from analysis)

    bold_percchange, cmro2_percchange: arrays of matching shape, percent (or
    relative) signal change of BOLD and CMRO2 for the same contrast (e.g.
    task vs control).
    """
    bold_percchange = np.asarray(bold_percchange)
    cmro2_percchange = np.asarray(cmro2_percchange)
    return np.sign(cmro2_percchange) * np.sign(bold_percchange)


def binary_label_from_concordance(concordance):
    """
    Maps the {-1, 0, +1} concordance array to a binary classification target:
        1 = concordant, 0 = discordant, NaN = undefined (voxel excluded).
    """
    concordance = np.asarray(concordance, dtype=float)
    out = np.full(concordance.shape, np.nan, dtype=float)
    out[concordance > 0] = 1.0
    out[concordance < 0] = 0.0
    return out
