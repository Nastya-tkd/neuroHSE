"""
Sanity checks for src/labeling.py.

The round-trip test is the actual "verify the labeling matches the model"
step the supervisor asked for: it checks that our DavisBOLD / DavisCMRO2
reimplementation is the correct algebraic inverse of itself, which is the
one property that must hold for ANY correct implementation of the Davis
model, independent of being able to read the paper's methods text directly.
"""

import numpy as np
import pytest

from src.labeling import (
    davis_bold_relchange,
    davis_bold_relchange_standard,
    davis_cmro2_relchange,
    davis_cbf_relchange,
    concordance_label,
    binary_label_from_concordance,
)


@pytest.mark.parametrize("a,b,m", [(0.2, 1.5, 0.05), (0.38, 1.3, 0.08)])
@pytest.mark.parametrize("cbf_rc,cmro2_rc", [(0.3, 0.1), (-0.2, -0.05), (0.5, 0.4)])
def test_davis_round_trip(a, b, m, cbf_rc, cmro2_rc):
    """The canonical (bug-free) forward model must be the exact inverse of
    davis_cmro2_relchange - this is what "the labeling matches the Davis
    model" actually means mathematically."""
    bold_rc = davis_bold_relchange_standard(a, b, m, cbf_rc, cmro2_rc)
    recovered_cmro2_rc = davis_cmro2_relchange(a, b, m, bold_rc, cbf_rc)
    assert recovered_cmro2_rc == pytest.approx(cmro2_rc, rel=1e-8)

    recovered_cbf_rc = davis_cbf_relchange(a, b, m, bold_rc, cmro2_rc)
    assert recovered_cbf_rc == pytest.approx(cbf_rc, rel=1e-8)


@pytest.mark.parametrize("a,b,m", [(0.2, 1.5, 0.05)])
@pytest.mark.parametrize("cbf_rc,cmro2_rc", [(0.3, 0.1), (-0.2, -0.05)])
def test_source_davis_bold_has_spurious_offset(a, b, m, cbf_rc, cmro2_rc):
    """Documents the discrepancy found in the source repo's DavisBOLD(): it
    is exactly 1 less than the canonical formula, and is never actually
    called anywhere in the merged notebooks (dead code), so it does not
    affect the concordant/discordant labels used in this project."""
    buggy = davis_bold_relchange(a, b, m, cbf_rc, cmro2_rc)
    standard = davis_bold_relchange_standard(a, b, m, cbf_rc, cmro2_rc)
    assert buggy == pytest.approx(standard - 1, rel=1e-8)


@pytest.mark.parametrize(
    "bold,cmro2,expected",
    [
        (1.0, 1.0, 1.0),      # both positive -> concordant
        (-1.0, -1.0, 1.0),    # both negative -> concordant
        (1.0, -1.0, -1.0),    # opposite signs -> discordant
        (-1.0, 1.0, -1.0),    # opposite signs -> discordant
        (0.0, 1.0, 0.0),      # undefined
    ],
)
def test_concordance_sign_rule(bold, cmro2, expected):
    assert concordance_label(bold, cmro2) == expected


def test_concordance_label_array():
    bold = np.array([1.0, -1.0, 1.0, -1.0, 0.0])
    cmro2 = np.array([1.0, -1.0, -1.0, 1.0, 2.0])
    conc = concordance_label(bold, cmro2)
    np.testing.assert_array_equal(conc, [1.0, 1.0, -1.0, -1.0, 0.0])


def test_binary_label_from_concordance():
    conc = np.array([1.0, -1.0, 0.0])
    binary = binary_label_from_concordance(conc)
    assert binary[0] == 1.0
    assert binary[1] == 0.0
    assert np.isnan(binary[2])
