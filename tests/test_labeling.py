"""
Проверки на здравый смысл для src/labeling.py.

Round-trip тест — это и есть тот самый шаг "проверить, что разметка
соответствует модели", о котором просил руководитель: он проверяет, что наша
переимплементация DavisBOLD / DavisCMRO2 является корректной алгебраической
обратной функцией самой себя — это единственное свойство, которое должно
выполняться для ЛЮБОЙ корректной реализации модели Davis, независимо от
возможности напрямую прочитать текст методов из статьи.
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
    """Каноническая (без бага) прямая модель должна быть точной обратной
    функцией davis_cmro2_relchange - именно это математически и означает
    "разметка соответствует модели Davis"."""
    bold_rc = davis_bold_relchange_standard(a, b, m, cbf_rc, cmro2_rc)
    recovered_cmro2_rc = davis_cmro2_relchange(a, b, m, bold_rc, cbf_rc)
    assert recovered_cmro2_rc == pytest.approx(cmro2_rc, rel=1e-8)

    recovered_cbf_rc = davis_cbf_relchange(a, b, m, bold_rc, cmro2_rc)
    assert recovered_cbf_rc == pytest.approx(cbf_rc, rel=1e-8)


@pytest.mark.parametrize("a,b,m", [(0.2, 1.5, 0.05)])
@pytest.mark.parametrize("cbf_rc,cmro2_rc", [(0.3, 0.1), (-0.2, -0.05)])
def test_source_davis_bold_has_spurious_offset(a, b, m, cbf_rc, cmro2_rc):
    """Документирует расхождение, найденное в DavisBOLD() из исходного
    репозитория: оно ровно на 1 меньше канонической формулы и нигде реально
    не вызывается в объединённых ноутбуках (мёртвый код), поэтому не влияет
    на метки concordant/discordant, используемые в этом проекте."""
    buggy = davis_bold_relchange(a, b, m, cbf_rc, cmro2_rc)
    standard = davis_bold_relchange_standard(a, b, m, cbf_rc, cmro2_rc)
    assert buggy == pytest.approx(standard - 1, rel=1e-8)


@pytest.mark.parametrize(
    "bold,cmro2,expected",
    [
        (1.0, 1.0, 1.0),      # оба положительны -> concordant
        (-1.0, -1.0, 1.0),    # оба отрицательны -> concordant
        (1.0, -1.0, -1.0),    # противоположные знаки -> discordant
        (-1.0, 1.0, -1.0),    # противоположные знаки -> discordant
        (0.0, 1.0, 0.0),      # неопределено
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
