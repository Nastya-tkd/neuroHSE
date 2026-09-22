"""
Формулы калиброванного BOLD модели Дэвиса и метка concordant/discordant для
вокселя.

Реализовано заново и построчно сверено с собственным кодом авторов из
https://github.com/NeuroenergeticsLab/two_modes_of_hemodynamics (Epp et al.,
"Two distinct modes of hemodynamic responses across the human cortex",
препринт: https://www.biorxiv.org/content/10.1101/2023.12.08.570806).

Исходные функции сверены (см. combined_pipeline.py, полученный
merge_notebooks.py из этого репозитория):
  - DavisBOLD, DavisCMRO2RelChange, DavisCBFRelChange  (модель Дэвиса, ~строки 498-521)
  - правило знаков concordant/discordant                 (~строки 10122-10124)

Примечание: link.springer.com, pubmed/ncbi и biorxiv.org заблокированы
политикой сетевого доступа этой сессии, поэтому текст опубликованных
методов не удалось получить напрямую. Проверка здесь опирается на (a)
собственную эталонную реализацию авторов выше и (b) стандартное уравнение
калиброванного BOLD Дэвиса и др. (1998), которое алгебраически
самосогласовано (см. проверку типа round-trip в test_labeling.py).
"""

import numpy as np


def davis_bold_relchange(a, b, m, cbf_relchange, cmro2_relchange):
    """
    Воспроизводит DavisBOLD() из исходного репозитория ТОЧНО, включая баг:
        m * (1 - (CBF_task/CBF_base)**(a-b) * (CMRO2_task/CMRO2_base)**b) - 1

    ИЗВЕСТНОЕ РАСХОЖДЕНИЕ (обнаружено при проверке разметки, см.
    test_labeling.py): каноническое уравнение Дэвиса и др. (1998) имеет вид
        dS/S0 = M * (1 - CBF_ratio**(a-b) * CMRO2_ratio**b)
    БЕЗ завершающего "-1". В DavisBOLD исходного репозитория он всё же есть,
    что нарушает согласованность round-trip с собственными парными
    функциями DavisCMRO2RelChange / DavisCBFRelChange (эти две корректно
    обращают уравнение БЕЗ "-1" и являются точными обратными друг для
    друга, см. test_davis_round_trip). DavisBOLD также единственная из трёх
    функций, которая нигде не вызывается в объединённых notebook'ах (grep
    подтверждает ноль мест вызова) - это неиспользуемый мёртвый код в
    исходном конвейере, поэтому данный баг не влияет ни на один
    опубликованный результат, ни на метки concordant/discordant этого
    проекта: они строятся из эмпирически измеренного BOLD_percchange и
    CMRO2 по принципу Фика (CBF*OEF*CaO2), а не из этой функции. Оставлено
    здесь только ради точного, документированного воспроизведения;
    используйте davis_bold_relchange_standard() ниже, если вам
    действительно нужна прямая модель.

    a: показатель степени Граба (CBV = k * CBF**a)
    b: показатель степени бета (чувствительность к дезоксигемоглобину, ~1.5 при 3Т)
    m: калибровочная константа (максимальный BOLD-сигнал при полном вымывании дезоксигемоглобина)
    cbf_relchange, cmro2_relchange: относительное изменение (task/base - 1), НЕ проценты
    """
    cbf_ratio = 1 + cbf_relchange
    cmro2_ratio = 1 + cmro2_relchange
    return m * (1 - cbf_ratio ** (a - b) * cmro2_ratio ** b) - 1


def davis_bold_relchange_standard(a, b, m, cbf_relchange, cmro2_relchange):
    """
    Каноническая прямая модель Дэвиса и др. (1998), без лишнего "-1" из
    исходного репозитория (см. docstring davis_bold_relchange). Это версия,
    являющаяся точной алгебраической обратной для davis_cmro2_relchange /
    davis_cbf_relchange ниже.
    """
    cbf_ratio = 1 + cbf_relchange
    cmro2_ratio = 1 + cmro2_relchange
    return m * (1 - cbf_ratio ** (a - b) * cmro2_ratio ** b)


def davis_cmro2_relchange(a, b, m, bold_relchange, cbf_relchange):
    """
    Относительное изменение CMRO2, полученное решением модели Дэвиса
    относительно CMRO2 при заданных BOLD и CBF.

    Соответствует DavisCMRO2RelChange() из исходного репозитория:
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
    Относительное изменение CBF, полученное решением модели Дэвиса
    относительно CBF при заданных BOLD и CMRO2.
    Соответствует DavisCBFRelChange() из исходного репозитория.
    """
    cmro2_ratio = 1 + cmro2_relchange
    product = 1 - bold_relchange / m
    cbf_ratio = (product / cmro2_ratio ** b) ** (1 / (a - b))
    return cbf_ratio - 1


def concordance_label(bold_percchange, cmro2_percchange):
    """
    Метка concordant/discordant по вокселю, вычисляемая точно так же, как в
    исходном репозитории:

        conc_disc = sign(CMRO2_percchange) * sign(BOLD_percchange)
        +1 : одинаковый знак у обоих  -> concordant ("красный")
        -1 : противоположные знаки    -> discordant ("синий")
         0 : одно из значений точно равно нулю (неопределено / исключается из анализа)

    bold_percchange, cmro2_percchange: массивы одинаковой формы, процентное
    (или относительное) изменение сигнала BOLD и CMRO2 для одного и того же
    контраста (например, задача vs control).
    """
    bold_percchange = np.asarray(bold_percchange)
    cmro2_percchange = np.asarray(cmro2_percchange)
    return np.sign(cmro2_percchange) * np.sign(bold_percchange)


def binary_label_from_concordance(concordance):
    """
    Отображает массив concordance {-1, 0, +1} в бинарную целевую переменную
    классификации:
        1 = concordant, 0 = discordant, NaN = не определено (воксель исключён).
    """
    concordance = np.asarray(concordance, dtype=float)
    out = np.full(concordance.shape, np.nan, dtype=float)
    out[concordance > 0] = 1.0
    out[concordance < 0] = 0.0
    return out
