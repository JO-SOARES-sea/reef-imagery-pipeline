"""Faixas de decisão do Noul "is_natural_reef" do TypeSafe Jev (offline, sem API)."""

import pytest

from src.analysis.gemini_analyzer import JEV_DECISION_LABELS, interpret_reef_noul
from src.constants import JEV_REEF_NO_MAX, JEV_REEF_YES_MIN


@pytest.mark.parametrize(
    "p, expected",
    [
        (0.95, "recife"),
        (JEV_REEF_YES_MIN, "recife"),
        (0.5, "revisao"),
        (0.6, "revisao"),
        (JEV_REEF_NO_MAX, "nao_recife"),
        (0.05, "nao_recife"),
    ],
)
def test_decision_bands(p, expected):
    assert interpret_reef_noul(p)["decision"] == expected


def test_half_is_undecided_not_yes():
    # 0.5 é indecisão do modelo — o limiar antigo (>= 0.5) contava-o como "recife"
    res = interpret_reef_noul(0.5)
    assert res["decision"] == "revisao"
    assert res["confidence"] == pytest.approx(0.0)


def test_confident_no_has_high_confidence():
    # p = 0.05 é um "não" seguro: antes aparecia como "5% de confiança"
    res = interpret_reef_noul(0.05)
    assert res["p_reef"] == pytest.approx(0.05)
    assert res["confidence"] == pytest.approx(0.9)


def test_not_executed_is_not_a_no():
    res = interpret_reef_noul(None)
    assert res == {"decision": "nao_executado", "p_reef": None, "confidence": None}


def test_every_decision_has_a_label():
    for p in (None, 0.0, 0.5, 1.0):
        assert interpret_reef_noul(p)["decision"] in JEV_DECISION_LABELS
