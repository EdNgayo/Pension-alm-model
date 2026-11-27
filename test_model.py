"""Sanity checks. Run with:  python -m pytest"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pension_model import (  # noqa: E402
    Bond, current_portfolio, duration_matched_portfolio, effective_duration,
    expected_cash_flows, make_sample_plan, present_value, survival_prob,
)

Y = 0.045


@pytest.fixture(scope="module")
def cfs():
    return expected_cash_flows(make_sample_plan())


def test_survival_is_a_decreasing_probability():
    p = survival_prob(65, np.arange(0, 56))
    assert p[0] == pytest.approx(1.0)
    assert np.all((p > 0) & (p <= 1))
    assert np.all(np.diff(p) < 0)


def test_pv_at_zero_rate_is_sum_of_cash_flows(cfs):
    assert present_value(cfs, 0.0) == pytest.approx(cfs.sum())


def test_liability_falls_when_rates_rise(cfs):
    assert present_value(cfs, Y + 0.01) < present_value(cfs, Y)


def test_strip_duration_matches_closed_form():
    # Modified duration of a zero-coupon bond = T / (1 + y)
    strip = Bond("strip", coupon=0.0, maturity=30)
    assert strip.duration(Y) == pytest.approx(30 / (1 + Y), rel=1e-4)


def test_par_bond_prices_at_100():
    assert Bond("par", coupon=Y, maturity=10).price(Y) == pytest.approx(100.0)


def test_matched_portfolio_dv01_equals_liability_dv01(cfs):
    liab = present_value(cfs, Y)
    d_l = effective_duration(lambda r: present_value(cfs, r), Y)
    port, w = duration_matched_portfolio(0.95 * liab, liab, d_l, Y)
    liab_dv01 = present_value(cfs, Y - 0.0001) - liab
    assert 0.0 < w < 1.0
    assert port.dv01(Y) == pytest.approx(liab_dv01, rel=1e-3)


def test_portfolios_start_at_target_funding(cfs):
    liab = present_value(cfs, Y)
    port = current_portfolio(0.95 * liab)
    assert port.value(Y, 0.0) / liab == pytest.approx(0.95)
