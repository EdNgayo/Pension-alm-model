"""
Pension liability model: funded-status sensitivity to interest rate shocks.

The model has four parts:
  1. A sample defined benefit (DB) plan: retirees and active members.
  2. Expected benefit cash flows, using the SOA Standard Ultimate Life Table
     (Makeham mortality) for survival probabilities.
  3. Liability value, duration and convexity at a flat discount rate.
  4. Two asset portfolios (current mix vs duration-matched) and the funded
     status A / L under parallel rate shocks.

Simplifications (see README): flat yield curve, same curve for assets and
liabilities, no inflation indexation, no salary growth, no new entrants,
equities assumed insensitive to rates.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Mortality: SOA Standard Ultimate Life Table (Makeham's law)
#    mu_x = A + B * c^x
# ---------------------------------------------------------------------------
A_MAKEHAM = 0.00022
B_MAKEHAM = 2.7e-6
C_MAKEHAM = 1.124
MAX_AGE = 120
RETIREMENT_AGE = 65


def survival_prob(age: float, t: np.ndarray) -> np.ndarray:
    """Probability that a life aged `age` survives t more years (tpx)."""
    t = np.asarray(t, dtype=float)
    log_c = np.log(C_MAKEHAM)
    return np.exp(
        -A_MAKEHAM * t
        - B_MAKEHAM * C_MAKEHAM**age * (C_MAKEHAM**t - 1.0) / log_c
    )


# ---------------------------------------------------------------------------
# 2. Sample plan and expected cash flows
# ---------------------------------------------------------------------------
def make_sample_plan() -> pd.DataFrame:
    """
    A small, made-up DB plan. Each row is a group of identical members.

    Actives: accrued pension = 1.5% x years of service x salary, paid from 65.
    Retirees: already receiving a fixed annual pension.
    """
    rows = []
    for age in range(30, 65):  # actives, one group per age
        service = age - 25
        salary = 70_000
        accrued = 0.015 * service * salary
        rows.append({"status": "active", "age": age, "count": 20,
                     "annual_benefit": accrued})
    for age in range(65, 95):  # retirees, fewer at older ages
        count = max(round(28 - 0.8 * (age - 65)), 4)
        rows.append({"status": "retiree", "age": age, "count": count,
                     "annual_benefit": 24_000})
    return pd.DataFrame(rows)


def expected_cash_flows(plan: pd.DataFrame, indexation: float = 0.0) -> pd.Series:
    """
    Expected benefit payments for years t = 1, 2, ..., paid at the end of
    each year while the member is alive and at least 65.
    """
    horizon = MAX_AGE - int(plan["age"].min())
    t = np.arange(1, horizon + 1)
    total = np.zeros(horizon)
    growth = (1.0 + indexation) ** t  # pension grows by (1 + i) each year
    for _, m in plan.iterrows():
        alive = survival_prob(m["age"], t)
        in_payment = (m["age"] + t) >= RETIREMENT_AGE
        within_table = (m["age"] + t) <= MAX_AGE
        total += m["count"] * m["annual_benefit"] * growth * alive * in_payment * within_table
    return pd.Series(total, index=pd.Index(t, name="year"), name="cash_flow")


# ---------------------------------------------------------------------------
# 3. Present value, duration, convexity
# ---------------------------------------------------------------------------
def present_value(cash_flows: pd.Series, y: float) -> float:
    """PV of cash flows at a flat annual rate y."""
    t = cash_flows.index.to_numpy(dtype=float)
    return float(np.sum(cash_flows.to_numpy() / (1.0 + y) ** t))


def effective_duration(price_fn, y: float, dy: float = 0.0001) -> float:
    """D = (P(y - dy) - P(y + dy)) / (2 * P(y) * dy)"""
    p0, p_down, p_up = price_fn(y), price_fn(y - dy), price_fn(y + dy)
    return (p_down - p_up) / (2.0 * p0 * dy)


def effective_convexity(price_fn, y: float, dy: float = 0.0001) -> float:
    """C = (P(y - dy) + P(y + dy) - 2 P(y)) / (P(y) * dy^2)"""
    p0, p_down, p_up = price_fn(y), price_fn(y - dy), price_fn(y + dy)
    return (p_down + p_up - 2.0 * p0) / (p0 * dy**2)


# ---------------------------------------------------------------------------
# 4. Assets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Bond:
    """Annual-coupon bond, face 100. coupon=0 gives a zero-coupon strip."""
    name: str
    coupon: float
    maturity: int

    def price(self, y: float) -> float:
        t = np.arange(1, self.maturity + 1, dtype=float)
        cfs = np.full(self.maturity, 100.0 * self.coupon)
        cfs[-1] += 100.0
        return float(np.sum(cfs / (1.0 + y) ** t))

    def duration(self, y: float) -> float:
        return effective_duration(self.price, y)


SHORT_BOND = Bond("5-year bond (4%)", coupon=0.04, maturity=5)
LONG_BOND = Bond("20-year bond (4.5%)", coupon=0.045, maturity=20)
LONG_STRIP = Bond("30-year strip", coupon=0.0, maturity=30)


@dataclass
class Portfolio:
    """Market values at the base rate. Equity value is rate-insensitive."""
    name: str
    equity: float
    bonds: dict  # Bond -> market value at base rate

    def value(self, y_base: float, shock: float) -> float:
        total = self.equity
        for bond, mv in self.bonds.items():
            total += mv * bond.price(y_base + shock) / bond.price(y_base)
        return total

    def dv01(self, y_base: float) -> float:
        """Dollar change in value for a 1 bp fall in rates."""
        return self.value(y_base, -0.0001) - self.value(y_base, 0.0)


def current_portfolio(total_assets: float) -> Portfolio:
    """A typical 'balanced' mix: mostly equity, short-to-medium bonds."""
    return Portfolio(
        name="Current mix (55% equity)",
        equity=0.55 * total_assets,
        bonds={SHORT_BOND: 0.30 * total_assets,
               LONG_BOND: 0.15 * total_assets},
    )


def duration_matched_portfolio(total_assets: float, liability: float,
                               liability_duration: float, y: float,
                               equity_weight: float = 0.40) -> tuple:
    """
    Keep `equity_weight` in equity and split the bond sleeve between a
    5-year bond and a 30-year strip so that asset DV01 equals liability DV01:

        A_FI * (w * D_long + (1 - w) * D_short) = L * D_L

    Returns (portfolio, w). w is clipped to [0, 1]; if it hits a bound,
    the hedge is only partial.
    """
    a_fi = (1.0 - equity_weight) * total_assets
    d_short, d_long = SHORT_BOND.duration(y), LONG_STRIP.duration(y)
    w = (liability * liability_duration / a_fi - d_short) / (d_long - d_short)
    w = float(np.clip(w, 0.0, 1.0))
    port = Portfolio(
        name=f"Duration-matched ({equity_weight:.0%} equity)",
        equity=equity_weight * total_assets,
        bonds={SHORT_BOND: (1.0 - w) * a_fi, LONG_STRIP: w * a_fi},
    )
    return port, w


# ---------------------------------------------------------------------------
# 5. Funded status under rate shocks
# ---------------------------------------------------------------------------
def shock_table(cash_flows: pd.Series, portfolios: list, y: float,
                shocks_bp: list) -> pd.DataFrame:
    """Liability, assets and funded ratio for each parallel shock."""
    rows = []
    d_l = effective_duration(lambda r: present_value(cash_flows, r), y)
    c_l = effective_convexity(lambda r: present_value(cash_flows, r), y)
    l0 = present_value(cash_flows, y)
    for bp in shocks_bp:
        s = bp / 10_000
        liab = present_value(cash_flows, y + s)
        row = {
            "shock_bp": bp,
            "liability": liab,
            "liab_change_exact_pct": 100 * (liab / l0 - 1),
            "liab_change_duration_pct": 100 * (-d_l * s),
            "liab_change_dur_conv_pct": 100 * (-d_l * s + 0.5 * c_l * s**2),
        }
        for p in portfolios:
            a = p.value(y, s)
            row[f"funded_{p.name}"] = a / liab
            row[f"surplus_{p.name}"] = a - liab
        rows.append(row)
    return pd.DataFrame(rows)
