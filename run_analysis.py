"""
Runs the model and writes results to output/.

    python run_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pension_model import (
    LONG_STRIP, SHORT_BOND, current_portfolio, duration_matched_portfolio,
    effective_convexity, effective_duration, expected_cash_flows,
    make_sample_plan, present_value, shock_table,
)

BASE_RATE = 0.045          # flat discount rate
INITIAL_FUNDED = 0.95      # assets = 95% of liabilities at the start
SHOCKS_BP = [-200, -150, -100, -50, 0, 50, 100, 150, 200]
OUT = Path("output")


def main():
    OUT.mkdir(exist_ok=True)

    # Liabilities
    plan = make_sample_plan()
    cfs = expected_cash_flows(plan)
    liab = present_value(cfs, BASE_RATE)
    pv = lambda r: present_value(cfs, r)
    d_l = effective_duration(pv, BASE_RATE)
    c_l = effective_convexity(pv, BASE_RATE)

    # Assets
    assets = INITIAL_FUNDED * liab
    current = current_portfolio(assets)
    matched, w_long = duration_matched_portfolio(assets, liab, d_l, BASE_RATE)
    liab_dv01 = pv(BASE_RATE - 0.0001) - liab

    # Shocks
    table = shock_table(cfs, [current, matched], BASE_RATE, SHOCKS_BP)
    table.to_csv(OUT / "shock_results.csv", index=False)

    # ---- Console summary --------------------------------------------------
    print("=" * 64)
    print("LIABILITIES")
    print(f"  Members: {plan['count'].sum():,} "
          f"({plan.loc[plan.status == 'active', 'count'].sum()} active, "
          f"{plan.loc[plan.status == 'retiree', 'count'].sum()} retirees)")
    print(f"  PV at {BASE_RATE:.1%}:        ${liab / 1e6:,.1f}M")
    print(f"  Effective duration:   {d_l:.2f}")
    print(f"  Effective convexity:  {c_l:.1f}")
    print(f"  DV01 (1 bp fall):     ${liab_dv01:,.0f}")
    print("ASSETS (95% funded)")
    for p in (current, matched):
        print(f"  {p.name:<32} DV01 ${p.dv01(BASE_RATE):>10,.0f}"
              f"   hedge ratio {p.dv01(BASE_RATE) / liab_dv01:.0%}")
    print(f"  Matched bond sleeve: {w_long:.0%} in {LONG_STRIP.name}, "
          f"{1 - w_long:.0%} in {SHORT_BOND.name}")
    print("FUNDED STATUS UNDER PARALLEL SHOCKS")
    cols = ["shock_bp", f"funded_{current.name}", f"funded_{matched.name}"]
    view = table[cols].copy()
    view.columns = ["shock (bp)", "current mix", "duration-matched"]
    print(view.to_string(index=False, float_format=lambda x: f"{x:.1%}"))
    print("DURATION APPROXIMATION vs EXACT (liability % change)")
    acc = table[["shock_bp", "liab_change_exact_pct",
                 "liab_change_duration_pct", "liab_change_dur_conv_pct"]]
    acc.columns = ["shock (bp)", "exact", "duration only", "dur + convexity"]
    print(acc.to_string(index=False, float_format=lambda x: f"{x:+.2f}%"))
    print("=" * 64)

    # ---- Chart 1: funded status vs shock ------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for p, style in ((current, "o-"), (matched, "s-")):
        ax.plot(table["shock_bp"], 100 * table[f"funded_{p.name}"], style,
                label=p.name)
    ax.axhline(100, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Parallel rate shock (bp)")
    ax.set_ylabel("Funded status A / L (%)")
    ax.set_title("Funded status sensitivity to interest rates")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "funded_status_shocks.png", dpi=150)

    # ---- Chart 2: expected benefit cash flows --------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cfs.index, cfs.values / 1e6, width=0.8)
    ax.set_xlim(0, 75)
    ax.set_xlabel("Year")
    ax.set_ylabel("Expected benefit payments ($M)")
    ax.set_title("Projected benefit cash flows")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "cash_flows.png", dpi=150)

    print(f"Saved results and charts to {OUT.resolve()}")


if __name__ == "__main__":
    main()
