#!/usr/bin/env python3
"""
fx_xsmom.py — Cross-Sectional FX Momentum Backtester

Hypothesis: Currencies that have outperformed over the past 3 months
continue to outperform over the next month, relative to their peers.

Reference: Menkhoff, Sarno, Schmeling & Schrimpf (2012), "Currency Momentum Strategies"
           Journal of Financial Economics, 106(3), 660-684.
           Eriksen (2019), "Cross-sectional return dispersion and currency momentum",
           Journal of Empirical Finance, 53, 91-108.

Rules (fixed, no tuning):
  - Universe: 7 major USD pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, USDJPY)
  - Signal: 3-month cumulative return per pair (cross-sectional rank)
  - Rebalance: First trading day of each month
  - Positions: Long top 3, short bottom 3, equal weight
  - Holding: One month
  - Costs: 1.5 pip spread per side per trade (applied on rebalance)

Data: Dukascopy daily bars, 2005-present.

Pre-registered gates (do not change after seeing results):
  - PASS if portfolio Sharpe > 0.5 AND monthly return CI lower bound > 0
  - FAIL otherwise
"""

import math
import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as inst_mod

# ============================================================
# CONFIG
# ============================================================
SYMBOLS = {
    'EURUSD': ('EUR/USD', 'INSTRUMENT_FX_MAJORS_EUR_USD', 0.0001),
    'GBPUSD': ('GBP/USD', 'INSTRUMENT_FX_MAJORS_GBP_USD', 0.0001),
    'AUDUSD': ('AUD/USD', 'INSTRUMENT_FX_MAJORS_AUD_USD', 0.0001),
    'NZDUSD': ('NZD/USD', 'INSTRUMENT_FX_MAJORS_NZD_USD', 0.0001),
    'USDCAD': ('USD/CAD', 'INSTRUMENT_FX_MAJORS_USD_CAD', 0.0001),
    'USDCHF': ('USD/CHF', 'INSTRUMENT_FX_MAJORS_USD_CHF', 0.0001),
    'USDJPY': ('USD/JPY', 'INSTRUMENT_FX_MAJORS_USD_JPY', 0.01),
}

# Parameters (fixed)
LOOKBACK_MONTHS = 3          # formation period (Menkhoff et al. 2012)
TOP_N = 3                    # long top 3, short bottom 3
BOTTOM_N = 3
REBALANCE_FREQ = 'MS'        # Month Start
START_DATE = pd.Timestamp(2005, 1, 1)
END_DATE = pd.Timestamp.now()

# Costs
SPREAD_PIPS = 1.5            # per side, per trade
SLIPPAGE_PIPS = 0.0          # not modeled separately (spread captures it)

# Portfolio
START_BALANCE = 10_000.0
RISK_PER_TRADE = 0.01        # 1% of equity per leg (for sizing only)

# Bootstrap
RNG_SEED = 42
BOOTSTRAP_ITERS = 5000

# ============================================================
# DATA LOADING
# ============================================================
def load_instruments():
    loaded = {}
    for sym, (name, attr, pip) in SYMBOLS.items():
        inst = getattr(inst_mod, attr, None)
        if inst is None:
            print(f"  [SKIP] {sym} ({attr} not available)")
            continue
        loaded[sym] = {'name': name, 'instrument': inst, 'pip_size': pip}
        print(f"  [LOADED] {sym} ({name})")
    return loaded


def fetch_daily(instrument, start, end):
    df = dukascopy_python.fetch(
        instrument,
        dukascopy_python.INTERVAL_DAY_1,
        dukascopy_python.OFFER_SIDE_BID,
        start.to_pydatetime(), end.to_pydatetime(),
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df[['close']].dropna()
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


# ============================================================
# DATA ALIGNMENT
# ============================================================
def build_price_panel(instruments, start, end):
    """Fetch all pairs, align to a common daily index."""
    frames = {}
    for sym, cfg in instruments.items():
        print(f"  Fetching {sym}...")
        df = fetch_daily(cfg['instrument'], start, end)
        if df.empty:
            print(f"    [WARN] {sym} empty")
            continue
        frames[sym] = df['close']
    if not frames:
        return pd.DataFrame()

    panel = pd.DataFrame(frames)
    panel = panel.sort_index()

    # Align to common dates (inner join)
    panel = panel.dropna(how='all')
    # Forward-fill gaps up to 3 days (weekends/holidays are not real gaps)
    panel = panel.ffill(limit=3)
    panel = panel.dropna()

    print(f"  Panel: {panel.shape[0]} bars × {panel.shape[1]} pairs")
    print(f"  Range: {panel.index[0].date()} → {panel.index[-1].date()}")
    return panel


# ============================================================
# SIGNAL: cross-sectional momentum
# ============================================================
def compute_cross_sectional_momentum(panel, lookback_months=3):
    """
    Compute cross-sectional momentum signal at each month-end.
    Returns a DataFrame of ranks per pair per rebalance date.
    Rank 1 = strongest, Rank N = weakest.
    """
    # Resample to month-end to get month-end prices
    month_end = panel.resample('ME').last()

    # Compute cumulative return over lookback period
    # Use the formula from Menkhoff et al.: product of (1+r) - 1 over past K months
    rets = month_end.pct_change()
    mom = (1 + rets).rolling(lookback_months).apply(np.prod, raw=True) - 1

    # Shift so that we use information available at month t to predict month t+1
    mom = mom.shift(1)

    # Cross-sectional rank (per row, rank from strongest to weakest)
    ranks = mom.rank(axis=1, ascending=False, method='min')

    return mom, ranks


def generate_target_positions(ranks, n_long=3, n_short=3):
    """
    At each rebalance date, assign target weights.
    +1/n for top n, -1/n for bottom n, 0 otherwise.
    """
    targets = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    for dt, row in ranks.iterrows():
        valid = row.dropna()
        if len(valid) < (n_long + n_short):
            continue
        top = valid.nsmallest(n_long).index
        bottom = valid.nlargest(n_short).index
        targets.loc[dt, top] = 1.0 / n_long
        targets.loc[dt, bottom] = -1.0 / n_short
    return targets


# ============================================================
# BACKTEST
# ============================================================
def backtest_xsmom(panel, targets, pip_sizes, spread_pips=1.5):
    """
    Walk forward day by day. At each rebalance date (month start),
    close old positions and open new ones. Track daily P&L.
    """
    # Align targets to month start (first trading day of month)
    daily_index = panel.index
    # Find first trading day of each month
    month_starts = pd.Series(daily_index).groupby([daily_index.year, daily_index.month]).min()
    month_start_dates = pd.DatetimeIndex(month_starts.values)

    # Build a daily target matrix
    daily_targets = pd.DataFrame(0.0, index=daily_index, columns=panel.columns)
    for dt in month_start_dates:
        if dt in targets.index:
            daily_targets.loc[dt] = targets.loc[dt]
    # Forward-fill between rebalances
    daily_targets = daily_targets.replace(0, np.nan).ffill().fillna(0.0)

    # Compute daily returns (long/short P&L)
    daily_rets = panel.pct_change()

    # Position P&L (signal from previous day's close, applied to today's return)
    signal = daily_targets.shift(1).fillna(0.0)
    gross_pnl = (signal * daily_rets).sum(axis=1)

    # Transaction costs: applied on rebalance days
    turnover = daily_targets.diff().abs().sum(axis=1)
    # Costs in pips converted to return terms
    # We approximate: cost = turnover * (spread_pips / 10000) for each leg
    # But since pairs have different pip sizes (JPY=0.01), we compute per-pair
    cost_series = pd.Series(0.0, index=daily_index)
    for sym in panel.columns:
        pip = pip_sizes[sym]
        # spread in price terms
        spread_price = spread_pips * pip
        # cost per unit turnover for this pair
        pair_turnover = daily_targets[sym].diff().abs()
        # The turnover is in weight units. We approximate position notional as weight * 1
        # (weights are fractions of equity). So cost in return terms = turnover * spread_price / price
        price = panel[sym]
        cost_series += (pair_turnover * spread_price / price).fillna(0.0)

    net_pnl = gross_pnl - cost_series
    equity = (1 + net_pnl).cumprod() * START_BALANCE

    return {
        'gross_pnl': gross_pnl,
        'cost': cost_series,
        'net_pnl': net_pnl,
        'equity': equity,
    }


# ============================================================
# METRICS
# ============================================================
def summarize_backtest(results):
    pnl = results['net_pnl'].dropna()
    if len(pnl) < 100:
        return {'n': 0}

    n = len(pnl)
    mean_daily = pnl.mean()
    std_daily = pnl.std()
    sharpe_daily = mean_daily / std_daily if std_daily > 0 else 0.0
    sharpe_ann = sharpe_daily * math.sqrt(252)

    # Monthly returns
    monthly = (1 + pnl).resample('ME').prod() - 1
    monthly_mean = monthly.mean()
    monthly_std = monthly.std()
    sharpe_monthly = (monthly_mean / monthly_std * math.sqrt(12)) if monthly_std > 0 else 0.0

    # Bootstrap CI on monthly returns
    rng = np.random.default_rng(RNG_SEED)
    boots = monthly.values[rng.integers(0, len(monthly), size=(BOOTSTRAP_ITERS, len(monthly)))].mean(axis=1)
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))

    total_return = (1 + pnl).prod() - 1
    years = (pnl.index[-1] - pnl.index[0]).days / 365.25
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # Drawdown
    eq = (1 + pnl).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    return {
        'n_days': n,
        'sharpe_ann': round(sharpe_ann, 3),
        'sharpe_monthly': round(sharpe_monthly, 3),
        'monthly_mean': round(monthly_mean, 6),
        'monthly_std': round(monthly_std, 6),
        'ci95_lo': round(ci_lo, 6),
        'ci95_hi': round(ci_hi, 6),
        'total_return_pct': round(total_return * 100, 2),
        'ann_return_pct': round(ann_return * 100, 2),
        'max_dd_pct': round(max_dd * 100, 2),
        'avg_cost_bps': round(results['cost'].mean() * 10000, 2),
    }


# ============================================================
# REPORT
# ============================================================
def print_report(summary, results):
    print("\n" + "=" * 60)
    print("     CROSS-SECTIONAL FX MOMENTUM — BACKTEST REPORT")
    print("=" * 60)

    if summary.get('n_days', 0) < 100:
        print("Insufficient data.")
        return

    print(f"  Sharpe (ann.):       {summary['sharpe_ann']:+.3f}")
    print(f"  Sharpe (monthly):    {summary['sharpe_monthly']:+.3f}")
    print(f"  Monthly mean:        {summary['monthly_mean']*100:+.4f}%")
    print(f"  Monthly std:         {summary['monthly_std']*100:.4f}%")
    print(f"  Monthly CI95:        [{summary['ci95_lo']*100:+.4f}%, {summary['ci95_hi']*100:+.4f}%]")
    print(f"  Total return:        {summary['total_return_pct']:+.2f}%")
    print(f"  Annualized return:   {summary['ann_return_pct']:+.2f}%")
    print(f"  Max drawdown:        {summary['max_dd_pct']:.2f}%")
    print(f"  Avg daily cost:      {summary['avg_cost_bps']:.2f} bps")

    # Verdict
    print("\n--- VERDICT ---")
    sharpe_ok = summary['sharpe_ann'] > 0.5
    ci_ok = summary['ci95_lo'] > 0
    if sharpe_ok and ci_ok:
        print("  ✅ PASS — strategy shows positive risk-adjusted returns.")
    elif not sharpe_ok and not ci_ok:
        print("  ❌ FAIL — Sharpe ≤ 0.5 and CI lower bound ≤ 0.")
    elif not sharpe_ok:
        print(f"  ❌ FAIL — Sharpe {summary['sharpe_ann']:.3f} ≤ 0.5")
    else:
        print(f"  ❌ FAIL — Monthly CI lower bound {summary['ci95_lo']*100:.4f}% ≤ 0")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  FX CROSS-SECTIONAL MOMENTUM BACKTEST")
    print(f"  Formation: {LOOKBACK_MONTHS} months | Long: {TOP_N} | Short: {BOTTOM_N}")
    print("=" * 60)

    instruments = load_instruments()
    if len(instruments) < 6:
        print("Not enough instruments. Aborting.")
        return

    panel = build_price_panel(instruments, START_DATE, END_DATE)
    if panel.empty:
        print("No data. Aborting.")
        return

    # Compute signals
    mom, ranks = compute_cross_sectional_momentum(panel, LOOKBACK_MONTHS)
    targets = generate_target_positions(ranks, TOP_N, BOTTOM_N)

    print(f"\n  Rebalance dates: {len(targets)}")
    if len(targets) == 0:
        print("No rebalance dates. Aborting.")
        return

    # Pip sizes
    pip_sizes = {sym: cfg['pip_size'] for sym, cfg in instruments.items()}

    # Backtest
    results = backtest_xsmom(panel, targets, pip_sizes, spread_pips=SPREAD_PIPS)
    summary = summarize_backtest(results)
    print_report(summary, results)

    # Save
    pd.DataFrame({
        'equity': results['equity'],
        'net_pnl': results['net_pnl'],
        'cost': results['cost'],
    }).to_csv('xsmom_equity.csv')
    print("\nSaved xsmom_equity.csv")


if __name__ == '__main__':
    main()
