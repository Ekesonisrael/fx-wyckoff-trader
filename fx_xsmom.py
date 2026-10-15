#!/usr/bin/env python3
"""
fx_xsmom.py — Cross-Sectional FX Momentum Backtester

Reference: Menkhoff, Sarno, Schmeling & Schrimpf (2012), JFE 106(3).

Rules:
  - Universe: 7 major USD pairs
  - Signal: 3-month cumulative return per pair, cross-sectionally ranked
  - Rebalance: first trading day of each month
  - Positions: long top 3, short bottom 3, equal weight
  - Holding: one month
  - Costs: 1.5 pip spread per side per trade

Gates (fixed before results):
  - PASS if Sharpe_ann > 0.5 AND monthly CI lower bound > 0
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

LOOKBACK_MONTHS = 3
TOP_N = 3
BOTTOM_N = 3
START_DATE = pd.Timestamp(2005, 1, 1)
END_DATE = pd.Timestamp.now()

SPREAD_PIPS = 1.5
START_BALANCE = 10_000.0
RNG_SEED = 42
BOOTSTRAP_ITERS = 5000


# ============================================================
# DATA
# ============================================================
def load_instruments():
    loaded = {}
    for sym, (name, attr, pip) in SYMBOLS.items():
        inst = getattr(inst_mod, attr, None)
        if inst is None:
            print(f"  [SKIP] {sym}")
            continue
        loaded[sym] = {'name': name, 'instrument': inst, 'pip_size': pip}
    print(f"  [LOADED] {len(loaded)} instruments")
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


def build_price_panel(instruments, start, end):
    frames = {}
    for sym, cfg in instruments.items():
        print(f"  Fetching {sym}...")
        df = fetch_daily(cfg['instrument'], start, end)
        if df.empty:
            continue
        frames[sym] = df['close']
    panel = pd.DataFrame(frames).sort_index()
    panel = panel.ffill(limit=3).dropna()
    print(f"  Panel: {panel.shape[0]} bars x {panel.shape[1]} pairs")
    print(f"  Range: {panel.index[0].date()} -> {panel.index[-1].date()}")
    return panel


# ============================================================
# SIGNALS
# ============================================================
def compute_signals(panel, lookback_months):
    """
    Returns a DataFrame indexed by month-end dates, columns = symbols,
    values = target weight (+1/N long, -1/N short, 0 flat).
    Uses only data available as of that month-end.
    """
    month_end = panel.resample('ME').last()
    print(f"  [DEBUG] month_end rows: {len(month_end)}")

    rets = month_end.pct_change()
    mom = (1 + rets).rolling(lookback_months).apply(np.prod, raw=True) - 1
    mom = mom.shift(1)  # use info available at t to predict t+1

    targets = pd.DataFrame(0.0, index=mom.index, columns=mom.columns)
    for dt, row in mom.iterrows():
        valid = row.dropna()
        if len(valid) < (TOP_N + BOTTOM_N):
            continue
        top = valid.nlargest(TOP_N).index
        bottom = valid.nsmallest(BOTTOM_N).index
        targets.loc[dt, top] = 1.0 / TOP_N
        targets.loc[dt, bottom] = -1.0 / BOTTOM_N

    n_nonzero = (targets.abs().sum(axis=1) > 0).sum()
    print(f"  [DEBUG] month-end rows with positions: {n_nonzero}")
    return targets


# ============================================================
# BACKTEST
# ============================================================
def backtest(panel, month_end_targets, pip_sizes, spread_pips):
    """
    Align month-end targets to the first trading day AFTER each month-end.
    Hold until the next rebalance. Track daily P&L.
    """
    daily_index = panel.index

    # Build a daily target frame, one row per month-end target
    # First, find the trading-day version of each month-end target
    aligned = pd.DataFrame(np.nan, index=daily_index, columns=panel.columns)

    for me_dt, row in month_end_targets.iterrows():
        future = daily_index[daily_index > me_dt]
        if len(future) == 0:
            continue
        apply_from = future[0]
        aligned.loc[apply_from] = row.values

    # Forward-fill across trading days (targets persist until next rebalance)
    aligned = aligned.ffill().fillna(0.0)

    n_nonzero = (aligned.abs().sum(axis=1) > 0).sum()
    print(f"  [DEBUG] trading days with nonzero positions: {n_nonzero} / {len(aligned)}")

    # Daily returns of the price panel
    daily_rets = panel.pct_change()

    # Signal = yesterday's target applied to today's return
    signal = aligned.shift(1).fillna(0.0)
    gross_pnl = (signal * daily_rets).sum(axis=1)

    # Costs on rebalance days (turnover)
    cost_series = pd.Series(0.0, index=daily_index)
    for sym in panel.columns:
        pip = pip_sizes[sym]
        spread_price = spread_pips * pip
        turnover = aligned[sym].diff().abs()
        price = panel[sym]
        cost_series += (turnover * spread_price / price).fillna(0.0)

    net_pnl = gross_pnl - cost_series
    equity = (1 + net_pnl).cumprod() * START_BALANCE

    print(f"  [DEBUG] gross PnL sum: {gross_pnl.sum():+.4f}")
    print(f"  [DEBUG] cost sum: {cost_series.sum():+.4f}")
    print(f"  [DEBUG] net PnL sum: {net_pnl.sum():+.4f}")

    return {'gross_pnl': gross_pnl, 'cost': cost_series, 'net_pnl': net_pnl, 'equity': equity}


# ============================================================
# METRICS
# ============================================================
def summarize(results):
    pnl = results['net_pnl'].dropna()
    if len(pnl) < 100:
        return {'n': 0}

    n = len(pnl)
    mean_d = pnl.mean()
    std_d = pnl.std()
    sharpe_ann = (mean_d / std_d) * math.sqrt(252) if std_d > 0 else 0.0

    monthly = (1 + pnl).resample('ME').prod() - 1
    m_mean = monthly.mean()
    m_std = monthly.std()
    sharpe_monthly = (m_mean / m_std * math.sqrt(12)) if m_std > 0 else 0.0

    rng = np.random.default_rng(RNG_SEED)
    boots = monthly.values[rng.integers(0, len(monthly), size=(BOOTSTRAP_ITERS, len(monthly)))].mean(axis=1)
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))

    total_ret = (1 + pnl).prod() - 1
    years = (pnl.index[-1] - pnl.index[0]).days / 365.25
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0

    eq = (1 + pnl).cumprod()
    peak = eq.cummax()
    max_dd = ((eq - peak) / peak).min()

    return {
        'n_days': n,
        'sharpe_ann': round(sharpe_ann, 3),
        'sharpe_monthly': round(sharpe_monthly, 3),
        'monthly_mean_pct': round(m_mean * 100, 4),
        'monthly_std_pct': round(m_std * 100, 4),
        'ci95_lo_pct': round(ci_lo * 100, 4),
        'ci95_hi_pct': round(ci_hi * 100, 4),
        'total_return_pct': round(total_ret * 100, 2),
        'ann_return_pct': round(ann_ret * 100, 2),
        'max_dd_pct': round(max_dd * 100, 2),
    }


def print_report(summary):
    print("\n" + "=" * 60)
    print("     CROSS-SECTIONAL FX MOMENTUM — BACKTEST REPORT")
    print("=" * 60)
    if summary.get('n_days', 0) < 100:
        print("Insufficient data.")
        return
    print(f"  Sharpe (ann.):     {summary['sharpe_ann']:+.3f}")
    print(f"  Sharpe (monthly):  {summary['sharpe_monthly']:+.3f}")
    print(f"  Monthly mean:      {summary['monthly_mean_pct']:+.4f}%")
    print(f"  Monthly std:       {summary['monthly_std_pct']:.4f}%")
    print(f"  Monthly CI95:      [{summary['ci95_lo_pct']:+.4f}%, {summary['ci95_hi_pct']:+.4f}%]")
    print(f"  Total return:      {summary['total_return_pct']:+.2f}%")
    print(f"  Annualized return: {summary['ann_return_pct']:+.2f}%")
    print(f"  Max drawdown:      {summary['max_dd_pct']:.2f}%")

    print("\n--- VERDICT ---")
    sharpe_ok = summary['sharpe_ann'] > 0.5
    ci_ok = summary['ci95_lo_pct'] > 0
    if sharpe_ok and ci_ok:
        print("  PASS — strategy shows positive risk-adjusted returns.")
    elif not sharpe_ok and not ci_ok:
        print("  FAIL — Sharpe <= 0.5 and CI lower bound <= 0.")
    elif not sharpe_ok:
        print(f"  FAIL — Sharpe {summary['sharpe_ann']:.3f} <= 0.5")
    else:
        print(f"  FAIL — Monthly CI lower bound {summary['ci95_lo_pct']:.4f}% <= 0")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  FX CROSS-SECTIONAL MOMENTUM BACKTEST")
    print(f"  Formation: {LOOKBACK_MONTHS}M | Long: {TOP_N} | Short: {BOTTOM_N}")
    print("=" * 60)

    instruments = load_instruments()
    if len(instruments) < 6:
        print("Not enough instruments.")
        return

    panel = build_price_panel(instruments, START_DATE, END_DATE)
    if panel.empty:
        print("No data.")
        return

    targets = compute_signals(panel, LOOKBACK_MONTHS)
    if (targets.abs().sum(axis=1) > 0).sum() == 0:
        print("No rebalance dates produced positions. Aborting.")
        return

    pip_sizes = {sym: cfg['pip_size'] for sym, cfg in instruments.items()}
    results = backtest(panel, targets, pip_sizes, SPREAD_PIPS)
    summary = summarize(results)
    print_report(summary)

    pd.DataFrame({
        'equity': results['equity'],
        'net_pnl': results['net_pnl'],
        'cost': results['cost'],
    }).to_csv('xsmom_equity.csv')
    print("\nSaved xsmom_equity.csv")


if __name__ == '__main__':
    main()
