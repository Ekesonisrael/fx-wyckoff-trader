#!/usr/bin/env python3
"""
fx_carry_v2.py — FX Carry with Daily VIX Scaling + Vol Weighting + DD Control

Improvements over v1:
  1. VIX checked DAILY (not monthly) — smooth exposure scaling
  2. Inverse-volatility weighting (not equal)
  3. Hard drawdown kill switch
  4. Uses 3M rolling carry average (smoother than spot rate)
"""

import io
import math
import numpy as np
import pandas as pd
import requests
import dukascopy_python
from dukascopy_python import instruments as inst_mod

PAIRS = {
    'EURUSD': {'attr': 'INSTRUMENT_FX_MAJORS_EUR_USD', 'pip': 0.0001, 'rate': 'IR3TIB01EZM156N', 'quote_is_usd': True},
    'GBPUSD': {'attr': 'INSTRUMENT_FX_MAJORS_GBP_USD', 'pip': 0.0001, 'rate': 'IR3TIB01GBM156N', 'quote_is_usd': True},
    'AUDUSD': {'attr': 'INSTRUMENT_FX_MAJORS_AUD_USD', 'pip': 0.0001, 'rate': 'IR3TIB01AUM156N', 'quote_is_usd': True},
    'NZDUSD': {'attr': 'INSTRUMENT_FX_MAJORS_NZD_USD', 'pip': 0.0001, 'rate': 'IR3TIB01NZM156N', 'quote_is_usd': True},
    'USDCAD': {'attr': 'INSTRUMENT_FX_MAJORS_USD_CAD', 'pip': 0.0001, 'rate': 'IR3TIB01CAM156N', 'quote_is_usd': False},
    'USDCHF': {'attr': 'INSTRUMENT_FX_MAJORS_USD_CHF', 'pip': 0.0001, 'rate': 'IR3TIB01CHM156N', 'quote_is_usd': False},
    'USDJPY': {'attr': 'INSTRUMENT_FX_MAJORS_USD_JPY', 'pip': 0.01,   'rate': 'IR3TIB01JPM156N', 'quote_is_usd': False},
}

US_RATE = 'IR3TIB01USM156N'
VIX_SERIES = 'VIXCLS'
START = pd.Timestamp(2005, 1, 1)
END = pd.Timestamp.now()
SPREAD_PIPS = 1.5
RNG_SEED = 42
BOOTSTRAP_ITERS = 5000


def fred_csv(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=30)
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        s = df.dropna().set_index('date')['value']
        print(f"  [FRED] {series_id}: {len(s)} obs")
        return s
    except Exception as e:
        print(f"  [FRED] {series_id}: ERROR {e}")
        return pd.Series(dtype=float)


def fetch_daily(attr, start, end):
    inst = getattr(inst_mod, attr, None)
    if inst is None:
        return pd.Series(dtype=float)
    df = dukascopy_python.fetch(
        inst, dukascopy_python.INTERVAL_DAY_1,
        dukascopy_python.OFFER_SIDE_BID,
        start.to_pydatetime(), end.to_pydatetime(),
    )
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df[['close']].dropna()
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df['close']


def load_all():
    print("--- FRED ---")
    vix = fred_csv(VIX_SERIES)
    us_rate = fred_csv(US_RATE)
    rates = {sym: fred_csv(cfg['rate']) for sym, cfg in PAIRS.items()}
    print("--- PRICES ---")
    frames = {}
    for sym, cfg in PAIRS.items():
        s = fetch_daily(cfg['attr'], START, END)
        if not s.empty:
            frames[sym] = s
    panel = pd.DataFrame(frames).sort_index().ffill(limit=3).dropna()
    print(f"  Panel: {panel.shape[0]} days x {panel.shape[1]} pairs")
    return panel, vix, us_rate, rates


def build_carry(panel, rates, us_rate):
    idx = panel.index
    us_d = us_rate.reindex(idx, method='ffill')
    carry = pd.DataFrame(0.0, index=idx, columns=panel.columns)
    for sym, cfg in PAIRS.items():
        r = rates[sym].reindex(idx, method='ffill')
        if sym in carry.columns:
            carry[sym] = (r.values - us_d.values) if cfg['quote_is_usd'] else (us_d.values - r.values)
    # 3-month rolling average of carry (smoother)
    carry = carry.rolling(63, min_periods=20).mean()
    return carry


def vix_exposure_scalar(vix_level):
    """Daily VIX scaling: full exposure < 15, zero above 30."""
    if pd.isna(vix_level):
        return 0.0
    if vix_level < 15:
        return 1.0
    if vix_level < 20:
        return 0.75
    if vix_level < 25:
        return 0.50
    if vix_level < 30:
        return 0.25
    return 0.0


def backtest(panel, carry, vix):
    idx = panel.index
    vix_d = vix.reindex(idx, method='ffill')

    # Realized vol for inverse-vol weighting
    rets = panel.pct_change()
    vol = rets.rolling(60, min_periods=20).std()

    # Month-start rebalance dates
    month_first = pd.Series(idx).groupby([idx.year, idx.month]).min()
    rebalances = pd.DatetimeIndex(month_first.values)

    targets = pd.DataFrame(0.0, index=idx, columns=panel.columns)

    for rd in rebalances:
        if rd not in carry.index:
            continue
        row = carry.loc[rd].dropna()
        if len(row) < 3:
            continue
        vix_lvl = vix_d.loc[rd]
        if pd.isna(vix_lvl) or vix_lvl >= 25:
            continue
        top = row.nlargest(3).index
        inv_vols = 1.0 / vol.loc[rd, top].replace(0, np.nan)
        inv_vols = inv_vols.fillna(0)
        if inv_vols.sum() == 0:
            continue
        weights = inv_vols / inv_vols.sum()
        targets.loc[rd, top] = weights.values

    # Forward-fill targets to daily
    targets = targets.replace(0, np.nan).ffill().fillna(0.0)

    # Apply daily VIX exposure scalar
    vix_scalar = vix_d.apply(vix_exposure_scalar)
    effective = targets.mul(vix_scalar, axis=0)

    n_active = (effective.abs().sum(axis=1) > 0).sum()
    print(f"\n  [DEBUG] active days: {n_active} / {len(effective)}")

    # Spot P&L
    signal = effective.shift(1).fillna(0.0)
    spot_pnl = (signal * rets).sum(axis=1)

    # Carry income P&L
    carry_daily = carry.reindex(idx, method='ffill').fillna(0.0) / 100.0 / 252.0
    carry_income = (signal * carry_daily).sum(axis=1)

    gross = spot_pnl + carry_income

    # Costs — only on actual rebalance days
    cost = pd.Series(0.0, index=idx)
    rebal_dates_set = set(rebalances)
    for i, dt in enumerate(idx):
        if dt not in rebal_dates_set:
            continue
        for sym in panel.columns:
            pip = PAIRS[sym]['pip']
            spread_price = SPREAD_PIPS * pip
            prev = targets[sym].iloc[i-1] if i > 0 else 0.0
            new = targets[sym].iloc[i]
            turnover = abs(new - prev)
            if turnover > 0:
                cost.iloc[i] += turnover * spread_price / panel[sym].iloc[i]

    # Apply drawdown control: scale down after -10%, halt after -20%
    net_raw = gross - cost
    equity_raw = (1 + net_raw).cumprod()
    peak = equity_raw.cummax()
    dd = (equity_raw - peak) / peak

    dd_scalar = pd.Series(1.0, index=idx)
    dd_scalar[dd < -0.10] = 0.5
    dd_scalar[dd < -0.20] = 0.0
    dd_scalar = dd_scalar.shift(1).fillna(1.0)  # avoid lookahead

    net = net_raw * dd_scalar
    equity = (1 + net).cumprod() * 10000

    print(f"  [DEBUG] spot_pnl:     {spot_pnl.sum():+.4f}")
    print(f"  [DEBUG] carry_income: {carry_income.sum():+.4f}")
    print(f"  [DEBUG] cost:         {cost.sum():+.4f}")
    print(f"  [DEBUG] net:          {net.sum():+.4f}")

    return net, equity


def summarize(net):
    pnl = net.dropna()
    if len(pnl) < 100:
        return {'n': 0}
    m, s = pnl.mean(), pnl.std()
    sharpe = (m / s) * math.sqrt(252) if s > 0 else 0.0
    monthly = (1 + pnl).resample('ME').prod() - 1
    rng = np.random.default_rng(RNG_SEED)
    boots = monthly.values[rng.integers(0, len(monthly), size=(BOOTSTRAP_ITERS, len(monthly)))].mean(axis=1)
    ci_lo = float(np.percentile(boots, 2.5))
    total = (1 + pnl).prod() - 1
    years = (pnl.index[-1] - pnl.index[0]).days / 365.25
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0
    eq = (1 + pnl).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    return {
        'n': len(pnl),
        'sharpe': round(sharpe, 3),
        'monthly_mean_pct': round(monthly.mean() * 100, 4),
        'ci_lo_pct': round(ci_lo * 100, 4),
        'total_pct': round(total * 100, 2),
        'ann_pct': round(ann * 100, 2),
        'max_dd_pct': round(dd * 100, 2),
    }


def report(s):
    print("\n" + "=" * 60)
    print("   FX CARRY v2 — DAILY VIX + VOL WEIGHT + DD CONTROL")
    print("=" * 60)
    if s.get('n', 0) < 100:
        print("Insufficient data.")
        return
    print(f"  Sharpe (ann.):    {s['sharpe']:+.3f}")
    print(f"  Monthly mean:     {s['monthly_mean_pct']:+.4f}%")
    print(f"  Monthly CI lower: {s['ci_lo_pct']:+.4f}%")
    print(f"  Total return:     {s['total_pct']:+.2f}%")
    print(f"  Annualized:       {s['ann_pct']:+.2f}%")
    print(f"  Max drawdown:     {s['max_dd_pct']:.2f}%")
    print("\n--- VERDICT (gate: Sharpe>0.3, CI_lo>0, DD<20%) ---")
    passed = s['sharpe'] > 0.3 and s['ci_lo_pct'] > 0 and s['max_dd_pct'] > -20
    if passed:
        print("  ✅ PASS")
    else:
        fails = []
        if s['sharpe'] <= 0.3: fails.append(f"Sharpe {s['sharpe']}")
        if s['ci_lo_pct'] <= 0: fails.append(f"CI_lo {s['ci_lo_pct']}")
        if s['max_dd_pct'] <= -20: fails.append(f"DD {s['max_dd_pct']}")
        print(f"  ❌ FAIL — {'; '.join(fails)}")
    print("=" * 60)


def main():
    panel, vix, us_rate, rates = load_all()
    if panel.empty:
        print("No data.")
        return
    carry = build_carry(panel, rates, us_rate)
    net, equity = backtest(panel, carry, vix)
    s = summarize(net)
    report(s)
    pd.DataFrame({'equity': equity, 'net': net}).to_csv('carry_v2.csv')
    print("\nSaved carry_v2.csv")


if __name__ == "__main__":
    main()
