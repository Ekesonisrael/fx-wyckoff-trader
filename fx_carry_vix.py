#!/usr/bin/env python3
"""
fx_carry_vix.py — FX Carry Trade with VIX Regime Filter

Rule: Long top-3 highest-carry pairs IF VIX < 20.
      Top-2 if VIX 20-25. Flat if VIX >= 25.
      Monthly rebalance. Equal weight.

Data: Dukascopy FX, FRED VIX (VIXCLS), FRED 3M interbank rates.

Returns come from TWO sources:
  1. Spot price change of the pairs held
  2. Daily interest income = (annual rate differential %) / 100 / 252
"""

import io
import math
import numpy as np
import pandas as pd
import requests
import dukascopy_python
from dukascopy_python import instruments as inst_mod

# ============================================================
# CONFIG
# ============================================================
PAIRS = {
    'EURUSD': {'attr': 'INSTRUMENT_FX_MAJORS_EUR_USD', 'pip': 0.0001,
               'rate': 'IR3TIB01EZM156N', 'quote_is_usd': True},
    'GBPUSD': {'attr': 'INSTRUMENT_FX_MAJORS_GBP_USD', 'pip': 0.0001,
               'rate': 'IR3TIB01GBM156N', 'quote_is_usd': True},
    'AUDUSD': {'attr': 'INSTRUMENT_FX_MAJORS_AUD_USD', 'pip': 0.0001,
               'rate': 'IR3TIB01AUM156N', 'quote_is_usd': True},
    'NZDUSD': {'attr': 'INSTRUMENT_FX_MAJORS_NZD_USD', 'pip': 0.0001,
               'rate': 'IR3TIB01NZM156N', 'quote_is_usd': True},
    'USDCAD': {'attr': 'INSTRUMENT_FX_MAJORS_USD_CAD', 'pip': 0.0001,
               'rate': 'IR3TIB01CAM156N', 'quote_is_usd': False},
    'USDCHF': {'attr': 'INSTRUMENT_FX_MAJORS_USD_CHF', 'pip': 0.0001,
               'rate': 'IR3TIB01CHM156N', 'quote_is_usd': False},
    'USDJPY': {'attr': 'INSTRUMENT_FX_MAJORS_USD_JPY', 'pip': 0.01,
               'rate': 'IR3TIB01JPM156N', 'quote_is_usd': False},
}

US_RATE_SERIES = 'IR3TIB01USM156N'
VIX_SERIES = 'VIXCLS'

START = pd.Timestamp(2005, 1, 1)
END = pd.Timestamp.now()

SPREAD_PIPS = 1.5

RNG_SEED = 42
BOOTSTRAP_ITERS = 5000


# ============================================================
# FRED DATA (no API key needed via CSV endpoint)
# ============================================================
def fred_csv(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"  [FRED] {series_id}: HTTP {r.status_code}")
            return pd.Series(dtype=float)
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        s = df.dropna().set_index('date')['value']
        print(f"  [FRED] {series_id}: {len(s)} obs | {s.index[0].date()} -> {s.index[-1].date()}")
        return s
    except Exception as e:
        print(f"  [FRED] {series_id}: ERROR {e}")
        return pd.Series(dtype=float)


def load_all_rates():
    print("\n--- FRED DATA ---")
    vix = fred_csv(VIX_SERIES)
    us_rate = fred_csv(US_RATE_SERIES)
    rates = {}
    for sym, cfg in PAIRS.items():
        rates[sym] = fred_csv(cfg['rate'])
    return vix, us_rate, rates


# ============================================================
# PRICES
# ============================================================
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


def load_prices():
    print("\n--- PRICES ---")
    frames = {}
    for sym, cfg in PAIRS.items():
        print(f"  {sym}...")
        s = fetch_daily(cfg['attr'], START, END)
        if s.empty:
            print(f"    [WARN] empty")
            continue
        frames[sym] = s
    panel = pd.DataFrame(frames).sort_index().ffill(limit=3).dropna()
    print(f"  Panel: {panel.shape[0]} days x {panel.shape[1]} pairs")
    print(f"  Range: {panel.index[0].date()} -> {panel.index[-1].date()}")
    return panel


# ============================================================
# CARRY SPREAD
# ============================================================
def build_carry_panel(rates, us_rate, panel):
    daily_idx = panel.index
    us_d = us_rate.reindex(daily_idx, method='ffill')

    carry = pd.DataFrame(0.0, index=daily_idx, columns=panel.columns)
    for sym, cfg in PAIRS.items():
        r = rates[sym].reindex(daily_idx, method='ffill')
        if sym in carry.columns:
            if cfg['quote_is_usd']:
                carry[sym] = r.values - us_d.values
            else:
                carry[sym] = us_d.values - r.values

    print("\n--- CARRY (last row, annualized %) ---")
    last = carry.iloc[-1].dropna()
    for sym, val in last.sort_values(ascending=False).items():
        print(f"  {sym}: {val:+.3f}%")

    return carry


# ============================================================
# BACKTEST (with carry income — the fix)
# ============================================================
def backtest(panel, carry, vix):
    daily_idx = panel.index
    vix_d = vix.reindex(daily_idx, method='ffill')

    month_first = pd.Series(daily_idx).groupby([daily_idx.year, daily_idx.month]).min()
    rebalance_dates = pd.DatetimeIndex(month_first.values)

    targets = pd.DataFrame(0.0, index=daily_idx, columns=panel.columns)
    for rd in rebalance_dates:
        if rd not in carry.index:
            continue
        vix_level = vix_d.loc[rd]
        if pd.isna(vix_level):
            continue
        if vix_level >= 25:
            n = 0
        elif vix_level >= 20:
            n = 2
        else:
            n = 3
        if n == 0:
            continue
        row = carry.loc[rd].dropna()
        if len(row) < n:
            continue
        top = row.nlargest(n).index
        targets.loc[rd, top] = 1.0 / n

    targets = targets.replace(0, np.nan).ffill().fillna(0.0)
    n_active = (targets.abs().sum(axis=1) > 0).sum()
    print(f"\n  [DEBUG] active days: {n_active} / {len(targets)}")

    # --- Spot price P&L ---
    rets = panel.pct_change()
    signal = targets.shift(1).fillna(0.0)
    spot_pnl = (signal * rets).sum(axis=1)

    # --- Carry income P&L (THE MISSING PIECE) ---
    # Annual rate differential (%) / 100 / 252 = daily income fraction
    carry_daily = carry.reindex(daily_idx, method='ffill').fillna(0.0) / 100.0 / 252.0
    carry_income = (signal * carry_daily).sum(axis=1)

    gross = spot_pnl + carry_income

    # --- Costs on rebalance days ---
    cost = pd.Series(0.0, index=daily_idx)
    for sym in panel.columns:
        pip = PAIRS[sym]['pip']
        spread_price = SPREAD_PIPS * pip
        turnover = targets[sym].diff().abs()
        cost += (turnover * spread_price / panel[sym]).fillna(0.0)

    net = gross - cost
    equity = (1 + net).cumprod() * 10000

    print(f"  [DEBUG] spot_pnl sum:    {spot_pnl.sum():+.4f}")
    print(f"  [DEBUG] carry_income sum:{carry_income.sum():+.4f}")
    print(f"  [DEBUG] gross sum:       {gross.sum():+.4f}")
    print(f"  [DEBUG] cost sum:        {cost.sum():+.4f}")
    print(f"  [DEBUG] net sum:         {net.sum():+.4f}")

    return net, equity, spot_pnl, carry_income


# ============================================================
# METRICS
# ============================================================
def summarize(net):
    pnl = net.dropna()
    if len(pnl) < 100:
        return {'n': 0}
    n = len(pnl)
    m = pnl.mean()
    s = pnl.std()
    sharpe = (m / s) * math.sqrt(252) if s > 0 else 0.0

    monthly = (1 + pnl).resample('ME').prod() - 1
    mm = monthly.mean()
    ms = monthly.std()
    rng = np.random.default_rng(RNG_SEED)
    boots = monthly.values[rng.integers(0, len(monthly), size=(BOOTSTRAP_ITERS, len(monthly)))].mean(axis=1)
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))

    total = (1 + pnl).prod() - 1
    years = (pnl.index[-1] - pnl.index[0]).days / 365.25
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0

    eq = (1 + pnl).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax()).min()

    return {
        'n': n, 'sharpe': round(sharpe, 3),
        'monthly_mean_pct': round(mm * 100, 4),
        'ci_lo_pct': round(ci_lo * 100, 4),
        'ci_hi_pct': round(ci_hi * 100, 4),
        'total_pct': round(total * 100, 2),
        'ann_pct': round(ann * 100, 2),
        'max_dd_pct': round(dd * 100, 2),
    }


def report(s):
    print("\n" + "=" * 60)
    print("     FX CARRY + VIX FILTER — REPORT")
    print("=" * 60)
    if s.get('n', 0) < 100:
        print("Insufficient data.")
        return
    print(f"  Sharpe (ann.):      {s['sharpe']:+.3f}")
    print(f"  Monthly mean:       {s['monthly_mean_pct']:+.4f}%")
    print(f"  Monthly CI95:       [{s['ci_lo_pct']:+.4f}%, {s['ci_hi_pct']:+.4f}%]")
    print(f"  Total return:       {s['total_pct']:+.2f}%")
    print(f"  Annualized return:  {s['ann_pct']:+.2f}%")
    print(f"  Max drawdown:       {s['max_dd_pct']:.2f}%")
    print("\n--- VERDICT ---")
    if s['sharpe'] > 0.5 and s['ci_lo_pct'] > 0:
        print("  PASS")
    elif s['sharpe'] > 0.3:
        print("  MARGINAL — promising but not a clear pass")
    else:
        print("  FAIL")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  FX CARRY + VIX FILTER BACKTEST")
    print("=" * 60)

    vix, us_rate, rates = load_all_rates()
    if vix.empty or us_rate.empty:
        print("FRED data missing. Aborting.")
        return

    panel = load_prices()
    if panel.empty:
        print("No price data.")
        return

    carry = build_carry_panel(rates, us_rate, panel)

    net, equity, spot_pnl, carry_income = backtest(panel, carry, vix)
    s = summarize(net)
    report(s)

    equity.to_csv('carry_equity.csv')
    pd.DataFrame({
        'equity': equity,
        'net': net,
        'spot': spot_pnl,
        'carry_income': carry_income,
    }).to_csv('carry_daily.csv')
    print("\nSaved carry_equity.csv and carry_daily.csv")


if __name__ == "__main__":
    main()
