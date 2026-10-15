#!/usr/bin/env python3
"""
fx_backtest.py — Daily trend-following backtester (100-day breakout, 3xATR trail).

Edge hypothesis: time-series momentum. Long when close > 100-day high close,
short when close < 100-day low close. Exit on 3xATR(20) trailing stop.
Portfolio-level judgement (monthly Sharpe, not per-trade R).

Data: Dukascopy, daily bars, 2005-present.
"""
import math
import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as inst_mod

# ============================================================
# SEALED HOLDOUT
# ============================================================
USE_HOLDOUT = False   # flip to True ONCE, at the very end

# ============================================================
# STRATEGY PARAMETERS (fixed — no tuning)
# ============================================================
LOOKBACK_DAYS = 100
ATR_PERIOD = 20
ATR_MULT = 3.0
RISK_PER_TRADE = 0.0025      # 0.25% of starting balance per trade
MAX_HOLD_BARS = 250
MIN_TRADES_REQUIRED = 150

# ============================================================
# COSTS (era-aware)
# ============================================================
ERA_CUTOVER_YEAR = 2012
SLIPPAGE_FRACTION_OF_SPREAD = 0.25

# ============================================================
# INSTRUMENTS (name, attr, pip size, spread_pre, spread_post, swap/day pips)
# ============================================================
INSTRUMENT_CANDIDATES = {
    'EURUSD': ('EUR/USD', 'INSTRUMENT_FX_MAJORS_EUR_USD', 0.0001, 2.0, 1.0, 0.2),
    'GBPUSD': ('GBP/USD', 'INSTRUMENT_FX_MAJORS_GBP_USD', 0.0001, 3.0, 1.5, 0.2),
    'USDJPY': ('USD/JPY', 'INSTRUMENT_FX_MAJORS_USD_JPY', 0.01,   2.0, 1.0, 0.2),
    'AUDUSD': ('AUD/USD', 'INSTRUMENT_FX_MAJORS_AUD_USD', 0.0001, 2.5, 1.5, 0.2),
    'USDCAD': ('USD/CAD', 'INSTRUMENT_FX_MAJORS_USD_CAD', 0.0001, 2.5, 1.8, 0.2),
    'USDCHF': ('USD/CHF', 'INSTRUMENT_FX_MAJORS_USD_CHF', 0.0001, 2.5, 1.8, 0.2),
    'NZDUSD': ('NZD/USD', 'INSTRUMENT_FX_MAJORS_NZD_USD', 0.0001, 3.0, 2.0, 0.2),
    'EURGBP': ('EUR/GBP', 'INSTRUMENT_FX_CROSSES_EUR_GBP', 0.0001, 3.0, 2.0, 0.2),
    'EURJPY': ('EUR/JPY', 'INSTRUMENT_FX_CROSSES_EUR_JPY', 0.01,   3.0, 2.0, 0.2),
    'GBPJPY': ('GBP/JPY', 'INSTRUMENT_FX_CROSSES_GBP_JPY', 0.01,   4.0, 3.0, 0.2),
    'XAUUSD': ('XAU/USD', 'INSTRUMENT_FX_METALS_XAU_USD', 0.01, 50.0, 30.0, 1.0),
    'XAGUSD': ('XAG/USD', 'INSTRUMENT_FX_METALS_XAG_USD', 0.001, 5.0, 3.0, 0.5),
}

START_BALANCE = 10_000.0

# Use pandas Timestamps — compatible with df.index (datetime64[ns])
DEV_START = pd.Timestamp(2005, 1, 1)
DEV_END   = pd.Timestamp(2018, 1, 1)
VAL_END   = pd.Timestamp(2022, 12, 31)
FINAL_END = pd.Timestamp.now()

RNG_SEED = 42
BOOTSTRAP_ITERS = 5000
RANDOM_ITERS = 500


# ============================================================
# INSTRUMENT LOADING
# ============================================================
def load_instruments():
    loaded = {}
    for sym, (name, attr, pip, pre, post, swap) in INSTRUMENT_CANDIDATES.items():
        inst = getattr(inst_mod, attr, None)
        if inst is None:
            print(f"  [SKIP] {sym} ({attr} not available)")
            continue
        loaded[sym] = {
            'name': name,
            'instrument': inst,
            'pip_size': pip,
            'spread_pre': pre,
            'spread_post': post,
            'swap_per_day_pips': swap,
        }
        print(f"  [LOADED] {sym} ({name})")
    return loaded


# ============================================================
# DATA
# ============================================================
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
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None).astype('datetime64[ns]')
    df = df[['open', 'high', 'low', 'close']].dropna()
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


def assert_data(df, sym):
    if df.empty:
        print(f"  [WARN] {sym}: empty")
        return
    first, last = df.index[0], df.index[-1]
    gaps = df.index.to_series().diff().dropna()
    max_gap_days = gaps.max().days if len(gaps) else 0
    bars_per_year = len(df) / max(1, (last - first).days / 365.25)
    print(f"  {sym}: {len(df):5d} bars | {first.date()} → {last.date()} | "
          f"gap≤{max_gap_days}d | ~{bars_per_year:.0f} bars/yr")
    if first.year > 2006:
        print(f"    [WARN] starts late ({first.year})")
    if max_gap_days > 10:
        print(f"    [WARN] max gap {max_gap_days} days")


# ============================================================
# INDICATORS
# ============================================================
def atr_series(high, low, close, n):
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    tr = np.maximum(hl, np.maximum(hc, lc))
    tr[0] = hl[0]
    atr = pd.Series(tr).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().values
    return atr


# ============================================================
# SIMULATOR
# ============================================================
def simulate_trend_trade(entry_idx, side, entry_open, stop_distance,
                         high, low, close, spread_price, swap_per_day_price,
                         max_hold_bars=MAX_HOLD_BARS):
    n = len(high)
    if entry_idx >= n - 1:
        return None
    if not (np.isfinite(entry_open) and entry_open > 0 and np.isfinite(stop_distance) and stop_distance > 0):
        return None

    slip = SLIPPAGE_FRACTION_OF_SPREAD * spread_price
    entry_filled = entry_open + side * (spread_price / 2.0 + slip)

    if side == 1:
        stop = entry_filled - stop_distance
        best = high[entry_idx]
    else:
        stop = entry_filled + stop_distance
        best = low[entry_idx]

    exit_idx = None
    exit_price = None
    status = None

    for j in range(entry_idx, min(entry_idx + max_hold_bars, n)):
        bh, bl = high[j], low[j]

        if side == 1 and bl <= stop:
            exit_price = stop; exit_idx = j; status = 'STOP'; break
        if side == -1 and bh >= stop:
            exit_price = stop; exit_idx = j; status = 'STOP'; break

        if side == 1:
            if bh > best:
                best = bh
            new_stop = best - stop_distance
            if new_stop > stop:
                stop = new_stop
        else:
            if bl < best:
                best = bl
            new_stop = best + stop_distance
            if new_stop < stop:
                stop = new_stop

    if exit_price is None:
        exit_idx = min(entry_idx + max_hold_bars, n - 1)
        exit_price = close[exit_idx]
        status = 'TIME'

    exit_filled = exit_price - side * (spread_price / 2.0 + slip)
    r_gross = side * (exit_filled - entry_filled) / stop_distance

    days_held = max(1, exit_idx - entry_idx)
    swap_r = (swap_per_day_price * days_held) / stop_distance
    r_net = r_gross - swap_r

    return {
        'entry_idx': entry_idx, 'exit_idx': exit_idx,
        'entry_price': entry_filled, 'exit_price': exit_filled,
        'stop_distance': stop_distance, 'status': status,
        'r_gross': r_gross, 'r_net': r_net, 'days_held': days_held,
        'side': 'LONG' if side == 1 else 'SHORT',
    }


# ============================================================
# TRADE GENERATION
# ============================================================
def generate_trades(df, sym, cfg, era_cutover=ERA_CUTOVER_YEAR):
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    idx = df.index

    atr = atr_series(h, l, c, ATR_PERIOD)

    cs = pd.Series(c)
    hh = cs.shift(1).rolling(LOOKBACK_DAYS).max().values
    ll = cs.shift(1).rolling(LOOKBACK_DAYS).min().values

    pip = cfg['pip_size']

    trades = []
    start_i = max(LOOKBACK_DAYS, ATR_PERIOD) + 1
    i = start_i
    next_free = start_i

    while i < len(df) - 2:
        if i < next_free:
            i += 1
            continue
        if not (np.isfinite(atr[i]) and np.isfinite(hh[i]) and np.isfinite(ll[i])):
            i += 1
            continue

        side = 0
        if c[i] > hh[i]:
            side = 1
        elif c[i] < ll[i]:
            side = -1
        if side == 0:
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= len(df):
            break
        stop_distance = ATR_MULT * atr[i]

        year = idx[i].year
        spread_pips = cfg['spread_pre'] if year < era_cutover else cfg['spread_post']
        spread_price = spread_pips * pip
        swap_price = cfg['swap_per_day_pips'] * pip

        t = simulate_trend_trade(
            entry_idx, side, o[entry_idx], stop_distance,
            h, l, c, spread_price, swap_price,
        )
        if t is not None:
            t['symbol'] = sym
            t['entry_time'] = idx[entry_idx]
            t['exit_time'] = idx[t['exit_idx']]
            trades.append(t)
            next_free = t['exit_idx'] + 2
            i = next_free
        else:
            i += 1

    return trades


# ============================================================
# CANARY TESTS
# ============================================================
def canary_buy_hold(df, sym):
    bh = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"  [1] {sym} buy&hold: {bh:+.1f}%")
    return True


def canary_random_entries(df, cfg, n_signals=200, iters=RANDOM_ITERS):
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    atr = atr_series(h, l, c, ATR_PERIOD)
    pip = cfg['pip_size']
    rng = np.random.default_rng(RNG_SEED)

    valid = np.arange(LOOKBACK_DAYS + ATR_PERIOD + 2, len(df) - MAX_HOLD_BARS - 2)
    if len(valid) < 100:
        return None, None

    def run(zero_cost):
        means = []
        for _ in range(iters):
            rs = []
            picks = rng.choice(valid, size=n_signals, replace=True)
            for i in picks:
                if not np.isfinite(atr[i]) or atr[i] <= 0:
                    continue
                side = 1 if rng.random() < 0.5 else -1
                spread_price = 0.0 if zero_cost else cfg['spread_post'] * pip
                swap_price = 0.0 if zero_cost else cfg['swap_per_day_pips'] * pip
                t = simulate_trend_trade(
                    i + 1, side, o[i + 1], ATR_MULT * atr[i],
                    h, l, c, spread_price, swap_price,
                )
                if t is not None:
                    rs.append(t['r_net'])
            if rs:
                means.append(np.mean(rs))
        return np.array(means)

    no_cost = run(True)
    with_cost = run(False)
    return no_cost, with_cost


def canary_lookahead(df, cfg):
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    atr = atr_series(h, l, c, ATR_PERIOD)
    rs = []
    for i in range(LOOKBACK_DAYS + ATR_PERIOD + 2, len(df) - 5, 5):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        nxt_up = c[i + 1] > c[i]
        side = 1 if nxt_up else -1
        sd = 0.1 * atr[i]
        t = simulate_trend_trade(
            i + 1, side, o[i + 1], sd,
            h, l, c, 0.0, 0.0,
            max_hold_bars=2,
        )
        if t is not None:
            rs.append(t['r_net'])
    return float(np.mean(rs)) if rs else 0.0


# ============================================================
# PORTFOLIO METRICS
# ============================================================
def portfolio_monthly_returns(trades):
    if not trades:
        return pd.Series(dtype=float)
    df = pd.DataFrame(trades)
    risk_dollars = START_BALANCE * RISK_PER_TRADE
    df['pnl'] = df['r_net'] * risk_dollars
    df['month'] = pd.to_datetime(df['exit_time']).dt.to_period('M')
    monthly = df.groupby('month')['pnl'].sum()
    return monthly / START_BALANCE


def portfolio_sharpe(monthly_returns):
    if len(monthly_returns) < 12:
        return float('nan')
    m, s = monthly_returns.mean(), monthly_returns.std()
    if s <= 0:
        return 0.0
    return float((m / s) * math.sqrt(12))


def portfolio_drawdown(monthly_returns):
    if len(monthly_returns) == 0:
        return 0.0
    eq = (1 + monthly_returns).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min()) * 100


def summarize_trades(trades):
    if not trades:
        return {'n': 0}
    df = pd.DataFrame(trades)
    r = df['r_net'].values
    n = len(r)
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    wins = int((r > 0).sum())
    rng = np.random.default_rng(RNG_SEED)
    boots = r[rng.integers(0, n, size=(BOOTSTRAP_ITERS, n))].mean(axis=1)
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    return {
        'n': n, 'mean_r': round(mean, 4), 'std_r': round(std, 4),
        'win_rate': round(wins / n * 100, 1),
        'ci95_lo': round(ci_lo, 4), 'ci95_hi': round(ci_hi, 4),
        'total_r': round(float(r.sum()), 2),
        't_stat': round(mean / (std / math.sqrt(n)), 3) if std > 0 else 0.0,
    }


# ============================================================
# REPORTING
# ============================================================
def print_summary_line(label, m):
    if m.get('n', 0) == 0:
        print(f"  {label:<12} no trades")
        return
    print(f"  {label:<12} n={m['n']:4d} | WR {m['win_rate']:5.1f}% | "
          f"mean {m['mean_r']:+.4f}R [{m['ci95_lo']:+.3f},{m['ci95_hi']:+.3f}] | "
          f"total {m['total_r']:+.2f}R | t={m['t_stat']:+.2f}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 72)
    print("   TREND-FOLLOWING BACKTESTER (100d breakout, 3xATR trail)")
    print(f"   USE_HOLDOUT = {USE_HOLDOUT}")
    print("=" * 72)

    print("\n--- LOADING INSTRUMENTS ---")
    instruments = load_instruments()
    if not instruments:
        print("No instruments loaded. Aborting.")
        return

    print("\n--- CANARY TESTS (per instrument) ---")
    canary_failures = []
    for sym, cfg in instruments.items():
        try:
            df = fetch_daily(cfg['instrument'], DEV_START, FINAL_END)
            if df.empty:
                print(f"  [SKIP] {sym} — no data")
                continue
            assert_data(df, sym)
            canary_buy_hold(df, sym)
            nc, wc = canary_random_entries(df, cfg, n_signals=150, iters=200)
            if nc is None or wc is None:
                canary_failures.append(sym)
                continue
            print(f"    random (no cost) mean R = {nc.mean():+.4f}  (expect ~0)")
            print(f"    random (with cost) mean R = {wc.mean():+.4f}  (expect <0)")
            la = canary_lookahead(df, cfg)
            print(f"    look-ahead mean R = {la:+.3f}  (expect > 0.5)")
            if la < 0.5:
                print(f"    [WARN] {sym} look-ahead weak")
                canary_failures.append(sym)
        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")
            canary_failures.append(sym)

    if canary_failures:
        print(f"\n[WARN] Canary issues on: {canary_failures}")
        print("Continuing, but treat results with caution.")

    print("\n--- GENERATING TRADES ---")
    all_trades = []
    for sym, cfg in instruments.items():
        try:
            df = fetch_daily(cfg['instrument'], DEV_START, FINAL_END)
            if df.empty:
                continue
            if not USE_HOLDOUT:
                df = df[df.index <= VAL_END]
            trades = generate_trades(df, sym, cfg)
            print(f"  {sym}: {len(trades)} trades")
            all_trades.extend(trades)
        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")

    if not all_trades:
        print("\nNo trades generated.")
        return

    all_trades.sort(key=lambda t: t['exit_time'])

    dev = [t for t in all_trades if t['entry_time'] < DEV_END]
    val = [t for t in all_trades if DEV_END <= t['entry_time'] <= VAL_END]
    hold = [t for t in all_trades if t['entry_time'] > VAL_END]

    print("\n" + "=" * 72)
    print("          TREND-FOLLOWING REPORT")
    print("=" * 72)

    print(f"\n--- ALL TRADES ---")
    print_summary_line('ALL', summarize_trades(all_trades))
    print(f"\n--- DEV (2005-2018) ---")
    print_summary_line('DEV', summarize_trades(dev))
    print(f"\n--- VALIDATION (2018-2022) ---")
    print_summary_line('VAL', summarize_trades(val))
    if USE_HOLDOUT:
        print(f"\n--- HOLDOUT (2023-now) ---")
        print_summary_line('HOLD', summarize_trades(hold))

    print(f"\n--- PER INSTRUMENT ---")
    for sym in instruments:
        sub = [t for t in all_trades if t['symbol'] == sym]
        print_summary_line(sym, summarize_trades(sub))

    print(f"\n--- PORTFOLIO METRICS ---")
    monthly = portfolio_monthly_returns(all_trades)
    if len(monthly) > 0:
        sharpe = portfolio_sharpe(monthly)
        dd_min = portfolio_drawdown(monthly)
        total_ret = ((1 + monthly).prod() - 1) * 100
        print(f"  Months tracked: {len(monthly)}")
        print(f"  Total return: {total_ret:+.1f}%")
        print(f"  Annualized Sharpe: {sharpe:.2f}")
        print(f"  Max drawdown: {dd_min:.1f}%")
        print(f"  Best month: {monthly.max()*100:+.2f}% | Worst: {monthly.min()*100:+.2f}%")

        rng = np.random.default_rng(RNG_SEED)
        boots = monthly.values[rng.integers(0, len(monthly), size=(BOOTSTRAP_ITERS, len(monthly)))].mean(axis=1)
        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))
        print(f"  Monthly mean CI95: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    else:
        print("  No monthly data.")

    print(f"\n--- VERDICT ---")
    n_trades = len(all_trades)
    passed = True
    if n_trades < MIN_TRADES_REQUIRED:
        print(f"  FAIL: only {n_trades} trades (< {MIN_TRADES_REQUIRED})")
        passed = False
    if len(monthly) > 0:
        sharpe = portfolio_sharpe(monthly)
        if sharpe < 0.30:
            print(f"  FAIL: Sharpe {sharpe:.2f} < 0.30")
            passed = False
        rng = np.random.default_rng(RNG_SEED)
        boots = monthly.values[rng.integers(0, len(monthly), size=(BOOTSTRAP_ITERS, len(monthly)))].mean(axis=1)
        if float(np.percentile(boots, 2.5)) <= 0:
            print(f"  FAIL: monthly mean CI lower bound ≤ 0")
            passed = False
    if passed:
        print("  PASS (directional). Forward-test on demo before any live capital.")
    print("=" * 72)

    pd.DataFrame(all_trades).to_csv('fx_trades.csv', index=False)
    print(f"\nSaved {len(all_trades)} trades to fx_trades.csv")


if __name__ == '__main__':
    main()
