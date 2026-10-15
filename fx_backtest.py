#!/usr/bin/env python3
"""
fx_backtest.py — Wyckoff Spring/UTAD swing strategy backtester for FX.
Assets: EURUSD, XAUUSD. Timeframe: 4H primary, 1D trend context.
Data: Dukascopy via dukascopy-python.
"""
import datetime as dt
import math
import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_METALS_XAU_USD,
)

# ============================================================
# CONFIG (fixed — not to be tuned after seeing results)
# ============================================================
INSTRUMENTS = {
    'EURUSD': INSTRUMENT_FX_MAJORS_EUR_USD,
    'XAUUSD': INSTRUMENT_FX_METALS_XAU_USD,
}

PIP_SIZE = {'EURUSD': 0.0001, 'XAUUSD': 0.01}
SPREAD_PIPS = {'EURUSD': 1.0, 'XAUUSD': 30.0}
SLIPPAGE_PIPS = {'EURUSD': 0.5, 'XAUUSD': 5.0}
SWAP_PIPS_PER_DAY = {'EURUSD': 0.2, 'XAUUSD': 1.0}

LOOKBACK = 40
TOUCH_TOL_ATR = 0.25
MIN_TOUCHES_PER_EDGE = 2
RANGE_WIDTH_MIN_ATR = 1.5
RANGE_WIDTH_MAX_ATR = 6.0
RANGE_WIDTH_MAX_PCT = 0.05
WICK_RATIO_MIN = 0.5
MIN_BARS_BETWEEN_SIGNALS = 20

RR_RATIO = 2.0
STOP_BUFFER_ATR = 0.25
TIME_STOP_BARS = 60
WARMUP_BARS = 300

STARTING_BALANCE = 10_000.0
RISK_PER_TRADE = 0.01
LEVERAGE_CAP = 3.0

# Timezone-aware split dates (fix: Dukascopy returns tz-aware timestamps)
DEV_START = dt.datetime(2005, 1, 1, tzinfo=dt.timezone.utc)
DEV_END = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
VAL_END = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
HOLDOUT_END = dt.datetime.now(tz=dt.timezone.utc)

GATES = {
    'min_oos_trades': 100,
    'min_oos_mean_r': 0.05,
    'ci_lower_gt_zero': True,
    'baseline_p_max': 0.05,
    'min_sharpe': 0.30,
}

RNG_SEED = 42
BOOTSTRAP_ITERS = 5000
MONTE_CARLO_ITERS = 3000


# ============================================================
# DATA
# ============================================================
def fetch_4h(instrument, start, end):
    print(f"  Fetching 4H from {start.date()} to {end.date()}...")
    df = dukascopy_python.fetch(
        instrument,
        dukascopy_python.INTERVAL_HOUR_4,
        dukascopy_python.OFFER_SIDE_BID,
        start,
        end,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


def compute_atr(df, n=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def compute_ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


# ============================================================
# WYCKOFF DETECTOR (strict)
# ============================================================
def detect_signals(df, pip_size):
    df = df.copy()
    df['atr'] = compute_atr(df)
    df['ema20'] = compute_ema(df['close'], 20)
    df['ema50'] = compute_ema(df['close'], 50)
    df = df.reset_index()

    signals = []
    n = len(df)
    last_signal_idx = -10**9

    for i in range(WARMUP_BARS, n - 1):
        if i - last_signal_idx < MIN_BARS_BETWEEN_SIGNALS:
            continue

        atr = df.loc[i, 'atr']
        if pd.isna(atr) or atr <= 0:
            continue

        win_start = i - LOOKBACK
        win = df.iloc[win_start:i]
        if win.empty:
            continue

        H = win['high'].max()
        L = win['low'].min()
        range_width = H - L
        avg_close = win['close'].mean()

        if range_width < RANGE_WIDTH_MIN_ATR * atr:
            continue
        if range_width > RANGE_WIDTH_MAX_ATR * atr:
            continue
        if range_width / avg_close > RANGE_WIDTH_MAX_PCT:
            continue

        tol = TOUCH_TOL_ATR * atr
        touches_high = int((win['high'] >= H - tol).sum())
        touches_low = int((win['low'] <= L + tol).sum())
        if touches_high < MIN_TOUCHES_PER_EDGE or touches_low < MIN_TOUCHES_PER_EDGE:
            continue

        bar = df.iloc[i]
        bar_range = bar['high'] - bar['low']
        if bar_range <= 0:
            continue

        # Spring (LONG)
        if bar['low'] < L and bar['close'] > L:
            lower_wick = min(bar['open'], bar['close']) - bar['low']
            if lower_wick / bar_range >= WICK_RATIO_MIN:
                stop = bar['low'] - STOP_BUFFER_ATR * atr
                signals.append({
                    'idx': i, 'type': 'Spring', 'side': 1,
                    'entry_idx': i + 1, 'stop_price': stop,
                })
                last_signal_idx = i
                continue

        # UTAD (SHORT)
        if bar['high'] > H and bar['close'] < H:
            upper_wick = bar['high'] - max(bar['open'], bar['close'])
            if upper_wick / bar_range >= WICK_RATIO_MIN:
                stop = bar['high'] + STOP_BUFFER_ATR * atr
                signals.append({
                    'idx': i, 'type': 'UTAD', 'side': -1,
                    'entry_idx': i + 1, 'stop_price': stop,
                })
                last_signal_idx = i

    return signals


# ============================================================
# SIMULATOR
# ============================================================
def simulate_one(df, signal, pip_size, spread_pips, slip_pips, swap_pips_per_day):
    entry_idx = signal['entry_idx']
    if entry_idx >= len(df):
        return None

    side = signal['side']
    entry_bar = df.iloc[entry_idx]
    entry_price = entry_bar['open']

    half_spread = (spread_pips / 2.0) * pip_size
    slip = slip_pips * pip_size
    entry_filled = entry_price + side * (half_spread + slip)

    stop = signal['stop_price']
    risk_distance = abs(entry_filled - stop)
    if risk_distance <= 0:
        return None

    if side == 1:
        target = entry_filled + RR_RATIO * risk_distance
    else:
        target = entry_filled - RR_RATIO * risk_distance

    for j in range(entry_idx, min(entry_idx + TIME_STOP_BARS, len(df))):
        bar = df.iloc[j]
        if side == 1:
            hit_sl = bar['low'] <= stop
            hit_tp = bar['high'] >= target
        else:
            hit_sl = bar['high'] >= stop
            hit_tp = bar['low'] <= target

        if hit_sl and hit_tp:
            status = 'LOSS'
            exit_price = stop
            exit_idx = j
            break
        if hit_sl:
            status = 'LOSS'
            exit_price = stop
            exit_idx = j
            break
        if hit_tp:
            status = 'WIN'
            exit_price = target
            exit_idx = j
            break
    else:
        j = min(entry_idx + TIME_STOP_BARS, len(df) - 1)
        bar = df.iloc[j]
        exit_price = bar['close']
        status = 'TIME_STOP'
        exit_idx = j

    exit_filled = exit_price - side * (half_spread + slip)
    r_gross = side * (exit_filled - entry_filled) / risk_distance

    days_held = max(1, (exit_idx - entry_idx + 1) * 4 / 24)
    swap_cost = swap_pips_per_day * days_held * pip_size
    swap_r = swap_cost / risk_distance
    r_net = r_gross - swap_r

    return {
        'entry_idx': entry_idx,
        'exit_idx': exit_idx,
        'entry_time': df.index[entry_idx],
        'exit_time': df.index[exit_idx],
        'type': signal['type'],
        'side': 'LONG' if side == 1 else 'SHORT',
        'entry': entry_filled,
        'stop': stop,
        'target': target,
        'exit': exit_filled,
        'status': status,
        'r_gross': r_gross,
        'r_net': r_net,
        'risk_price': risk_distance,
        'stop_pct': risk_distance / entry_filled * 100,
    }


# ============================================================
# METRICS
# ============================================================
def summarize(trades):
    if not trades:
        return {'n': 0}
    df = pd.DataFrame(trades).sort_values('exit_time')
    n = len(df)
    r = df['r_net'].values
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0

    eq_r = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate([[0], eq_r]))[1:]
    dd_r = float((eq_r - peak).min())

    rng = np.random.default_rng(RNG_SEED)
    boots = r[rng.integers(0, n, size=(BOOTSTRAP_ITERS, n))].mean(axis=1)
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    wins = int((df['status'] == 'WIN').sum())
    losses = int((df['status'] == 'LOSS').sum())
    ts = int((df['status'] == 'TIME_STOP').sum())
    win_rate = wins / n * 100 if n else 0

    return {
        'n': n,
        'wins': wins, 'losses': losses, 'time_stops': ts,
        'win_rate': round(win_rate, 1),
        'mean_r': round(mean, 4),
        'median_r': round(float(np.median(r)), 4),
        'ci95_lo': round(ci_lo, 4),
        'ci95_hi': round(ci_hi, 4),
        'std_r': round(std, 4),
        'total_r': round(float(r.sum()), 2),
        'max_dd_r': round(dd_r, 2),
        't_stat': round(mean / (std / math.sqrt(n)), 3) if std > 0 else 0.0,
    }


def random_entry_baseline(df, n_signals, pip_size, spread_pips, slip_pips, swap_pips_per_day,
                           iters=MONTE_CARLO_ITERS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    returns = []
    valid_idxs = np.arange(WARMUP_BARS, len(df) - 100)

    for _ in range(iters):
        rs = []
        idxs = rng.choice(valid_idxs, size=n_signals, replace=True)
        for i in idxs:
            bar = df.iloc[i]
            side = 1 if rng.random() < 0.5 else -1
            entry_price = bar['open']
            half_spread = (spread_pips / 2.0) * pip_size
            slip = slip_pips * pip_size
            entry_filled = entry_price + side * (half_spread + slip)

            stop_distance = entry_price * 0.01
            stop = entry_filled - side * stop_distance
            target = entry_filled + side * RR_RATIO * stop_distance

            r_net = _simulate_quick(df, i + 1, side, entry_filled, stop, target,
                                    stop_distance, pip_size, half_spread, slip,
                                    swap_pips_per_day)
            if r_net is not None:
                rs.append(r_net)
        if rs:
            returns.append(np.mean(rs))
    return np.array(returns)


def _simulate_quick(df, entry_idx, side, entry_filled, stop, target, risk_distance,
                     pip_size, half_spread, slip, swap_pips_per_day):
    if entry_idx >= len(df):
        return None
    for j in range(entry_idx, min(entry_idx + TIME_STOP_BARS, len(df))):
        bar = df.iloc[j]
        if side == 1:
            hit_sl = bar['low'] <= stop
            hit_tp = bar['high'] >= target
        else:
            hit_sl = bar['high'] >= stop
            hit_tp = bar['low'] <= target
        if hit_sl:
            r_gross = side * (stop - entry_filled) / risk_distance
            days_held = max(1, (j - entry_idx + 1) * 4 / 24)
            return r_gross - (swap_pips_per_day * days_held * pip_size / risk_distance)
        if hit_tp:
            r_gross = side * (target - entry_filled) / risk_distance
            days_held = max(1, (j - entry_idx + 1) * 4 / 24)
            return r_gross - (swap_pips_per_day * days_held * pip_size / risk_distance)
    return None


# ============================================================
# CANARY TESTS
# ============================================================
def _lookahead_profit(df, pip_size):
    rs = []
    for i in range(WARMUP_BARS, len(df) - 2, 10):
        nxt = df.iloc[i + 1]
        entry = df.iloc[i]['close']
        if nxt['close'] > entry:
            side = 1
            exit_price = nxt['high']
        else:
            side = -1
            exit_price = nxt['low']
        r = side * (exit_price - entry) / (entry * 0.001)
        rs.append(r)
    return float(np.mean(rs)) if rs else 0.0


def canary_tests(df, pip_size):
    print("\n--- CANARY TESTS ---")
    passed = True

    bh = (df['close'].iloc[-1] / df['close'].iloc[WARMUP_BARS] - 1) * 100
    print(f"  [1] Buy & Hold over window: {bh:+.1f}%")
    if not (isinstance(bh, float) and math.isfinite(bh)):
        print("      FAIL — buy&hold did not compute")
        passed = False
    else:
        print("      PASS — buy&hold reproducible")

    print(f"  [2] Random entries with costs...")
    n_random = 200
    rand_returns = random_entry_baseline(df, n_random, pip_size, 1.0, 0.5, 0.2,
                                          iters=200, seed=RNG_SEED)
    if len(rand_returns):
        mean_rand = float(rand_returns.mean())
        print(f"      mean R = {mean_rand:+.4f}  (expected: ≤ 0 after costs)")
        if mean_rand > 0.15:
            print("      FAIL — random entries profitable → costs too low or bug")
            passed = False
        else:
            print("      PASS — random entries lose to costs as expected")
    else:
        print("      FAIL — random baseline returned no values")
        passed = False

    print(f"  [3] Look-ahead canary (cheat: uses next bar's high)...")
    lookahead_profit = _lookahead_profit(df, pip_size)
    print(f"      look-ahead expectancy: {lookahead_profit:+.3f}R per trade")
    if lookahead_profit > 1.0:
        print("      PASS — look-ahead produces absurd profit, engine correct")
    else:
        print("      FAIL — look-ahead not extracting absurd profit → engine issue")
        passed = False

    print(f"--- CANARY RESULT: {'PASS' if passed else 'FAIL'} ---")
    return passed


# ============================================================
# BACKTEST RUNNER
# ============================================================
def run_pair(pair, instrument):
    print(f"\n=== {pair} ===")
    df = fetch_4h(instrument, DEV_START, HOLDOUT_END)
    if df.empty:
        print(f"  No data for {pair}")
        return None

    print(f"  Bars: {len(df)} | {df.index[0].date()} → {df.index[-1].date()}")
    pip = PIP_SIZE[pair]
    spread = SPREAD_PIPS[pair]
    slip = SLIPPAGE_PIPS[pair]
    swap = SWAP_PIPS_PER_DAY[pair]

    if not canary_tests(df, pip):
        print(f"  ⚠️ Canary failed for {pair}. Aborting.")
        return None

    signals = detect_signals(df, pip)
    print(f"  Signals after strict filter: {len(signals)}")

    trades = []
    for sig in signals:
        t = simulate_one(df, sig, pip, spread, slip, swap)
        if t is not None:
            t['pair'] = pair
            trades.append(t)

    print(f"  Simulated trades: {len(trades)}")
    return trades, df


def _print_summary_line(label, m):
    if m.get('n', 0) == 0:
        print(f"  {label:<8} no trades")
        return
    print(f"  {label:<8} n={m['n']:4d} | WR {m['win_rate']:5.1f}% | "
          f"mean {m['mean_r']:+.4f}R [{m['ci95_lo']:+.3f},{m['ci95_hi']:+.3f}] | "
          f"total {m['total_r']:+.2f}R | DD {m['max_dd_r']:+.2f}R | t={m['t_stat']:+.2f}")


def print_report(all_trades, per_pair_df):
    print("\n" + "=" * 70)
    print("          FX WYCKOFF BACKTEST REPORT")
    print("=" * 70)

    if not all_trades:
        print("No trades.")
        return None

    dev = [t for t in all_trades if t['entry_time'] < DEV_END]
    val = [t for t in all_trades if DEV_END <= t['entry_time'] < VAL_END]
    holdout = [t for t in all_trades if t['entry_time'] >= VAL_END]

    print(f"\n--- ALL TRADES ---")
    all_m = summarize(all_trades)
    _print_summary_line('ALL', all_m)

    print(f"\n--- DEV (2005-2018) ---")
    dev_m = summarize(dev)
    _print_summary_line('DEV', dev_m)

    print(f"\n--- VALIDATION (2018-2023) ---")
    val_m = summarize(val)
    _print_summary_line('VAL', val_m)

    print(f"\n--- HOLDOUT (2023-now) ---")
    hol_m = summarize(holdout)
    _print_summary_line('HOLD', hol_m)

    print(f"\n--- PER PAIR ---")
    for pair in per_pair_df:
        sub = [t for t in all_trades if t['pair'] == pair]
        _print_summary_line(pair, summarize(sub))

    print(f"\n--- BY TYPE ---")
    for t_type in ['Spring', 'UTAD']:
        sub = [t for t in all_trades if t['type'] == t_type]
        _print_summary_line(t_type, summarize(sub))

    print(f"\n--- VERDICT ---")
    if not holdout or hol_m['n'] < 20:
        print(f"  HOLDOUT has only {hol_m.get('n', 0)} trades — INCONCLUSIVE")
    elif hol_m['mean_r'] <= 0:
        print(f"  HOLDOUT mean R = {hol_m['mean_r']:+.4f} → FAIL")
    elif hol_m['ci95_lo'] <= 0:
        print(f"  HOLDOUT 95% CI lower = {hol_m['ci95_lo']:+.4f} → INCONCLUSIVE")
    else:
        print(f"  HOLDOUT mean R = {hol_m['mean_r']:+.4f}, CI lo = {hol_m['ci95_lo']:+.4f} → PASS")

    print("=" * 70)
    return {'all': all_m, 'dev': dev_m, 'val': val_m, 'holdout': hol_m}


# ============================================================
# MAIN
# ============================================================
def main():
    all_trades = []
    per_pair_df = {}

    for pair, instrument in INSTRUMENTS.items():
        result = run_pair(pair, instrument)
        if result is None:
            continue
        trades, df = result
        all_trades.extend(trades)
        per_pair_df[pair] = df

    all_trades.sort(key=lambda t: t['entry_time'])
    print_report(all_trades, per_pair_df)

    if all_trades:
        out = pd.DataFrame(all_trades)
        out.to_csv('fx_trades.csv', index=False)
        print(f"\nSaved {len(all_trades)} trades to fx_trades.csv")


if __name__ == "__main__":
    main()
