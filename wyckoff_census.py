# wyckoff_census.py
import pandas as pd
import numpy as np
import datetime as dt
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_EUR_USD, INSTRUMENT_FX_METALS_XAU_USD

# --- PARAMETERS (Fixed, not optimized) ---
LOOKBACK_BARS = 40
RANGE_WIDTH_TOLERANCE = 0.25
SPRING_WICK_RATIO = 1.5
CONFIRMATION_BARS = 1

# --- DATA CONFIG ---
START_DATE = dt.datetime(2005, 1, 1)
END_DATE = dt.datetime.now()
PAIRS = {
    'EURUSD': INSTRUMENT_FX_MAJORS_EUR_USD,
    'XAUUSD': INSTRUMENT_FX_METALS_XAU_USD
}

def fetch_dukascopy_data(instrument, start, end):
    print(f"Fetching {instrument} data...")
    df = dukascopy_python.fetch(
        instrument,
        dukascopy_python.INTERVAL_HOUR_4,
        dukascopy_python.OFFER_SIDE_BID,
        start,
        end
    )
    return df

def detect_wyckoff_signals(df):
    """Simple Wyckoff Spring/UTAD detector for a census."""
    signals = []
    df = df.copy()
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    for i in range(LOOKBACK_BARS, len(df)):
        window = df.iloc[i-LOOKBACK_BARS:i]
        current_bar = df.iloc[i]
        
        support_level = window['low'].min()
        resistance_level = window['high'].max()
        range_width = resistance_level - support_level
        atr = current_bar['atr']
        
        if pd.isna(atr) or range_width < 0.5 * atr or range_width > 5 * atr:
            continue

        # --- Spring Detection (Long) ---
        if current_bar['low'] < support_level:
            if current_bar['close'] > support_level:
                lower_wick = current_bar['open'] - current_bar['low'] if current_bar['open'] > current_bar['close'] else current_bar['close'] - current_bar['low']
                body = abs(current_bar['open'] - current_bar['close'])
                if body > 0 and lower_wick > SPRING_WICK_RATIO * body:
                    signals.append({'type': 'Spring', 'date': df.index[i]})

        # --- UTAD Detection (Short) ---
        if current_bar['high'] > resistance_level:
            if current_bar['close'] < resistance_level:
                upper_wick = current_bar['high'] - current_bar['open'] if current_bar['open'] < current_bar['close'] else current_bar['high'] - current_bar['close']
                body = abs(current_bar['open'] - current_bar['close'])
                if body > 0 and upper_wick > SPRING_WICK_RATIO * body:
                    signals.append({'type': 'UTAD', 'date': df.index[i]})

    return signals

if __name__ == "__main__":
    total_signals = 0
    for pair, instrument in PAIRS.items():
        print(f"\n--- Processing {pair} ---")
        try:
            df = fetch_dukascopy_data(instrument, START_DATE, END_DATE)
            if df is not None and not df.empty:
                print(f"Fetched {len(df)} bars.")
                signals = detect_wyckoff_signals(df)
                print(f"Found {len(signals)} total signals on {pair}:")
                if signals:
                    signal_types = pd.Series([s['type'] for s in signals]).value_counts()
                    print(signal_types)
                total_signals += len(signals)
            else:
                print(f"Could not fetch data for {pair}.")
        except Exception as e:
            print(f"An error occurred processing {pair}: {e}")

    print(f"\n--- CENSUS COMPLETE ---")
    print(f"Total signals found across all pairs: {total_signals}")
    print("If this number is below 150, the strategy cannot be statistically validated.")
