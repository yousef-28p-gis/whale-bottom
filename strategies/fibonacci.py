#!/usr/bin/env python3
"""
Fibonacci Retracement — built on ZigZag pivot points
Draws fib levels for every bullish wave (L→H).

Usage:
    from strategies.zigzag import zigzag
    from strategies.fibonacci import get_waves, fib_levels
    
    pivots = zigzag(highs, lows, depth=10, dev=1.0)
    waves = get_waves(pivots)
    for L, H in waves:
        levels = fib_levels(L[1], H[1])
"""

def get_waves(pivots):
    """
    Extract bullish waves (L→H) from zigzag pivots.
    Returns list of ((idx_L, price_L, 'L'), (idx_H, price_H, 'H'))
    """
    return [(pivots[i], pivots[i+1])
            for i in range(len(pivots) - 1)
            if pivots[i][2] == 'L' and pivots[i+1][2] == 'H']


def fib_levels(low_price, high_price, levels=None):
    """
    Calculate Fibonacci retracement levels for a wave.
    
    Args:
        low_price: wave low (start)
        high_price: wave high (end)
        levels: list of fib ratios (default: standard set)
    
    Returns:
        dict {ratio: price}
    """
    if levels is None:
        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    wave_height = high_price - low_price
    return {lvl: high_price - wave_height * lvl for lvl in levels}


def retracement_pct(wave_L, wave_H, current_low):
    """
    Calculate current retracement percentage.
    
    Returns:
        float: 0.0 to 1.0 (e.g., 0.618 = 61.8% retracement)
    """
    wave_height = wave_H - wave_L
    if wave_height <= 0:
        return 1.0
    return (wave_H - current_low) / wave_height


# ── Example ──
if __name__ == "__main__":
    import ccxt
    import pandas as pd
    from zigzag import zigzag

    exchange = ccxt.binance({'timeout': 15000})
    ohlcv = exchange.fetch_ohlcv('FET/USDT', '3m', limit=500)
    df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')

    pivots = zigzag(df['h'].values, df['l'].values, depth=10, dev=1.0)
    waves = get_waves(pivots)
    
    print(f"Total waves: {len(waves)}")
    if waves:
        L, H = waves[-1]
        levels = fib_levels(L[1], H[1])
        print(f"\nLast wave: {L[1]:.6f} → {H[1]:.6f}")
        for lvl, price in levels.items():
            print(f"  Fib {lvl:.3f}: {price:.6f}")
        
        current_low = df['l'].iloc[H[0]:].min()
        ret = retracement_pct(L[1], H[1], current_low)
        print(f"\nCurrent retracement: {ret:.1%}")
