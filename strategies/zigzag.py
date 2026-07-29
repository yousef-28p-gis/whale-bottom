#!/usr/bin/env python3
"""
ZigZag Indicator — Python port of TradingView Pine Script ZigZag
Matches: depth=10 (5 bars each side), deviation threshold

Usage:
    from zigzag import zigzag
    pivots = zigzag(highs, lows, depth=10, dev=1.0)

Returns list of (bar_index, price, 'H'/'L') tuples.
"""

def zigzag(highs, lows, depth=10, dev=1.0):
    """
    Compute ZigZag pivot points.

    Args:
        highs: list/array of high prices
        lows:  list/array of low prices
        depth: total depth (bars on each side = depth // 2)
        dev:   minimum % change to confirm a new leg

    Returns:
        List of (bar_index, price, type) where type is 'H' or 'L'
    """
    D = depth // 2  # bars on each side
    h = list(highs)
    l = list(lows)
    n = len(h)

    # ── Pivot detection (Pine Script logic) ──
    # Left side: strict comparison  (< or >)
    # Right side: non-strict comparison (<= or >=)
    pivots = []
    for i in range(D, n - D):
        # Pivot High
        is_ph = all(h[j] < h[i] for j in range(i - D, i)) and \
                all(h[j] <= h[i] for j in range(i + 1, i + D + 1))
        if is_ph:
            pivots.append((i, h[i], 'H'))

        # Pivot Low
        is_pl = all(l[j] > l[i] for j in range(i - D, i)) and \
                all(l[j] >= l[i] for j in range(i + 1, i + D + 1))
        if is_pl:
            pivots.append((i, l[i], 'L'))

    pivots.sort(key=lambda x: x[0])

    # ── Deviation filter + alternation ──
    # Same-type pivots: replace if more extreme
    # Opposite-type: require deviation >= threshold
    filtered = []
    for idx, price, pt in pivots:
        if not filtered:
            filtered.append((idx, price, pt))
        elif pt == filtered[-1][2]:
            # Same direction: replace if more extreme
            if (pt == 'H' and price > filtered[-1][1]) or \
               (pt == 'L' and price < filtered[-1][1]):
                filtered[-1] = (idx, price, pt)
        else:
            # Opposite direction: check deviation
            pct = abs(price - filtered[-1][1]) / filtered[-1][1] * 100
            if pct >= dev:
                filtered.append((idx, price, pt))

    return filtered


# ── Example usage ──
if __name__ == "__main__":
    import ccxt
    import pandas as pd

    exchange = ccxt.binance({'timeout': 15000})
    ohlcv = exchange.fetch_ohlcv('FET/USDT', '3m', limit=500)
    df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')

    for dev_val in [1.0, 2.0]:
        res = zigzag(df['h'].values, df['l'].values, depth=10, dev=dev_val)
        print(f"\ndev={dev_val}%: {len(res)} pivots")
        for idx, price, pt in res:
            print(f"  [{idx}] {df['dt'].iloc[idx]}  {pt}  {price:.6f}")
