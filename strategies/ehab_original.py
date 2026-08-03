#!/usr/bin/env python3
"""
🦅 استراتيجية ايهاب الاصلية (EA Free Signals Clone)
──────────────────────────────────────────────────
Reverse-engineered from EA Free Signals Telegram channel.

Entry: Volume > 3× average + Break 5-bar high + Green candle
TP:    Entry + 0.7%
SL:    Entry - 1.8%
Max:   12 hours (48 bars on 15m)

Filters:
- Cooldown 12h per coin
- Top 30 coins by volume only

Performance (3 months, 15m):
- ~200 trades, WR ~68%, R:R ~0.33
- High WR but small R:R — needs modification for profitability
"""

COMM = 0.2        # 0.2%
TP1_PCT = 0.7     # +0.7% target
SL_PCT = 1.8      # -1.8% stop
MAX_BARS = 48     # 12 hours
VOL_MULT = 3.0    # Volume > 3× average
COOLDOWN_BARS = 48 # 12h cooldown
LOOKBACK_BARS = 5  # Break 5-bar high
TOP_N_COINS = 30   # Top coins by volume

import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta

def fetch_top_coins(exchange, n=TOP_N_COINS):
    """Get top N coins by 24h USDT volume"""
    tickers = exchange.fetch_tickers()
    skip = {'USDC','BUSD','DAI','TUSD','USDE','FDUSD','USDD','FRAX','LUSD',
            'PYUSD','USDJ','RLUSD','XAUT','EUR','PAXG','BTC','ETH'}
    vols = {}
    for sym, t in tickers.items():
        if sym.endswith('/USDT') and t.get('quoteVolume'):
            coin = sym.replace('/USDT','')
            if coin not in skip:
                vols[coin] = t['quoteVolume']
    return sorted(vols, key=vols.get, reverse=True)[:n]

def fetch_ohlcv(exchange, coin, tf='15m', days=90):
    """Fetch OHLCV data for a coin"""
    since = exchange.parse8601((datetime.now() - timedelta(days=days+3)).isoformat())
    return exchange.fetch_ohlcv(f'{coin}/USDT', tf, since=since, limit=10000)

def check_entry(c, h, v, avg_vol, i, last_entry):
    """Check if entry conditions are met"""
    if i - last_entry < COOLDOWN_BARS:
        return False
    if v[i] <= avg_vol[i] * VOL_MULT:
        return False
    if c[i] <= max(h[max(0,i-LOOKBACK_BARS):i]):
        return False
    if c[i] <= c[i-1]:
        return False
    return True

def simulate_exit(c, h, l, i, ep):
    """Simulate exit for a trade entered at bar i with price ep"""
    tp = ep * (1 + TP1_PCT/100)
    sl = ep * (1 - SL_PCT/100)
    n = len(c)
    
    for j in range(i+1, min(i+MAX_BARS, n)):
        if l[j] <= sl:
            return sl, 'SL', j-i
        elif h[j] >= tp:
            return tp, 'TP', j-i
    
    end = min(i+MAX_BARS, n-1)
    return c[end], 'TIME', end-i

# Alias for clarity
EhabStrategy = {
    'name': 'استراتيجية ايهاب الاصلية',
    'tp_pct': TP1_PCT,
    'sl_pct': SL_PCT,
    'vol_mult': VOL_MULT,
    'cooldown': COOLDOWN_BARS,
    'lookback': LOOKBACK_BARS,
    'max_bars': MAX_BARS,
    'top_coins': TOP_N_COINS,
}

if __name__ == '__main__':
    print(f"🦅 {EhabStrategy['name']}")
    print(f"   TP={TP1_PCT}% SL={SL_PCT}% VOL>{VOL_MULT}x COOLDOWN={COOLDOWN_BARS/4}h")
