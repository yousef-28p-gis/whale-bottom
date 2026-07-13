#!/usr/bin/env python3
"""Check signals for multiple pairs and strategies."""
import ccxt
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import sys

def analyze(symbol, timeframe='4h'):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)

    price = df['close'].iloc[-1]
    change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100

    # SMA
    sma20 = SMAIndicator(df['close'], 20).sma_indicator().iloc[-1]
    sma50 = SMAIndicator(df['close'], 50).sma_indicator().iloc[-1]
    sma_signal = 1 if sma20 > sma50 else -1

    # RSI
    rsi = RSIIndicator(df['close'], 14).rsi().iloc[-1]
    if rsi < 30:
        rsi_signal = 1
    elif rsi > 70:
        rsi_signal = -1
    else:
        rsi_signal = 0

    # MACD
    macd = MACD(df['close'], 26, 12, 9)
    macd_line = macd.macd().iloc[-1]
    signal_line = macd.macd_signal().iloc[-1]
    macd_signal = 1 if macd_line > signal_line else -1

    # Bollinger
    bb = BollingerBands(df['close'], 20, 2)
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    if price < lower:
        bb_signal = 1
    elif price > upper:
        bb_signal = -1
    else:
        bb_signal = 0

    # Consensus
    signals = [sma_signal, rsi_signal, macd_signal, bb_signal]
    bullish = sum(1 for s in signals if s == 1)
    bearish = sum(1 for s in signals if s == -1)

    if bullish > bearish:
        consensus = "🟢 BULLISH"
    elif bearish > bullish:
        consensus = "🔴 BEARISH"
    else:
        consensus = "⚪ NEUTRAL"

    return {
        'symbol': symbol,
        'price': price,
        'change': change,
        'consensus': consensus,
        'bullish_count': bullish,
        'bearish_count': bearish,
        'sma': 'BULL' if sma_signal == 1 else 'BEAR',
        'rsi': round(rsi, 1),
        'macd': 'BULL' if macd_signal == 1 else 'BEAR',
        'bb': 'BUY' if bb_signal == 1 else ('SELL' if bb_signal == -1 else 'HOLD'),
    }

# Default pairs to scan
PAIRS = sys.argv[1:] if len(sys.argv) > 1 else ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']

print(f"📊 Multi-Pair Signal Scanner ({len(PAIRS)} pairs, 4h TF)")
print("=" * 70)

results = []
for symbol in PAIRS:
    try:
        r = analyze(symbol)
        results.append(r)
        print(f"{r['consensus']} {r['symbol']:<12} ${r['price']:>10,.2f} ({r['change']:+.2f}%)")
        print(f"   SMA:{r['sma']} | RSI:{r['rsi']} | MACD:{r['macd']} | BB:{r['bb']} ({r['bullish_count']}/{r['bearish_count']}/4)")
    except Exception as e:
        print(f"⚠️  {symbol}: Error - {e}")

# Summary
bullish_count = sum(1 for r in results if r['consensus'] == '🟢 BULLISH')
bearish_count = sum(1 for r in results if r['consensus'] == '🔴 BEARISH')
print(f"\n📈 Summary: {bullish_count} Bullish | {bearish_count} Bearish | {len(results)-bullish_count-bearish_count} Neutral")

# Top picks
if bullish_count > 0:
    print(f"\n🏆 Top Bullish: ", end="")
    for r in results:
        if r['consensus'] == '🟢 BULLISH':
            print(f"{r['symbol']} (+{r['change']:.1f}%) ", end="")
    print()

if bearish_count > 0:
    print(f"📉 Top Bearish: ", end="")
    for r in results:
        if r['consensus'] == '🔴 BEARISH':
            print(f"{r['symbol']} ({r['change']:.1f}%) ", end="")
    print()
