#!/usr/bin/env python3
"""Scan top movers on Binance — best for finding trading opportunities."""
import ccxt
import pandas as pd
import sys

def scan_movers(min_volume=1_000_000, top_n=20):
    exchange = ccxt.binance()
    exchange.load_markets()

    tickers = exchange.fetch_tickers()
    usdt_pairs = [s for s in tickers if s.endswith('/USDT')]

    movers = []
    for symbol in usdt_pairs:
        t = tickers[symbol]
        vol = t.get('quoteVolume', 0) or 0
        change = t.get('percentage', 0) or 0
        if vol >= min_volume:
            movers.append({
                'symbol': symbol,
                'price': t['last'],
                'change_24h': round(change, 2),
                'volume_usdt': int(vol),
                'high_24h': t.get('high', 0),
                'low_24h': t.get('low', 0),
            })

    df = pd.DataFrame(movers)
    df = df.sort_values('change_24h', ascending=False)

    print("📊 Top Gainers (24h):")
    print("-" * 60)
    for _, row in df.head(top_n).iterrows():
        emoji = "🟢" if row['change_24h'] > 0 else "🔴"
        print(f"{emoji} {row['symbol']:<12} ${row['price']:>10,.4f}  {row['change_24h']:>+6.1f}%  Vol: ${row['volume_usdt']:>12,}")

    print(f"\n📉 Top Losers (24h):")
    print("-" * 60)
    for _, row in df.tail(top_n).iloc[::-1].iterrows():
        emoji = "🟢" if row['change_24h'] > 0 else "🔴"
        print(f"{emoji} {row['symbol']:<12} ${row['price']:>10,.4f}  {row['change_24h']:>+6.1f}%  Vol: ${row['volume_usdt']:>12,}")

    # Summary
    print(f"\n📈 Market Summary:")
    avg_change = df['change_24h'].mean()
    btc = df[df['symbol'] == 'BTC/USDT']
    btc_change = btc['change_24h'].values[0] if len(btc) > 0 else 0
    print(f"   BTC: {btc_change:+.1f}% | Avg: {avg_change:+.1f}%")
    print(f"   Pairs scanned: {len(df)} (vol > ${min_volume:,})")

if __name__ == '__main__':
    min_vol = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    scan_movers(min_vol, top_n)
