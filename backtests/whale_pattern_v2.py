#!/usr/bin/env python3
"""
WHALE PATTERN STRATEGY v2 — Tighter filters
Based on pattern discovery insights:
- Multiple whale bars matter more than single
- Lower RSI = better
- Previous day red = strong signal
"""
import ccxt
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
os.makedirs(DATA_DIR, exist_ok=True)

BACKTEST_DAYS = 60
TIMEFRAME = '1h'
LOOKBACK_CANDLES = 500

# ── STRATEGY PARAMETERS ─────────────────────────────────
WHALE_STD = 2.0           # whale = volume > mean + N*std
MIN_WHALE_IN_24H = 2      # at least N whale bars in last 24h
RSI_MAX = 40              # max RSI at entry
PRICE_POS_MAX = 0.35      # price must be in bottom 35% of 48h range
REQUIRE_PREV_CANDLE_RED = True  # candle just before whale must be red

TP_PCT = 0.07
SL_PCT = 0.03
MAX_HOLD_HOURS = 48
COMMISSION = 0.002
INITIAL_CAPITAL = 1000

# ── Load coins ──────────────────────────────────────────
with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
coins_raw = config['halal'] + config['halal2']
seen = set()
coins_raw = [c for c in coins_raw if not (c in seen or seen.add(c))]
blacklist = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','ROBO','PYTH','ANKR'}
coins = [c for c in coins_raw if c not in blacklist]

print(f"🎯 Whale Pattern v2 — Tighter Filters")
print(f"   Whale≥{MIN_WHALE_IN_24H} in 24h | RSI<{RSI_MAX} | Pos<{PRICE_POS_MAX} | PrevRed={REQUIRE_PREV_CANDLE_RED}")
print(f"   TP:+{TP_PCT*100:.0f}% SL:-{SL_PCT*100:.0f}%")

# ── Exchange ────────────────────────────────────────────
_EXCHANGE = None
def get_exchange():
    global _EXCHANGE
    if _EXCHANGE is None:
        _EXCHANGE = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})
    return _EXCHANGE

# ── Fetch ───────────────────────────────────────────────
def fetch_data():
    cache_file = os.path.join(DATA_DIR, 'hourly_60d.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    
    exchange = get_exchange()
    since = exchange.parse8601((datetime.now() - timedelta(days=BACKTEST_DAYS)).isoformat())
    all_data = {}
    
    for i, coin in enumerate(coins):
        symbol = f"{coin}/USDT"
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=LOOKBACK_CANDLES)
            if len(ohlcv) >= 100:
                all_data[coin] = {
                    'ts': [int(o[0]) for o in ohlcv],
                    'open': [float(o[1]) for o in ohlcv],
                    'high': [float(o[2]) for o in ohlcv],
                    'low': [float(o[3]) for o in ohlcv],
                    'close': [float(o[4]) for o in ohlcv],
                    'volume': [float(o[5]) for o in ohlcv],
                }
            if (i+1) % 20 == 0:
                print(f"  📊 {i+1}/{len(coins)}")
        except:
            pass
        time.sleep(0.05)
    
    with open(cache_file, 'w') as f:
        json.dump(all_data, f)
    print(f"✅ {len(all_data)} coins")
    return all_data

# ── Backtest ────────────────────────────────────────────
def backtest(all_data):
    all_signals = []
    
    for coin, data in all_data.items():
        close = np.array(data['close'])
        volume = np.array(data['volume'])
        ts = data['ts']
        n = len(close)
        
        if n < 100:
            continue
        
        # Rolling stats for whale detection
        vol_mean = pd.Series(volume).rolling(24).mean().values
        vol_std = pd.Series(volume).rolling(24).std().values
        
        # RSI(14)
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = (100 - (100 / (1 + rs))).values
        
        # Price position (48h range)
        price_pos = np.full(n, 0.5)
        for i in range(48, n):
            rng = max(close[i-48:i]) - min(close[i-48:i])
            if rng > 0:
                price_pos[i] = (close[i] - min(close[i-48:i])) / rng
        
        # Detect whale bars and check filters
        for i in range(48, n - 1):  # -1: enter on next candle
            # Whale check
            if vol_std[i] <= 0 or np.isnan(vol_std[i]):
                continue
            
            is_whale = volume[i] > vol_mean[i] + WHALE_STD * vol_std[i]
            if not is_whale:
                continue
            
            # Count whales in last 24h (including this one)
            whale_count = 0
            for j in range(max(0, i-23), i+1):
                if vol_std[j] > 0 and volume[j] > vol_mean[j] + WHALE_STD * vol_std[j]:
                    whale_count += 1
            
            if whale_count < MIN_WHALE_IN_24H:
                continue
            
            # RSI filter
            if np.isnan(rsi[i]) or rsi[i] > RSI_MAX:
                continue
            
            # Price position filter
            if price_pos[i] > PRICE_POS_MAX:
                continue
            
            # Previous candle must be red
            if REQUIRE_PREV_CANDLE_RED and close[i] >= close[i-1]:
                continue
            
            # Entry on next candle
            entry_idx = i + 1
            if entry_idx >= n:
                continue
            
            all_signals.append({
                'coin': coin,
                'idx': entry_idx,
                'entry_price': close[entry_idx],
                'ts': ts[entry_idx],
                'rsi': float(rsi[i]),
                'whale_count': whale_count,
                'price_pos': float(price_pos[i]),
            })
    
    print(f"\n🎯 Signals: {len(all_signals)}")
    all_signals.sort(key=lambda s: s['ts'])
    
    # Process trades
    trades = []
    capital = INITIAL_CAPITAL
    active = {}  # coin -> exit_idx
    
    for sig in all_signals:
        coin = sig['coin']
        entry_idx = sig['idx']
        
        if coin in active and active[coin] > entry_idx:
            continue
        
        data = all_data[coin]
        close_arr = np.array(data['close'])
        high_arr = np.array(data['high'])
        low_arr = np.array(data['low'])
        ts_arr = data['ts']
        n = len(close_arr)
        
        if entry_idx >= n - 1:
            continue
        
        tp_price = sig['entry_price'] * (1 + TP_PCT)
        sl_price = sig['entry_price'] * (1 - SL_PCT)
        
        exit_price = None
        exit_type = None
        exit_idx = None
        
        for j in range(entry_idx + 1, min(entry_idx + MAX_HOLD_HOURS, n)):
            if low_arr[j] <= sl_price:
                exit_price = sl_price
                exit_type = 'SL'
                exit_idx = j
                break
            elif high_arr[j] >= tp_price:
                exit_price = tp_price
                exit_type = 'TP'
                exit_idx = j
                break
        
        if exit_price is None:
            end = min(entry_idx + MAX_HOLD_HOURS, n - 1)
            exit_price = close_arr[end]
            exit_type = 'TIME' if end == entry_idx + MAX_HOLD_HOURS else 'EOD'
            exit_idx = end
        
        pnl_pct = (exit_price / sig['entry_price'] - 1) * 100 - COMMISSION * 100
        size = capital * 0.10
        pnl_usd = size * pnl_pct / 100
        capital += pnl_usd
        
        trades.append({
            'coin': coin,
            'entry': round(sig['entry_price'], 8),
            'exit': round(exit_price, 8),
            'entry_time': datetime.fromtimestamp(sig['ts']/1000).strftime('%Y-%m-%d %H:%M'),
            'type': exit_type,
            'pnl_pct': round(pnl_pct, 2),
            'rsi': sig['rsi'],
            'whales': sig['whale_count'],
            'capital': round(capital, 2),
            'hold_h': exit_idx - entry_idx,
        })
        
        active[coin] = exit_idx
        # Clean
        active = {k: v for k, v in active.items() if v > entry_idx}
    
    return trades, capital

# ── Report ──────────────────────────────────────────────
def report(trades, final_capital):
    if not trades:
        print("❌ No trades")
        return
    
    df = pd.DataFrame(trades)
    wins = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]
    wr = len(wins) / len(df) * 100
    
    # DD
    eq = [INITIAL_CAPITAL]
    for t in trades:
        eq.append(t['capital'])
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()
    
    total_ret = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    print(f"\n{'='*60}")
    print(f"📊 BACKTEST RESULTS (v2 — tight filters)")
    print(f"{'='*60}")
    print(f"   Trades:     {len(df)}")
    print(f"   Win Rate:   {wr:.1f}%")
    print(f"   Wins:       {len(wins)} | Losses: {len(losses)}")
    if len(wins) > 0:
        print(f"   Avg Win:    +{wins['pnl_pct'].mean():.2f}%")
    if len(losses) > 0:
        print(f"   Avg Loss:   {losses['pnl_pct'].mean():.2f}%")
    print(f"   Avg Trade:  {df['pnl_pct'].mean():.2f}%")
    if len(losses) > 0 and losses['pnl_usd'].sum() != 0:
        pf = abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum()) if 'pnl_usd' in df.columns else 0
        print(f"   PF:         {pf:.2f}")
    print(f"   Return:     {total_ret:+.1f}%")
    print(f"   Final:      ${final_capital:,.2f}")
    print(f"   Max DD:     {max_dd:.2f}%")
    if 'hold_h' in df.columns:
        print(f"   Avg Hold:   {df['hold_h'].mean():.1f}h")
    
    print(f"\n## By Exit:")
    for et in ['TP', 'SL', 'TIME', 'EOD']:
        s = df[df['type'] == et]
        if len(s) > 0:
            print(f"   {et:5s}: {len(s)} trades, avg {s['pnl_pct'].mean():+.2f}%")
    
    print(f"\n## By Whale Count:")
    for wc in sorted(df['whales'].unique()):
        s = df[df['whales'] == wc]
        swr = len(s[s['pnl_pct'] > 0]) / len(s) * 100
        print(f"   {int(wc)} whales: {len(s)} trades, WR {swr:.0f}%, avg {s['pnl_pct'].mean():+.2f}%")
    
    print(f"\n## By RSI:")
    for r_range, (lo, hi) in [('<20', (0,20)), ('20-30', (20,30)), ('30-40', (30,40))]:
        s = df[(df['rsi'] >= lo) & (df['rsi'] < hi)]
        if len(s) > 0:
            swr = len(s[s['pnl_pct'] > 0]) / len(s) * 100
            print(f"   RSI {r_range}: {len(s)} trades, WR {swr:.0f}%, avg {s['pnl_pct'].mean():+.2f}%")
    
    print(f"\n## Top Coins:")
    cs = df.groupby('coin').agg(n=('pnl_pct','count'), avg=('pnl_pct','mean'), total=('pnl_pct','sum')).sort_values('total', ascending=False)
    for coin, row in cs.head(10).iterrows():
        print(f"   {coin:8s}: {int(row['n'])} trades, avg {row['avg']:+.2f}%, total {row['total']:+.1f}%")

# ── Main ─────────────────────────────────────────────────
print("=" * 60)
print("🐋 WHALE PATTERN v2 — Tight Filters")
print("=" * 60)

all_data = fetch_data()
trades, final_cap = backtest(all_data)
report(trades, final_cap)
print("\n✅ Done!")
