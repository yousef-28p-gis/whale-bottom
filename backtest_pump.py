#!/usr/bin/env python3
"""
Backtest Pump Detector signals against Hunter Whale strategy.
Filters: LONG only, non-stable, volume >= 200K, whale confirmed.
"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from collections import defaultdict

CACHE = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_pumpdetector.json'
THIS_MONTH = '2026-07'

# Strategy params (same as hunter_live.py)
TP = 2.5; SL = 2.0; PL = 40; TRAIL = 0.10; MAX_H = 2; STR = 50; WHALE_MIN = 0.35
COMMISSION = 0.20  # 0.1% buy + 0.1% sell

STABLES = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDE', 'XUSD',
    'BFUSD', 'FDUSD', 'USDD', 'FRAX', 'LUSD', 'PYUSD',
    'USDJ', 'RLUSD', 'XAUT', 'USD1', 'EUR'
}

def load_cached(sym, mon):
    fpath = f'{CACHE}/{sym}_{mon}.json'
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
        return df.sort_values('ts').reset_index(drop=True)
    except:
        return None

def whale_indicator(df):
    df = df.copy()
    LB, WF, WS, VM = 30, 2, 5, 1.0
    df['lo'] = df['low'].rolling(LB).min()
    df['lc'] = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
    df['sm'] = df['lc'].ewm(span=3, adjust=False).mean()
    df['hi'] = df['sm'].rolling(LB).max()
    df['raw'] = np.where(df['low'] <= df['lo'], (df['sm'] + df['hi'] * 2) / 3, 0)
    df['whale'] = df['raw'].ewm(span=3, adjust=False).mean().fillna(0)
    df['spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.03)
    df['wf'] = df['whale'].rolling(WF).mean()
    df['ws'] = df['whale'].rolling(WS).mean()
    df['wp'] = df['whale'].rolling(50).max()
    df['str'] = (df['whale'] / df['wp'].replace(0, np.nan) * 100).fillna(0)
    df['vma'] = df['volume'].rolling(20).mean()
    df['entry'] = (df['spike'] & (df['wf'] > df['ws']) &
                   (df['str'] > STR) & (df['volume'] > df['vma'] * VM))
    return df

def simulate_position(df, entry_idx, entry_price):
    """Simulate TP/SL/PL+Trail/Time exit and return (exit_pnl, exit_status, exit_detail)."""
    tp_price = entry_price * (1 + TP/100)
    sl_price = entry_price * (1 - SL/100)
    pl_price = entry_price * (1 + PL/100)
    
    pl_triggered = False
    peak = entry_price
    trail_price = None
    
    max_idx = min(entry_idx + 8, len(df))  # 8 candles = 2h (15m each)
    
    for i in range(entry_idx + 1, max_idx):
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']
        close = df.iloc[i]['close']
        
        # Check TP
        if high >= tp_price:
            return (TP, '🎯 هدف', f'وصل الهدف +{TP}% | سعر {tp_price}')
        
        # Check SL
        if low <= sl_price:
            return (-SL, '🛑 ستوب', f'ضرب الستوب -{SL}% | سعر {sl_price}')
        
        # PL + Trail
        if pl_triggered:
            if high > peak:
                peak = high
                trail_price = peak * (1 - TRAIL/100)
            if trail_price and low <= trail_price:
                trail_pnl = round((trail_price - entry_price) / entry_price * 100, 4)
                return (trail_pnl, '🐌 تريل', f'ارتد من القمة | إغلاق تريل {close}')
        else:
            if high >= pl_price:
                pl_triggered = True
                peak = high
                trail_price = peak * (1 - TRAIL/100)
    
    # Time exit
    last_close = df.iloc[max_idx - 1]['close']
    pnl = round((last_close - entry_price) / entry_price * 100, 4)
    return (pnl, '⏰ وقت', f'انتهت المدة ({MAX_H}h) | إغلاق بسعر {last_close}')

# ── Main ──
with open(SIGNALS_FILE) as f:
    all_signals = json.load(f)

print(f'Total signals: {len(all_signals)}')

# Filter: non-stable, volume >= 200K
MIN_VOL = 200_000
filtered = []
for s in all_signals:
    if s['symbol'] in STABLES:
        continue
    if s['volume_usdt'] < MIN_VOL:
        continue
    filtered.append(s)

print(f'After volume/stable filter: {len(filtered)}')

# Group signals needing cache check
symbols_needed = set()
cache_hits = set()
cache_misses = set()

for s in filtered:
    dt = datetime.fromisoformat(s['dt'])
    mon = dt.strftime('%Y-%m')
    sym = s['symbol']
    key = (sym, mon)
    symbols_needed.add(key)
    if load_cached(sym, mon) is not None:
        cache_hits.add(key)
    else:
        cache_misses.add(key)

print(f'Unique symbol-months needed: {len(symbols_needed)}')
print(f'Cache hits: {len(cache_hits)}')
print(f'Cache misses: {len(cache_misses)}')

# Process signals
results = {
    'total_filtered': len(filtered),
    'cache_miss': 0,
    'whale_rejected': 0,
    'trades': [],
    'no_data': [],
}

for i, s in enumerate(filtered):
    dt = datetime.fromisoformat(s['dt'])
    mon = dt.strftime('%Y-%m')
    sym = s['symbol']
    
    df = load_cached(sym, mon)
    if df is None:
        results['cache_miss'] += 1
        results['no_data'].append({'symbol': sym, 'dt': s['dt'], 'reason': 'no_cache'})
        continue
    
    # Find closest candle BEFORE signal time
    df_w = whale_indicator(df)
    # Convert signal time to UTC, strip tzinfo to match cache tz-naive timestamps
    signal_ts = pd.Timestamp(dt).tz_convert('UTC').tz_localize(None)
    # Find candles up to signal time
    prior = df_w[df_w['ts'] <= signal_ts]
    if len(prior) < 50:
        results['no_data'].append({'symbol': sym, 'dt': s['dt'], 'reason': 'insufficient_data'})
        continue
    
    # Check entry at the candle just before signal
    last_idx = prior.index[-1]
    last_row = df_w.iloc[last_idx]
    
    if not last_row['entry']:
        results['whale_rejected'] += 1
        continue
    
    entry_price = last_row['close']
    whale_val = last_row['whale']
    whale_str = last_row['str']
    
    # Simulate
    exit_pnl, exit_status, exit_detail = simulate_position(df_w, last_idx, entry_price)
    net_pnl = round(exit_pnl - COMMISSION, 4)
    
    results['trades'].append({
        'symbol': sym,
        'dt': s['dt'],
        'entry_price': round(entry_price, 8),
        'whale_val': round(float(whale_val), 4),
        'whale_str': round(float(whale_str), 1),
        'exit_pnl': exit_pnl,
        'exit_net': net_pnl,
        'exit_status': exit_status,
        'exit_detail': exit_detail,
        'volume_usdt': s['volume_usdt']
    })
    
    if (i + 1) % 500 == 0:
        print(f'  Processed {i+1}/{len(filtered)}... {len(results["trades"])} trades so far')

# ── Summary ──
trades = results['trades']
print(f'\n{"="*60}')
print(f'📊 نتائج باك تيست — Pump Detector Signals')
print(f'{"="*60}')
print(f'إجمالي الإشارات: {len(all_signals):,}')
print(f'بعد فلترة الحجم/ستيبل: {len(filtered):,}')
print(f'بدون كاش: {results["cache_miss"]}')
print(f'مرفوضة من الحوت: {results["whale_rejected"]}')
print(f'صفقات منفذة: {len(trades)}')
print()

if trades:
    wins = sum(1 for t in trades if t['exit_net'] > 0)
    losses = sum(1 for t in trades if t['exit_net'] <= 0)
    wr = round(wins / len(trades) * 100, 1)
    total_net = sum(t['exit_net'] for t in trades)
    total_gross = sum(t['exit_pnl'] for t in trades)
    avg_net = round(total_net / len(trades), 2)
    
    # Status breakdown
    status_count = defaultdict(int)
    status_pnl = defaultdict(float)
    for t in trades:
        status_count[t['exit_status']] += 1
        status_pnl[t['exit_status']] += t['exit_net']
    
    print(f'📈 ملخص:')
    print(f'  الرابحة: {wins} 🟢 | الخاسرة: {losses} 🔴 | النسبة: {wr}%')
    print(f'  الإجمالي الخام: {total_gross:+.2f}% | الصافي (بعد العمولة): {total_net:+.2f}%')
    print(f'  متوسط الصفقة: {avg_net:+.2f}%')
    print()
    print(f'📋 تفصيل الإغلاقات:')
    for status, count in sorted(status_count.items(), key=lambda x: -x[1]):
        avg = round(status_pnl[status] / count, 2)
        print(f'  {status}: {count} صفقة | المجموع: {status_pnl[status]:+.2f}% | المتوسط: {avg:+.2f}%')
    
    # Worst losers
    print(f'\n🔴 أسوأ 5 صفقات:')
    worst = sorted(trades, key=lambda t: t['exit_net'])[:5]
    for t in worst:
        print(f'  {t["symbol"]:<10} | {t["exit_net"]:+.2f}% | {t["exit_status"]} | {t["dt"][:10]}')
    
    # Best winners
    print(f'\n🟢 أفضل 5 صفقات:')
    best = sorted(trades, key=lambda t: -t['exit_net'])[:5]
    for t in best:
        print(f'  {t["symbol"]:<10} | {t["exit_net"]:+.2f}% | {t["exit_status"]} | {t["dt"][:10]}')

# Save results
out_path = '/data/trading28/backtest_pumpdetector.json'
with open(out_path, 'w') as f:
    json.dump(results, f, default=str, indent=2)
print(f'\n✅ النتائج محفوظة: {out_path}')
