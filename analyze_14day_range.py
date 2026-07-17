#!/usr/bin/env python3
"""Max rise & max drop within 14 days of Whale Sniper signal (Apr-Jun 2026)."""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

CACHE = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'
DAYS = 14; MAX_CANDLES = DAYS * 24 * 4  # 1344 candles

MIN_VOL = 200000
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
BLOCKED = {'SUPER','ORCA','VANA','W','DOGS','MET','XLM','BB','COS','LUNA','S'}

def load_cached(sym, mon):
    fpath = f'{CACHE}/{sym}_{mon}.json'
    if not os.path.exists(fpath): return None
    with open(fpath) as f: data = json.load(f)
    df = pd.DataFrame(data)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    return df.sort_values('ts').reset_index(drop=True)

with open(SIGNALS_FILE) as f:
    raw = json.load(f)

signals = []
for s in raw:
    if s['symbol'] in STABLES or s['symbol'] in BLOCKED: continue
    if s.get('direction', 'LONG') != 'LONG': continue
    if s.get('volume_usdt', 0) < MIN_VOL: continue
    dt = datetime.fromisoformat(s['dt'])
    if dt.month not in (4, 5, 6) or dt.year != 2026: continue
    signals.append({'symbol': s['symbol'], 'dt': dt, 'month': dt.strftime('%Y-%m')})

print(f'Signals: {len(signals)}')

by_pair = defaultdict(list)
for sig in signals:
    by_pair[(sig['symbol'], sig['month'])].append(sig)

results = []
no_data = 0

for (sym, mon), sigs in by_pair.items():
    df = load_cached(sym, mon)
    if df is None: no_data += len(sigs); continue
    
    for sig in sigs:
        sig_ts = sig['dt'].replace(tzinfo=None)
        df['td'] = abs((df['ts'] - sig_ts).dt.total_seconds())
        nearest = df['td'].idxmin()
        
        sig_price = df.iloc[nearest]['close']
        
        # Look forward 14 days
        max_idx = min(len(df), nearest + MAX_CANDLES)
        if max_idx <= nearest + 1:
            no_data += 1; continue
        
        forward = df.iloc[nearest + 1:max_idx]
        if len(forward) == 0:
            no_data += 1; continue
        
        max_price = forward['high'].max()
        min_price = forward['low'].min()
        max_pct = round((max_price - sig_price) / sig_price * 100, 2)
        min_pct = round((min_price - sig_price) / sig_price * 100, 2)
        
        # When did max/min occur?
        max_idx_rel = forward['high'].idxmax()
        min_idx_rel = forward['low'].idxmin()
        max_hours = round((max_idx_rel - nearest) * 0.25, 1)
        min_hours = round((min_idx_rel - nearest) * 0.25, 1)
        
        results.append({
            'symbol': sym, 'dt': sig['dt'],
            'sig_price': sig_price,
            'max_pct': max_pct, 'max_hours': max_hours, 'max_days': round(max_hours/24, 1),
            'min_pct': min_pct, 'min_hours': min_hours, 'min_days': round(min_hours/24, 1),
        })

print(f'Analyzed: {len(results)} signals ({no_data} no data)')

# ── Stats ──
max_vals = [r['max_pct'] for r in results]
min_vals = [r['min_pct'] for r in results]

print(f'\n{"="*55}')
print(f'📊 خلال {DAYS} يوم من نزول التوصية ({len(results)} إشارة)')
print(f'{"="*55}')

print(f'\n🟢 أعلى نسبة صعود:')
print(f'  متوسط: +{np.mean(max_vals):.1f}%')
print(f'  وسيط: +{np.median(max_vals):.1f}%')
print(f'  أقصى: +{max(max_vals):.1f}%')
print(f'  نسبة >0%: {sum(1 for v in max_vals if v > 0)/len(max_vals)*100:.0f}%')

print(f'\n🔴 أكبر نسبة هبوط:')
print(f'  متوسط: {np.mean(min_vals):.1f}%')
print(f'  وسيط: {np.median(min_vals):.1f}%')
print(f'  أقصى: {min(min_vals):.1f}%')

# Distribution
print(f'\n📈 توزيع أعلى صعود:')
bins = [(-100, 0), (0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1000)]
for lo, hi in bins:
    items = [r for r in results if lo <= r['max_pct'] < hi]
    if not items: continue
    bar = '█' * int(len(items)/len(results)*50)
    label = f'{lo:+d}%' if lo > 0 else f'{lo}%'
    print(f'  {label} — {hi:+d}%: {len(items):>4} ({len(items)/len(results)*100:.0f}%) {bar}')

print(f'\n📉 توزيع أكبر هبوط:')
bins_drop = [(-100, -50), (-50, -30), (-30, -20), (-20, -10), (-10, -5), (-5, 0), (0, 100)]
for lo, hi in bins_drop:
    items = [r for r in results if lo <= r['min_pct'] < hi]
    if not items: continue
    bar = '█' * int(len(items)/len(results)*50)
    print(f'  {lo:+d}% — {hi:+d}%: {len(items):>4} ({len(items)/len(results)*100:.0f}%) {bar}')

# Timing
print(f'\n⏱️ متى تتحقق أعلى قمة؟')
day_bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 14)]
for lo, hi in day_bins:
    items = [r for r in results if lo <= r['max_days'] < hi]
    if not items: continue
    bar = '█' * int(len(items)/len(results)*50)
    print(f'  {lo:.1f}-{hi:.0f} يوم: {len(items):>4} ({len(items)/len(results)*100:.0f}%) {bar}')

# Net opportunity (max rise - abs(max drop))
print(f'\n💡 الفرصة الصافية (أعلى صعود - |أكبر هبوط|):')
net_opp = [r['max_pct'] - abs(r['min_pct']) for r in results]
pos_opp = sum(1 for v in net_opp if v > 0)
print(f'  إيجابية (الصعود > الهبوط): {pos_opp}/{len(results)} ({pos_opp/len(results)*100:.0f}%)')
print(f'  متوسط: {np.mean(net_opp):+.1f}%')
print(f'  وسيط: {np.median(net_opp):+.1f}%')
