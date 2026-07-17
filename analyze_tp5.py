#!/usr/bin/env python3
"""If TP=5%: when do trades hit it? What's the optimal SL?"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

CACHE = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'
DAYS = 14; MAX_CANDLES = DAYS * 24 * 4

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
        target = sig_price * 1.05  # +5%
        
        max_idx = min(len(df), nearest + MAX_CANDLES)
        if max_idx <= nearest + 1: no_data += 1; continue
        
        # Track: first time to 5%, and max drawdown before that
        hit_5pct = False
        hit_hours = None
        max_dd_before = 0  # most negative % before reaching 5%
        peak_so_far = sig_price
        
        for j in range(nearest + 1, max_idx):
            row = df.iloc[j]
            hours = (j - nearest) * 0.25
            
            # Track max drawdown from signal price
            dd = (row['low'] - sig_price) / sig_price * 100
            if dd < max_dd_before:
                max_dd_before = dd
            
            # Check if hit 5%
            if row['high'] >= target:
                hit_5pct = True
                hit_hours = hours
                break
        
        if hit_5pct:
            results.append({
                'symbol': sym, 'dt': sig['dt'],
                'hours': hit_hours,
                'days': round(hit_hours / 24, 1),
                'max_dd': round(max_dd_before, 1),
            })

print(f'Hit +5% within 14d: {len(results)}/{len(signals)-no_data} ({len(results)/(len(signals)-no_data)*100:.0f}%)')

if not results: exit()

# ── Time to 5% ──
print(f'\n{"="*55}')
print(f'⏱️ متى يتحقق هدف +5%؟')
print(f'{"="*55}')
print(f'  متوسط: {np.mean([r["hours"] for r in results]):.1f} ساعة')
print(f'  وسيط: {np.median([r["hours"] for r in results]):.1f} ساعة')

print(f'\n📅 توزيع الوقت:')
day_bins = [(0, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 14)]
cum = 0
for lo, hi in day_bins:
    items = [r for r in results if lo <= r['days'] < hi]
    cum += len(items)
    if not items: continue
    bar = '█' * int(len(items)/len(results)*50)
    print(f'  {lo*24:.0f}س-{hi*24:.0f}س ({lo:.1f}-{hi:.0f}يوم): {len(items):>4} ({len(items)/len(results)*100:.0f}%) تراكمي {cum/len(results)*100:.0f}% {bar}')

# ── Max DD before 5% → Optimal SL ──
print(f'\n{"="*55}')
print(f'📉 أقصى هبوط قبل تحقيق +5% → أنسب ستوب')
print(f'{"="*55}')

dd_vals = [r['max_dd'] for r in results]
print(f'  متوسط: {np.mean(dd_vals):.1f}%')
print(f'  وسيط: {np.median(dd_vals):.1f}%')

# Test different SL levels
print(f'\n🧪 اختبار مستويات ستوب مختلفة:')
for sl in [1, 1.5, 2, 2.5, 3, 4, 5, 7, 10]:
    killed = sum(1 for dd in dd_vals if dd <= -sl)
    survived = len(results) - killed
    wr = survived / len(results) * 100
    bar = '█' * int(wr/2)
    print(f'  SL={sl}%: ينجو {survived}/{len(results)} ({wr:.0f}%) {bar}')
    if wr >= 85:
        print(f'         ⬆ أنسب ستوب لـ Win Rate ~{wr:.0f}%')

# ── SL distribution ──
print(f'\n📊 توزيع أقصى هبوط قبل الهدف:')
dd_bins = [(0, -1), (-1, -2), (-2, -3), (-3, -5), (-5, -8), (-8, -10), (-10, -15), (-15, -100)]
for lo, hi in dd_bins:
    items = [r for r in results if lo >= r['max_dd'] > hi]
    bar = '█' * int(len(items)/len(results)*50)
    print(f'  {lo:+d}% — {hi:+d}%: {len(items):>4} ({len(items)/len(results)*100:.0f}%) {bar}')

# ── By hour bucket ──
print(f'\n📊 حسب سرعة تحقيق الهدف:')
for lo, hi in [(0, 1), (1, 4), (4, 12), (12, 24), (24, 72), (72, 168), (168, 336)]:
    items = [r for r in results if lo <= r['hours'] < hi]
    if not items: continue
    avg_dd = np.mean([r['max_dd'] for r in items])
    med_dd = np.median([r['max_dd'] for r in items])
    print(f'  {lo}-{hi}س: {len(items)} صفقة | متوسط هبوط: {avg_dd:.1f}% | وسيط: {med_dd:.1f}%')
