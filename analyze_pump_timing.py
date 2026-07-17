#!/usr/bin/env python3
"""Analyze: how many days after signal does price spike/pump?"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

CACHE = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'

TP = 2.5; SL = 2.0; PL = 40; TRAIL = 0.10; MAX_H = 2
STR = 50; WHALE_MIN = 0.35; MIN_VOL = 200000

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

def whale_indicator(df):
    df = df.copy()
    LB = 30
    df['lo'] = df['low'].rolling(LB).min()
    df['lc'] = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
    df['sm'] = df['lc'].ewm(span=3, adjust=False).mean()
    df['hi'] = df['sm'].rolling(LB).max()
    df['raw'] = np.where(df['low'] <= df['lo'], (df['sm'] + df['hi'] * 2) / 3, 0)
    df['whale'] = df['raw'].ewm(span=3, adjust=False).mean().fillna(0)
    df['spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.03)
    df['wf'] = df['whale'].rolling(2).mean()
    df['ws'] = df['whale'].rolling(5).mean()
    df['wp'] = df['whale'].rolling(50).max()
    df['str'] = (df['whale'] / df['wp'].replace(0, np.nan) * 100).fillna(0)
    df['vma'] = df['volume'].rolling(20).mean()
    df['entry'] = (df['spike'] & (df['wf'] > df['ws']) &
                   (df['str'] > STR) & (df['volume'] > df['vma'] * 1.0))
    return df

# ── Load signals ──
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

# ── For each signal, find max price in next 14 days (in 15m candles) ──
analysis = []
by_pair = defaultdict(list)
for sig in signals:
    by_pair[(sig['symbol'], sig['month'])].append(sig)

for (sym, mon), sigs in by_pair.items():
    df = load_cached(sym, mon)
    if df is None: continue
    df_w = whale_indicator(df)
    
    for sig in sigs:
        df_w['td'] = abs((df_w['ts'] - sig['dt']).dt.total_seconds())
        nearest = df_w['td'].idxmin()
        
        # Get candle at signal time
        sig_price = df_w.iloc[nearest]['close']
        
        # Look forward: find whale entry and track price after
        forward = df_w.iloc[nearest:].reset_index(drop=True)
        
        # Find whale entry
        wi = None; wv = 0
        for j, row in forward.iterrows():
            if j * 0.25 > 24: break
            if row['entry']: wi = j; wv = float(row['whale']); break
        
        if wi is None or wv < WHALE_MIN: continue
        
        entry_idx = wi
        entry_price = forward.iloc[entry_idx]['close']
        
        # Track max price for next 14 days (14*24*4 = 1344 candles)
        max_fwd = min(len(forward) - 1, entry_idx + 14*24*4)
        
        max_pct = 0; max_hours = 0
        first_2pct_hours = None
        first_5pct_hours = None
        first_10pct_hours = None
        
        for j in range(entry_idx + 1, max_fwd):
            row = forward.iloc[j]
            pct = (row['high'] - entry_price) / entry_price * 100
            hours = (j - entry_idx) * 0.25
            
            if pct > max_pct:
                max_pct = pct; max_hours = hours
            
            if first_2pct_hours is None and row['high'] >= entry_price * 1.02:
                first_2pct_hours = hours
            if first_5pct_hours is None and row['high'] >= entry_price * 1.05:
                first_5pct_hours = hours
            if first_10pct_hours is None and row['high'] >= entry_price * 1.10:
                first_10pct_hours = hours
        
        # Also check TP/SL outcome
        tp_price = entry_price * (1 + TP/100)
        sl_price = entry_price * (1 - SL/100)
        pl_price = entry_price + (tp_price - entry_price) * (PL/100)
        
        outcome = 'TIMEOUT'
        outcome_pnl = 0
        pl_triggered = False; peak = entry_price; trail = 0
        
        for j in range(entry_idx + 1, min(len(forward), entry_idx + 8 + 1)):
            row = forward.iloc[j]
            hours = (j - entry_idx) * 0.25
            if hours > MAX_H:
                outcome = 'TIMEOUT'; outcome_pnl = round((row['close'] - entry_price)/entry_price*100, 4); break
            
            if not pl_triggered and row['high'] >= pl_price:
                pl_triggered = True; peak = row['high']; trail = row['high'] * (1 - TRAIL/100)
            if pl_triggered:
                if row['high'] > peak: peak = row['high']; trail = row['high'] * (1 - TRAIL/100)
                if row['low'] <= trail:
                    outcome = 'TRAIL'; outcome_pnl = round((trail - entry_price)/entry_price*100, 4); break
            if row['high'] >= tp_price:
                outcome = 'TP'; outcome_pnl = round(TP, 4); break
            if row['low'] <= sl_price:
                outcome = 'SL'; outcome_pnl = round(-SL, 4); break
        
        analysis.append({
            'symbol': sym, 'dt': sig['dt'],
            'entry_price': entry_price,
            'max_pct': round(max_pct, 2),
            'max_hours': round(max_hours, 1),
            'max_days': round(max_hours / 24, 1),
            'first_2pct_h': first_2pct_hours,
            'first_5pct_h': first_5pct_hours,
            'first_10pct_h': first_10pct_hours,
            'outcome': outcome, 'outcome_pnl': outcome_pnl
        })

print(f'Trades analyzed: {len(analysis)}')

# ── Distribution analysis ──
# Max price timing
print(f'\n{"="*55}')
print(f'📊 متى يتحقق أعلى سعر بعد الدخول؟')
print(f'{"="*55}')

max_pct_bins = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 100)]
for lo, hi in max_pct_bins:
    items = [a for a in analysis if lo <= a['max_pct'] < hi]
    if not items: continue
    avg_h = np.mean([a['max_hours'] for a in items])
    med_h = np.median([a['max_hours'] for a in items])
    print(f'  قمة {lo}-{hi}%: {len(items)} صفقة | متوسط {avg_h:.0f}س | وسيط {med_h:.0f}س')

# Max by days
print(f'\n📅 توزيع القمة حسب الأيام:')
day_bins = [(0, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 7), (7, 14)]
for lo, hi in day_bins:
    items = [a for a in analysis if lo <= a['max_days'] < hi]
    if not items: continue
    bar = '█' * int(len(items) / len(analysis) * 50)
    print(f'  {lo:.1f}-{hi:.1f} يوم: {len(items):>4} صفقة ({len(items)/len(analysis)*100:.0f}%) {bar}')

# First time to 2%
print(f'\n🎯 أول مرة يوصل +2%:')
items_2 = [a for a in analysis if a['first_2pct_h'] is not None]
if items_2:
    avg = np.mean([a['first_2pct_h'] for a in items_2])
    med = np.median([a['first_2pct_h'] for a in items_2])
    print(f'  {len(items_2)}/{len(analysis)} صفقة ({len(items_2)/len(analysis)*100:.0f}%)')
    print(f'  متوسط: {avg:.1f}س | وسيط: {med:.1f}س')
else:
    print(f'  لا يوجد')

# First to 5%
items_5 = [a for a in analysis if a['first_5pct_h'] is not None]
if items_5:
    avg = np.mean([a['first_5pct_h'] for a in items_5])
    med = np.median([a['first_5pct_h'] for a in items_5])
    print(f'\n🎯 أول مرة يوصل +5%:')
    print(f'  {len(items_5)}/{len(analysis)} صفقة ({len(items_5)/len(analysis)*100:.0f}%)')
    print(f'  متوسط: {avg:.1f}س | وسيط: {med:.1f}س')

# First to 10%
items_10 = [a for a in analysis if a['first_10pct_h'] is not None]
if items_10:
    avg = np.mean([a['first_10pct_h'] for a in items_10])
    med = np.median([a['first_10pct_h'] for a in items_10])
    print(f'\n🎯 أول مرة يوصل +10%:')
    print(f'  {len(items_10)}/{len(analysis)} صفقة ({len(items_10)/len(analysis)*100:.0f}%)')
    print(f'  متوسط: {avg:.1f}س | وسيط: {med:.1f}س')

# By outcome
print(f'\n📊 حسب نتيجة الصفقة:')
for out in ['TP', 'TRAIL', 'SL', 'TIMEOUT']:
    items = [a for a in analysis if a['outcome'] == out]
    if not items: continue
    avg_max = np.mean([a['max_pct'] for a in items])
    avg_h = np.mean([a['max_hours'] for a in items])
    print(f'  {out}: {len(items)} صفقة | أقصى ارتفاع متوسط: {avg_max:.1f}% | وقت القمة: {avg_h:.0f}س')

# Top performers
print(f'\n🏆 أفضل 10 صفقات (أعلى ارتفاع):')
top = sorted(analysis, key=lambda a: -a['max_pct'])[:10]
for a in top:
    print(f'  {a["symbol"]:<10} | +{a["max_pct"]:.1f}% | بعد {a["max_days"]:.1f} يوم | نتيجة: {a["outcome"]} {a["outcome_pnl"]:+.1f}% | {a["dt"]}')
