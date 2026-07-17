#!/usr/bin/env python3
"""Test TP=15%, SL=10% from signal price within 14 days (Apr-Jun 2026)."""
import json, os, numpy as np, pandas as pd
from datetime import datetime
from collections import defaultdict

CACHE = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'
DAYS = 14; MAX_CANDLES = DAYS * 24 * 4
TP = 15; SL = 10

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
        
        tp_price = sig_price * (1 + TP/100)
        sl_price = sig_price * (1 - SL/100)
        
        max_idx = min(len(df), nearest + MAX_CANDLES)
        if max_idx <= nearest + 1: no_data += 1; continue
        
        tp_hit = False; tp_hours = None
        sl_hit = False; sl_hours = None
        
        for j in range(nearest + 1, max_idx):
            row = df.iloc[j]
            hours = (j - nearest) * 0.25
            
            if row['low'] <= sl_price:
                sl_hit = True; sl_hours = hours; break
            if row['high'] >= tp_price:
                tp_hit = True; tp_hours = hours; break
        
        results.append({
            'symbol': sym, 'dt': sig['dt'],
            'sig_price': sig_price,
            'tp_hit': tp_hit, 'tp_hours': tp_hours,
            'sl_hit': sl_hit, 'sl_hours': sl_hours,
        })

total = len(results)
tp_wins = [r for r in results if r['tp_hit'] and not r['sl_hit']]
sl_loss = [r for r in results if r['sl_hit'] and not r['tp_hit']]
both = [r for r in results if r['tp_hit'] and r['sl_hit']]  # hit both in same candle? shouldn't happen
neither = [r for r in results if not r['tp_hit'] and not r['sl_hit']]

# For 'both' cases, check which happened first
tp_first = [r for r in both if r['tp_hours'] < r['sl_hours']]
sl_first = [r for r in both if r['sl_hours'] >= r['tp_hours']]

real_wins = len(tp_wins) + len(tp_first)
real_losses = len(sl_loss) + len(sl_first)
wr = real_wins / total * 100

print(f'Analyzed: {total} ({no_data} no data)')
print(f'\n{"="*55}')
print(f'🎯 TP={TP}% | 🛑 SL={SL}% | ⏱️ {DAYS} يوم')
print(f'{"="*55}')
print(f'  TP فقط (قبل SL):  {len(tp_wins):>4} ({len(tp_wins)/total*100:.0f}%)')
print(f'  SL فقط (قبل TP):  {len(sl_loss):>4} ({len(sl_loss)/total*100:.0f}%)')
print(f'  TP ثم SL (كلاهما): {len(tp_first):>4}')
print(f'  SL ثم TP (كلاهما): {len(sl_first):>4}')
print(f'  لا هذا ولا ذاك:    {len(neither):>4} ({len(neither)/total*100:.0f}%)')
print(f'  ─────────────────────')
print(f'  إجمالي ربح: {real_wins} | خسارة: {real_losses}')
print(f'  **Win Rate: {wr:.1f}%**')
print(f'  R:R = {TP/SL:.1f}:1')

# Time to TP
if tp_wins or tp_first:
    all_tp = tp_wins + tp_first
    tp_times = [r['tp_hours'] for r in all_tp]
    print(f'\n⏱️ وقت تحقيق +{TP}%:')
    print(f'  متوسط: {np.mean(tp_times):.1f}س | وسيط: {np.median(tp_times):.1f}س')
    
    print(f'\n📅 توزيع الوقت:')
    bins = [(0,6),(6,12),(12,24),(24,48),(48,72),(72,120),(120,168),(168,336)]
    for lo, hi in bins:
        items = [t for t in tp_times if lo <= t < hi]
        if not items: continue
        bar = '█' * int(len(items)/len(tp_times)*40)
        print(f'  {lo}-{hi}س: {len(items):>4} ({len(items)/len(tp_times)*100:.0f}%) {bar}')

# Time to SL
if sl_loss or sl_first:
    all_sl = sl_loss + sl_first
    sl_times = [r['sl_hours'] for r in all_sl]
    print(f'\n⏱️ وقت ضرب الستوب -{SL}%:')
    print(f'  متوسط: {np.mean(sl_times):.1f}س | وسيط: {np.median(sl_times):.1f}س')

# Net outcome (expected value)
avg_win = TP  # +15%
avg_loss = -SL  # -10%
ev = (wr/100 * avg_win) + ((1-wr/100) * avg_loss)
print(f'\n💰 القيمة المتوقعة:')
print(f'  EV = {wr:.0f}% × +{TP}% + {100-wr:.0f}% × -{SL}% = {ev:+.1f}%')

# By month
print(f'\n📊 حسب الشهر:')
for mon, name in [('2026-04','أبريل'),('2026-05','مايو'),('2026-06','يونيو')]:
    mr = [r for r in results if r['dt'].strftime('%Y-%m') == mon]
    if not mr: continue
    mw = sum(1 for r in mr if r['tp_hit'] and (not r['sl_hit'] or r['tp_hours'] < r['sl_hours']))
    ml = len(mr) - mw
    mwr = mw/len(mr)*100
    print(f'  {name}: {len(mr)} صفقة | WR {mwr:.1f}% | ربح {mw} | خسارة {ml}')
