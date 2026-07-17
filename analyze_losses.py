#!/usr/bin/env python3 -u
"""تحليل الصفقات الخاسرة — صفقتين × 50%"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict

CACHE_DIR = '/data/trading28/cache/ohlcv'
MONTHS = ['2026-04', '2026-05', '2026-06']
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.40; COMM=0.20
coins_skip = 0

# Re-run to collect all trades with full details
all_trades = []
for fname in sorted(os.listdir(CACHE_DIR)):
    if not fname.endswith('.json'): continue
    parts = fname.replace('.json','').rsplit('_',1)
    if len(parts) != 2: continue
    sym, mon = parts
    if mon not in MONTHS: continue
    
    with open(f'{CACHE_DIR}/{fname}') as f:
        try: data = json.load(f)
        except: continue
    
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    if len(df) < 500: continue
    
    # Whale
    LB=30
    df['lo']=df['low'].rolling(LB).min()
    df['lc']=abs(df['low']-df['low'].shift(1))/df['low']*100
    df['sm']=df['lc'].ewm(span=3,adjust=False).mean()
    df['hi']=df['sm'].rolling(LB).max()
    df['raw']=np.where(df['low']<=df['lo'],(df['sm']+df['hi']*2)/3,0)
    df['whale']=df['raw'].ewm(span=3,adjust=False).mean().fillna(0)
    df['spike']=(df['whale']>df['whale'].shift(1))&(df['whale'].shift(1)<=0.03)
    df['wf']=df['whale'].rolling(2).mean(); df['ws']=df['whale'].rolling(5).mean()
    df['wp']=df['whale'].rolling(50).max()
    df['str']=(df['whale']/df['wp'].replace(0,np.nan)*100).fillna(0)
    df['vma']=df['volume'].rolling(20).mean()
    df['entry']=(df['spike']&(df['wf']>df['ws'])&(df['str']>STR)&(df['volume']>df['vma']*1.0))
    
    for i in range(50, len(df)-10):
        row = df.iloc[i]
        if row['ts'].month < 4: continue
        if not (row['entry'] and float(row['whale']) >= WHALE_MIN): continue
        if i+1 < len(df) and float(df.iloc[i+1]['whale']) >= 0.35: continue
        ps = max(0, i-96); pb = float(df.iloc[ps]['close'])
        ep = float(row['close'])
        if (ep-pb)/pb*100 >= 0: continue
        
        tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
        pl_p=ep+(tp_p-ep)*(PL/100)
        pl_trig=False; peak=ep; trail_p=0
        for k in range(i+1, len(df)):
            cur=float(df.iloc[k]['close']); h=(k-i)*0.25
            if h>MH: pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
            if cur>=tp_p: pnl=round(TP-COMM,4); exit_='TP'; break
            if cur<=sl_p: pnl=round(-SL-COMM,4); exit_='SL'; break
            if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
            if pl_trig:
                if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
                if cur<=trail_p: pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
        else: pnl=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4); exit_='EOD'
        
        wv = float(row['whale'])
        pump24 = (ep-pb)/pb*100
        all_trades.append({
            'sym': sym, 'dt': row['ts'], 'pnl': pnl, 'exit': exit_,
            'whale_val': wv, 'pump24': pump24,
            'hour': row['ts'].hour, 'weekday': row['ts'].strftime('%A'),
            'month': row['ts'].month
        })

losses = [t for t in all_trades if t['pnl'] <= 0]
wins = [t for t in all_trades if t['pnl'] > 0]

print(f'🔍 تحليل {len(losses)} صفقة خاسرة من {len(all_trades)} إجمالي')
print()

# ── By exit reason ──
print('='*50)
print('📌 حسب سبب الخروج:')
reasons = defaultdict(list)
for t in losses: reasons[t['exit']].append(t)
for r in ['SL','TIME','EOD','TRAIL']:
    if r not in reasons: continue
    lt = reasons[r]
    avg_loss = np.mean([t['pnl'] for t in lt])
    print(f'  {r}: {len(lt)} خسارة | متوسط {avg_loss:+.2f}%')

# ── Top losing coins ──
print()
print('='*50)
print('📌 أكثر العملات خسارة:')
coin_losses = defaultdict(lambda: {'count':0, 'total_pnl':0.0, 'total_trades':0})
for t in all_trades:
    c = coin_losses[t['sym']]
    c['total_trades'] += 1
    if t['pnl'] <= 0:
        c['count'] += 1
        c['total_pnl'] += t['pnl']

sorted_coins = sorted(coin_losses.items(), key=lambda x: x[1]['count'], reverse=True)
for sym, c in sorted_coins[:15]:
    loss_rate = c['count']/c['total_trades']*100
    print(f'  {sym:<8} | {c["count"]:>3}L/{c["total_trades"]:>3}T | خسارة: {c["total_pnl"]:+.1f}% | نسبة {loss_rate:.0f}%')

# ── By hour ──
print()
print('='*50)
print('📌 حسب الساعة:')
hour_data = defaultdict(lambda: {'loss':0, 'total':0})
for t in all_trades:
    h = t['hour']
    hour_data[h]['total'] += 1
    if t['pnl'] <= 0: hour_data[h]['loss'] += 1

for h in sorted(hour_data):
    d = hour_data[h]
    if d['total'] < 10: continue
    wr = (d['total']-d['loss'])/d['total']*100
    bar = '█' * int(d['loss']/5)
    print(f'  {h:02d}:00 | {d["loss"]:>3}خ/{d["total"]:>3}ت | WR {wr:.0f}% | {bar}')

# ── By weekday ──
print()
print('='*50)
print('📌 حسب اليوم:')
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_data = defaultdict(lambda: {'loss':0, 'total':0})
for t in all_trades:
    day_data[t['weekday']]['total'] += 1
    if t['pnl'] <= 0: day_data[t['weekday']]['loss'] += 1

for d in days:
    dd = day_data[d]
    if dd['total'] == 0: continue
    wr = (dd['total']-dd['loss'])/dd['total']*100
    print(f'  {d:<10} | {dd["loss"]:>3}خ/{dd["total"]:>3}ت | WR {wr:.0f}%')

# ── By month ──
print()
print('='*50)
print('📌 حسب الشهر:')
months_ar = {4:'أبريل',5:'مايو',6:'يونيو'}
mon_data = defaultdict(lambda: {'loss':0, 'total':0, 'sl':0, 'time':0, 'trail':0})
for t in all_trades:
    m = t['month']
    mon_data[m]['total'] += 1
    if t['pnl'] <= 0:
        mon_data[m]['loss'] += 1
        if t['exit'] == 'SL': mon_data[m]['sl'] += 1
        elif t['exit'] in ('TIME','EOD'): mon_data[m]['time'] += 1
        elif t['exit'] == 'TRAIL': mon_data[m]['trail'] += 1

for m in [4,5,6]:
    md = mon_data[m]
    wr = (md['total']-md['loss'])/md['total']*100
    print(f'  {months_ar[m]}: {md["total"]}ت | WR {wr:.0f}% | SL={md["sl"]} TIME={md["time"]} TRAIL={md["trail"]}')

# ── Whale value vs loss ──
print()
print('='*50)
print('📌 توزيع قيمة الحوت للصفقات الخاسرة:')
loss_whale = [t['whale_val'] for t in losses]
win_whale = [t['whale_val'] for t in wins]
bins = [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.0]
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    lc = sum(1 for w in loss_whale if lo <= w < hi)
    wc = sum(1 for w in win_whale if lo <= w < hi)
    tot = lc + wc
    wr = wc/tot*100 if tot > 0 else 0
    print(f'  {lo}-{hi}: {lc}خ/{tot}ت | WR {wr:.0f}%')

# ── Pump24 vs loss ──
print()
print('='*50)
print('📌 Pump24 للخاسرة vs الرابحة:')
print(f'  رابحة: متوسط pump24 = {np.mean(win_whale):+.2f}%' if win_whale else '')
loss_pump = [t['pump24'] for t in losses]
win_pump = [t['pump24'] for t in wins]
print(f'  خاسرة: متوسط pump24 = {np.mean(loss_pump):+.2f}%')
print(f'  رابحة: متوسط pump24 = {np.mean(win_pump):+.2f}%')

# ── PnL distribution for losses ──
print()
print('='*50)
print('📌 توزيع حجم الخسارة:')
loss_pnls = [t['pnl'] for t in losses]
ranges = [(-10, -5), (-5, -3), (-3, -2), (-2, -1.5), (-1.5, -1), (-1, 0)]
for lo, hi in ranges:
    cnt = sum(1 for p in loss_pnls if lo <= p < hi)
    print(f'  {lo}% to {hi}%: {cnt} خسارة')
