#!/usr/bin/env python3
"""Grid search: optimize PL, MAX_H, TRAIL for close-only (live-style) simulation."""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

CACHE = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'

TP = 2.5; SL = 2.0
STR = 50; WHALE_MIN = 0.35; MIN_VOL = 200000; COMMISSION = 0.20

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
    df = df.copy(); LB = 30
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
    df['entry'] = (df['spike'] & (df['wf'] > df['ws']) & (df['str'] > STR) & (df['volume'] > df['vma'] * 1.0))
    return df

def simulate_close(df, entry_idx, PL, TRAIL, MAX_H):
    ep = df.iloc[entry_idx]['close']
    tp_p = ep * (1+TP/100); sl_p = ep * (1-SL/100)
    pl_p = ep + (tp_p - ep) * (PL/100)
    pl_trig = False; peak = ep; trail_p = 0
    for j in range(entry_idx + 1, len(df)):
        row = df.iloc[j]; cur = row['close']
        hours = (j - entry_idx) * 0.25
        if hours > MAX_H:
            return ('T', round((cur-ep)/ep*100, 4))
        if cur >= tp_p: return ('P', round(TP, 4))
        if cur <= sl_p: return ('S', round(-SL, 4))
        if not pl_trig and cur >= pl_p:
            pl_trig = True; peak = cur; trail_p = cur * (1-TRAIL/100)
        if pl_trig:
            if cur > peak: peak = cur; trail_p = cur * (1-TRAIL/100)
            if cur <= trail_p: return ('L', round((trail_p-ep)/ep*100, 4))
    return ('E', round((df.iloc[-1]['close']-ep)/ep*100, 4))

with open(SIGNALS_FILE) as f: raw = json.load(f)
signals = []
for s in raw:
    if s['symbol'] in STABLES or s['symbol'] in BLOCKED: continue
    if s.get('direction', 'LONG') != 'LONG': continue
    if s.get('volume_usdt', 0) < MIN_VOL: continue
    dt = datetime.fromisoformat(s['dt'])
    if dt.month not in (4, 5, 6) or dt.year != 2026: continue
    signals.append({'symbol': s['symbol'], 'dt': dt, 'month': dt.strftime('%Y-%m')})

by_pair = defaultdict(list)
for sig in signals: by_pair[(sig['symbol'], sig['month'])].append(sig)

print('Loading data...')
all_entries = []
for (sym, mon), sigs in by_pair.items():
    df = load_cached(sym, mon)
    if df is None: continue
    df_w = whale_indicator(df)
    for sig in sigs:
        df_w['td'] = abs((df_w['ts'] - sig['dt']).dt.total_seconds())
        nearest = df_w['td'].idxmin()
        forward = df_w.iloc[nearest:].reset_index(drop=True)
        for j, row in forward.iterrows():
            if j * 0.25 > 24: break
            if row['entry']:
                wv = float(row['whale'])
                if wv >= WHALE_MIN:
                    all_entries.append((forward, j, sig['month']))
                break

print(f'Entries: {len(all_entries)}')

print()
print('Grid Search - Close-Only Simulation')
print('=' * 75)
header = f'{"PL":>4} {"H":>3} {"TR":>4} {"#":>5} {"WR":>6} {"Net":>8} {"AvgW":>6} {"AvgL":>6} {"P":>3} {"L":>3} {"T":>3} {"S":>3}'
print(header)
print('-' * 75)

results = []
for PL in [30, 40, 50, 60]:
    for MAX_H in [2, 3, 4, 6, 8]:
        for TRAIL in [0.10, 0.15, 0.20, 0.30]:
            trades = []
            for forward, ei, mon in all_entries:
                st, pnl = simulate_close(forward, ei, PL, TRAIL, MAX_H)
                trades.append({'pnl': pnl, 'net': round(pnl - COMMISSION, 4), 'exit': st})
            
            wins = [t for t in trades if t['net'] > 0]
            losses = [t for t in trades if t['net'] <= 0]
            wr = len(wins)/len(trades)*100
            net = sum(t['net'] for t in trades)
            avg_win = np.mean([t['net'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['net'] for t in losses]) if losses else 0
            
            exits = defaultdict(int)
            for t in trades: exits[t['exit']] += 1
            
            results.append((PL, MAX_H, TRAIL, len(trades), wr, net, avg_win, avg_loss, exits))

results.sort(key=lambda x: (x[5], x[4]), reverse=True)

print()
print('TOP 15 (sorted by net profit):')
for i, (pl, mh, tr, nt, wr, net, aw, al, exits) in enumerate(results[:15]):
    flag = ' <-- BEST' if i == 0 else ''
    p = exits.get('P',0); l = exits.get('L',0); t = exits.get('T',0); s = exits.get('S',0)
    print(f'{pl:>4} {mh:>3}h {tr:>4.2f} {nt:>5} {wr:>5.1f}% {net:>+7.1f}% {aw:>+5.1f}% {al:>+5.1f}% {p:>3} {l:>3} {t:>3} {s:>3}{flag}')

print()
print('ORIGINAL config for comparison:')
for pl, mh, tr, nt, wr, net, aw, al, exits in results:
    if pl == 40 and mh == 2 and tr == 0.10:
        p = exits.get('P',0); l = exits.get('L',0); t = exits.get('T',0); s = exits.get('S',0)
        print(f'{pl:>4} {mh:>3}h {tr:>4.2f} {nt:>5} {wr:>5.1f}% {net:>+7.1f}% {aw:>+5.1f}% {al:>+5.1f}% {p:>3} {l:>3} {t:>3} {s:>3} <-- ORIGINAL')
        break

# Portfolio for best
best = results[0]
best_pl, best_mh, best_tr = best[0], best[1], best[2]
print(f'\nBEST: PL={best_pl} | MAX_H={best_mh}h | TRAIL={best_tr}')
print(f'  WR={best[4]:.1f}% | Net={best[5]:+.1f}% | AvgWin={best[6]:+.2f}% | AvgLoss={best[7]:+.2f}%')
