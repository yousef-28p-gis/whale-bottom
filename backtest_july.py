#!/usr/bin/env python3 -u
"""باك تيست يوليو 2026 — WhaleSniper + RSI<25"""
import ccxt, json, os, numpy as np, pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
import time

CACHE_DIR = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'
os.makedirs(CACHE_DIR, exist_ok=True)

TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20

STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
BLOCKED={'SUPER','ORCA','VANA','W','DOGS','MET','XLM','BB','COS','LUNA','S'}

# ── Phase 1: Build July cache ──
exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})

with open(SIGNALS_FILE) as f: raw = json.load(f)
july_pairs = set()
for s in raw:
    if s['symbol'] in STABLES or s['symbol'] in BLOCKED: continue
    if s.get('direction','LONG') != 'LONG': continue
    if s.get('volume_usdt',0) < 200000: continue
    dt = datetime.fromisoformat(s['dt'])
    if dt.month != 7 or dt.year != 2026: continue
    july_pairs.add(s['symbol'])

print(f'📦 بناء كاش يوليو لـ {len(july_pairs)} عملة...', flush=True)
t0 = time.time()

for sym in sorted(july_pairs):
    fpath = f'{CACHE_DIR}/{sym}_2026-07.json'
    if os.path.exists(fpath): continue
    
    since = exchange.parse8601('2026-07-01T00:00:00Z')
    end_ts = exchange.parse8601('2026-08-01T00:00:00Z')
    
    all_candles = []
    fetch_since = since
    for _ in range(6):
        try:
            candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=fetch_since, limit=1000)
        except:
            break
        if not candles: break
        all_candles.extend(candles)
        if candles[-1][0] >= end_ts or len(candles) < 1000: break
        fetch_since = candles[-1][0] + 1
    
    if all_candles:
        with open(fpath, 'w') as f:
            json.dump(all_candles, f)

elapsed = time.time() - t0
print(f'✅ تم بناء الكاش: {elapsed:.0f}ث\n', flush=True)

# ── Phase 2: Backtest ──
signals = []
for s in raw:
    if s['symbol'] in STABLES or s['symbol'] in BLOCKED: continue
    if s.get('direction','LONG') != 'LONG': continue
    if s.get('volume_usdt',0) < 200000: continue
    dt = datetime.fromisoformat(s['dt'])
    if dt.month != 7 or dt.year != 2026: continue
    signals.append({'symbol': s['symbol'], 'dt': dt})

print(f'📊 {len(signals)} إشارة يوليو', flush=True)

all_trades = []
for sig in signals:
    sym = sig['symbol']
    fpath = f'{CACHE_DIR}/{sym}_2026-07.json'
    if not os.path.exists(fpath): continue
    
    with open(fpath) as f:
        try: data = json.load(f)
        except: continue
    
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    if len(df) < 500: continue
    
    # Whale + RSI
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
    df['entry']=(df['spike']&(df['wf']>df['ws'])&(df['str']>50)&(df['volume']>df['vma']*1.0))
    delta = df['close'].diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100/(1+rs))
    
    # Find nearest entry to signal
    df['td'] = abs((df['ts'] - sig['dt']).dt.total_seconds())
    n = df['td'].idxmin()
    fwd = df.iloc[n:].reset_index(drop=True)
    
    for j, row in fwd.iterrows():
        if j * 0.25 > 24: break
        if not row['entry']: continue
        if float(row['whale']) < WHALE_MIN: continue
        if j+1 < len(fwd) and float(fwd.iloc[j+1]['whale']) >= 0.35: continue
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi >= 25: continue  # 🔥
        if row['ts'].weekday() == 3: continue
        if row['ts'].hour in (1,3,6,12,0,4): continue
        
        ps = max(0, n+j-96)
        pb = float(df.iloc[ps]['close']) if ps < len(df) else float(row['close'])
        ep = float(row['close'])
        if (ep-pb)/pb*100 >= 0: continue
        
        tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
        pl_p=ep+(tp_p-ep)*(PL/100)
        pl_trig=False; peak=ep; trail_p=0
        for k in range(j+1, len(fwd)):
            cur=float(fwd.iloc[k]['close']); h=(k-j)*0.25
            if h>MH: pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
            if cur>=tp_p: pnl=round(TP-COMM,4); exit_='TP'; break
            if cur<=sl_p: pnl=round(-SL-COMM,4); exit_='SL'; break
            if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
            if pl_trig:
                if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
                if cur<=trail_p: pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
        else: pnl=round((float(fwd.iloc[-1]['close'])-ep)/ep*100-COMM,4); exit_='EOD'
        all_trades.append({'dt': row['ts'], 'pnl': pnl, 'exit': exit_, 'sym': sym})
        break

print(f'✅ {len(all_trades)} إشارة مؤهلة', flush=True)

# ── Stats ──
nets=[t['pnl'] for t in all_trades]
wins=sum(1 for n in nets if n>0)
exits=defaultdict(int)
for t in all_trades: exits[t['exit']] += 1

print(f'\n📊 يوليو 2026 — إشارات مؤهلة: {len(all_trades)}')
print(f'🟢 رابحة: {wins} | 🔴 خاسرة: {len(all_trades)-wins}')
print(f'📈 WR: {wins/len(all_trades)*100:.1f}%' if all_trades else 'N/A')
print(f'🎯 TP={exits.get("TP",0)} 🛑 SL={exits.get("SL",0)} 🐌 TRAIL={exits.get("TRAIL",0)} ⏰ TIME={exits.get("TIME",0)+exits.get("EOD",0)}')
print()

# ── Portfolio: 2 × 50% ──
trades_sorted = sorted(all_trades, key=lambda x: x['dt'])
capital = 1000.0; peak = 1000.0; max_dd = 0.0
active = []; skipped = 0; taken = 0; exec_trades = []

for t in trades_sorted:
    dt = t['dt']
    still_active = []
    for exit_dt, cost, pnl_amt in active:
        if dt >= exit_dt:
            capital += cost + pnl_amt
        else:
            still_active.append((exit_dt, cost, pnl_amt))
    active = still_active
    
    if len(active) >= 2:
        skipped += 1; continue
    
    pos_size = capital * 0.50
    if capital < pos_size:
        skipped += 1; continue
    
    pnl_amt = pos_size * t['pnl'] / 100
    capital -= pos_size
    active.append((dt + timedelta(hours=MH), pos_size, pnl_amt))
    taken += 1
    exec_trades.append(t)
    
    equity = capital + sum(pc + pd for _, pc, pd in active)
    if equity > peak: peak = equity
    dd = (equity - peak) / peak * 100
    if dd < max_dd: max_dd = dd

for _, cost, pnl_amt in active:
    capital += cost + pnl_amt

exec_nets = [t['pnl'] for t in exec_trades]
exec_wins = sum(1 for n in exec_nets if n > 0)

print(f'{"="*60}')
print(f'🔥 RSI<25 + حوت≥0.50 — يوليو 2026')
print(f'{"="*60}')
print(f'📋 إشارات مؤهلة: {len(all_trades)}')
print(f'✅ منفذة: {taken} | ⏭️ متخطية: {skipped}')
print(f'🟢 رابحة: {exec_wins} | 🔴 خاسرة: {taken-exec_wins}')
print(f'📈 WR منفذة: {exec_wins/taken*100:.1f}%' if taken>0 else 'N/A')
print(f'💼 محفظة: $1000 → ${capital:.0f} ({capital/10-100:+.1f}%)')
print(f'📉 سحب: {max_dd:.2f}%')

total_time = time.time() - t0
print(f'\n⏱️ الوقت: {total_time:.0f}ث')
