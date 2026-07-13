#!/usr/bin/env python3
"""Whale v3: 200-bar whale test"""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m.csv', parse_dates=['ts'])

# ─── Whale: 200-bar ─────────────────────────────────────────────
print("🐋 Computing whale (200-bar)...", flush=True)
lowest = df['low'].rolling(200).min()
at_low = (df['low'] <= lowest).astype(float)
low_change = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
smooth = low_change.ewm(span=3, adjust=False).mean()
highest = smooth.rolling(200).max()
strength = np.where(at_low > 0, (smooth + highest * 2) / 3, 0)
df['whale'] = pd.Series(strength).ewm(span=3, adjust=False).mean().fillna(0)
df['whale_spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.02)

df['w_ma50'] = df['whale'].rolling(50).mean()
df['w_ma200'] = df['whale'].rolling(200).mean()
df['w_peak50'] = df['whale'].rolling(50).max()
df['w_strength'] = df['whale'] / df['w_peak50'].replace(0, np.nan) * 100
df['atr'] = (df['high'] - df['low']).rolling(14).mean()
df['atr_ma20'] = df['atr'].rolling(20).mean()
df['vol_ma20'] = df['volume'].rolling(20).mean()

# Swings
lb = 5
sh = np.zeros(len(df), dtype=bool)
sl_arr = np.zeros(len(df), dtype=bool)
for i in range(lb*2, len(df)):
    w = df['high'].iloc[i-lb*2:i+1]; m = i-lb
    if df['high'].iloc[m]==w.max() and w.values.argmax()==lb: sh[i]=True
    w = df['low'].iloc[i-lb*2:i+1]
    if df['low'].iloc[m]==w.min() and w.values.argmin()==lb: sl_arr[i]=True

def nsl(idx):
    for j in range(idx-1,max(0,idx-100),-1):
        if sl_arr[j]: return df['low'].iloc[j]
    return df['low'].iloc[idx]*0.95
def nsh(idx):
    for j in range(idx-1,max(0,idx-100),-1):
        if sh[j]: return df['high'].iloc[j]
    return df['high'].iloc[idx]*1.05

print(f"🐋 Whale spikes: {df['whale_spike'].sum()}", flush=True)

# ─── Test ───────────────────────────────────────────────────────
FEE = 0.001; CAPITAL = 1000

for ws in [50, 60, 70]:
    for vm in [1.0, 1.5]:
        long_ok = df['w_ma50'] > df['w_ma200']
        short_ok = df['w_ma50'] < df['w_ma200']
        
        long_entry = (df['whale_spike'] & (df['w_strength'] > ws) & long_ok &
                      (df['volume'] > df['vol_ma20'] * vm) & (df['atr'] > df['atr_ma20']))
        short_entry = (df['whale_spike'] & (df['w_strength'] > ws) & short_ok &
                       (df['volume'] > df['vol_ma20'] * vm) & (df['atr'] > df['atr_ma20']))
        
        entry_idxs = np.where(long_entry | short_entry)[0]
        if len(entry_idxs) == 0: continue
        
        trades = []; in_trade = False; exi_done = 0; equity = CAPITAL
        cmon=df['ts'].iloc[300].month; cyr=df['ts'].iloc[300].year; mstart=CAPITAL
        
        for ei in entry_idxs:
            if ei < 400: continue
            if in_trade and ei < exi_done: continue
            ts = df['ts'].iloc[ei]
            if ts.month != cmon or ts.year != cyr: cmon,cyr=ts.month,ts.year; mstart=equity
            if (equity-mstart)/mstart*100 < -7: continue
            
            is_long = long_entry.iloc[ei]; entry = df['close'].iloc[ei]
            if is_long:
                sl = nsl(ei) * 0.998; tp = 99999
            else:
                sl = nsh(ei) * 1.002; tp = entry - df['atr'].iloc[ei] * 3
            
            end = min(ei+192, len(df)); result=None; exit_px=entry; exi=ei
            for j in range(ei+1, end):
                if is_long:
                    if df['low'].iloc[j]<=sl: result='SL'; exit_px=sl; exi=j; break
                    if short_entry.iloc[j] and df['w_strength'].iloc[j]>ws: result='REV'; exit_px=df['close'].iloc[j]; exi=j; break
                else:
                    if df['high'].iloc[j]>=sl: result='SL'; exit_px=sl; exi=j; break
                    if df['low'].iloc[j]<=tp: result='TP'; exit_px=tp; exi=j; break
            if result is None: result='TIME'; exit_px=df['close'].iloc[end-1]; exi=end-1
            
            pnl = (exit_px-entry)/entry*100
            if is_long: pnl -= 0.2
            else: pnl = -pnl - 0.2
            
            trades.append({'is_long':is_long,'result':result,'pnl':pnl,'ei':ei,'exi':exi})
            in_trade=True; exi_done=exi; equity+=CAPITAL*(pnl/100)
        
        n=len(trades)
        if n==0: continue
        
        wins=[t for t in trades if t['pnl']>0]; wr=len(wins)/n*100
        pnls=[t['pnl'] for t in trades]
        sp=np.mean(pnls)/np.std(pnls)*np.sqrt(n) if np.std(pnls)>0 else 0
        
        eqs=[CAPITAL]
        for t in trades: eqs.append(eqs[-1]+CAPITAL*(t['pnl']/100))
        peak=np.maximum.accumulate(eqs); dd=(np.array(eqs)-peak)/peak*100
        
        lt=[t for t in trades if t['is_long']]; st=[t for t in trades if not t['is_long']]
        lwr=len([t for t in lt if t['pnl']>0])/len(lt)*100 if lt else 0
        swr=len([t for t in st if t['pnl']>0])/len(st)*100 if st else 0
        
        rev=sum(1 for t in trades if t['result']=='REV')
        tp=sum(1 for t in trades if t['result']=='TP')
        sl_c=sum(1 for t in trades if t['result']=='SL')
        
        print(f"  {ws}%/{vm}x: {n}T | WR:{wr:.0f}% | ${equity:,.0f} | L/S:{lwr:.0f}/{swr:.0f} | S:{sp:.2f} | DD:{dd.min():.1f}% | R/T/S:{rev}/{tp}/{sl_c}", flush=True)

print("\n✅ Done")
