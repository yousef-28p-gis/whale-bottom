#!/usr/bin/env python3
"""Whale 200-bar: LONG only + sell signal exit"""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m.csv', parse_dates=['ts'])
FEE=0.001; CAPITAL=1000
BARS=200

# Whale
lowest=df['low'].rolling(BARS).min()
at_low=(df['low']<=lowest).astype(float)
lc=abs(df['low']-df['low'].shift(1))/df['low']*100
sm=lc.ewm(span=3,adjust=False).mean()
hi=sm.rolling(BARS).max()
st=np.where(at_low>0,(sm+hi*2)/3,0)
df['whale']=pd.Series(st).ewm(span=3,adjust=False).mean().fillna(0)
df['spike']=(df['whale']>df['whale'].shift(1))&(df['whale'].shift(1)<=0.02)
df['wma50']=df['whale'].rolling(50).mean()
df['wma200']=df['whale'].rolling(200).mean()
df['wstr']=df['whale']/df['whale'].rolling(50).max().replace(0,np.nan)*100
df['atr']=(df['high']-df['low']).rolling(14).mean()
df['atr_ma']=df['atr'].rolling(20).mean()
df['vma']=df['volume'].rolling(20).mean()

# RSI
delta=df['close'].diff(); g=delta.clip(lower=0); l=-delta.clip(upper=0)
ag=g.ewm(alpha=1/14,adjust=False).mean(); al=l.ewm(alpha=1/14,adjust=False).mean()
df['rsi']=100-(100/(1+ag/al.replace(0,np.nan)))

# Sell exhaustion (6-cond)
vs=df['volume'].rolling(20).mean(); hh20=df['high'].rolling(20).max().shift(1)
ll10=df['low'].rolling(10).min().shift(1)
c=np.zeros(len(df))
c+=((df['volume']>vs*1.5)&(df['close']<df['open'])).astype(int)
c+=((df['high']>hh20)&(df['close']<hh20)).astype(int)
c+=((df['high']>hh20)&(df['close']<df['open'])).astype(int)
c+=((df['close'].shift(1)>df['open'].shift(1))&(df['volume']>vs*1.5)&(df['close']<df['open'])).astype(int)
c+=(df['low']<ll10).astype(int)
c+=((df['high']>df['high'].shift(1))&(df['rsi']<df['rsi'].shift(1))).astype(int)
df['sell_str']=c/6*100

# Swings for SL
lb=5; sl_arr=np.zeros(len(df),dtype=bool)
for i in range(lb*2,len(df)):
    w=df['low'].iloc[i-lb*2:i+1]; m=i-lb
    if df['low'].iloc[m]==w.min() and w.values.argmin()==lb: sl_arr[i]=True
def nsl(idx):
    for j in range(idx-1,max(0,idx-100),-1):
        if sl_arr[j]: return df['low'].iloc[j]
    return df['low'].iloc[idx]*0.95

print(f"🐋 Whale {BARS}-bar | LONG only | Sell signal exit", flush=True)

for ws in [50, 60, 70]:
    for vm in [1.0, 1.5]:
        for sell_thresh in [60, 70, 80]:
            long_ok = df['wma50'] > df['wma200']
            entry = (df['spike'] & (df['wstr'] > ws) & long_ok &
                     (df['volume'] > df['vma'] * vm) & (df['atr'] > df['atr_ma']))
            
            eis = np.where(entry)[0]
            if len(eis) == 0: continue
            
            trades = []; it = False; ed = 0; eq = CAPITAL
            cm = df['ts'].iloc[400].month; cy = df['ts'].iloc[400].year; ms = CAPITAL
            
            for ei in eis:
                if ei < 500: continue
                if it and ei < ed: continue
                ts = df['ts'].iloc[ei]
                if ts.month != cm or ts.year != cy: cm,cy=ts.month,ts.year; ms=eq
                if (eq-ms)/ms*100 < -7: continue
                
                e = df['close'].iloc[ei]; sl = nsl(ei) * 0.998
                
                end = min(ei+192, len(df)); r=None; ep=e; ex=ei
                for j in range(ei+1, end):
                    if df['low'].iloc[j] <= sl: r='SL'; ep=sl; ex=j; break
                    if df['sell_str'].iloc[j] >= sell_thresh: r='SELL'; ep=df['close'].iloc[j]; ex=j; break
                
                if r is None: r='TIME'; ep=df['close'].iloc[end-1]; ex=end-1
                
                pnl = (ep-e)/e*100 - 0.2
                trades.append({'r':r,'pnl':pnl,'ei':ei,'exi':ex})
                it=True; ed=ex; eq+=CAPITAL*(pnl/100)
            
            n=len(trades)
            if n<10: continue
            
            wins=[t for t in trades if t['pnl']>0]; wr=len(wins)/n*100
            pnls=[t['pnl'] for t in trades]
            sp=np.mean(pnls)/np.std(pnls)*np.sqrt(n) if np.std(pnls)>0 else 0
            
            eqs=[CAPITAL]
            for t in trades: eqs.append(eqs[-1]+CAPITAL*(t['pnl']/100))
            pk=np.maximum.accumulate(eqs); dd=(np.array(eqs)-pk)/pk*100
            
            sell_n=sum(1 for t in trades if t['r']=='SELL')
            sl_n=sum(1 for t in trades if t['r']=='SL')
            time_n=sum(1 for t in trades if t['r']=='TIME')
            avg_w=np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_l=np.mean([t['pnl'] for t in trades if t['pnl']<=0]) if (n-len(wins))>0 else 0
            
            label = f"SELL≥{sell_thresh}"
            print(f"  {ws}%/{vm}x {label}: {n}T | WR:{wr:.0f}% | ${eq:,.0f} | S:{sp:.2f} | DD:{dd.min():.1f}% | Sell/SL/T:{sell_n}/{sl_n}/{time_n} | W:{avg_w:+.1f}% L:{avg_l:+.1f}%", flush=True)

print("\n✅ Done")
