#!/usr/bin/env python3
"""Test different whale MA combinations for trend filter"""
import pandas as pd, numpy as np, ccxt, os

CACHE='/data/trading28/backtests/cache'; FEE=0.001; B=200

ddf=pd.read_csv(f"{CACHE}/FET_USDT_1d.csv",parse_dates=['ts'])
ddf['ema50']=ddf['close'].ewm(span=50,adjust=False).mean()
ddf['date']=ddf['ts'].dt.date
ema_map=ddf.set_index('date')['ema50'].to_dict()

df=pd.read_csv(f"{CACHE}/FET_USDT_15m_FULL.csv",parse_dates=['ts'])
df['date']=df['ts'].dt.date
df['dema50']=df['date'].map(ema_map)

# Whale
lo=df['low'].rolling(B).min();al=(df['low']<=lo).astype(float)
lc=abs(df['low']-df['low'].shift(1))/df['low']*100;sm=lc.ewm(span=3,adjust=False).mean()
hi=sm.rolling(B).max();st=np.where(al>0,(sm+hi*2)/3,0)
df['w']=pd.Series(st).ewm(span=3,adjust=False).mean().fillna(0)
df['spike']=(df['w']>df['w'].shift(1))&(df['w'].shift(1)<=0.02)

# Precompute all whale MAs
for p in [20,50,100,200]:
    df[f'w{p}']=df['w'].rolling(p).mean()

# Sell signal
delta=df['close'].diff();g=delta.clip(lower=0);l=-delta.clip(upper=0)
ag=g.ewm(alpha=1/14,adjust=False).mean();al=l.ewm(alpha=1/14,adjust=False).mean()
df['rsi']=100-(100/(1+ag/al.replace(0,np.nan)))
vs=df['volume'].rolling(20).mean();hh20=df['high'].rolling(20).max().shift(1)
ll10=df['low'].rolling(10).min().shift(1)
c=np.zeros(len(df))
c+=((df['volume']>vs*1.5)&(df['close']<df['open'])).astype(int)
c+=((df['high']>hh20)&(df['close']<hh20)).astype(int)
c+=((df['high']>hh20)&(df['close']<df['open'])).astype(int)
c+=((df['close'].shift(1)>df['open'].shift(1))&(df['volume']>vs*1.5)&(df['close']<df['open'])).astype(int)
c+=(df['low']<ll10).astype(int);c+=((df['high']>df['high'].shift(1))&(df['rsi']<df['rsi'].shift(1))).astype(int)
df['sell']=c/6*100

market_ok=df['close']>df['dema50']

# Test MA combos
combos=[(20,50),(20,100),(50,100),(50,200),(20,200),(100,200)]

print(f"{'MA':<12} {'Signals':>7} {'Trades':>6} {'WR':>5} {'PF':>8} {'DD':>6} {'R:R':>5}")
print("-"*55)

for fast,slow in combos:
    trend=df[f'w{fast}']>df[f'w{slow}']
    entry_sig=df['spike'] & trend & market_ok
    eis=np.where(entry_sig)[0]
    
    trades=[];it=False;ed=0;equity=1000;pos=25
    cmon=df['ts'].iloc[500].month;cyr=df['ts'].iloc[500].year;mstart=1000
    
    for ei in eis:
        if ei<500:continue
        if it and ei<ed:continue
        ts=df['ts'].iloc[ei]
        if ts.month!=cmon or ts.year!=cyr:cmon,cyr=ts.month,ts.year;mstart=equity
        if (equity-mstart)/mstart*100<=-7:continue
        e=df['close'].iloc[ei]
        end=min(ei+192,len(df));r=None;ep=e;ex=ei
        for j in range(ei+1,end):
            if df['sell'].iloc[j]>=60:r='SELL';ep=df['close'].iloc[j];ex=j;break
        if not r:r='TIME';ep=df['close'].iloc[end-1];ex=end-1
        pnl=(ep-e)/e*100-FEE*200
        dollar=equity*(pos/100)*(pnl/100);equity+=dollar
        trades.append({'pnl':pnl,'r':r,'eq':equity})
        it=True;ed=ex
    
    n=len(trades)
    if n<5:continue
    wins=[t for t in trades if t['pnl']>0];nw=len(wins);wr=nw/n*100
    aw=np.mean([t['pnl'] for t in wins]) if wins else 0
    aloss=np.mean([t['pnl'] for t in trades if t['pnl']<=0]) if (n-nw) else 0
    rr=abs(aw/aloss) if aloss else 0
    eqs=[1000]
    for t in trades:eqs.append(t['eq'])
    peak=np.maximum.accumulate(eqs);dd=(np.array(eqs)-peak)/peak*100
    
    sigs=len(eis)
    print(f"w{fast}/w{slow:<7} {sigs:>7} {n:>6} {wr:>4.0f}% ${equity:>7,.0f} {dd.min():>5.1f}% {rr:>4.1f}x")
