#!/usr/bin/env python3
"""Test SMA vs EMA vs WMA for daily market filter"""
import pandas as pd, numpy as np

CACHE='/data/trading28/backtests/cache'; FEE=0.001; B=200

ddf=pd.read_csv(f"{CACHE}/FET_USDT_1d.csv",parse_dates=['ts'])
ddf['date']=ddf['ts'].dt.date

for p in [50]:
    ddf[f'sma{p}']=ddf['close'].rolling(p).mean()
    ddf[f'ema{p}']=ddf['close'].ewm(span=p,adjust=False).mean()
    # WMA: weighted moving average
    def wma(series, period):
        w=np.arange(1,period+1)
        return series.rolling(period).apply(lambda x:np.sum(x*w)/w.sum(),raw=True)
    ddf[f'wma{p}']=wma(ddf['close'],p)

df=pd.read_csv(f"{CACHE}/FET_USDT_15m_FULL.csv",parse_dates=['ts'])
df['date']=df['ts'].dt.date

# Whale + sell signal (same as before)
lo=df['low'].rolling(B).min();al=(df['low']<=lo).astype(float)
lc=abs(df['low']-df['low'].shift(1))/df['low']*100;sm=lc.ewm(span=3,adjust=False).mean()
hi=sm.rolling(B).max();st=np.where(al>0,(sm+hi*2)/3,0)
df['w']=pd.Series(st).ewm(span=3,adjust=False).mean().fillna(0)
df['spike']=(df['w']>df['w'].shift(1))&(df['w'].shift(1)<=0.02)
df['w20']=df['w'].rolling(20).mean();df['w50']=df['w'].rolling(50).mean()
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

trend_ok=df['w20']>df['w50']

print(f"{'نوع المتوسط':<12} {'إشارات':>6} {'صفقات':>6} {'WR':>5} {'محفظة':>8} {'DD':>6}")
print("-"*50)

for ma_type in ['sma50','ema50','wma50']:
    ma_map=ddf.set_index('date')[ma_type].to_dict()
    df['ma']=df['date'].map(ma_map)
    entry_sig=df['spike'] & trend_ok & (df['close']>df['ma'])
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
        trades.append({'pnl':pnl,'eq':equity})
        it=True;ed=ex
    
    n=len(trades)
    if n<5:continue
    wins=[t for t in trades if t['pnl']>0];nw=len(wins);wr=nw/n*100
    eqs=[1000]
    for t in trades:eqs.append(t['eq'])
    peak=np.maximum.accumulate(eqs);dd=(np.array(eqs)-peak)/peak*100
    sigs=len(eis)
    print(f"{ma_type.upper():<12} {sigs:>6} {n:>6} {wr:>4.0f}% ${equity:>7,.0f} {dd.min():>5.1f}%")
