#!/usr/bin/env python3
"""Test different exit strategies"""
import pandas as pd, numpy as np

CACHE='/data/trading28/backtests/cache'; FEE=0.001; B=200

ddf=pd.read_csv(f"{CACHE}/FET_USDT_1d.csv",parse_dates=['ts'])
ddf['date']=ddf['ts'].dt.date
ddf['sma50']=ddf['close'].rolling(50).mean()

df=pd.read_csv(f"{CACHE}/FET_USDT_15m_FULL.csv",parse_dates=['ts'])
df['date']=df['ts'].dt.date
df['sma50d']=df['date'].map(ddf.set_index('date')['sma50'].to_dict())

# Whale
lo=df['low'].rolling(B).min();al=(df['low']<=lo).astype(float)
lc=abs(df['low']-df['low'].shift(1))/df['low']*100;sm=lc.ewm(span=3,adjust=False).mean()
hi=sm.rolling(B).max();st=np.where(al>0,(sm+hi*2)/3,0)
df['w']=pd.Series(st).ewm(span=3,adjust=False).mean().fillna(0)
df['spike']=(df['w']>df['w'].shift(1))&(df['w'].shift(1)<=0.02)
df['w20']=df['w'].rolling(20).mean();df['w50']=df['w'].rolling(50).mean()
df['atr']=(df['high']-df['low']).rolling(14).mean()

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

# Swings for SL
lb=5;swl=np.zeros(len(df),dtype=bool)
for i in range(lb*2,len(df)):
    w=df['low'].iloc[i-lb*2:i+1];m=i-lb
    if df['low'].iloc[m]==w.min() and w.values.argmax()==lb:swl[i]=True
def nsl(i):
    for j in range(i-1,max(0,i-100),-1):
        if swl[j]:return df['low'].iloc[j]
    return df['low'].iloc[i]*0.95

# Entry: whale spike + w20>w50 + close>SMA50d
entry_sig=df['spike'] & (df['w20']>df['w50']) & (df['close']>df['sma50d'])
eis=np.where(entry_sig)[0]

def test_exit(name, exit_fn):
    """exit_fn(entry_idx, entry_price, df) -> (exit_idx, exit_price, result)"""
    trades=[];it=False;ed=0;equity=1000;pos=25
    cmon=df['ts'].iloc[500].month;cyr=df['ts'].iloc[500].year;mstart=1000
    for ei in eis:
        if ei<500:continue
        if it and ei<ed:continue
        ts=df['ts'].iloc[ei]
        if ts.month!=cmon or ts.year!=cyr:cmon,cyr=ts.month,ts.year;mstart=equity
        if (equity-mstart)/mstart*100<=-7:continue
        e=df['close'].iloc[ei]
        ex,ep,r=exit_fn(ei,e)
        pnl=(ep-e)/e*100-FEE*200
        dollar=equity*(pos/100)*(pnl/100);equity+=dollar
        trades.append({'pnl':pnl,'eq':equity});it=True;ed=ex
    n=len(trades)
    if n<5:return None
    wins=[t for t in trades if t['pnl']>0];nw=len(wins);wr=nw/n*100
    eqs=[1000]
    for t in trades:eqs.append(t['eq'])
    peak=np.maximum.accumulate(eqs);dd=(np.array(eqs)-peak)/peak*100
    aw=np.mean([t['pnl'] for t in wins]) if wins else 0
    aloss=np.mean([t['pnl'] for t in trades if t['pnl']<=0]) if (n-nw) else 0
    rr=abs(aw/aloss) if aloss else 0
    return {'name':name,'n':n,'wr':wr,'eq':equity,'dd':dd.min(),'rr':rr,'aw':aw,'al':aloss}

# Exit strategies
def exit_sell(ei,e,thresh=60):
    end=min(ei+192,len(df))
    for j in range(ei+1,end):
        if df['sell'].iloc[j]>=thresh:return (j,df['close'].iloc[j],'SELL')
    return (end-1,df['close'].iloc[end-1],'TIME')

def exit_tp_atr(ei,e,mult=2):
    tp=e+df['atr'].iloc[ei]*mult
    end=min(ei+192,len(df))
    for j in range(ei+1,end):
        if df['high'].iloc[j]>=tp:return (j,tp,'TP')
    return (end-1,df['close'].iloc[end-1],'TIME')

def exit_sl_fixed(ei,e,pct=8):
    sl=e*(1-pct/100)
    end=min(ei+192,len(df))
    for j in range(ei+1,end):
        if df['low'].iloc[j]<=sl:return (j,sl,'SL')
        if df['sell'].iloc[j]>=60:return (j,df['close'].iloc[j],'SELL')
    return (end-1,df['close'].iloc[end-1],'TIME')

def exit_sl_swing(ei,e):
    sl=nsl(ei)*0.998
    end=min(ei+192,len(df))
    for j in range(ei+1,end):
        if df['low'].iloc[j]<=sl:return (j,sl,'SL')
        if df['sell'].iloc[j]>=60:return (j,df['close'].iloc[j],'SELL')
    return (end-1,df['close'].iloc[end-1],'TIME')

def exit_tp_sl(ei,e):
    tp=e+df['atr'].iloc[ei]*3;sl=e*(1-8/100)
    end=min(ei+192,len(df))
    for j in range(ei+1,end):
        if df['low'].iloc[j]<=sl:return (j,sl,'SL')
        if df['high'].iloc[j]>=tp:return (j,tp,'TP')
    return (end-1,df['close'].iloc[end-1],'TIME')

def exit_tp_sell(ei,e):
    tp=e+df['atr'].iloc[ei]*3
    end=min(ei+192,len(df))
    for j in range(ei+1,end):
        if df['high'].iloc[j]>=tp:return (j,tp,'TP')
        if df['sell'].iloc[j]>=60:return (j,df['close'].iloc[j],'SELL')
    return (end-1,df['close'].iloc[end-1],'TIME')

results=[]
for name,fn in [
    ("إشارة بيع ≥60% (الحالي)", lambda ei,e: exit_sell(ei,e,60)),
    ("إشارة بيع ≥70%", lambda ei,e: exit_sell(ei,e,70)),
    ("إشارة بيع ≥80%", lambda ei,e: exit_sell(ei,e,80)),
    ("TP=2ATR فقط", lambda ei,e: exit_tp_atr(ei,e,2)),
    ("TP=3ATR فقط", lambda ei,e: exit_tp_atr(ei,e,3)),
    ("SL=-8% + بيع≥60%", exit_sl_fixed),
    ("SL سوينج + بيع≥60%", exit_sl_swing),
    ("TP=3ATR + SL=-8%", exit_tp_sl),
    ("TP=3ATR + بيع≥60%", exit_tp_sell),
]:
    r=test_exit(name,fn)
    if r:results.append(r)

print(f"{'استراتيجية الخروج':<28} {'صفقات':>5} {'WR':>5} {'محفظة':>8} {'DD':>6} {'R:R':>5} {'W/L':>10}")
print("-"*75)
for r in sorted(results,key=lambda x:x['eq'],reverse=True):
    wl=f"+{r['aw']:.1f}/{r['al']:.1f}"
    print(f"{r['name']:<28} {r['n']:>5} {r['wr']:>4.0f}% ${r['eq']:>7,.0f} {r['dd']:>5.1f}% {r['rr']:>4.1f}x {wl:>10}")
