#!/usr/bin/env python3
"""Test MA25, MA50, MA100 as price filters for LONG/SHORT direction"""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv', parse_dates=['ts'])
B=200; FEE=0.001

# Whale
lo=df['low'].rolling(B).min();al=(df['low']<=lo).astype(float)
lc=abs(df['low']-df['low'].shift(1))/df['low']*100;sm=lc.ewm(span=3,adjust=False).mean()
hi=sm.rolling(B).max();st=np.where(al>0,(sm+hi*2)/3,0)
df['w']=pd.Series(st).ewm(span=3,adjust=False).mean().fillna(0)
df['spike']=(df['w']>df['w'].shift(1))&(df['w'].shift(1)<=0.02)
df['w50']=df['w'].rolling(50).mean();df['w200']=df['w'].rolling(200).mean()
df['wstr']=df['w']/df['w'].rolling(50).max().replace(0,np.nan)*100
df['atr']=(df['high']-df['low']).rolling(14).mean();df['vma']=df['volume'].rolling(20).mean()

# Price MAs
df['pma25']=df['close'].rolling(25).mean()
df['pma50']=df['close'].rolling(50).mean()
df['pma100']=df['close'].rolling(100).mean()

# Sell/Buy
delta=df['close'].diff();g=delta.clip(lower=0);l=-delta.clip(upper=0)
ag=g.ewm(alpha=1/14,adjust=False).mean();al=l.ewm(alpha=1/14,adjust=False).mean()
df['rsi']=100-(100/(1+ag/al.replace(0,np.nan)))
vs=df['volume'].rolling(20).mean();hh20=df['high'].rolling(20).max().shift(1)
ll20=df['low'].rolling(20).min().shift(1);ll10=df['low'].rolling(10).min().shift(1)
hh10=df['high'].rolling(10).max().shift(1)
c=np.zeros(len(df))
c+=((df['volume']>vs*1.5)&(df['close']<df['open'])).astype(int)
c+=((df['high']>hh20)&(df['close']<hh20)).astype(int)
c+=((df['high']>hh20)&(df['close']<df['open'])).astype(int)
c+=((df['close'].shift(1)>df['open'].shift(1))&(df['volume']>vs*1.5)&(df['close']<df['open'])).astype(int)
c+=(df['low']<ll10).astype(int);c+=((df['high']>df['high'].shift(1))&(df['rsi']<df['rsi'].shift(1))).astype(int)
df['sell']=c/6*100
c2=np.zeros(len(df))
c2+=((df['volume']>vs*1.5)&(df['close']>df['open'])).astype(int)
c2+=((df['low']<ll20)&(df['close']>ll20)).astype(int)
c2+=((df['low']<ll20)&(df['close']>df['open'])).astype(int)
c2+=((df['close'].shift(1)<df['open'].shift(1))&(df['volume']>vs*1.5)&(df['close']>df['open'])).astype(int)
c2+=(df['high']>hh10).astype(int);c2+=((df['low']<df['low'].shift(1))&(df['rsi']>df['rsi'].shift(1))).astype(int)
df['buy_sig']=c2/6*100

# Swings
lb=5;swl=np.zeros(len(df),dtype=bool);swh=np.zeros(len(df),dtype=bool)
for i in range(lb*2,len(df)):
    w=df['high'].iloc[i-lb*2:i+1];m=i-lb
    if df['high'].iloc[m]==w.max() and w.values.argmax()==lb:swh[i]=True
    w=df['low'].iloc[i-lb*2:i+1]
    if df['low'].iloc[m]==w.min() and w.values.argmax()==lb:swl[i]=True
def nsl(i):
    for j in range(i-1,max(0,i-100),-1):
        if swl[j]:return df['low'].iloc[j]
    return df['low'].iloc[i]*0.95
def nsh(i):
    for j in range(i-1,max(0,i-100),-1):
        if swh[j]:return df['high'].iloc[j]
    return df['high'].iloc[i]*1.05

# Test each MA period
for pma_len in [25, 50, 100, 200]:
    pma=df['close'].rolling(pma_len).mean()
    price_up=df['close']>pma;price_down=df['close']<pma
    
    long_entry=(df['spike']&(df['wstr']>50)&(df['w50']>df['w200'])&price_up&(df['volume']>df['vma'])&(df['atr']>df['atr'].rolling(20).mean()))
    short_entry=(df['spike']&(df['wstr']>50)&(df['w50']<df['w200'])&price_down&(df['volume']>df['vma'])&(df['atr']>df['atr'].rolling(20).mean()))
    
    all_eis=[]
    for i in np.where(long_entry)[0]:
        if i>=500:all_eis.append((i,'LONG'))
    for i in np.where(short_entry)[0]:
        if i>=500:all_eis.append((i,'SHORT'))
    all_eis.sort()
    
    trades=[];it=False;ed=0;equity=1000;pos=25
    cmon=df['ts'].iloc[500].month;cyr=df['ts'].iloc[500].year;mstart=1000
    
    for ei,dirn in all_eis:
        if it and ei<ed:continue
        ts=df['ts'].iloc[ei]
        if ts.month!=cmon or ts.year!=cyr:cmon,cyr=ts.month,ts.year;mstart=equity
        if (equity-mstart)/mstart*100<=-7:continue
        is_long=dirn=='LONG';e=df['close'].iloc[ei]
        if is_long:sl=nsl(ei)*0.998
        else:sl=nsh(ei)*1.002
        end=min(ei+192,len(df));r=None;ep=e;ex=ei
        for j in range(ei+1,end):
            if is_long:
                if df['low'].iloc[j]<=sl:r='SL';ep=sl;ex=j;break
                if df['sell'].iloc[j]>=60:r='SELL';ep=df['close'].iloc[j];ex=j;break
            else:
                if df['high'].iloc[j]>=sl:r='SL';ep=sl;ex=j;break
                if df['buy_sig'].iloc[j]>=60:r='BUY';ep=df['close'].iloc[j];ex=j;break
        if not r:r='TIME';ep=df['close'].iloc[end-1];ex=end-1
        pnl=(ep-e)/e*100-FEE*200
        if not is_long:pnl=-pnl
        dollar=equity*(pos/100)*(pnl/100);equity+=dollar
        trades.append({'dir':dirn,'pnl':pnl,'r':r,'eq':equity})
        it=True;ed=ex
    
    n=len(trades)
    if n<5:continue
    wins=[t for t in trades if t['pnl']>0];nw=len(wins);wr=nw/n*100
    aw=np.mean([t['pnl'] for t in wins]) if wins else 0
    aloss=np.mean([t['pnl'] for t in trades if t['pnl']<=0]) if (n-nw) else 0
    rr=abs(aw/aloss) if aloss else 0
    lt=[t for t in trades if t['dir']=='LONG'];st=[t for t in trades if t['dir']=='SHORT']
    lwr=len([t for t in lt if t['pnl']>0])/len(lt)*100 if lt else 0
    swr=len([t for t in st if t['pnl']>0])/len(st)*100 if st else 0
    eqs=[1000]
    for t in trades:eqs.append(t['eq'])
    peak=np.maximum.accumulate(eqs);dd=(np.array(eqs)-peak)/peak*100
    
    # Last year
    ycut=df['ts'].max()-pd.Timedelta(days=365)
    yt=[t for t in trades if df['ts'].iloc[t['ei'] if 'ei' in t else 0]>=ycut] if False else []
    # Simplified: just report overall
    print(f"MA{pma_len}: {n}T | WR:{wr:.0f}% | ${equity:,.0f} | L:{len(lt)}T/{lwr:.0f}% | S:{len(st)}T/{swr:.0f}% | DD:{dd.min():.1f}% | R:R:{rr:.1f}x", flush=True)
