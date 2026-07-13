import pandas as pd, numpy as np
CACHE='/data/trading28/backtests/cache';FEE=0.001;B=200

ddf=pd.read_csv(f'{CACHE}/FET_USDT_1d.csv',parse_dates=['ts'])
ddf['date']=ddf['ts'].dt.date;ddf['sma50']=ddf['close'].rolling(50).mean().shift(1)
df=pd.read_csv(f'{CACHE}/FET_USDT_15m_FULL.csv',parse_dates=['ts'])
df['date']=df['ts'].dt.date;df['sma50d']=df['date'].map(ddf.set_index('date')['sma50'].to_dict())

lo=df['low'].rolling(B).min();al=(df['low']<=lo).astype(float)
lc=abs(df['low']-df['low'].shift(1))/df['low']*100;sm=lc.ewm(span=3,adjust=False).mean()
hi=sm.rolling(B).max();st=np.where(al>0,(sm+hi*2)/3,0)
df['w']=pd.Series(st).ewm(span=3,adjust=False).mean().fillna(0)
df['spike']=(df['w']>df['w'].shift(1))&(df['w'].shift(1)<=0.02)
df['w20']=df['w'].rolling(20).mean();df['w50']=df['w'].rolling(50).mean()
df['wstr']=df['w']/df['w'].rolling(50).max().replace(0,np.nan)*100
df['atr']=(df['high']-df['low']).rolling(14).mean()
delta=df['close'].diff();g=delta.clip(lower=0);l=-delta.clip(upper=0)
ag=g.ewm(alpha=1/14,adjust=False).mean();al=l.ewm(alpha=1/14,adjust=False).mean()
df['rsi']=100-(100/(1+ag/al.replace(0,np.nan)))
vs=df['volume'].rolling(20).mean();hh20=df['high'].rolling(20).max().shift(1)
ll20=df['low'].rolling(20).min().shift(1);ll10=df['low'].rolling(10).min().shift(1)
hh10=df['high'].rolling(10).max().shift(1)

# Sell signal
c=np.zeros(len(df))
c+=((df['volume']>vs*1.5)&(df['close']<df['open'])).astype(int)
c+=((df['high']>hh20)&(df['close']<hh20)).astype(int)
c+=((df['high']>hh20)&(df['close']<df['open'])).astype(int)
c+=((df['close'].shift(1)>df['open'].shift(1))&(df['volume']>vs*1.5)&(df['close']<df['open'])).astype(int)
c+=(df['low']<ll10).astype(int);c+=((df['high']>df['high'].shift(1))&(df['rsi']<df['rsi'].shift(1))).astype(int)
df['sell']=c/6*100

# Buy signal (opposite for SHORT exit)
c2=np.zeros(len(df))
c2+=((df['volume']>vs*1.5)&(df['close']>df['open'])).astype(int)
c2+=((df['low']<ll20)&(df['close']>ll20)).astype(int)
c2+=((df['low']<ll20)&(df['close']>df['open'])).astype(int)
c2+=((df['close'].shift(1)<df['open'].shift(1))&(df['volume']>vs*1.5)&(df['close']>df['open'])).astype(int)
c2+=(df['high']>hh10).astype(int);c2+=((df['low']<df['low'].shift(1))&(df['rsi']>df['rsi'].shift(1))).astype(int)
df['buy_sig']=c2/6*100

def run(name, use_short, ws, ml=7):
    # LONG: above SMA50d + whale trend up
    long_ok = (df['w20']>df['w50']) & (df['close']>df['sma50d'])
    long_entry = df['spike'] & long_ok & (df['wstr']>ws)
    
    if use_short:
        # SHORT: below SMA50d + whale trend down
        short_ok = (df['w20']<df['w50']) & (df['close']<df['sma50d'])
        short_entry = df['spike'] & short_ok & (df['wstr']>ws)
    else:
        short_entry = pd.Series(False, index=df.index)
    
    all_eis = []
    for i in np.where(long_entry)[0]:
        if i>=500: all_eis.append((i,'LONG'))
    for i in np.where(short_entry)[0]:
        if i>=500: all_eis.append((i,'SHORT'))
    all_eis.sort()
    
    trades=[];it=False;ed=0;equity=1000;peak_eq=1000
    cmon=df['ts'].iloc[500].month;cyr=df['ts'].iloc[500].year;mstart=1000
    for ei,dirn in all_eis:
        if it and ei<ed:continue
        ts=df['ts'].iloc[ei]
        if ts.month!=cmon or ts.year!=cyr:cmon,cyr=ts.month,ts.year;mstart=equity
        if (equity-mstart)/mstart*100<=-ml:continue
        is_long=dirn=='LONG';e=df['close'].iloc[ei]
        if is_long:tp=e+df['atr'].iloc[ei]*3
        else:tp=e-df['atr'].iloc[ei]*3
        end=min(ei+192,len(df));r=None;ep=e;ex=ei
        for j in range(ei+1,end):
            if is_long:
                if df['high'].iloc[j]>=tp:r='TP';ep=tp;ex=j;break
                if df['sell'].iloc[j]>=60:r='SELL';ep=df['close'].iloc[j];ex=j;break
            else:
                if df['low'].iloc[j]<=tp:r='TP';ep=tp;ex=j;break
                if df['buy_sig'].iloc[j]>=60:r='BUY';ep=df['close'].iloc[j];ex=j;break
        if not r:r='TIME';ep=df['close'].iloc[end-1];ex=end-1
        pnl=(ep-e)/e*100-0.2
        if not is_long:pnl=-pnl
        equity+=equity*(pnl/100)
        if equity>peak_eq:peak_eq=equity
        trades.append(dict(pnl=pnl,r=r,dir=dirn,eq=equity))
        it=True;ed=ex
    
    n=len(trades)
    if n<5:return None
    wins=[t for t in trades if t['pnl']>0];nw=len(wins)
    lt=[t for t in trades if t['dir']=='LONG'];st=[t for t in trades if t['dir']=='SHORT']
    lwr=len([t for t in lt if t['pnl']>0])/len(lt)*100 if lt else 0
    swr=len([t for t in st if t['pnl']>0])/len(st)*100 if st else 0
    eqs=[1000]
    for t in trades:eqs.append(t['eq'])
    peak=np.maximum.accumulate(eqs);dd=(np.array(eqs)-peak)/peak*100
    
    # Yearly
    tdf=pd.DataFrame(trades);tdf['yr']=tdf['r'].apply(lambda x:0)
    for i,t in enumerate(trades):
        t['yr'] = 0  # skip for now
    
    return dict(name=name,n=n,wr=nw/n*100,eq=equity,dd=dd.min(),
                lwr=lwr,swr=swr,ln=len(lt),sn=len(st))

print(f"{'الاستراتيجية':<32} {'T':>4} {'WR':>4} {'محفظة':>8} {'DD':>6} {'L/S':>12}")
print("-"*72)

for name, use_short, ws in [
    ("LONG فقط (الحالي)", False, 70),
    ("LONG+SHORT", True, 70),
    ("LONG+SHORT قوة>50%", True, 50),
]:
    r=run(name, use_short, ws)
    if r:
        ls=f"{r['lwr']:.0f}%/{r['swr']:.0f}% ({r['ln']}/{r['sn']})"
        print(f"{r['name']:<32} {r['n']:>4} {r['wr']:>3.0f}% ${r['eq']:>7,.0f} {r['dd']:>5.1f}% {ls:>12}")
