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
df['atr_ma']=df['atr'].rolling(20).mean()
df['vol_ma']=df['volume'].rolling(20).mean()
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

base = (df['w20']>df['w50']) & (df['close']>df['sma50d']) & df['spike']

def run(name, extra_filter, ml=7):
    entry = base & extra_filter
    eis = np.where(entry)[0]
    trades=[];it=False;ed=0;equity=1000;peak_eq=1000
    cmon=df['ts'].iloc[500].month;cyr=df['ts'].iloc[500].year;mstart=1000
    cons_losses=0
    for ei in eis:
        if ei<500:continue
        if it and ei<ed:continue
        ts=df['ts'].iloc[ei]
        if ts.month!=cmon or ts.year!=cyr:cmon,cyr=ts.month,ts.year;mstart=equity;cons_losses=0
        if (equity-mstart)/mstart*100<=-ml:continue
        e=df['close'].iloc[ei];tp=e+df['atr'].iloc[ei]*3
        end=min(ei+192,len(df));r=None;ep=e;ex=ei
        for j in range(ei+1,end):
            if df['high'].iloc[j]>=tp:r='TP';ep=tp;ex=j;break
            if df['sell'].iloc[j]>=60:r='SELL';ep=df['close'].iloc[j];ex=j;break
        if not r:r='TIME';ep=df['close'].iloc[end-1];ex=end-1
        pnl=(ep-e)/e*100-0.2;equity+=equity*(pnl/100)
        if pnl<=0:cons_losses+=1
        else:cons_losses=0
        if equity>peak_eq:peak_eq=equity
        trades.append(dict(pnl=pnl,eq=equity))
        it=True;ed=ex
    n=len(trades)
    if n<5:return None
    wins=[t for t in trades if t['pnl']>0];nw=len(wins)
    eqs=[1000]
    for t in trades:eqs.append(t['eq'])
    peak=np.maximum.accumulate(eqs);dd_vals=(np.array(eqs)-peak)/peak*100
    return dict(name=name,n=n,wr=nw/n*100,eq=equity,dd=dd_vals.min())

results=[run('الان (بدون فلاتر اضافية)', True)]

# Strength filters
results.append(run('قوة>50%', df['wstr']>50))
results.append(run('قوة>70%', df['wstr']>70))

# Volume filters
results.append(run('حجم>1.5x', df['volume']>df['vol_ma']*1.5))
results.append(run('حجم>2x', df['volume']>df['vol_ma']*2))

# ATR filter (not too extreme)
results.append(run('ATR>متوسط', df['atr']>df['atr_ma']))

# Combined
results.append(run('قوة>50%+حجم>1.5x', (df['wstr']>50)&(df['volume']>df['vol_ma']*1.5)))
results.append(run('قوة>70%+حجم>1.5x', (df['wstr']>70)&(df['volume']>df['vol_ma']*1.5)))
results.append(run('قوة>50%+حجم>2x', (df['wstr']>50)&(df['volume']>df['vol_ma']*2)))
results.append(run('قوة>70%+ATR>متوسط', (df['wstr']>70)&(df['atr']>df['atr_ma'])))

results = [r for r in results if r]

print(f"{'الفلتر':<28} {'T':>4} {'WR':>4} {'محفظة':>8} {'DD':>6} {'تغييرDD':>8}")
print("-"*65)
base_dd = results[0]['dd']
for r in sorted(results,key=lambda x:x['eq']-abs(x['dd'])*10,reverse=True):
    dd_chg = round(r['dd'] - base_dd, 1)
    dd_str = f"+{dd_chg}%" if dd_chg > 0 else f"{dd_chg}%"
    print(f"{r['name']:<28} {r['n']:>4} {r['wr']:>3.0f}% ${r['eq']:>7,.0f} {r['dd']:>5.1f}% {dd_str:>8}")
