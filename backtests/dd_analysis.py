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
df['atr']=(df['high']-df['low']).rolling(14).mean()
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

entry_sig=df['spike'] & (df['w20']>df['w50']) & (df['close']>df['sma50d'])
eis=np.where(entry_sig)[0]

trades=[];it=False;ed=0;equity=1000;peak_eq=1000
cmon=df['ts'].iloc[500].month;cyr=df['ts'].iloc[500].year;mstart=1000
for ei in eis:
    if ei<500:continue
    if it and ei<ed:continue
    ts=df['ts'].iloc[ei]
    if ts.month!=cmon or ts.year!=cyr:cmon,cyr=ts.month,ts.year;mstart=equity
    if (equity-mstart)/mstart*100<=-7:continue
    e=df['close'].iloc[ei];tp=e+df['atr'].iloc[ei]*3
    end=min(ei+192,len(df));r=None;ep=e;ex=ei
    for j in range(ei+1,end):
        if df['high'].iloc[j]>=tp:r='TP';ep=tp;ex=j;break
        if df['sell'].iloc[j]>=60:r='SELL';ep=df['close'].iloc[j];ex=j;break
    if not r:r='TIME';ep=df['close'].iloc[end-1];ex=end-1
    pnl=(ep-e)/e*100-0.2
    equity+=equity*(pnl/100)
    if equity>peak_eq:peak_eq=equity
    dd_now=(equity-peak_eq)/peak_eq*100
    trades.append(dict(pnl=pnl,r=r,eq=equity,peak=peak_eq,dd=dd_now,ets=df['ts'].iloc[ei],xts=df['ts'].iloc[ex],e=e,ep=ep))
    it=True;ed=ex

worst=max(trades,key=lambda t:abs(t['dd']))
print("=" * 50)
print(f"📉 اقصى انخفاض: {worst['dd']:.1f}%")
print(f"   القمة: ${worst['peak']:,.0f} -> القاع: ${worst['eq']:,.0f}")
lost = worst['peak'] - worst['eq']
print(f"   خسرت: ${lost:,.0f}")
print(f"   التاريخ: {str(worst['ets'])[:16]} | {worst['r']} | {worst['e']:.4f}->{worst['ep']:.4f} | {worst['pnl']:+.1f}%")

trades_df=pd.DataFrame(trades)
trades_df['year']=trades_df['ets'].dt.year
print("\n📊 اقصى انخفاض سنوي:")
for yr in sorted(trades_df['year'].unique()):
    yt=trades_df[trades_df['year']==yr]
    print(f"  {yr}: {yt['dd'].min():.0f}%")

print("\nاسوا 5 فترات:")
for t in sorted(trades,key=lambda t:t['dd'])[:5]:
    print(f"  {t['dd']:+.0f}% | {str(t['ets'])[:16]} | {t['r']} | {t['pnl']:+.1f}% | قمة${t['peak']:,.0f}->${t['eq']:,.0f}")
