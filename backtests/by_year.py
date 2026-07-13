#!/usr/bin/env python3
"""Performance by market regime"""
import pandas as pd, numpy as np

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv', parse_dates=['ts'])
B=200; FEE=0.001

lo=df['low'].rolling(B,min_periods=1).min();al=(df['low']<=lo).astype(float)
lc=abs(df['low']-df['low'].shift(1))/df['low']*100;sm=lc.ewm(span=3,adjust=False).mean()
hi=sm.rolling(B,min_periods=1).max();st=np.where(al>0,(sm+hi*2)/3,0)
df['w']=pd.Series(st).ewm(span=3,adjust=False).mean().fillna(0)
df['spike']=(df['w']>df['w'].shift(1))&(df['w'].shift(1)<=0.02)
df['w50']=df['w'].rolling(50,min_periods=1).mean();df['w200']=df['w'].rolling(200,min_periods=1).mean()
df['wstr']=df['w']/df['w'].rolling(50,min_periods=1).max().replace(0,np.nan)*100
df['atr']=(df['high']-df['low']).rolling(14).mean();df['vma']=df['volume'].rolling(20,min_periods=1).mean()
delta=df['close'].diff();g=delta.clip(lower=0);l=-delta.clip(upper=0)
ag=g.ewm(alpha=1/14,adjust=False).mean();al=l.ewm(alpha=1/14,adjust=False).mean()
df['rsi']=100-(100/(1+ag/al.replace(0,np.nan)))
vs=df['volume'].rolling(20,min_periods=1).mean();hh20=df['high'].rolling(20).max().shift(1)
ll10=df['low'].rolling(10).min().shift(1)
c=np.zeros(len(df))
c+=((df['volume']>vs*1.5)&(df['close']<df['open'])).astype(int)
c+=((df['high']>hh20)&(df['close']<hh20)).astype(int)
c+=((df['high']>hh20)&(df['close']<df['open'])).astype(int)
c+=((df['close'].shift(1)>df['open'].shift(1))&(df['volume']>vs*1.5)&(df['close']<df['open'])).astype(int)
c+=(df['low']<ll10).astype(int);c+=((df['high']>df['high'].shift(1))&(df['rsi']<df['rsi'].shift(1))).astype(int)
df['sell']=c/6*100
lb=5;swl=np.zeros(len(df),dtype=bool)
for i in range(lb*2,len(df)):
    w=df['low'].iloc[i-lb*2:i+1];m=i-lb
    if df['low'].iloc[m]==w.min() and w.values.argmax()==lb:swl[i]=True
def nsl(i):
    for j in range(i-1,max(0,i-100),-1):
        if swl[j]:return df['low'].iloc[j]
    return df['low'].iloc[i]*0.95

long_ok=df['w50']>df['w200']
entry_sig=(df['spike']&(df['wstr']>50)&long_ok&(df['volume']>df['vma'])&(df['atr']>df['atr'].rolling(20,min_periods=1).mean()))
eis=np.where(entry_sig)[0]

all_trades=[];it=False;ed=0
for ei in eis:
    if ei<500:continue
    if it and ei<ed:continue
    e=df['close'].iloc[ei]
    end=min(ei+192,len(df));r=None;ep=e;ex=ei
    for j in range(ei+1,end):
        if df['sell'].iloc[j]>=60:r='SELL';ep=df['close'].iloc[j];ex=j;break
    if not r:r='TIME';ep=df['close'].iloc[end-1];ex=end-1
    pnl=(ep-e)/e*100-0.2
    all_trades.append({'pnl':pnl,'ets':df['ts'].iloc[ei],'e':e,'ep':ep,'r':r})
    it=True;ed=ex

tdf=pd.DataFrame(all_trades)
tdf['year']=tdf['ets'].dt.year

print("📊 الأداء حسب السنة:\n")
for yr in sorted(tdf['year'].unique()):
    yt=tdf[tdf['year']==yr];ym=df[(df['ts'].dt.year==yr)]
    s0=ym['close'].iloc[0];s1=ym['close'].iloc[-1];chg=(s1/s0-1)*100
    n=len(yt);w=len(yt[yt['pnl']>0]);wr=w/n*100;net=sum(yt['pnl'])
    arrow="📈" if chg>0 else "📉";emoji="🟢" if net>0 else "🔴"
    print(f"  {yr}: {arrow} FET {chg:+.0f}% | {emoji} {n}T WR:{wr:.0f}% | صافي:{net:+.1f}%")

# Group by regime
for label, cond in [("🐂 صعود 2021", tdf['year']==2021), ("🐂 صعود 2023-24", tdf['year'].isin([2023,2024])),
                     ("🐻 هبوط 2022", tdf['year']==2022), ("🐻 هبوط 2025-26", tdf['year'].isin([2025,2026]))]:
    yt=tdf[cond]
    if len(yt)==0:continue
    n=len(yt);w=len(yt[yt['pnl']>0]);wr=w/n*100;net=sum(yt['pnl'])
    aw=np.mean(yt[yt['pnl']>0]['pnl']) if w else 0;aloss=np.mean(yt[yt['pnl']<=0]['pnl']) if (n-w) else 0
    print(f"\n  {label}: {n}T WR:{wr:.0f}% | صافي:{net:+.1f}% | AvgW:{aw:+.1f}% AvgL:{aloss:+.1f}%")
