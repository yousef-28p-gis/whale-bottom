#!/usr/bin/env python3
"""Fresh start: Whale 200 + wMA200 only — LONG only"""
import pandas as pd, numpy as np

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv', parse_dates=['ts'])
B=200; FEE=0.001

# Whale
lo=df['low'].rolling(B).min()
al=(df['low']<=lo).astype(float)
lc=abs(df['low']-df['low'].shift(1))/df['low']*100
sm=lc.ewm(span=3,adjust=False).mean()
hi=sm.rolling(B).max()
st=np.where(al>0,(sm+hi*2)/3,0)
df['w']=pd.Series(st).ewm(span=3,adjust=False).mean().fillna(0)
df['spike']=(df['w']>df['w'].shift(1))&(df['w'].shift(1)<=0.02)
df['w50']=df['w'].rolling(50).mean()
df['w200']=df['w'].rolling(200).mean()

# Sell signal only
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
c+=(df['low']<ll10).astype(int)
c+=((df['high']>df['high'].shift(1))&(df['rsi']<df['rsi'].shift(1))).astype(int)
df['sell']=c/6*100

# Entry: ONLY whale spike + w50 > w200. Nothing else.
long_ok=df['w50']>df['w200']
entry_sig=df['spike'] & long_ok
eis=np.where(entry_sig)[0]

print(f"📊 {len(df):,} candles | {df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}")
print(f"🐋 Signals: {len(eis)} (LONG only, whale spike + wMA50>wMA200)")

# Simulate: no SL, sell signal exit, 48h timeout, 25% position
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
    trades.append({'pnl':pnl,'r':r,'eq':equity,'e':e,'ep':ep,'ets':df['ts'].iloc[ei]})
    it=True;ed=ex

# Results
n=len(trades);wins=[t for t in trades if t['pnl']>0];nw=len(wins);nl=n-nw
wr=nw/n*100
tp=sum(t['pnl'] for t in wins);tl=abs(sum(t['pnl'] for t in trades if t['pnl']<=0))
aw=np.mean([t['pnl'] for t in wins]) if wins else 0
aloss=np.mean([t['pnl'] for t in trades if t['pnl']<=0]) if nl else 0
rr=abs(aw/aloss) if aloss else 0
eqs=[1000]
for t in trades:eqs.append(t['eq'])
peak=np.maximum.accumulate(eqs);dd=(np.array(eqs)-peak)/peak*100

# By year
tdf=pd.DataFrame(trades);tdf['year']=tdf['ets'].dt.year

print(f"\n📋 {n} صفقة | 🟢{nw} 🔴{nl} | WR:{wr:.0f}%")
print(f"🟢 +{aw:.2f}% | 🔴 {aloss:.2f}% | R:R:{rr:.1f}x")
print(f"🏦 ${equity:,.0f} ({(equity/1000-1)*100:+.1f}%) | DD:{dd.min():.1f}%")
print(f"أفضل سنة: +{tdf.groupby('year')['pnl'].sum().max():.0f}% | أسوأ سنة: {tdf.groupby('year')['pnl'].sum().min():.0f}%")

print(f"\n📊 حسب السنة:")
for yr in sorted(tdf['year'].unique()):
    yt=tdf[tdf['year']==yr];ym=df[(df['ts'].dt.year==yr)]
    s0=ym['close'].iloc[0];s1=ym['close'].iloc[-1];chg=(s1/s0-1)*100
    n=len(yt);w=len(yt[yt['pnl']>0]);wr_y=w/n*100;net=sum(yt['pnl'])
    arrow="📈" if chg>0 else "📉";emoji="🟢" if net>0 else "🔴"
    print(f"  {yr}: {arrow} FET {chg:+.0f}% | {emoji} {n}T WR:{wr_y:.0f}% | صافي:{net:+.0f}%")
