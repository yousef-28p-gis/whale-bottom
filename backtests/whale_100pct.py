#!/usr/bin/env python3
"""Final strategy: 100% compounding"""
import pandas as pd, numpy as np

CACHE='/data/trading28/backtests/cache'; FEE=0.001; B=200

ddf=pd.read_csv(f"{CACHE}/FET_USDT_1d.csv",parse_dates=['ts'])
ddf['date']=ddf['ts'].dt.date;ddf['sma50']=ddf['close'].rolling(50).mean()
df=pd.read_csv(f"{CACHE}/FET_USDT_15m_FULL.csv",parse_dates=['ts'])
df['date']=df['ts'].dt.date;df['sma50d']=df['date'].map(ddf.set_index('date')['sma50'].to_dict())

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

entry_sig=df['spike'] & (df['w20']>df['w50']) & (df['close']>df['sma50d'])
eis=np.where(entry_sig)[0]
print(f"🐋 Signals: {len(eis)} | 100% compounding | TP=3ATR + بيع≥60%")

trades=[];it=False;ed=0;equity=1000
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
    
    pnl=(ep-e)/e*100-FEE*200
    dollar=equity*(pnl/100)  # 100% of equity
    equity+=dollar
    trades.append({'pnl':pnl,'r':r,'eq':equity,'ets':df['ts'].iloc[ei],'xts':df['ts'].iloc[ex],'e':e,'ep':ep})
    it=True;ed=ex

n=len(trades);wins=[t for t in trades if t['pnl']>0];nw=len(wins)
wr=nw/n*100 if n else 0
aw=np.mean([t['pnl'] for t in wins]) if wins else 0
aloss=np.mean([t['pnl'] for t in trades if t['pnl']<=0]) if (n-nw) else 0
rr=abs(aw/aloss) if aloss else 0
eqs=[1000]
for t in trades:eqs.append(t['eq'])
peak=np.maximum.accumulate(eqs);dd=(np.array(eqs)-peak)/peak*100

tdf2=pd.DataFrame(trades);tdf2['year']=tdf2['ets'].dt.year

print(f"\n📋 {n} صفقة | 🟢{nw} 🔴{n-nw} | WR:{wr:.0f}%")
print(f"🟢 +{aw:.2f}% | 🔴 {aloss:.2f}% | R:R:{rr:.1f}x")
print(f"🏦 $1,000 → ${equity:,.0f} ({(equity/1000-1)*100:+.1f}%) | DD:{dd.min():.1f}%")

print(f"\n📊 حسب السنة (100% مركب):")
for yr in sorted(tdf2['year'].unique()):
    yt=tdf2[tdf2['year']==yr];ym=df[(df['ts'].dt.year==yr)]
    s0=ym['close'].iloc[0];s1=ym['close'].iloc[-1];chg=(s1/s0-1)*100
    yn=len(yt);yw=len(yt[yt['pnl']>0]);ywr=yw/yn*100 if yn else 0
    
    # Compound within year
    yeq=1000
    for _,t in yt.iterrows():yeq+=yeq*(t['pnl']/100)
    
    arrow="📈" if chg>0 else "📉";emoji="🟢" if yeq>1000 else "🔴"
    print(f"  {yr}: {arrow} FET {chg:+.0f}% | {emoji} {yn}T WR:{ywr:.0f}% | ${yeq:,.0f} ({(yeq/1000-1)*100:+.0f}%)")

print(f"\n📋 آخر 5 صفقات:")
for t in trades[-5:]:
    em="🟢" if t['pnl']>0 else "🔴"
    print(f"  {em} {str(t['ets'])[:16]} → {str(t['xts'])[:16]} | {t['r']:<4} | {t['e']:.4f}→{t['ep']:.4f} | {t['pnl']:+.2f}% | ${t['eq']:,.0f}")
