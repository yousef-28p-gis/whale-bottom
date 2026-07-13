#!/usr/bin/env python3
"""Last 30 days test — step by step"""
import pandas as pd, numpy as np

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv', parse_dates=['ts'])
cutoff = df['ts'].max() - pd.Timedelta(days=30)
tdf = df[df['ts'] >= cutoff].copy().reset_index(drop=True)
print(f"📅 آخر 30 يوم: {tdf['ts'].iloc[0]} → {tdf['ts'].iloc[-1]} | شموع: {len(tdf)}")
print(f"💰 نطاق السعر: ${tdf['low'].min():.4f} → ${tdf['high'].max():.4f} | الحالي: ${tdf['close'].iloc[-1]:.4f}")

# Whale 200
B=200
lo=tdf['low'].rolling(B,min_periods=1).min()
al=(tdf['low']<=lo).astype(float)
lc=abs(tdf['low']-tdf['low'].shift(1))/tdf['low']*100
sm=lc.ewm(span=3,adjust=False).mean()
hi=sm.rolling(B,min_periods=1).max()
st=np.where(al>0,(sm+hi*2)/3,0)
tdf['w']=pd.Series(st).ewm(span=3,adjust=False).mean().fillna(0)
tdf['spike']=(tdf['w']>tdf['w'].shift(1))&(tdf['w'].shift(1)<=0.02)
tdf['w50']=tdf['w'].rolling(50,min_periods=1).mean()
tdf['w200']=tdf['w'].rolling(200,min_periods=1).mean()
tdf['wstr']=tdf['w']/tdf['w'].rolling(50,min_periods=1).max().replace(0,np.nan)*100
tdf['atr']=(tdf['high']-tdf['low']).rolling(14).mean()
tdf['vma']=tdf['volume'].rolling(20,min_periods=1).mean()

delta=tdf['close'].diff();g=delta.clip(lower=0);l=-delta.clip(upper=0)
ag=g.ewm(alpha=1/14,adjust=False).mean();al=l.ewm(alpha=1/14,adjust=False).mean()
tdf['rsi']=100-(100/(1+ag/al.replace(0,np.nan)))
vs=tdf['volume'].rolling(20,min_periods=1).mean()
hh20=tdf['high'].rolling(20,min_periods=1).max().shift(1)
ll10=tdf['low'].rolling(10,min_periods=1).min().shift(1)
c=np.zeros(len(tdf))
c+=((tdf['volume']>vs*1.5)&(tdf['close']<tdf['open'])).astype(int)
c+=((tdf['high']>hh20)&(tdf['close']<hh20)).astype(int)
c+=((tdf['high']>hh20)&(tdf['close']<tdf['open'])).astype(int)
c+=((tdf['close'].shift(1)>tdf['open'].shift(1))&(tdf['volume']>vs*1.5)&(tdf['close']<tdf['open'])).astype(int)
c+=(tdf['low']<ll10).astype(int)
c+=((tdf['high']>tdf['high'].shift(1))&(tdf['rsi']<tdf['rsi'].shift(1))).astype(int)
tdf['sell']=c/6*100

lb=5;swl=np.zeros(len(tdf),dtype=bool)
for i in range(lb*2,len(tdf)):
    w=tdf['low'].iloc[i-lb*2:i+1];m=i-lb
    if tdf['low'].iloc[m]==w.min() and w.values.argmax()==lb:swl[i]=True
def nsl(idx):
    for j in range(idx-1,max(0,idx-100),-1):
        if swl[j]:return tdf['low'].iloc[j]
    return tdf['low'].iloc[idx]*0.95

# Entry signals
long_ok=tdf['w50']>tdf['w200']
entry_sig=(tdf['spike']&(tdf['wstr']>50)&long_ok&(tdf['volume']>tdf['vma'])&(tdf['atr']>tdf['atr'].rolling(20,min_periods=1).mean()))
eis=np.where(entry_sig)[0]
print(f"\n🔍 إشارات الدخول: {len(eis)}")

if len(eis)>0:
    print(f"\n📋 تفاصيل الإشارات:")
    for ei in eis:
        if ei<10:continue
        t=tdf['ts'].iloc[ei];px=tdf['close'].iloc[ei];ws=tdf['wstr'].iloc[ei]
        wh=tdf['w'].iloc[ei];vo=tdf['volume'].iloc[ei]>tdf['vma'].iloc[ei]
        ao=tdf['atr'].iloc[ei]>tdf['atr'].rolling(20,min_periods=1).mean().iloc[ei]
        print(f"  {t} | ${px:.4f} | قوة:{ws:.0f}% | حوت:{wh:.3f} | حجم:{'✅' if vo else '❌'} | ATR:{'✅' if ao else '❌'}")

# Simulation
trades=[];it=False;ed=0;equity=1000;pos=25
for ei in eis:
    if ei<10:continue
    if it and ei<ed:continue
    e=tdf['close'].iloc[ei];sl=nsl(ei)*0.998
    end=min(ei+192,len(tdf));r=None;ep=e;ex=ei
    for j in range(ei+1,end):
        if tdf['low'].iloc[j]<=sl:r='SL';ep=sl;ex=j;break
        if tdf['sell'].iloc[j]>=60:r='SELL';ep=tdf['close'].iloc[j];ex=j;break
    if not r:r='TIME';ep=tdf['close'].iloc[end-1];ex=end-1
    pnl_pct=(ep-e)/e*100-0.2
    dollar=equity*(pos/100)*(pnl_pct/100)
    equity+=dollar
    trades.append({'ets':tdf['ts'].iloc[ei],'e':e,'xts':tdf['ts'].iloc[ex],'xp':ep,'r':r,'pnl':pnl_pct,'d':dollar,'sl':sl,'eq':equity})
    it=True;ed=ex

n=len(trades)
if n>0:
    wins=[t for t in trades if t['pnl']>0]
    print(f"\n📊 نتائج التداول ({pos}% من المحفظة):")
    print(f"{'تاريخ':<19} {'دخول':>7} {'SL':>7} {'خروج':>7} {'نتيجة':<5} {'ربح%':>7} {'ربح$':>8} {'محفظة':>8}")
    print("-"*75)
    for t in trades:
        em="🟢" if t['pnl']>0 else "🔴"
        print(f"{em}{str(t['ets'])[:19]:<18} {t['e']:>7.4f} {t['sl']:>7.4f} {t['xp']:>7.4f} {t['r']:<5} {t['pnl']:>+6.2f}% ${t['d']:>+7.2f} ${t['eq']:>7,.0f}")
    print(f"\n🏦 المحفظة: $1000 → ${equity:,.0f} ({(equity/1000-1)*100:+.1f}%) | صفقات: {n} | ربح: {len(wins)}/{n} ({len(wins)/n*100:.0f}%)")
else:
    print("\n⚠️ لا توجد صفقات في آخر 30 يوم")
