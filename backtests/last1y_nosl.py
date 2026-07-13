#!/usr/bin/env python3
"""Last 1 year — no SL, sell signal exit only"""
import pandas as pd, numpy as np

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv', parse_dates=['ts'])
cutoff = df['ts'].max() - pd.Timedelta(days=365)
tdf = df[df['ts'] >= cutoff].copy().reset_index(drop=True)
s0, s1 = tdf['close'].iloc[0], tdf['close'].iloc[-1]
chg = (s1/s0-1)*100
print(f"📅 سنة: {tdf['ts'].iloc[0]} → {tdf['ts'].iloc[-1]}")
print(f"💰 ${s0:.4f} → ${s1:.4f} | {'📈' if chg>0 else '📉'} {chg:+.1f}% | شموع: {len(tdf):,}")

# Whale
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

long_ok=tdf['w50']>tdf['w200']
entry_sig=(tdf['spike']&(tdf['wstr']>50)&long_ok&(tdf['volume']>tdf['vma'])&(tdf['atr']>tdf['atr'].rolling(20,min_periods=1).mean()))
eis=np.where(entry_sig)[0]
print(f"🔍 إشارات: {len(eis)}")

# No SL — sell signal only + 48h timeout
trades=[];it=False;ed=0;equity=1000;pos=25
for ei in eis:
    if ei<10:continue
    if it and ei<ed:continue
    e=tdf['close'].iloc[ei]
    end=min(ei+192,len(tdf));r=None;ep=e;ex=ei
    for j in range(ei+1,end):
        if tdf['sell'].iloc[j]>=60:r='SELL';ep=tdf['close'].iloc[j];ex=j;break
    if not r:r='TIME';ep=tdf['close'].iloc[end-1];ex=end-1
    pnl=(ep-e)/e*100-0.2
    dollar=equity*(pos/100)*(pnl/100)
    equity+=dollar
    trades.append({'pnl':pnl,'r':r,'eq':equity,'e':e,'ep':ep,'ets':tdf['ts'].iloc[ei],'xts':tdf['ts'].iloc[ex]})
    it=True;ed=ex

n=len(trades);wins=[t for t in trades if t['pnl']>0];nw=len(wins);nl=n-nw
wr=nw/n*100
tp=sum(t['pnl'] for t in wins)
tl=sum(t['pnl'] for t in trades if t['pnl']<=0)
aw=np.mean([t['pnl'] for t in wins]) if wins else 0
aloss=np.mean([t['pnl'] for t in trades if t['pnl']<=0]) if nl else 0
rr=abs(aw/aloss) if aloss else 0

eqs=[1000]
for t in trades:eqs.append(t['eq'])
peak=np.maximum.accumulate(eqs)
dd=(np.array(eqs)-peak)/peak*100

# Monthly
tdf2=pd.DataFrame(trades)
tdf2['month']=tdf2['ets'].dt.to_period('M')
mo=tdf2.groupby('month')['pnl'].sum()

print(f"\n📋 عدد الصفقات: {n}")
print(f"🟢 صفقات رابحة: {nw} | 🔴 صفقات خاسرة: {nl}")
print(f"📈 Win Rate: {wr:.0f}%")
print(f"💰 صافي: {sum(t['pnl'] for t in trades):+.1f}%")
print(f"🟢 متوسط الربح: +{aw:.2f}%")
print(f"🔴 متوسط الخسارة: {aloss:.2f}%")
print(f"📊 R:R: {rr:.1f}x")
print(f"📉 أقصى انخفاض: {dd.min():.1f}%")
print(f"🏦 المحفظة: $1000 → ${equity:,.0f} ({(equity/1000-1)*100:+.1f}%)")
print(f"🏆 أفضل شهر: +{mo.max():.1f}% | 💀 أسوأ شهر: {mo.min():.1f}%")

print(f"\n📋 كل الصفقات:")
print(f"{'تاريخ':<19} {'دخول':>7} {'خروج':>7} {'نوع':<5} {'ربح%':>7} {'محفظة':>8}")
print("-"*55)
for t in trades:
    em="🟢" if t['pnl']>0 else "🔴"
    print(f"{em}{str(t['ets'])[:19]:<18} {t['e']:>7.4f} {t['ep']:>7.4f} {t['r']:<5} {t['pnl']:>+6.2f}% ${t['eq']:>7,.0f}")
