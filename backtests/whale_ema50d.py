#!/usr/bin/env python3
"""Whale 200 + EMA50 daily filter — LONG only above EMA50"""
import pandas as pd, numpy as np, ccxt, os

CACHE='/data/trading28/backtests/cache'
FEE=0.001; B=200

# ─── Fetch daily for EMA50 ────────────────────────────────────
daily_file=f"{CACHE}/FET_USDT_1d.csv"
if not os.path.exists(daily_file):
    print("📡 Fetching daily data...")
    ex=ccxt.binance(); candles=[]
    since=ex.parse8601('2019-01-01T00:00:00Z')
    while True:
        try:
            c=ex.fetch_ohlcv('FET/USDT','1d',since=since,limit=1000)
            if not c:break
            candles.extend(c); since=c[-1][0]+1
            if len(c)<1000:break
        except:break
    ddf=pd.DataFrame(candles,columns=['ts','open','high','low','close','volume'])
    ddf['ts']=pd.to_datetime(ddf['ts'],unit='ms')
    ddf.to_csv(daily_file,index=False)
else:
    ddf=pd.read_csv(daily_file,parse_dates=['ts'])

ddf['ema50']=ddf['close'].ewm(span=50,adjust=False).mean()
ddf['date']=ddf['ts'].dt.date
print(f"📅 Daily: {len(ddf)} candles | {ddf['ts'].iloc[0].date()} → {ddf['ts'].iloc[-1].date()}")

# ─── 15m data ──────────────────────────────────────────────────
df=pd.read_csv(f"{CACHE}/FET_USDT_15m_FULL.csv",parse_dates=['ts'])
df['date']=df['ts'].dt.date

# Merge daily EMA50 onto 15m
ema_map=ddf.set_index('date')['ema50'].to_dict()
df['dema50']=df['date'].map(ema_map)

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
c+=(df['low']<ll10).astype(int)
c+=((df['high']>df['high'].shift(1))&(df['rsi']<df['rsi'].shift(1))).astype(int)
df['sell']=c/6*100

# Entry: whale spike + w50>w200 + price > daily EMA50
market_ok=df['close']>df['dema50']
entry_sig=df['spike'] & (df['w50']>df['w200']) & market_ok
eis=np.where(entry_sig)[0]

print(f"🐋 Signals: {len(eis)} | Market filter: close > EMA50 daily")
print(f"   Days above EMA50: {(df['close']>df['dema50']).sum()/96:.0f} ({(df['close']>df['dema50']).mean()*100:.0f}%)")

# Simulate
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

n=len(trades);wins=[t for t in trades if t['pnl']>0];nw=len(wins);nl=n-nw
wr=nw/n*100 if n else 0
tp=sum(t['pnl'] for t in wins);tl=abs(sum(t['pnl'] for t in trades if t['pnl']<=0))
aw=np.mean([t['pnl'] for t in wins]) if wins else 0
aloss=np.mean([t['pnl'] for t in trades if t['pnl']<=0]) if nl else 0
rr=abs(aw/aloss) if aloss else 0
eqs=[1000]
for t in trades:eqs.append(t['eq'])
peak=np.maximum.accumulate(eqs);dd=(np.array(eqs)-peak)/peak*100

tdf2=pd.DataFrame(trades);tdf2['year']=tdf2['ets'].dt.year

print(f"\n📋 {n} صفقة | 🟢{nw} 🔴{nl} | WR:{wr:.0f}%")
print(f"🟢 +{aw:.2f}% | 🔴 {aloss:.2f}% | R:R:{rr:.1f}x")
print(f"🏦 ${equity:,.0f} ({(equity/1000-1)*100:+.1f}%) | DD:{dd.min():.1f}%")

print(f"\n📊 حسب السنة:")
for yr in sorted(tdf2['year'].unique()):
    yt=tdf2[tdf2['year']==yr];ym=df[(df['ts'].dt.year==yr)]
    s0=ym['close'].iloc[0];s1=ym['close'].iloc[-1];chg=(s1/s0-1)*100
    yn=len(yt);yw=len(yt[yt['pnl']>0]);ywr=yw/yn*100 if yn else 0;net=sum(yt['pnl'])
    arrow="📈" if chg>0 else "📉";emoji="🟢" if net>0 else "🔴"
    print(f"  {yr}: {arrow} FET {chg:+.0f}% | {emoji} {yn}T WR:{ywr:.0f}% | صافي:{net:+.0f}%")
