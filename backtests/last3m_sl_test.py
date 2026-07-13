#!/usr/bin/env python3
"""Last 3 months — different SL modes"""
import pandas as pd, numpy as np

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv', parse_dates=['ts'])
cutoff = df['ts'].max() - pd.Timedelta(days=90)
tdf = df[df['ts'] >= cutoff].copy().reset_index(drop=True)
print(f"📅 آخر 3 شهور: {tdf['ts'].iloc[0]} → {tdf['ts'].iloc[-1]} | شموع: {len(tdf)}")
rng = tdf['close'].max() - tdf['close'].min()
print(f"💰 السعر: ${tdf['close'].min():.4f} → ${tdf['close'].max():.4f} | {'📈 صاعد' if tdf['close'].iloc[-1]>tdf['close'].iloc[0] else '📉 هابط'} | تغير: {(tdf['close'].iloc[-1]/tdf['close'].iloc[0]-1)*100:+.1f}%")

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

# Sell signal
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

# Swings
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
print(f"🔍 إشارات: {len(eis)}")

# ─── Test 3 SL modes ──────────────────────────────────────────
def simulate(name, sl_mode):
    """sl_mode: 'swing'=trailing swing, 'fixed_swing'=swing at entry only, 'atr'=2ATR, 'none'=no SL"""
    trades=[];it=False;ed=0;equity=1000;pos=25
    for ei in eis:
        if ei<10:continue
        if it and ei<ed:continue
        e=tdf['close'].iloc[ei]
        
        if sl_mode=='swing':sl=nsl(ei)*0.998
        elif sl_mode=='fixed_swing':sl=nsl(ei)*0.998  # same but never updated
        elif sl_mode=='atr':sl=e-tdf['atr'].iloc[ei]*2
        elif sl_mode=='none':sl=0  # never triggers
        
        end=min(ei+192,len(tdf));r=None;ep=e;ex=ei
        for j in range(ei+1,end):
            if sl_mode!='none' and tdf['low'].iloc[j]<=sl:r='SL';ep=sl;ex=j;break
            if tdf['sell'].iloc[j]>=60:r='SELL';ep=tdf['close'].iloc[j];ex=j;break
        if not r:r='TIME';ep=tdf['close'].iloc[end-1];ex=end-1
        pnl=(ep-e)/e*100-0.2
        dollar=equity*(pos/100)*(pnl/100)
        equity+=dollar
        trades.append({'pnl':pnl,'r':r,'d':dollar,'eq':equity,'e':e,'ep':ep,'ets':tdf['ts'].iloc[ei]})
        it=True;ed=ex
    n=len(trades);wins=[t for t in trades if t['pnl']>0]
    return {'name':name,'n':n,'wr':len(wins)/n*100 if n else 0,'eq':equity,'trades':trades}

modes = [
    simulate("SL سوينج متحرك", 'swing'),
    simulate("SL سوينج ثابت", 'fixed_swing'),
    simulate("SL = 2×ATR", 'atr'),
    simulate("بدون SL (إشارة بيع فقط)", 'none'),
]

print(f"\n{'='*65}")
print(f"📊 مقارنة أنواع SL — آخر 3 شهور (25% حجم)")
print(f"{'='*65}")
print(f"{'النوع':<22} {'صفقات':>5} {'WR':>5} {'محفظة':>8} {'ربح/خسارة':>12}")
print("-"*55)
for m in modes:
    wins=[t for t in m['trades'] if t['pnl']>0]
    losses=[t for t in m['trades'] if t['pnl']<=0]
    wpct=sum(t['pnl'] for t in wins) if wins else 0
    lpct=sum(t['pnl'] for t in losses) if losses else 0
    print(f"{m['name']:<22} {m['n']:>5} {m['wr']:>4.0f}% ${m['eq']:>7,.0f} {wpct:>+5.1f}%/{lpct:>+5.1f}%")

# Show details of best mode
best = max(modes, key=lambda m: m['eq'])
print(f"\n🔍 تفاصيل أفضل نوع: {best['name']}")
print(f"{'تاريخ':<19} {'دخول':>7} {'خروج':>7} {'نتيجة':<5} {'ربح%':>7}")
print("-"*50)
for t in best['trades']:
    em="🟢" if t['pnl']>0 else "🔴"
    print(f"{em}{str(t['ets'])[:19]:<18} {t['e']:>7.4f} {t['ep']:>7.4f} {t['r']:<5} {t['pnl']:>+6.2f}%")
