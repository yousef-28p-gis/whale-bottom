"""
تحليل علاقة قوة الحوت بالصفقات الخاسرة
"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd
import numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts'])
df = df.sort_values('timestamp').reset_index(drop=True)

CAP=1000.0
whale = whale_indicator(df, 200)
wma20 = whale_ma(whale, 20)
wma50 = whale_ma(whale, 50)
strength = whale_strength(whale, 50)
spike = whale_spike(whale)
vol_ok = volume_filter(df)
sma50 = sma50_daily(df)

entry = (
    spike & (wma20>wma50) & (strength>50) & vol_ok & (df['close']>sma50)
)
ema = ema21(df); sell = sell_signal(df); sw_mask = swing_lows(df,5)
n=len(df)
capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}; trades=[]
in_trade=False; trade=None

for i in range(500,n):
    row=df.iloc[i]; ts=row['timestamp']
    mk=f"{ts.year}-{ts.month:02d}"
    if monthly_pnl.get(mk,0.0)<=-7 and not in_trade: continue
    if not in_trade:
        if entry.iloc[i]:
            ep=row['close']
            if i<1 or pd.isna(ema.iloc[i-1]): continue
            tp=ema.iloc[i-1]
            if tp<=ep: continue
            sw_s=max(0,i-60); sw_r=df.iloc[sw_s:i][sw_mask[sw_s:i]]
            sl=sw_r['low'].min()*0.998 if len(sw_r)>0 else ep*0.95
            trade={'ei':i,'et':ts,'e1':ep,'e2':None,'ae':ep,'al':0.5,
                   'sl':sl,'tp':tp,'dca':False,'str':strength.iloc[i]}
            in_trade=True
    else:
        if not trade['dca']:
            s2=max(0,trade['ei']+1); ns=df.iloc[s2:i+1][sw_mask[s2:i+1]]
            if len(ns)>0 and ns['low'].min()<trade['e1']:
                trade['e2']=row['close']; trade['ae']=(trade['e1']+trade['e2'])/2
                trade['al']=1.0; trade['dca']=True
                trade['sl']=ns['low'].min()*0.998
        st=max(0,i-100); swt=df.iloc[st:i+1][sw_mask[st:i+1]]
        if len(swt)>0:
            nsl=swt['low'].min()*0.998
            if nsl>trade['sl']: trade['sl']=nsl
        er=None; epx=None; hrs=(ts-trade['et']).total_seconds()/3600
        tp_h=row['high']>=trade['tp']
        sl_h=(row['high']>=trade['sl']) if trade['sl']>trade['ae'] else (row['low']<=trade['sl'])
        if tp_h: er,epx='TP',trade['tp']
        elif i>=2 and sell.iloc[i-1]>=60: er,epx='SELL',row['close']
        elif sl_h: 
            er='SL_UP' if trade['sl']>trade['ae'] else 'SL'
            epx=(min(trade['sl'],row['high']) if trade['sl']>trade['ae'] else max(trade['sl'],row['low']))
        elif hrs>=4: er,epx='TIME',row['close']
        if er:
            pnl=(epx-trade['ae'])/trade['ae']-0.002; eff=pnl*trade['al']
            monthly_pnl[mk]=monthly_pnl.get(mk,0.0)+eff*100
            capital*=(1+eff)
            if capital>peak: peak=capital
            dd=(capital-peak)/peak
            if dd<max_dd: max_dd=dd
            trades.append({'str':trade['str'],'pnl':pnl*100,'er':er,
                           'dca':trade['dca'],'y':ts.year})
            in_trade=False; trade=None

tdf=pd.DataFrame(trades)
print(f"📊 {len(tdf)} صفقة | WR={len(tdf[tdf['pnl']>0])/len(tdf)*100:.1f}%\n")

# ── توزيع القوة ──
print("="*65)
print("🔬 علاقة قوة الحوت بالربح/الخسارة")
print("="*65)

# تقسيم القوة إلى فئات
bins=[50,55,60,65,70,75,80,85,90,95,100]
labels=[f"{bins[i]}-{bins[i+1]}%" for i in range(len(bins)-1)]
tdf['str_bin']=pd.cut(tdf['str'],bins=bins,labels=labels)

print(f"\n{'القوة':<12} {'صفقات':>6} {'WR%':>7} {'متوسط PnL%':>12} {'كم خسارة':>10}")
print("-"*50)
for b in labels:
    sub=tdf[tdf['str_bin']==b]
    if len(sub)==0: continue
    wr=len(sub[sub['pnl']>0])/len(sub)*100
    print(f"{b:<12} {len(sub):>6} {wr:>6.1f}% {sub['pnl'].mean():>11.2f}% {len(sub[sub['pnl']<=0]):>8}")

# المجموع
print("-"*50)
print(f"{'الكل':<12} {len(tdf):>6} {len(tdf[tdf['pnl']>0])/len(tdf)*100:>6.1f}% {tdf['pnl'].mean():>11.2f}% {len(tdf[tdf['pnl']<=0]):>8}")

# ── مقارنة رابحة vs خاسرة ──
print(f"\n{'='*65}")
print("🔬 مقارنة مباشرة: رابحة vs خاسرة")
print(f"="*65)
wins=tdf[tdf['pnl']>0]
loss=tdf[tdf['pnl']<=0]
print(f"رابحة ({len(wins)}): متوسط القوة = {wins['str'].mean():.1f}% | وسيط = {wins['str'].median():.1f}%")
print(f"خاسرة ({len(loss)}): متوسط القوة = {loss['str'].mean():.1f}% | وسيط = {loss['str'].median():.1f}%")

# توزيع
print(f"\nتوزيع القوة للرابحة:")
print(f"  Min={wins['str'].min():.1f}% | Q1={wins['str'].quantile(0.25):.1f}% | Median={wins['str'].median():.1f}% | Q3={wins['str'].quantile(0.75):.1f}% | Max={wins['str'].max():.1f}%")
print(f"توزيع القوة للخاسرة:")
print(f"  Min={loss['str'].min():.1f}% | Q1={loss['str'].quantile(0.25):.1f}% | Median={loss['str'].median():.1f}% | Q3={loss['str'].quantile(0.75):.1f}% | Max={loss['str'].max():.1f}%")

# ── أعلى وأدنى قوة ──
print(f"\n{'='*65}")
print("🔬 أقوى وأضعف الصفقات")
print(f"="*65)
top25 = tdf.nlargest(10, 'str')
bot25 = tdf.nsmallest(10, 'str')
print(f"\n🔝 أعلى 10 صفقات قوة:")
for _,t in top25.iterrows():
    print(f"  قوة={t['str']:.1f}% | PnL={t['pnl']:+.2f}% | {t['er']} | DCA={t['dca']}")

print(f"\n🔻 أدنى 10 صفقات قوة:")
for _,t in bot25.iterrows():
    print(f"  قوة={t['str']:.1f}% | PnL={t['pnl']:+.2f}% | {t['er']} | DCA={t['dca']}")

print("\n✅ تم")
