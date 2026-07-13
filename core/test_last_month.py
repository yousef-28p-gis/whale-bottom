"""
LB50 — آخر شهر (يونيو-يوليو 2026)
"""
import sys; sys.path.insert(0,'/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)

# آخر 30 يوم
end = df['timestamp'].iloc[-1]
start = end - pd.Timedelta(days=30)
df_sub = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)].reset_index(drop=True)

print(f"📅 {start.date()} → {end.date()} | {len(df_sub)} شمعة\n")

CAP=1000.0
lookback=50; strength_min=10; wma_fast=3; wma_slow=10

whale = whale_indicator(df_sub, lookback)
entry = (
    whale_spike(whale) & (whale_ma(whale,wma_fast) > whale_ma(whale,wma_slow)) &
    (whale_strength(whale,50) > strength_min)
)
ema=ema21(df_sub); sell=sell_signal(df_sub); sm=swing_lows(df_sub,5)
n=len(df_sub)

capital=CAP; peak=CAP; max_dd=0.0; total_eff=0
trades=[]; in_trade=False; trade=None

for i in range(500,n):
    row=df_sub.iloc[i]; ts=row['timestamp']
    if not in_trade:
        if i < len(entry) and entry.iloc[i]:
            ep=row['close']; tp=ema.iloc[i-1] if i>=1 and not pd.isna(ema.iloc[i-1]) else None
            if tp is None or tp<=ep: continue
            sw_s=max(0,i-60); sw_r=df_sub.iloc[sw_s:i][sm[sw_s:i]]
            sl=sw_r['low'].min()*0.998 if len(sw_r)>0 else ep*0.95
            pl=ep+(tp-ep)*60/100
            trade={'ei':i,'et':ts,'e1':ep,'e2':None,'ae':ep,'al':0.25,'sl':sl,'tp':tp,'pl':pl,'pl_act':False,'hi':ep,'dca':False}
            in_trade=True
    else:
        if row['high']>trade['hi']: trade['hi']=row['high']
        if not trade['pl_act'] and row['high']>=trade['pl']: trade['pl_act']=True
        if not trade['dca']:
            s2=max(0,trade['ei']+1); ns=df_sub.iloc[s2:i+1][sm[s2:i+1]]
            if len(ns)>0 and ns['low'].min()<trade['e1']:
                trade['e2']=row['close']
                trade['ae']=(trade['e1']*25+trade['e2']*75)/100
                trade['al']=1.0; trade['dca']=True
                trade['sl']=ns['low'].min()*0.998
                trade['pl']=trade['ae']+(trade['tp']-trade['ae'])*60/100
                if row['high']>=trade['pl']: trade['pl_act']=True
        st2=max(0,i-100); swt=df_sub.iloc[st2:i+1][sm[st2:i+1]]
        if len(swt)>0:
            nsl=swt['low'].min()*0.998
            if nsl>trade['sl']: trade['sl']=nsl
        if trade['pl_act']:
            ts2=trade['hi']*(1-0.3/100)
            if ts2>trade['sl']: trade['sl']=ts2
        er=None; epx=None; hrs=(ts-trade['et']).total_seconds()/3600
        tp_h=row['high']>=trade['tp']
        sl_h=(row['high']>=trade['sl']) if trade['sl']>trade['ae'] else (row['low']<=trade['sl'])
        if tp_h: er,epx='TP',trade['tp']
        elif i>=2 and sell.iloc[i-1]>=60: er,epx='SELL',row['close']
        elif sl_h:
            if trade['pl_act']: er,epx='PL',trade['sl']
            else: er='SL_UP' if trade['sl']>trade['ae'] else 'SL'; epx=max(min(trade['sl'],row['high']),row['low'])
        elif hrs>=4: er,epx='TIME',row['close']
        if er:
            pnl=((epx-trade['ae'])/trade['ae']-0.002)*100
            eff=pnl*trade['al']
            total_eff+=eff
            capital*=(1+eff/100)
            if capital>peak: peak=capital
            dd=(capital-peak)/peak*100
            if dd<max_dd: max_dd=dd
            trades.append({'entry':ts,'exit':ts,'pnl':pnl,'eff':eff,'er':er,'dca':trade['dca']})
            in_trade=False; trade=None

tdf=pd.DataFrame(trades)
if len(tdf)==0:
    print("❌ لا توجد صفقات في آخر شهر")
else:
    wins=tdf[tdf['pnl']>0]; losses=tdf[tdf['pnl']<=0]
    wr=len(wins)/len(tdf)*100
    print(f"{'='*70}")
    print(f"🏆 آخر 30 يوم | LB50 WMA3/10")
    print(f"{'='*70}")
    print(f"صفقات: {len(tdf)} | WR: {wr:.1f}%")
    print(f"رأس مال: ${capital:.0f} | DD: {max_dd:.1f}%")
    print(f"AvgWin: {wins['pnl'].mean():.2f}% | AvgLoss: {losses['pnl'].mean():.2f}%")
    print(f"مخارج: {tdf['er'].value_counts().to_dict()}")
    print(f"DCA: {tdf['dca'].sum()}")
    print(f"\n📋 كل الصفقات:")
    for _,t in tdf.iterrows():
        icon='🟢' if t['pnl']>0 else '🔴'
        print(f"  {icon} {str(t['entry'])[:16]} → {str(t['exit'])[:16]} | {t['er']:5s} | {t['pnl']:+.2f}% | DCA={'✓' if t['dca'] else '✗'}")

print(f"\n✅")
