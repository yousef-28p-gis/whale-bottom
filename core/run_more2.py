"""
المزيد من الصفقات — توليفات بدون حجم مع فلاتر مخففة
"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)
CAP=1000.0; n=len(df)

def runq(df, lookback=200, strength_min=50, use_vol=True, use_sma50=True,
         wma_fast=20, wma_slow=50):
    whale = whale_indicator(df, lookback)
    entry = (
        whale_spike(whale) & (whale_ma(whale, wma_fast) > whale_ma(whale, wma_slow)) &
        (whale_strength(whale, 50) > strength_min)
    )
    if use_vol: entry = entry & volume_filter(df)
    if use_sma50: entry = entry & (df['close'] > sma50_daily(df))
    sigs = entry.sum()
    if sigs < 5: return None

    ema=ema21(df); sell=sell_signal(df); sm=swing_lows(df,5)
    capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}
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
                sw_s=max(0,i-60); sw_r=df.iloc[sw_s:i][sm[sw_s:i]]
                sl=sw_r['low'].min()*0.998 if len(sw_r)>0 else ep*0.95
                pl=ep+(tp-ep)*60/100
                trade={'ei':i,'et':ts,'e1':ep,'e2':None,'ae':ep,'al':0.25,
                       'sl':sl,'tp':tp,'pl':pl,'pl_act':False,'hi':ep,'dca':False}
                in_trade=True
        else:
            if row['high']>trade['hi']: trade['hi']=row['high']
            if not trade['pl_act'] and row['high']>=trade['pl']: trade['pl_act']=True
            if not trade['dca']:
                s2=max(0,trade['ei']+1); ns=df.iloc[s2:i+1][sm[s2:i+1]]
                if len(ns)>0 and ns['low'].min()<trade['e1']:
                    trade['e2']=row['close']
                    trade['ae']=(trade['e1']*25+trade['e2']*75)/100
                    trade['al']=1.0; trade['dca']=True
                    trade['sl']=ns['low'].min()*0.998
                    trade['pl']=trade['ae']+(trade['tp']-trade['ae'])*60/100
                    if row['high']>=trade['pl']: trade['pl_act']=True
            st2=max(0,i-100); swt=df.iloc[st2:i+1][sm[st2:i+1]]
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
                else:
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
                in_trade=False; trade=None

    if capital==CAP: return None
    return {'c':capital,'ret':(capital/CAP-1)*100,'dd':max_dd*100,'sigs':sigs}

# ═══════════════════════════════════════════════
print(f"📦 {len(df)} candles\n")
print(f"{'='*90}")
print(f"🏆 المزيد من الصفقات — بدون فلتر حجم + فلاتر مخففة")
print(f"{'='*90}")
print(f"{'الإعداد':<45} {'إشارات':>8} {'محفظة':>10} {'عائد%':>9} {'DD%':>7}")
print(f"{'-'*90}")

tests = [
    ('v10 الأساس', {}),
    ('بدون حجم فقط', {'use_vol': False}),
    ('بدون حجم + قوة>30%', {'use_vol': False, 'strength_min': 30}),
    ('بدون حجم + بدون SMA50', {'use_vol': False, 'use_sma50': False}),
    ('بدون حجم + LB150', {'use_vol': False, 'lookback': 150}),
    ('بدون حجم + wMA10>30', {'use_vol': False, 'wma_fast': 10, 'wma_slow': 30}),
    ('بدون حجم + LB150 + wMA10>30', {'use_vol': False, 'lookback': 150, 'wma_fast': 10, 'wma_slow': 30}),
    ('بدون حجم + SMA50 + LB150', {'use_vol': False, 'use_sma50': False, 'lookback': 150}),
    ('بدون حجم + SMA50 + LB100', {'use_vol': False, 'use_sma50': False, 'lookback': 100}),
]

for name, ov in tests:
    params = {'lookback':200,'strength_min':50,'use_vol':True,'use_sma50':True,'wma_fast':20,'wma_slow':50}
    params.update(ov)
    r = runq(df, **params)
    if r is None:
        print(f"{name:<45} {'—':>8}")
        continue
    star = ' ⬅' if name.startswith('v10') else ''
    icon = '✅' if r['dd'] > -10 else ('⚠️' if r['dd'] > -15 else '❌')
    print(f"{name:<45} {r['sigs']:>8} ${r['c']:>9.0f} {r['ret']:>8.1f}% {r['dd']:>6.1f}% {icon}{star}")

print(f"\n✅ تم")
