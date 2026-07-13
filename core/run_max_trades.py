"""
أقصى صفقات — LB50, LB30, بدون فلتر قوة, WMA سريع
"""
import sys; sys.path.insert(0,'/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)
CAP=1000.0; n=len(df)

def fast_linear(lookback=100, strength_min=30, wma_fast=20, wma_slow=50):
    whale = whale_indicator(df, lookback)
    entry = (
        whale_spike(whale) &
        (whale_ma(whale,wma_fast) > whale_ma(whale,wma_slow)) &
        (whale_strength(whale,50) > strength_min)
    )
    # بدون حجم وبدون SMA50
    
    sig_count = entry.sum()
    if sig_count < 10: return None
    
    ema=ema21(df); sell=sell_signal(df); sm=swing_lows(df,5)
    total_eff = 0; running = 0; peak = 0; max_dd = 0
    wins = 0; total_trades = 0
    in_trade=False; trade=None
    
    for i in range(500,n):
        row=df.iloc[i]; ts=row['timestamp']
        if not in_trade:
            if entry.iloc[i]:
                ep=row['close']; tp=ema.iloc[i-1] if i>=1 and not pd.isna(ema.iloc[i-1]) else None
                if tp is None or tp<=ep: continue
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
                pnl_pct = ((epx-trade['ae'])/trade['ae']-0.002) * 100
                eff = pnl_pct * trade['al']
                total_eff += eff
                total_trades += 1
                if pnl_pct > 0: wins += 1
                running += eff
                if running > peak: peak = running
                dd = (running - peak) / (100 + peak) * 100
                if dd < max_dd: max_dd = dd
                in_trade=False; trade=None
    
    if total_trades == 0: return None
    wr = wins/total_trades*100
    linear_cap = CAP * (1 + total_eff/100)
    yearly = total_eff / ((df['timestamp'].iloc[-1]-df['timestamp'].iloc[0]).days/365)
    
    return {'sigs':sig_count,'t':total_trades,'wr':wr,'eff':total_eff,'cap':linear_cap,'dd':max_dd,'yearly':yearly}

# ═══════════════════════════════════════════════
print(f"📦 {len(df)} candles\n")
print(f"{'الإعداد':<40} {'إشارات':>7} {'صفقات':>6} {'WR%':>6} {'إجمالي%':>8} {'رأس مال':>9} {'DD%':>7} {'سنوي%':>7}")
print(f"{'-'*92}")

tests = [
    ('🟢 v10 (مرجع)', {'lookback':200,'strength_min':50,'wma_fast':20,'wma_slow':50}),
    ('🐯 شرس LB150', {'lookback':150,'strength_min':50,'wma_fast':20,'wma_slow':50}),
    ('LB100 WMA20/50', {'lookback':100,'strength_min':50,'wma_fast':20,'wma_slow':50}),
    ('LB100 WMA10/30', {'lookback':100,'strength_min':50,'wma_fast':10,'wma_slow':30}),
    ('LB100 WMA3/10 STR>30', {'lookback':100,'strength_min':30,'wma_fast':3,'wma_slow':10}),
    ('LB80 WMA3/10 STR>10', {'lookback':80,'strength_min':10,'wma_fast':3,'wma_slow':10}),
    ('LB50 WMA3/10 STR>10', {'lookback':50,'strength_min':10,'wma_fast':3,'wma_slow':10}),
    ('⚡ LB50 WMA3/10 STR>0', {'lookback':50,'strength_min':0,'wma_fast':3,'wma_slow':10}),
]

for name, params in tests:
    r = fast_linear(**params)
    if r is None:
        print(f"{name:<40} {'—':>7}")
        continue
    icon = '✅' if r['dd']>-20 else ('⚠️' if r['dd']>-25 else '💀')
    print(f"{name:<40} {r['sigs']:>7} {r['t']:>6} {r['wr']:>5.1f}% {r['eff']:>7.1f}% ${r['cap']:>8.0f} {r['dd']:>6.1f}% {r['yearly']:>6.1f}% {icon}")

print(f"\n✅")
