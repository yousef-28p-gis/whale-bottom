"""
إعادة حساب بدون Compounding + فحص الأخطاء
"""
import sys; sys.path.insert(0,'/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)
CAP=1000.0; n=len(df)

def run_linear(lookback=200, strength_min=50, use_vol=True, use_sma50=True, wma_fast=20, wma_slow=50):
    """Linear P&L — no compounding. Tracks total P&L sum and max concurrent DD."""
    whale = whale_indicator(df, lookback)
    entry = (
        whale_spike(whale) & (whale_ma(whale,wma_fast) > whale_ma(whale,wma_slow)) &
        (whale_strength(whale,50) > strength_min)
    )
    if use_vol: entry = entry & volume_filter(df)
    if use_sma50: entry = entry & (df['close'] > sma50_daily(df))
    
    sig_count = entry.sum()
    if sig_count < 5: return None
    
    ema=ema21(df); sell=sell_signal(df); sm=swing_lows(df,5)
    trades = []
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
                eff_pnl = pnl_pct * trade['al']
                trades.append({'pnl':pnl_pct,'eff':eff_pnl,'er':er,'dca':trade['dca']})
                in_trade=False; trade=None
    
    tdf = pd.DataFrame(trades)
    if len(tdf)==0: return None
    
    wins = tdf[tdf['pnl']>0]; losses = tdf[tdf['pnl']<=0]
    wr = len(wins)/len(tdf)*100
    
    # LINEAR: sum all P&L (no compounding)
    total_linear_pnl = tdf['eff'].sum()
    linear_capital = CAP * (1 + total_linear_pnl/100)
    
    # DD: track running P&L sum, find max drawdown
    running = 0; peak_running = 0; max_dd = 0
    for _, t in tdf.iterrows():
        running += t['eff']
        if running > peak_running: peak_running = running
        dd = (running - peak_running) / (CAP/100 + peak_running) * 100 if peak_running > -CAP/100 else 0
        if dd < max_dd: max_dd = dd
    
    # COMPOUNDED (for comparison)
    cap_c = CAP
    peak_c = CAP; max_dd_c = 0
    for _, t in tdf.iterrows():
        cap_c *= (1 + t['eff']/100)
        if cap_c > peak_c: peak_c = cap_c
        ddc = (cap_c - peak_c)/peak_c*100
        if ddc < max_dd_c: max_dd_c = ddc
    
    yearly = tdf['eff'].sum() / ((df['timestamp'].iloc[-1]-df['timestamp'].iloc[0]).days/365)
    
    return {
        't':len(tdf),'wr':wr,'sigs':sig_count,
        'linear_pnl':total_linear_pnl,'linear_cap':linear_capital,
        'dd':max_dd,'dd_comp':max_dd_c,
        'cap_comp':cap_c,'yearly_avg':yearly,
        'avg_win':wins['pnl'].mean() if len(wins)>0 else 0,
        'avg_loss':losses['pnl'].mean() if len(losses)>0 else 0,
        'dca':tdf['dca'].sum()
    }

# ═══════════════════════════════════════════════
print("⏳ جاري الحساب بدون compounding...\n")

tests = [
    ('v10 الأساس', {'lookback':200,'strength_min':50,'use_vol':True,'use_sma50':True,'wma_fast':20,'wma_slow':50}),
    ('🐯 شرس', {'lookback':150,'strength_min':50,'use_vol':False,'use_sma50':False,'wma_fast':20,'wma_slow':50}),
    ('+ LB100', {'lookback':100,'strength_min':50,'use_vol':False,'use_sma50':False,'wma_fast':20,'wma_slow':50}),
    ('+ LB100 + قوة>30%', {'lookback':100,'strength_min':30,'use_vol':False,'use_sma50':False,'wma_fast':20,'wma_slow':50}),
]

print(f"{'='*100}")
print(f"🏆 نتائج بدون Compounding (Linear P&L)")
print(f"{'='*100}")
print(f"{'الإعداد':<30} {'إشارات':>7} {'صفقات':>6} {'WR%':>6} {'AvgW%':>6} {'AvgL%':>6} {'إجمالي%':>9} {'خطي$':>9} {'مركب$':>10} {'DD%':>7} {'سنوي%':>7}")
print(f"{'-'*100}")

for name, params in tests:
    r = run_linear(**params)
    if r is None: continue
    star = ' ⬅' if 'v10' in name else ''
    print(f"{name:<30} {r['sigs']:>7} {r['t']:>6} {r['wr']:>5.1f}% {r['avg_win']:>5.2f}% {r['avg_loss']:>5.2f}% {r['linear_pnl']:>8.1f}% ${r['linear_cap']:>8.0f} ${r['cap_comp']:>9.0f} {r['dd']:>6.1f}% {r['yearly_avg']:>6.1f}%{star}")

print(f"\n💡 'خطي$' = رأس المال بدون compounding (مجموع الأرباح البسيط)")
print(f"💡 'مركب$' = رأس المال مع compounding (نظري فقط!)")
print(f"✅")
