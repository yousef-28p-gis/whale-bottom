"""
Walk-Forward: نفس البارامترات على نوافذ متحركة
Train 2 years → Test 1 year
"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)
CAP=1000.0

def run_strategy(df_sub):
    n=len(df_sub)
    whale = whale_indicator(df_sub,200)
    entry = (
        whale_spike(whale) & (whale_ma(whale,20) > whale_ma(whale,50)) &
        (whale_strength(whale,50) > 50) & volume_filter(df_sub) &
        (df_sub['close'] > sma50_daily(df_sub))
    )
    ema=ema21(df_sub); sell=sell_signal(df_sub); sm=swing_lows(df_sub,5)
    capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}; trades=[]
    in_trade=False; trade=None
    for i in range(500,n):
        row=df_sub.iloc[i]; ts=row['timestamp']
        mk=f"{ts.year}-{ts.month:02d}"
        if monthly_pnl.get(mk,0.0)<=-7 and not in_trade: continue
        if not in_trade:
            if entry.iloc[i]:
                ep=row['close']
                if i<1 or pd.isna(ema.iloc[i-1]): continue
                tp=ema.iloc[i-1]
                if tp<=ep: continue
                sw_s=max(0,i-60); sw_r=df_sub.iloc[sw_s:i][sm[sw_s:i]]
                sl=sw_r['low'].min()*0.998 if len(sw_r)>0 else ep*0.95
                pl=ep+(tp-ep)*60/100
                trade={'ei':i,'et':ts,'e1':ep,'e2':None,'ae':ep,'al':0.25,
                       'sl':sl,'tp':tp,'pl':pl,'pl_act':False,'hi':ep,'dca':False}
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
                trades.append({'pnl':pnl*100,'eff':eff*100,'er':er})
                in_trade=False; trade=None
    tdf=pd.DataFrame(trades)
    if len(tdf)==0: return {}
    w=tdf[tdf['pnl']>0]; wr=len(w)/len(tdf)*100
    rets=tdf['eff'].values/100
    sh=rets.mean()/rets.std()*np.sqrt(len(rets)) if rets.std()>0 else 0
    return {'t':len(tdf),'wr':wr,'c':capital,'ret':(capital/CAP-1)*100,'dd':max_dd*100,'sh':sh}

# ═══════════════════════════════════════════════
print(f"📦 {len(df)} candles\n")
print(f"{'='*95}")
print(f"🏆 Walk-Forward | تدريب سنتين → اختبار سنة | نفس البارامترات")
print(f"{'='*95}")
print(f"{'نافذة':<22} {'تدريب':>12} {'← اختبار':>12} {'صفقات':>6} {'WR%':>6} {'رأس مال':>9} {'عائد%':>8} {'DD%':>7} {'Sharpe':>7}")
print(f"{'-'*95}")

total_test_trades = 0
total_test_wins = 0
total_test_capital = 1000  # compounded

for train_end_year, test_year in [(2020,2021),(2021,2022),(2022,2023),(2023,2024),(2024,2025),(2025,2026)]:
    train_start = '2019-01-01'
    train_end = f'{train_end_year}-12-31'
    test_start = f'{test_year}-01-01'
    test_end = f'{test_year}-12-31'
    
    df_train = df[(df['timestamp']>=train_start)&(df['timestamp']<=train_end)]
    df_test = df[(df['timestamp']>=test_start)&(df['timestamp']<=test_end)]
    
    if len(df_train)<500 or len(df_test)<500: continue
    
    r_train = run_strategy(df_train.reset_index(drop=True))
    r_test = run_strategy(df_test.reset_index(drop=True))
    
    if not r_train or not r_test: continue
    
    total_test_trades += r_test['t']
    
    label = f"19-{train_end_year} → {test_year}"
    print(f"{label:<22} {df_train['timestamp'].iloc[0].date().strftime('%Y-%m'):>12} {df_test['timestamp'].iloc[0].date().strftime('%Y-%m'):>12} {r_test['t']:>6} {r_test['wr']:>5.1f}% ${r_test['c']:>8.0f} {r_test['ret']:>7.1f}% {r_test['dd']:>6.1f}% {r_test['sh']:>6.2f}")

print(f"{'-'*95}")
print(f"{'المجموع (كل فترات الاختبار)':<22} {'':>12} {'':>12} {total_test_trades:>6}")

# احسب compounded
comp = 1000
for train_end_year, test_year in [(2020,2021),(2021,2022),(2022,2023),(2023,2024),(2024,2025),(2025,2026)]:
    test_start = f'{test_year}-01-01'; test_end = f'{test_year}-12-31'
    df_test = df[(df['timestamp']>=test_start)&(df['timestamp']<=test_end)]
    if len(df_test)<500: continue
    r_test = run_strategy(df_test.reset_index(drop=True))
    if r_test: comp *= (1 + r_test['ret']/100)

print(f"{'مركب (compounded)':<22} {'':>12} {'':>12} {'':>6} {'':>6} ${comp:>8.0f} {(comp/1000-1)*100:>7.1f}%")

print(f"\n✅ تم")
