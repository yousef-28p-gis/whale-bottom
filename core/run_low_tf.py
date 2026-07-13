"""
DCA مقارنة: 3m vs 5m vs 15m — نفس الفترة (6 شهور)
"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd
import numpy as np
from core.indicators import (
    whale_indicator, whale_ma, whale_strength, whale_spike,
    volume_filter, sma50_daily, ema21, atr, sell_signal, swing_lows
)

CAP=1000.0

def run_dca_bars(df_sub, max_bars):
    df_sub = df_sub.sort_values('timestamp').reset_index(drop=True)
    whale = whale_indicator(df_sub, 200)
    entry = (
        whale_spike(whale) & (whale_ma(whale,20) > whale_ma(whale,50)) &
        (whale_strength(whale,50) > 50) & volume_filter(df_sub) &
        (df_sub['close'] > sma50_daily(df_sub))
    )
    ema = ema21(df_sub); sell = sell_signal(df_sub)
    sw_mask = swing_lows(df_sub, 5)
    n = len(df_sub)
    capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}; trades=[]
    in_trade=False; trade=None

    for i in range(500, n):
        row=df_sub.iloc[i]; ts=row['timestamp']
        mk=f"{ts.year}-{ts.month:02d}"
        if monthly_pnl.get(mk,0.0) <= -7 and not in_trade: continue

        if not in_trade:
            if entry.iloc[i]:
                ep=row['close']
                if i<1 or pd.isna(ema.iloc[i-1]): continue
                tp=ema.iloc[i-1]
                if tp<=ep: continue
                sw_s=max(0,i-60); sw_r=df_sub.iloc[sw_s:i][sw_mask[sw_s:i]]
                sl=sw_r['low'].min()*0.998 if len(sw_r)>0 else ep*0.95
                trade={'ei':i,'et':ts,'e1':ep,'e2':None,'ae':ep,'al':0.5,'sl':sl,'tp':tp,'dca':False}
                in_trade=True
        else:
            if not trade['dca']:
                s2=max(0,trade['ei']+1); ns=df_sub.iloc[s2:i+1][sw_mask[s2:i+1]]
                if len(ns)>0 and ns['low'].min() < trade['e1']:
                    trade['e2']=row['close']; trade['ae']=(trade['e1']+trade['e2'])/2
                    trade['al']=1.0; trade['dca']=True
                    trade['sl']=ns['low'].min()*0.998
            st=max(0,i-100); swt=df_sub.iloc[st:i+1][sw_mask[st:i+1]]
            if len(swt)>0:
                nsl=swt['low'].min()*0.998
                if nsl>trade['sl']: trade['sl']=nsl

            er=None; epx=None; be=i-trade['ei']
            tp_h=row['high']>=trade['tp']
            sl_h=(row['high']>=trade['sl']) if trade['sl']>trade['ae'] else (row['low']<=trade['sl'])

            if tp_h: er,epx='TP',trade['tp']
            elif i>=2 and sell.iloc[i-1]>=60: er,epx='SELL',row['close']
            elif sl_h:
                er='SL_UP' if trade['sl']>trade['ae'] else 'SL'
                epx=(min(trade['sl'],row['high']) if trade['sl']>trade['ae'] else max(trade['sl'],row['low']))
            elif be>=max_bars: er,epx='TIME',row['close']

            if er:
                pnl=(epx-trade['ae'])/trade['ae']-0.002; eff=pnl*trade['al']
                monthly_pnl[mk]=monthly_pnl.get(mk,0.0)+eff*100
                capital*=(1+eff)
                if capital>peak: peak=capital
                dd=(capital-peak)/peak
                if dd<max_dd: max_dd=dd
                trades.append({'pnl':pnl*100,'er':er,'dca':trade['dca'],'y':ts.year})
                in_trade=False; trade=None

    tdf=pd.DataFrame(trades)
    if len(tdf)==0: return {'t':0,'c':CAP,'dd':0,'wr':0}
    w=tdf[tdf['pnl']>0]; wr=len(w)/len(tdf)*100
    return {'t':len(tdf),'wr':wr,'c':capital,'ret':(capital/CAP-1)*100,'dd':max_dd*100,'dca':tdf['dca'].sum(),'tdf':tdf}

# ── تحميل ──
print("⏳ تحميل البيانات...")
df3 = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_3m_6M.csv')
df3['timestamp'] = pd.to_datetime(df3['ts'])
df5 = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_5m_6M.csv')
df5['timestamp'] = pd.to_datetime(df5['ts'])
df15 = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_6M.csv')
df15['timestamp'] = pd.to_datetime(df15['timestamp'])

print(f"📊 3m={len(df3)} | 5m={len(df5)} | 15m={len(df15)}")
print(f"   الفترة: {df3['timestamp'].iloc[0]} → {df3['timestamp'].iloc[-1]}\n")

# ── اختبار ──
print(f"{'='*85}")
print(f"🏆 DCA | 3m vs 5m vs 15m | نفس الفترة (6 شهور)")
print(f"{'='*85}")
print(f"{'فريم':<6} {'حد':<10} {'صفقات':>6} {'WR%':>6} {'رأس المال':>10} {'DD%':>7} {'DCA':>5}")
print(f"{'-'*85}")

for tf, df_tf, mb, label in [
    ('3m', df3, 80, '4h=80b'),
    ('3m', df3, 40, '2h=40b'),
    ('3m', df3, 20, '1h=20b'),
    ('5m', df5, 48, '4h=48b'),
    ('5m', df5, 24, '2h=24b'),
    ('5m', df5, 12, '1h=12b'),
    ('15m', df15, 16, '4h=16b'),
    ('15m', df15, 8, '2h=8b'),
    ('15m', df15, 4, '1h=4b'),
]:
    r = run_dca_bars(df_tf, mb)
    if r['t']==0:
        print(f"{tf:<6} {label:<10} {'—':>6} {'—':>6} {'—':>10} {'—':>7} {'—':>5}")
        continue
    star = ' ⬅' if tf=='15m' and mb==16 else ''
    print(f"{tf:<6} {label:<10} {r['t']:>6} {r['wr']:>5.1f}% ${r['c']:>9.0f} {r['dd']:>6.1f}% {r['dca']:>5}{star}")

print(f"\n✅ تم")
