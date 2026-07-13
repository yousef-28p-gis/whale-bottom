"""اختبار آخر شهرين فقط"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)

# نقطع من 1 مايو 2026
cutoff = pd.Timestamp('2026-05-01')
df = df[df['timestamp'] >= cutoff].reset_index(drop=True)
CAP=1000.0; n=len(df)

print(f"📅 من {df['timestamp'].iloc[0].strftime('%Y-%m-%d')} إلى {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"📦 {n} candle | السعر: {df['close'].iloc[0]:.4f} → {df['close'].iloc[-1]:.4f} ({((df['close'].iloc[-1]/df['close'].iloc[0]-1)*100):+.1f}%)\n")

configs = [
    ("LB50+WMA5/10+SMA50 (جديد)", 50, 5, 10, 30, True, False),
    ("LB50+WMA3/10 بلا SMA50 (شرس)", 50, 3, 10, 10, False, False),
    ("LB200+WMA20/50+SMA50 (v10)", 200, 20, 50, 50, True, True),
]

for name, lb, wf, ws, str_min, use_sma50, use_vol in configs:
    whale = whale_indicator(df, lb)
    entry = whale_spike(whale) & (whale_ma(whale, wf) > whale_ma(whale, ws)) & (whale_strength(whale, 50) > str_min)
    if use_sma50: entry &= (df['close'] > sma50_daily(df))
    if use_vol: entry &= volume_filter(df)
    
    ema=ema21(df); sell=sell_signal(df); sm=swing_lows(df,5)
    
    capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}; trades=[]
    in_trade=False; trade=None
    
    for i in range(200, n):
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
                pl_price = ep + (tp-ep)*60/100
                trade={'ei':i,'et':ts,'e1':ep,'e2':None,'ae':ep,'al':0.25,
                       'sl':sl,'tp':tp,'pl':pl_price,'pl_act':False,
                       'hi':ep,'dca':False,'first':25,'sec':75}
                in_trade=True
        else:
            if row['high']>trade['hi']: trade['hi']=row['high']
            if not trade['pl_act'] and row['high'] >= trade['pl']:
                trade['pl_act']=True
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
                trail_sl=trade['hi']*(1-0.3/100)
                if trail_sl>trade['sl']: trade['sl']=trail_sl
            er=None; epx=None; hrs=(ts-trade['et']).total_seconds()/3600
            tp_h=row['high']>=trade['tp']
            sl_h=(row['high']>=trade['sl']) if trade['sl']>trade['ae'] else (row['low']<=trade['sl'])
            if tp_h: er,epx='TP',trade['tp']
            elif i>=2 and sell.iloc[i-1]>=60: er,epx='SELL',row['close']
            elif sl_h:
                if trade['pl_act']: er,epx='PL',max(trade['sl'],row['low'])
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
                trades.append({'pnl':pnl*100,'eff_pnl':eff*100,'er':er,'dca':trade['dca']})
                in_trade=False; trade=None
    
    tdf=pd.DataFrame(trades)
    if len(tdf)==0:
        print(f"❌ {name}: 0 صفقات\n")
        continue
    w=tdf[tdf['pnl']>0]; 
    wr=len(w)/len(tdf)*100
    net = tdf['pnl'].sum()
    print(f"{'='*60}")
    print(f"🔍 {name}")
    print(f"{'='*60}")
    print(f"  صفقات: {len(tdf)} | 🟢 {len(w)} | 🔴 {len(tdf)-len(w)} | WR: {wr:.0f}%")
    print(f"  💰 صافي خطي: {net:+.1f}% | 🏦 محفظة: ${capital:.0f} ({((capital/CAP-1)*100):+.1f}%)")
    print(f"  📉 DD: {((capital-peak)/peak*100) if peak>capital else 0:.1f}% (actual max: {max_dd:.1f}%)")
    print(f"  🟢 متوسط ربح: +{w['pnl'].mean():.2f}% | 🔴 متوسط خسارة: {tdf[tdf['pnl']<=0]['pnl'].mean():.2f}%")
    print(f"  خروج: {tdf['er'].value_counts().to_dict()}")
    print()

print("✅ تم")
