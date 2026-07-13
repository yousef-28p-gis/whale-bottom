"""باك تست مباشر لنفس كود TradingView"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)
CAP=1000.0; n=len(df)

def run(lb, wf, ws, smin, vol, sma50, label):
    whale = whale_indicator(df, lb)
    entry = (whale_spike(whale) & (whale_ma(whale, wf) > whale_ma(whale, ws)) &
             (whale_strength(whale, 50) > smin))
    if vol: entry &= volume_filter(df)
    if sma50: entry &= (df['close'] > sma50_daily(df))
    
    ema=ema21(df); sell=sell_signal(df); sm=swing_lows(df,5)
    capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}; trades=[]
    in_trade=False; trade=None
    
    for i in range(500,n):
        row=df.iloc[i]; ts=row['timestamp']
        mk=f"{ts.year}-{ts.month:02d}"
        if monthly_pnl.get(mk,0.0)<=-7 and not in_trade: continue
        if not in_trade:
            if entry.iloc[i] and not pd.isna(ema.iloc[i-1]) and ema.iloc[i-1]>row['close']:
                sw_s=max(0,i-60); sw_r=df.iloc[sw_s:i][sm[sw_s:i]]
                sl=sw_r['low'].min()*0.998 if len(sw_r)>0 else row['close']*0.95
                pl=row['close']+(ema.iloc[i-1]-row['close'])*60/100
                trade={'ei':i,'et':ts,'e1':row['close'],'e2':None,'ae':row['close'],'al':0.25,
                       'sl':sl,'tp':ema.iloc[i-1],'pl':pl,'pl_act':False,'hi':row['close'],'dca':False}
                in_trade=True
        else:
            if row['high']>trade['hi']: trade['hi']=row['high']
            if not trade['pl_act'] and row['high']>=trade['pl']: trade['pl_act']=True
            if not trade['dca']:
                s2=max(0,trade['ei']+1); ns=df.iloc[s2:i+1][sm[s2:i+1]]
                if len(ns)>0 and ns['low'].min()<trade['e1']:
                    trade['e2']=row['close']
                    trade['ae']=(trade['e1']*25+row['close']*75)/100
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
                trades.append({'pnl':pnl*100,'eff':eff*100,'er':er,'dca':trade['dca']})
                in_trade=False; trade=None
    
    tdf=pd.DataFrame(trades)
    if len(tdf)==0: return None
    w=tdf[tdf['pnl']>0]; l=tdf[tdf['pnl']<=0]
    months=len(set(df.iloc[500:n]['timestamp'].dt.to_period('M')))
    linear_ret = tdf['pnl'].sum()
    return {
        't':len(tdf),'wins':len(w),'losses':len(l),'wr':len(w)/len(tdf)*100,
        'total_win':w['pnl'].sum(),'total_loss':l['pnl'].sum(),'net':linear_ret,
        'avg_win':w['pnl'].mean(),'avg_loss':l['pnl'].mean() if len(l)>0 else 0,
        'c':capital,'dd':max_dd*100,'dca_n':tdf['dca'].sum(),
        'per_month':len(tdf)/months,
        'er_counts':tdf['er'].value_counts().to_dict()
    }

configs = [
    (200,20,50,50,True,True,'آمن v10'),
    (200,20,50,50,False,True,'آمن بدون حجم'),
    (50,3,10,10,False,False,'شرس LB50'),
    (50,5,10,30,False,True,'وسط LB50+SMA50'),
]

print(f"📦 {len(df)} شمعة | 2019-2026\n")
print(f"{'='*95}")
print(f"{'الإعدادات':<25} {'صفقات':>6} {'WR%':>6} {'شهرياً':>7} {'صافي%':>8} {'DD%':>7} {'محفظة$':>10} {'DCA':>5}")
print(f"{'-'*95}")

for lb,wf,ws,sm,vol,sma50,lbl in configs:
    r = run(lb,wf,ws,sm,vol,sma50,lbl)
    if r:
        print(f"{lbl:<25} {r['t']:>6} {r['wr']:>5.1f}% {r['per_month']:>6.1f} {r['net']:>7.1f}% {r['dd']:>6.1f}% ${r['c']:>9.0f} {r['dca_n']:>5}")

print(f"\n⚠️ المحفظة = تراكمي (compounded) — الرقم الخطي (net) هو الأصدق")

# اختبار سريع لآخر 3 أشهر
print(f"\n{'='*95}")
print(f"📍 آخر 3 أشهر (مايو-يوليو 2026) | FET/USDT: -19%")
print(f"{'-'*95}")
cut = pd.Timestamp('2026-05-01')
for lb,wf,ws,sm,vol,sma50,lbl in configs[:2]+configs[2:]:
    # تشغيل مصغر
    sub = df[df['timestamp']>=cut].reset_index(drop=True)
    whale = whale_indicator(sub, lb)
    entry = (whale_spike(whale) & (whale_ma(whale, wf) > whale_ma(whale, ws)) &
             (whale_strength(whale, 50) > sm))
    if vol: entry &= volume_filter(sub)
    if sma50: entry &= (sub['close'] > sma50_daily(sub))
    print(f"  {lbl:<25}: {entry.sum()} إشارة")
print()

print(f"✅ تم")
