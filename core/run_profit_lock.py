"""
Profit Lock: trailing stop بعد وصول السعر لنسبة من TP
"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)

CAP=1000.0; n=len(df)
whale = whale_indicator(df,200)
entry = (
    whale_spike(whale) & (whale_ma(whale,20) > whale_ma(whale,50)) &
    (whale_strength(whale,50) > 50) & volume_filter(df) &
    (df['close'] > sma50_daily(df))
)
ema=ema21(df); sell=sell_signal(df); sm=swing_lows(df,5)

def run_pl(entry_signal, pl_pct=70, trail_pct=0.3):
    """
    pl_pct: نسبة من TP نفعّل عندها profit lock
    trail_pct: نسبة التراجع تحت أعلى سعر للخروج
    """
    capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}; trades=[]
    in_trade=False; trade=None

    for i in range(500,n):
        row=df.iloc[i]; ts=row['timestamp']
        mk=f"{ts.year}-{ts.month:02d}"
        if monthly_pnl.get(mk,0.0)<=-7 and not in_trade: continue
        if not in_trade:
            if entry_signal.iloc[i]:
                ep=row['close']
                if i<1 or pd.isna(ema.iloc[i-1]): continue
                tp=ema.iloc[i-1]
                if tp<=ep: continue
                sw_s=max(0,i-60); sw_r=df.iloc[sw_s:i][sm[sw_s:i]]
                sl=sw_r['low'].min()*0.998 if len(sw_r)>0 else ep*0.95
                pl_price = ep + (tp-ep)*pl_pct/100  # سعر تفعيل profit lock
                trade={'ei':i,'et':ts,'e1':ep,'e2':None,'ae':ep,'al':0.5,
                       'sl':sl,'tp':tp,'pl':pl_price,'pl_act':False,
                       'hi':ep,'dca':False}
                in_trade=True
        else:
            # تتبع أعلى سعر
            if row['high']>trade['hi']: trade['hi']=row['high']

            # تفعيل Profit Lock
            if not trade['pl_act'] and row['high'] >= trade['pl']:
                trade['pl_act']=True

            # DCA
            if not trade['dca']:
                s2=max(0,trade['ei']+1); ns=df.iloc[s2:i+1][sm[s2:i+1]]
                if len(ns)>0 and ns['low'].min()<trade['e1']:
                    trade['e2']=row['close']; trade['ae']=(trade['e1']+trade['e2'])/2
                    trade['al']=1.0; trade['dca']=True
                    trade['sl']=ns['low'].min()*0.998
                    # إعادة حساب PL
                    trade['pl']=trade['ae']+(trade['tp']-trade['ae'])*pl_pct/100
                    if row['high']>=trade['pl']: trade['pl_act']=True

            # Trail SL (swing)
            st2=max(0,i-100); swt=df.iloc[st2:i+1][sm[st2:i+1]]
            if len(swt)>0:
                nsl=swt['low'].min()*0.998
                if nsl>trade['sl']: trade['sl']=nsl

            # إذا profit lock مفعّل، SL = أعلى من (swing SL, trailing stop)
            if trade['pl_act']:
                trail_sl = trade['hi'] * (1 - trail_pct/100)
                if trail_sl > trade['sl']:
                    trade['sl'] = trail_sl

            er=None; epx=None; hrs=(ts-trade['et']).total_seconds()/3600
            tp_h=row['high']>=trade['tp']
            sl_h=(row['high']>=trade['sl']) if trade['sl']>trade['ae'] else (row['low']<=trade['sl'])

            if tp_h: er,epx='TP',trade['tp']
            elif i>=2 and sell.iloc[i-1]>=60: er,epx='SELL',row['close']
            elif sl_h:
                if trade['pl_act']:
                    er='PL'  # خرج بـ Profit Lock
                    epx=trade['sl']
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
                trades.append({'pnl':pnl*100,'er':er,'dca':trade['dca'],'pl':trade['pl_act']})
                in_trade=False; trade=None

    tdf=pd.DataFrame(trades)
    if len(tdf)==0: return {'t':0}
    w=tdf[tdf['pnl']>0]; wr=len(w)/len(tdf)*100
    pl_trades=tdf[tdf['er']=='PL']
    pl_avg=pl_trades['pnl'].mean() if len(pl_trades)>0 else 0
    return {'t':len(tdf),'wr':wr,'c':capital,'ret':(capital/CAP-1)*100,
            'dd':max_dd*100,'pl_n':len(pl_trades),'pl_avg':pl_avg,
            'tp':len(tdf[tdf['er']=='TP']),'sl':len(tdf[tdf['er'].isin(['SL','SL_UP'])]),
            'time':len(tdf[tdf['er']=='TIME']),'sell':len(tdf[tdf['er']=='SELL']),
            'tdf':tdf}

# ═══════════════════════════════════════════════
print(f"📦 {len(df)} candles | 🚦 {entry.sum()} signals\n")
print(f"{'='*95}")
print(f"🏆 Profit Lock — إنقاذ الصفقات اللي صعدت ثم نزلت")
print(f"{'='*95}")
print(f"{'PL عند':<12} {'Trail':<8} {'صفقات':>6} {'WR%':>6} {'المحفظة':>10} {'عائد%':>8} {'DD%':>7} {'PL':>5} {'AvgPL%':>7}")
print(f"{'-'*95}")

best_cap = 0
for pl, trail, label in [
    (0,0,'بدون PL'), (60,0.3,'60% + 0.3%'), (60,0.5,'60% + 0.5%'),
    (70,0.3,'70% + 0.3%'), (70,0.5,'70% + 0.5%'),
    (80,0.3,'80% + 0.3%'), (80,0.5,'80% + 0.5%'),
]:
    r = run_pl(entry, pl if pl>0 else 200, trail)  # 200 = never activates
    if r['t']==0: continue
    star = ' ⬅' if r['c'] > best_cap else ''
    if r['c'] > best_cap: best_cap = r['c']
    print(f"{label:<12} {f'{trail}%':<8} {r['t']:>6} {r['wr']:>5.1f}% ${r['c']:>9.0f} {r['ret']:>7.1f}% {r['dd']:>6.1f}% {r['pl_n']:>5} {r['pl_avg']:>6.2f}%{star}")
    if pl==0:
        print(f"   مخارج: TP={r['tp']} SL={r['sl']} SELL={r['sell']} TIME={r['time']}")
    else:
        print(f"   مخارج: TP={r['tp']} SL={r['sl']} SELL={r['sell']} TIME={r['time']} PL={r['pl_n']}")

print(f"\n✅ تم")
