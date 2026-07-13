"""
باك تست LB50 + WMA5/20 + قوة 30% + بدون حجم + مع SMA50 يومي
"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)
CAP=1000.0; n=len(df)

# الإعدادات الجديدة
whale = whale_indicator(df, 50)  # LB50
entry = (
    whale_spike(whale) &
    (whale_ma(whale, 5) > whale_ma(whale, 10)) &  # WMA5/20? لا — WMA5/10
    (whale_strength(whale, 50) > 30) &  # قوة 30%
    (df['close'] > sma50_daily(df))  # مع SMA50 يومي
    # بدون فلتر حجم
)
ema=ema21(df); sell=sell_signal(df); sm=swing_lows(df,5)

def run_dca_ratio(entry_signal, first_pct=25):
    second_pct = 100 - first_pct
    initial_al = first_pct / 100
    
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
                pl_price = ep + (tp-ep)*60/100
                trade={'ei':i,'et':ts,'e1':ep,'e2':None,'ae':ep,'al':initial_al,
                       'sl':sl,'tp':tp,'pl':pl_price,'pl_act':False,
                       'hi':ep,'dca':False,'first':first_pct,'sec':second_pct}
                in_trade=True
        else:
            if row['high']>trade['hi']: trade['hi']=row['high']
            if not trade['pl_act'] and row['high'] >= trade['pl']:
                trade['pl_act']=True
            
            if not trade['dca']:
                s2=max(0,trade['ei']+1); ns=df.iloc[s2:i+1][sm[s2:i+1]]
                if len(ns)>0 and ns['low'].min()<trade['e1']:
                    trade['e2']=row['close']
                    trade['ae'] = (trade['e1']*trade['first'] + trade['e2']*trade['sec']) / 100
                    trade['al']=1.0; trade['dca']=True
                    trade['sl']=ns['low'].min()*0.998
                    trade['pl']=trade['ae']+(trade['tp']-trade['ae'])*60/100
                    if row['high']>=trade['pl']: trade['pl_act']=True
            
            st2=max(0,i-100); swt=df.iloc[st2:i+1][sm[st2:i+1]]
            if len(swt)>0:
                nsl=swt['low'].min()*0.998
                if nsl>trade['sl']: trade['sl']=nsl
            
            if trade['pl_act']:
                trail_sl = trade['hi'] * (1 - 0.3/100)
                if trail_sl > trade['sl']: trade['sl'] = trail_sl
            
            er=None; epx=None; hrs=(ts-trade['et']).total_seconds()/3600
            tp_h=row['high']>=trade['tp']
            sl_h=(row['high']>=trade['sl']) if trade['sl']>trade['ae'] else (row['low']<=trade['sl'])
            
            if tp_h: er,epx='TP',trade['tp']
            elif i>=2 and sell.iloc[i-1]>=60: er,epx='SELL',row['close']
            elif sl_h:
                if trade['pl_act']:
                    er,epx='PL',max(trade['sl'],row['low'])
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
                trades.append({'pnl':pnl*100,'eff_pnl':eff*100,'er':er,'dca':trade['dca'],
                              'ae':trade['ae'],'epx':epx,'tp':trade['tp'],'sl':trade['sl']})
                in_trade=False; trade=None
    
    tdf=pd.DataFrame(trades)
    if len(tdf)==0: return {}
    w=tdf[tdf['pnl']>0]; l=tdf[tdf['pnl']<=0]
    return {
        't':len(tdf),'wins':len(w),'losses':len(l),
        'wr':len(w)/len(tdf)*100,
        'total_win':w['pnl'].sum(),'total_loss':l['pnl'].sum(),
        'net':tdf['pnl'].sum(),
        'avg_win':w['pnl'].mean(),'avg_loss':l['pnl'].mean() if len(l)>0 else 0,
        'c':capital,'ret':(capital/CAP-1)*100,
        'dd':max_dd*100,'dca_n':tdf['dca'].sum(),
        'er_counts':tdf['er'].value_counts().to_dict()
    }

# تشغيل
r25 = run_dca_ratio(entry, 25)
r50 = run_dca_ratio(entry, 50)

print(f"📦 {len(df)} candles | 🚦 {entry.sum()} signals")
print(f"\n{'='*80}")
print(f"🐋 LB50 + WMA5/10 + قوة 30% + SMA50 يومي + بدون حجم")
print(f"{'='*80}")

for label, r in [("25/75 DCA", r25), ("50/50 DCA", r50)]:
    if not r: continue
    print(f"\n📋 {label}:")
    print(f"   عدد الصفقات: {r['t']}")
    print(f"   🟢 رابحة: {r['wins']} | 🔴 خاسرة: {r['losses']}")
    print(f"   📈 Win Rate: {r['wr']:.1f}%")
    print(f"   💵 إجمالي الربح: +{r['total_win']:.1f}%")
    print(f"   💸 إجمالي الخسارة: {r['total_loss']:.1f}%")
    print(f"   💰 صافي: {r['net']:+.1f}%")
    print(f"   🟢 متوسط الربح: +{r['avg_win']:.1f}%")
    print(f"   🔴 متوسط الخسارة: {r['avg_loss']:.1f}%")
    print(f"   R:R: {abs(r['avg_win']/r['avg_loss']):.1f}x" if r['avg_loss']!=0 else "   R:R: N/A")
    print(f"   🏦 المحفظة: $1000 → ${r['c']:.0f} ({r['ret']:+.1f}%)")
    print(f"   📉 DD: {r['dd']:.1f}%")
    print(f"   🔄 DCA: {r['dca_n']}")
    print(f"   📊 خروج: {r['er_counts']}")
    # صفقات بالشهر
    months = len(set(df.iloc[500:n]['timestamp'].dt.to_period('M')))
    print(f"   📅 صفقات/شهر: {r['t']/months:.1f}")

print(f"\n✅ تم")
