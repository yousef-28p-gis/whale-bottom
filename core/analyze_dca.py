"""
تحليل DCA: مع DCA vs بدون DCA (100% من البداية) — Profit Lock 60% + 0.3%
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

def run_full(entry_signal, use_dca=True, use_pl=True, pl_pct=60, trail_pct=0.3):
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
                al = 0.5 if use_dca else 1.0
                pl_price = ep + (tp-ep)*pl_pct/100 if use_pl else tp*2
                trade={'ei':i,'et':ts,'e1':ep,'e2':None,'ae':ep,'al':al,
                       'sl':sl,'tp':tp,'pl':pl_price,'pl_act':False,
                       'hi':ep,'dca':False,'use_dca':use_dca}
                in_trade=True
        else:
            if row['high']>trade['hi']: trade['hi']=row['high']
            if use_pl and not trade['pl_act'] and row['high'] >= trade['pl']:
                trade['pl_act']=True
            
            if use_dca and not trade['dca']:
                s2=max(0,trade['ei']+1); ns=df.iloc[s2:i+1][sm[s2:i+1]]
                if len(ns)>0 and ns['low'].min()<trade['e1']:
                    trade['e2']=row['close']; trade['ae']=(trade['e1']+trade['e2'])/2
                    trade['al']=1.0; trade['dca']=True
                    trade['sl']=ns['low'].min()*0.998
                    if use_pl:
                        trade['pl']=trade['ae']+(trade['tp']-trade['ae'])*pl_pct/100
                        if row['high']>=trade['pl']: trade['pl_act']=True
            
            st2=max(0,i-100); swt=df.iloc[st2:i+1][sm[st2:i+1]]
            if len(swt)>0:
                nsl=swt['low'].min()*0.998
                if nsl>trade['sl']: trade['sl']=nsl
            
            if trade['pl_act']:
                trail_sl = trade['hi'] * (1 - trail_pct/100)
                if trail_sl > trade['sl']: trade['sl'] = trail_sl
            
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
                trades.append({
                    'pnl':pnl*100,'eff_pnl':eff*100,'er':er,
                    'dca':trade['dca'],'pl':trade['pl_act'],
                    'al':trade['al']*100
                })
                in_trade=False; trade=None
    
    tdf=pd.DataFrame(trades)
    if len(tdf)==0: return {}
    w=tdf[tdf['pnl']>0]; wr=len(w)/len(tdf)*100
    return {'t':len(tdf),'wr':wr,'c':capital,'ret':(capital/CAP-1)*100,
            'dd':max_dd*100,'tdf':tdf}

# ═══════════════════════════════════════════════
print("⏳ تشغيل ٣ سيناريوهات...")

r1 = run_full(entry, use_dca=True, use_pl=False)   # DCA فقط
r2 = run_full(entry, use_dca=False, use_pl=True)    # PL فقط (100%)
r3 = run_full(entry, use_dca=True, use_pl=True)     # DCA + PL

print(f"\n{'='*85}")
print(f"🏆 تحليل DCA — مع/بدون تعزيز — مع/بدون Profit Lock")
print(f"{'='*85}")
print(f"{'الاستراتيجية':<30} {'صفقات':>6} {'WR%':>6} {'المحفظة':>10} {'عائد%':>8} {'DD%':>7}")
print(f"{'-'*85}")
for name, r in [('DCA فقط',r1), ('PL فقط (100%)',r2), ('DCA + PL ⬅',r3)]:
    print(f"{name:<30} {r['t']:>6} {r['wr']:>5.1f}% ${r['c']:>9.0f} {r['ret']:>7.1f}% {r['dd']:>6.1f}%")

# ── تحليل DCA داخل r3 (DCA+PL) ──
tdf = r3['tdf']
dca_yes = tdf[tdf['dca']==True]
dca_no = tdf[tdf['dca']==False]

print(f"\n{'='*85}")
print(f"🔬 تحليل التعزيز داخل DCA + PL")
print(f"{'='*85}")
print(f"\n📊 صفقات بتعزيز: {len(dca_yes)} ({len(dca_yes)/len(tdf)*100:.0f}%)")
print(f"   WR: {len(dca_yes[dca_yes['pnl']>0])/len(dca_yes)*100:.1f}%")
print(f"   متوسط PnL: {dca_yes['eff_pnl'].mean():.2f}%")
print(f"   ربحانة: {len(dca_yes[dca_yes['pnl']>0])} | خاسرة: {len(dca_yes[dca_yes['pnl']<=0])}")

print(f"\n📊 صفقات بدون تعزيز: {len(dca_no)} ({len(dca_no)/len(tdf)*100:.0f}%)")
print(f"   WR: {len(dca_no[dca_no['pnl']>0])/len(dca_no)*100:.1f}%")
print(f"   متوسط PnL: {dca_no['eff_pnl'].mean():.2f}%")
print(f"   ربحانة: {len(dca_no[dca_no['pnl']>0])} | خاسرة: {len(dca_no[dca_no['pnl']<=0])}")

# ── DCA: ماذا كان سيحدث بدون تعزيز؟ ──
# نقارن مع سيناريو PL فقط (100% allocation)
tdf_plonly = r2['tdf']
# نفس الصفقات — قارن النتيجة
dca_wins = dca_yes[dca_yes['pnl']>0]
dca_loss = dca_yes[dca_yes['pnl']<=0]

print(f"\n{'='*85}")
print(f"🔬 مكسب/خسارة التعزيز")
print(f"{'='*85}")

# PL only gives 100% allocation, DCA+PL gives 50%→100%
# DCA trades have higher effective exposure after DCA
print(f"\n💰 الصفقات اللي تعززت:")
print(f"   ربحانة: {len(dca_wins)} | متوسط ربح فعّال: {dca_wins['eff_pnl'].mean():.2f}%")
print(f"   خاسرة: {len(dca_loss)} | متوسط خسارة فعّالة: {dca_loss['eff_pnl'].mean():.2f}%")

# مقارنة: لو كان 100% من البداية بدون DCA
# pnl_pct في dca_yes = ربح/خسارة كنسبة من متوسط الدخول
# effective_pnl = pnl * 1.0 (لأن DCA يكمل لـ 100%)
# بدون DCA (100% من البداية): effective = pnl_from_entry1 * 1.0
# 
# للتبسيط: متوسط eff_pnl للصفقات المعززة vs الصفقات المشابهة في PL only

print(f"\n📈 خرجوا بـ:")
for er in ['TP','PL','SL','SL_UP','SELL','TIME']:
    cnt = len(dca_yes[dca_yes['er']==er])
    if cnt>0: print(f"   {er}: {cnt}")

print(f"\n✅ تم")
