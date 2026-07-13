"""
مقارنة عادلة: الأساس vs تعزيز ذكي
"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)
CAP=1000.0; n=len(df)
whale_raw = whale_indicator(df,200)

entry = (
    whale_spike(whale_raw) & (whale_ma(whale_raw,20) > whale_ma(whale_raw,50)) &
    (whale_strength(whale_raw,50) > 50) & volume_filter(df) &
    (df['close'] > sma50_daily(df))
)
ema=ema21(df); sell=sell_signal(df); sm=swing_lows(df,5)

def run_simple_dca():
    """25/75 DCA + PL 60% + 0.3% trail + 4hr"""
    capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}; trades=[]
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
                trades.append({'pnl':pnl*100,'eff':eff*100,'er':er})
                in_trade=False; trade=None
    tdf=pd.DataFrame(trades)
    if len(tdf)==0: return {}
    w=tdf[tdf['pnl']>0]; wr=len(w)/len(tdf)*100
    rets=tdf['eff'].values/100
    sh=rets.mean()/rets.std()*np.sqrt(len(rets)) if rets.std()>0 else 0
    return {'t':len(tdf),'wr':wr,'c':capital,'ret':(capital/CAP-1)*100,'dd':max_dd*100,'sh':sh}

def run_smart_dca():
    """25→35→40 + شرط الحوت يرتفع"""
    capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}; trades=[]
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
                trade={'ei':i,'et':ts,'e1':ep,'e2':None,'e3':None,'ae':ep,'al':0.25,
                       'sl':sl,'tp':tp,'pl':pl,'pl_act':False,'hi':ep,
                       'dca1':False,'dca2':False}
                in_trade=True
        else:
            if row['high']>trade['hi']: trade['hi']=row['high']
            if not trade['pl_act'] and row['high']>=trade['pl']: trade['pl_act']=True
            
            # Smart DCA
            s2=max(0,trade['ei']+1); ns=df.iloc[s2:i+1][sm[s2:i+1]]
            if len(ns)>0 and ns['low'].min()<trade['e1']:
                lowest=ns['low'].min()
                whale_rising = whale_raw.iloc[i] > whale_raw.iloc[i-1]
                
                if not trade['dca1'] and whale_rising:
                    trade['e2']=row['close']
                    trade['ae']=(trade['e1']*25+trade['e2']*35)/60
                    trade['al']=0.60; trade['dca1']=True
                    trade['sl']=lowest*0.998
                    trade['pl']=trade['ae']+(trade['tp']-trade['ae'])*60/100
                    if row['high']>=trade['pl']: trade['pl_act']=True
                
                elif trade['dca1'] and not trade['dca2'] and lowest<trade['e2'] and whale_rising:
                    trade['e3']=row['close']
                    trade['ae']=(trade['e1']*25+trade['e2']*35+trade['e3']*40)/100
                    trade['al']=1.0; trade['dca2']=True
                    trade['sl']=lowest*0.998
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
                trades.append({'pnl':pnl*100,'eff':eff*100,'er':er,'dca1':trade['dca1'],'dca2':trade['dca2']})
                in_trade=False; trade=None
    tdf=pd.DataFrame(trades)
    if len(tdf)==0: return {}
    w=tdf[tdf['pnl']>0]; wr=len(w)/len(tdf)*100
    rets=tdf['eff'].values/100
    sh=rets.mean()/rets.std()*np.sqrt(len(rets)) if rets.std()>0 else 0
    dca1=tdf['dca1'].sum(); dca2=tdf['dca2'].sum()
    return {'t':len(tdf),'wr':wr,'c':capital,'ret':(capital/CAP-1)*100,'dd':max_dd*100,'sh':sh,
            'dca1':dca1,'dca2':dca2}

# ═══════════════════════════════════════════════
print(f"📦 {len(df)} candles | 🚦 {entry.sum()} signals\n")

r1 = run_simple_dca()
r2 = run_smart_dca()

print(f"{'='*90}")
print(f"🏆 الأساس (25/75) vs التعزيز الذكي (25→35→40 + شرط الحوت)")
print(f"{'='*90}")
print(f"{'الاستراتيجية':<35} {'صفقات':>6} {'WR%':>6} {'المحفظة':>10} {'عائد%':>8} {'DD%':>7} {'Sharpe':>7}")
print(f"{'-'*90}")
for name, r in [('1️⃣ 25/75 DCA (الأساس)', r1), ('2️⃣ تعزيز ذكي (25→35→40)', r2)]:
    dca_info = f" | DCA1={r.get('dca1','?')} DCA2={r.get('dca2','?')}" if 'dca1' in r else ''
    print(f"{name:<35} {r['t']:>6} {r['wr']:>5.1f}% ${r['c']:>9.0f} {r['ret']:>7.1f}% {r['dd']:>6.1f}% {r['sh']:>6.2f}{dca_info}")

print(f"\n✅ تم")
