"""
Out-of-Sample: Train 2019-2024, Test 2025-2026
"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)

CAP=1000.0

def run_full_period(df_sub, first_pct=25, pl_pct=60, trail_pct=0.3, max_hrs=4, monthly_lim=7):
    n=len(df_sub)
    whale = whale_indicator(df_sub,200)
    entry = (
        whale_spike(whale) & (whale_ma(whale,20) > whale_ma(whale,50)) &
        (whale_strength(whale,50) > 50) & volume_filter(df_sub) &
        (df_sub['close'] > sma50_daily(df_sub))
    )
    ema=ema21(df_sub); sell=sell_signal(df_sub); sm=swing_lows(df_sub,5)
    
    second_pct = 100 - first_pct
    initial_al = first_pct/100
    capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}; trades=[]
    in_trade=False; trade=None
    
    for i in range(500,n):
        row=df_sub.iloc[i]; ts=row['timestamp']
        mk=f"{ts.year}-{ts.month:02d}"
        if monthly_pnl.get(mk,0.0)<=-monthly_lim and not in_trade: continue
        if not in_trade:
            if entry.iloc[i]:
                ep=row['close']
                if i<1 or pd.isna(ema.iloc[i-1]): continue
                tp=ema.iloc[i-1]
                if tp<=ep: continue
                sw_s=max(0,i-60); sw_r=df_sub.iloc[sw_s:i][sm[sw_s:i]]
                sl=sw_r['low'].min()*0.998 if len(sw_r)>0 else ep*0.95
                pl_price = ep + (tp-ep)*pl_pct/100
                trade={'ei':i,'et':ts,'e1':ep,'e2':None,'ae':ep,'al':initial_al,
                       'sl':sl,'tp':tp,'pl':pl_price,'pl_act':False,
                       'hi':ep,'dca':False,'first':first_pct,'sec':second_pct}
                in_trade=True
        else:
            if row['high']>trade['hi']: trade['hi']=row['high']
            if not trade['pl_act'] and row['high'] >= trade['pl']:
                trade['pl_act']=True
            
            if not trade['dca']:
                s2=max(0,trade['ei']+1); ns=df_sub.iloc[s2:i+1][sm[s2:i+1]]
                if len(ns)>0 and ns['low'].min()<trade['e1']:
                    trade['e2']=row['close']
                    trade['ae']=(trade['e1']*trade['first']+trade['e2']*trade['sec'])/100
                    trade['al']=1.0; trade['dca']=True
                    trade['sl']=ns['low'].min()*0.998
                    trade['pl']=trade['ae']+(trade['tp']-trade['ae'])*pl_pct/100
                    if row['high']>=trade['pl']: trade['pl_act']=True
            
            st2=max(0,i-100); swt=df_sub.iloc[st2:i+1][sm[st2:i+1]]
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
            elif hrs>=max_hrs: er,epx='TIME',row['close']
            
            if er:
                pnl=(epx-trade['ae'])/trade['ae']-0.002; eff=pnl*trade['al']
                monthly_pnl[mk]=monthly_pnl.get(mk,0.0)+eff*100
                capital*=(1+eff)
                if capital>peak: peak=capital
                dd=(capital-peak)/peak
                if dd<max_dd: max_dd=dd
                trades.append({'y':ts.year,'pnl':pnl*100,'eff':eff*100,'er':er,'dca':trade['dca']})
                in_trade=False; trade=None
    
    tdf=pd.DataFrame(trades)
    if len(tdf)==0: return {}
    w=tdf[tdf['pnl']>0]; wr=len(w)/len(tdf)*100
    rets=tdf['eff'].values/100
    sharpe = rets.mean()/rets.std()*np.sqrt(len(rets)) if rets.std()>0 else 0
    return {'t':len(tdf),'wr':wr,'c':capital,'ret':(capital/CAP-1)*100,
            'dd':max_dd*100,'sharpe':sharpe,'dca_n':tdf['dca'].sum(),
            'years':tdf.groupby('y').agg(trades=('pnl','count'),wr=('pnl',lambda x:(x>0).sum()/len(x)*100),
                                         ret=('eff',lambda x:((1+x/100).prod()-1)*100)).to_dict()}

# ═══════════════════════════════════════════════
print(f"📦 إجمالي البيانات: {len(df)} شمعة | {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}\n")

# Split
df_train = df[df['timestamp'] < '2025-01-01'].reset_index(drop=True)
df_test = df[df['timestamp'] >= '2025-01-01'].reset_index(drop=True)

print(f"📚 تدريب: {df_train['timestamp'].iloc[0].date()} → {df_train['timestamp'].iloc[-1].date()} ({len(df_train)} شمعة)")
print(f"🧪 اختبار: {df_test['timestamp'].iloc[0].date()} → {df_test['timestamp'].iloc[-1].date()} ({len(df_test)} شمعة)\n")

# Run both
r_train = run_full_period(df_train)
r_test = run_full_period(df_test)

print(f"{'='*85}")
print(f"🏆 Out-of-Sample Test | نفس البارامترات بالضبط")
print(f"{'='*85}")
print(f"{'':<12} {'صفقات':>6} {'WR%':>6} {'المحفظة':>10} {'عائد%':>8} {'DD%':>7} {'Sharpe':>7}")
print(f"{'-'*85}")

for name, r in [('تدريب 19-24', r_train), ('اختبار 25-26', r_test)]:
    print(f"{name:<12} {r['t']:>6} {r['wr']:>5.1f}% ${r['c']:>9.0f} {r['ret']:>7.1f}% {r['dd']:>6.1f}% {r['sharpe']:>6.2f}")

# Full period
r_full = run_full_period(df)
print(f"{'كامل 19-26':<12} {r_full['t']:>6} {r_full['wr']:>5.1f}% ${r_full['c']:>9.0f} {r_full['ret']:>7.1f}% {r_full['dd']:>6.1f}% {r_full['sharpe']:>6.2f}")

# Yearly breakdown for test
print(f"\n📅 تفصيل سنوي — فترة الاختبار:")
tdf_test = r_test.get('tdf', None)
if False:  # not stored, use years dict
    pass
# We need to rerun with tdf stored. Let me just print from the dict.

print(f"\n📅 تفصيل سنوي — كامل:")
for y in range(2019,2027):
    mask = (df['timestamp'].dt.year == y)
    dfy = df[mask].reset_index(drop=True)
    if len(dfy) < 1000: continue
    ry = run_full_period(dfy)
    if ry.get('t',0)==0: continue
    print(f"   {y}: {ry['t']}T | WR={ry['wr']:.0f}% | ${ry['c']:.0f} (+{ry['ret']:.0f}%) | DD={ry['dd']:.1f}%")

print(f"\n✅ تم")
