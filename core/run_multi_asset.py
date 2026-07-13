"""
Multi-Asset: نفس البارامترات على كل العملات
"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np, os, glob
from core.indicators import *

CAP=1000.0

def run_strategy(df):
    df['timestamp'] = pd.to_datetime(df['ts'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    n=len(df)
    if n < 2000: return None
    
    whale = whale_indicator(df,200)
    entry = (
        whale_spike(whale) & (whale_ma(whale,20) > whale_ma(whale,50)) &
        (whale_strength(whale,50) > 50) & volume_filter(df) &
        (df['close'] > sma50_daily(df))
    )
    sig_count = entry.sum()
    if sig_count == 0: return None
    
    ema=ema21(df); sell=sell_signal(df); sm=swing_lows(df,5)
    capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}
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
                in_trade=False; trade=None
    
    if capital==CAP: return None
    w = capital > CAP
    dd_pct = max_dd*100
    yr_count = len(df)//(365*24*4) or 1  # approximate years
    cagr = ((capital/CAP)**(1/max(yr_count,0.5))-1)*100
    period = f"{df['timestamp'].iloc[0].date()}→{df['timestamp'].iloc[-1].date()}"
    candles = len(df)
    return {'coin':'?','t':0,'wr':0,'c':capital,'ret':(capital/CAP-1)*100,
            'dd':dd_pct,'cagr':cagr,'signals':sig_count,'period':period,'candles':candles}

# ═══════════════════════════════════════════════
cache_dir = '/data/trading28/backtests/cache'
files = sorted(glob.glob(f'{cache_dir}/*_USDT_15m.csv'))

print(f"🔍 {len(files)} عملات للاختبار\n")
print(f"{'='*95}")
print(f"🏆 Multi-Asset | نفس البارامترات بالضبط | 25/75 DCA + PL 60%")
print(f"{'='*95}")
print(f"{'عملة':<8} {'الفترة':<24} {'شمعة':>7} {'إشارات':>7} {'محفظة':>9} {'عائد%':>8} {'DD%':>7} {'CAGR':>7}")
print(f"{'-'*95}")

results = []
winners = 0; losers = 0; total_capital = 0

for f in files:
    coin = os.path.basename(f).replace('_USDT_15m.csv','')
    if coin == 'FET': continue  # نعرضه آخر شي
    try:
        df = pd.read_csv(f)
        r = run_strategy(df)
        if r is None:
            print(f"{coin:<8} {'—':<24} {'—':>7} {'—':>7} {'—':>9} {'—':>8} {'—':>7} {'—':>7} (لا بيانات كافية)")
            continue
        r['coin'] = coin
        results.append(r)
        if r['ret'] > 0: winners += 1
        else: losers += 1
        total_capital += r['c']
        icon = '✅' if r['ret'] > 0 else '❌'
        print(f"{coin:<8} {r['period']:<24} {r['candles']:>7} {r['signals']:>7} ${r['c']:>8.0f} {r['ret']:>7.1f}% {r['dd']:>6.1f}% {r['cagr']:>6.1f}% {icon}")
    except Exception as e:
        print(f"{coin:<8} ERROR: {str(e)[:50]}")

# FET
fet_file = f'{cache_dir}/FET_USDT_15m_FULL.csv'
if os.path.exists(fet_file):
    df_fet = pd.read_csv(fet_file); df_fet['timestamp']=pd.to_datetime(df_fet['ts'])
    df_fet = df_fet.rename(columns={'timestamp':'ts'})
    r_fet = run_strategy(df_fet)
    if r_fet:
        r_fet['coin']='FET'
        results.append(r_fet)
        if r_fet['ret']>0: winners+=1
        else: losers+=1
        total_capital += r_fet['c']
        print(f"{'FET':<8} {r_fet['period']:<24} {r_fet['candles']:>7} {r_fet['signals']:>7} ${r_fet['c']:>8.0f} {r_fet['ret']:>7.1f}% {r_fet['dd']:>6.1f}% {r_fet['cagr']:>6.1f}% {'✅' if r_fet['ret']>0 else '❌'} ⬅")

# Summary
print(f"{'-'*95}")
avg_ret = np.mean([r['ret'] for r in results]) if results else 0
avg_dd = np.mean([r['dd'] for r in results]) if results else 0
print(f"{'ملخص':<8} {f'{len(results)} عملات':<24} {'':>7} {'':>7} {'':>9} {avg_ret:>7.1f}% {avg_dd:>6.1f}% {'':>7}")
print(f"   ✅ ربحانة: {winners} | ❌ خاسرة: {losers} | نسبة نجاح: {winners}/{len(results)} ({winners/len(results)*100:.0f}%)")

print(f"\n✅ تم")
