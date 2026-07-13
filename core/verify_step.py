"""تحقق من التراكم خطوة بخطوة"""
import sys; sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)
cut = df['timestamp'].max() - pd.Timedelta(days=7)
df = df[df['timestamp'] >= cut].reset_index(drop=True)
n=len(df)

lb=50; wf=3; ws=10; smin=10
whale = whale_indicator(df, lb)
entry = (whale_spike(whale) & (whale_ma(whale, wf) > whale_ma(whale, ws)) & (whale_strength(whale, 50) > smin))
ema=ema21(df); sm=swing_lows(df,5)
capital=1000.0; trades=[]; in_trade=False; trade=None; monthly_pnl={}

for i in range(200,n):
    row=df.iloc[i]; ts=row['timestamp']
    mk='{}-{:02d}'.format(ts.year, ts.month)
    if monthly_pnl.get(mk,0.0)<=-7 and not in_trade: continue
    if not in_trade:
        if entry.iloc[i] and not pd.isna(ema.iloc[i-1]) and ema.iloc[i-1]>row['close']:
            sl=df.iloc[max(0,i-60):i][sm[max(0,i-60):i]]['low'].min()*0.998 if sm[max(0,i-60):i].sum()>0 else row['close']*0.95
            trade={'ei':i,'et':ts,'e1':row['close'],'ae':row['close'],'al':0.25,'sl':sl,'tp':ema.iloc[i-1],'pl_act':False,'hi':row['close'],'dca':False}
            trade['pl']=row['close']+(trade['tp']-row['close'])*60/100; in_trade=True
    else:
        if row['high']>trade['hi']: trade['hi']=row['high']
        if not trade['pl_act'] and row['high']>=trade['pl']: trade['pl_act']=True
        if not trade['dca']:
            s2=max(0,trade['ei']+1); ns=df.iloc[s2:i+1][sm[s2:i+1]]
            if len(ns)>0 and ns['low'].min()<trade['e1']:
                trade['ae']=(trade['e1']*25+row['close']*75)/100; trade['al']=1.0; trade['dca']=True
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
        elif sl_h:
            if trade['pl_act']: er,epx='PL',max(trade['sl'],row['low'])
            elif trade['sl']<=trade['ae']: er,epx='SL',max(trade['sl'],row['low'])
            else: er,epx='SL_UP',min(trade['sl'],row['high'])
        elif hrs>=4: er,epx='TIME',row['close']
        if er:
            pnl=(epx-trade['ae'])/trade['ae']-0.002; eff=pnl*trade['al']
            monthly_pnl[mk]=monthly_pnl.get(mk,0.0)+eff*100
            trades.append({'pnl_pct':pnl*100,'eff':eff*100,'er':er,'dca':trade['dca'],'al':trade['al']})
            capital*=(1+eff)
            in_trade=False; trade=None

tdf=pd.DataFrame(trades)

print('='*75)
print('VERIFICATION - Step by step compounding:')
print('='*75)
print('{:>3} {:>2} {:>5} {:>6} {:>8} {:>14} {:>14}'.format('#','','Type','DCA','PnL%','Capital Before','Capital After'))
print('-'*75)

cap = 1000.0
sum_pnl = 0.0
for i, (_, t) in enumerate(tdf.iterrows()):
    pct = float(t['eff'])
    cap_before = cap
    cap *= (1 + pct/100)
    sum_pnl += float(t['pnl_pct'])
    sig = '+' if t['pnl_pct'] > 0 else ''
    dca = 'YES' if t['dca'] else '-'
    print('{:>3} {} {:>5} {:>6} {:>+6.1f}% ${:>12.2f} ${:>12.2f}'.format(
        i+1, 'W' if t['pnl_pct']>0 else 'L', t['er'], dca, float(t['pnl_pct']), cap_before, cap))

print('='*75)
print('Total PnL (sum): {:+.1f}%'.format(sum_pnl))
print('Final Capital: $1000.00 -> ${:.2f} ({:+.1f}%)'.format(cap, (cap/1000-1)*100))
print()

# WHY it works
wins = tdf[tdf['pnl_pct']>0]['pnl_pct']
losses = tdf[tdf['pnl_pct']<=0]['pnl_pct']
print('KEY INSIGHT:')
print('  Avg Win:  +{:.1f}%  x {} wins  = +{:.1f}%'.format(wins.mean(), len(wins), wins.sum()))
print('  Avg Loss: {:.1f}%  x {} losses = {:.1f}%'.format(losses.mean(), len(losses), losses.sum()))
print('  Net = {:.1f}% + ({:.1f}%) = {:+.1f}%'.format(wins.sum(), losses.sum(), wins.sum()+losses.sum()))
print()
print('Even with MORE losses ({} vs {}), wins are BIGGER -> net positive'.format(len(losses), len(wins)))
