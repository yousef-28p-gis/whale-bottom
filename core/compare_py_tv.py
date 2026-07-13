"""Python vs TradingView — شرس 7 أيام"""
import sys; sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)
cut = df['timestamp'].max() - pd.Timedelta(days=7)
df = df[df['timestamp'] >= cut].reset_index(drop=True)
CAP=1000.0; n=len(df)

ps=float(df['close'].iloc[0]); pe=float(df['close'].iloc[-1])
print('='*100)
print('🐋 شرس | FET/USDT 15m | {} -> {}'.format(df['timestamp'].iloc[0], df['timestamp'].iloc[-1]))
print('💰 Price: ${:.4f} -> ${:.4f} ({:+.1f}%) | {} candles'.format(ps, pe, (pe/ps-1)*100, n))
print('='*100)

lb=50; wf=3; ws=10; smin=10
whale = whale_indicator(df, lb)
entry = (whale_spike(whale) & (whale_ma(whale, wf) > whale_ma(whale, ws)) & (whale_strength(whale, 50) > smin))
ema=ema21(df); sm=swing_lows(df,5)
capital=CAP; trades=[]; in_trade=False; trade=None; monthly_pnl={}

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
            capital*=(1+eff)
            trades.append({'n':len(trades)+1,'pnl':pnl*100,'er':er,'dca':trade['dca'],'entry_px':trade['e1'],'avg':trade['ae'],'exit_px':epx,'et':trade['et'],'xt':ts,'bars':i-trade['ei']})
            in_trade=False; trade=None

tdf=pd.DataFrame(trades)
if len(tdf)==0:
    print('ZERO TRADES')
    exit()
w=tdf[tdf['pnl']>0]; l=tdf[tdf['pnl']<=0]

print()
print('-'*100)
print('TRADE LOG - compare with TradingView:')
print('-'*100)
print('{:>3} {:<17} {:<17} {:>4} {:>8} {:>8} {:>4} {:>5} {:>7}'.format('#','Entry','Exit','Bars','Avg$','Exit$','DCA','Type','P&L'))
print('-'*100)
for _, t in tdf.iterrows():
    et_str = t['et'].strftime('%m/%d %H:%M')
    xt_str = t['xt'].strftime('%m/%d %H:%M')
    dca_str = 'YES' if t['dca'] else '-'
    print('{:>3} {:<17} {:<17} {:>4} ${:>7.4f} ${:>7.4f} {:>4} {:>5} {:>+6.1f}%'.format(int(t['n']), et_str, xt_str, int(t['bars']), float(t['avg']), float(t['exit_px']), dca_str, t['er'], float(t['pnl'])))

print()
print('='*100)
print('PYTHON BACKTEST SUMMARY')
print('='*100)
print('  Trades: {}'.format(len(tdf)))
print('  Wins: {} | Losses: {}'.format(len(w), len(l)))
print('  Win Rate: {:.1f}%'.format(len(w)/len(tdf)*100))
print('  Total Win: +{:.1f}%'.format(w['pnl'].sum()))
print('  Total Loss: {:.1f}%'.format(l['pnl'].sum()))
print('  Net P&L: {:+.1f}%'.format(tdf['pnl'].sum()))
print('  Avg Win: +{:.1f}%'.format(w['pnl'].mean()))
print('  Avg Loss: {:.1f}%'.format(l['pnl'].mean()))
rr = abs(w['pnl'].mean()/l['pnl'].mean()) if len(l)>0 else 0
print('  R:R: {:.1f}x'.format(rr))
print('  DCA Trades: {}'.format(int(tdf['dca'].sum())))
print('  Portfolio: $1000 -> ${:.0f} ({:+.1f}%)'.format(capital, (capital/1000-1)*100))
er_counts = tdf['er'].value_counts()
for k in ['TP','PL','SL','SL_UP','TIME']:
    if k in er_counts: print('  {}: {}'.format(k, er_counts[k]))

print()
print('-'*100)
print('COMPARE: Open same period on TV -> Sharss code -> match each trade')
print('  DCA in TV is approximate -> expect small diff in avg price')
print('  Key: TP=EMA21[1], SL=Swing Low x 0.998')
print('-'*100)
