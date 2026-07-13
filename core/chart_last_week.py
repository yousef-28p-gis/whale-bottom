"""رسم صفقات آخر أسبوع"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)

# آخر أسبوع
cut = df['timestamp'].max() - pd.Timedelta(days=7)
df = df[df['timestamp'] >= cut].reset_index(drop=True)
CAP=1000.0; n=len(df)

# شرس LB50 بدون SMA50 (الوحيد اللي عنده صفقات حديثة)
lb=50; wf=3; ws=10; smin=10; use_vol=False; use_sma50=False
whale = whale_indicator(df, lb)
entry = (whale_spike(whale) & (whale_ma(whale, wf) > whale_ma(whale, ws)) &
         (whale_strength(whale, 50) > smin))
if use_vol: entry &= volume_filter(df)
if use_sma50: entry &= (df['close'] > sma50_daily(df))

ema=ema21(df); sell=sell_signal(df); sm=swing_lows(df,5)
capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}; trades=[]
in_trade=False; trade=None

for i in range(200,n):
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
            trades.append({'ei':trade['ei'],'xi':i,'pnl':pnl*100,'er':er,'dca':trade['dca'],
                          'ae':trade['ae'],'epx':epx,'et':trade['et'],'xt':ts})
            in_trade=False; trade=None

tdf=pd.DataFrame(trades)
print(f"📅 {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
print(f"📦 {n} candle | 🚦 {entry.sum()} signals | 💼 {len(tdf)} trades")
if len(tdf)>0:
    w=tdf[tdf['pnl']>0]
    print(f"📈 WR: {len(w)/len(tdf)*100:.0f}% | 💰 صافي: {tdf['pnl'].sum():+.1f}%")
    print(f"🟢 {len(w)} | 🔴 {len(tdf)-len(w)}")
    print(f"خروج: {tdf['er'].value_counts().to_dict()}")

# ═══════════════ رسم ═══════════════
fig = make_subplots(rows=1, cols=1)

# شموع
fig.add_trace(go.Candlestick(
    x=df['timestamp'], open=df['open'], high=df['high'],
    low=df['low'], close=df['close'], name='FET/USDT',
    increasing_line_color='#00ff88', decreasing_line_color='#ff4466'
))

# EMA21
fig.add_trace(go.Scatter(x=df['timestamp'], y=ema, name='EMA21',
    line=dict(color='orange', width=1), opacity=0.7))

# إشارات دخول
entry_idx = df[entry].index
fig.add_trace(go.Scatter(
    x=df.iloc[entry_idx]['timestamp'], y=df.iloc[entry_idx]['low']*0.998,
    mode='markers', name='🐋 دخول', marker=dict(symbol='triangle-up', size=10, color='cyan')
))

# صفقات
for _, t in tdf.iterrows():
    color = '#00ff88' if t['pnl'] > 0 else '#ff4466'
    fig.add_trace(go.Scatter(
        x=[df.iloc[int(t['ei'])]['timestamp'], df.iloc[int(t['xi'])]['timestamp']],
        y=[t['ae'], t['epx']],
        mode='lines+markers', name='',
        line=dict(color=color, width=2), marker=dict(size=5),
        showlegend=False,
        hovertemplate=f"{t['er']} | {t['pnl']:+.1f}%<extra></extra>"
    ))

fig.update_layout(
    title=f"🐋 آخر أسبوع | FET/USDT 15m | {len(tdf)} صفقة | WR {len(w)/len(tdf)*100:.0f}%",
    xaxis_title='', yaxis_title='سعر',
    template='plotly_dark', height=700,
    xaxis_rangeslider_visible=False,
    hovermode='x unified'
)

fig.write_html('/data/trading28/backtests/charts/last_week.html')
print(f"\n✅ محفوظ: /data/trading28/backtests/charts/last_week.html")
