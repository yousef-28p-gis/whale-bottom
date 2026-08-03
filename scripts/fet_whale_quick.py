#!/usr/bin/env python3
"""FET 7d + Whale — smaller PNG"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings; warnings.filterwarnings('ignore')

ex = ccxt.binance({'timeout': 15000})
since = ex.parse8601((datetime.utcnow() - timedelta(days=7)).isoformat())
all_c = []
while True:
    batch = ex.fetch_ohlcv('FET/USDT', '15m', since=since, limit=1000)
    if not batch: break
    all_c.extend(batch)
    since = batch[-1][0] + 1
    if len(batch) < 1000: break

df = pd.DataFrame(all_c, columns=['ts','open','high','low','close','volume'])
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
df.set_index('ts', inplace=True); df.sort_index(inplace=True)

c=df['close'].values; l=df['low'].values; h=df['high'].values; n=len(c)

lookback=200
low_change=np.zeros(n)
for i in range(1,n): low_change[i]=abs(l[i]-l[i-1])/l[i]*100
sc=pd.Series(low_change).ewm(span=3,adjust=False).mean().values
ln=pd.Series(l).rolling(lookback).min().values
hc=pd.Series(sc).rolling(lookback).max().values
sr=np.zeros(n)
for i in range(lookback,n):
    if l[i]<=ln[i]: sr[i]=(sc[i]+hc[i]*2)/3
whale=pd.Series(sr).ewm(span=3,adjust=False).mean().values
w20=pd.Series(whale).rolling(20).apply(lambda x: np.average(x,weights=np.arange(1,21))).values
w50=pd.Series(whale).rolling(50).apply(lambda x: np.average(x,weights=np.arange(1,51))).values
hw=pd.Series(whale).rolling(50).max().values
ws=np.zeros(n)
for i in range(50,n):
    if hw[i]>0: ws[i]=whale[i]/hw[i]*100
spike=np.zeros(n,bool)
for i in range(1,n):
    if whale[i]>whale[i-1] and whale[i-1]<=0.02: spike[i]=True

ema21=pd.Series(c).ewm(span=21,adjust=False).mean().values

fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.5,0.25,0.25],vertical_spacing=0.03)

fig.add_trace(go.Candlestick(x=df.index,open=df['open'],high=df['high'],low=df['low'],close=df['close'],name='FET',showlegend=False),row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=ema21,name='EMA21',line=dict(color='orange',width=1)),row=1,col=1)

si=df.index[spike]; spx=c[spike]
fig.add_trace(go.Scatter(x=si,y=spx,mode='markers',marker=dict(symbol='triangle-up',size=10,color='cyan'),name='🐋'),row=1,col=1)

fig.add_trace(go.Scatter(x=df.index,y=ws,name='Strength%',line=dict(color='purple',width=1)),row=2,col=1)
fig.add_hline(y=50,line_dash='dash',line_color='gray',row=2,col=1)

fig.add_trace(go.Scatter(x=df.index,y=whale,name='Whale',line=dict(color='cyan',width=1.5)),row=3,col=1)
fig.add_trace(go.Scatter(x=df.index,y=w20,name='WMA20',line=dict(color='lime',width=1)),row=3,col=1)
fig.add_trace(go.Scatter(x=df.index,y=w50,name='WMA50',line=dict(color='red',width=1)),row=3,col=1)

fig.update_layout(title='🐋 الحوت الخام — FET 15m آخر 7 أيام',template='plotly_white',height=700,hovermode='x unified',xaxis_rangeslider_visible=False,margin=dict(l=50,r=20,t=50,b=20))
fig.write_image('/data/trading28/charts/fet_whale_7d.png',width=1200,height=700,scale=1)
print(f'Done: {n} candles, {spike.sum()} spikes')
