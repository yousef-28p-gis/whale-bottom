#!/usr/bin/env python3
"""ADA 1m — Whale Pump + SSL indicators — last 3 days"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings('ignore')

ex=ccxt.binance({'timeout':15000})
since=ex.parse8601((datetime.utcnow()-timedelta(days=3)).isoformat())
all_c=[]
while True:
    batch=ex.fetch_ohlcv('ADA/USDT','1m',since=since,limit=1000)
    if not batch: break
    all_c.extend(batch)
    since=batch[-1][0]+1
    if len(batch)<1000: break

df=pd.DataFrame(all_c,columns=['ts','open','high','low','close','volume'])
df['ts']=pd.to_datetime(df['ts'],unit='ms')
df.set_index('ts',inplace=True); df.sort_index(inplace=True)

c=df['close'].values; h=df['high'].values
l_=df['low'].values; o=df['open'].values; n=len(c); idx=df.index

# ── Whale Pump ──
LB=50
ln=pd.Series(l_).rolling(LB).min().values
at_low=l_<=ln
lc=np.zeros(n)
for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
sc=pd.Series(lc).ewm(span=3,adjust=False).mean().values
hc=pd.Series(sc).rolling(LB).max().values
strength=np.where(at_low,(sc+hc*2)/3,0)
wp=pd.Series(strength).ewm(span=3,adjust=False).mean().values
wp_up=wp>np.roll(wp,1)

# ── SSL ──
ssl_period=10
sup=pd.Series(h).rolling(ssl_period).mean().values
sdn=pd.Series(l_).rolling(ssl_period).mean().values

# ── Plot ──
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(18,9),sharex=True,gridspec_kw={'height_ratios':[2.5,1]})
fig.patch.set_facecolor('white')

# TOP: Price + SSL
ax1.set_facecolor('white')
colors=['green' if c[i]>=o[i] else 'red' for i in range(n)]
ax1.bar(idx,h-l_,bottom=l_,color=colors,width=0.0003,alpha=0.25)
ax1.bar(idx,abs(c-o),bottom=np.minimum(c,o),color=colors,width=0.00025)
ax1.plot(idx,sup,'dodgerblue',linewidth=1.2,label='SSL Up (10)',alpha=0.9)
ax1.plot(idx,sdn,'tomato',linewidth=1.2,label='SSL Dn (10)',alpha=0.9)

# Whale entry markers
le=np.zeros(n,bool)
for i in range(200,n):
    if wp_up[i] and wp[i]>wp[i-2]*2 and c[i]>sup[i]: le[i]=True
ax1.scatter(idx[le],c[le],color='cyan',s=50,zorder=5,marker='D',label=f'🐋 Entry ({le.sum()})')
ax1.set_ylabel('ADA/USDT',fontweight='bold')
ax1.legend(loc='upper left',ncol=3)
ax1.grid(True,alpha=0.2)

# BOTTOM: Whale Pump
ax2.set_facecolor('white')
green_bars=wp_up & (wp>0)
red_bars=(~wp_up) & (wp>0)
ax2.bar(idx[green_bars],wp[green_bars],width=0.0003,color='limegreen',alpha=0.7)
ax2.bar(idx[red_bars],wp[red_bars],width=0.0003,color='red',alpha=0.7)
ax2.axhline(y=0,color='gray',linewidth=0.5)
ax2.set_ylabel('Whale Pump',fontweight='bold')
ax2.grid(True,alpha=0.2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))

fig.suptitle('🐋 ADA/USDT 1m — Whale Pump (LB50/E3) + SSL (10) — آخر 3 أيام',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig('/data/trading28/charts/ada_1m_indicators.png',dpi=130,facecolor='white',bbox_inches='tight')
plt.close()

print(f'✅ Saved | {n} candles | {le.sum()} signals')
