#!/usr/bin/env python3
"""FET 1m — Whale+SSL trades chart — TP5/SL2.5"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000

ex=ccxt.binance({'timeout':15000})
since=ex.parse8601((datetime.utcnow()-timedelta(days=7)).isoformat())
all_c=[]
while True:
    batch=ex.fetch_ohlcv('FET/USDT','1m',since=since,limit=1000)
    if not batch: break
    all_c.extend(batch)
    since=batch[-1][0]+1
    if len(batch)<1000: break

df=pd.DataFrame(all_c,columns=['ts','open','high','low','close','volume'])
df['ts']=pd.to_datetime(df['ts'],unit='ms')
df.set_index('ts',inplace=True); df.sort_index(inplace=True)

c=df['close'].values; h=df['high'].values
l_=df['low'].values; o=df['open'].values; n=len(c); idx=df.index

# SSL v2
period=10
sma_h=pd.Series(h).rolling(period).mean().values
sma_l=pd.Series(l_).rolling(period).mean().values
ssl=np.full(n,np.nan); ssl_c=np.zeros(n,int)
for i in range(period,n):
    if h[i-1]>sma_h[i-1]: ssl[i]=sma_l[i]; ssl_c[i]=1
    else: ssl[i]=sma_h[i]; ssl_c[i]=-1

# Whale
LB=50
ln=pd.Series(l_).shift(1).rolling(LB).min().values
lc=np.zeros(n)
for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
sc=pd.Series(lc).ewm(span=3,adjust=False).mean().values
hc=pd.Series(sc).rolling(LB).max().values
sr=np.where(l_<=ln,(sc+hc*2)/3,0)
wp=pd.Series(sr).ewm(span=3,adjust=False).mean().values
wp_up=wp>np.roll(wp,1)

# Entry
le=np.zeros(n,bool)
for i in range(200,n):
    if i>0 and ssl_c[i-1]==1 and wp_up[i-1] and wp[i-1]>wp[max(0,i-3)]*2 and wp[i-1]>0:
        le[i]=True

# Simulate trades
tp_pct=5.0; sl_pct=2.5
trades=[]
pos=0; ep=0; entry_i=0
for i in range(200,n):
    if pos:
        if c[i]>=ep*(1+tp_pct/100):
            trades.append({'entry_i':entry_i,'exit_i':i,'ep':ep,'xp':c[i],'type':'TP'})
            pos=0
        elif c[i]<=ep*(1-sl_pct/100):
            trades.append({'entry_i':entry_i,'exit_i':i,'ep':ep,'xp':c[i],'type':'SL'})
            pos=0
    if not pos and le[i]: pos=1; ep=o[i]; entry_i=i

pnls=[]
for t in trades:
    pnl=(t['xp']/t['ep']-1)*100-COMM*100
    pnls.append(pnl)
print(f'FET 1m | {len(trades)} trades')
for i,t in enumerate(trades):
    print(f'  {i+1}. {t["type"]} | {idx[t["entry_i"]]} → {idx[t["exit_i"]]} | ${t["ep"]:.5f} → ${t["xp"]:.5f} | {pnls[i]:+.2f}%')

# ── Plot full week ──
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,11),sharex=True,gridspec_kw={'height_ratios':[2.5,0.8,0.8]})
fig.patch.set_facecolor('white')

# Price + SSL + trades
ax1.set_facecolor('white')
colors=['green' if c[i]>=o[i] else 'red' for i in range(n)]
ax1.bar(idx,h-l_,bottom=l_,color=colors,width=0.0003,alpha=0.2)
ax1.bar(idx,abs(c-o),bottom=np.minimum(c,o),color=colors,width=0.00025)

# SSL
blue=ssl_c==1; red=ssl_c==-1
ax1.plot(idx[blue],ssl[blue],'dodgerblue',linewidth=1,alpha=0.8)
ax1.plot(idx[red],ssl[red],'tomato',linewidth=1,alpha=0.8)

# Entry signals
ax1.scatter(idx[le],o[le],color='lime',s=40,zorder=5,marker='^',edgecolors='darkgreen',linewidth=0.5,label=f'🐋 ({le.sum()})')

# Trades
for t in trades:
    ei=t['entry_i']; xi=t['exit_i']
    color='gold' if t['type']=='TP' else 'magenta'
    style='-' if t['type']=='TP' else ':'
    ax1.plot([idx[ei],idx[xi]],[t['ep'],t['xp']],color=color,linewidth=3,alpha=0.7,linestyle=style)

ax1.set_ylabel('FET/USDT',fontweight='bold')
ax1.legend(loc='upper left',ncol=2)
ax1.grid(True,alpha=0.15)

# Whale Pump
ax2.set_facecolor('white')
ax2.fill_between(idx,0,wp,where=wp_up&(wp>0),color='limegreen',alpha=0.5)
ax2.fill_between(idx,0,wp,where=(~wp_up)&(wp>0),color='red',alpha=0.5)
ax2.axhline(y=0,color='gray',linewidth=0.5)
ax2.set_ylabel('Whale',fontweight='bold')
ax2.grid(True,alpha=0.15)

# PnL curve
eq_curve=[CAP]; eq=CAP
for i in range(len(trades)):
    pnl=pnls[i]
    eq*=(1+pnl/100)
    eq_curve.append(eq)
eq_x=[idx[0]]+[idx[t['exit_i']] for t in trades]
ax3.set_facecolor('white')
ax3.fill_between(eq_x,1000,eq_curve,where=np.array(eq_curve)>1000,color='green',alpha=0.3)
ax3.fill_between(eq_x,1000,eq_curve,where=np.array(eq_curve)<=1000,color='red',alpha=0.3)
ax3.plot(eq_x,eq_curve,'black',linewidth=2)
ax3.axhline(y=1000,color='gray',linewidth=0.5,linestyle='--')
ax3.set_ylabel('Equity',fontweight='bold')
ax3.grid(True,alpha=0.15)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

fig.suptitle(f'🐋 FET/USDT 1m — Whale+SSL — 7d (-9.8%) — {len(trades)} صفقة — خسارة ${1000-eq:.0f}',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig('/data/trading28/charts/fet_1m_trades.png',dpi=130,facecolor='white',bbox_inches='tight')
plt.close()

print(f'\n✅ Saved: fet_1m_trades.png | Net: ${eq:.1f}')
