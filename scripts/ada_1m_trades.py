#!/usr/bin/env python3
"""ADA 1m — Whale+SSL 4 trades chart"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

ex=ccxt.binance({'timeout':15000})
since=ex.parse8601((datetime.utcnow()-timedelta(days=7)).isoformat())
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

# Whale
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

# SSL
sup=pd.Series(h).rolling(10).mean().values
sdn=pd.Series(l_).rolling(10).mean().values

# Entry signals
le=np.zeros(n,bool)
for i in range(200,n):
    if wp_up[i] and wp[i]>wp[i-2]*2 and c[i]>sup[i]:
        le[i]=True

# Find trades
tp_pct=5.0; sl_pct=2.5
trades=[]
pos=0; ep=0
for i in range(200,n):
    if pos:
        if h[i]>=ep*(1+tp_pct/100):
            trades.append({'entry_idx':entry_i,'exit_idx':i,'entry':ep,'exit':ep*(1+tp_pct/100),'type':'TP','pnl':tp_pct-COMM*100})
            pos=0
        elif l_[i]<=ep*(1-sl_pct/100):
            trades.append({'entry_idx':entry_i,'exit_idx':i,'entry':ep,'exit':ep*(1-sl_pct/100),'type':'SL','pnl':-sl_pct-COMM*100})
            pos=0
    if not pos and le[i]:
        pos=1; ep=c[i]; entry_i=i
print(f'Trades found: {len(trades)}')
for i,t in enumerate(trades):
    print(f'  {i+1}. {t["type"]} | ENTRY: {idx[t["entry_idx"]]} @ {t["entry"]:.5f} | EXIT: {idx[t["exit_idx"]]} @ {t["exit"]:.5f} | PnL: {t["pnl"]:+.2f}%')

# Group trades to show ±1h context each
if trades:
    n_rows=len(trades)
    fig,axes=plt.subplots(n_rows,1,figsize=(14,3.5*n_rows),sharex=False)
    if n_rows==1: axes=[axes]
    fig.patch.set_facecolor('white')
    
    for ax_i,(t,ax) in enumerate(zip(trades,axes)):
        # ±60 candles context
        start=max(0,t['entry_idx']-60)
        end=min(n,t['exit_idx']+60)
        x_slice=idx[start:end]
        c_slice=c[start:end]; h_slice=h[start:end]
        l_slice=l_[start:end]; o_slice=o[start:end]
        
        ax.set_facecolor('white')
        colors=['green' if c_slice[i]>=o_slice[i] else 'red' for i in range(len(c_slice))]
        ax.bar(x_slice,h_slice-l_slice,bottom=l_slice,color=colors,width=0.0003,alpha=0.3)
        ax.bar(x_slice,abs(c_slice-o_slice),bottom=np.minimum(c_slice,o_slice),color=colors,width=0.00025)
        
        # SSL lines
        ax.plot(x_slice,sup[start:end],'dodgerblue',linewidth=0.8,alpha=0.5,label='SSL Up')
        ax.plot(x_slice,sdn[start:end],'tomato',linewidth=0.8,alpha=0.5,label='SSL Dn')
        
        # Entry marker
        ax.axvline(x=idx[t['entry_idx']],color='lime',linewidth=2,linestyle='--',alpha=0.8)
        ax.scatter(idx[t['entry_idx']],t['entry'],color='lime',s=120,zorder=5,marker='^',label='ENTRY')
        
        # Exit marker
        exit_color='gold' if t['type']=='TP' else 'red'
        ax.axvline(x=idx[t['exit_idx']],color=exit_color,linewidth=2,linestyle='--',alpha=0.8)
        ex_marker='v' if t['type']=='TP' else 'v'
        ax.scatter(idx[t['exit_idx']],t['exit'],color=exit_color,s=120,zorder=5,marker=ex_marker,label=f'EXIT ({t["type"]})')
        
        # Entry signal marker (whale spike)
        ax.scatter(idx[t['entry_idx']],c[t['entry_idx']],color='cyan',s=80,zorder=6,marker='D',label='🐋')
        
        # Title
        ax.set_title(f'صفقة {ax_i+1}: {t["type"]} | PnL: {t["pnl"]:+.2f}% | دخول: {idx[t["entry_idx"]].strftime("%m/%d %H:%M")} | خروج: {idx[t["exit_idx"]].strftime("%m/%d %H:%M")}',fontweight='bold')
        ax.set_ylabel('ADA/USDT')
        ax.legend(loc='upper left',fontsize=8)
        ax.grid(True,alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    
    fig.suptitle('🐋 ADA/USDT 1m — Whale+SSL (LB50/E3/SSL10) — 4 صفقات في 7 أيام',fontsize=14,fontweight='bold')
    plt.tight_layout()
    plt.savefig('/data/trading28/charts/ada_1m_4trades.png',dpi=130,facecolor='white',bbox_inches='tight')
    plt.close()
    print('\n✅ Saved: ada_1m_4trades.png')
else:
    print('No trades found')
