#!/usr/bin/env python3
"""
SSL صحيح — خط واحد (أزرق/أحمر) + Whale Pump
ADA 1m + اختبار
"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

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

# ── SSL v2: خط واحد ينتقل بين الأزرق والأحمر ──
period=10
sma_high=pd.Series(h).rolling(period).mean().values  # Hlv
sma_low=pd.Series(l_).rolling(period).mean().values   # lv

ssl=np.full(n,np.nan)
ssl_color=np.zeros(n,dtype=int)  # 1=blue(up), -1=red(dn)

for i in range(period,n):
    if h[i-1] > sma_high[i-1]:
        ssl[i]=sma_low[i]
        ssl_color[i]=1  # blue
    else:
        ssl[i]=sma_high[i]
        ssl_color[i]=-1  # red

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

# ── Entry: SSL أزرق + Whale rising ──
le=np.zeros(n,bool)
for i in range(200,n):
    # SSL just turned blue OR already blue + whale rising from near zero
    ssl_blue=ssl_color[i]==1
    ssl_flip=ssl_color[i]==1 and ssl_color[i-1]==-1  # تحول للأزرق
    whale_active=wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0
    
    if ssl_blue and whale_active:
        le[i]=True

print(f'ADA 1m | {n} candles | {le.sum()} signals')

# ── Plot ──
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(18,9),sharex=True,gridspec_kw={'height_ratios':[2.5,1]})
fig.patch.set_facecolor('white')

# Price + SSL
ax1.set_facecolor('white')
colors=['green' if c[i]>=o[i] else 'red' for i in range(n)]
ax1.bar(idx,h-l_,bottom=l_,color=colors,width=0.0003,alpha=0.25)
ax1.bar(idx,abs(c-o),bottom=np.minimum(c,o),color=colors,width=0.00025)

# SSL — single line with color change
blue_mask=ssl_color==1
red_mask=ssl_color==-1
ax1.plot(idx[blue_mask],ssl[blue_mask],'dodgerblue',linewidth=1.5,alpha=0.9)
ax1.plot(idx[red_mask],ssl[red_mask],'tomato',linewidth=1.5,alpha=0.9)

# Entry signals
ax1.scatter(idx[le],c[le],color='lime',s=80,zorder=5,marker='^',edgecolors='darkgreen',linewidth=1,label=f'🐋 شراء ({le.sum()})')
ax1.set_ylabel('ADA/USDT',fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True,alpha=0.2)

# Whale Pump
ax2.set_facecolor('white')
ax2.bar(idx[wp_up & (wp>0)],wp[wp_up & (wp>0)],width=0.0003,color='limegreen',alpha=0.7)
ax2.bar(idx[(~wp_up) & (wp>0)],wp[(~wp_up) & (wp>0)],width=0.0003,color='red',alpha=0.7)
ax2.axhline(y=0,color='gray',linewidth=0.5)
ax2.set_ylabel('Whale Pump',fontweight='bold')
ax2.grid(True,alpha=0.2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))

fig.suptitle('🐋 ADA/USDT 1m — SSL (خط واحد أزرق/أحمر) + Whale Pump (LB50/E3)',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig('/data/trading28/charts/ada_1m_ssl_v2.png',dpi=130,facecolor='white',bbox_inches='tight')
plt.close()

# ── Backtest ──
print(f'\n📊 باك تست:')
print(f'{"TP/SL":>10} {"T":>4} {"WR":>6} {"R:R":>5} {"DD":>6} {"Equity":>9}')
print('-'*45)

for tp,sl in [(0.5,0.25),(0.8,0.4),(1.0,0.5),(1.5,0.75),(2.0,1.0),(3.0,1.5),(5.0,2.5)]:
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                t.append(tp-COMM*100); eq*=(1+(tp-COMM*100)/100); pos=0; cool=12
            elif l_[i]<=ep*(1-sl/100):
                t.append(-sl-COMM*100); eq*=(1+(-sl-COMM*100)/100); pos=0; cool=12
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); cv.append(eq)
    if len(t)<2: continue
    w=[p for p in t if p>0]; lo=[p for p in t if p<=0]
    wr=len(w)/len(t)*100
    aw=np.mean(w) if w else 0; al=abs(np.mean(lo)) if lo else 0
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    ico='+' if eq>CAP else '-'
    print(f'{tp:.1f}%/{sl:.1f}%   {len(t):>3} {wr:>5.1f}% {aw/(al+0.001):>4.2f}x {dd:>5.1f}% {ico}${eq-CAP:>+8.1f}')

print(f'\n✅ Saved: ada_1m_ssl_v2.png')
