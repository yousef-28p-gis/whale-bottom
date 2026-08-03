#!/usr/bin/env python3
"""FET 1m — دخول عند أول حوت بعد SSL يتحول أزرق"""
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
p=10
sma_h=pd.Series(h).rolling(p).mean().values
sma_l=pd.Series(l_).rolling(p).mean().values
ssl=np.full(n,np.nan); ssl_c=np.zeros(n,int)
for i in range(p,n):
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

# ── دخول: أول حوت بعد SSL يتحول لأزرق ──
le=np.zeros(n,bool)
ssl_flip_blue=np.zeros(n,bool)   # SSL flipped to blue
ssl_flip_red=np.zeros(n,bool)     # SSL flipped to red

for i in range(1,n):
    if ssl_c[i]==1 and ssl_c[i-1]==-1:
        ssl_flip_blue[i]=True
    if ssl_c[i]==-1 and ssl_c[i-1]==1:
        ssl_flip_red[i]=True

# بعد SSL يتحول أزرق → انتظر أول حوت صاعد → ادخل
waiting_for_whale=False
for i in range(200,n):
    if ssl_flip_blue[i]:
        waiting_for_whale=True
    if waiting_for_whale and wp_up[i] and wp[i]>wp[i-2]*1.5 and wp[i]>0:
        le[i]=True
        waiting_for_whale=False  # دخلنا — ما ننتظر إشارة ثانية
    if ssl_flip_red[i]:
        waiting_for_whale=False  # SSL رجع أحمر — إلغاء الانتظار

p_change=(c[-1]-c[0])/c[0]*100
print(f'FET 1m | 7d | n={n} | Δ {p_change:+.1f}%')
print(f'SSL flips blue: {ssl_flip_blue.sum()} | red: {ssl_flip_red.sum()}')
print(f'Entries: {le.sum()} (القديم كان 65!)')
print(f'{"TP/SL":>10} {"T":>4} {"WR":>6} {"DD":>6} {"Eq":>9} {"W/L":>8}')
print('-'*50)

for tp,sl in [(2.0,1.0),(3.0,1.5),(5.0,2.5)]:
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if c[i]>=ep*(1+tp/100):
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=30
            elif c[i]<=ep*(1-sl/100):
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=30
        if not pos and cool==0 and le[i]: pos=1; ep=o[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); cv.append(eq)
    if len(t)<2: continue
    w=[p for p in t if p>0]; lo=[p for p in t if p<=0]
    wr=len(w)/len(t)*100
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    ico='✅' if eq>CAP else '❌'
    print(f'{tp:.1f}%/{sl:.1f}%   {len(t):>3} {wr:>5.1f}% {dd:>5.1f}% {ico}${eq-CAP:>+8.1f} {len(w)}W/{len(lo)}L')

# Show trades
print(f'\nتفاصيل الصفقات (TP5/SL2.5):')
t=[]; pos=0; ep=0; entry_i=0; trades=[]
for i in range(200,n):
    if pos:
        if c[i]>=ep*(1+5/100):
            trades.append({'ei':entry_i,'xi':i,'ep':ep,'xp':c[i],'t':'TP'})
            pos=0
        elif c[i]<=ep*(1-2.5/100):
            trades.append({'ei':entry_i,'xi':i,'ep':ep,'xp':c[i],'t':'SL'})
            pos=0
    if not pos and le[i]: pos=1; ep=o[i]; entry_i=i

for i,trade in enumerate(trades):
    pnl=(trade['xp']/trade['ep']-1)*100-COMM*100
    dur=trade['xi']-trade['ei']
    print(f'  {i+1}. {trade["t"]} | {idx[trade["ei"]]} → {idx[trade["xi"]]} | ${trade["ep"]:.5f}→${trade["xp"]:.5f} | {pnl:+.2f}% | {dur}min')

# ── Chart ──
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(20,9),sharex=True,gridspec_kw={'height_ratios':[2.5,1]})
fig.patch.set_facecolor('white')

ax1.set_facecolor('white')
clrs=['green' if c[i]>=o[i] else 'red' for i in range(n)]
ax1.bar(idx,h-l_,bottom=l_,color=clrs,width=0.0003,alpha=0.2)
ax1.bar(idx,abs(c-o),bottom=np.minimum(c,o),color=clrs,width=0.00025)

blue=ssl_c==1; red=ssl_c==-1
ax1.plot(idx[blue],ssl[blue],'dodgerblue',linewidth=1.5,alpha=0.9)
ax1.plot(idx[red],ssl[red],'tomato',linewidth=1.5,alpha=0.9)

# SSL flips
ax1.scatter(idx[ssl_flip_blue],c[ssl_flip_blue],color='dodgerblue',s=30,zorder=4,marker='o',alpha=0.6,label=f'SSL→أزرق ({ssl_flip_blue.sum()})')

# Entries
ax1.scatter(idx[le],o[le],color='lime',s=80,zorder=5,marker='^',edgecolors='darkgreen',linewidth=1,label=f'🐋 دخول ({le.sum()})')

# Trades
for trade in trades:
    color='gold' if trade['t']=='TP' else 'magenta'
    ax1.plot([idx[trade['ei']],idx[trade['xi']]],[trade['ep'],trade['xp']],color=color,linewidth=3,alpha=0.7)

ax1.set_ylabel('FET/USDT',fontweight='bold')
ax1.legend(loc='upper left',ncol=2)
ax1.grid(True,alpha=0.15)

# Whale
ax2.set_facecolor('white')
ax2.fill_between(idx,0,wp,where=wp_up&(wp>0),color='limegreen',alpha=0.5)
ax2.fill_between(idx,0,wp,where=(~wp_up)&(wp>0),color='red',alpha=0.5)
ax2.axhline(y=0,color='gray',linewidth=0.5)
ax2.set_ylabel('Whale',fontweight='bold')
ax2.grid(True,alpha=0.15)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H'))

fig.suptitle(f'🐋 FET/USDT 1m — أول حوت بعد SSL أزرق — {le.sum()} دخول — {len(trades)} صفقة — Δ{p_change:+.1f}%',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig('/data/trading28/charts/fet_1m_first_whale.png',dpi=130,facecolor='white',bbox_inches='tight')
plt.close()
print(f'\n✅ Saved: fet_1m_first_whale.png')
