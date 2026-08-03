#!/usr/bin/env python3
"""Plot 10 steep angle + pullback trades — FET 1h"""
import ccxt, pandas as pd, numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SYMBOL = 'FET/USDT'; COMM = 0.002; DAYS = 180; CAP = 1000

def fetch(tf, days):
    ex = ccxt.binance({'timeout': 15000})
    since = ex.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    all_c = []
    while True:
        batch = ex.fetch_ohlcv(SYMBOL, tf, since=since, limit=1000)
        if not batch: break
        all_c.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000: break
    df = pd.DataFrame(all_c, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True); df.sort_index(inplace=True)
    return df

def slope(c, period):
    y = c[-period:]; x = np.arange(period)
    return np.polyfit(x, y, 1)[0] / np.mean(y) * 100

print('Fetching FET 1h...')
df = fetch('1h', DAYS)
c=df['close'].values; h=df['high'].values; l=df['low'].values; o=df['open'].values
n=len(c); idx=df.index

# Pre-compute slopes & pullbacks (lookback=15, same as before)
lookback=15; slopes=np.full(n,np.nan); pullbacks=np.full(n,np.nan)
for i in range(50,n):
    slopes[i]=slope(c[i-lookback+1:i+1],lookback)
    peak5=h[i-5:i+1].max()
    pullbacks[i]=(peak5-c[i])/peak5*100 if peak5>0 else 0

# Generate trades with exact same logic as best config: angle>0.4, PB 1.5-2.0, TP2 SL2
angle_min=0.4; pb_min=1.5; pb_max=2.0; tp=2.0; sl=2.0
trades=[]; pos=0; ep=0; ei=0
for i in range(200,n):
    if np.isnan(slopes[i]): continue
    steep=any(not np.isnan(slopes[j]) and slopes[j]>angle_min for j in range(max(0,i-5),i+1))
    steep_down=any(not np.isnan(slopes[j]) and slopes[j]<-angle_min for j in range(max(0,i-5),i+1))
    
    le=(steep and pullbacks[i]>pb_min and pullbacks[i]<pb_max and c[i]>h[i-1] and c[i]>o[i])
    se=(steep_down and (c[i]-l[max(0,i-3):i+1].min())/l[max(0,i-3):i+1].min()*100>pb_min and 
        (c[i]-l[max(0,i-3):i+1].min())/l[max(0,i-3):i+1].min()*100<pb_max and c[i]<l[i-1] and c[i]<o[i])
    
    if pos==1:
        ex=False; xp=c[i]; reason=''
        if h[i]>=ep*(1+tp/100): ex=True; xp=ep*(1+tp/100); reason='TP'
        elif c[i]<=ep*(1-sl/100): ex=True; xp=c[i]; reason='SL'
        elif se: ex=True; xp=c[i]; reason='REV'
        if ex:
            pnl=(xp/ep-1)*100-COMM*100
            trades.append({'type':'L','ei':ei,'xi':i,'ep':ep,'xp':xp,'pnl':pnl,'reason':reason})
            pos=0
    elif pos==-1:
        ex=False; xp=c[i]; reason=''
        if l[i]<=ep*(1-tp/100): ex=True; xp=ep*(1-tp/100); reason='TP'
        elif c[i]>=ep*(1+sl/100): ex=True; xp=c[i]; reason='SL'
        elif le: ex=True; xp=c[i]; reason='REV'
        if ex:
            pnl=(1-xp/ep)*100-COMM*100
            trades.append({'type':'S','ei':ei,'xi':i,'ep':ep,'xp':xp,'pnl':pnl,'reason':reason})
            pos=0
    
    if pos==0:
        if le: pos=1; ep=c[i]; ei=i
        elif se: pos=-1; ep=c[i]; ei=i

completed=[t for t in trades if 'pnl' in t]
print(f'Total completed trades: {len(completed)}')
plot_trades=completed[-10:]

# ═══════════ PLOT ═══════════
fig,ax=plt.subplots(figsize=(22,13))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

start_i=max(0,plot_trades[0]['ei']-15)
end_i=min(n-1,plot_trades[-1]['xi']+15)

for i in range(start_i,end_i+1):
    clr='#00ff88' if c[i]>=o[i] else '#ff4466'
    ax.plot([i,i],[l[i],h[i]],color=clr,linewidth=0.6)
    ax.plot([i,i],[o[i],c[i]],color=clr,linewidth=4)

for t in plot_trades:
    ei=t['ei']; xi=t['xi']; clr='#00ff88' if t['type']=='L' else '#ff4466'
    # Entry
    ax.scatter(ei,t['ep'],color=clr,s=120,marker='^' if t['type']=='L' else 'v',zorder=5,edgecolors='white',linewidths=1)
    # Exit
    ax.scatter(xi,t['xp'],color='white',s=80,marker='o',zorder=5,edgecolors=clr,linewidths=2)
    # Line
    ax.plot([ei,xi],[t['ep'],t['xp']],color=clr,linewidth=1.5,alpha=0.5,linestyle='--')
    # PnL label
    mid_i=(ei+xi)//2; mid_px=max(t['ep'],t['xp'])+0.015
    ax.annotate(f"{t['pnl']:+.1f}% {t['reason']}",(mid_i,mid_px),color='white' if t['pnl']>0 else '#ff4466',
                fontsize=7,ha='center',fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2',facecolor='#1a1a2e',edgecolor=clr,alpha=0.8))

date_labels=[idx[i].strftime('%m/%d %H') for i in range(start_i,end_i+1)]
step=max(1,(end_i-start_i)//12)
tick_pos=list(range(start_i,end_i+1,step))
ax.set_xticks(tick_pos)
ax.set_xticklabels([date_labels[i-start_i] for i in tick_pos],rotation=45,fontsize=8,color='white')
ax.tick_params(axis='y',colors='white')
ax.grid(alpha=0.15,color='white')
ax.set_ylabel('FET/USDT',color='white',fontsize=12)

wins=sum(1 for t in plot_trades if t['pnl']>0)
losses=len(plot_trades)-wins
win_avg=np.mean([t['pnl'] for t in plot_trades if t['pnl']>0]) if wins else 0
loss_avg=np.mean([t['pnl'] for t in plot_trades if t['pnl']<=0]) if losses else 0
ax.set_title(f'Steep Angle + Pullback — FET 1h — Last 10 Trades\nangle>0.4% PB1.5-2% TP2/SL2 — {wins}W/{losses}L WR {wins/len(plot_trades)*100:.0f}% aW{win_avg:+.1f}% aL{loss_avg:+.1f}%',
             color='white',fontsize=13,fontweight='bold')

plt.tight_layout()
path='/data/trading28/charts/steep_angle_10trades.png'
plt.savefig(path,dpi=150,bbox_inches='tight',facecolor='#1a1a2e')
print(f'Saved: {path}')
print(f'\nTrade details:')
for i,t in enumerate(plot_trades):
    print(f'  {i+1}. {t["type"]} {idx[t["ei"]].strftime("%m/%d %H:%M")}->{idx[t["xi"]].strftime("%m/%d %H:%M")} | {t["pnl"]:+.2f}% [{t["reason"]}]')
