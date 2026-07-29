#!/usr/bin/env python3
"""Plot 4 V-Shape trades with Half TP + BE"""
import pandas as pd, numpy as np, os, sys, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag

COMM=0.20; D='/data/trading28/data/3m_4months'
DEPTH=10; DEV=1.0; CONFIRM=5; TIME_BARS=120

def find_zpatterns(pv):
    pats=[]
    for i in range(len(pv)-3):
        p0,p1,p2,p3=pv[i],pv[i+1],pv[i+2],pv[i+3]
        if p0[2]=='H' and p1[2]=='L' and p2[2]=='H' and p3[2]=='L':
            A=p0[1]-p1[1]; B=p2[1]-p1[1]; C=p2[1]-p3[1]
            if A>0 and B>0 and C>0 and 0.38<=B/A<=0.79 and p3[1]<p1[1]:
                pats.append((p0,p1,p2,p3))
    return pats

def sim_one(close,high,low,coin):
    n=len(close); pv=zigzag(high,low,DEPTH,DEV)
    if len(pv)<4: return[]
    pats=find_zpatterns(pv); trades=[]
    for H1,L1,H2,L2 in pats:
        eb=L2[0]+CONFIRM
        if eb>=n: continue
        ep=close[eb]
        if (ep-L2[1])/L2[1]*100>0.5: continue
        sl=L2[1]*0.995
        if sl>=ep: continue
        tp_full=ep*1.01; tp_half=ep*1.005; be=ep
        
        half_exited=False; half1_pnl=0.0; exit_type='TIME'; exit_pnl=0.0
        exit_idx=eb; exit_price=ep
        
        for j in range(eb+1, min(n, eb+TIME_BARS+1)):
            bh=high[j]; bl=low[j]; bc=close[j]
            
            if not half_exited:
                if bh>=tp_half:
                    half1_pnl=(tp_half/ep-1)*100-COMM/2; half_exited=True
                    if bh>=tp_full:
                        h2=(tp_full/ep-1)*100-COMM/2
                        exit_idx=j; exit_price=tp_full; exit_type='TP'
                        exit_pnl=(half1_pnl+h2)/2; break
                    continue
                if bc<=sl:
                    exit_idx=j; exit_price=bc; exit_type='SL'
                    exit_pnl=(bc/ep-1)*100-COMM; half_exited=True; break
            
            if half_exited:
                if bh>=tp_full:
                    h2=(tp_full/ep-1)*100-COMM/2
                    exit_idx=j; exit_price=tp_full; exit_type='TP'
                    exit_pnl=(half1_pnl+h2)/2; break
                if bc<=be:
                    h2=(be/ep-1)*100-COMM/2
                    exit_idx=j; exit_price=be; exit_type='BE'
                    exit_pnl=(half1_pnl+h2)/2; break
        else:
            exit_idx=min(eb+TIME_BARS, n-1); exit_price=close[exit_idx]
            if half_exited: exit_pnl=(half1_pnl+(exit_price/ep-1)*100-COMM/2)/2
            else: exit_pnl=(exit_price/ep-1)*100-COMM
        
        trades.append({'coin':coin,'eb':eb,'exit':exit_idx,'ep':ep,'xp':exit_price,
            'type':exit_type,'pnl':round(exit_pnl,4),'bars':exit_idx-eb,
            'H1':H1,'L1':L1,'H2':H2,'L2':L2,'sl':sl,'tp':tp_full,'be':be,'tp_half':tp_half})
    return trades

# Collect all trades
with open('/data/trading28/config/shariah_coins.json') as f:
    sh=json.load(f)
COINS=[c for c in sh['halal']+sh['halal2'] if c not in {'USDT','USDC','BUSD','DAI','TUSD'}]

all_t=[]
for cn in COINS[:60]:  # enough for picks
    fp=f'{D}/{cn}.json'
    if not os.path.exists(fp): continue
    df=pd.read_json(fp)
    df=df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    all_t.extend(sim_one(df['close'].values,df['high'].values,df['low'].values,cn))

all_t.sort(key=lambda t:t['eb'])

# Pick: 2 best wins (TP on both halves), 1 BE win, 1 loss
tp_wins=[t for t in all_t if t['type']=='TP' and t['pnl']>0.5]
be_wins=[t for t in all_t if t['type']=='BE' and t['pnl']>0]
losses=[t for t in all_t if t['type']=='SL']

picks=[]
if tp_wins: picks.append(tp_wins[-1])   # recent TP win
if len(tp_wins)>1: picks.append(tp_wins[len(tp_wins)//3])  # middle TP win
if be_wins: picks.append(be_wins[len(be_wins)//2])  # middle BE win
if losses: picks.append(losses[-2] if len(losses)>1 else losses[-1])  # loss

while len(picks)<4:
    remaining=[t for t in all_t if t not in picks]
    if not remaining: break
    picks.append(remaining[-1])

fig,axes=plt.subplots(2,2,figsize=(22,14)); axes=axes.flatten()
fig.patch.set_facecolor('#0d1117')

for idx,t in enumerate(picks[:4]):
    cn=t['coin']; df=pd.read_json(f'{D}/{cn}.json')
    df=df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    pv=zigzag(df['high'].values,df['low'].values,DEPTH,DEV); n=len(df)
    pad=35; start=max(0,t['H1'][0]-pad); end=min(n-1,t['exit']+pad); win=end-start
    ax=axes[idx]; ax.set_facecolor('#0d1117')
    
    # Candlesticks
    for i in range(win):
        ii=start+i; row=df.iloc[ii]
        clr='#26a69a' if row['close']>=row['open'] else '#ef5350'
        ax.plot([i,i],[row['low'],row['high']],color=clr,linewidth=0.5,alpha=0.7)
        ax.plot([i-0.2,i+0.2],[row['open'],row['close']],color=clr,linewidth=1.8,alpha=0.9)
    
    # ZigZag line
    zx,zy=[],[]
    for pi,pr,pt in pv:
        if start<=pi<=end: zx.append(pi-start); zy.append(pr)
    if zx: ax.plot(zx,zy,color='#1565C0',linewidth=2,zorder=4,alpha=0.4)
    
    # Pattern points
    H1=t['H1']; L1=t['L1']; H2=t['H2']; L2=t['L2']
    ph1=H1[0]-start; ph2=H2[0]-start; pl1=L1[0]-start; pl2=L2[0]-start
    
    ax.scatter(ph1,H1[1],color='#FF6D00',s=130,zorder=8,marker='v')
    ax.scatter(pl1,L1[1],color='#00E676',s=130,zorder=8,marker='^')
    ax.scatter(ph2,H2[1],color='#FF6D00',s=110,zorder=8,marker='v')
    ax.scatter(pl2,L2[1],color='cyan',s=160,zorder=8,marker='o')
    
    # Pattern line
    pts=[(ph1,H1[1]),(pl1,L1[1]),(ph2,H2[1]),(pl2,L2[1])]
    vpts=[(x,y) for x,y in pts if 0<=x<win]
    if len(vpts)>=2: 
        xs,ys=zip(*vpts); ax.plot(xs,ys,color='#FFD600',linewidth=3,zorder=5,alpha=0.8)
    
    # Wave labels
    for name,x,y in [('A',(ph1+pl1)//2,(H1[1]+L1[1])//2),('B',(pl1+ph2)//2,(L1[1]+H2[1])//2),('C',(ph2+pl2)//2,(H2[1]+L2[1])//2)]:
        if 0<=x<win: ax.text(x,y,name,fontsize=12,color='#FFD600',fontweight='bold',ha='center',va='center',
            bbox=dict(boxstyle='round',facecolor='black',alpha=0.7))
    
    # Entry & Exit
    ei=t['eb']-start; xi=t['exit']-start
    edge='lime' if t['pnl']>0 else 'red'
    
    if 0<=ei<win:
        ax.scatter(ei,t['ep'],color='yellow',s=220,zorder=10,marker='o',edgecolors=edge,linewidths=3)
        ax.annotate(f'ENTRY\n${t["ep"]:.5f}',(ei,t['ep']),
            xytext=(ei,t['ep']+(H1[1]-L1[1])*0.35),fontsize=9,color=edge,fontweight='bold',ha='center',
            arrowprops=dict(arrowstyle='->',color=edge,lw=2))
    if 0<=xi<win:
        ax.scatter(xi,t['xp'],color='yellow',s=220,zorder=10,marker='s',edgecolors=edge,linewidths=3)
        ax.annotate(f'{t["type"]}\n${t["xp"]:.5f}',(xi,t['xp']),
            xytext=(xi,t['xp']-(H1[1]-L1[1])*0.3),fontsize=9,color=edge,fontweight='bold',ha='center',
            arrowprops=dict(arrowstyle='->',color=edge,lw=2))
    
    # SL / TP / BE levels
    if 0<=ei<win:
        x_end=min(xi if 0<=xi<win else win-1, win-1)
        ax.axhline(y=t['sl'],xmin=ei/win,xmax=x_end/win,color='red',linewidth=1.5,linestyle='--',alpha=0.6)
        ax.axhline(y=t['tp'],xmin=ei/win,xmax=x_end/win,color='lime',linewidth=1.5,linestyle='--',alpha=0.6)
        ax.axhline(y=t['be'],xmin=ei/win,xmax=x_end/win,color='orange',linewidth=1.5,linestyle=':',alpha=0.5)
        ax.axhline(y=t['tp_half'],xmin=ei/win,xmax=x_end/win,color='cyan',linewidth=1,linestyle='--',alpha=0.4)
    
    # Legend
    from matplotlib.lines import Line2D
    leg=[Line2D([0],[0],color='lime',lw=2,ls='--',label='TP 1%'),
         Line2D([0],[0],color='cyan',lw=1,ls='--',label='Half TP 0.5%'),
         Line2D([0],[0],color='orange',lw=1.5,ls=':',label='BE (دخول)'),
         Line2D([0],[0],color='red',lw=2,ls='--',label='SL L2-0.5%')]
    ax.legend(handles=leg,loc='upper right',fontsize=8,facecolor='#161b22',edgecolor='#30363d',labelcolor='white')
    
    # Fill trade zone
    if 0<=ei<win:
        ax.fill_between([max(0,ei),min(xi if 0<=xi<win else win-1,win-1)],
            ax.get_ylim()[0],ax.get_ylim()[1],alpha=0.08,color=edge)
    
    # Title
    tin=pd.to_datetime(df['ts'].iloc[t['eb']],unit='ms')
    tout=pd.to_datetime(df['ts'].iloc[t['exit']],unit='ms')
    e='WIN' if t['pnl']>0 else 'LOSS'
    ax.set_title(f'{e} — {cn} | {tin.strftime("%m/%d %H:%M")}->{tout.strftime("%m/%d %H:%M")} | {t["type"]} | {t["pnl"]:+.2f}% | {t["bars"]}b',
        fontsize=11,fontweight='bold',color=edge)
    ax.set_ylabel('USDT',color='white'); ax.tick_params(colors='white')
    ax.grid(True,alpha=0.06)
    step=max(1,win//8); ticks=list(range(0,win,step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([pd.to_datetime(df['ts'].iloc[start+i],unit='ms').strftime('%m/%d %H:%M') for i in ticks],
        rotation=45,ha='right',fontsize=7)
    ax.spines['bottom'].set_color('#30363d'); ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.suptitle('ZigZag V-Shape Correction | Half TP 0.5% + BE | TP=1% | SL=L2-0.5% | TIME=120',
    fontsize=14,fontweight='bold',y=0.995,color='white')
plt.tight_layout()
out='/data/trading28/scripts/vshape_trades_4.png'
plt.savefig(out,dpi=150,facecolor='#0d1117')
print(f'✅ {out}')
for t in picks[:4]:
    print(f"  {t['coin']} {t['type']} {t['pnl']:+.2f}% {t['bars']}b")
