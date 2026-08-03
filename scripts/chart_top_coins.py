#!/usr/bin/env python3
"""Charts for top 3 coins — TRX, ALLO, RAD — last 60 days of data"""
import json, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000; DATA='/data/trading28/data/whale_15m_1y'
COOLDOWN=48

def load(sym):
    with open(os.path.join(DATA, f'{sym}.json')) as f: d=json.load(f)
    return {'c':np.array(d['c'],float),'h':np.array(d['h'],float),
            'l':np.array(d['l'],float),'o':np.array(d['o'],float),
            'ts':pd.to_datetime(d['ts'],unit='ms')}

def make_chart(sym, LB, ssl_p, tp, sl):
    d=load(sym)
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; idx=d['ts']; n=len(c)
    
    # Whale
    ln=pd.Series(l_).shift(1).rolling(LB).min().values
    lc=np.zeros(n)
    for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
    sc=pd.Series(lc).ewm(span=3,adjust=False).mean().values
    hc=pd.Series(sc).rolling(LB).max().values
    sr=np.where(l_<=ln,(sc+hc*2)/3,0)
    wp=pd.Series(sr).ewm(span=3,adjust=False).mean().values
    wp_up=wp>np.roll(wp,1)
    
    # SSL
    sma_h=pd.Series(h).rolling(ssl_p).mean().values
    sma_l=pd.Series(l_).rolling(ssl_p).mean().values
    ssl_c=np.zeros(n,int); ssl_line=np.full(n,np.nan)
    for i in range(ssl_p,n):
        if h[i-1]>sma_h[i-1]: ssl_c[i]=1; ssl_line[i]=sma_l[i]
        else: ssl_c[i]=-1; ssl_line[i]=sma_h[i]
    
    # Entry
    le=np.zeros(n,bool)
    for i in range(200,n):
        if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0:
            le[i]=True
    
    # Sim trades
    trades=[]; pos=0; ep=0; cool=0; ei=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                trades.append({'ei':ei,'xi':i,'ep':ep,'xp':ep*(1+tp/100),'t':'TP'}); pos=0; cool=COOLDOWN
            elif l_[i]<=ep*(1-sl/100):
                raw=(c[i]/ep-1)*100-COMM*100; pnl=max(raw,-sl*1.5-COMM*100)
                xp=ep*(1+pnl/100+COMM)
                trades.append({'ei':ei,'xi':i,'ep':ep,'xp':xp,'t':'SL'}); pos=0; cool=COOLDOWN
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]; ei=i
        if not pos and cool>0: cool-=1
    
    # Last 6000 candles for chart (~60 days)
    start_chart=max(0,n-6000)
    x=idx[start_chart:]
    c2=c[start_chart:]; h2=h[start_chart:]; l2=l_[start_chart:]; o2=o[start_chart:]
    ssl_line2=ssl_line[start_chart:]; ssl_c2=ssl_c[start_chart:]
    wp2=wp[start_chart:]; wp_up2=wp_up[start_chart:]
    le2=le[start_chart:]
    
    # Filter trades in chart range
    chart_trades=[t for t in trades if t['ei']>=start_chart]
    
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(20,9),sharex=True,gridspec_kw={'height_ratios':[2.5,1]})
    fig.patch.set_facecolor('white')
    
    # Price + SSL
    ax1.set_facecolor('white')
    clrs=['limegreen' if c2[i]>=o2[i] else 'red' for i in range(len(c2))]
    ax1.bar(x,h2-l2,bottom=l2,color=clrs,width=0.0004,alpha=0.2)
    ax1.bar(x,abs(c2-o2),bottom=np.minimum(c2,o2),color=clrs,width=0.0003)
    
    # SSL line
    blue=ssl_c2==1; red=ssl_c2==-1
    ax1.plot(x[blue],ssl_line2[blue],'dodgerblue',linewidth=1.5,alpha=0.9)
    ax1.plot(x[red],ssl_line2[red],'tomato',linewidth=1.5,alpha=0.9)
    
    # Entries
    ax1.scatter(x[le2],c2[le2],color='lime',s=60,zorder=5,marker='^',edgecolors='darkgreen',linewidth=1,label=f'🐋 دخول ({le2.sum()})')
    
    # Trades
    for t in chart_trades:
        color='gold' if t['t']=='TP' else 'magenta'
        ax1.plot([idx[t['ei']],idx[t['xi']]],[t['ep'],t['xp']],color=color,linewidth=2.5,alpha=0.7)
    
    pnl_total=sum((t['xp']/t['ep']-1)*100-COMM*100 for t in chart_trades)
    w=[t for t in chart_trades if t['t']=='TP']
    wr=len(w)/len(chart_trades)*100 if chart_trades else 0
    
    ax1.set_ylabel(f'{sym}/USDT',fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True,alpha=0.15)
    ax1.set_title(f'{sym}/USDT 15m | {len(chart_trades)} صفقة | WR {wr:.0f}% | PnL {pnl_total:+.1f}% | LB{LB}/SSL{ssl_p} TP{tp}/SL{sl}',fontweight='bold')
    
    # Whale
    ax2.set_facecolor('white')
    ax2.fill_between(x,0,wp2,where=wp_up2&(wp2>0),color='limegreen',alpha=0.5)
    ax2.fill_between(x,0,wp2,where=(~wp_up2)&(wp2>0),color='red',alpha=0.5)
    ax2.axhline(y=0,color='gray',linewidth=0.5)
    ax2.set_ylabel('Whale',fontweight='bold')
    ax2.grid(True,alpha=0.15)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    plt.tight_layout()
    plt.savefig(f'/data/trading28/charts/{sym}_trades.png',dpi=130,facecolor='white',bbox_inches='tight')
    plt.close()
    return len(chart_trades),wr,pnl_total

# ── Generate charts ──
configs={
    'TRX':(70,10,3,1.5),
    'ALLO':(70,20,2,1),
    'RAD':(50,5,5,2.5),
    'KAITO':(70,10,5,2.5),
}

for sym,(LB,ssl_p,tp,sl) in configs.items():
    nt,wr,pnl=make_chart(sym,LB,ssl_p,tp,sl)
    print(f'{sym}: {nt} trades, WR {wr:.0f}%, PnL {pnl:+.1f}% | saved')

print('\n✅ Done')
