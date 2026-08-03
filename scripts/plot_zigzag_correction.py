import pandas as pd, numpy as np, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMM=0.2;D='/data/trading28/data/3m_4months'
COINS=['FET','ETH','SOL','ADA','DOGE','XRP','AVAX','LINK','DOT','UNI']

def zz(h,l,d=10,dev=1.0):
    D=d//2;h=list(h);l=list(l);n=len(h);p=[]
    for i in range(D,n-D):
        if all(h[j]<h[i] for j in range(i-D,i)) and all(h[j]<=h[i] for j in range(i+1,i+D+1)):p.append((i,h[i],'H'))
        if all(l[j]>l[i] for j in range(i-D,i)) and all(l[j]>=l[i] for j in range(i+1,i+D+1)):p.append((i,l[i],'L'))
    p.sort(key=lambda x:x[0]);f=[]
    for i,pr,pt in p:
        if not f:f.append((i,pr,pt))
        elif pt==f[-1][2]:
            if (pt=='H' and pr>f[-1][1])or(pt=='L' and pr<f[-1][1]):f[-1]=(i,pr,pt)
        elif abs(pr-f[-1][1])/f[-1][1]*100>=dev:f.append((i,pr,pt))
    return f

def find_zpatterns(pv):
    pats=[]
    for i in range(len(pv)-3):
        p0,p1,p2,p3=pv[i],pv[i+1],pv[i+2],pv[i+3]
        if p0[2]=='H' and p1[2]=='L' and p2[2]=='H' and p3[2]=='L':
            A=p0[1]-p1[1];B=p2[1]-p1[1];C=p2[1]-p3[1]
            if A>0 and B>0 and C>0 and 0.38<=B/A<=0.79 and p3[1]<p1[1]:
                pats.append((p0,p1,p2,p3,A,B,C,B/A))
    return pats

# Collect trades with pattern context
all_t=[]
for cn in COINS:
    df=pd.read_json(f'{D}/{cn}.json')
    pv=zz(df['h'].values,df['l'].values,10,1.0)
    pats=find_zpatterns(pv)
    c=df['c'].values;h=df['h'].values;l=df['l'].values;n=len(df)
    
    for H1,L1,H2,L2,A,B,C,ret_B in pats:
        eb=L2[0]+1
        if eb>=n-10:continue
        ep=c[eb];sl=ep*0.99;tp=ep*1.02
        for i in range(eb,n):
            bh=i-eb;rs=None;xp=c[i]
            if l[i]<=sl:rs='SL';xp=sl
            elif h[i]>=tp:rs='TP';xp=tp
            elif bh>=480:rs='TIME'
            if rs:
                all_t.append({
                    'coin':cn,'pnl':(xp/ep-1)*100-COMM,'r':rs,'bars':bh,
                    'ep':ep,'xp':xp,'sl':sl,'tp':tp,
                    'H1':H1,'L1':L1,'H2':H2,'L2':L2,
                    'in':eb,'out':i,'ts_in':int(df['ts'].iloc[eb])
                })
                break

all_t.sort(key=lambda x:x['ts_in'])

# Pick 2 wins + 2 losses
wins=[t for t in all_t if t['pnl']>0]
losses=[t for t in all_t if t['pnl']<=0]
picks=losses[-2:]+wins[-2:]

fig,axes=plt.subplots(2,2,figsize=(20,14));axes=axes.flatten()

for idx,t in enumerate(picks):
    cn=t['coin'];df=pd.read_json(f'{D}/{cn}.json')
    pv=zz(df['h'].values,df['l'].values,10,1.0)
    n=len(df)
    pad=40;start=max(0,t['H1'][0]-pad);end=min(n-1,t['out']+pad);win=end-start
    ax=axes[idx]
    
    for i in range(win):
        ii=start+i;row=df.iloc[ii];clr='#26a69a' if row['c']>=row['o'] else '#ef5350'
        ax.plot([i,i],[row['l'],row['h']],color=clr,linewidth=0.5,alpha=0.7)
        ax.plot([i-0.2,i+0.2],[row['o'],row['c']],color=clr,linewidth=1.5,alpha=0.8)
    
    # ZigZag
    zx,zy=[],[]
    for pi,pr,pt in pv:
        if start<=pi<=end:zx.append(pi-start);zy.append(pr)
    if zx:ax.plot(zx,zy,color='#1565C0',linewidth=2,zorder=4,alpha=0.5)
    
    # Pattern points: H1, L1, H2, L2
    H1=t['H1'];L1=t['L1'];H2=t['H2'];L2=t['L2']
    
    # Mark pattern
    ph1=H1[0]-start;ph2=H2[0]-start;pl1=L1[0]-start;pl2=L2[0]-start
    
    if 0<=ph1<win:ax.scatter(ph1,H1[1],color='#FF6D00',s=120,zorder=8,marker='v')
    if 0<=pl1<win:ax.scatter(pl1,L1[1],color='#00E676',s=120,zorder=8,marker='^')
    if 0<=ph2<win:ax.scatter(ph2,H2[1],color='#FF6D00',s=100,zorder=8,marker='v')
    if 0<=pl2<win:ax.scatter(pl2,L2[1],color='cyan',s=150,zorder=8,marker='o')
    
    # Connect pattern: H1→L1→H2→L2
    pts=[(ph1,H1[1]),(pl1,L1[1]),(ph2,H2[1]),(pl2,L2[1])]
    valid_pts=[(x,y) for x,y in pts if 0<=x<win]
    if len(valid_pts)>=2:
        xs,ys=zip(*valid_pts)
        ax.plot(xs,ys,color='#FFD600',linewidth=3,linestyle='-',zorder=5,alpha=0.8)
    
    # Label waves
    for name,x,y in [('A',(ph1+pl1)//2,(H1[1]+L1[1])//2),('B',(pl1+ph2)//2,(L1[1]+H2[1])//2),('C',(ph2+pl2)//2,(H2[1]+L2[1])//2)]:
        if 0<=x<win:
            ax.text(x,y,name,fontsize=11,color='#FFD600',fontweight='bold',ha='center',va='center',
                   bbox=dict(boxstyle='round',facecolor='black',alpha=0.7))
    
    # Entry/Exit
    ei=t['in']-start;xi=t['out']-start
    edge='lime' if t['pnl']>0 else 'red'
    
    if 0<=ei<win:
        ax.scatter(ei,t['ep'],color='yellow',s=200,zorder=10,marker='o',edgecolors=edge,linewidths=3)
        ax.annotate('ENTRY',(ei,t['ep']),xytext=(ei,t['ep']+(H1[1]-L1[1])*0.3),fontsize=9,color=edge,fontweight='bold',ha='center',arrowprops=dict(arrowstyle='->',color=edge,lw=2))
    if 0<=xi<win:
        ax.scatter(xi,t['xp'],color='yellow',s=200,zorder=10,marker='s',edgecolors=edge,linewidths=3)
        ax.annotate(f'EXIT {t["r"]}',(xi,t['xp']),xytext=(xi,t['xp']-(H1[1]-L1[1])*0.3),fontsize=9,color=edge,fontweight='bold',ha='center',arrowprops=dict(arrowstyle='->',color=edge,lw=2))
    
    # SL/TP
    if 0<=ei<win and 0<=xi<win:
        ax.axhline(y=t['sl'],xmin=ei/win,xmax=xi/win,color='red',linewidth=1.5,linestyle='--',alpha=0.7)
        ax.axhline(y=t['tp'],xmin=ei/win,xmax=xi/win,color='lime',linewidth=1.5,linestyle='--',alpha=0.7)
    
    ax.fill_between([max(0,ei),min(xi,win-1)],ax.get_ylim()[0],ax.get_ylim()[1],alpha=0.08,color=edge)
    
    tin=pd.to_datetime(df['ts'].iloc[t['in']],unit='ms')
    tout=pd.to_datetime(df['ts'].iloc[t['out']],unit='ms')
    e='WIN' if t['pnl']>0 else 'LOSS'
    ax.set_title(f'{e} - {cn} | {tin.strftime("%m/%d %H:%M")} -> {tout.strftime("%m/%d %H:%M")} | {t["r"]} | {t["pnl"]:+.2f}% | {t["bars"]}b',fontsize=11,fontweight='bold',color=edge)
    ax.set_ylabel('USDT');ax.grid(True,alpha=0.06)
    step=10;ticks=list(range(0,win,step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([pd.to_datetime(df['ts'].iloc[start+i],unit='ms').strftime('%H:%M') for i in ticks],rotation=45,ha='right',fontsize=6)

plt.suptitle('Zigzag Correction Pattern (V-Shape) | H1->L1->H2->L2 | Entry at L2+1 | TP=2% SL=1%',fontsize=14,fontweight='bold',y=0.995)
plt.tight_layout()
plt.savefig('/data/trading28/scripts/zigzag_correction.png',dpi=150)
print('OK')
for t in picks:
    print(f"  {t['coin']} {t['r']} {t['pnl']:+.2f}% {t['bars']}b")
