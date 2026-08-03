import pandas as pd, numpy as np, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMM=0.2;D='/data/trading28/data/3m_4months'

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

def get_trades(df,pv):
    wv=[(pv[i],pv[i+1]) for i in range(len(pv)-1) if pv[i][2]=='L' and pv[i+1][2]=='H']
    c=df['c'].values;h=df['h'].values;l=df['l'].values;n=len(df)
    trades=[];it=False;w={'L':None,'H':None,'r':1e10,'a':False}
    for i in range(1,n):
        for wL,wH in wv:
            if wH[0]<=i and (w['L'] is None or wL[0]>w.get('_b',-1)):
                w['L']=wL[1];w['H']=wH[1];w['_b']=wL[0];w['r']=wH[1];w['a']=True
        if w['a']:w['r']=min(w['r'],l[i])
        if not it and w['a']:
            wh=w['H']-w['L']
            ret_pct=(w['H']-w['r'])/wh if wh>0 else 0
            if wh>0 and 0.7<=ret_pct<=0.8:
                if c[i]>w['H'] and c[i-1]<=w['H']:
                    it=True;ep=c[i];sl=ep*0.985;tp=ep+wh*0.5;st=i
        if it:
            bh=i-st;rs=None;xp=c[i]
            if l[i]<=sl:rs='SL';xp=sl
            elif h[i]>=tp:rs='TP';xp=tp
            elif bh>=480:rs='TIME'
            if rs:
                pnl=(xp/ep-1)*100-COMM
                trades.append({'in':st,'out':i,'pnl':pnl,'r':rs,'ep':ep,'xp':xp,'sl':sl,'tp':tp,'wh':wh,'wL':w['L'],'wH':w['H'],'ret':ret_pct,'bars':bh})
                it=False
    return trades

COINS=['FET','ETH','SOL','ADA','DOGE','XRP','AVAX','LINK','DOT','UNI']
all_t=[]
for cn in COINS:
    df=pd.read_json(f'{D}/{cn}.json')
    pv=zz(df['h'].values,df['l'].values,10,1.0)
    ts=get_trades(df,pv)
    for t in ts:t['coin']=cn;t['ts_in']=int(df['ts'].iloc[t['in']]);t['ts_out']=int(df['ts'].iloc[t['out']])
    all_t.extend(ts)
all_t.sort(key=lambda x:x['ts_in'])

wins=[t for t in all_t if t['pnl']>0]
losses=[t for t in all_t if t['pnl']<=0]
picks=losses[-2:]+wins[-2:]

fig,axes=plt.subplots(2,2,figsize=(20,14));axes=axes.flatten()

for idx,t in enumerate(picks):
    cn=t['coin'];df=pd.read_json(f'{D}/{cn}.json')
    pv=zz(df['h'].values,df['l'].values,10,1.0)
    n=len(df)
    pad=30;start=max(0,t['in']-pad);end=min(n-1,t['out']+pad);win=end-start
    ax=axes[idx]
    
    for i in range(win):
        ii=start+i;row=df.iloc[ii];clr='#26a69a' if row['c']>=row['o'] else '#ef5350'
        ax.plot([i,i],[row['l'],row['h']],color=clr,linewidth=0.6,alpha=0.8)
        ax.plot([i-0.2,i+0.2],[row['o'],row['c']],color=clr,linewidth=1.8,alpha=0.9)
    
    zx,zy=[],[]
    for pi,pr,pt in pv:
        if start<=pi<=end:zx.append(pi-start);zy.append(pr)
    if zx:ax.plot(zx,zy,color='#1565C0',linewidth=2,zorder=4,alpha=0.6)
    
    ei=t['in']-start;xi=t['out']-start
    edge='lime' if t['pnl']>0 else 'red'
    
    ax.scatter(ei,t['ep'],color='yellow',s=200,zorder=10,marker='o',edgecolors=edge,linewidths=3)
    ax.scatter(xi,t['xp'],color='yellow',s=200,zorder=10,marker='s',edgecolors=edge,linewidths=3)
    ax.annotate('ENTRY',(ei,t['ep']),xytext=(ei,t['ep']+t['wh']*0.3),fontsize=9,color=edge,fontweight='bold',ha='center',arrowprops=dict(arrowstyle='->',color=edge,lw=2))
    ax.annotate(f'EXIT {t["r"]}',(xi,t['xp']),xytext=(xi,t['xp']-t['wh']*0.3),fontsize=9,color=edge,fontweight='bold',ha='center',arrowprops=dict(arrowstyle='->',color=edge,lw=2))
    
    ax.axhline(y=t['sl'],xmin=ei/win,xmax=xi/win,color='red',linewidth=2,linestyle='--',alpha=0.8)
    ax.text(ei+2,t['sl'],f'SL {t["sl"]:.4f}',fontsize=7,color='red',va='bottom')
    ax.axhline(y=t['tp'],xmin=ei/win,xmax=xi/win,color='lime',linewidth=2,linestyle='--',alpha=0.8)
    ax.text(ei+2,t['tp'],f'TP {t["tp"]:.4f}',fontsize=7,color='lime',va='top')
    
    ax.axhline(y=t['wH'],xmin=ei/win-0.02,xmax=ei/win,color='cyan',linewidth=2,linestyle='-',alpha=0.7)
    ax.text(ei-3,t['wH'],f'Wave H={t["wH"]:.4f}',fontsize=7,color='cyan',va='bottom',ha='right')
    
    wh=t['wh'];wL=t['wL'];wH=t['wH']
    for lvl,clr in [(0.618,'#FFD600'),(0.7,'lime'),(0.786,'#CE93D8')]:
        p_=wH-wh*lvl
        ax.axhline(y=p_,xmin=0,xmax=ei/win,color=clr,linewidth=1,linestyle='--',alpha=0.5)
        ax.text(ei-3,p_,f'Fib {lvl:.3f}',fontsize=6,color=clr,va='center',ha='right')
    
    ax.fill_between([ei,xi],ax.get_ylim()[0],ax.get_ylim()[1],alpha=0.1,color=edge)
    
    tin=pd.to_datetime(df['ts'].iloc[t['in']],unit='ms')
    tout=pd.to_datetime(df['ts'].iloc[t['out']],unit='ms')
    e='WIN' if t['pnl']>0 else 'LOSS'
    ax.set_title(f'{e} - {cn} - {tin.strftime("%m/%d %H:%M")} to {tout.strftime("%m/%d %H:%M")} - {t["r"]} - {t["pnl"]:+.2f}% - {t["bars"]}b - ret={t["ret"]:.3f}',fontsize=11,fontweight='bold',color=edge)
    ax.set_ylabel('USDT');ax.grid(True,alpha=0.08)
    step=10;ticks=list(range(0,win,step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([pd.to_datetime(df['ts'].iloc[start+i],unit='ms').strftime('%H:%M') for i in ticks],rotation=45,ha='right',fontsize=6)

plt.suptitle('Entry and Exit Signals - ZigZag + Fib 0.7-0.8 - TP=0.5x - SL=1.5%',fontsize=15,fontweight='bold',y=0.995)
plt.tight_layout()
plt.savefig('/data/trading28/scripts/zz_entry_exit.png',dpi=150)
print('OK')
