#!/usr/bin/env python3
"""Plot 3 more winning trades"""
import json, numpy as np, pandas as pd, os, sys, random
sys.path.insert(0,'/data/trading28')
from strategies.zigzag import zigzag
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA='/data/trading28/data/3m_4months'
DEPTH=10; DEV=1.0

with open('/data/trading28/elliot_completed_trades.json') as f:
    all_trades = json.load(f)

winners = [t for t in all_trades if t['pnl'] > 0.5]  # bigger winners
random.seed(99)
picked = random.sample(winners, 3)

OUTC='/data/trading28/elliot_charts/completed'

for pi, t in enumerate(picked):
    cn=t['coin']
    fp=f'{DATA}/{cn}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw=json.load(f)
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    
    eb=t['eb']; ex=t['exit_bar']
    start=max(0,eb-40); end=min(len(df),ex+30)
    segment=df.iloc[start:end].copy()
    rel_eb=eb-start; rel_ex=ex-start
    
    fig,ax=plt.subplots(figsize=(10,5),facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    x=range(len(segment))
    ax.plot(x,segment['close'].values,color='#00bcd4',linewidth=1,alpha=0.8)
    
    seg_h=segment['high'].values; seg_l=segment['low'].values
    pv=zigzag(seg_h,seg_l,DEPTH,DEV)
    for j in range(len(pv)-1):
        c='#ff6b6b' if pv[j][2]=='H' else '#4ecdc4'
        ax.plot([pv[j][0],pv[j+1][0]],[pv[j][1],pv[j+1][1]],color=c,linewidth=1.2,alpha=0.7)
    
    ax.axvline(x=rel_eb,color='cyan',linestyle='-',linewidth=1.5)
    ax.scatter(rel_eb,t['ep'],color='cyan',s=150,marker='>',zorder=5,edgecolors='white')
    ax.annotate(f'دخول\n${t["ep"]:.4f}',(rel_eb,t['ep']),textcoords="offset points",xytext=(12,0),color='cyan',fontsize=9,fontweight='bold')
    
    exit_clr='lime'
    ax.axvline(x=rel_ex,color=exit_clr,linestyle='-',linewidth=1.5)
    if rel_ex<len(segment):
        ex_price=segment['close'].iloc[rel_ex]
        ax.scatter(rel_ex,ex_price,color=exit_clr,s=150,marker='<',zorder=5,edgecolors='white')
        ax.annotate(f'خروج\n${ex_price:.4f}',(rel_ex,ex_price),textcoords="offset points",xytext=(-12,0),ha='right',color=exit_clr,fontsize=9,fontweight='bold')
    
    l3=t['L3']; h3=t['H3']; w5=h3-l3
    for lbl,mult in [('F0.5',0.5),('F1.0',1.0)]:
        price=l3+mult*w5
        ax.axhline(y=price,color='gold',linestyle='--',linewidth=0.7,alpha=0.5)
        ax.text(len(segment)-1,price,lbl,color='gold',fontsize=7,va='bottom',ha='right')
    
    i_l3=t.get('i_L3',0); i_h3=t.get('i_H3',0)
    rel_l3=i_l3-start; rel_h3=i_h3-start
    if 0<=rel_l3<len(segment):
        ax.scatter(rel_l3,l3,color='lime',s=80,marker='^',zorder=4,edgecolors='white')
        ax.annotate('L3',(rel_l3,l3),textcoords="offset points",xytext=(0,-14),ha='center',color='lime',fontsize=8)
    if 0<=rel_h3<len(segment):
        ax.scatter(rel_h3,h3,color='red',s=80,marker='v',zorder=4,edgecolors='white')
        ax.annotate('H3',(rel_h3,h3),textcoords="offset points",xytext=(0,10),ha='center',color='red',fontsize=8)
    
    ax.set_title(f'{cn} | {t["type"]} | WIN +{t["pnl"]:.2f}% | w5={t["w5_pct"]:.1f}%',color='white',fontsize=12,fontweight='bold')
    ax.tick_params(colors='gray',labelsize=8)
    ax.grid(alpha=0.12)
    ax.spines['bottom'].set_color('#333'); ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fname=f'{OUTC}/trade_{pi+8}_{cn}_{t["type"]}.png'
    fig.savefig(fname,dpi=120,facecolor='#1a1a2e')
    plt.close(fig)
    print(f'  ✅ {fname} — +{t["pnl"]:.2f}%')

print('done')
