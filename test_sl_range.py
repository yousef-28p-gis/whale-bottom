#!/usr/bin/env python3
"""Test different SL values on last month — realistic SL formula"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timezone

COMM=0.002; COOLDOWN=2; MAX_SLIP=1.5
CUTOFF=int(datetime(2026,7,3,tzinfo=timezone.utc).timestamp()*1000)

def ld(sym):
    p=os.path.join('/data/trading28/data/whale_15m_1y',sym+'.json')
    if not os.path.exists(p): return None
    with open(p) as f: j=json.load(f)
    ts=j.get('ts',[]);c=np.array(j['c'],float);h=np.array(j['h'],float)
    l=np.array(j['l'],float);o=np.array(j['o'],float)
    mask=np.array(ts)>=CUTOFF
    if mask.sum()<200: return None
    return c[mask],h[mask],l[mask],o[mask],[t for i,t in enumerate(ts) if mask[i]]

def r8h(c,h,l,o,ts):
    idx=pd.to_datetime(np.array(ts),unit='ms')
    df=pd.DataFrame({'o':o,'h':h,'l':l,'c':c},index=idx)
    r=df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
    return r['c'].values,r['h'].values,r['l'].values,r['o'].values,r.index

def tr(c,h,l,o,idx,sl):
    tk,kj,sk=3,9,18; n=len(c)
    if n<sk+30: return[]
    ht=pd.Series(h).rolling(tk).max().values;lt=pd.Series(l).rolling(tk).min().values
    ta=(ht+lt)/2;hk=pd.Series(h).rolling(kj).max().values;lk=pd.Series(l).rolling(kj).min().values
    ka=(hk+lk)/2;hs=pd.Series(h).rolling(sk).max().values;ls=pd.Series(l).rolling(sk).min().values
    sb=(hs+ls)/2;sa=(ta+ka)/2;sh=kj
    saf=np.full(n,np.nan);sbf=np.full(n,np.nan)
    for i in range(max(sh,sk),n-sh):
        if i+sh<n: saf[i+sh]=sa[i];sbf[i+sh]=sb[i]
    ct=np.maximum(saf,sbf);trs=[];pos=0;ep=0;cool=0;eb=0
    for i in range(sk+sh,n):
        if np.isnan(saf[i]): continue
        ab=c[i]>ct[i];gd=ta[i]>ka[i] and ta[i-1]<=ka[i-1]
        if pos:
            if h[i]>=ep*1.05: trs.append((eb,i,5-COMM*100));pos=0;cool=COOLDOWN
            elif l[i]<=ep*(1-sl/100):
                pnl=max((c[i]/ep-1)*100-COMM*100,-sl*MAX_SLIP-COMM*100)
                trs.append((eb,i,pnl));pos=0;cool=COOLDOWN
        if not pos and cool==0 and ab and gd: pos=1;ep=c[i];eb=i
        if not pos and cool>0: cool-=1
    if pos: trs.append((eb,n-1,(c[-1]/ep-1)*100-COMM*100))
    return trs

def r2p(ct):
    eq=1000;cv=[1000];op={};tl=[]
    for sym,trs in ct.items():
        for e,x,pnl in trs: tl.append((e,'entry',sym,pnl));tl.append((x,'exit',sym,pnl))
    tl.sort();exe=0;win=0
    for t,ty,sym,pnl in tl:
        if ty=='entry':
            if len(op)<2: op[sym]=eq/2
        elif ty=='exit':
            if sym in op:
                al=op.pop(sym);eq+=al*(1+pnl/100)-al;cv.append(eq);exe+=1
                if pnl>0: win+=1
    s=pd.Series(cv);pk=s.expanding().max();dd=((s-pk)/pk*100).min()
    return{'pnl':eq-1000,'dd':dd,'t':exe,'wr':win/exe*100 if exe else 0}

with open('/data/trading28/config/shariah_coins.json') as f: d=json.load(f)
tradeable=sorted(d['halal']+d['halal2'])

coin3={}
for pdir in['2023','prev','1y']:
    for sym in tradeable:
        fp=os.path.join(f'/data/trading28/data/whale_15m_{pdir}',sym+'.json')
        if not os.path.exists(fp): continue
        with open(fp) as f: j=json.load(f)
        c=np.array(j['c'],float);h=np.array(j['h'],float)
        l=np.array(j['l'],float);o=np.array(j['o'],float)
        ts=j.get('ts',[])
        rp=r8h(c,h,l,o,ts)
        if rp is None: continue
        trs=tr(*rp,2.5)
        if len(trs)>=3: coin3[sym]=coin3.get(sym,0)+sum(p for _,_,p in trs)
top60=set(c for c,_ in sorted(coin3.items(),key=lambda x:x[1],reverse=True)[:60])

print(' TP5/SL متغير | يوليو 2026 | أفضل 60\n')
print('  SL   | صفقات |  WR   |  ربح$  |  سحب  | خسارة قصوى')
print('-'*58)
for sl in[1.5,2.0,2.5,3.0]:
    ct={}
    for sym in top60:
        d=ld(sym)
        if d is None: continue
        rp=r8h(*d)
        if rp is None: continue
        trs=tr(*rp,sl)
        if len(trs)>=1: ct[sym]=trs
    m=r2p(ct)
    maxloss=-sl*MAX_SLIP-COMM*100
    print(f'SL={sl:.1f}% | {m["t"]:5d} | {m["wr"]:4.1f}% | ${m["pnl"]:+6.0f} | {m["dd"]:5.1f}% | {maxloss:+.1f}%')
