#!/usr/bin/env python3
"""Cloud Hunter LONG ONLY — fixed 2.5% SL — last month"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timezone

COMM = 0.002; COOLDOWN = 2
CUTOFF = int(datetime(2026, 7, 3, tzinfo=timezone.utc).timestamp() * 1000)

def load_lm(sym):
    p = os.path.join('/data/trading28/data/whale_15m_1y', sym+'.json')
    if not os.path.exists(p): return None
    with open(p) as f: j = json.load(f)
    ts = j.get('ts',[]); c=np.array(j['c'],float); h=np.array(j['h'],float)
    l=np.array(j['l'],float); o=np.array(j['o'],float)
    mask=np.array(ts)>=CUTOFF
    if mask.sum()<200: return None
    return c[mask],h[mask],l[mask],o[mask],[t for i,t in enumerate(ts) if mask[i]]

def r8h(c,h,l,o,ts):
    idx=pd.to_datetime(np.array(ts),unit='ms')
    df=pd.DataFrame({'o':o,'h':h,'l':l,'c':c},index=idx)
    r=df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
    return r['c'].values,r['h'].values,r['l'].values,r['o'].values,r.index

def trades(c,h,l,o,idx):
    tk,kj,sk=3,9,18; n=len(c)
    if n<sk+30: return[]
    ht=pd.Series(h).rolling(tk).max().values; lt=pd.Series(l).rolling(tk).min().values
    ta=(ht+lt)/2
    hk=pd.Series(h).rolling(kj).max().values; lk=pd.Series(l).rolling(kj).min().values
    ka=(hk+lk)/2
    hs=pd.Series(h).rolling(sk).max().values; ls=pd.Series(l).rolling(sk).min().values
    sb=(hs+ls)/2; sa=(ta+ka)/2; sh=kj
    sa_f=np.full(n,np.nan); sb_f=np.full(n,np.nan)
    for i in range(max(sh,sk),n-sh):
        if i+sh<n: sa_f[i+sh]=sa[i]; sb_f[i+sh]=sb[i]
    ct=np.maximum(sa_f,sb_f)
    tr=[]; pos=0; ep=0; cool=0; eb=0
    for i in range(sk+sh,n):
        if np.isnan(sa_f[i]): continue
        ab=c[i]>ct[i]; gd=ta[i]>ka[i] and ta[i-1]<=ka[i-1]
        if pos:
            if h[i]>=ep*1.05: tr.append((eb,i,5-COMM*100,'TP')); pos=0; cool=COOLDOWN
            elif l[i]<=ep*0.975: tr.append((eb,i,-2.5-COMM*100,'SL')); pos=0; cool=COOLDOWN
        if not pos and cool==0 and ab and gd: pos=1; ep=c[i]; eb=i
        if not pos and cool>0: cool-=1
    if pos: tr.append((eb,n-1,(c[-1]/ep-1)*100-COMM*100,'OPEN'))
    return tr

def run2p(ct):
    eq=1000; cv=[1000]; op={}
    tl=[]
    for sym,trs in ct.items():
        for e,x,pnl,r in trs: tl.append((e,'entry',sym,pnl)); tl.append((x,'exit',sym,pnl))
    tl.sort()
    exe=0; win=0
    for t,ty,sym,pnl in tl:
        if ty=='entry':
            if len(op)<2: op[sym]=eq/2
        elif ty=='exit':
            if sym in op:
                al=op.pop(sym); eq+=al*(1+pnl/100)-al; cv.append(eq); exe+=1
                if pnl>0: win+=1
    s=pd.Series(cv); pk=s.expanding().max(); dd=((s-pk)/pk*100).min()
    return{'pnl':eq-1000,'dd':dd,'t':exe,'wr':win/exe*100 if exe else 0,'eq':eq}

with open('/data/trading28/config/shariah_coins.json') as f:
    d=json.load(f)
tradeable=sorted(d['halal']+d['halal2'])

# Get top 60 from 3y
coin3={}
for pdir in['2023','prev','1y']:
    for sym in tradeable:
        fp=os.path.join(f'/data/trading28/data/whale_15m_{pdir}',sym+'.json')
        if not os.path.exists(fp): continue
        with open(fp) as f: j=json.load(f)
        c=np.array(j['c'],float);h=np.array(j['h'],float);l=np.array(j['l'],float)
        o=np.array(j['o'],float);ts=j.get('ts',[])
        rp=r8h(c,h,l,o,ts)
        if rp is None: continue
        trs=trades(*rp)
        if len(trs)>=3: coin3[sym]=coin3.get(sym,0)+sum(p for _,_,p,_ in trs)

top60=set(c for c,_ in sorted(coin3.items(),key=lambda x:x[1],reverse=True)[:60])

# Last month
ct_all={}; ct_60={}
for sym in tradeable:
    d=load_lm(sym)
    if d is None: continue
    rp=r8h(*d)
    if rp is None: continue
    trs=trades(*rp)
    if len(trs)>=1:
        ct_all[sym]=trs
        if sym in top60: ct_60[sym]=trs

print('صياد السحابة — SL ثابت -2.7% (2.5% + عمولة)\n')
for label,ct in[('كل العملات',ct_all),('أفضل 60',ct_60)]:
    avail=sum(len(v) for v in ct.values())
    m=run2p(ct)
    print(f'{label}: {len(ct)} عملة | {avail} إشارة | {m["t"]} منفذ | {m["wr"]:.1f}% WR | ${m["pnl"]:+,.0f} | DD={m["dd"]:.1f}% | ${m["eq"]:,.0f}')
