#!/usr/bin/env python3
"""Test cooldown impact"""
import json, os, numpy as np, pandas as pd

COMM=0.002; MAX_SLIPPAGE=1.5

def load(s,p):
    f=os.path.join(f'/data/trading28/data/whale_15m_{p}',f'{s}.json')
    if not os.path.exists(f): return None
    with open(f) as fh: j=json.load(fh)
    return np.array(j['c'],float),np.array(j['h'],float),np.array(j['l'],float),np.array(j['o'],float),j.get('ts',[])

def r8h(c,h,l,o,ts):
    idx=pd.to_datetime(np.array(ts),unit='ms')
    df=pd.DataFrame({'o':o,'h':h,'l':l,'c':c},index=idx)
    r=df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
    return r['c'].values,r['h'].values,r['l'].values,r['o'].values,r.index

def rsi(c,p=14):
    n=len(c);r=np.full(n,np.nan)
    if n<p+1: return r
    d=np.diff(c);g=np.maximum(d,0);l=np.abs(np.minimum(d,0))
    for i in range(p+1,n+1):
        ag=np.mean(g[i-p:i]);al=np.mean(l[i-p:i])
        r[i-1]=100-100/(1+ag/al) if al!=0 else 100
    return r

def trades(c,h,l,o,idx,COOLDOWN=2):
    tk,kj,sk=3,9,18;tp,sl=5,2.5;n=len(c)
    if n<200: return [],0
    ht=pd.Series(h).rolling(tk).max().values;lt=pd.Series(l).rolling(tk).min().values
    ta=(ht+lt)/2
    hk=pd.Series(h).rolling(kj).max().values;lk=pd.Series(l).rolling(kj).min().values
    ka=(hk+lk)/2
    hs=pd.Series(h).rolling(sk).max().values;ls=pd.Series(l).rolling(sk).min().values
    sb=(hs+ls)/2;sa=(ta+ka)/2;sh=kj
    saf=np.full(n,np.nan);sbf=np.full(n,np.nan)
    for i in range(max(sh,sk),n-sh):
        if i+sh<n: saf[i+sh]=sa[i];sbf[i+sh]=sb[i]
    ri=rsi(c)
    trs=[];sig=0;pos=0;ep=0;cool=0;eb=0
    for i in range(sk+sh,n):
        if np.isnan(saf[i]) or np.isnan(sbf[i]): continue
        ct=max(saf[i],sbf[i])
        ab=c[i]>ct;gd=ta[i]>ka[i] and ta[i-1]<=ka[i-1]
        s=ab and gd and not np.isnan(ri[i]) and ri[i]>50
        if s: sig+=1
        if pos:
            if h[i]>=ep*1.05: trs.append((eb,idx[i],tp-COMM*100));pos=0;cool=COOLDOWN
            elif l[i]<=ep*0.975:
                pnl=max((c[i]/ep-1)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                trs.append((eb,idx[i],pnl));pos=0;cool=COOLDOWN
        if not pos and cool==0 and s: pos=1;ep=c[i];eb=idx[i]
        if not pos and cool>0: cool-=1
    if pos: trs.append((eb,idx[-1],(c[-1]/ep-1)*100-COMM*100))
    return trs,sig

def run_pf(coin_trades):
    eq=1000;cv=[1000];op={};tl=[]
    for sym,(trs,_) in coin_trades.items():
        for et,xt,pnl in trs:
            tl.append((et.value//10**6,'e',sym,pnl));tl.append((xt.value//10**6,'x',sym,pnl))
    tl.sort();ex=0;wi=0
    for t,ty,sym,pnl in tl:
        if ty=='e':
            if len(op)<2: op[sym]=eq*0.5
        elif ty=='x':
            if sym in op:
                al=op.pop(sym);nv=al*(1+pnl/100);eq+=nv-al;cv.append(eq);ex+=1
                if pnl>0: wi+=1
    for sym,al in list(op.items()): eq+=al*0.99;del op[sym]
    s=pd.Series(cv);pk=s.expanding().max();dd=((s-pk)/pk*100).min()
    return{'pnl':eq-1000,'dd':dd,'t':ex,'wr':wi/ex*100 if ex else 0}

with open('/data/trading28/config/shariah_coins.json') as f: d=json.load(f)
ac=sorted(d['halal']+d['halal2'])

for cd,label in[(0,'بدون تبريد'),(2,'تبريد 16h (2 شمعة)')]:
    print(f"📋 {label}")
    print(f"{'':>6s} | {'عملات':>4s} | {'إشارات':>5s} | {'منفذ':>4s} | {'WR':>5s} | {'ربح$':>8s} | {'سحب':>6s}")
    print("─"*58)
    g=0
    for pn,pdir in[('2023','2023'),('PREV','prev'),('CUR','1y')]:
        ct={}
        for s in ac:
            d_=load(s,pdir)
            if d_ is None: continue
            rp=r8h(*d_)
            if rp is None: continue
            c8,h8,l8,o8,ix=rp
            trs,sig=trades(c8,h8,l8,o8,ix,COOLDOWN=cd)
            if len(trs)>=3: ct[s]=(trs,sig)
        N=len(ct);ts=sum(v[1] for v in ct.values())
        m=run_pf(ct)
        print(f"{pn:>6s} | {N:4d} | {ts:5d} | {m['t']:4d} | {m['wr']:4.1f}% | ${m['pnl']:+7,.0f} | {m['dd']:5.1f}%")
        g+=m['pnl']
    print(f"{'─'*58}")
    print(f"💰 | {'':>4s} | {'':>5s} | {'':>4s} | {'':>5s} | ${g:+7,.0f}\n\n")
