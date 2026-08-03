#!/usr/bin/env python3
"""Cloud Hunter — Best combos: top60 + filters"""
import json, os, numpy as np, pandas as pd

COMM=0.002; MAX_SLIPPAGE=1.5; COOLDOWN=2

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

def trades(c,h,l,o,idx,flt=None):
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
    ri=rsi(c);ema4h=pd.Series(c).ewm(span=25,adjust=False).mean().values
    trs=[];sig=0;pos=0;ep=0;cool=0;eb=0
    for i in range(sk+sh,n):
        if np.isnan(saf[i]) or np.isnan(sbf[i]): continue
        ct=max(saf[i],sbf[i])
        ab=c[i]>ct;gd=ta[i]>ka[i] and ta[i-1]<=ka[i-1]
        s=ab and gd and not np.isnan(ri[i]) and ri[i]>50
        if s and flt=='4h': s=s and c[i]>ema4h[i]
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

def run_pf(coin_trades,mx,pc,ll=0,ph=0):
    eq=1000;cv=[1000];op={};tl=[]
    for sym,(trs,_) in coin_trades.items():
        for et,xt,pnl in trs:
            em=et.value//10**6;xm=xt.value//10**6
            tl.append((em,'e',sym,pnl));tl.append((xm,'x',sym,pnl))
    tl.sort();ex=0;wi=0;cl=0;pu=0
    for t,ty,sym,pnl in tl:
        if ty=='e':
            if t<pu: continue
            if len(op)<mx: op[sym]=eq*pc
        elif ty=='x':
            if sym in op:
                al=op.pop(sym);nv=al*(1+pnl/100);eq+=nv-al;cv.append(eq);ex+=1
                if pnl>0: wi+=1;cl=0
                else: cl+=1
                if ll>0 and cl>=ll: pu=t+ph*3600*1000;cl=0
    for sym,al in list(op.items()): eq+=al*0.99;del op[sym]
    s=pd.Series(cv);pk=s.expanding().max();dd=((s-pk)/pk*100).min()
    wr=wi/ex*100 if ex else 0
    return{'pnl':eq-1000,'dd':dd,'t':ex,'wr':wr,'eq':eq}

with open('/data/trading28/config/shariah_coins.json') as f: d=json.load(f)
ac=sorted(d['halal']+d['halal2'])

# Rank coins by total PnL (no filter)
cp={}
for pdir in['2023','prev','1y']:
    for s in ac:
        d_=load(s,pdir)
        if d_ is None: continue
        rp=r8h(*d_)
        if rp is None: continue
        c8,h8,l8,o8,ix=rp
        trs,_=trades(c8,h8,l8,o8,ix)
        if len(trs)>=3: cp[s]=cp.get(s,0)+sum(p for _,_,p in trs)
rk=sorted(cp.items(),key=lambda x:x[1],reverse=True)
top60=set(c for c,_ in rk[:60])

# Exclude negative in 2+ periods
cpp={}
for pdir in['2023','prev','1y']:
    for s in top60:
        d_=load(s,pdir)
        if d_ is None: continue
        rp=r8h(*d_)
        if rp is None: continue
        c8,h8,l8,o8,ix=rp
        trs,_=trades(c8,h8,l8,o8,ix)
        if trs:
            pnl=sum(p for _,_,p in trs)
            if s not in cpp: cpp[s]={}
            cpp[s][pdir]=pnl
ex=set()
for s,pp in cpp.items():
    if sum(1 for p in pp.values() if p<0)>=2: ex.add(s)
clean=sorted(top60-ex)

configs=[
    (None,2,0.50,0,0,f'RSI>50 + {len(clean)}عملة + صفقتين'),
    ('4h',2,0.50,0,0,f'RSI>50+4hEMA + {len(clean)}عملة + صفقتين'),
    ('4h',3,0.33,2,24,f'RSI>50+4hEMA + {len(clean)}عملة + 3×33% + تبريد'),
]

print(f"☁️ صياد السحابة | 3 سنوات\n")
for flt,mx,pc,ll,ph,label in configs:
    print(f"📋 {label}")
    print(f"{'':>6s} | {'عملات':>4s} | {'إشارات':>5s} | {'منفذ':>4s} | {'WR':>5s} | {'ربح$':>8s} | {'سحب':>6s} | {'نهائي$':>8s}")
    print("─"*68)
    g=0;gs=0
    for pn,pdir in[('2023','2023'),('PREV','prev'),('CUR','1y')]:
        ct={}
        for s in clean:
            d_=load(s,pdir)
            if d_ is None: continue
            rp=r8h(*d_)
            if rp is None: continue
            c8,h8,l8,o8,ix=rp
            trs,sig=trades(c8,h8,l8,o8,ix,flt=flt)
            if len(trs)>=3: ct[s]=(trs,sig)
        N=len(ct);ts=sum(v[1] for v in ct.values())
        m=run_pf(ct,mx,pc,ll,ph)
        gs+=ts
        print(f"{pn:>6s} | {N:4d} | {ts:5d} | {m['t']:4d} | {m['wr']:4.1f}% | ${m['pnl']:+7,.0f} | {m['dd']:5.1f}% | ${m['eq']:7,.0f}")
        g+=m['pnl']
    print(f"{'─'*68}")
    print(f"💰 | {'':>4s} | {gs:5d} | {'':>4s} | {'':>5s} | ${g:+7,.0f}\n\n")
