#!/usr/bin/env python3
"""Cloud Hunter — filters V2 — FIXED signal count — last month"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timezone

COMM=0.002; MAX_SLIPPAGE=1.5; COOLDOWN=2
TP_PNL=5-COMM*100
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

def compute_ichimoku(c,h,l,o):
    tk,kj,sk=3,9,18;n=len(c)
    if n<sk+30: return None
    ht=pd.Series(h).rolling(tk).max().values;lt=pd.Series(l).rolling(tk).min().values
    ta=(ht+lt)/2
    hk=pd.Series(h).rolling(kj).max().values;lk=pd.Series(l).rolling(kj).min().values
    ka=(hk+lk)/2
    hs=pd.Series(h).rolling(sk).max().values;ls=pd.Series(l).rolling(sk).min().values
    sb=(hs+ls)/2;sa=(ta+ka)/2;sh=kj
    saf=np.full(n,np.nan);sbf=np.full(n,np.nan)
    for i in range(max(sh,sk),n-sh):
        if i+sh<n: saf[i+sh]=sa[i];sbf[i+sh]=sb[i]
    ct=np.maximum(saf,sbf)
    ema50=pd.Series(c).ewm(span=50,adjust=False).mean().values
    ema200=pd.Series(c).ewm(span=200,adjust=False).mean().values
    sma50=pd.Series(c).rolling(50).mean().values
    sma100=pd.Series(c).rolling(100).mean().values
    # RSI
    delta=np.diff(c);gain=np.maximum(delta,0);loss=np.abs(np.minimum(delta,0))
    rsi=np.full(n,np.nan)
    for i in range(15,n):
        ag=np.mean(gain[i-14:i]);al=np.mean(loss[i-14:i])
        if al==0: rsi[i]=100
        else: rsi[i]=100-100/(1+ag/al)
    # ATR
    tr1=h-l;tr2=np.abs(h-np.roll(c,1));tr3=np.abs(l-np.roll(c,1))
    tr=np.maximum(np.maximum(tr1,tr2),tr3)
    atr=pd.Series(tr).rolling(14).mean().values
    # Body
    body_pct=np.abs(c-o)/(h-l+1e-10)
    # Consecutive above cloud
    cons=np.zeros(n,int)
    for i in range(1,n):
        if c[i]>ct[i] if not np.isnan(ct[i]) else False: cons[i]=cons[i-1]+1
        else: cons[i]=0
    return ct,ta,ka,ema50,ema200,sma50,sma100,rsi,atr,body_pct,cons

def check_filter(flt,i,c,ct,ta,ka,ema50,ema200,sma50,sma100,rsi,atr,body_pct,cons):
    if flt is None: return True
    if flt=='ema200': return not np.isnan(ema200[i]) and c[i]>ema200[i]
    if flt=='ema50': return not np.isnan(ema50[i]) and c[i]>ema50[i]
    if flt=='sma50_100': return i>=100 and not np.isnan(sma50[i]) and not np.isnan(sma100[i]) and sma50[i]>sma100[i]
    if flt=='rsi70': return not np.isnan(rsi[i]) and rsi[i]<70
    if flt=='rsi50': return not np.isnan(rsi[i]) and rsi[i]>50
    if flt=='body40': return body_pct[i]>0.4
    if flt=='dist8': return (c[i]-ct[i])/ct[i]*100<8
    if flt=='cons3': return cons[i]>=3
    if flt=='strong': return i>=50 and not np.isnan(sma50[i]) and not np.isnan(atr[i]) and c[i]>sma50[i]+2*atr[i]
    if flt=='ema200_rsi70': return (not np.isnan(ema200[i]) and c[i]>ema200[i]) and (not np.isnan(rsi[i]) and rsi[i]<70)
    if flt=='ema200_body40': return (not np.isnan(ema200[i]) and c[i]>ema200[i]) and body_pct[i]>0.4
    if flt=='rsi50_body40': return (not np.isnan(rsi[i]) and rsi[i]>50) and body_pct[i]>0.4
    if flt=='ema200_rsi50': return (not np.isnan(ema200[i]) and c[i]>ema200[i]) and (not np.isnan(rsi[i]) and rsi[i]>50)
    return True

def run(c,h,l,o,idx,ich_data,flt):
    ct,ta,ka,ema50,ema200,sma50,sma100,rsi,atr,body_pct,cons=ich_data
    n=len(c);trs=[];pos=0;ep=0;cool=0;eb=0;signals=0
    for i in range(27,n):  # senkou+shift = 18+9 = 27
        if np.isnan(ct[i]): continue
        ab=c[i]>ct[i];gd=ta[i]>ka[i] and ta[i-1]<=ka[i-1]
        signal=ab and gd
        if signal:
            if check_filter(flt,i,c,ct,ta,ka,ema50,ema200,sma50,sma100,rsi,atr,body_pct,cons):
                signals+=1  # filtered signal
        filtered_signal=signal and check_filter(flt,i,c,ct,ta,ka,ema50,ema200,sma50,sma100,rsi,atr,body_pct,cons)
        if pos:
            if h[i]>=ep*1.05: trs.append((eb,i,TP_PNL));pos=0;cool=COOLDOWN
            elif l[i]<=ep*0.975: pnl=max((c[i]/ep-1)*100-COMM*100,-2.5*MAX_SLIPPAGE-COMM*100);trs.append((eb,i,pnl));pos=0;cool=COOLDOWN
        if not pos and cool==0 and filtered_signal: pos=1;ep=c[i];eb=i
        if not pos and cool>0: cool-=1
    if pos: trs.append((eb,n-1,(c[-1]/ep-1)*100-COMM*100))
    return trs,signals

def r2p(ct):
    eq=1000;cv=[1000];op={};tl=[]
    for sym,trs_sig in ct.items():
        trs=trs_sig[0]
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
    return{'pnl':eq-1000,'dd':dd,'t':exe,'wr':win/exe*100 if exe else 0,'eq':eq}

with open('/data/trading28/config/shariah_coins.json') as f: d=json.load(f)
tradeable=sorted(d['halal']+d['halal2'])

# Top 60 from 3y
coin3={}
for pdir in['2023','prev','1y']:
    for sym in tradeable:
        fp=os.path.join(f'/data/trading28/data/whale_15m_{pdir}',sym+'.json')
        if not os.path.exists(fp): continue
        with open(fp) as f: j=json.load(f)
        c=np.array(j['c'],float);h=np.array(j['h'],float)
        l=np.array(j['l'],float);o=np.array(j['o'],float);ts=j.get('ts',[])
        rp=r8h(c,h,l,o,ts)
        if rp is None: continue
        c8,h8,l8,o8,idx=rp
        ich=compute_ichimoku(c8,h8,l8,o8)
        if ich is None: continue
        trs,_=run(c8,h8,l8,o8,idx,ich,None)
        if len(trs)>=3: coin3[sym]=coin3.get(sym,0)+sum(p for _,_,p in trs)
top60=set(c for c,_ in sorted(coin3.items(),key=lambda x:x[1],reverse=True)[:60])

filters=[
    (None,'بدون فلتر'),
    ('ema200','السعر > EMA200'),
    ('ema50','السعر > EMA50'),
    ('sma50_100','SMA50 > SMA100'),
    ('rsi70','RSI < 70'),
    ('rsi50','RSI > 50'),
    ('body40','جسم > 40%'),
    ('dist8','بُعد < 8%'),
    ('cons3','3 فوق السحابة'),
    ('ema200_rsi70','EMA200 + RSI<70'),
    ('ema200_body40','EMA200 + جسم>40%'),
    ('rsi50_body40','RSI>50 + جسم>40%'),
    ('ema200_rsi50','EMA200 + RSI>50'),
]

print(f'☁️ صياد السحابة | SL=-2.7% | يوليو 2026 | أفضل 60\n')
print(f'{"فلتر":>24s} | {"إشارات":>5s} | {"منفذ":>4s} | {"WR":>5s} | {"ربح$":>7s} | {"سحب":>6s}')
print('─'*67)

for flt,label in filters:
    ct={};total_sig=0
    for sym in top60:
        d=ld(sym)
        if d is None: continue
        c,h,l,o,ts=d
        rp=r8h(c,h,l,o,ts)
        if rp is None: continue
        c8,h8,l8,o8,idx=rp
        ich=compute_ichimoku(c8,h8,l8,o8)
        if ich is None: continue
        trs,sigs=run(c8,h8,l8,o8,idx,ich,flt)
        total_sig+=sigs
        if len(trs)>=1: ct[sym]=(trs,sigs)
    if not ct: continue
    m=r2p(ct)
    print(f'{label:>24s} | {total_sig:5d} | {m["t"]:4d} | {m["wr"]:4.1f}% | ${m["pnl"]:+6.0f} | {m["dd"]:5.1f}%')
