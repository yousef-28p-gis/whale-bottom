#!/usr/bin/env python3
"""Cloud Hunter — test 10+ filters with fixed SL — last month"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timezone

COMM=0.002; COOLDOWN=2; SL=-2.5-COMM*100; TP=5-COMM*100
CUTOFF=int(datetime(2026,7,3,tzinfo=timezone.utc).timestamp()*1000)

def ld(sym):
    p=os.path.join('/data/trading28/data/whale_15m_1y',sym+'.json')
    if not os.path.exists(p): return None
    with open(p) as f: j=json.load(f)
    ts=j.get('ts',[]);c=np.array(j['c'],float);h=np.array(j['h'],float)
    l=np.array(j['l'],float);o=np.array(j['o'],float);v=np.array(j.get('v',[0]*len(c)),float)
    mask=np.array(ts)>=CUTOFF
    if mask.sum()<200: return None
    return c[mask],h[mask],l[mask],o[mask],v[mask] if len(v)>0 else None,[t for i,t in enumerate(ts) if mask[i]]

def r8h(c,h,l,o,ts):
    idx=pd.to_datetime(np.array(ts),unit='ms')
    df=pd.DataFrame({'o':o,'h':h,'l':l,'c':c},index=idx)
    r=df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
    return r['c'].values,r['h'].values,r['l'].values,r['o'].values,r.index

def ichimoku_all(c,h,l,o):
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
    # Filters
    ema20=pd.Series(c).ewm(span=20,adjust=False).mean().values
    ema50=pd.Series(c).ewm(span=50,adjust=False).mean().values
    ema200=pd.Series(c).ewm(span=200,adjust=False).mean().values
    sma50=pd.Series(c).rolling(50).mean().values
    sma100=pd.Series(c).rolling(100).mean().values
    # ATR
    tr1=h-l;tr2=np.abs(h-np.roll(c,1));tr3=np.abs(l-np.roll(c,1))
    tr=np.maximum(np.maximum(tr1,tr2),tr3)
    atr=pd.Series(tr).rolling(14).mean().values
    # RSI
    delta=np.diff(c);gain=np.maximum(delta,0);loss=np.abs(np.minimum(delta,0))
    avg_gain=np.full(n,np.nan);avg_loss=np.full(n,np.nan)
    for i in range(14,n): avg_gain[i]=np.mean(gain[i-14:i]);avg_loss[i]=np.mean(loss[i-14:i])
    rsi=np.full(n,np.nan)
    for i in range(14,n):
        if avg_loss[i]==0: rsi[i]=100
        else: rsi[i]=100-100/(1+avg_gain[i]/avg_loss[i])
    # Body %
    body_pct=np.abs(c-o)/(h-l+1e-10)
    return ct,ta,ka,saf,sbf,ema20,ema50,ema200,sma50,sma100,atr,rsi,body_pct

def run(c,h,l,o,idx,ct,ta,ka,flt):
    n=len(c);trs=[];pos=0;ep=0;cool=0;eb=0
    sh=9;sk=18
    for i in range(sk+sh,n):
        if np.isnan(ct[i]): continue
        # Base signal
        ab=c[i]>ct[i]; gd=ta[i]>ka[i] and ta[i-1]<=ka[i-1]
        signal=ab and gd
        # Apply filter
        if signal and flt is not None:
            if flt=='ema200':
                if not (pd.Series(c).ewm(span=200).mean().values[i] and c[i]>pd.Series(c).ewm(span=200).mean().values[i]): signal=False
            elif flt=='ema50':
                if not (pd.Series(c).ewm(span=50).mean().values[i] and c[i]>pd.Series(c).ewm(span=50).mean().values[i]): signal=False
            elif flt=='sma50_100':
                s50=pd.Series(c).rolling(50).mean().values; s100=pd.Series(c).rolling(100).mean().values
                if i<100 or np.isnan(s50[i]) or not s50[i]>s100[i]: signal=False
            elif flt=='no_overbought':
                if pd.Series(np.diff(c)).rolling(14).mean().values[i-1] and get_rsi(c,i)>70: signal=False
            elif flt=='strong_trend':
                sma50=pd.Series(c).rolling(50).mean().values; atr14=get_atr(c,h,l,i)
                if i<50 or atr14==0: signal=False
                elif c[i]<sma50[i]+2*atr14: signal=False  # far above SMA
            elif flt=='volume':
                body=abs(c[i]-o[i])/(h[i]-l[i]+1e-10)
                if body<0.4: signal=False  # require strong candle
            elif flt=='cloud_distance':
                dist=(c[i]-ct[i])/ct[i]*100
                if dist>8: signal=False  # too far above cloud
            elif flt=='consecutive':
                if i<3: signal=False
                elif not all(c[j]>ct[j] for j in range(i-2,i+1)): signal=False
        
        if pos:
            if h[i]>=ep*1.05: trs.append((eb,i,TP));pos=0;cool=COOLDOWN
            elif l[i]<=ep*0.975: trs.append((eb,i,SL));pos=0;cool=COOLDOWN
        if not pos and cool==0 and signal: pos=1;ep=c[i];eb=i
        if not pos and cool>0: cool-=1
    if pos: trs.append((eb,n-1,(c[-1]/ep-1)*100-COMM*100))
    return trs

def get_rsi(c,i):
    if i<15: return 50
    delta=np.diff(c[:i+1]);gain=np.maximum(delta,0);loss=np.abs(np.minimum(delta,0))
    ag=np.mean(gain[-14:]);al=np.mean(loss[-14:])
    if al==0: return 100
    return 100-100/(1+ag/al)

def get_atr(c,h,l,i):
    if i<15: return 0
    tr1=h[i-13:i+1]-l[i-13:i+1];tr2=np.abs(h[i-13:i+1]-np.roll(c[i-13:i+1],1))
    tr3=np.abs(l[i-13:i+1]-np.roll(c[i-13:i+1],1));tr=np.maximum(np.maximum(tr1,tr2),tr3)
    return np.mean(tr)

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
    return{'pnl':eq-1000,'dd':dd,'t':exe,'wr':win/exe*100 if exe else 0,'eq':eq}

with open('/data/trading28/config/shariah_coins.json') as f: d=json.load(f)
tradeable=sorted(d['halal']+d['halal2'])

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
        ich=ichimoku_all(c8,h8,l8,o8)
        if ich is None: continue
        ct,ta,ka,_,_,_,_,_,_,_,_,_,_=ich
        trs=run(c8,h8,l8,o8,idx,ct,ta,ka,None)
        if len(trs)>=3: coin3[sym]=coin3.get(sym,0)+sum(p for _,_,p in trs)
top60=set(c for c,_ in sorted(coin3.items(),key=lambda x:x[1],reverse=True)[:60])

filters={
    None:'بدون فلتر',
    'ema200':'السعر > EMA200',
    'ema50':'السعر > EMA50',
    'sma50_100':'SMA50 > SMA100',
    'no_overbought':'RSI < 70',
    'strong_trend':'سعر > SMA50+2ATR',
    'volume':'جسم الشمعة > 40%',
    'cloud_distance':'بُعد عن السحابة < 8%',
    'consecutive':'3 شمعات فوق السحابة',
}

print(f'☁️ صياد السحابة — SL ثابت -2.7% — يوليو 2026 — أفضل 60\n')
print(f'{"فلتر":>22s} | {"إشارات":>5s} | {"منفذ":>4s} | {"WR":>5s} | {"ربح$":>7s} | {"سحب":>6s}')
print('─'*65)

for flt,label in filters.items():
    ct={}
    total_signals=0
    for sym in top60:
        d=ld(sym)
        if d is None: continue
        c,h,l,o,v,ts=d
        rp=r8h(c,h,l,o,ts)
        if rp is None: continue
        c8,h8,l8,o8,idx=rp
        ich=ichimoku_all(c8,h8,l8,o8)
        if ich is None: continue
        _,ta,ka,_,_,_,_,_,_,_,_,_,_=ich
        trs=run(c8,h8,l8,o8,idx,ich[0],ta,ka,flt)
        sig_count=sum(1 for i in range(18+9,len(c8)) if not np.isnan(ich[0][i]) and c8[i]>ich[0][i] and ta[i]>ka[i] and ta[i-1]<=ka[i-1])
        total_signals+=sig_count
        if len(trs)>=1: ct[sym]=trs
    if not ct: continue
    m=r2p(ct)
    print(f'{label:>22s} | {total_signals:5d} | {m["t"]:4d} | {m["wr"]:4.1f}% | ${m["pnl"]:+6.0f} | {m["dd"]:5.1f}%')
