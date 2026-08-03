#!/usr/bin/env python3
"""Cloud Hunter — all coins RSI>50 — detailed report format"""
import json, os, numpy as np, pandas as pd

COMM=0.002; MAX_SLIPPAGE=1.5; COOLDOWN=2; CAP=1000

def load(s,p):
    f=os.path.join(f'/data/trading28/data/whale_15m_{p}',f'{s}.json')
    if not os.path.exists(f): return None
    with open(f) as fh: j=json.load(fh)
    return np.array(j['c'],float),np.array(j['h'],float),np.array(j['l'],float),np.array(j['o'],float),j.get('ts',[])

def r8h(c,h,l,o,ts):
    try:
        idx=pd.to_datetime(np.array(ts),unit='ms')
        df=pd.DataFrame({'o':o,'h':h,'l':l,'c':c},index=idx)
        r=df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
        return r['c'].values,r['h'].values,r['l'].values,r['o'].values,r.index
    except: return None

def rsi(c,p=14):
    n=len(c);r=np.full(n,np.nan)
    if n<p+1: return r
    d=np.diff(c);g=np.maximum(d,0);l=np.abs(np.minimum(d,0))
    for i in range(p+1,n+1):
        ag=np.mean(g[i-p:i]);al=np.mean(l[i-p:i])
        r[i-1]=100-100/(1+ag/al) if al!=0 else 100
    return r

def trades(c,h,l,o,idx):
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

def run_pf(coin_trades,mx,pc):
    eq=CAP;cv=[CAP];op={};tl=[]
    for sym,(trs,_) in coin_trades.items():
        for et,xt,pnl in trs:
            em=et.value//10**6;xm=xt.value//10**6
            tl.append((em,'e',sym,pnl));tl.append((xm,'x',sym,pnl))
    tl.sort();ex=0;wi=0
    for t,ty,sym,pnl in tl:
        if ty=='e':
            if len(op)<mx: op[sym]=eq*pc
        elif ty=='x':
            if sym in op:
                al=op.pop(sym);nv=al*(1+pnl/100);eq+=nv-al;cv.append(eq);ex+=1
                if pnl>0: wi+=1
    for sym,al in list(op.items()): eq+=al*0.99;del op[sym]
    s=pd.Series(cv);pk=s.expanding().max();dd=((s-pk)/pk*100).min()
    wr=wi/ex*100 if ex else 0
    rets=np.array([(cv[i+1]-cv[i])/cv[i]*100 for i in range(len(cv)-1)])
    sharpe=(np.mean(rets)/np.std(rets)*np.sqrt(len(rets))).round(2) if len(rets)>1 and np.std(rets)>0 else 0
    return{'pnl':eq-CAP,'dd':dd,'t':ex,'wr':wr,'eq':eq,'sharpe':sharpe,'cv':cv}

with open('/data/trading28/config/shariah_coins.json') as f: d=json.load(f)
ac=sorted(d['halal']+d['halal2'])

# Run per period with full stats
configs=[(2,0.50,'صفقتين × 50%'),(3,0.33,'3 صفقات × 33%'),(1,1.0,'صفقة × 100%')]

print("☁️ صياد السحابة — RSI>50 — كل العملات الحلال\n")
print("⚙️ 8h | 3/9/18 | TP=5% | SL=2.5% | MAX_SLIPPAGE=1.5 | تبريد 16h\n")
print("="*75)

for mx,pc,plabel in configs:
    print(f"\n📋 إدارة رأس المال: {plabel}")
    print("─"*75)
    gp=0;gt=0;gw=0;gl=0;gcv=[]
    for pn,pdir in[('2023','2023'),('PREV','prev'),('CUR','1y')]:
        ct={}
        for s in ac:
            d_=load(s,pdir)
            if d_ is None: continue
            rp=r8h(*d_)
            if rp is None: continue
            c8,h8,l8,o8,ix=rp
            trs,sig=trades(c8,h8,l8,o8,ix)
            if len(trs)>=3: ct[s]=(trs,sig)
        N=len(ct);ts=sum(v[1] for v in ct.values())
        m=run_pf(ct,mx,pc)
        # Per-period stats
        all_pnls=[]
        for sym,(trs,_) in ct.items():
            for _,_,pnl in trs: all_pnls.append(pnl)
        wins=[p for p in all_pnls if p>0];losses=[p for p in all_pnls if p<0]
        avg_win=np.mean(wins) if wins else 0
        avg_loss=np.mean(losses) if losses else 0
        rr=abs(avg_win/avg_loss) if avg_loss!=0 else 0
        
        months=(len(ix)*8)/(24*30)
        annual=((m['eq']/CAP)**(12/months)-1)*100 if months>0 else 0
        
        gp+=m['pnl'];gt+=m['t'];gw+=len(wins);gl+=len(losses)
        
        print(f"\n📅 {pn} | ⏱️ 8h | 📊 {N} عملة | {ts} إشارة | 🔍 LA=shift(1)")
        print(f"📋 {m['t']} صفقة | 🟢 {len(wins)} ربح | 🔴 {len(losses)} خسارة | 📈 WR={m['wr']:.1f}%")
        print(f"💵 ربح=${m['pnl']:+,.0f} | 💸 إنفاق=${m['eq']-m['pnl']:,.0f} | 💰 نهائي=${m['eq']:,.0f}")
        print(f"🟢 م.ربح={avg_win:+.2f}% | 🔴 م.خسارة={avg_loss:+.2f}% | 📊 R:R=1:{rr:.2f} | شارپ={m['sharpe']}")
        print(f"🏦 سحب={m['dd']:.1f}% | 📈 سنوي={annual:+.1f}% | 🎯 TP=5% | 🛑 SL=2.5% | 🐌 تبريد=16h")
    
    # Grand total
    print(f"\n{'─'*75}")
    print(f"💰 المجموع الكلي: {gt} صفقة | 🟢 {gw} ربح | 🔴 {gl} خسارة | 📈 WR={gw/gt*100:.1f}% | ${gp:+,.0f}")
    print(f"💵 عائد = {gp/CAP*100:+.0f}% على رأس المال")
