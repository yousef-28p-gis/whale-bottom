#!/usr/bin/env python3
"""3 solutions for out-of-sample robustness"""
import json, os, numpy as np, pandas as pd, gc
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN=48

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def load_json(path):
    with open(path) as f: return json.load(f)

def compute_trends(c, ts, n):
    try:
        if ts and len(ts)==n:
            idx=pd.to_datetime(np.array(ts),unit='ms')
            df=pd.DataFrame({'c':c},index=idx)
            c4h=df['c'].resample('4h').last().dropna().values
            e50=ema(c4h,50); e200=ema(c4h,200)
            e50a=np.zeros(n); e200a=np.zeros(n)
            for i in range(n):
                j=i//16
                if j<len(e50): e50a[i]=e50[j]; e200a[i]=e200[j]
            t4=e50a>e200a
            c1h=df['c'].resample('1h').last().dropna().values
            e20=ema(c1h,20); e50h=ema(c1h,50)
            e20a=np.zeros(n); e50a2=np.zeros(n)
            for i in range(n):
                j=i//4
                if j<len(e20): e20a[i]=e20[j]; e50a2[i]=e50h[j]
            t1=e20a>e50a2
            return t4, t1
    except: pass
    return np.ones(n,bool), np.ones(n,bool)

def make_entries(c,h,l_,o,n,LB,sp,wp_w=None,wp_s=None):
    sm=3
    ln=pd.Series(l_).shift(1).rolling(LB).min().values
    lc=np.zeros(n)
    for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
    sc=pd.Series(lc).ewm(span=sm,adjust=False).mean().values
    hc=pd.Series(sc).rolling(LB).max().values
    sr=np.where(l_<=ln,(sc+hc*2)/3,0)
    wp=pd.Series(sr).ewm(span=sm,adjust=False).mean().values
    wp_up=wp>np.roll(wp,1)
    sma_h=pd.Series(h).rolling(sp).mean().values
    ssl_c=np.zeros(n,int)
    for i in range(sp,n):
        if h[i-1]>sma_h[i-1]: ssl_c[i]=1
        else: ssl_c[i]=-1
    mult = wp_w if wp_w else 2
    def entry_gen(filt_arr):
        le=np.zeros(n,bool)
        for i in range(200,n):
            if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[max(0,i-2)]*mult and wp[i]>0 and filt_arr[i]:
                le[i]=True
        return le
    return entry_gen

def simulate(le, c, h, l_, n, tp, sl):
    if le.sum()<3: return None
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l_[i]<=ep*(1-sl/100):
                raw=(c[i]/ep-1)*100-COMM*100
                pnl=max(raw,-sl*MAX_SLIPPAGE-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=[p for p in t if p>0]
    wr=len(w)/len(t)*100
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    return {'t':len(t),'wr':wr,'dd':dd,'eq':eq,'pnl':eq-CAP}

def test_config(c,h,l_,o,n,t4,entry_gen,tp,sl):
    le=entry_gen(t4)
    return simulate(le,c,h,l_,n,tp,sl)

def load_period(sym, period):
    path = f'/data/trading28/data/whale_15m_{period}/{sym}.json'
    if not os.path.exists(path): return None
    d=load_json(path)
    c=np.array(d['c'],float); h=np.array(d['h'],float)
    l_=np.array(d['l'],float); o=np.array(d['o'],float)
    ts=d.get('ts',[]); n=len(c)
    t4,t1=compute_trends(c,ts,n)
    return c,h,l_,o,n,ts,t4,t1

# ── Load BTC for filter ──
btc_cur=load_period('BTC','1y')
btc_prev=load_period('BTC','prev')

coins=sorted(set(f.replace('.json','') for f in os.listdir('/data/trading28/data/whale_15m_1y') 
    if f.endswith('.json') and f!='_manifest.json'))

# ── Solution 1: Optimize on prev, test on cur ──
print("Sol 1: Optimize on PREV, test on CUR...", flush=True)
LB_opts=[30,50,70]; SSL_opts=[5,10,20]; TP_SL=[(2,1),(3,1.5),(5,2.5)]
s1_cur={'t':0,'w':0,'pnl':0,'dd':0,'coins':0}
s1_prev={'t':0,'w':0,'pnl':0,'dd':0,'coins':0}

for sym in coins:
    if sym in ['0G','ALLO','ASTER']: continue  # skip new coins
    p=load_period(sym,'prev')
    if p is None: continue
    c,h,l_,o,n,ts,t4,t1=p
    
    best_eq=0; best_cfg=None
    for LB in LB_opts:
        eg=make_entries(c,h,l_,o,n,LB,10)
        for sp in SSL_opts:
            eg2=make_entries(c,h,l_,o,n,LB,sp)
            for tp,sl in TP_SL:
                r=test_config(c,h,l_,o,n,t4,eg2,tp,sl)
                if r and r['eq']>best_eq: best_eq=r['eq']; best_cfg=(LB,sp,tp,sl)
    
    if best_cfg is None: continue
    LB,sp,tp,sl=best_cfg
    
    # Test on prev (training)
    eg=make_entries(c,h,l_,o,n,LB,sp)
    r=test_config(c,h,l_,o,n,t4,eg,tp,sl)
    if r: s1_prev['t']+=r['t']; s1_prev['w']+=int(r['t']*r['wr']/100); s1_prev['pnl']+=r['pnl']; s1_prev['coins']+=1
    
    # Test on cur (validation)
    p2=load_period(sym,'1y')
    if p2 is None: continue
    c2,h2,l2,o2,n2,ts2,t4_2,t1_2=p2
    eg2=make_entries(c2,h2,l2,o2,n2,LB,sp)
    r2=test_config(c2,h2,l2,o2,n2,t4_2,eg2,tp,sl)
    if r2: s1_cur['t']+=r2['t']; s1_cur['w']+=int(r2['t']*r2['wr']/100); s1_cur['pnl']+=r2['pnl']; s1_cur['coins']+=1
    
    gc.collect()

wr1p=s1_prev['w']/s1_prev['t']*100 if s1_prev['t']>0 else 0
wr1c=s1_cur['w']/s1_cur['t']*100 if s1_cur['t']>0 else 0
print(f"  PREV(train): {s1_prev['coins']}c {s1_prev['t']}t WR={wr1p:.1f}% ${s1_prev['pnl']:+.0f}")
print(f"  CUR(test):   {s1_cur['coins']}c {s1_cur['t']}t WR={wr1c:.1f}% ${s1_cur['pnl']:+.0f}")

# ── Solution 2: Fixed params for all coins (LB=50, SSL=10, TP5/SL2.5) ──
print("\nSol 2: Fixed params LB50/SSL10/TP5/SL2.5...", flush=True)
s2_cur={'t':0,'w':0,'pnl':0,'coins':0}; s2_prev={'t':0,'w':0,'pnl':0,'coins':0}
for sym in coins:
    for pname,plabel in [('prev',s2_prev),('1y',s2_cur)]:
        p=load_period(sym,pname)
        if p is None: continue
        c,h,l_,o,n,ts,t4,t1=p
        eg=make_entries(c,h,l_,o,n,50,10)
        r=test_config(c,h,l_,o,n,t4,eg,5,2.5)
        if r: plabel['t']+=r['t']; plabel['w']+=int(r['t']*r['wr']/100); plabel['pnl']+=r['pnl']; plabel['coins']+=1
    gc.collect()

wr2p=s2_prev['w']/s2_prev['t']*100 if s2_prev['t']>0 else 0
wr2c=s2_cur['w']/s2_cur['t']*100 if s2_cur['t']>0 else 0
print(f"  PREV: {s2_prev['coins']}c {s2_prev['t']}t WR={wr2p:.1f}% ${s2_prev['pnl']:+.0f}")
print(f"  CUR:  {s2_cur['coins']}c {s2_cur['t']}t WR={wr2c:.1f}% ${s2_cur['pnl']:+.0f}")

# ── Solution 3: BTC filter ──
print("\nSol 3: BTC 4h filter...", flush=True)
s3_cur={'t':0,'w':0,'pnl':0,'coins':0}; s3_prev={'t':0,'w':0,'pnl':0,'coins':0}
for sym in coins:
    for pname,plabel,btc_data in [('prev',s3_prev,btc_prev),('1y',s3_cur,btc_cur)]:
        p=load_period(sym,pname)
        if p is None or btc_data is None: continue
        c,h,l_,o,n,ts,t4,t1=p
        bc,bh,bl,bo,bn,bts,bt4,bt1=btc_data
        if n>bn: n=bn
        eg=make_entries(c[:n],h[:n],l_[:n],o[:n],n,50,10)
        btc_filt = t4[:n] & bt4[:n]  # coin 4h AND BTC 4h both up
        r=test_config(c[:n],h[:n],l_[:n],o[:n],n,btc_filt,eg,5,2.5)
        if r: plabel['t']+=r['t']; plabel['w']+=int(r['t']*r['wr']/100); plabel['pnl']+=r['pnl']; plabel['coins']+=1
    gc.collect()

wr3p=s3_prev['w']/s3_prev['t']*100 if s3_prev['t']>0 else 0
wr3c=s3_cur['w']/s3_cur['t']*100 if s3_cur['t']>0 else 0
print(f"  PREV: {s3_prev['coins']}c {s3_prev['t']}t WR={wr3p:.1f}% ${s3_prev['pnl']:+.0f}")
print(f"  CUR:  {s3_cur['coins']}c {s3_cur['t']}t WR={wr3c:.1f}% ${s3_cur['pnl']:+.0f}")

# ── FINAL SUMMARY ──
print(f"\n{'='*60}")
print(f"{'Solution':<25} {'Period':>8} {'Trades':>6} {'WR':>6} {'PnL$':>9}")
print(f"{'─'*55}")
for name, d1, d2 in [
    ('1. Reverse opt', s1_prev, s1_cur),
    ('2. Fixed params', s2_prev, s2_cur),
    ('3. BTC filter', s3_prev, s3_cur),
]:
    for label, d in [('PREV',d1),('CUR',d2)]:
        wr=d['w']/d['t']*100 if d['t']>0 else 0
        print(f"{name:<25} {label:>8} {d['t']:>6} {wr:>5.1f}% ${d['pnl']:>+8.0f}")
print('Done')
