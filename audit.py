#!/usr/bin/env python3
"""AUDIT: Verify strategy is real — look-ahead, sample size, random baseline"""
import json, os, numpy as np, pandas as pd, gc
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN=48

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def load(path):
    with open(path) as f: d=json.load(f)
    return (np.array(d['c'],float), np.array(d['h'],float),
            np.array(d['l'],float), np.array(d['o'],float),
            d.get('ts',[]), len(d['c']))

def trends(c,ts,n):
    try:
        idx=pd.to_datetime(np.array(ts),unit='ms')
        df=pd.DataFrame({'c':c},index=idx)
        c4h=df['c'].resample('4h').last().dropna().values
        e50=ema(c4h,50); e200=ema(c4h,200)
        e50a=np.zeros(n); e200a=np.zeros(n)
        for i in range(n):
            j=i//16
            if j<len(e50): e50a[i]=e50[j]; e200a[i]=e200[j]
        return e50a>e200a
    except: return np.ones(n,bool)

def whale_ssl_entries(c,h,l,o,n, LB,sp,LA=True):
    """LA=True = correct (shift), LA=False = buggy (no shift)"""
    sm=3
    if LA:
        ln=pd.Series(l).shift(1).rolling(LB).min().values
    else:
        ln=pd.Series(l).rolling(LB).min().values  # BUG: look-ahead!
    lc=np.zeros(n)
    for i in range(1,n): lc[i]=abs(l[i]-l[i-1])/l[i]*100
    sc=pd.Series(lc).ewm(span=sm,adjust=False).mean().values
    hc=pd.Series(sc).rolling(LB).max().values
    sr=np.where(l<=ln,(sc+hc*2)/3,0)
    wp=pd.Series(sr).ewm(span=sm,adjust=False).mean().values
    wp_up=wp>np.roll(wp,1)
    sma_h=pd.Series(h).rolling(sp).mean().values
    ssl_c=np.zeros(n,int)
    for i in range(sp,n):
        if h[i-1]>sma_h[i-1]: ssl_c[i]=1
        else: ssl_c[i]=-1
    
    le=np.zeros(n,bool)
    for i in range(200,n):
        if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[max(0,i-2)]*2 and wp[i]>0:
            le[i]=True
    return le

def simulate(le, c, h, l, n, tp, sl, filt):
    le_filt = le & filt[:n]
    if le_filt.sum()<3: return None
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l[i]<=ep*(1-sl/100):
                raw=(c[i]/ep-1)*100-COMM*100
                pnl=max(raw,-sl*MAX_SLIPPAGE-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0 and le_filt[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    wr=w/len(t)*100
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    return {'t':len(t),'wr':wr,'dd':dd,'pnl':eq-CAP,'w':w,'l':len(t)-w}

# ── TEST 1: Look-ahead bias (correct vs buggy) ──
print("═══ TEST 1: Look-ahead audit ═══", flush=True)
for sym in ['ADA','ETH','BTC','RAD','KAITO','OG']:
    p1=f'/data/trading28/data/whale_15m_1y/{sym}.json'
    p2=f'/data/trading28/data/whale_15m_prev/{sym}.json'
    for pname,path in [('CUR',p1),('PREV',p2)]:
        if not os.path.exists(path): continue
        c,h,l,o,ts,n=load(path); t4=trends(c,ts,n)
        le_correct=whale_ssl_entries(c,h,l,o,n,50,10,LA=True)
        le_buggy=whale_ssl_entries(c,h,l,o,n,50,10,LA=False)
        r_c=simulate(le_correct,c,h,l,n,5,2.5,t4)
        r_b=simulate(le_buggy,c,h,l,n,5,2.5,t4)
        if r_c and r_b:
            diff=r_b['pnl']-r_c['pnl']
            print(f"  {sym} {pname}: correct={r_c['t']}t WR{r_c['wr']:.0f}% ${r_c['pnl']:+7.0f} | buggy={r_b['t']}t WR{r_b['wr']:.0f}% ${r_b['pnl']:+7.0f} | Δ=${diff:+7.0f}")
    gc.collect()

# ── TEST 2: Walk-forward 6mo train / 6mo test ──
print("\n═══ TEST 2: Walk-forward 6mo/6mo ═══", flush=True)
coins=sorted(set(f.replace('.json','') for f in os.listdir('/data/trading28/data/whale_15m_prev')
    if f.endswith('.json') and f!='_manifest.json'))

wf_results={'train':{'t':0,'w':0,'pnl':0,'c':0},'test':{'t':0,'w':0,'pnl':0,'c':0}}

for sym in coins[:40]:  # 40 coins for speed
    path=f'/data/trading28/data/whale_15m_prev/{sym}.json'
    if not os.path.exists(path): continue
    c,h,l,o,ts,n=load(path)
    if n<6000: continue
    t4=trends(c,ts,n)
    
    # Split: first 6mo train, last 6mo test
    mid=n//2
    c_tr,h_tr,l_tr,o_tr=c[:mid],h[:mid],l[:mid],o[:mid]
    c_te,h_te,l_te,o_te=c[mid:],h[mid:],l[mid:],o[mid:]
    t4_tr,t4_te=t4[:mid],t4[mid:]
    
    best_eq=0; best_cfg=(50,10,5,2.5)
    for LB in [30,50,70]:
        le_tr=whale_ssl_entries(c_tr,h_tr,l_tr,o_tr,mid,LB,10,LA=True)
        for tp,sl in [(2,1),(3,1.5),(5,2.5)]:
            r=simulate(le_tr,c_tr,h_tr,l_tr,mid,tp,sl,t4_tr)
            if r and r['pnl']>best_eq:
                best_eq=r['pnl']; best_cfg=(LB,10,tp,sl)
    
    LB,sp,tp,sl=best_cfg
    
    # Test on train
    le_tr=whale_ssl_entries(c_tr,h_tr,l_tr,o_tr,mid,LB,sp,LA=True)
    rt=simulate(le_tr,c_tr,h_tr,l_tr,mid,tp,sl,t4_tr)
    if rt: wf_results['train']['t']+=rt['t']; wf_results['train']['w']+=rt['w']; wf_results['train']['pnl']+=rt['pnl']; wf_results['train']['c']+=1
    
    # Test on test
    le_te=whale_ssl_entries(c_te,h_te,l_te,o_te,len(c_te),LB,sp,LA=True)
    re=simulate(le_te,c_te,h_te,l_te,len(c_te),tp,sl,t4_te)
    if re: wf_results['test']['t']+=re['t']; wf_results['test']['w']+=re['w']; wf_results['test']['pnl']+=re['pnl']; wf_results['test']['c']+=1
    gc.collect()

for label,d in [('TRAIN(6mo)',wf_results['train']),('TEST(6mo)',wf_results['test'])]:
    wr=d['w']/d['t']*100 if d['t']>0 else 0
    print(f"  {label}: {d['c']}c {d['t']}t WR={wr:.1f}% ${d['pnl']:+.0f}")

# ── TEST 3: Minimum 15 trades ──
print("\n═══ TEST 3: Min 15 trades filter ═══", flush=True)
s3={'t':0,'w':0,'pnl':0,'c':0}
for sym in coins:
    p1=f'/data/trading28/data/whale_15m_1y/{sym}.json'
    p2=f'/data/trading28/data/whale_15m_prev/{sym}.json'
    for pname,path in [('CUR',p1),('PREV',p2)]:
        if not os.path.exists(path): continue
        c,h,l,o,ts,n=load(path); t4=trends(c,ts,n)
        le=whale_ssl_entries(c,h,l,o,n,50,10,LA=True)
        r=simulate(le,c,h,l,n,5,2.5,t4)
        if r and r['t']>=15:
            s3['t']+=r['t']; s3['w']+=r['w']; s3['pnl']+=r['pnl']; s3['c']+=1
        elif r:
            pass  # skip small samples
    gc.collect()
wr3=s3['w']/s3['t']*100 if s3['t']>0 else 0
print(f"  Fixed LB50/SSL10/TP5/SL2.5, min 15t: {s3['c']}c {s3['t']}t WR={wr3:.1f}% ${s3['pnl']:+.0f}")

# ── TEST 4: Random baseline ──
print("\n═══ TEST 4: Random entry baseline ═══", flush=True)
np.random.seed(42)
s4_correct={'t':0,'w':0,'pnl':0,'c':0}; s4_random={'t':0,'w':0,'pnl':0,'c':0}
for sym in coins[:30]:
    path=f'/data/trading28/data/whale_15m_1y/{sym}.json'
    if not os.path.exists(path): continue
    c,h,l,o,ts,n=load(path); t4=trends(c,ts,n)
    # Whale entry
    le_w=whale_ssl_entries(c,h,l,o,n,50,10,LA=True)
    # Random entry: same density as whale entries
    density=le_w[200:].sum()/(n-200) if n>200 else 0.01
    le_r=np.zeros(n,bool)
    for i in range(200,n):
        if t4[i] and np.random.random()<density: le_r[i]=True
    
    rw=simulate(le_w,c,h,l,n,5,2.5,t4)
    rr=simulate(le_r,c,h,l,n,5,2.5,t4)
    if rw: s4_correct['t']+=rw['t']; s4_correct['w']+=rw['w']; s4_correct['pnl']+=rw['pnl']; s4_correct['c']+=1
    if rr: s4_random['t']+=rr['t']; s4_random['w']+=rr['w']; s4_random['pnl']+=rr['pnl']; s4_random['c']+=1
    gc.collect()

wr_w=s4_correct['w']/s4_correct['t']*100 if s4_correct['t']>0 else 0
wr_r=s4_random['w']/s4_random['t']*100 if s4_random['t']>0 else 0
print(f"  Whale+SSL: {s4_correct['c']}c {s4_correct['t']}t WR={wr_w:.1f}% ${s4_correct['pnl']:+.0f}")
print(f"  Random:    {s4_random['c']}c {s4_random['t']}t WR={wr_r:.1f}% ${s4_random['pnl']:+.0f}")

print("\n═══ VERDICT ═══")
print("If Random ≈ Whale → no edge")
print("If buggy >> correct → look-ahead inflation")
print("If TEST << TRAIN in WF → overfitting")
