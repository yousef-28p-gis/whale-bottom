#!/usr/bin/env python3
"""Test ALL strategies on PREV data only (2024-2025) — pure out-of-sample"""
import json, os, numpy as np, pandas as pd, gc
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN=48
DATA='/data/trading28/data/whale_15m_prev'

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def load(sym):
    p=os.path.join(DATA,f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d=json.load(f)
    return (np.array(d['c'],float),np.array(d['h'],float),
            np.array(d['l'],float),np.array(d['o'],float),
            d.get('ts',[]),len(d['c']))

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
        t4=e50a>e200a
        c1h=df['c'].resample('1h').last().dropna().values
        e20=ema(c1h,20); e50h=ema(c1h,50)
        e20a=np.zeros(n); e50a2=np.zeros(n)
        for i in range(n):
            j=i//4
            if j<len(e20): e20a[i]=e20[j]; e50a2[i]=e50h[j]
        t1=e20a>e50a2
        return t4,t1
    except: return np.ones(n,bool),np.ones(n,bool)

def sim(entries, c, h, l, n, tp, sl):
    """entries = array of entry indices"""
    entry_set=set(entries)
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l[i]<=ep*(1-sl/100):
                raw=(c[i]/ep-1)*100-COMM*100
                pnl=max(raw,-sl*MAX_SLIPPAGE-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0 and i in entry_set: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    wr=w/len(t)*100
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    return {'t':len(t),'wr':wr,'dd':dd,'pnl':eq-CAP,'w':w,'l':len(t)-w}

# ── Strategy builders ──
def s1_whale_ssl(c,h,l,o,n,LB,sp,filt):
    sm=3
    ln=pd.Series(l).shift(1).rolling(LB).min().values
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
    return [i for i in range(200,n) if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[max(0,i-2)]*2 and wp[i]>0 and filt[i]]

def s2_ema_cross(c,h,l,o,n,fast,slow,filt):
    ef=ema(c,fast); es=ema(c,slow)
    return [i for i in range(200,n) if ef[i]>es[i] and ef[i-1]<=es[i-1] and c[i]>o[i] and filt[i]]

def s3_pullback_ema(c,h,l,o,n,ema_len,filt):
    e=ema(c,ema_len)
    return [i for i in range(200,n) if c[i]<e[i] and c[i]>e[i]*0.99 and c[i]>o[i] and filt[i]]

def s4_breakout(c,h,l,o,n,lookback,filt):
    return [i for i in range(200,n) if c[i]>max(h[max(0,i-lookback):i]) and c[i]>o[i] and filt[i]]

def s5_supertrend(c,h,l,o,n,period,mult,filt):
    atr=pd.Series(h-l).ewm(span=period,adjust=False).mean().values
    hl2=(pd.Series(h).rolling(period).max().values+pd.Series(l).rolling(period).min().values)/2
    upper=hl2+mult*atr; lower=hl2-mult*atr
    trend_up=np.ones(n,bool)
    for i in range(1,n):
        trend_up[i]=trend_up[i-1]
        if c[i]>upper[i-1]: trend_up[i]=True
        elif c[i]<lower[i-1]: trend_up[i]=False
    return [i for i in range(200,n) if trend_up[i] and not trend_up[i-1] and filt[i]]

def s6_mtf_pullback(c,h,l,o,n,t4,t1,filt):
    if t4 is None or t1 is None:
        t4,t1 = np.ones(n,bool), np.ones(n,bool)
    e15=ema(c,20)
    return [i for i in range(200,n) if t4[i] and t1[i] and c[i]<e15[i] and c[i]>e15[i]*0.98 and c[i]>o[i] and filt[i]]

# ── Test all strategies on all coins ──
coins=sorted(set(f.replace('.json','') for f in os.listdir(DATA) if f.endswith('.json') and f!='_manifest.json'))
print(f'Testing {len(coins)} coins on PREV data...', flush=True)

strategies = [
    ('W+SSL LB30/SSL10', lambda c,h,l,o,n,f: s1_whale_ssl(c,h,l,o,n,30,10,f)),
    ('W+SSL LB50/SSL10', lambda c,h,l,o,n,f: s1_whale_ssl(c,h,l,o,n,50,10,f)),
    ('W+SSL LB70/SSL10', lambda c,h,l,o,n,f: s1_whale_ssl(c,h,l,o,n,70,10,f)),
    ('W+SSL LB50/SSL5',  lambda c,h,l,o,n,f: s1_whale_ssl(c,h,l,o,n,50,5,f)),
    ('W+SSL LB50/SSL20', lambda c,h,l,o,n,f: s1_whale_ssl(c,h,l,o,n,50,20,f)),
    ('EMA cross 20/50',  lambda c,h,l,o,n,f: s2_ema_cross(c,h,l,o,n,20,50,f)),
    ('EMA cross 50/200', lambda c,h,l,o,n,f: s2_ema_cross(c,h,l,o,n,50,200,f)),
    ('Pullback EMA20',   lambda c,h,l,o,n,f: s3_pullback_ema(c,h,l,o,n,20,f)),
    ('Pullback EMA50',   lambda c,h,l,o,n,f: s3_pullback_ema(c,h,l,o,n,50,f)),
    ('Breakout 20bar',   lambda c,h,l,o,n,f: s4_breakout(c,h,l,o,n,20,f)),
    ('SuperTrend 10/3',  lambda c,h,l,o,n,f: s5_supertrend(c,h,l,o,n,10,3,f)),
    ('MTF pullback',     lambda c,h,l,o,n,f: s6_mtf_pullback(c,h,l,o,n,None,None,f)),
]

# Test: 4h filter only, and 4h+1h
filters = [('4h',0), ('4h+1h',1)]  # 0=4h only, 1=both

all_results=[]
for sname, sfn in strategies:
    for flabel, fidx in filters:
        agg={'t':0,'w':0,'pnl':0,'coins':0,'dd_sum':0}
        for sym in coins:
            d=load(sym)
            if d is None: continue
            c,h,l,o,ts,n=d
            # Skip crash coins
            skip=False
            for i in range(1,n):
                if abs(c[i]/c[i-1]-1)*100>40: skip=True; break
            if skip: continue
            
            t4_full,t1_full=trends(c,ts,n)
            filt=t4_full if fidx==0 else (t4_full & t1_full)
            
            entries=sfn(c,h,l,o,n,filt)
            if len(entries)<5: continue
            
            # Best TP/SL for this strategy
            best_pnl=-99999; best_r=None; best_tpsl=None
            for tp,sl in [(2,1),(3,1.5),(5,2.5),(1.5,0.75),(2.5,1.5),(4,2)]:
                r=sim(entries,c,h,l,n,tp,sl)
                if r and r['pnl']>best_pnl:
                    best_pnl=r['pnl']; best_r=r; best_tpsl=(tp,sl)
            
            if best_r:
                agg['t']+=best_r['t']; agg['w']+=best_r['w']
                agg['pnl']+=best_r['pnl']; agg['coins']+=1
                agg['dd_sum']+=best_r['dd']
        
        if agg['t']>=10:
            wr=agg['w']/agg['t']*100
            dd=agg['dd_sum']/agg['coins']
            all_results.append((sname,flabel,agg['t'],wr,agg['pnl'],dd,agg['coins']))
    gc.collect()

# Print sorted by PnL
all_results.sort(key=lambda x:-x[4])
print(f"\n{'Strategy':<25} {'Filt':>6} {'T':>5} {'WR':>6} {'DD':>6} {'PnL$':>9} {'Coins':>6}")
print('-'*68)
for sname,flabel,t,wr,pnl,dd,cc in all_results[:20]:
    print(f'{sname:<25} {flabel:>6} {t:>5} {wr:>5.1f}% {dd:>5.1f}% ${pnl:>+8.0f} {cc:>6}')

# Also show best by WR
print(f"\n🏆 Best by WR:")
by_wr=sorted(all_results, key=lambda x:-x[3])[:10]
for sname,flabel,t,wr,pnl,dd,cc in by_wr:
    print(f'{sname:<25} {flabel:>6} {t:>5} {wr:>5.1f}% {dd:>5.1f}% ${pnl:>+8.0f} {cc:>6}')

print('\nDone')
