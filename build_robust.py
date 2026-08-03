#!/usr/bin/env python3
"""Final: optimize on PREV, save config, test on both periods"""
import json, os, numpy as np, pandas as pd, gc
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN=48

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

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
            return e50a>e200a
    except: pass
    return np.ones(n,bool)

def make_backtest(c,h,l_,o,n,LB,sp,tp,sl,filt):
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
    le=np.zeros(n,bool)
    for i in range(200,n):
        if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[max(0,i-2)]*2 and wp[i]>0 and filt[i]:
            le[i]=True
    if le.sum()<3: return None
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; crash=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l_[i]<=ep*(1-sl/100):
                raw=(c[i]/ep-1)*100-COMM*100
                pnl=max(raw,-sl*MAX_SLIPPAGE-COMM*100)
                if raw<pnl: crash+=1
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=[p for p in t if p>0]
    wr=len(w)/len(t)*100
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    return {'t':len(t),'wr':wr,'dd':dd,'eq':eq,'pnl':eq-CAP,'crash':crash}

# Load coins
coins=sorted(set(f.replace('.json','') for f in os.listdir('/data/trading28/data/whale_15m_prev')
    if f.endswith('.json') and f!='_manifest.json'))

LB_opts=[30,50,70]; SSL_opts=[5,10,20]; TP_SL=[(2,1),(3,1.5),(5,2.5)]
final_configs=[]

print('Optimizing on PREV 2024-2025...', flush=True)
for si, sym in enumerate(coins):
    pp=f'/data/trading28/data/whale_15m_prev/{sym}.json'
    cp=f'/data/trading28/data/whale_15m_1y/{sym}.json'
    if not os.path.exists(pp): continue
    
    with open(pp) as f: d=json.load(f)
    c=np.array(d['c'],float); h=np.array(d['h'],float)
    l_=np.array(d['l'],float); o=np.array(d['o'],float)
    n=len(c); t4=compute_trends(c,d.get('ts',[]),n)
    
    # Crash filter
    skip=False
    for i in range(1,n):
        if abs(c[i]/c[i-1]-1)*100>40: skip=True; break
    if skip: continue
    
    best_eq=0; best_cfg=None
    for LB in LB_opts:
        for sp in SSL_opts:
            for tp,sl in TP_SL:
                r=make_backtest(c,h,l_,o,n,LB,sp,tp,sl,t4)
                if r and r['eq']>best_eq:
                    best_eq=r['eq']; best_cfg=(LB,sp,tp,sl)
    
    if best_cfg is None: continue
    LB,sp,tp,sl=best_cfg
    r = make_backtest(c,h,l_,o,n,LB,sp,tp,sl,t4)
    if r is None: continue
    
    # Test on CUR for validation
    cur_t=cur_wr=0; cur_pnl=0; cur_dd=0
    if os.path.exists(cp):
        with open(cp) as f: d2=json.load(f)
        c2=np.array(d2['c'],float); h2=np.array(d2['h'],float)
        l2=np.array(d2['l'],float); o2=np.array(d2['o'],float)
        n2=len(c2); t4_2=compute_trends(c2,d2.get('ts',[]),n2)
        r2=make_backtest(c2,h2,l2,o2,n2,LB,sp,tp,sl,t4_2)
        if r2: cur_t=r2['t']; cur_wr=r2['wr']; cur_pnl=r2['pnl']; cur_dd=r2['dd']
    
    green_pct=t4.sum()/n*100
    final_configs.append({'sym':sym,'LB':int(LB),'ssl':int(sp),'tp':float(tp),'sl':float(sl),
        'prev_t':r['t'],'prev_wr':r['wr'],'prev_dd':r['dd'],'prev_pnl':r['pnl'],
        'cur_t':cur_t,'cur_wr':cur_wr,'cur_dd':cur_dd,'cur_pnl':cur_pnl,
        'green':green_pct})
    
    if (si+1)%20==0: print(f'  {si+1}/{len(coins)}...', flush=True)
    gc.collect()

# Filter: positive on BOTH periods
passed=[c for c in final_configs if c['prev_pnl']>0 and c['cur_pnl']>0]
both_green=[c for c in final_configs if c['prev_pnl']>0]  # profit on prev (training)

print(f'\nTotal: {len(final_configs)} | Prev+Cur+ : {len(passed)} | Prev+: {len(both_green)}')

# Save
with open('/data/trading28/final_robust_config.json','w') as f:
    json.dump(final_configs, f)

# Summary
print(f'\n═══ ROBUST CONFIG (optimized on PREV) ═══')
total_prev=sum(c['prev_pnl'] for c in final_configs)
total_cur=sum(c['cur_pnl'] for c in final_configs)
avg_wr_prev=np.mean([c['prev_wr'] for c in final_configs])
avg_wr_cur=np.mean([c['cur_wr'] for c in final_configs if c['cur_t']>0])
print(f'Coins: {len(final_configs)} | PREV: WR={avg_wr_prev:.1f}% +${total_prev:.0f} | CUR: WR={avg_wr_cur:.1f}% +${total_cur:.0f}')

# Both-green only
total_both=sum(c['prev_pnl']+c['cur_pnl'] for c in passed)
print(f'Both profitable: {len(passed)} coins | Combined: +${total_both:.0f}')

# Top 10 by combined PnL
top=sorted(final_configs, key=lambda x:-(x['prev_pnl']+x['cur_pnl']))[:10]
print(f'\nTop 10 (combined):')
for c in top:
    print(f"  {c['sym']:<10} LB{c['LB']}/SSL{c['ssl']} TP{c['tp']}/SL{c['sl']}  PREV:{c['prev_t']}t WR{c['prev_wr']:.0f}% +${c['prev_pnl']:.0f}  CUR:{c['cur_t']}t WR{c['cur_wr']:.0f}% +${c['cur_pnl']:.0f}")

print(f'\nSaved: final_robust_config.json')
