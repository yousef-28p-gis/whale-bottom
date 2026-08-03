#!/usr/bin/env python3
"""Test Ichimoku 8h Ultra 3/9/18 on all 3 years"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN=2

D2023='/data/trading28/data/whale_15m_2023'
DP='/data/trading28/data/whale_15m_prev'
DC='/data/trading28/data/whale_15m_1y'

c2023=set(f.replace('.json','') for f in os.listdir(D2023) if f.endswith('.json') and f!='_manifest.json')
cp=set(f.replace('.json','') for f in os.listdir(DP) if f.endswith('.json') and f!='_manifest.json')
cc=set(f.replace('.json','') for f in os.listdir(DC) if f.endswith('.json') and f!='_manifest.json')
common=sorted(c2023 & cp & cc)
print(f"Coins in all 3 periods: {len(common)}")

def load(d,s):
    p=os.path.join(d,f'{s}.json')
    if not os.path.exists(p): return None
    with open(p) as f: j=json.load(f)
    return (np.array(j['c'],float),np.array(j['h'],float),np.array(j['l'],float),np.array(j['o'],float),j.get('ts',[]))

def resample_8h(c,h,l,o,ts):
    try:
        idx=pd.to_datetime(np.array(ts),unit='ms')
        df=pd.DataFrame({'o':o,'h':h,'l':l,'c':c},index=idx)
        r=df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
        return r['c'].values,r['h'].values,r['l'].values,r['o'].values
    except: return None

def ichimoku(c,h,l,o,tenkan,kijun,senkou,tp,sl,cooldown):
    n=len(c)
    if n<senkou+30: return None
    h_t=pd.Series(h).rolling(tenkan).max().values; l_t=pd.Series(l).rolling(tenkan).min().values
    t_arr=(h_t+l_t)/2
    h_k=pd.Series(h).rolling(kijun).max().values; l_k=pd.Series(l).rolling(kijun).min().values
    k_arr=(h_k+l_k)/2
    h_s=pd.Series(h).rolling(senkou).max().values; l_s=pd.Series(l).rolling(senkou).min().values
    sb_raw=(h_s+l_s)/2; sa_raw=(t_arr+k_arr)/2
    shift=kijun
    sa=np.full(n,np.nan); sb=np.full(n,np.nan)
    for i in range(shift,n-shift):
        if i+shift<n: sa[i+shift]=sa_raw[i]
    for i in range(senkou,n-shift):
        if i+shift<n: sb[i+shift]=sb_raw[i]
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; side=0
    for i in range(senkou+shift,n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        cloud_top=max(sa[i],sb[i]); cloud_bot=min(sa[i],sb[i])
        above=c[i]>cloud_top; below=c[i]<cloud_bot
        golden=t_arr[i]>k_arr[i] and t_arr[i-1]<=k_arr[i-1]
        death=t_arr[i]<k_arr[i] and t_arr[i-1]>=k_arr[i-1]
        if pos:
            if side==1:
                if h[i]>=ep*(1+tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cooldown
                elif l[i]<=ep*(1-sl/100):
                    pnl=max((c[i]/ep-1)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cooldown
            else:
                if l[i]<=ep*(1-tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cooldown
                elif h[i]>=ep*(1+sl/100):
                    pnl=max((1-c[i]/ep)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cooldown
        if not pos and cool==0:
            if above and golden: pos=1; ep=c[i]; side=1
            elif below and death: pos=1; ep=c[i]; side=-1
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100 if side==1 else (1-c[-1]/ep)*100-COMM*100
        t.append(pnl); eq*=(1+pnl/100)
    if len(t)<3: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

tp,sl=5,2.5; tenkan,kijun,senkou=3,9,18; cool=2
all_data={'2023':[],'PREV':[],'CUR':[]}

for sym in common:
    for period,d,results in [('2023',D2023,all_data['2023']),('PREV',DP,all_data['PREV']),('CUR',DC,all_data['CUR'])]:
        data=load(d,sym)
        if data is None: continue
        c4,h4,l4,o4=resample_8h(data[0],data[1],data[2],data[3],data[4])
        if c4 is None: continue
        r=ichimoku(c4,h4,l4,o4,tenkan,kijun,senkou,tp,sl,cool)
        if r: r['sym']=sym; results.append(r)

total_pnl=0
for label in ['2023','PREV','CUR']:
    res=all_data[label]
    tt=sum(x['t'] for x in res); tw=sum(x['w'] for x in res)
    tl=sum(x['l'] for x in res); tp_=sum(x['pnl'] for x in res)
    wr=tw/tt*100 if tt>0 else 0
    dd=np.mean([x['dd'] for x in res]); gr=sum(1 for x in res if x['pnl']>0)
    theo=sl/(tp+sl)*100
    total_pnl+=tp_
    print(f"{label}: {len(res)}Ⓜ️ {tt}T 🟢{tw} 🔴{tl} WR={wr:.1f}% (random={theo:.1f}%, Δ={wr-theo:+.1f}%) DD={dd:.1f}% ${tp_:+,.0f} g={gr}")

print(f"\n3-YEAR TOTAL: ${total_pnl:+,.0f} on {len(common)} coins | Ichimoku 8h Ultra 3/9/18 TP5/SL2.5")
