"""Optimized: High WR with trend filters — 30 coins"""
import json, os, numpy as np, pandas as pd
COMM, DATA = 0.002, 'data/whale_15m_1y'

def load(sym):
    p = os.path.join(DATA, f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d = json.load(f)
    return {'c': np.array(d['c'],float), 'h': np.array(d['h'],float),
            'l': np.array(d['l'],float), 'o': np.array(d['o'],float)}

coins = sorted([f.replace('.json','') for f in os.listdir(DATA) 
                if f.endswith('.json') and f!='_manifest.json'])[:30]

TP_SL_TESTS = [(0.5,3.0),(0.75,3.0),(0.5,2.0),(0.75,2.5),(0.5,2.5),(1.0,3.0)]

def sim_many(entries, c, h, l_, n, tp, sl):
    wins=0; losses=0; pnl=0.0
    for ei in entries:
        ep=c[ei]; end=min(ei+48,n)
        tp_idx=-1; sl_idx=-1
        for j in range(ei+1,end):
            if tp_idx<0 and h[j]>=ep*(1+tp/100): tp_idx=j
            if sl_idx<0 and l_[j]<=ep*(1-sl/100): sl_idx=j
            if tp_idx>=0 and sl_idx>=0: break
        if tp_idx>=0 and sl_idx<0: wins+=1; pnl+=tp-COMM*100
        elif sl_idx>=0 and tp_idx<0: losses+=1; pnl+=-sl-COMM*100
        else: pnl+=(c[end-1]/ep-1)*100-COMM*100
    return len(entries),wins,losses,pnl

results=[]
for fname in ['NoFilter','4h','1h','15m','4h+1h']:
    for tp,sl in TP_SL_TESTS:
        tr=0; tw=0; tl=0; tp_pnl=0.0; cc=0
        for sym in coins:
            d=load(sym)
            if d is None or len(d['c'])<500: continue
            c,h,l_,o=d['c'],d['h'],d['l'],d['o']; n=len(c)
            # Whale+SSL
            LB,sm=50,3
            ln=pd.Series(l_).rolling(LB).min().values
            lc=np.zeros(n)
            for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
            sc=pd.Series(lc).ewm(span=sm,adjust=False).mean().values
            hc=pd.Series(sc).rolling(LB).max().values
            strength=np.where(l_<=ln,(sc+hc*2)/3,0)
            wp=pd.Series(strength).ewm(span=sm,adjust=False).mean().values
            wp_up=wp>np.roll(wp,1)
            sup=pd.Series(h).rolling(10).mean().values
            # Trend
            t_4h=pd.Series(c).ewm(span=800,adjust=False).mean().values>pd.Series(c).ewm(span=3200,adjust=False).mean().values
            t_1h=pd.Series(c).ewm(span=80,adjust=False).mean().values>pd.Series(c).ewm(span=200,adjust=False).mean().values
            t_15=pd.Series(c).ewm(span=20,adjust=False).mean().values>pd.Series(c).ewm(span=50,adjust=False).mean().values
            if fname=='NoFilter': filt=np.ones(n,bool)
            elif fname=='4h': filt=t_4h
            elif fname=='1h': filt=t_1h
            elif fname=='15m': filt=t_15
            elif fname=='4h+1h': filt=t_4h & t_1h
            else: continue
            entries=[i for i in range(300,n) if wp_up[i] and c[i]>sup[i] and c[i]>o[i] and filt[i]]
            if len(entries)<3: continue
            cc+=1
            t,w,l,pnl=sim_many(entries,c,h,l_,n,tp,sl)
            tr+=t; tw+=w; tl+=l; tp_pnl+=pnl
        if tr>=10:
            wr=tw/tr*100; avg=tp_pnl/tr
            results.append((fname,tp,sl,tr,wr,tw,tl,tp_pnl,avg,cc))

results.sort(key=lambda x:-x[4])
print(f"{'Filter':<10} {'TP/SL':>10} {'T':>5} {'WR':>7} {'W':>4} {'L':>4} {'Avg$/T':>8} {'PnL$':>9} {'C':>4}")
print("-"*68)
for fn,tp,sl,tr,wr,w,l,eq,avg,cc in results[:25]:
    print(f"{fn:<10} TP{tp}/SL{sl} {tr:>5} {wr:>6.1f}% {w:>4} {l:>4} ${avg:>+7.2f} ${eq:>+8.1f} {cc:>4}")
