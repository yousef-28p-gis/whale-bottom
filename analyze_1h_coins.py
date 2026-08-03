"""Analyze why only 27 coins with +1h filter"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; DATA='data/whale_15m_1y'

def load(sym):
    with open(os.path.join(DATA, f'{sym}.json')) as f: d=json.load(f)
    return {'c':np.array(d['c'],float),'h':np.array(d['h'],float),
            'l':np.array(d['l'],float),'o':np.array(d['o'],float),
            'ts':pd.to_datetime(d['ts'],unit='ms')}

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

coins=sorted([f.replace('.json','') for f in os.listdir(DATA) if f.endswith('.json') and f!='_manifest.json'])
stats = []
for sym in coins:
    d=load(sym)
    if d is None or len(d['c'])<2000: continue
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; n=len(c); idx=d['ts']
    skip=False
    for i in range(1,n):
        if abs(c[i]/c[i-1]-1)*100>40: skip=True; break
    if skip: continue
    try:
        df=pd.DataFrame({'c':c},index=idx)
        c4h=df['c'].resample('4h').last().dropna().values
        e50_4h=ema(c4h,50); e200_4h=ema(c4h,200)
        e50_a=np.zeros(n); e200_a=np.zeros(n)
        for i in range(n):
            j=i//16
            if j<len(e50_4h): e50_a[i]=e50_4h[j]; e200_a[i]=e200_4h[j]
        t4=e50_a>e200_a
        c1h=df['c'].resample('1h').last().dropna().values
        e20_1h=ema(c1h,20); e50_1h=ema(c1h,50)
        e20_1h_a=np.zeros(n); e50_1h_a=np.zeros(n)
        for i in range(n):
            j=i//4
            if j<len(e20_1h): e20_1h_a[i]=e20_1h[j]; e50_1h_a[i]=e50_1h[j]
        t1=e20_1h_a>e50_1h_a
    except:
        continue
    green_4h=t4.sum()/n*100; green_1h=t1.sum()/n*100; both_green=(t4&t1).sum()/n*100
    LB=50; sm=3; sp=10
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
    cnt_4h=sum(1 for i in range(200,n) if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0 and t4[i])
    cnt_1h=sum(1 for i in range(200,n) if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0 and t4[i] and t1[i])
    stats.append({'sym':sym,'g4':green_4h,'g1':green_1h,'gb':both_green,'e4':cnt_4h,'e1':cnt_1h})

print(f'Total coins: {len(stats)}')
print(f'≥5 entries 4h: {sum(1 for s in stats if s["e4"]>=5)}')
print(f'≥5 entries +1h: {sum(1 for s in stats if s["e1"]>=5)}')
print(f'≥3 entries +1h: {sum(1 for s in stats if s["e1"]>=3)}')
print(f'≥1 entry  +1h: {sum(1 for s in stats if s["e1"]>=1)}')
print(f'Median e4h: {np.median([s["e4"] for s in stats]):.0f}')
print(f'Median e1h: {np.median([s["e1"] for s in stats]):.0f}')
print(f'Avg green%: 4h={np.mean([s["g4"] for s in stats]):.0f}% 1h={np.mean([s["g1"] for s in stats]):.0f}%')
print(f'\nTop 15 by +1h entries:')
for s in sorted(stats, key=lambda x:-x['e1'])[:15]:
    print(f'  {s["sym"]:<12} 4h={s["g4"]:.0f}% 1h={s["g1"]:.0f}% e4h={s["e4"]} e1h={s["e1"]}')
