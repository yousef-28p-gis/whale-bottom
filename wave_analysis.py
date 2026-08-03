import json, os, sys, numpy as np
sys.path.insert(0,'/data/trading28')
from strategies.zigzag import zigzag

DATA='/data/trading28/data/3m_4months'
with open('/data/trading28/config/shariah_coins.json') as f: sh=json.load(f)
COINS=[c for c in sh['halal']+sh['halal2'] if c not in {'USDT','USDC','BUSD','DAI','TUSD'}]

w5_w1=[]; w5_w3=[]; w3_w1=[]

for cn in COINS:
    fp=f'{DATA}/{cn}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw=json.load(f)
    if len(raw)<200: continue
    h=[r['h'] for r in raw]; l=[r['l'] for r in raw]
    pv=zigzag(h,l,10,1.0)
    if len(pv)<6: continue
    
    for i in range(len(pv)-5):
        p=pv[i:i+6]
        if [pt[2] for pt in p]!=['H','L','H','L','H','L']: continue
        H1=p[0][1];L1=p[1][1];H2=p[2][1];L2=p[3][1];H3=p[4][1];L3=p[5][1]
        w1=H1-L1;w2=H2-L1;w3=H2-L2;w4=H3-L2;w5=H3-L3
        if w1<=0 or w2<=0 or w3<=0 or w4<=0 or w5<=0: continue
        if w2>=w1 or w3<=min(w1,w5): continue
        if H3>=L1: continue
        if L3>=L2: continue
        
        if w1>0: w5_w1.append(w5/w1); w3_w1.append(w3/w1)
        if w3>0: w5_w3.append(w5/w3)

print(f'نماذج: {len(w5_w1)}')
print()
print('W5/W1:')
for p in [10,25,50,75,90]:
    print(f'  P{p}: {np.percentile(w5_w1,p):.2f}')
print(f'  متوسط: {np.mean(w5_w1):.2f}')
for r in [0.382,0.5,0.618,0.786,1.0,1.272,1.618]:
    cnt=sum(1 for x in w5_w1 if abs(x-r)<=0.15)
    print(f'  قرب {r:.3f}: {cnt} ({cnt/len(w5_w1)*100:.1f}%)')

print()
print('W3/W1:')
print(f'  متوسط: {np.mean(w3_w1):.2f}')
for r in [1.0,1.272,1.618,2.0,2.618]:
    cnt=sum(1 for x in w3_w1 if abs(x-r)<=0.2)
    print(f'  قرب {r:.3f}: {cnt} ({cnt/len(w3_w1)*100:.1f}%)')
