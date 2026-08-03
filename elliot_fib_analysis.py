#!/usr/bin/env python3
"""Elliot 5-Wave Fib Extension Analysis"""
import json, os, sys, numpy as np
sys.path.insert(0,'/data/trading28')
from strategies.zigzag import zigzag

DATA='/data/trading28/data/3m_4months'
with open('/data/trading28/config/shariah_coins.json') as f: sh=json.load(f)
COINS=[c for c in sh['halal']+sh['halal2'] if c not in {'USDT','USDC','BUSD','DAI','TUSD'}]

def find_5waves(pv, direction='down'):
    pats=[]
    for i in range(len(pv)-5):
        p=pv[i:i+6]
        if direction=='down' and [pt[2] for pt in p]!=['H','L','H','L','H','L']: continue
        if direction=='up'   and [pt[2] for pt in p]!=['L','H','L','H','L','H']: continue
        
        a0,a1,a2,a3,a4,a5 = p[0][1],p[1][1],p[2][1],p[3][1],p[4][1],p[5][1]
        if direction=='down':
            w1=a0-a1; w2=a2-a1; w3=a2-a3; w4=a4-a3; w5=a4-a5
            ok = a3<a1 and a5<a3  # lower lows
        else:
            w1=a1-a0; w2=a1-a2; w3=a3-a2; w4=a3-a4; w5=a5-a4
            ok = a3>a1 and a5>a3  # higher highs
        
        if w1<=0 or w2<=0 or w3<=0 or w4<=0 or w5<=0: continue
        if w2>=w1 or w3<=min(w1,w5): continue
        if direction=='down' and a4>=a1: continue  # wave4 < wave1
        if direction=='up'   and a4<=a1: continue  # wave4 > wave1
        if not ok: continue
        
        pats.append((p[0],p[1],p[2],p[3],p[4],p[5]))
    return pats

# Fib levels to track
FIB_LEVELS = [0.382, 0.50, 0.618, 0.786, 1.0, 1.272, 1.618]
FIB_TOL = 0.03  # 3% tolerance

results = {lev: {'reached':0, 'reversed':0, 'bars':[]} for lev in FIB_LEVELS}
total_down = total_up = 0
never_reached = 0

for ci,cn in enumerate(COINS):
    fp=f'{DATA}/{cn}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw=json.load(f)
    if len(raw)<200: continue
    c=[r['c'] for r in raw]; h=[r['h'] for r in raw]; l=[r['l'] for r in raw]
    n=len(c)
    pv=zigzag(h,l,10,1.0)
    if len(pv)<6: continue
    
    for direction in ['down','up']:
        pats=find_5waves(pv,direction)
        for H1,L1,H2,L2,H3,L3 in pats:
            if direction=='down':
                total_down+=1; start_p=H1[1]; end_p=L3[1]; start_bar=H1[0]; end_bar=L3[0]
            else:
                total_up+=1; start_p=L1[1]; end_p=H3[1]; start_bar=L1[0]; end_bar=H3[0]
            
            wave_range = abs(end_p - start_p)
            if wave_range == 0: continue
            
            # Track price and see which fib levels are hit
            conf_bar = end_bar + 5
            if conf_bar >= n-10: continue
            
            reached_levels = set()
            for j in range(conf_bar, min(n, conf_bar+240)):
                cur_price = c[j]
                
                for lev in FIB_LEVELS:
                    if lev in reached_levels: continue
                    
                    if direction=='down':
                        target = end_p + wave_range * lev
                        hit = cur_price >= target * (1-FIB_TOL)
                    else:
                        target = end_p - wave_range * lev
                        hit = cur_price <= target * (1+FIB_TOL)
                    
                    if hit:
                        reached_levels.add(lev)
                        results[lev]['reached'] += 1
                        results[lev]['bars'].append(j - conf_bar)
                
                if len(reached_levels) == len(FIB_LEVELS):
                    break  # all levels reached
            
            if not reached_levels:
                never_reached += 1

print(f'''
═══ Elliot 5-Wave Fib Extension Analysis ═══
نماذج هابطة: {total_down} | صاعدة: {total_up} | الإجمالي: {total_down+total_up}
لم يصل أي فيبو: {never_reached}

فibo  |  وصل  |  %  | متوسط الشمعات
''')
for lev in FIB_LEVELS:
    r=results[lev]
    pct=r['reached']/(total_down+total_up)*100
    avg=np.mean(r['bars']) if r['bars'] else 0
    print(f"  {lev:.3f}  | {r['reached']:>5} | {pct:>4.1f}% | {avg:.0f}b")
