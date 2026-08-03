#!/usr/bin/env python3
"""
Ribbon فقط — 198 عملة × سنة — بيانات whale المحملة
"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

COMM = 0.002; CAP = 1000; DATA = '/data/trading28/data/whale_15m_1y'

def load(sym):
    p = os.path.join(DATA, f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d = json.load(f)
    return {
        'ts': pd.to_datetime(d['ts'], unit='ms'),
        'c': np.array(d['c'],float), 'h': np.array(d['h'],float),
        'l': np.array(d['l'],float), 'o': np.array(d['o'],float),
        'v': np.array(d['v'],float),
    }

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def sim(le, c, h, l_, n, tp, sl, cd=12):
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                t.append(tp-COMM*100); eq*=(1+(tp-COMM*100)/100); pos=0; cool=cd
            elif l_[i]<=ep*(1-sl/100):
                t.append(-sl-COMM*100); eq*=(1+(-sl-COMM*100)/100); pos=0; cool=cd
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); cv.append(eq)
    return t,cv,eq

def met(le, c, h, l_, n, tp, sl):
    tr,cv,eq = sim(le,c,h,l_,n,tp,sl)
    if len(tr)<5: return None
    w=[p for p in tr if p>0]; l=[p for p in tr if p<=0]
    wr=len(w)/len(tr)*100
    aw=np.mean(w) if w else 0; al=abs(np.mean(l)) if l else 0
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    return {'t':len(tr),'wr':wr,'aw':aw,'al':al,'dd':dd,'eq':eq,'sigs':le.sum()}

def test(sym):
    d=load(sym)
    if d is None or len(d['c'])<500: return None
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; n=len(c)
    
    e5=ema(c,5); e9=ema(c,9); e13=ema(c,13)
    e21=ema(c,21); e34=ema(c,34); e50=ema(c,50); e200=ema(c,200)
    
    # ── Ribbon alignment: price > e5 > e9 > e13 > e21 > e34 > e50 ──
    le = np.zeros(n, bool)
    for i in range(200,n):
        if (c[i] > e5[i] > e9[i] > e13[i] > e21[i] > e34[i] > e50[i] and
            not (c[i-1] > e5[i-1] > e9[i-1] > e13[i-1] > e21[i-1] > e34[i-1] > e50[i-1])):
            le[i] = True
    
    # 4h filter
    try:
        df4 = pd.DataFrame({'c':c}, index=d['ts']).resample('4h').last().dropna()
        c4 = df4['c'].values; e50_4=ema(c4,50); e200_4=ema(c4,200)
        mtf4 = np.zeros(n, bool)
        for i in range(200,n):
            j = i//16
            if j>=len(e50_4) or j>=len(e200_4): continue
            if np.isnan(e50_4[j]) or np.isnan(e200_4[j]): continue
            if e50_4[j] > e200_4[j]: mtf4[i] = True
    except:
        mtf4 = np.ones(n,bool)
    
    # 15m trend filter: e50 > e200 on 15m itself
    mtf15 = np.zeros(n,bool)
    for i in range(200,n):
        if not np.isnan(e50[i]) and not np.isnan(e200[i]) and e50[i] > e200[i]:
            mtf15[i] = True
    
    # Filters to test
    filters = {
        'بدون فلتر': np.ones(n,bool),
        '4h↑': mtf4,
        '15m↑': mtf15,
        '4h+15m↑': mtf4 & mtf15,
    }
    
    # TP/SL combos: small AND larger
    tp_sl = [
        (0.8,0.4),(1.0,0.5),(1.5,0.75),(2.0,1.0),
        (3.0,1.5),(4.0,2.0),(5.0,2.5),
    ]
    
    results = []
    for fname, fmask in filters.items():
        le_f = le & fmask
        for tp,sl in tp_sl:
            m = met(le_f, c, h, l_, n, tp, sl)
            if m: m['filter']=fname; m['tp']=tp; m['sl']=sl; results.append(m)
    
    return {'sym':sym,'res':results,'n':n,'green_15m':mtf15.sum()/n*100,'green_4h':mtf4.sum()/n*100}

# ── RUN ──
print('🔄 Ribbon solo — 198 عملة...')
coins = sorted([f.replace('.json','') for f in os.listdir(DATA) if f.endswith('.json') and f!='_manifest.json'])
all_data=[]
for i,sym in enumerate(coins):
    r=test(sym)
    if r: all_data.append(r)
    if (i+1)%40==0: print(f'  {i+1}/{len(coins)}...')
print(f'\n✅ {len(all_data)} عملة\n')

# ── AGGREGATE BY FILTER & TP/SL ──
for fname in ['4h↑','15m↑','4h+15m↑','بدون فلتر']:
    print(f'{"="*80}')
    print(f'🔍 فلتر: {fname}')
    print(f'{"TP/SL":>10} {"عملات":>5} {"صفقات":>7} {"WR":>6} {"R:R":>5} {"سحب":>6} {"محفظة":>10} {"✅ربح":>6}')
    print('-'*60)
    
    by_tp = defaultdict(list)
    for coin in all_data:
        for r in coin['res']:
            if r['filter']==fname:
                by_tp[(r['tp'],r['sl'])].append({**r,'sym':coin['sym']})
    
    for (tp,sl),items in sorted(by_tp.items(), key=lambda x: -sum(i['eq'] for i in x[1])):
        tt=sum(i['t'] for i in items)
        if tt<10: continue
        eqs=sum(i['eq'] for i in items)
        awr=np.mean([i['wr'] for i in items])
        add=np.mean([i['dd'] for i in items])
        arr=np.mean([i['aw']/(i['al']+0.001) for i in items])
        win=sum(1 for i in items if i['eq']>CAP)
        ico='+' if eqs>CAP*len(items) else '-'
        pr=eqs-CAP*len(items)
        print(f'{tp:.1f}%/{sl:.1f}%  {len(items):>4} {tt:>7} {awr:>5.1f}% {arr:>4.2f}x {add:>5.1f}% {ico}${pr:>+9.0f} {win:>5}')

# ── TOP COINS ──
print(f'\n{"="*80}')
print(f'🏆 أفضل 30 عملة (Ribbon + 4h↑ + TP5/SL2.5):')
best=[]
for coin in all_data:
    for r in coin['res']:
        if r['filter']=='4h↑' and r['tp']==5.0 and r['sl']==2.5 and r['eq']>CAP:
            best.append({**r,'sym':coin['sym'],'green_15m':coin['green_15m'],'green_4h':coin['green_4h']})
best.sort(key=lambda x:-x['eq'])
for r in best[:30]:
    print(f'{r["sym"]:<12} G15m{r["green_15m"]:>5.0f}% G4h{r["green_4h"]:>5.0f}% | {r["t"]:>4d}t WR{r["wr"]:>5.1f}% DD{r["dd"]:>5.1f}% +${r["eq"]-CAP:>+7.1f}')

# ── BEST PER FILTER ──
print(f'\n📊 ملخص كل فلتر — TP5/SL2.5:')
for fname in ['بدون فلتر','4h↑','15m↑','4h+15m↑']:
    items=[]
    for coin in all_data:
        for r in coin['res']:
            if r['filter']==fname and r['tp']==5.0 and r['sl']==2.5:
                items.append({**r,'sym':coin['sym']})
    if not items: continue
    tt=sum(i['t'] for i in items); eqs=sum(i['eq'] for i in items)
    awr=np.mean([i['wr'] for i in items])
    win=sum(1 for i in items if i['eq']>CAP)
    pr=eqs-CAP*len(items)
    ico='✅' if pr>0 else '❌'
    print(f'{fname:<15} {len(items):>3} ع | {tt:>5}t | WR{awr:>5.1f}% | {win}✅ | {ico} total ${pr:>+.0f}')

print('\n✅ تم')
