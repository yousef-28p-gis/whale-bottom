#!/usr/bin/env python3
"""
لكل عملة استراتيجيتها — تحسين فردي
Whale / SSL / Whale+SSL — بارامترات مختلفة — TP/SL مختلف
"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

COMM = 0.002; CAP = 1000
DATA = '/data/trading28/data/whale_15m_1y'

def load(sym):
    p = os.path.join(DATA, f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d = json.load(f)
    return {
        'ts': pd.to_datetime(d['ts'], unit='ms'),
        'c': np.array(d['c'],float), 'h': np.array(d['h'],float),
        'l': np.array(d['l'],float), 'o': np.array(d['o'],float),
    }

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def whale_signal(l_, n, LB=50, smooth=3):
    ln = pd.Series(l_).shift(1).rolling(LB).min().values
    lc = np.zeros(n)
    for i in range(1,n): lc[i] = abs(l_[i]-l_[i-1])/l_[i]*100
    sc = pd.Series(lc).ewm(span=smooth,adjust=False).mean().values
    hc = pd.Series(sc).rolling(LB).max().values
    sr = np.where(l_<=ln, (sc+hc*2)/3, 0)
    wp = pd.Series(sr).ewm(span=smooth,adjust=False).mean().values
    return wp, wp>np.roll(wp,1)

def ssl_lines(h, l_, n, period=10):
    sma_h = pd.Series(h).rolling(period).mean().values
    sma_l = pd.Series(l_).rolling(period).mean().values
    ssl_c = np.zeros(n, int)
    for i in range(period, n):
        if h[i-1]>sma_h[i-1]: ssl_c[i]=1
        else: ssl_c[i]=-1
    return ssl_c

def sim(le, c, h, l_, n, tp, sl, cd=12):
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cd
            elif l_[i]<=ep*(1-sl/100):
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cd
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    return t,cv,eq

def score_metric(tr, cv, eq):
    """Score = combines WR, profit, stability"""
    if len(tr)<5: return -999
    w=[p for p in tr if p>0]; lo=[p for p in tr if p<=0]
    wr=len(w)/len(tr)*100
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    profit=eq-CAP
    return profit + wr*2 - abs(dd)*5  # higher=better

def test_coin(sym):
    d = load(sym)
    if d is None or len(d['c'])<2000: return None
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; n=len(c)
    
    best_result = None
    best_score = -999
    all_results = []
    
    # Configs to test
    whale_configs = [(30,3),(50,3),(70,3)]
    ssl_configs = [5,10,20]
    tp_sl_pairs = [(2.0,1.0),(3.0,1.5),(5.0,2.5)]
    
    for LB, sm in whale_configs:
        wp, wp_up = whale_signal(l_, n, LB=LB, smooth=sm)
        
        for sp in ssl_configs:
            ssl_c = ssl_lines(h, l_, n, period=sp)
            
            # Strategy 1: Whale only
            le_w = np.zeros(n, bool)
            for i in range(200,n):
                if wp_up[i] and wp[i]>wp[i-2]*1.5 and wp[i]>0:
                    le_w[i]=True
            
            # Strategy 2: SSL only
            le_s = np.zeros(n, bool)
            for i in range(200,n):
                if ssl_c[i]==1 and c[i]>c[i-1] and c[i]>o[i]:
                    le_s[i]=True
            
            # Strategy 3: Whale + SSL
            le_ws = np.zeros(n, bool)
            for i in range(200,n):
                if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0:
                    le_ws[i]=True
            
            for tp, sl in tp_sl_pairs:
                for name, le in [('Whale',le_w), ('SSL',le_s), ('W+SSL',le_ws)]:
                    if le.sum()<5: continue
                    tr,cv,eq = sim(le, c, h, l_, n, tp, sl)
                    score = score_metric(tr, cv, eq)
                    
                    w=[p for p in tr if p>0]; lo=[p for p in tr if p<=0]
                    wr=len(w)/len(tr)*100 if tr else 0
                    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min() if tr else 0
                    
                    result = {
                        'name':name, 'LB':LB, 'sm':sm, 'ssl':sp,
                        'tp':tp, 'sl':sl, 't':len(tr), 'wr':wr,
                        'dd':dd, 'eq':eq, 'sigs':le.sum(),
                        'w':len(w), 'l':len(lo), 'score':score
                    }
                    all_results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
    
    if best_result is None: return None
    return {'sym':sym, 'best':best_result, 'all':all_results, 'n':n}

# ── Run ──
print('🔄 تحسين فردي لكل عملة — 198 عملة...')
coins = sorted([f.replace('.json','') for f in os.listdir(DATA) if f.endswith('.json') and f!='_manifest.json'])

all_data = []
for i, sym in enumerate(coins):
    r = test_coin(sym)
    if r: all_data.append(r)
    if (i+1)%30 == 0: print(f'  {i+1}/{len(coins)}...')

print(f'\n✅ {len(all_data)} عملة\n')

# ── Summary ──
profitable = [d for d in all_data if d['best']['eq'] > CAP]
unprofitable = [d for d in all_data if d['best']['eq'] <= CAP]

print(f'{"="*70}')
print(f'📊 النتائج النهائية — تحسين فردي لكل عملة')
print(f'{"="*70}')
print(f'عملات مختبرة: {len(all_data)}')
print(f'ربحانة: {len(profitable)} ({len(profitable)/max(1,len(all_data))*100:.0f}%)')
print(f'خسرانة: {len(unprofitable)}')

if profitable:
    avg_profit = np.mean([d['best']['eq']-CAP for d in profitable])
    avg_wr = np.mean([d['best']['wr'] for d in profitable])
    total_profit = sum(d['best']['eq']-CAP for d in profitable)
    print(f'\n🟢 الربحانة:')
    print(f'   متوسط ربح: +${avg_profit:.1f}')
    print(f'   متوسط WR: {avg_wr:.1f}%')
    print(f'   إجمالي ربح: +${total_profit:.0f}')
    
    # Strategy distribution
    strats = defaultdict(int)
    for d in profitable: strats[d['best']['name']] += 1
    print(f'   توزيع الاستراتيجيات: {dict(strats)}')
    
    # TP/SL distribution  
    tp_sl_dist = defaultdict(int)
    for d in profitable: tp_sl_dist[(d['best']['tp'], d['best']['sl'])] += 1
    print(f'   توزيع TP/SL: {dict(tp_sl_dist)}')

if unprofitable:
    avg_loss = np.mean([d['best']['eq']-CAP for d in unprofitable])
    total_loss = sum(d['best']['eq']-CAP for d in unprofitable)
    print(f'\n🔴 الخسرانة:')
    print(f'   متوسط خسارة: -${abs(avg_loss):.1f}')
    print(f'   إجمالي خسارة: -${abs(total_loss):.0f}')

# ── Top 20 ──
print(f'\n{"="*70}')
print(f'🏆 أفضل 20 عملة (تحسين فردي):')
print(f'{"عملة":<10} {"استراتيجية":<8} {"LB/SSL":>8} {"TP/SL":>8} {"صفقات":>5} {"WR":>6} {"سحب":>6} {"ربح":>8}')
print('-'*65)

best_coins = sorted(all_data, key=lambda x: -x['best']['eq'])[:20]
for d in best_coins:
    b = d['best']
    params = f'{b["LB"]}/{b["ssl"]}' if b['name']=='W+SSL' else f'{b["LB"]}' if 'Whale' in b['name'] else f'{b["ssl"]}'
    ico = '+' if b['eq']>CAP else '-'
    print(f'{d["sym"]:<10} {b["name"]:<8} {params:>8} {b["tp"]:.0f}/{b["sl"]:.1f}  {b["t"]:>5} {b["wr"]:>5.1f}% {b["dd"]:>5.1f}% {ico}${b["eq"]-CAP:>+7.1f}')

# Save profitable coins to file
with open('/data/trading28/per_coin_configs.json','w') as f:
    out = [{'sym':d['sym'], **{k:v for k,v in d['best'].items() if k!='trades'}} for d in profitable]
    json.dump(out, f, indent=2)
print(f'\n💾 حفظت إعدادات {len(profitable)} عملة ربحانة → per_coin_configs.json')

print('\n✅ Done')
