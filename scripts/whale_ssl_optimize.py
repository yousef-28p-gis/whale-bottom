#!/usr/bin/env python3
"""
Whale Pump + SSL — 198 عملة × سنة — تحسين الإعدادات
"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

COMM = 0.002; CAP = 1000; DATA = '/data/trading28/data/whale_15m_1y'

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

def whale_signal(l, n, LB=30, smooth=3):
    """Returns whale_pump array, bool: up"""
    ln = pd.Series(l).rolling(LB).min().values
    at_low = l <= ln
    low_change = np.zeros(n)
    for i in range(1,n): low_change[i] = abs(l[i]-l[i-1])/l[i]*100
    sc = pd.Series(low_change).ewm(span=smooth,adjust=False).mean().values
    hc = pd.Series(sc).rolling(LB).max().values
    strength = np.where(at_low, (sc + hc*2)/3, 0)
    wp = pd.Series(strength).ewm(span=smooth,adjust=False).mean().values
    up = wp > np.roll(wp, 1)
    return wp, up

def ssl_lines(h, l, n, period=10):
    """SSL up/down lines"""
    ssl_up = pd.Series(h).rolling(period).mean().values
    ssl_dn = pd.Series(l).rolling(period).mean().values
    return ssl_up, ssl_dn

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
    w=[p for p in tr if p>0]; lo=[p for p in tr if p<=0]
    wr=len(w)/len(tr)*100
    aw=np.mean(w) if w else 0; al=abs(np.mean(lo)) if lo else 0
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    return {'t':len(tr),'wr':wr,'aw':aw,'al':al,'dd':dd,'eq':eq,'sigs':le.sum()}

def test_coin(sym, whale_LB, whale_smooth, ssl_period):
    d = load(sym)
    if d is None or len(d['c'])<500: return None
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; n=len(c)
    
    # Whale
    wp, wp_up = whale_signal(l_, n, LB=whale_LB, smooth=whale_smooth)
    
    # SSL
    sup, sdn = ssl_lines(h, l_, n, period=ssl_period)
    
    # Entry conditions
    le_whale_only = np.zeros(n, bool)  # whale rising + price > SSL up
    le_ssl_only = np.zeros(n, bool)    # SSL cross only
    le_both = np.zeros(n, bool)        # whale rising AND price > SSL up
    
    for i in range(200, n):
        # Whale + SSL: whale rising + price above SSL up
        if wp_up[i] and wp[i] > wp[i-2]*2 and c[i] > sup[i]:
            le_both[i] = True
        
        # Whale only rising
        if wp_up[i] and wp[i] > wp[i-2]*1.5 and c[i] > o[i]:
            le_whale_only[i] = True
    
    # SSL cross above
    for i in range(200, n):
        if c[i] > sup[i] and c[i-1] <= sup[i-1] and c[i] > o[i]:
            le_ssl_only[i] = True
    
    results = []
    for tp, sl in [(1.0,0.5),(1.5,0.75),(2.0,1.0),(3.0,1.5),(5.0,2.5)]:
        m = met(le_both, c, h, l_, n, tp, sl)
        if m: m['entry']='Whale+SSL'; m['tp']=tp; m['sl']=sl; results.append(m)
        
        m = met(le_whale_only, c, h, l_, n, tp, sl)
        if m: m['entry']='Whale فقط'; m['tp']=tp; m['sl']=sl; results.append(m)
        
        m = met(le_ssl_only, c, h, l_, n, tp, sl)
        if m: m['entry']='SSL فقط'; m['tp']=tp; m['sl']=sl; results.append(m)
    
    if not results: return None
    
    # Find best result per entry type
    best = {}
    for etype in ['Whale+SSL','Whale فقط','SSL فقط']:
        er = [r for r in results if r['entry']==etype]
        if er: best[etype] = max(er, key=lambda x: x['eq'])
    
    return {'sym':sym, 'best':best, 'all':results}

# ── Configs ──
configs = [
    (30, 3, 10, 'LB30/E3/SSL10'),    # default
    (20, 3, 10, 'LB20/E3/SSL10'),    # faster whale
    (50, 3, 10, 'LB50/E3/SSL10'),    # slower whale
    (30, 2, 10, 'LB30/E2/SSL10'),    # less smooth
    (30, 5, 10, 'LB30/E5/SSL10'),    # more smooth
    (30, 3, 5,  'LB30/E3/SSL5'),     # faster SSL
    (30, 3, 20, 'LB30/E3/SSL20'),    # slower SSL
    (20, 2, 5,  'LB20/E2/SSL5'),     # aggressive
]

print('🔄 Whale+SSL — 198 عملة...')
coins = sorted([f.replace('.json','') for f in os.listdir(DATA) if f.endswith('.json') and f!='_manifest.json'])

# Test first 10 coins with all configs to find best
SAMPLE = coins[:20]  # 20 coins for config sweep
print(f'   Config sweep on {len(SAMPLE)} coins...')

config_results = {}
for LB, sm, ssl, label in configs:
    data = []
    for sym in SAMPLE:
        r = test_coin(sym, LB, sm, ssl)
        if r: data.append(r)
    
    # Aggregate
    agg = {'Whale+SSL': [], 'Whale فقط': [], 'SSL فقط': []}
    for d in data:
        for etype, b in d['best'].items():
            agg[etype].append(b)
    
    config_results[label] = agg
    print(f'   {label}: Whale+SSL={len(agg["Whale+SSL"])} coins, '
          f'avg eq ${np.mean([b["eq"] for b in agg["Whale+SSL"]]):.0f}' if agg['Whale+SSL'] else 'no results')

# ── Best config summary ──
print(f'\n{"="*80}')
print(f'🏆 أفضل إعدادات (20 عملة):')
print(f'{"الإعدادات":<20} {"استراتيجية":<12} {"عملات":>5} {"متوسط EQ":>8} {"متوسط WR":>7}')
print('-'*60)
all_summaries = []
for label, agg in config_results.items():
    for etype in ['Whale+SSL','Whale فقط','SSL فقط']:
        if agg[etype]:
            avg_eq = np.mean([b['eq'] for b in agg[etype]])
            avg_wr = np.mean([b['wr'] for b in agg[etype]])
            all_summaries.append((label, etype, len(agg[etype]), avg_eq, avg_wr))

all_summaries.sort(key=lambda x: -x[3])
for label, etype, coins_n, avg_eq, avg_wr in all_summaries[:15]:
    ico = '+' if avg_eq > CAP else '-'
    print(f'{label:<20} {etype:<12} {coins_n:>4} {ico}${avg_eq-CAP:>+7.1f} {avg_wr:>6.1f}%')

# ── Best config on ALL coins ──
best_config = all_summaries[0]
print(f'\n🚀 تشغيل أفضل إعدادات ({best_config[0]} + {best_config[1]}) على 198 عملة...')
best_label = best_config[0]
best_entry = best_config[1]
# Find the LB/sm/ssl for this label
best_params = None
for LB, sm, ssl, label in configs:
    if label == best_label:
        best_params = (LB, sm, ssl)
        break

if best_params:
    LB, sm, ssl = best_params
    all_data = []
    for i, sym in enumerate(coins):
        r = test_coin(sym, LB, sm, ssl)
        if r and best_entry in r['best']:
            all_data.append(r)
        if (i+1) % 40 == 0: print(f'   {i+1}/{len(coins)}...')
    
    print(f'\n📊 النتائج النهائية — {best_label} — {best_entry} — {len(all_data)} عملة:')
    
    total_t = 0; total_eq = 0; all_wr = []; winning = 0
    for d in all_data:
        b = d['best'][best_entry]
        total_t += b['t']
        total_eq += b['eq']
        all_wr.append(b['wr'])
        if b['eq'] > CAP: winning += 1
    
    avg_wr = np.mean(all_wr)
    avg_eq = total_eq / len(all_data)
    print(f'   صفقات: {total_t} | متوسط WR: {avg_wr:.1f}% | متوسط EQ: ${avg_eq:.0f}')
    print(f'   ربحانة: {winning}/{len(all_data)} | محفظة كلية: ${total_eq:.0f} | ربح: ${total_eq - CAP*len(all_data):+.0f}')
    
    # Top coins
    top = sorted(all_data, key=lambda x: -x['best'][best_entry]['eq'])[:15]
    print(f'\n🏆 أفضل 15 عملة ({best_label} + {best_entry}):')
    for d in top:
        b = d['best'][best_entry]
        print(f'{d["sym"]:<12} {b["t"]:>4d}t WR{b["wr"]:>5.1f}% DD{b["dd"]:>5.1f}% +${b["eq"]-CAP:>+7.1f} (TP{b["tp"]:.1f}/SL{b["sl"]:.1f})')

print('\n✅ تم')
