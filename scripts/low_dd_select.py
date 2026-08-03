#!/usr/bin/env python3
"""
تحسين لتقليل DD + استبعاد الخسرانة
score = ربح - DD×10 - صفقات منهارة×100
"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

COMM = 0.002; CAP = 1000; DATA = '/data/trading28/data/whale_15m_1y'
MAX_SLIPPAGE = 1.5

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
    crash_trades = 0
    for i in range(200,n):
        if pos:
            if h[i] >= ep*(1+tp/100):
                pnl = tp - COMM*100
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cd
            elif l_[i] <= ep*(1-sl/100):
                raw_pnl = (c[i]/ep - 1)*100 - COMM*100
                max_loss = -sl*MAX_SLIPPAGE - COMM*100
                pnl = max(raw_pnl, max_loss)
                if raw_pnl < max_loss: crash_trades += 1
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cd
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    return t,cv,eq,crash_trades

def test_coin(sym):
    d = load(sym)
    if d is None or len(d['c'])<2000: return None
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; n=len(c)
    
    # filter extreme bars
    for i in range(1,n):
        if abs(c[i]/c[i-1]-1)*100 > 40: return None
    
    best_result = None; best_score = -999999
    
    whale_configs = [(30,3),(50,3),(70,3)]
    ssl_configs = [5,10,20]
    tp_sl_pairs = [(2.0,1.0),(3.0,1.5),(5.0,2.5)]
    
    for LB, sm in whale_configs:
        wp, wp_up = whale_signal(l_, n, LB=LB, smooth=sm)
        for sp in ssl_configs:
            ssl_c = ssl_lines(h, l_, n, period=sp)
            
            # Whale only
            le_w = np.zeros(n,bool)
            for i in range(200,n):
                if wp_up[i] and wp[i]>wp[i-2]*1.5 and wp[i]>0: le_w[i]=True
            
            # W+SSL
            le_ws = np.zeros(n,bool)
            for i in range(200,n):
                if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0: le_ws[i]=True
            
            for tp, sl in tp_sl_pairs:
                for name, le in [('Whale',le_w), ('W+SSL',le_ws)]:
                    if le.sum()<5: continue
                    tr,cv,eq,crash = sim(le, c, h, l_, n, tp, sl)
                    if len(tr)<5: continue
                    
                    w=[p for p in tr if p>0]; lo=[p for p in tr if p<=0]
                    wr=len(w)/len(tr)*100
                    dd_val=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
                    
                    # NEW SCORE: heavily penalize DD and crashes, reward profit
                    score = (eq-CAP) - abs(dd_val)*15 - crash*100 + wr*1.5
                    
                    result = {'name':name,'LB':LB,'sm':sm,'ssl':sp,'tp':tp,'sl':sl,
                              't':len(tr),'wr':wr,'dd':dd_val,'eq':eq,'sigs':le.sum(),
                              'w':len(w),'l':len(lo),'score':score,'crash':crash}
                    
                    if score > best_score:
                        best_score = score; best_result = result
    
    if best_result is None: return None
    return {'sym':sym,'best':best_result}

# ── Run ──
print('🔄 تحسين لتقليل DD...')
coins = sorted([f.replace('.json','') for f in os.listdir(DATA) if f.endswith('.json') and f!='_manifest.json'])

all_data=[]
for i,sym in enumerate(coins):
    r=test_coin(sym)
    if r: all_data.append(r)
    if (i+1)%30==0: print(f'  {i+1}/{len(coins)}...')

# ── Apply strict filters ──
FILTERS = {
    'min_wr': 42,      # WR ≥ 42%
    'max_dd': 20,       # DD ≤ 20%
    'min_trades': 10,   # 10+ trades
    'max_crash': 3,     # max 3 crash trades
}

passed = [d for d in all_data if 
    d['best']['eq'] > CAP and
    d['best']['wr'] >= FILTERS['min_wr'] and
    abs(d['best']['dd']) <= FILTERS['max_dd'] and
    d['best']['t'] >= FILTERS['min_trades'] and
    d['best']['crash'] <= FILTERS['max_crash']]

failed = [d for d in all_data if d not in passed]

print(f'\n{"="*70}')
print(f'📊 بعد الفلاتر الصارمة:')
print(f'   WR ≥ {FILTERS["min_wr"]}% | DD ≤ {FILTERS["max_dd"]}%')
print(f'   صفقات ≥ {FILTERS["min_trades"]} | انهيارات ≤ {FILTERS["max_crash"]}')
print(f'{"="*70}')
print(f'   ✅ اجتازوا: {len(passed)}/{len(all_data)}')
print(f'   ❌ رُفضوا: {len(failed)}')

if not passed:
    print('\nلا توجد عملات تجتاز الفلاتر! جرب تخفيف...')
    # Try relaxed
    FILTERS = {'min_wr':38,'max_dd':25,'min_trades':8,'max_crash':5}
    passed = [d for d in all_data if 
        d['best']['eq']>CAP and d['best']['wr']>=FILTERS['min_wr'] and
        abs(d['best']['dd'])<=FILTERS['max_dd'] and d['best']['t']>=FILTERS['min_trades'] and
        d['best']['crash']<=FILTERS['max_crash']]
    print(f'   🟡 مخففة: {len(passed)} عملة')
else:
    avg_profit = np.mean([d['best']['eq']-CAP for d in passed])
    avg_wr = np.mean([d['best']['wr'] for d in passed])
    avg_dd = np.mean([d['best']['dd'] for d in passed])
    total = sum(d['best']['eq']-CAP for d in passed)
    
    print(f'\n🟢 المحفظة النهائية:')
    print(f'   عملات: {len(passed)}')
    print(f'   متوسط ربح: +${avg_profit:.1f}')
    print(f'   متوسط WR: {avg_wr:.1f}%')
    print(f'   متوسط DD: {avg_dd:.1f}%')
    print(f'   إجمالي ربح: +${total:.0f}')
    
    strats = defaultdict(int)
    for d in passed: strats[d['best']['name']] += 1
    print(f'   استراتيجيات: {dict(strats)}')
    
    print(f'\n{"="*70}')
    print(f'🏆 العملات المختارة:')
    print(f'{"عملة":<12} {"استراتيجية":<8} {"إعدادات":>10} {"TP/SL":>8} {"صفقات":>5} {"WR":>6} {"DD":>6} {"💥":>4} {"ربح":>9}')
    print('-'*75)
    
    best_coins = sorted(passed, key=lambda x: -x['best']['eq'])
    for d in best_coins:
        b = d['best']
        if b['name']=='W+SSL': params=f'{b["LB"]}/{b["ssl"]}'
        else: params=str(b['LB'])
        ico = '+' if b['eq']>CAP else '-'
        print(f'{d["sym"]:<12} {b["name"]:<8} {params:>10} {b["tp"]:.0f}/{b["sl"]:.1f}  {b["t"]:>5} {b["wr"]:>5.1f}% {b["dd"]:>5.1f}% {b["crash"]:>3} {ico}${b["eq"]-CAP:>+8.1f}')

    # Save
    with open('/data/trading28/final_portfolio.json','w') as f:
        out = []
        for d in passed:
            b = d['best']
            out.append({'sym':d['sym'],'strategy':b['name'],'LB':int(b['LB']),'ssl':int(b['ssl']),
                         'tp':float(b['tp']),'sl':float(b['sl']),'t':int(b['t']),
                         'wr':float(b['wr']),'dd':float(b['dd']),'eq':float(b['eq'])})
        json.dump(out, f)
    print(f'\n💾 حفظت → final_portfolio.json')

print('\n✅ Done')
