#!/usr/bin/env python3
"""
Whale+SSL — 198 عملة × سنة — فقط في فترات الترند الصاعد
فلتر: 4h EMA50>200 + 15m EMA50>200 — لازم الاتنين
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

def resample_4h(d):
    df = pd.DataFrame({'c': d['c']}, index=d['ts'])
    return df['c'].resample('4h').last().dropna().values

def test_coin(sym):
    d = load(sym)
    if d is None or len(d['c'])<2000: return None
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; n=len(c); idx=d['ts']
    
    # ── Trend filter: 4h EMA50>200 + 15m EMA50>200 ──
    try:
        c4h=resample_4h(d)
        e50_4=ema(c4h,50); e200_4=ema(c4h,200)
        e50_15=ema(c,50); e200_15=ema(c,200)
    except:
        return None
    
    trend_ok=np.zeros(n,bool)
    for i in range(200,n):
        j=i//16  # 15m→4h mapping
        if j>=len(e50_4) or j>=len(e200_4): continue
        if np.isnan(e50_4[j]) or np.isnan(e200_4[j]): continue
        if np.isnan(e50_15[i]) or np.isnan(e200_15[i]): continue
        # Both TFs must be in uptrend
        if e50_15[i]>e200_15[i] and e50_4[j]>e200_4[j]:
            trend_ok[i]=True
    
    green_pct=trend_ok.sum()/n*100
    if green_pct<3: return None  # skip dead coins
    
    # ── SSL ──
    p=10
    sma_h=pd.Series(h).rolling(p).mean().values
    sma_l=pd.Series(l_).rolling(p).mean().values
    ssl_c=np.zeros(n,int)
    for i in range(p,n):
        if h[i-1]>sma_h[i-1]: ssl_c[i]=1
        else: ssl_c[i]=-1
    
    # ── Whale ──
    LB=50
    ln=pd.Series(l_).shift(1).rolling(LB).min().values
    lc=np.zeros(n)
    for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
    sc=pd.Series(lc).ewm(span=3,adjust=False).mean().values
    hc=pd.Series(sc).rolling(LB).max().values
    sr=np.where(l_<=ln,(sc+hc*2)/3,0)
    wp=pd.Series(sr).ewm(span=3,adjust=False).mean().values
    wp_up=wp>np.roll(wp,1)
    
    # ── Entry: SSL blue + whale rising + trend OK ──
    le=np.zeros(n,bool)
    for i in range(200,n):
        if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0 and trend_ok[i]:
            le[i]=True
    
    if le.sum()<5: return None
    
    # ── Test TP/SL ──
    results={}
    for tp,sl in [(3.0,1.5),(5.0,2.5)]:
        t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
        for i in range(200,n):
            if pos:
                if h[i]>=ep*(1+tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=12
                elif l_[i]<=ep*(1-sl/100):
                    pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=12
            if not pos and cool==0 and le[i]: pos=1; ep=c[i]
            if not pos and cool>0: cool-=1
            cv.append(eq)
        if pos:
            pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
        if len(t)<5: continue
        w=[p for p in t if p>0]; lo=[p for p in t if p<=0]
        wr=len(w)/len(t)*100
        aw=np.mean(w) if w else 0; al=abs(np.mean(lo)) if lo else 0
        dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
        results[(tp,sl)]={'t':len(t),'wr':wr,'dd':dd,'eq':eq,'w':len(w),'l':len(lo)}
    
    if not results: return None
    return {'sym':sym,'green':green_pct,'res':results,'sigs':le.sum(),'n':n}

# ── Run ──
print('🔄 Whale+SSL + ترند 4h+15m — 198 عملة...')
coins=sorted([f.replace('.json','') for f in os.listdir(DATA) if f.endswith('.json') and f!='_manifest.json'])

all_data=[]
for i,sym in enumerate(coins):
    r=test_coin(sym)
    if r: all_data.append(r)
    if (i+1)%40==0: print(f'  {i+1}/{len(coins)}...')

print(f'\n✅ {len(all_data)} عملة عندها فترات صاعدة + صفقات كافية\n')

# ── Aggregate ──
for tp,sl in [(3.0,1.5),(5.0,2.5)]:
    items=[d for d in all_data if (tp,sl) in d['res']]
    if not items: continue
    
    total_t=sum(d['res'][(tp,sl)]['t'] for d in items)
    total_w=sum(d['res'][(tp,sl)]['w'] for d in items)
    total_l=sum(d['res'][(tp,sl)]['l'] for d in items)
    eq_sum=sum(d['res'][(tp,sl)]['eq'] for d in items)
    avg_wr=np.mean([d['res'][(tp,sl)]['wr'] for d in items])
    avg_dd=np.mean([d['res'][(tp,sl)]['dd'] for d in items])
    win=sum(1 for d in items if d['res'][(tp,sl)]['eq']>CAP)
    profit=eq_sum-CAP*len(items)
    ico='✅' if profit>0 else '❌'
    
    print(f'TP{tp}/SL{sl}: {len(items)} عملة | {total_t} صفقة | WR {total_w/(total_w+total_l)*100:.1f}% | سحب {avg_dd:.1f}% | {ico} ${profit:+.0f} | {win}✅/{len(items)-win}❌')

# ── Top 20 ──
print(f'\n🏆 أفضل 20 — TP5/SL2.5:')
best=[]
for d in all_data:
    if (5.0,2.5) in d['res']:
        r=d['res'][(5.0,2.5)]
        best.append({'sym':d['sym'],'green':d['green'],**r})
best.sort(key=lambda x:-x['eq'])

print(f'{"عملة":<12} {"أخضر%":>6} {"صفقات":>5} {"WR":>6} {"سحب":>6} {"ربح":>8}')
print('-'*50)
for r in best[:20]:
    ico='+' if r['eq']>CAP else '-'
    print(f'{r["sym"]:<12} {r["green"]:>5.0f}% {r["t"]:>5} {r["wr"]:>5.1f}% {r["dd"]:>5.1f}% {ico}${r["eq"]-CAP:>+7.1f}')

# Quick stats
print(f'\n📊 إحصائيات TP5/SL2.5:')
items=[d for d in all_data if (5.0,2.5) in d['res']]
prof=[d['res'][(5.0,2.5)] for d in items if d['res'][(5.0,2.5)]['eq']>CAP]
print(f'   عملات: {len(items)} | ربحانة: {len(prof)} ({len(prof)/max(1,len(items))*100:.0f}%)')
if prof:
    print(f'   متوسط WR للربحانة: {np.mean([p["wr"] for p in prof]):.1f}%')
    print(f'   متوسط ربح: +${np.mean([p["eq"]-CAP for p in prof]):.1f}')
    print(f'   أفضل: +${max(p["eq"]-CAP for p in prof):.1f} | أسوأ ربحانة: +${min(p["eq"]-CAP for p in prof):.1f}')

print('\n✅ Done')
