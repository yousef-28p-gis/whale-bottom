#!/usr/bin/env python3
"""Deep analysis: WHY did the 30 coins succeed?"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')

def load(path):
    if not os.path.exists(path): return None
    with open(path) as f: d=json.load(f)
    return (np.array(d['c'],float), d.get('ts',[]))

def trends(c,ts,n):
    def e(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values
    try:
        idx=pd.to_datetime(np.array(ts),unit='ms')
        df=pd.DataFrame({'c':c},index=idx)
        c4h=df['c'].resample('4h').last().dropna().values
        e50=e(c4h,50); e200=e(c4h,200)
        e50a=np.zeros(n); e200a=np.zeros(n)
        for i in range(n):
            j=i//16
            if j<len(e50): e50a[i]=e50[j]; e200a[i]=e200[j]
        return (e50a>e200a).sum()/n*100
    except: return 50

with open('/data/trading28/per_coin_best.json') as f: data = json.load(f)

both = [c for c in data if c['prev_pnl']>0 and c['cur_pnl']>0]
losers = [c for c in data if c['prev_pnl']<=0 or c['cur_pnl']<=0]

print("═══ تحليل أسباب النجاح ═══\n")

# 1. Market context: BTC performance
print("1️⃣ سياق السوق (BTC):")
for pname, pdir in [('PREV','prev'),('CUR','1y')]:
    for sym in ['BTC','ETH']:
        p=f'/data/trading28/data/whale_15m_{pdir}/{sym}.json'
        if os.path.exists(p):
            c,ts=load(p); n=len(c)
            green=trends(c,ts,n)
            first=c[0]; last=c[-1]; chg=(last/first-1)*100
            print(f"  {sym} {pname}: 4h↑={green:.0f}% | {first:.4f}→{last:.4f} ({chg:+.1f}%)")

# 2. Trend stats for winners vs losers
print("\n2️⃣ نسبة الوقت في ترند صاعد (4h EMA50>200):")
w_prev=[c for c in both]
l_prev=[c for c in losers]

# Read green% from configs
for label, coins in [('✅ 30 ربحانة', both), ('❌ 55 خسرانة', losers)]:
    # Compute from data
    greens_prev=[]; greens_cur=[]
    for c in coins:
        p1=f'/data/trading28/data/whale_15m_prev/{c["sym"]}.json'
        p2=f'/data/trading28/data/whale_15m_1y/{c["sym"]}.json'
        for p,gl in [(p1,greens_prev),(p2,greens_cur)]:
            if os.path.exists(p):
                cc,ts=load(p)
                if cc is not None: gl.append(trends(cc,ts,len(cc)))
    g_prev=np.mean(greens_prev) if greens_prev else 0
    g_cur=np.mean(greens_cur) if greens_cur else 0
    print(f"  {label}: PREV 4h↑={g_prev:.0f}% | CUR 4h↑={g_cur:.0f}%")

# 3. Buy & hold comparison
print("\n3️⃣ هل الاستراتيجية تغلبت على الشراء والاحتفاظ؟")
for label, coins in [('✅ Winners',both), ('❌ Losers',losers)]:
    bh_prev=0; bh_cur=0; count=0
    for c in coins:
        for pname,pdir in [('PREV','prev'),('CUR','1y')]:
            p=f'/data/trading28/data/whale_15m_{pdir}/{c["sym"]}.json'
            if os.path.exists(p):
                cc,ts=load(p)
                if cc is not None:
                    ret=(cc[-1]/cc[0]-1)*100
                    if pname=='PREV': bh_prev+=ret
                    else: bh_cur+=ret
                    count+=1 if pname=='PREV' else 0
    n=len(coins)
    print(f"  {label}: PREV B&H={bh_prev/n:.0f}% | Strategy: ${sum(c['prev_pnl'] for c in coins):+.0f} | Beat? {'✅' if sum(c['prev_pnl'] for c in coins)>bh_prev else '❌'}")
    print(f"           CUR  B&H={bh_cur/n:.0f}% | Strategy: ${sum(c['cur_pnl'] for c in coins):+.0f} | Beat? {'✅' if sum(c['cur_pnl'] for c in coins)>bh_cur else '❌'}")

# 4. Correlation: green% vs PnL
print("\n4️⃣ ارتباط الاخضرار بالربح:")
all_coins=[]
for c in data:
    p1=f'/data/trading28/data/whale_15m_prev/{c["sym"]}.json'
    p2=f'/data/trading28/data/whale_15m_1y/{c["sym"]}.json'
    g1=0; g2=0
    if os.path.exists(p1):
        cc,ts=load(p1)
        if cc is not None: g1=trends(cc,ts,len(cc))
    if os.path.exists(p2):
        cc,ts=load(p2)
        if cc is not None: g2=trends(cc,ts,len(cc))
    all_coins.append({'sym':c['sym'],'g_prev':g1,'g_cur':g2,
        'pnl_prev':c['prev_pnl'],'pnl_cur':c['cur_pnl'],
        'won': c['prev_pnl']>0 and c['cur_pnl']>0})

# Group by green%
for g_threshold in [10,20,30,40]:
    high_g = [c for c in all_coins if c['g_prev']>=g_threshold and c['g_cur']>=g_threshold]
    won = sum(1 for c in high_g if c['won'])
    print(f"  Green≥{g_threshold}%: {len(high_g)} coins, {won} won ({won/len(high_g)*100:.0f}%)")

# 5. Average trade count
print("\n5️⃣ متوسط الصفقات:")
print(f"  ✅ فائزين: PREV={np.mean([c['prev_t'] for c in both]):.0f} صفقة | CUR={np.mean([c['cur_t'] for c in both]):.0f}")
print(f"  ❌ خاسرين: PREV={np.mean([c['prev_t'] for c in losers]):.0f} صفقة | CUR={np.mean([c['cur_t'] for c in losers]):.0f}")

# 6. Strategy breakdown for winners
from collections import Counter
print("\n6️⃣ توزيع الاستراتيجيات (الفائزين vs الخاسرين):")
w_strat = Counter(c['strat'] for c in both)
l_strat = Counter(c['strat'] for c in losers)
for s in sorted(set(list(w_strat)+list(l_strat))):
    wt=w_strat.get(s,0); lt=l_strat.get(s,0); total=wt+lt
    print(f"  {s:<12}: ✅{wt} ❌{lt} (نجاح {wt/total*100:.0f}%)")

print("\n═══ الخلاصة ═══")
g_winners = np.mean([c['g_prev'] for c in all_coins if c['won']])
g_losers = np.mean([c['g_prev'] for c in all_coins if not c['won']])
print(f"الفائزين متوسط اخضرارهم: {g_winners:.0f}%")
print(f"الخاسرين متوسط اخضرارهم: {g_losers:.0f}%")
print(f"الفرق: {g_winners-g_losers:.0f}% — {'كبير (السوق هو السبب)' if g_winners-g_losers>10 else 'صغير (الاستراتيجية لها دور)'}")
