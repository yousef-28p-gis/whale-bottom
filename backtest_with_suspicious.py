#!/usr/bin/env python3 -u
"""باك تيست 5 سنوات — حلال + مشبوهة قابلة للاختبار"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}

# Blacklisted (from previous halal clean run)
BLACKLIST = {
    'QTUM', 'ZRO', 'IOTX', 'DYM', 'DGB', 'SAPIEN', 'XLM',
    'EDU', 'BTC', 'INIT', 'PARTI', '0G', 'ROBO', 'PYTH', 'ANKR'
}

# The 4 testable categories
TEST_CATEGORIES = {
    '🔄 منصات DEX': ['UNI','SUSHI','CAKE','DYDX','1INCH','COW','DODO','JOE','JUP','ORCA','QUICK','RAY','VELODROME'],
    '⚽ مشجعين': ['ACM','ALPINE','ASR','ATM','BAR','CITY','JUV','LAZIO','OG','PORTO','PSG','SANTOS'],
    '🎮 ألعاب': ['AGLD','ALICE','ANIME','AXS','BIGTIME','CATI','ENJ','GALA','HMSTR','ILV','MAGIC','MANA','NOT','PIXEL','SAND','TLM','YGG'],
    '🔒 خصوصية': ['DASH','PIVX','SCRT','XVG','ZEC'],
}

ALL_TESTABLE = set()
for coins in TEST_CATEGORIES.values():
    ALL_TESTABLE.update(coins)

print(f'⚙️ TP={TP} SL={SL} PL={PL} TR={TRAIL} MH={MH}h WHALE≥{WHALE_MIN} RSI<25')
print(f'🔍 109 حلال نظيف + {len(ALL_TESTABLE)} مشبوهة للاختبار | 5 سنوات | صفقتين×50%')
print()

all_trades = []
done = 0
coin_files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])

for fname in coin_files:
    sym = fname.replace('_15m.json','')
    if sym in BLACKLIST: continue
    
    with open(f'{CACHE_DIR}/{fname}') as f:
        data = json.load(f)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    if len(df) < 500: continue
    
    LB=30
    df['lo']=df['low'].rolling(LB).min()
    df['lc']=abs(df['low']-df['low'].shift(1))/df['low']*100
    df['sm']=df['lc'].ewm(span=3,adjust=False).mean()
    df['hi']=df['sm'].rolling(LB).max()
    df['raw']=np.where(df['low']<=df['lo'],(df['sm']+df['hi']*2)/3,0)
    df['whale']=df['raw'].ewm(span=3,adjust=False).mean().fillna(0)
    df['spike']=(df['whale']>df['whale'].shift(1))&(df['whale'].shift(1)<=0.03)
    df['wf']=df['whale'].rolling(2).mean(); df['ws']=df['whale'].rolling(5).mean()
    df['wp']=df['whale'].rolling(50).max()
    df['str']=(df['whale']/df['wp'].replace(0,np.nan)*100).fillna(0)
    df['vma']=df['volume'].rolling(20).mean()
    df['entry']=(df['spike']&(df['wf']>df['ws'])&(df['str']>STR)&(df['volume']>df['vma']*1.0))
    delta = df['close'].diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100/(1+rs))
    
    sym_trades = 0
    for i in range(50, len(df)-10):
        row = df.iloc[i]
        if not row['entry']: continue
        if float(row['whale']) < WHALE_MIN: continue
        if i+1 < len(df) and float(df.iloc[i+1]['whale']) >= 0.35: continue
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi >= 25: continue
        if row['ts'].weekday() == 3: continue
        if row['ts'].hour in BLOCK_HOURS: continue
        ps=max(0,i-96); pb=float(df.iloc[ps]['close']); ep=float(row['close'])
        if (ep-pb)/pb*100 >= 0: continue
        
        tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
        pl_p=ep+(tp_p-ep)*(PL/100)
        pl_trig=False; peak=ep; trail_p=0
        for k in range(i+1, len(df)):
            cur=float(df.iloc[k]['close']); h=(k-i)*0.25
            if h>MH: pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
            if cur>=tp_p: pnl=round(TP-COMM,4); exit_='TP'; break
            if cur<=sl_p: pnl=round(-SL-COMM,4); exit_='SL'; break
            if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
            if pl_trig:
                if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
                if cur<=trail_p: pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
        else: pnl=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4); exit_='EOD'
        all_trades.append({'sym':sym, 'dt':row['ts'], 'pnl':pnl, 'exit':exit_})
        sym_trades += 1
    
    done += 1
    if done % 20 == 0:
        print(f'  {done}/{len(coin_files)-len(BLACKLIST)} عملة...', flush=True)

print(f'  ✅ {done}/{len(coin_files)-len(BLACKLIST)} عملة مكتملة', flush=True)

# Overall stats
nets=[t['pnl'] for t in all_trades]
wins=sum(1 for n in nets if n>0)
print(f'\n{"="*60}')
print(f'📊 الإجمالي: {len(all_trades)} إشارة | WR {wins/len(all_trades)*100:.1f}% | صافي {sum(nets):+.1f}%')
print(f'{"="*60}')

# Per-coin stats
coin_stats = defaultdict(lambda: {'t':0,'w':0,'pnl':0.0})
for t in all_trades:
    c = coin_stats[t['sym']]
    c['t'] += 1; c['pnl'] += t['pnl']
    if t['pnl']>0: c['w']+=1

# Group by category
print('\n📊 نتائج حسب الفئة:')
print(f'{"الفئة":<25} {"عملات":>5} {"إشارات":>6} {"WR":>6} {"صافي":>8}')
print('-' * 55)

# Halal (baseline)
halal_coins = set(coin_stats.keys()) - ALL_TESTABLE
h_t = sum(coin_stats[c]['t'] for c in halal_coins)
h_w = sum(coin_stats[c]['w'] for c in halal_coins)
h_p = sum(coin_stats[c]['pnl'] for c in halal_coins)
h_wr = h_w/h_t*100 if h_t>0 else 0
print(f'{"✅ حلال نظيف":<25} {len(halal_coins):>5} {h_t:>6} {h_wr:>5.1f}% {h_p:>+7.1f}%')

for cat_name, cat_coins in TEST_CATEGORIES.items():
    c_t = sum(coin_stats[c]['t'] for c in cat_coins if c in coin_stats)
    c_w = sum(coin_stats[c]['w'] for c in cat_coins if c in coin_stats)
    c_p = sum(coin_stats[c]['pnl'] for c in cat_coins if c in coin_stats)
    c_wr = c_w/c_t*100 if c_t>0 else 0
    n = sum(1 for c in cat_coins if c in coin_stats)
    tag = '✅' if c_p > 0 else '❌'
    print(f'{tag} {cat_name:<23} {n:>5} {c_t:>6} {c_wr:>5.1f}% {c_p:>+7.1f}%')

# Detail per category
for cat_name, cat_coins in TEST_CATEGORIES.items():
    print(f'\n{"─"*55}')
    print(f'{cat_name}:')
    for sym in sorted(cat_coins):
        if sym not in coin_stats: continue
        c = coin_stats[sym]
        wr = c['w']/c['t']*100 if c['t']>0 else 0
        tag = '🟢' if c['pnl'] > 0 else '🔴'
        print(f'  {tag} {sym:<14} {c["t"]:>3}ت | WR {wr:.0f}% | صافي {c["pnl"]:+.1f}%')

# Portfolio: all combined (halal + testable)
trades_sorted = sorted(all_trades, key=lambda x: x['dt'])
capital = 1000.0; peak = 1000.0; max_dd = 0.0
active = []; skipped = 0; taken = 0

for t in trades_sorted:
    dt = t['dt']
    still_active = []
    for exit_dt, cost, pnl_amt in active:
        if dt >= exit_dt:
            capital += cost + pnl_amt
        else:
            still_active.append((exit_dt, cost, pnl_amt))
    active = still_active
    
    if len(active) >= 2:
        skipped += 1; continue
    pos_size = capital * 0.50
    if capital < pos_size:
        skipped += 1; continue
    pnl_amt = pos_size * t['pnl'] / 100
    capital -= pos_size
    active.append((dt + timedelta(hours=MH), pos_size, pnl_amt))
    taken += 1
    
    equity = capital + sum(pc + pd for _, pc, pd in active)
    if equity > peak: peak = equity
    dd = (equity - peak) / peak * 100
    if dd < max_dd: max_dd = dd

for _, cost, pnl_amt in active:
    capital += cost + pnl_amt

exec_nets = [t['pnl'] for t in trades_sorted[:taken]]  # approximate
years = 5

print(f'\n{"="*60}')
print(f'💼 المحفظة المجمعة — {years} سنوات')
print(f'{"="*60}')
print(f'✅ منفذة: {taken} | ⏭️ متخطية: {skipped}')
print(f'💼 محفظة: $1000 → ${capital:.0f} ({capital/10-100:+.1f}%)')
print(f'📉 سحب: {max_dd:.2f}%')
ann_return = (capital/1000)**(1/years)-1 if capital > 0 else -1
print(f'📈 عائد سنوي: {ann_return*100:+.1f}%')
