#!/usr/bin/env python3
"""اختبار 5 أفكار جديدة — 20 عملة"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

CACHE_DIR = '/data/trading28/data/5year_halal'
MH=6; COMM=0.20; WHALE_MIN=0.50; STR=50; BLOCK_HOURS={1,3,6,12,0,4}
BLACKLIST = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','0G','ROBO','PYTH','ANKR'}
COINS = ['ADA','ETH','SOL','DOGE','AVAX','LINK','DOT','ATOM','GRT','SAND',
         'MATIC','NEAR','FIL','AR','FET','XRP','LTC','UNI','AAVE','THETA']

# Baseline: 15m + فلتر3 + تريل0.1%
BASELINE = {'tp': 3.5, 'sl': 1.5, 'pl': 30, 'trail': 0.10, 'step': 1, 'grace': 0, 'vol_filter': False}

VARIANTS = [
    ('1. 1h + تريل 0.5%',           {'tp': 3.5, 'sl': 1.5, 'pl': 30, 'trail': 0.50, 'step': 4, 'grace': 0, 'vol_filter': False}),
    ('2. خروج 30m',                  {'tp': 3.5, 'sl': 1.5, 'pl': 30, 'trail': 0.10, 'step': 2, 'grace': 0, 'vol_filter': False}),
    ('3. TP=5% SL=2% بدون تريل',     {'tp': 5.0, 'sl': 2.0, 'pl': 100, 'trail': 0,   'step': 1, 'grace': 0, 'vol_filter': False}),
    ('4. سماح 3 شمعات',              {'tp': 3.5, 'sl': 1.5, 'pl': 30, 'trail': 0.10, 'step': 1, 'grace': 3, 'vol_filter': False}),
    ('5. فلتر حجم 2×',               {'tp': 3.5, 'sl': 1.5, 'pl': 30, 'trail': 0.10, 'step': 1, 'grace': 0, 'vol_filter': True}),
    ('0. حالي (مرجع)',               BASELINE),
]

results = {}

for vname, cfg in VARIANTS:
    vtp=cfg['tp']; vsl=cfg['sl']; vpl=cfg['pl']; vtrail=cfg['trail']
    step=cfg['step']; grace=cfg['grace']; volf=cfg['vol_filter']
    use_trail = vtrail > 0 and vpl < 100
    
    all_trades = []
    exits = defaultdict(int)
    
    for sym in COINS:
        fp = f'{CACHE_DIR}/{sym}_15m.json'
        if not os.path.exists(fp): continue
        if sym in BLACKLIST: continue
        
        with open(fp) as f: data = json.load(f)
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
        df['vma50']=df['volume'].rolling(50).mean()
        df['entry']=(df['spike']&(df['wf']>df['ws'])&(df['str']>STR)&(df['volume']>df['vma']*1.0))
        delta = df['close'].diff()
        gain = delta.where(delta>0,0).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100/(1+rs))
        
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
            
            # Filter 3: green confirm
            if i+1 >= len(df): continue
            if float(df.iloc[i+1]['close']) <= float(df.iloc[i+1]['open']): continue
            
            # Volume filter
            if volf:
                vol_ratio = float(row['volume']) / float(df.iloc[i]['vma50']) if not np.isnan(df.iloc[i]['vma50']) and float(df.iloc[i]['vma50']) > 0 else 0
                if vol_ratio < 2.0:
                    continue
            
            tp_p=ep*(1+vtp/100); sl_p=ep*(1-vsl/100)
            pl_p=ep+(tp_p-ep)*(vpl/100)
            pl_trig=False; peak=ep; trail_p=0
            pnl = None; exit_ = 'EOD'
            
            if step == 1:
                # Every candle
                for k in range(i+1, len(df)):
                    cur=float(df.iloc[k]['close']); h=(k-i)*0.25
                    if k-i <= grace: continue  # skip exit checks during grace period
                    if h>MH: pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
                    if cur>=tp_p: pnl=round(vtp-COMM,4); exit_='TP'; break
                    if cur<=sl_p: pnl=round(-vsl-COMM,4); exit_='SL'; break
                    if use_trail:
                        if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-vtrail/100)
                        if pl_trig:
                            if cur>peak: peak=cur; trail_p=cur*(1-vtrail/100)
                            if cur<=trail_p: pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
            else:
                # Step-based exit
                k = i + step
                while k < len(df):
                    cur=float(df.iloc[k]['close']); h=(k-i)*0.25
                    if k-i <= grace: k += step; continue
                    if h>MH: pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
                    if cur>=tp_p: pnl=round(vtp-COMM,4); exit_='TP'; break
                    if cur<=sl_p: pnl=round(-vsl-COMM,4); exit_='SL'; break
                    if use_trail:
                        if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-vtrail/100)
                        if pl_trig:
                            if cur>peak: peak=cur; trail_p=cur*(1-vtrail/100)
                            if cur<=trail_p: pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
                    k += step
            
            if pnl is None:
                pnl=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4); exit_='EOD'
            
            all_trades.append({'sym':sym, 'dt':row['ts'], 'pnl':pnl, 'exit':exit_})
            exits[exit_] += 1
    
    arr = np.array([t['pnl'] for t in all_trades])
    wins_arr = arr[arr>0]; loss_arr = arr[arr<0]
    wc = len(wins_arr); lc = len(loss_arr)
    tp_agg = wins_arr.sum() if wc > 0 else 0
    tl_agg = loss_arr.sum() if lc > 0 else 0
    aw = wins_arr.mean() if wc > 0 else 0
    al = loss_arr.mean() if lc > 0 else 0
    rr = aw/abs(al) if al != 0 else 0
    sharpe = arr.mean()/arr.std()*np.sqrt(365*24*4) if arr.std() > 0 else 0
    
    # Portfolio
    trades_sorted = sorted(all_trades, key=lambda x: x['dt'])
    capital = 1000.0; peak=1000.0; max_dd=0.0; active=[]; skipped=0; taken=0
    for t in trades_sorted:
        dt=t['dt']
        still_active=[]
        for ed,c,p in active:
            if dt>=ed: capital+=c+p
            else: still_active.append((ed,c,p))
        active=still_active
        if len(active)>=2: skipped+=1; continue
        ps2=capital*0.50
        if capital<ps2: skipped+=1; continue
        pa=ps2*t['pnl']/100
        capital-=ps2
        active.append((dt+timedelta(hours=MH),ps2,pa))
        taken+=1
        eq=capital+sum(pc+pd for _,pc,pd in active)
        if eq>peak: peak=eq
        dd=(eq-peak)/peak*100
        if dd<max_dd: max_dd=dd
    for _,c,p in active: capital+=c+p
    ann=(capital/1000)**(1/5)-1 if capital>0 else -1
    
    tp_c=exits.get('TP',0); sl_c=exits.get('SL',0)
    tr_c=exits.get('TRAIL',0); tm_c=exits.get('TIME',0)+exits.get('EOD',0)
    
    results[vname] = {
        't': len(arr), 'w': wc, 'l': lc, 'wr': wc/len(arr)*100,
        'tprof': tp_agg, 'tloss': tl_agg, 'net': tp_agg+tl_agg,
        'aw': aw, 'al': al, 'rr': rr, 'sharpe': sharpe,
        'cap': capital, 'dd': max_dd, 'ann': ann,
        'exec': taken, 'skip': skipped,
        'tp_c': tp_c, 'sl_c': sl_c, 'tr_c': tr_c, 'tm_c': tm_c
    }
    
    print(f'{vname:<35} {len(arr):>4}ت WR {wc/len(arr)*100:>5.1f}%  محفظة \${capital:>8,.0f}  سنوي {ann*100:>+6.1f}%  سحب {max_dd:>+5.2f}%', flush=True)

print(f'\n{"="*80}')
print(f'📊 مقارنة 5 أفكار جديدة — {len(COINS)} عملة')
print(f'{"="*80}')
print(f'{"":<30} {"ت":>4} {"WR":>6} {"صافي":>8} {"محفظة":>9} {"سحب":>7} {"سنوي":>7} {"R:R":>5} {"TP":>4} {"SL":>4} {"TRAIL":>5}')
print(f'{"-"*80}')

names_order = ['0. حالي (مرجع)', '1. 1h + تريل 0.5%', '2. خروج 30m', '3. TP=5% SL=2% بدون تريل', '4. سماح 3 شمعات', '5. فلتر حجم 2×']
for vname in names_order:
    r = results.get(vname)
    if not r: continue
    print(f'{vname:<30} {r["t"]:>4} {r["wr"]:>5.1f}% {r["net"]:>+7.1f}% \${r["cap"]:>8,.0f} {r["dd"]:>6.2f}% {r["ann"]*100:>+6.1f}% {r["rr"]:>4.1f}x {r["tp_c"]:>4} {r["sl_c"]:>4} {r["tr_c"]:>5}')

print(f'\n✅ تم')
