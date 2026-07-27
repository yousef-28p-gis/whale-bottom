#!/usr/bin/env python3
"""اختبار أوامر حد — شراء بسعر أقل من الإشارة"""
import json, numpy as np, pandas as pd, os, time, gc

COMM = 0.20; TF_MIN = 3; CAPITAL = 1000; MAX_POS = 2
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# TP1.3 SL0.5 PL12 TRAIL0.02 MH4 WHALE≥0.10 RSI<35 تأكيد بدون تباعد
# مع أوامر حد بنسب مختلفة
LIMIT_OFFSETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def compute_indicators(df):
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    w = (df['low'].values-df['low_raw'].values)/np.where(df['low_raw'].values!=0, df['low_raw'].values, np.nan)*100
    df['whale'] = np.clip(w,0,None)
    vm = df['volume'].rolling(20).mean().values
    df['spike'] = df['volume'].values/np.where(vm!=0, vm, np.nan)
    delta = df['close'].diff().values
    gain = pd.Series(np.where(delta>0,delta,0)).rolling(14).mean().values
    loss = pd.Series(np.where(delta<0,-delta,0)).rolling(14).mean().values
    df['rsi'] = 100-100/(1+gain/np.where(loss!=0,loss,np.nan))
    return df

def find_signals(df):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    wh=df['whale'].values; sp=df['spike'].values; rs=df['rsi'].values
    mask=(wh>=0.10)&(sp>=1.5)&(rs<35)&~np.isnan(wh)&~np.isnan(sp)&~np.isnan(rs)
    mask[:50]=False
    ng=np.zeros(n,dtype=bool); ng[:-1]=df['close'].values[1:]>df['open'].values[1:]; mask&=ng
    return np.where(mask)[0]

def simulate_limit(df, signal_idxs, tp, sl, pl, trail, max_h, limit_offset_pct):
    """
    Limit order simulation:
    - Signal at candle i with close price P
    - Place limit at P × (1 - limit_offset/100)
    - Scan forward from i+1: first candle where LOW <= limit_price → entry
    - If filled: standard TP/SL/TRAIL/TIME logic
    - If never filled within max candes or price goes above entry+2% first: missed
    """
    n = len(df)
    close_arr = df['close'].values
    low_arr = df['low'].values
    high_arr = df['high'].values
    ts_arr = df['ts'].values.astype('datetime64[ns]').astype('int64')
    
    max_bars = int(max_h * 60 / TF_MIN)
    tp_r = 1 + tp/100
    sl_r = 1 - sl/100
    tr_r = 1 - trail/100
    
    trades = []
    active = []
    missed = 0
    filled = 0
    
    for i in range(n):
        cur_close = close_arr[i]
        
        # Check new signals → place limit orders
        if i in signal_idxs:
            signal_price = close_arr[i]
            limit_price = signal_price * (1 - limit_offset_pct/100)
            
            # Place as pending limit order (not yet active)
            active.append({
                'type': 'pending',
                'signal_price': signal_price,
                'limit': limit_price,
                'tp': limit_price * tp_r,
                'sl': limit_price * sl_r,
                'pl_ok': False,
                'peak': limit_price,
                'trail': limit_price,
                'entry_i': i,
                'entry_ns': int(ts_arr[i]),
                'bars_waiting': 0,
            })
        
        # Check pending orders: did price reach limit?
        for j in range(len(active)-1, -1, -1):
            p = active[j]
            
            if p.get('type') == 'active':
                # Active position — standard exit logic
                e = p['limit']  # entry price
                bh = i - p['entry_i']
                
                if bh >= max_bars:
                    pnl = round((cur_close/e - 1)*100 - COMM, 4)
                    p['pnl'] = pnl; p['xt'] = 'TIME'; trades.append(p); del active[j]; continue
                
                if high_arr[i] >= p['tp']:
                    pnl = round(tp - COMM, 4)
                    p['pnl'] = pnl; p['xt'] = 'TP'; trades.append(p); del active[j]; continue
                
                if low_arr[i] <= p['sl']:
                    pnl = round(-sl - COMM, 4)
                    p['pnl'] = pnl; p['xt'] = 'SL'; trades.append(p); del active[j]; continue
                
                if p['pl_ok']:
                    if high_arr[i] > p['peak']:
                        p['peak'] = high_arr[i]
                        p['trail'] = p['peak'] * tr_r
                    if low_arr[i] <= p['trail']:
                        tr_pnl = round((p['trail']/e - 1)*100 - COMM, 4)
                        p['pnl'] = tr_pnl; p['xt'] = 'TRAIL'; trades.append(p); del active[j]
                else:
                    pl_price = e + (p['tp'] - e) * (pl/100)
                    if high_arr[i] >= pl_price:
                        p['pl_ok'] = True
                        p['peak'] = high_arr[i]
                        p['trail'] = p['peak'] * tr_r
            
            else:
                # Pending limit order — check if filled
                p['bars_waiting'] += 1
                
                # Check: did price dip to our limit?
                if low_arr[i] <= p['limit']:
                    # Filled! Convert to active position
                    p['type'] = 'active'
                    p['entry_i'] = i  # entry at this candle
                    p['entry_ns'] = int(ts_arr[i])
                    # Recalculate TP/SL based on actual limit fill
                    p['tp'] = p['limit'] * tp_r
                    p['sl'] = p['limit'] * sl_r
                    p['pl_ok'] = False
                    p['peak'] = p['limit']
                    p['trail'] = p['limit']
                    filled += 1
                    continue
                
                # Check: price ran away (went above signal + 1%)? Missed
                if high_arr[i] > p['signal_price'] * 1.01:
                    missed += 1
                    del active[j]
                    continue
                
                # Check: waited too long (20 bars = 1 hour)?
                if p['bars_waiting'] > 20:
                    missed += 1
                    del active[j]
                    continue
    
    return trades, missed, filled

print("⏳ جمع الإشارات...", flush=True)
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]

# Collect all signals and their data
all_coins_data = {}
for coin in COINS:
    fpath = f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw = json.load(f)
    if len(raw) < 200: continue
    df = pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    df = compute_indicators(df)
    idxs = find_signals(df)
    if len(idxs) > 0:
        all_coins_data[coin] = (df, idxs)
    else:
        del df

print(f"✅ {len(all_coins_data)} عملة بإشارات\n", flush=True)

# ═══════════════ اختبار كل نسبة حد ═══════════════
print(f"{'='*90}")
print(f"📊 أوامر حد — شراء أقل من سعر الإشارة | TP=1.3% SL=0.5%")
print(f"{'='*90}")
print(f"  {'حد':>6} {'إشارات':>7} {'✅نفذت':>7} {'❌فاتت':>7} {'%تنفيذ':>7} {'WR':>7} {'R:R':>6} {'🟢':>6} {'🔴':>6} {'ثابت$':>9} {'سحب':>6}")
print(f"  {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*9} {'─'*6}")

for offset in LIMIT_OFFSETS:
    label = f"{offset:.2f}%" if offset > 0 else "ماركت"
    
    all_trades_list = []
    total_missed = 0
    total_filled = 0
    total_signals = 0
    
    for coin, (df, idxs) in all_coins_data.items():
        total_signals += len(idxs)
        trades, missed, filled = simulate_limit(df, idxs, 1.3, 0.5, 12, 0.02, 4, offset)
        all_trades_list.extend(trades)
        total_missed += missed
        total_filled += filled
    
    if not all_trades_list:
        print(f"  {label:>6} {total_signals:>7,} {0:>7} {total_missed:>7,} 0.0% - - - - - -")
        continue
    
    # Sort by entry time
    all_trades_list.sort(key=lambda t: t.get('entry_ns', 0))
    
    # Portfolio simulation
    eq = float(CAPITAL); peak = float(CAPITAL); mdd = 0.0
    slots = [None]*MAX_POS; executed_pnls = []; skipped = 0
    
    for t in all_trades_list:
        en = t.get('entry_ns', 0)
        xn = t.get('exit_ns', en + 1) if 'exit_ns' not in t else t['exit_ns']
        pnl_pct = t['pnl']
        
        for s in range(MAX_POS):
            if slots[s] is not None:
                sex, spnl = slots[s]
                if sex <= en:
                    eq += eq * 0.5 * (spnl/100)
                    slots[s] = None
                    if eq > peak: peak = eq
                    if eq < peak: mdd = min(mdd, (eq-peak)/peak*100)
        
        free = -1
        for s in range(MAX_POS):
            if slots[s] is None: free = s; break
        if free == -1: skipped += 1; continue
        executed_pnls.append(pnl_pct)
        slots[free] = (xn, pnl_pct)
    
    for s in range(MAX_POS):
        if slots[s] is not None:
            sex, spnl = slots[s]
            eq += eq * 0.5 * (spnl/100)
    
    wins = sum(1 for p in executed_pnls if p > 0)
    losses = len(executed_pnls) - wins
    wr = wins/len(executed_pnls)*100 if executed_pnls else 0
    aw = np.mean([p for p in executed_pnls if p > 0]) if wins else 0
    al = np.mean([p for p in executed_pnls if p <= 0]) if losses else 0
    rr = aw/abs(al) if al != 0 else 0
    
    fixed_pnl = sum(p/100*500 for p in executed_pnls)
    exec_rate = total_filled/total_signals*100 if total_signals else 0
    
    print(f"  {label:>6} {total_signals:>7,} {total_filled:>7,} {total_missed:>7,} {exec_rate:>6.1f}% {wr:>6.1f}% {rr:>5.2f}x {aw:>+5.2f}% {al:>+5.2f}% ${1000+fixed_pnl:>8,.0f} {mdd:>5.1f}%")

print(f"\n  ⚠️ ملاحظة: أوامر الحد تستخدم High/Low داخل الشمعة للتنفيذ")
print(f"  ⚠️ ماركت = سعر الإغلاق (close-only)")
