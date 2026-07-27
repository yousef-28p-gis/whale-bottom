#!/usr/bin/env python3
"""🧪 باك تيست 3m — vectorized + numpy arrays — 212 عملة × 4 شهور — close-only"""
import json, numpy as np, pandas as pd, os, time, gc

COMM = 0.20; TF_MIN = 3; CAPITAL = 1000; MAX_POS = 2
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

TESTS = [
    ("TP2.5_الأساسي",    2.5, 2.0, 40, 0.20, 8, 0.05, 50, False),
    ("TP2.0_أضيق",       2.0, 1.5, 30, 0.10, 6, 0.05, 50, False),
    ("TP1.5_أضيق",       1.5, 1.0, 20, 0.10, 6, 0.05, 50, False),
    ("TP2.5_تريل0.1",    2.5, 2.0, 40, 0.10, 8, 0.05, 50, False),
    ("TP2.0_تأكيد",      2.0, 1.5, 30, 0.10, 6, 0.05, 50, True),
    ("TP2.0_فلاترأضيق",  2.0, 1.0, 25, 0.10, 6, 0.10, 35, False),
    ("TP3.0_واسع",       3.0, 2.0, 40, 0.15, 8, 0.05, 50, False),
    ("TP1.5_SL1_PL15",   1.5, 1.0, 15, 0.10, 4, 0.05, 50, False),
]

def compute_indicators(df):
    """Vectorized indicators on full DataFrame"""
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    whale = (df['low'].values - df['low_raw'].values) / np.where(df['low_raw'].values!=0, df['low_raw'].values, np.nan) * 100
    df['whale'] = np.clip(whale, 0, None)
    vol_ma = df['volume'].rolling(20).mean().values
    df['spike'] = df['volume'].values / np.where(vol_ma!=0, vol_ma, np.nan)
    delta = df['close'].diff().values
    gain = np.where(delta>0, delta, 0)
    loss = np.where(delta<0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = avg_gain / np.where(avg_loss!=0, avg_loss, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def find_signals_fast(df, whale_min, rsi_max, confirm):
    """Vectorized signal detection"""
    n = len(df)
    if n < 100: return np.array([], dtype=int)
    
    whale = df['whale'].values
    spike = df['spike'].values
    rsi = df['rsi'].values
    
    # Base mask
    mask = (whale >= whale_min) & (spike >= 1.5) & (rsi < rsi_max) & ~np.isnan(whale) & ~np.isnan(spike) & ~np.isnan(rsi)
    mask[:50] = False
    
    # Remove consecutive (تباعد 3 شمعات)
    # Mark indices that have an entry within previous 3 bars
    has_prev = np.zeros(n, dtype=bool)
    for shift in [1, 2, 3]:
        shifted = np.zeros(n, dtype=bool)
        shifted[shift:] = mask[:-shift]
        has_prev |= shifted
    mask &= ~has_prev
    
    # Confirmation filter
    if confirm:
        opens = df['open'].values
        closes = df['close'].values
        next_green = np.zeros(n, dtype=bool)
        next_green[:-1] = closes[1:] > opens[1:]
        mask &= next_green
    
    return np.where(mask)[0]

def simulate_numpy(close_arr, signal_idxs, entry_prices, tp, sl, pl, trail, max_h):
    """Fast simulation using numpy arrays"""
    n = len(close_arr)
    max_bars = int(max_h * 60 / TF_MIN)
    tp_ratio = 1 + tp/100; sl_ratio = 1 - sl/100; trail_ratio = 1 - trail/100
    
    trades = []; active = []; skipped = 0
    
    # Map signal indices to entry prices
    sig_map = dict(zip(signal_idxs, entry_prices))
    
    for i in range(n):
        current = close_arr[i]
        
        # New entries
        if i in sig_map:
            entry = sig_map[i]
            if len(active) >= MAX_POS:
                skipped += 1
            else:
                active.append({
                    'entry': entry,
                    'tp': entry * tp_ratio,
                    'sl': entry * sl_ratio,
                    'pl_triggered': False, 'peak': entry,
                    'trail_price': entry,
                    'entry_idx': i,
                })
        
        # Check exits (iterate in reverse to allow removal)
        for j in range(len(active)-1, -1, -1):
            pos = active[j]
            entry = pos['entry']; bars_held = i - pos['entry_idx']
            
            if bars_held >= max_bars:
                pnl = round((current/entry - 1)*100 - COMM, 4)
                pos['exit'] = ('TIME', pnl)
                trades.append(pos); del active[j]; continue
            
            if current >= pos['tp']:
                pos['exit'] = ('TP', round(tp - COMM, 4))
                trades.append(pos); del active[j]; continue
            
            if current <= pos['sl']:
                pos['exit'] = ('SL', round(-sl - COMM, 4))
                trades.append(pos); del active[j]; continue
            
            if pos['pl_triggered']:
                if current > pos['peak']:
                    pos['peak'] = current
                    pos['trail_price'] = current * trail_ratio
                if current <= pos['trail_price']:
                    tr_pnl = round((pos['trail_price']/entry - 1)*100 - COMM, 4)
                    pos['exit'] = ('TRAIL', tr_pnl)
                    trades.append(pos); del active[j]
            else:
                pl_price = entry + (pos['tp'] - entry) * (pl / 100)
                if current >= pl_price:
                    pos['pl_triggered'] = True
                    pos['peak'] = current
                    pos['trail_price'] = current * trail_ratio
    
    return trades, skipped

def calc_portfolio(trades):
    equity = CAPITAL; peak_eq = CAPITAL; max_dd = 0.0
    for t in trades:
        pnl = t['exit'][1]
        pos_cap = equity / MAX_POS
        equity += pos_cap * (pnl / 100)
        if equity > peak_eq: peak_eq = equity
        dd = (equity - peak_eq) / peak_eq * 100
        if dd < max_dd: max_dd = dd
    return equity, max_dd

# ═══════════════ MAIN ═══════════════
print("⏳ تجهيز...", flush=True)
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
print(f"📋 {len(COINS)} عملة\n", flush=True)

# Accumulators
all_test_trades = {t[0]: [] for t in TESTS}
all_skipped = {t[0]: 0 for t in TESTS}
total_candles = 0; processed = 0; t_total = time.time()

for ci, coin in enumerate(COINS):
    fpath = f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    
    with open(fpath) as f: raw = json.load(f)
    total_candles += len(raw)
    
    if len(raw) < 200:
        del raw; continue
    
    # Build DataFrame
    df = pd.DataFrame(raw)
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    
    # Compute indicators once
    t_coin = time.time()
    df = compute_indicators(df)
    
    # Get close array for simulation
    close_arr = df['close'].values
    
    # Run each test config
    for name, tp, sl, pl, trail, max_h, whale, rsi, confirm in TESTS:
        signal_idxs = find_signals_fast(df, whale, rsi, confirm)
        if len(signal_idxs) == 0: continue
        entry_prices = close_arr[signal_idxs].tolist()
        trades, skipped = simulate_numpy(close_arr, signal_idxs, entry_prices, tp, sl, pl, trail, max_h)
        all_test_trades[name].extend(trades)
        all_skipped[name] += skipped
    
    dt = time.time() - t_coin
    del df; gc.collect()
    processed += 1
    
    if processed % 20 == 0:
        elapsed = time.time()-t_total
        eta = elapsed/processed*(len(COINS)-processed)
        print(f"  ⏳ {processed}/{len(COINS)} | {elapsed:.0f}s | ETA {eta:.0f}s | ~{dt:.1f}s/coin", flush=True)

elapsed = time.time()-t_total
print(f"\n✅ {processed} عملة | {total_candles:,} شمعة | {elapsed:.0f}s\n", flush=True)

# ═══════════════ النتائج ═══════════════
print(f"{'='*80}")
print("📊 نتائج الباك تيست — 3m | 4 شهور | close-only | تكوين حقيقي")
print(f"{'='*80}")

all_results = []
for name, tp, sl, pl, trail, max_h, whale, rsi, confirm in TESTS:
    trades = all_test_trades[name]
    if not trades:
        print(f"\n📊 {name}: ❌ 0 صفقات"); continue
    
    wins = [t for t in trades if t['exit'][1]>0]
    losses = [t for t in trades if t['exit'][1]<=0]
    wr = len(wins)/len(trades)*100
    total_net = sum(t['exit'][1] for t in trades)
    total_profit = sum(t['exit'][1] for t in wins) if wins else 0
    total_loss = sum(t['exit'][1] for t in losses) if losses else 0
    avg_win = np.mean([t['exit'][1] for t in wins]) if wins else 0
    avg_loss = np.mean([t['exit'][1] for t in losses]) if losses else 0
    rr = avg_win/abs(avg_loss) if avg_loss!=0 else 0
    tp_c = sum(1 for t in trades if t['exit'][0]=='TP')
    sl_c = sum(1 for t in trades if t['exit'][0]=='SL')
    tr_c = sum(1 for t in trades if t['exit'][0]=='TRAIL')
    tm_c = sum(1 for t in trades if t['exit'][0]=='TIME')
    equity, max_dd = calc_portfolio(trades)
    returns_arr = [t['exit'][1] for t in trades]
    sharpe = np.mean(returns_arr)/np.std(returns_arr)*np.sqrt(len(returns_arr)) if len(returns_arr)>1 else 0
    days=122; annual_ret=((equity/1000)**(365/days)-1)*100
    
    all_results.append({
        'name':name,'trades':len(trades),'skipped':all_skipped[name],
        'wins':len(wins),'losses':len(losses),'wr':wr,
        'total_profit':total_profit,'total_loss':total_loss,'net':total_net,
        'avg_win':avg_win,'avg_loss':avg_loss,'rr':rr,
        'tp_c':tp_c,'sl_c':sl_c,'tr_c':tr_c,'tm_c':tm_c,
        'sharpe':sharpe,'dd':max_dd,'equity':equity,'annual':annual_ret,
    })

sorted_r = sorted(all_results, key=lambda x: x['wr'], reverse=True)

for r in sorted_r:
    print(f"\n{'─'*60}")
    print(f"📊 {r['name']}")
    print(f"📋 صفقات: {r['trades']} | ⏭️ متخطية: {r['skipped']}")
    print(f"🟢 ربح: {r['wins']} | 🔴 خسارة: {r['losses']}")
    print(f"📈 WR: {r['wr']:.1f}%")
    print(f"💵 إجمالي الربح: +{r['total_profit']:.1f}% | 💸 إجمالي الخسارة: {r['total_loss']:.1f}%")
    print(f"💰 صافي: {r['net']:+.1f}%")
    print(f"🟢 متوسط ربح: +{r['avg_win']:.2f}% | 🔴 متوسط خسارة: {r['avg_loss']:.2f}%")
    print(f"📊 R:R: {r['rr']:.1f}x | 📊 شارپ: {r['sharpe']:.2f} | 📉 سحب: {r['dd']:.1f}%")
    print(f"🏦 محفظة: $1,000 → ${r['equity']:,.0f} (+{(r['equity']/10-100):.1f}%)")
    print(f"📈 عائد سنوي: {r['annual']:+.1f}%")
    print(f"🎯 TP:{r['tp_c']} 🛑 SL:{r['sl_c']} 🐌 TRAIL:{r['tr_c']} ⏰ TIME:{r['tm_c']}")

print(f"\n{'='*80}")
print("⚖️ ملخص مضغوط (مرتب حسب WR)")
print(f"{'='*80}")
print(f"  {'التجربة':<20} {'صفقات':>6} {'WR':>7} {'صافي':>8} {'محفظة':>9} {'DD':>7} {'سنوي':>8} {'شارپ':>6} {'R:R':>5}")
print(f"  {'─'*20} {'─'*6} {'─'*7} {'─'*8} {'─'*9} {'─'*7} {'─'*8} {'─'*6} {'─'*5}")
for r in sorted_r:
    print(f"  {r['name']:<20} {r['trades']:>6} {r['wr']:>6.1f}% {r['net']:>+7.1f}% ${r['equity']:>8,.0f} {r['dd']:>6.1f}% {r['annual']:>+7.1f}% {r['sharpe']:>6.2f} {r['rr']:>4.1f}x")
