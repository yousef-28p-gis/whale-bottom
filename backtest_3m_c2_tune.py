#!/usr/bin/env python3
"""🧪 تحسين C2 — 15 تكوين — دخول قوي جداً — close-only"""
import json, numpy as np, pandas as pd, os, time, gc

COMM = 0.20; TF_MIN = 3; CAPITAL = 1000
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

def compute_indicators_full(df):
    n = len(df)
    close = df['close'].values; high = df['high'].values; low = df['low'].values
    vol = df['volume'].values
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    w = (low - df['low_raw'].values) / np.where(df['low_raw'].values!=0, df['low_raw'].values, np.nan) * 100
    df['whale'] = np.clip(w, 0, None)
    vol_ma = df['volume'].rolling(20).mean().values
    df['spike'] = vol / np.where(vol_ma!=0, vol_ma, np.nan)
    delta = df['close'].diff().values
    gain = np.where(delta>0, delta, 0); loss = np.where(delta<0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = avg_gain / np.where(avg_loss!=0, avg_loss, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    tr1 = high - low; tr2 = np.abs(high - np.roll(close, 1)); tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    df['atr'] = pd.Series(tr).rolling(14).mean().values
    return df

def find_signals(df, whale_min, rsi_max, spike_min):
    n = len(df)
    if n < 100: return np.array([], dtype=int)
    whale = df['whale'].values; spike = df['spike'].values; rsi = df['rsi'].values
    mask = (whale >= whale_min) & (spike >= spike_min) & (rsi < rsi_max)
    mask &= ~np.isnan(whale) & ~np.isnan(spike) & ~np.isnan(rsi)
    mask[:50] = False
    has_prev = np.zeros(n, dtype=bool)
    for shift in [1,2,3]:
        shifted = np.zeros(n, dtype=bool)
        shifted[shift:] = mask[:-shift]
        has_prev |= shifted
    mask &= ~has_prev
    return np.where(mask)[0]

def simulate(close_arr, signal_idxs, entry_prices, tp, sl, pl, trail, max_h, max_pos):
    n = len(close_arr); max_bars = int(max_h * 60 / TF_MIN)
    tp_ratio = 1 + tp/100; trail_ratio = 1 - trail/100
    trades = []; active = []; skipped = 0
    sig_map = dict(zip(signal_idxs, entry_prices))
    
    for i in range(n):
        current = close_arr[i]
        if i in sig_map:
            entry = sig_map[i]
            if len(active) >= max_pos:
                skipped += 1
            else:
                dyn_sl = entry * (1 - sl/100) if sl < 90 else 0.0001
                active.append({
                    'entry': entry, 'tp': entry * tp_ratio, 'sl': dyn_sl,
                    'pl_triggered': False, 'peak': entry, 'trail_price': entry,
                    'entry_idx': i,
                })
        
        for j in range(len(active)-1, -1, -1):
            pos = active[j]; entry = pos['entry']; bars_held = i - pos['entry_idx']
            if bars_held >= max_bars:
                pnl = round((current/entry - 1)*100 - COMM, 4)
                pos['exit'] = ('TIME', pnl)
                trades.append(pos); del active[j]; continue
            if current >= pos['tp']:
                actual = round((pos['tp']/entry - 1)*100 - COMM, 4)
                pos['exit'] = ('TP', actual)
                trades.append(pos); del active[j]; continue
            if sl < 90 and current <= pos['sl']:
                actual = round((current/entry - 1)*100 - COMM, 4)
                pos['exit'] = ('SL', actual)
                trades.append(pos); del active[j]; continue
            if pos['pl_triggered']:
                if current > pos['peak']:
                    pos['peak'] = current; pos['trail_price'] = current * trail_ratio
                if current <= pos['trail_price']:
                    tr_pnl = round((pos['trail_price']/entry - 1)*100 - COMM, 4)
                    pos['exit'] = ('TRAIL', tr_pnl)
                    trades.append(pos); del active[j]
            else:
                pl_price = entry + (pos['tp'] - entry) * (pl / 100)
                if current >= pl_price:
                    pos['pl_triggered'] = True; pos['peak'] = current
                    pos['trail_price'] = current * trail_ratio
    return trades, skipped

def calc_portfolio(trades, max_pos):
    equity = CAPITAL; peak_eq = CAPITAL; max_dd = 0.0
    for t in trades:
        pnl = t['exit'][1]; pos_cap = equity / max_pos
        equity += pos_cap * (pnl / 100)
        if equity > peak_eq: peak_eq = equity
        dd = (equity - peak_eq) / peak_eq * 100
        if dd < max_dd: max_dd = dd
    return equity, max_dd

# ═══════════════ 15 تكوين — كلها دخول C2 (WHALE≥0.25 RSI<25 SPK≥2.0) ═══════════════
# (name, tp, sl, pl, trail, max_h, max_pos)
# sl=99 → بدون SL

TESTS = [
    # Group 1: C2 baseline + TP/SL variants
    ("1_C2base_TP2_SL1.5_TR05_6h",   2.0, 1.5, 30, 0.05, 6, 1),
    ("2_TP1.5_SL1.0_TR05_6h",        1.5, 1.0, 30, 0.05, 6, 1),
    ("3_TP2.5_SL1.5_TR05_6h",        2.5, 1.5, 30, 0.05, 6, 1),
    ("4_TP2.0_SL2.0_TR05_6h",        2.0, 2.0, 30, 0.05, 6, 1),
    ("5_TP2.5_SL2.0_TR05_6h",        2.5, 2.0, 30, 0.05, 6, 1),
    ("6_TP3.0_SL2.0_TR05_8h",        3.0, 2.0, 30, 0.05, 8, 1),
    
    # Group 2: Trail variants
    ("7_TR03_TP2_SL1.5_6h",          2.0, 1.5, 30, 0.03, 6, 1),
    ("8_TR08_TP2_SL1.5_6h",          2.0, 1.5, 30, 0.08, 6, 1),
    ("9_TR10_TP2_SL1.5_6h",          2.0, 1.5, 30, 0.10, 6, 1),
    
    # Group 3: No SL (trail+time only)
    ("10_NoSL_TP1.5_TR05_4h",        1.5, 99, 30, 0.05, 4, 1),
    ("11_NoSL_TP2.0_TR05_6h",        2.0, 99, 30, 0.05, 6, 1),
    ("12_NoSL_TP2.5_TR08_6h",        2.5, 99, 30, 0.08, 6, 1),
    ("13_NoSL_TP2.0_TR10_8h",        2.0, 99, 30, 0.10, 8, 1),
    
    # Group 4: Short time + MP2
    ("14_TP2_SL1.5_TR05_4h",         2.0, 1.5, 30, 0.05, 4, 1),
    ("15_TP2_SL1.5_TR05_6h_MP2",     2.0, 1.5, 30, 0.05, 6, 2),
]

# ═══════════════ MAIN ═══════════════
print("⏳ تجهيز...", flush=True)
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
print(f"📋 {len(COINS)} عملة | 🔬 15 تكوين | 🎯 دخول: WHALE≥0.25 RSI<25 SPK≥2.0\n", flush=True)

all_test_trades = {t[0]: [] for t in TESTS}
all_skipped = {t[0]: 0 for t in TESTS}
total_candles = 0; processed = 0; t_total = time.time()

for ci, coin in enumerate(COINS):
    fpath = f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw = json.load(f)
    total_candles += len(raw)
    if len(raw) < 200: del raw; continue
    
    df = pd.DataFrame(raw)
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw
    df = compute_indicators_full(df)
    close_arr = df['close'].values
    
    signal_idxs = find_signals(df, 0.25, 25, 2.0)  # C2 entry
    if len(signal_idxs) == 0: del df; gc.collect(); processed += 1; continue
    entry_prices = close_arr[signal_idxs].tolist()
    
    for name, tp, sl, pl, trail, max_h, max_pos in TESTS:
        trades, skipped = simulate(close_arr, signal_idxs, entry_prices, tp, sl, pl, trail, max_h, max_pos)
        all_test_trades[name].extend(trades)
        all_skipped[name] += skipped
    
    del df; gc.collect(); processed += 1
    if processed % 30 == 0:
        elapsed = time.time()-t_total
        eta = elapsed/processed*(len(COINS)-processed)
        print(f"  ⏳ {processed}/{len(COINS)} | {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)

elapsed = time.time()-t_total
print(f"\n✅ {processed} عملة | {total_candles:,} شمعة | {elapsed:.0f}s\n", flush=True)

# ═══════════════ النتائج ═══════════════
print(f"{'='*90}")
print("📊 نتائج تحسين C2 — دخول WHALE≥0.25 RSI<25 SPK≥2.0 — close-only")
print(f"{'='*90}")

all_results = []
for name, tp, sl, pl, trail, max_h, max_pos in TESTS:
    trades = all_test_trades[name]
    if not trades: print(f"\n📊 {name}: ❌ 0 صفقات"); continue
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
    equity, max_dd = calc_portfolio(trades, max_pos)
    returns_arr = [t['exit'][1] for t in trades]
    sharpe = np.mean(returns_arr)/np.std(returns_arr)*np.sqrt(len(returns_arr)) if len(returns_arr)>1 else 0
    days=122; annual_ret=((equity/1000)**(365/days)-1)*100
    
    sl_type = "بدون" if sl==99 else f"{sl}%"
    
    all_results.append({
        'name':name,'trades':len(trades),'skipped':all_skipped[name],
        'wins':len(wins),'losses':len(losses),'wr':wr,
        'total_profit':total_profit,'total_loss':total_loss,'net':total_net,
        'avg_win':avg_win,'avg_loss':avg_loss,'rr':rr,
        'tp_c':tp_c,'sl_c':sl_c,'tr_c':tr_c,'tm_c':tm_c,
        'sharpe':sharpe,'dd':max_dd,'equity':equity,'annual':annual_ret,
        'max_pos':max_pos,'sl_type':sl_type,'tp_val':tp,'trail':trail,'max_h':max_h,
    })

sorted_r = sorted(all_results, key=lambda x: x['equity'], reverse=True)

for r in sorted_r:
    print(f"\n{'─'*70}")
    print(f"📊 {r['name']}")
    print(f"   ⚙️ TP={r['tp_val']}% | SL={r['sl_type']} | TRAIL={r['trail']}% | TIME={r['max_h']}h | MP={r['max_pos']}")
    print(f"📋 صفقات: {r['trades']} | ⏭️ متخطية: {r['skipped']}")
    print(f"🟢 ربح: {r['wins']} | 🔴 خسارة: {r['losses']}")
    print(f"📈 WR: {r['wr']:.1f}%")
    print(f"💵 ربح: +{r['total_profit']:.1f}% | 💸 خسارة: {r['total_loss']:.1f}%")
    print(f"💰 صافي: {r['net']:+.1f}%")
    print(f"🟢 متوسط ربح: +{r['avg_win']:.2f}% | 🔴 متوسط خسارة: {r['avg_loss']:.2f}%")
    print(f"📊 R:R: {r['rr']:.1f}x | شارپ: {r['sharpe']:.2f} | 📉 سحب: {r['dd']:.1f}%")
    print(f"🏦 محفظة: $1,000 → ${r['equity']:,.0f} (+{(r['equity']/10-100):.1f}%)")
    print(f"📈 سنوي: {r['annual']:+.1f}%")
    print(f"🎯 TP:{r['tp_c']} 🛑 SL:{r['sl_c']} 🐌 TRAIL:{r['tr_c']} ⏰ TIME:{r['tm_c']}")

print(f"\n{'='*90}")
print("⚖️ ملخص مضغوط (مرتب حسب المحفظة)")
print(f"{'='*90}")
print(f"  {'التجربة':<28} {'صفقات':>6} {'WR':>7} {'صافي':>8} {'محفظة':>9} {'DD':>7} {'سنوي':>8} {'شارپ':>6} {'R:R':>5}")
print(f"  {'─'*28} {'─'*6} {'─'*7} {'─'*8} {'─'*9} {'─'*7} {'─'*8} {'─'*6} {'─'*5}")
for r in sorted_r:
    print(f"  {r['name']:<28} {r['trades']:>6} {r['wr']:>6.1f}% {r['net']:>+7.1f}% ${r['equity']:>8,.0f} {r['dd']:>6.1f}% {r['annual']:>+7.1f}% {r['sharpe']:>6.2f} {r['rr']:>4.1f}x")
