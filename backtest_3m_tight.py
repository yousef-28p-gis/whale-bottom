#!/usr/bin/env python3
"""🧪 باك تيست 3m — فلاتر مشددة — إشارات أقل، جودة أعلى"""
import json, numpy as np, pandas as pd, os, time, gc

COMM = 0.20; TF_MIN = 3; CAPITAL = 1000; MAX_POS = 2
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# فلاتر أضيق — كلها بتركيبة الإشارات الأقوى
TESTS = [
    # فلتر 1: حوت أقوى
    ("TP2.5_حوت0.10",        2.5, 2.0, 40, 0.20, 8, 0.10, 50, False),
    ("TP2.0_حوت0.10",        2.0, 1.5, 30, 0.10, 6, 0.10, 50, False),
    ("TP1.5_حوت0.10",        1.5, 1.0, 20, 0.10, 6, 0.10, 50, False),
    # فلتر 2: RSI أضيق
    ("TP2.5_RSI35",          2.5, 2.0, 40, 0.20, 8, 0.05, 35, False),
    ("TP2.0_RSI35",          2.0, 1.5, 30, 0.10, 6, 0.05, 35, False),
    # فلتر 3: تأكيد (شمعة خضراء)
    ("TP2.5_تأكيد",          2.5, 2.0, 40, 0.20, 8, 0.05, 50, True),
    ("TP2.0_تأكيد_حوت0.10",  2.0, 1.5, 30, 0.10, 6, 0.10, 50, True),
    # فلتر 4: مركب (الأقوى)
    ("TP2.5_مركب",           2.5, 2.0, 40, 0.20, 8, 0.10, 35, True),
    ("TP2.0_مركب",           2.0, 1.5, 30, 0.10, 6, 0.10, 35, True),
    ("TP1.5_مركب",           1.5, 1.0, 20, 0.10, 6, 0.10, 35, True),
]

def compute_indicators(df):
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    w = (df['low'].values - df['low_raw'].values) / np.where(df['low_raw'].values!=0, df['low_raw'].values, np.nan) * 100
    df['whale'] = np.clip(w, 0, None)
    vm = df['volume'].rolling(20).mean().values
    df['spike'] = df['volume'].values / np.where(vm!=0, vm, np.nan)
    delta = df['close'].diff().values
    gain = pd.Series(np.where(delta>0, delta, 0)).rolling(14).mean().values
    loss = pd.Series(np.where(delta<0, -delta, 0)).rolling(14).mean().values
    df['rsi'] = 100 - 100/(1 + gain/np.where(loss!=0, loss, np.nan))
    return df

def find_signals(df, whale_min, rsi_max, confirm):
    n = len(df)
    if n < 100: return np.array([], dtype=int)
    whale = df['whale'].values; spike = df['spike'].values; rsi = df['rsi'].values
    mask = (whale >= whale_min) & (spike >= 1.5) & (rsi < rsi_max) & ~np.isnan(whale) & ~np.isnan(spike) & ~np.isnan(rsi)
    mask[:50] = False
    has_prev = np.zeros(n, dtype=bool)
    for shift in [1, 2, 3]:
        s = np.zeros(n, dtype=bool); s[shift:] = mask[:-shift]; has_prev |= s
    mask &= ~has_prev
    if confirm:
        next_green = np.zeros(n, dtype=bool)
        next_green[:-1] = df['close'].values[1:] > df['open'].values[1:]
        mask &= next_green
    return np.where(mask)[0]

def simulate(close_arr, signal_idxs, entry_prices, tp, sl, pl, trail, max_h):
    n = len(close_arr); max_bars = int(max_h*60/TF_MIN)
    tp_r = 1+tp/100; sl_r = 1-sl/100; tr_r = 1-trail/100
    trades = []; active = []; skipped = 0
    sig_map = dict(zip(signal_idxs, entry_prices))
    for i in range(n):
        cur = close_arr[i]
        if i in sig_map:
            if len(active) >= MAX_POS: skipped += 1
            else: active.append({'e':sig_map[i],'tp':sig_map[i]*tp_r,'sl':sig_map[i]*sl_r,'pl_ok':False,'pk':sig_map[i],'tr':sig_map[i],'ei':i})
        for j in range(len(active)-1,-1,-1):
            p = active[j]; e = p['e']; bh = i-p['ei']
            if bh>=max_bars: p['exit']=('TIME',round((cur/e-1)*100-COMM,4)); trades.append(p); del active[j]; continue
            if cur>=p['tp']: p['exit']=('TP',round(tp-COMM,4)); trades.append(p); del active[j]; continue
            if cur<=p['sl']: p['exit']=('SL',round(-sl-COMM,4)); trades.append(p); del active[j]; continue
            if p['pl_ok']:
                if cur>p['pk']: p['pk']=cur; p['tr']=cur*tr_r
                if cur<=p['tr']: p['exit']=('TRAIL',round((p['tr']/e-1)*100-COMM,4)); trades.append(p); del active[j]
            else:
                pl_p = e+(p['tp']-e)*(pl/100)
                if cur>=pl_p: p['pl_ok']=True; p['pk']=cur; p['tr']=cur*tr_r
    return trades, skipped

# ═══════════════ MAIN ═══════════════
print("⏳ تجهيز...", flush=True)
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
print(f"📋 {len(COINS)} عملة\n", flush=True)

all_trades = {t[0]: [] for t in TESTS}
all_skipped = {t[0]: 0 for t in TESTS}
processed = 0; t_total = time.time()

for ci, coin in enumerate(COINS):
    fpath = f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw = json.load(f)
    if len(raw) < 200: del raw; continue
    df = pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw
    
    df = compute_indicators(df)
    close_arr = df['close'].values
    
    for name, tp, sl, pl, trail, max_h, whale, rsi, confirm in TESTS:
        idxs = find_signals(df, whale, rsi, confirm)
        if len(idxs) == 0: continue
        trades, skipped = simulate(close_arr, idxs, close_arr[idxs].tolist(), tp, sl, pl, trail, max_h)
        all_trades[name].extend(trades)
        all_skipped[name] += skipped
    
    del df; gc.collect()
    processed += 1
    if processed % 30 == 0:
        e = time.time()-t_total
        print(f"  ⏳ {processed}/{len(COINS)} | {e:.0f}s | ETA {e/processed*(len(COINS)-processed):.0f}s", flush=True)

elapsed = time.time()-t_total
print(f"\n✅ {processed} عملة | {elapsed:.0f}s\n", flush=True)

# ═══════════════ النتائج ═══════════════
print(f"{'='*80}")
print("📊 نتائج الباك تيست — 3m | فلاتر مشددة | 4 شهور | close-only")
print(f"{'='*80}")

all_results = []
for name, tp, sl, pl, trail, max_h, whale, rsi, confirm in TESTS:
    trades = all_trades[name]
    if not trades: print(f"\n📊 {name}: ❌ 0"); continue
    wins = [t for t in trades if t['exit'][1]>0]; losses = [t for t in trades if t['exit'][1]<=0]
    wr = len(wins)/len(trades)*100
    net = sum(t['exit'][1] for t in trades)
    total_profit = sum(t['exit'][1] for t in wins) if wins else 0
    total_loss = sum(t['exit'][1] for t in losses) if losses else 0
    avg_win = np.mean([t['exit'][1] for t in wins]) if wins else 0
    avg_loss = np.mean([t['exit'][1] for t in losses]) if losses else 0
    rr = avg_win/abs(avg_loss) if avg_loss!=0 else 0
    tp_c = sum(1 for t in trades if t['exit'][0]=='TP')
    sl_c = sum(1 for t in trades if t['exit'][0]=='SL')
    tr_c = sum(1 for t in trades if t['exit'][0]=='TRAIL')
    tm_c = sum(1 for t in trades if t['exit'][0]=='TIME')
    
    # Portfolio — proper compounding, 2% risk per trade to avoid explosion
    equity = CAPITAL
    for t in trades:
        pnl = t['exit'][1]
        equity *= (1 + pnl/100 * 0.02)  # 2% capital at risk per trade
    
    max_dd = 0.0; peak = CAPITAL; eq = CAPITAL
    for t in trades:
        eq *= (1 + t['exit'][1]/100 * 0.02)
        if eq > peak: peak = eq
        dd = (eq-peak)/peak*100
        if dd < max_dd: max_dd = dd
    
    returns_arr = [t['exit'][1] for t in trades]
    sharpe = np.mean(returns_arr)/np.std(returns_arr)*np.sqrt(len(returns_arr)) if len(returns_arr)>1 else 0
    days=122; annual_ret=((equity/1000)**(365/days)-1)*100
    
    all_results.append({
        'name':name,'trades':len(trades),'skipped':all_skipped[name],
        'wins':len(wins),'losses':len(losses),'wr':wr,
        'total_profit':total_profit,'total_loss':total_loss,'net':net,
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
print("⚖️ ملخص (مرتب حسب WR)")
print(f"{'='*80}")
print(f"  {'التجربة':<22} {'صفقات':>6} {'WR':>7} {'R:R':>5} {'محفظة':>9} {'DD':>7}")
print(f"  {'─'*22} {'─'*6} {'─'*7} {'─'*5} {'─'*9} {'─'*7}")
for r in sorted_r:
    print(f"  {r['name']:<22} {r['trades']:>6} {r['wr']:>6.1f}% {r['rr']:>4.1f}x ${r['equity']:>8,.0f} {r['dd']:>6.1f}%")
