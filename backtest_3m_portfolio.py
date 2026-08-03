#!/usr/bin/env python3
"""محاكاة محفظة صحيحة: 2 صفقة متزامنة مع تداخل زمني — 50% لكل صفقة"""
import json, numpy as np, pandas as pd, os, time, gc

COMM = 0.20; TF_MIN = 3; CAPITAL = 1000; MAX_POS = 2
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

CONFIGS = [
    ("⚡ صياد البرق", 1.3, 0.5, 12, 0.02, 4, 0.10, 35, True),
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

# ═══════════════ جمع الصفقات ═══════════════
print("⏳ جمع الصفقات...", flush=True)
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
EXCLUDE = {'ETH','BTC','TRX','XRP','QI','LSK','GLMR','XTZ','YFI',
           'TLM','0G','LA','DYM','VANRY','SENT','VET','COOKIE','HEI','ACT','CKB','AR','RSR','AXS','XEC',
           'LPT','KNC','LTC','SFP','IOST','KAVA','VTHO','ZRX','1INCH','CVX','WAXP','ZIL','VANA','YGG','SUI',
           'STEEM','SOL',
           'HBAR','DYDX','SUSHI','RED','SHIB','BCH','DOGE','GAS','SPELL','ETC','VIRTUAL','UNI','SYRUP','MAV',
           'ARK','IMX','NEO','RLC','LINK','FLOKI','CVC','BOME','SCR','ASTER','BROCCOLI714','ADA','TFUEL','DOT'}
COINS = [c for c in COINS if c not in EXCLUDE]

all_trades = {c[0]: [] for c in CONFIGS}
processed = 0; t0 = time.time()

for coin in COINS:
    fpath = f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw = json.load(f)
    if len(raw) < 200: del raw; continue
    df = pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw
    
    df = compute_indicators(df)
    close_arr = df['close'].values
    ts_arr = df['ts'].values.astype('datetime64[ns]').astype('int64')  # nanoseconds
    
    for name, tp, sl, pl, trail, max_h, whale, rsi, confirm in CONFIGS:
        idxs = find_signals(df, whale, rsi, confirm)
        if len(idxs) == 0: continue
        
        max_bars = int(max_h*60/TF_MIN)
        tp_r = 1+tp/100; sl_r = 1-sl/100; tr_r = 1-trail/100
        active = []; sig_map = dict(zip(idxs, close_arr[idxs]))
        
        for i in range(len(df)):
            cur = close_arr[i]
            if i in sig_map:
                active.append({'symbol':coin,'entry':sig_map[i],
                    'tp':sig_map[i]*tp_r,'sl':sig_map[i]*sl_r,
                    'pl_ok':False,'peak':sig_map[i],'trail':sig_map[i],
                    'entry_i':i,'entry_ns':int(ts_arr[i])})
            for j in range(len(active)-1,-1,-1):
                p = active[j]; e = p['entry']; bh = i-p['entry_i']
                if bh>=max_bars:
                    p['pnl']=round((cur/e-1)*100-COMM,4); p['exit_type']='TIME'; p['exit_ns']=int(ts_arr[i]); all_trades[name].append(p); del active[j]
                elif cur>=p['tp']:
                    p['pnl']=round(tp-COMM,4); p['exit_type']='TP'; p['exit_ns']=int(ts_arr[i]); all_trades[name].append(p); del active[j]
                elif cur<=p['sl']:
                    p['pnl']=round((cur/e-1)*100-COMM,4); p['exit_type']='SL'; p['exit_ns']=int(ts_arr[i]); all_trades[name].append(p); del active[j]
                elif p['pl_ok']:
                    if cur>p['peak']: p['peak']=cur; p['trail']=cur*tr_r
                    if cur<=p['trail']:
                        p['pnl']=round((cur/e-1)*100-COMM,4); p['exit_type']='TRAIL'; p['exit_ns']=int(ts_arr[i]); all_trades[name].append(p); del active[j]
                else:
                    pl_p = e+(p['tp']-e)*(pl/100)
                    if cur>=pl_p: p['pl_ok']=True; p['peak']=cur; p['trail']=cur*tr_r
    
    del df; gc.collect()
    processed += 1
    if processed % 50 == 0:
        print(f"  ⏳ {processed}/{len(COINS)} | {time.time()-t0:.0f}s", flush=True)

print(f"✅ {time.time()-t0:.0f}s\n", flush=True)

# ═══════════════ محاكاة المحفظة مع تداخل زمني ═══════════════
print(f"{'='*80}")
print("📊 محاكاة المحفظة — MAX_POS=2 عالمي | تداخل زمني حقيقي | 50% لكل صفقة")
print(f"{'='*80}")

for name, tp, sl, pl, trail, max_h, whale, rsi, confirm in CONFIGS:
    trades = all_trades[name]
    trades.sort(key=lambda t: t['entry_ns'])
    
    equity = float(CAPITAL)
    peak = float(CAPITAL)
    max_dd = 0.0
    active_slots = [None, None]  # each: (exit_ns, pnl_pct) or None
    executed = 0; skipped = 0
    
    for t in trades:
        entry_ns = t['entry_ns']
        exit_ns = t['exit_ns']
        pnl_pct = t['pnl']
        
        # Free any slots whose position has ended
        for s in range(MAX_POS):
            if active_slots[s] is not None:
                slot_exit_ns, slot_pnl = active_slots[s]
                if slot_exit_ns <= entry_ns:
                    # Position closed, apply PnL
                    pos_cap = equity * 0.5
                    pnl_dollar = pos_cap * (slot_pnl / 100)
                    equity += pnl_dollar
                    active_slots[s] = None
                    
                    if equity > peak: peak = equity
                    dd = (equity - peak) / peak * 100
                    if dd < max_dd: max_dd = dd
        
        # Find free slot
        free = -1
        for s in range(MAX_POS):
            if active_slots[s] is None:
                free = s; break
        
        if free == -1:
            skipped += 1
            continue
        
        executed += 1
        active_slots[free] = (exit_ns, pnl_pct)
    
    # Close remaining positions
    for s in range(MAX_POS):
        if active_slots[s] is not None:
            slot_exit_ns, slot_pnl = active_slots[s]
            pos_cap = equity * 0.5
            pnl_dollar = pos_cap * (slot_pnl / 100)
            equity += pnl_dollar
            active_slots[s] = None
    
    if equity > peak: peak = equity
    dd = (equity - peak) / peak * 100
    if dd < max_dd: max_dd = dd
    
    # Statistics
    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = sum(1 for t in trades if t['pnl'] <= 0)
    wr = wins/len(trades)*100
    total_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    total_loss = sum(t['pnl'] for t in trades if t['pnl'] <= 0)
    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if losses else 0
    rr = avg_win/abs(avg_loss) if avg_loss != 0 else 0
    tp_c = sum(1 for t in trades if t['exit_type'] == 'TP')
    sl_c = sum(1 for t in trades if t['exit_type'] == 'SL')
    tr_c = sum(1 for t in trades if t['exit_type'] == 'TRAIL')
    tm_c = sum(1 for t in trades if t['exit_type'] == 'TIME')
    
    returns = [t['pnl'] for t in trades]
    sharpe = np.mean(returns)/np.std(returns)*np.sqrt(len(returns)) if len(returns)>1 else 0
    days = 122
    annual_ret = ((equity/1000)**(365/days)-1)*100
    
    print(f"\n{'─'*60}")
    print(f"📊 {name}")
    print(f"📋 إجمالي الإشارات: {len(trades):,}")
    print(f"✅ منفذة: {executed:,} | ⏭️ متخطية: {skipped:,}")
    print(f"🟢 ربح: {wins:,} | 🔴 خسارة: {losses:,}")
    print(f"📈 WR: {wr:.1f}%")
    print(f"💵 إجمالي الربح: +{total_profit:.1f}% | 💸 إجمالي الخسارة: {total_loss:.1f}%")
    print(f"💰 صافي (مجموع): {total_profit+total_loss:+.1f}%")
    print(f"🟢 متوسط ربح: +{avg_win:.2f}% | 🔴 متوسط خسارة: {avg_loss:.2f}%")
    print(f"📊 R:R: {rr:.1f}x | 📊 شارپ: {sharpe:.2f}")
    print(f"🏦 محفظة: $1,000 → ${equity:,.0f} (+{(equity/10-100):.1f}%)")
    print(f"📉 أقصى سحب: {max_dd:.1f}%")
    print(f"📈 عائد سنوي: {annual_ret:+.1f}%")
    print(f"🎯 TP:{tp_c:,} 🛑 SL:{sl_c:,} 🐌 TRAIL:{tr_c:,} ⏱️ TIME:{tm_c:,}")
