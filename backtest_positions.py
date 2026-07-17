#!/usr/bin/env python3 -u
"""بناء كاش OHLCV لـ ٣ أشهر (أبريل-يونيو 2026) — ثم باك تيست"""
import ccxt, numpy as np, pandas as pd, json, os
from collections import defaultdict
from datetime import datetime, timedelta
import time, gc

TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.40; COMM=0.20
CACHE_DIR = '/data/trading28/cache/ohlcv'
os.makedirs(CACHE_DIR, exist_ok=True)
MONTHS = ['2026-04', '2026-05', '2026-06']

exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})
markets = exchange.load_markets()
coins = [s.replace('/USDT','') for s in markets if s.endswith('/USDT') and markets[s]['active']]

print(f'📦 بناء كاش {len(coins)} زوج × {len(MONTHS)} أشهر')
t0 = time.time()

# ── Phase 1: Build cache ──
for sym in coins[:459]:
    sym_t0 = time.time()
    for mon in MONTHS:
        fpath = f'{CACHE_DIR}/{sym}_{mon}.json'
        if os.path.exists(fpath):
            continue  # already cached
        
        # Fetch full month with pagination
        st = datetime.strptime(mon, '%Y-%m')
        if st.month == 12:
            end_st = datetime(st.year + 1, 1, 1)
        else:
            end_st = datetime(st.year, st.month + 1, 1)
        
        since = int(st.timestamp() * 1000)
        end_ts = int(end_st.timestamp() * 1000)
        
        all_candles = []
        fetch_since = since
        for _ in range(6):  # max 6 iterations = 6000 candles
            try:
                candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=fetch_since, limit=1000)
            except:
                break
            if not candles:
                break
            all_candles.extend(candles)
            if candles[-1][0] >= end_ts or len(candles) < 1000:
                break
            fetch_since = candles[-1][0] + 1
        
        if all_candles:
            with open(fpath, 'w') as f:
                json.dump(all_candles, f)
    
    elapsed = time.time() - t0
    if (coins.index(sym) + 1) % 30 == 0:
        print(f'  📦 {coins.index(sym)+1}/{len(coins)} | {elapsed:.0f}ث', flush=True)

print(f'\n✅ تم بناء الكاش! {elapsed:.0f}ث\n', flush=True)

# ── Phase 2: Backtest ──
def load_cached(sym, mon):
    fpath = f'{CACHE_DIR}/{sym}_{mon}.json'
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath) as f:
            data = json.load(f)
    except:
        return None
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    return df

all_trades = []
pairs_done = 0

for sym in coins[:459]:
    for mon in MONTHS:
        df = load_cached(sym, mon)
        if df is None or len(df) < 100:
            continue
        
        # Whale indicator
        LB = 30
        df['lo'] = df['low'].rolling(LB).min()
        df['lc'] = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
        df['sm'] = df['lc'].ewm(span=3, adjust=False).mean()
        df['hi'] = df['sm'].rolling(LB).max()
        df['raw'] = np.where(df['low'] <= df['lo'], (df['sm'] + df['hi'] * 2) / 3, 0)
        df['whale'] = df['raw'].ewm(span=3, adjust=False).mean().fillna(0)
        df['spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.03)
        df['wf'] = df['whale'].rolling(2).mean()
        df['ws'] = df['whale'].rolling(5).mean()
        df['wp'] = df['whale'].rolling(50).max()
        df['str'] = (df['whale'] / df['wp'].replace(0, np.nan) * 100).fillna(0)
        df['vma'] = df['volume'].rolling(20).mean()
        df['entry'] = (df['spike'] & (df['wf'] > df['ws']) & (df['str'] > STR) & (df['volume'] > df['vma'] * 1.0))
        
        # Find + simulate
        for i in range(50, len(df) - 10):
            row = df.iloc[i]
            if not (row['entry'] and float(row['whale']) >= WHALE_MIN):
                continue
            if i + 1 < len(df) and float(df.iloc[i + 1]['whale']) >= 0.35:
                continue
            ps = max(0, i - 96)
            pb = float(df.iloc[ps]['close'])
            ep = float(row['close'])
            if (ep - pb) / pb * 100 >= 0:
                continue
            
            tp_p = ep * (1 + TP / 100)
            sl_p = ep * (1 - SL / 100)
            pl_p = ep + (tp_p - ep) * (PL / 100)
            pl_trig = False
            peak = ep
            trail_p = 0
            for k in range(i + 1, len(df)):
                cur = float(df.iloc[k]['close'])
                h = (k - i) * 0.25
                if h > MH:
                    pnl = round((cur - ep) / ep * 100 - COMM, 4)
                    exit_ = 'TIME'
                    break
                if cur >= tp_p:
                    pnl = round(TP - COMM, 4)
                    exit_ = 'TP'
                    break
                if cur <= sl_p:
                    pnl = round(-SL - COMM, 4)
                    exit_ = 'SL'
                    break
                if not pl_trig and cur >= pl_p:
                    pl_trig = True
                    peak = cur
                    trail_p = cur * (1 - TRAIL / 100)
                if pl_trig:
                    if cur > peak:
                        peak = cur
                        trail_p = cur * (1 - TRAIL / 100)
                    if cur <= trail_p:
                        pnl = round((trail_p - ep) / ep * 100 - COMM, 4)
                        exit_ = 'TRAIL'
                        break
            else:
                pnl = round((float(df.iloc[-1]['close']) - ep) / ep * 100 - COMM, 4)
                exit_ = 'EOD'
            
            all_trades.append({'dt': row['ts'], 'pnl': pnl, 'exit': exit_, 'sym': sym})
    
    pairs_done += 1
    if pairs_done % 50 == 0:
        print(f'  🔍 {pairs_done}/{len(coins)} | {len(all_trades)} صفقة', flush=True)
    del df
    gc.collect()

total_elapsed = time.time() - t0
print(f'\n✅ باك تيست: {len(all_trades)} صفقة | {total_elapsed:.0f}ث\n', flush=True)

# ── Portfolio ──
def simulate_portfolio(trades, max_pos, pos_pct):
    trades_sorted = sorted(trades, key=lambda x: x['dt'])
    capital = 1000.0
    peak = 1000.0
    max_dd = 0.0
    active = []
    skipped, taken = 0, 0
    exec_trades = []
    
    for t in trades_sorted:
        dt = t['dt']
        still_active = []
        for exit_dt, cost, pnl_amt in active:
            if dt >= exit_dt:
                capital += cost + pnl_amt
            else:
                still_active.append((exit_dt, cost, pnl_amt))
        active = still_active
        
        if len(active) >= max_pos:
            skipped += 1
            continue
        
        pos_size = capital * pos_pct / 100
        if capital < pos_size:
            skipped += 1
            continue
        
        pnl_amt = pos_size * t['pnl'] / 100
        capital -= pos_size
        active.append((dt + timedelta(hours=MH), pos_size, pnl_amt))
        taken += 1
        exec_trades.append(t)
        
        equity = capital + sum(pc + pd for _, pc, pd in active)
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
    
    for _, cost, pnl_amt in active:
        capital += cost + pnl_amt
    return capital, max_dd, skipped, taken, exec_trades

# ── Results ──
nets = [t['pnl'] for t in all_trades]
wins = sum(1 for n in nets if n > 0)
exits = defaultdict(int)
for t in all_trades:
    exits[t['exit']] += 1

print(f'📊 إجمالي الإشارات المؤهلة (أشهر كاملة): {len(all_trades)}')
print(f'🟢 رابحة: {wins} | 🔴 خاسرة: {len(all_trades) - wins}')
print(f'📈 WR: {wins / len(all_trades) * 100:.1f}%')
print(f'🎯 TP={exits.get("TP", 0)} 🛑 SL={exits.get("SL", 0)} 🐌 TRAIL={exits.get("TRAIL", 0)} ⏰ TIME={exits.get("TIME", 0) + exits.get("EOD", 0)}')
print()

months_ar = {4: 'أبريل', 5: 'مايو', 6: 'يونيو'}

for max_pos, pos_pct in [(2, 50), (3, 33)]:
    cap, dd, skipped, taken, exec_trades = simulate_portfolio(all_trades, max_pos, pos_pct)
    exec_nets = [t['pnl'] for t in exec_trades]
    exec_wins = sum(1 for n in exec_nets if n > 0)
    monthly_return = (cap / 1000) ** (1 / 3) - 1
    
    label = '🐋 صفقتين × 50%' if max_pos == 2 else '🐋 ٣ صفقات × 33%'
    print(f'{"=" * 60}')
    print(f'{label}')
    print(f'{"=" * 60}')
    print(f'📋 إشارات مؤهلة: {len(all_trades)}')
    print(f'✅ منفذة: {taken} | ⏭️ متخطية: {skipped}')
    print(f'🟢 رابحة: {exec_wins} | 🔴 خاسرة: {taken - exec_wins}')
    if taken > 0:
        print(f'📈 WR: {exec_wins / taken * 100:.1f}%')
    print(f'💼 محفظة: $1000 → ${cap:.0f} ({cap / 10 - 100:+.1f}%)')
    print(f'📈 عائد شهري: {monthly_return * 100:+.1f}%')
    print(f'📉 أقصى سحب: {dd:.2f}%')
    
    for m in [4, 5, 6]:
        mt = [t for t in exec_trades if t['dt'].month == m]
        if not mt:
            continue
        mw = sum(1 for t in mt if t['pnl'] > 0)
        print(f'  {months_ar[m]}: {len(mt)} صفقة | WR {mw / len(mt) * 100:.0f}%')
