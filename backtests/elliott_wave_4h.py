#!/usr/bin/env python3
"""Elliott Wave Strategy — 4h timeframe on FET/USDT"""
import ccxt, pandas as pd, numpy as np, os, sys, time
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

CACHE = '/data/trading28/backtests/cache'
SYMBOL = 'FET/USDT'
TF = '4h'

os.makedirs(CACHE, exist_ok=True)
cache_file = f"{CACHE}/{SYMBOL.replace('/','_')}_4h.csv"

# ─── Fetch & Cache ──────────────────────────────────────────────
if not os.path.exists(cache_file):
    print(f"📡 Fetching {SYMBOL} {TF}...", flush=True)
    exchange = ccxt.binance()
    all_candles = []
    years = ['2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01']
    for yr_start in years:
        since = exchange.parse8601(f'{yr_start}T00:00:00Z')
        yr = yr_start[:4]
        print(f"  📥 {yr}...", flush=True)
        while True:
            try:
                candles = exchange.fetch_ohlcv(SYMBOL, TF, since=since, limit=1000)
                if not candles: break
                all_candles.extend(candles)
                since = candles[-1][0] + 1
                last_ts = datetime.fromtimestamp(candles[-1][0]/1000)
                if last_ts.year > int(yr): break
                if len(candles) < 1000: break
            except Exception as e:
                print(f"  ⚠️ retry: {e}", flush=True)
                time.sleep(2)
    
    df = pd.DataFrame(all_candles, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.drop_duplicates(subset='ts').sort_values('ts').reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    print(f"  ✅ Fetched {len(df)} candles", flush=True)
else:
    print(f"📦 Loading cached: {cache_file}", flush=True)
    df = pd.read_csv(cache_file, parse_dates=['ts'])

print(f"📊 Total candles: {len(df)}", flush=True)
print(f"📅 Range: {df['ts'].iloc[0]} → {df['ts'].iloc[-1]}", flush=True)

# ─── Zigzag — detect swing highs/lows ──────────────────────────
print("\n🔍 Computing strategy...", flush=True)

def find_swings(df, lookback=3):
    """3-bar fractal for 4h (equivalent to ~5-bar on 15m)"""
    highs = np.zeros(len(df), dtype=bool)
    lows = np.zeros(len(df), dtype=bool)
    
    for i in range(lookback * 2, len(df)):
        window_h = df['high'].iloc[i - lookback*2 : i + 1]
        mid_idx = i - lookback
        if df['high'].iloc[mid_idx] == window_h.max() and window_h.values.argmax() == lookback:
            highs[i] = True
        
        window_l = df['low'].iloc[i - lookback*2 : i + 1]
        if df['low'].iloc[mid_idx] == window_l.min() and window_l.values.argmin() == lookback:
            lows[i] = True
    
    return highs, lows

swing_h, swing_l = find_swings(df, lookback=3)

# Indicators
df['atr'] = (df['high'] - df['low']).rolling(14).mean()
df['atr_ma20'] = df['atr'].rolling(20).mean()
df['vol_ma20'] = df['volume'].rolling(20).mean()

# ─── Elliott Wave Detection ─────────────────────────────────────
print("🌊 Detecting Wave 1 → 2 → 3 patterns...", flush=True)

wave3_signals = []
swing_indices = np.where(swing_l | swing_h)[0]

for idx, pos in enumerate(swing_indices):
    if pos < 200: continue
    if idx + 2 >= len(swing_indices): continue
    
    s0 = swing_indices[idx]      # Wave 1 start (low)
    s1 = swing_indices[idx + 1]  # Wave 1 end (high)
    s2 = swing_indices[idx + 2]  # Wave 2 end (low)
    
    if not swing_l[s0] or not swing_h[s1] or not swing_l[s2]: continue
    
    wave1_start = df['low'].iloc[s0]
    wave1_end = df['high'].iloc[s1]
    if wave1_end <= wave1_start: continue
    wave1_size = wave1_end - wave1_start
    
    wave2_low = df['low'].iloc[s2]
    if wave2_low <= wave1_start: continue
    
    # Fibonacci retracement: 38% - 62%
    retrace = (wave1_end - wave2_low) / wave1_size
    if retrace < 0.38 or retrace > 0.62: continue
    
    # Scan for breakout above Wave 1 high
    for j in range(s2 + 1, min(s2 + 50, len(df))):  # 50 bars on 4h = ~8 days
        if df['close'].iloc[j] > wave1_end:
            vol_ok = df['volume'].iloc[j] > df['vol_ma20'].iloc[j] * 1.5
            atr_ok = df['atr'].iloc[j] > df['atr_ma20'].iloc[j]
            
            if vol_ok and atr_ok:
                entry_price = df['close'].iloc[j]
                
                # Original params
                sl = wave2_low
                fib_ext = wave1_start + (wave1_size * 1.618)
                
                wave3_signals.append({
                    'entry_idx': j, 'entry_price': entry_price,
                    'sl': sl, 'tp': fib_ext,
                    'wave1_start': wave1_start, 'wave1_end': wave1_end,
                    'wave2_low': wave2_low, 'wave1_size': wave1_size,
                    'retrace_pct': retrace * 100,
                    's0': s0, 's1': s1, 's2': s2,
                })
            break

print(f"  📊 Found {len(wave3_signals)} Wave 3 signals", flush=True)

if len(wave3_signals) == 0:
    print("⚠️ No signals found!", flush=True)
    sys.exit(0)

# ─── Trade Simulation ───────────────────────────────────────────
print("💸 Simulating trades...", flush=True)

fee = 0.001
capital = 500
monthly_limit = 7

wave3_signals.sort(key=lambda x: x['entry_idx'])

trades_completed = []
in_trade = False
trade_exit_idx = 0
current_month = df['ts'].iloc[0].month
current_year = df['ts'].iloc[0].year
monthly_pnl = 0

for sig in wave3_signals:
    entry_idx = sig['entry_idx']
    entry_ts = df['ts'].iloc[entry_idx]
    
    if in_trade and entry_idx < trade_exit_idx:
        continue
    
    em = entry_ts.month; ey = entry_ts.year
    if em != current_month or ey != current_year:
        current_month = em; current_year = ey
        monthly_pnl = 0
    
    if monthly_pnl < -monthly_limit:
        continue
    
    entry_price = sig['entry_price']
    sl_price = sig['sl']
    tp_price = sig['tp']
    
    result = None
    exit_price = None
    exit_idx = entry_idx
    
    # Max hold: 30 bars (5 days on 4h)
    max_hold = 30
    end_idx = min(entry_idx + max_hold, len(df))
    
    for j in range(entry_idx + 1, end_idx):
        if df['low'].iloc[j] <= sl_price:
            result = 'SL'; exit_price = sl_price; exit_idx = j; break
        if df['high'].iloc[j] >= tp_price:
            result = 'TP'; exit_price = tp_price; exit_idx = j; break
    
    if result is None:
        result = 'TIME'; exit_price = df['close'].iloc[end_idx - 1]; exit_idx = end_idx - 1
    
    pnl_pct = (exit_price - entry_price) / entry_price * 100 - (fee * 200)
    dollar_pnl = capital * (pnl_pct / 100)
    
    trades_completed.append({
        **sig, 'result': result, 'exit_price': exit_price, 'exit_idx': exit_idx,
        'pnl_pct': pnl_pct, 'dollar_pnl': dollar_pnl,
        'entry_ts': entry_ts, 'exit_ts': df['ts'].iloc[exit_idx],
    })
    
    in_trade = True
    trade_exit_idx = exit_idx
    monthly_pnl += dollar_pnl

# ─── Results ─────────────────────────────────────────────────────
n = len(trades_completed)
if n == 0: print("❌ No trades", flush=True); sys.exit(0)

wins = [t for t in trades_completed if t['pnl_pct'] > 0]
losses = [t for t in trades_completed if t['pnl_pct'] <= 0]
n_wins, n_losses = len(wins), len(losses)
wr = n_wins / n * 100

total_profit = sum(t['pnl_pct'] for t in wins)
total_loss = abs(sum(t['pnl_pct'] for t in losses))
net_pnl = sum(t['pnl_pct'] for t in trades_completed)
avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

# Max concurrent
events = []
for t in trades_completed:
    events.append((t['entry_idx'], 1)); events.append((t['exit_idx'], -1))
events.sort()
cur, max_cur = 0, 0
for _, delta in events: cur += delta; max_cur = max(max_cur, cur)

# Avg concurrent
df_cur = np.zeros(len(df))
for t in trades_completed: df_cur[t['entry_idx']:t['exit_idx']+1] += 1
avg_cur = df_cur[df_cur > 0].mean() if df_cur.sum() > 0 else 0

risk_pct = np.mean([abs(t['entry_price'] - t['sl']) / t['entry_price'] * 100 for t in trades_completed])

portfolio = 1000
for t in trades_completed: portfolio += portfolio * 0.05 * (t['pnl_pct'] / 100)

tp_count = sum(1 for t in trades_completed if t['result'] == 'TP')
sl_count = sum(1 for t in trades_completed if t['result'] == 'SL')
time_count = sum(1 for t in trades_completed if t['result'] == 'TIME')

print(f"\n{'='*50}")
print(f"  🌊 ELLIOTT WAVE — {SYMBOL} {TF}")
print(f"{'='*50}")
print(f"""
📋 عدد الصفقات: {n}
🟢 صفقات رابحة: {n_wins} | 🔴 صفقات خاسرة: {n_losses}
📈 Win Rate: {wr:.1f}%
💵 إجمالي الربح: +{total_profit:.1f}%
💸 إجمالي الخسارة: -{total_loss:.1f}%
💰 صافي الربح/الخسارة: {net_pnl:+.1f}%
🟢 متوسط الربح للصفقة: +{avg_win:.2f}%
🔴 متوسط الخسارة للصفقة: -{avg_loss:.2f}%
📊 نسبة الربح للخسارة: {rr_ratio:.1f}x
🔄 أقصى صفقات مفتوحة معاً: {max_cur}
📊 متوسط الصفقات المفتوحة: {avg_cur:.1f}
💼 حجم المخاطرة بالصفقة: ~{risk_pct:.1f}% من رأس المال
🏦 المحفظة: $1000 → ${portfolio:,.0f} ({portfolio/1000*100-100:+.1f}%)

🎯 خروج TP: {tp_count} | ⛔ خروج SL: {sl_count} | ⏱️ خروج زمني: {time_count}
""")

if trades_completed:
    retracements = [t['retrace_pct'] for t in trades_completed]
    print(f"📐 متوسط تصحيح Wave 2: {np.mean(retracements):.1f}%")
    print(f"📏 متوسط حجم Wave 1: ${np.mean([t['wave1_size'] for t in trades_completed]):.4f}")
    hold_times = [(t['exit_ts'] - t['entry_ts']).total_seconds() / 3600 for t in trades_completed]
    print(f"⏰ متوسط وقت الصفقة: {np.mean(hold_times):.1f} ساعة")
    
    print(f"\n📋 آخر 5 صفقات:")
    print(f"{'تاريخ':<19} {'نتيجة':<6} {'دخول':>9} {'خروج':>9} {'ربح%':>7} {'تصحيح%':>7}")
    for t in trades_completed[-5:]:
        print(f"{str(t['entry_ts'])[:19]:<19} {t['result']:<6} {t['entry_price']:>9.4f} {t['exit_price']:>9.4f} {t['pnl_pct']:>+6.2f}% {t['retrace_pct']:>6.1f}%")
