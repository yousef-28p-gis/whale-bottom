#!/usr/bin/env python3
"""Elliott Wave Strategy — Zigzag + Momentum + Fibonacci on FET/USDT 15m"""
import ccxt, pandas as pd, numpy as np, os, sys, time
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

CACHE = '/data/trading28/backtests/cache'
SYMBOL = 'FET/USDT'

# ─── Fetch & Cache ──────────────────────────────────────────────
os.makedirs(CACHE, exist_ok=True)
cache_file = f"{CACHE}/{SYMBOL.replace('/','_')}_15m.csv"

if not os.path.exists(cache_file):
    print(f"📡 Fetching {SYMBOL} 15m (3 years)...", flush=True)
    exchange = ccxt.binance()
    all_candles = []
    years = ['2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01']
    for yr_start in years:
        since = exchange.parse8601(f'{yr_start}T00:00:00Z')
        yr = yr_start[:4]
        print(f"  📥 {yr}...", flush=True)
        while True:
            try:
                candles = exchange.fetch_ohlcv(SYMBOL, '15m', since=since, limit=1000)
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

# ─── Strategy Computation ───────────────────────────────────────
print("\n🔍 Computing strategy...", flush=True)

# 1. Zigzag — detect swing highs/lows (5-bar fractal)
def find_swings(df, lookback=5):
    """Find swing highs and lows using fractal detection"""
    highs = np.zeros(len(df), dtype=bool)
    lows = np.zeros(len(df), dtype=bool)
    
    for i in range(lookback * 2, len(df)):
        # Swing High: highest of 2L+1 bars
        window = df['high'].iloc[i - lookback*2 : i + 1]
        mid_idx = i - lookback
        mid_val = df['high'].iloc[mid_idx]
        if mid_val == window.max() and window.values.argmax() == lookback:
            highs[i] = True
        
        # Swing Low: lowest of 2L+1 bars
        window_low = df['low'].iloc[i - lookback*2 : i + 1]
        mid_val_low = df['low'].iloc[mid_idx]
        if mid_val_low == window_low.min() and window_low.values.argmin() == lookback:
            lows[i] = True
    
    return highs, lows

df['swing_high'] = False
df['swing_low'] = False

# Use 5-bar lookback (5 candles on each side = 11-candle window)
swing_h, swing_l = find_swings(df, lookback=5)
df['swing_high'] = swing_h
df['swing_low'] = swing_l

# 2. Indicators
df['atr'] = (df['high'] - df['low']).rolling(14).mean()
df['atr_ma20'] = df['atr'].rolling(20).mean()
df['vol_ma20'] = df['volume'].rolling(20).mean()
df['returns'] = df['close'].pct_change()

# ─── Elliott Wave Detection ─────────────────────────────────────
print("🌊 Detecting Wave 1 → 2 → 3 patterns...", flush=True)

# Store identified Wave 3 opportunities
wave3_signals = []

# Walk through candles looking for Wave 1 → 2 → 3 pattern
i = 300  # start after warmup (enough for indicators + MA200)
swing_indices = np.where(df['swing_low'] | df['swing_high'])[0]

for idx, pos in enumerate(swing_indices):
    if pos < 200: continue  # need warmup
    
    # We need: Low → High → Low pattern (Wave 1 up, Wave 2 down)
    if idx + 2 >= len(swing_indices): continue
    
    s0 = swing_indices[idx]      # potential Wave 1 start (low)
    s1 = swing_indices[idx + 1]  # potential Wave 1 end (high)
    s2 = swing_indices[idx + 2]  # potential Wave 2 end (low)
    
    # Must be: Low → High → Low
    if not df['swing_low'].iloc[s0]: continue
    if not df['swing_high'].iloc[s1]: continue
    if not df['swing_low'].iloc[s2]: continue
    
    # Wave 1 must go up
    wave1_start = df['low'].iloc[s0]
    wave1_end = df['high'].iloc[s1]
    if wave1_end <= wave1_start: continue
    wave1_size = wave1_end - wave1_start
    
    # Wave 2 must correct down (but not below Wave 1 start)
    wave2_low = df['low'].iloc[s2]
    if wave2_low <= wave1_start: continue  # invalidated
    
    # Fibonacci retracement: 50% - 78.6% (deeper correction → stronger Wave 3)
    retrace = (wave1_end - wave2_low) / wave1_size
    if retrace < 0.50 or retrace > 0.786: continue
    
    # Now scan candles AFTER Wave 2 low to find breakout above Wave 1 high
    # Entry: close > Wave 1 high + volume spike + ATR confirmed
    for j in range(s2 + 1, min(s2 + 100, len(df))):
        if df['close'].iloc[j] > wave1_end:
            # Entry conditions
            vol_ok = df['volume'].iloc[j] > df['vol_ma20'].iloc[j] * 2
            atr_ok = df['atr'].iloc[j] > df['atr_ma20'].iloc[j]
            
            if vol_ok and atr_ok:
                entry_price = df['close'].iloc[j]
                entry_idx = j
                
                # SL = Wave 2 low
                sl = wave2_low
                
                # TP = max(261.8% Fibonacci extension, 3×ATR)
                fib_2618 = wave1_start + (wave1_size * 2.618)
                atr_tp = entry_price + (df['atr'].iloc[j] * 3)
                fib_ext = max(fib_2618, atr_tp)
                
                wave3_signals.append({
                    'entry_idx': entry_idx,
                    'entry_price': entry_price,
                    'sl': sl,
                    'tp': fib_ext,
                    'wave1_start': wave1_start,
                    'wave1_end': wave1_end,
                    'wave2_low': wave2_low,
                    'wave1_size': wave1_size,
                    'retrace_pct': retrace * 100,
                    'swing0': s0, 'swing1': s1, 'swing2': s2,
                })
            break  # only first breakout

print(f"  📊 Found {len(wave3_signals)} Wave 3 signals", flush=True)

# ─── Trade Simulation ───────────────────────────────────────────
print("💸 Simulating trades (long-only, single position, 7% monthly limit)...", flush=True)

if len(wave3_signals) == 0:
    print("⚠️ No signals found!", flush=True)
    sys.exit(0)

fee = 0.001  # 0.1% per side
capital = 500  # per trade
monthly_limit = 7  # %

# Sort by entry time
wave3_signals.sort(key=lambda x: x['entry_idx'])

trades = []
in_trade = False
trade_exit_idx = 0
month_start_equity = 500  # we track per-trade capital
equity = 500
trade_idx = 0

# Actually let's simulate properly with timeline
# We need to scan from the start of the data and handle overlaps

# Simpler approach: process each signal, skip if still in a trade
trades_completed = []
current_month = df['ts'].iloc[0].month
current_year = df['ts'].iloc[0].year
monthly_start_eq = 500
monthly_pnl = 0

for sig_idx, sig in enumerate(wave3_signals):
    entry_idx = sig['entry_idx']
    entry_ts = df['ts'].iloc[entry_idx]
    
    # Check if we're still in a trade
    if in_trade and entry_idx < trade_exit_idx:
        continue  # skip overlapping signals
    
    # Monthly loss limit check
    em = entry_ts.month; ey = entry_ts.year
    if em != current_month or ey != current_year:
        current_month = em; current_year = ey
        monthly_start_eq = 500
        monthly_pnl = 0
    
    if monthly_pnl < -monthly_limit:
        continue  # skip rest of month
    
    # Simulate this trade
    entry_price = sig['entry_price']
    sl_price = sig['sl']
    tp_price = sig['tp']
    
    result = None
    exit_price = None
    exit_idx = entry_idx
    
    # Max hold: 240 candles (60 hours / 2.5 days)
    max_hold = 240
    end_idx = min(entry_idx + max_hold, len(df))
    
    for j in range(entry_idx + 1, end_idx):
        low_j = df['low'].iloc[j]
        high_j = df['high'].iloc[j]
        
        if low_j <= sl_price:
            result = 'SL'
            exit_price = sl_price
            exit_idx = j
            break
        
        if high_j >= tp_price:
            result = 'TP'
            exit_price = tp_price
            exit_idx = j
            break
    
    if result is None:
        # Timed out — close at market
        result = 'TIME'
        exit_price = df['close'].iloc[end_idx - 1]
        exit_idx = end_idx - 1
    
    # Calculate P&L
    if result == 'SL':
        pnl_pct = (exit_price - entry_price) / entry_price * 100 - (fee * 2 * 100)
    else:
        pnl_pct = (exit_price - entry_price) / entry_price * 100 - (fee * 2 * 100)
    
    dollar_pnl = capital * (pnl_pct / 100)
    
    trades_completed.append({
        **sig,
        'result': result,
        'exit_price': exit_price,
        'exit_idx': exit_idx,
        'pnl_pct': pnl_pct,
        'dollar_pnl': dollar_pnl,
        'entry_ts': entry_ts,
        'exit_ts': df['ts'].iloc[exit_idx],
    })
    
    in_trade = True
    trade_exit_idx = exit_idx
    monthly_pnl += dollar_pnl
    equity += dollar_pnl

# ─── Results ─────────────────────────────────────────────────────
print("\n" + "="*50)
print(f"  🌊 ELLIOTT WAVE STRATEGY — {SYMBOL} 15m")
print("="*50)

n = len(trades_completed)
if n == 0:
    print("❌ No trades simulated", flush=True)
    sys.exit(0)

wins = [t for t in trades_completed if t['pnl_pct'] > 0]
losses = [t for t in trades_completed if t['pnl_pct'] <= 0]
n_wins = len(wins)
n_losses = len(losses)
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
    events.append((t['entry_idx'], 1))
    events.append((t['exit_idx'], -1))
events.sort()
cur, max_cur = 0, 0
for _, delta in events:
    cur += delta
    max_cur = max(max_cur, cur)

# Average concurrent
df_cur = np.zeros(len(df))
for t in trades_completed:
    df_cur[t['entry_idx']:t['exit_idx']+1] += 1
avg_cur = df_cur[df_cur > 0].mean() if df_cur.sum() > 0 else 0

# Risk per trade
risk_pct = np.mean([abs(t['entry_price'] - t['sl']) / t['entry_price'] * 100 for t in trades_completed])

# Portfolio: $1000 with 5% risk per trade
portfolio = 1000
for t in trades_completed:
    portfolio += portfolio * 0.05 * (t['pnl_pct'] / 100)

# Result breakdown
tp_count = sum(1 for t in trades_completed if t['result'] == 'TP')
sl_count = sum(1 for t in trades_completed if t['result'] == 'SL')
time_count = sum(1 for t in trades_completed if t['result'] == 'TIME')

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

# Additional Elliott-specific stats
if trades_completed:
    retracements = [t['retrace_pct'] for t in trades_completed]
    print(f"📐 متوسط نسبة تصحيح Wave 2: {np.mean(retracements):.1f}%")
    print(f"📏 متوسط حجم Wave 1: ${np.mean([t['wave1_size'] for t in trades_completed]):.4f}")
    
    # Average hold time
    hold_times = [(t['exit_ts'] - t['entry_ts']).total_seconds() / 3600 for t in trades_completed]
    print(f"⏰ متوسط وقت الصفقة: {np.mean(hold_times):.1f} ساعة")
    
    # Show last 5 trades
    print(f"\n📋 آخر 5 صفقات:")
    print(f"{'تاريخ':<19} {'نتيجة':<6} {'دخول':>8} {'خروج':>8} {'ربح%':>7} {'تصحيح%':>7}")
    for t in trades_completed[-5:]:
        print(f"{str(t['entry_ts'])[:19]:<19} {t['result']:<6} {t['entry_price']:>8.4f} {t['exit_price']:>8.4f} {t['pnl_pct']:>+6.2f}% {t['retrace_pct']:>6.1f}%")
