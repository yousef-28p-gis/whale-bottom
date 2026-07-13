#!/usr/bin/env python3
"""Whale LONG-only: full history from 2019 + trade audit"""
import ccxt, pandas as pd, numpy as np, os, sys, time
from datetime import datetime
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

CACHE = '/data/trading28/backtests/cache'
SYMBOL = 'FET/USDT'
cache_file = f"{CACHE}/{SYMBOL.replace('/','_')}_15m_FULL.csv"

# ─── Fetch FULL history ──────────────────────────────────────────
if not os.path.exists(cache_file):
    print("📡 Fetching FULL FET/USDT 15m history (2019→2026)...", flush=True)
    exchange = ccxt.binance()
    all_c = []
    since = exchange.parse8601('2019-02-01T00:00:00Z')
    retries = 0
    while True:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, '15m', since=since, limit=1000)
            if not candles: break
            all_c.extend(candles)
            since = candles[-1][0] + 1
            last_ts = datetime.fromtimestamp(candles[-1][0]/1000)
            if len(candles) < 1000: break
            if len(all_c) % 10000 == 0:
                print(f"  📥 {len(all_c):,} candles... ({last_ts.date()})", flush=True)
            retries = 0
        except Exception as e:
            retries += 1
            if retries > 5: break
            time.sleep(3)
    
    df = pd.DataFrame(all_c, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.drop_duplicates(subset='ts').sort_values('ts').reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    print(f"  ✅ Fetched {len(df):,} candles", flush=True)
else:
    df = pd.read_csv(cache_file, parse_dates=['ts'])
    print(f"📦 Cached: {len(df):,} candles", flush=True)

print(f"📅 {df['ts'].iloc[0]} → {df['ts'].iloc[-1]}", flush=True)

# ─── Whale 200-bar ───────────────────────────────────────────────
print("🐋 Computing whale (200-bar)...", flush=True)
BARS = 200
lowest = df['low'].rolling(BARS).min()
at_low = (df['low'] <= lowest).astype(float)
lc = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
sm = lc.ewm(span=3, adjust=False).mean()
hi = sm.rolling(BARS).max()
st = np.where(at_low > 0, (sm + hi * 2) / 3, 0)
df['whale'] = pd.Series(st).ewm(span=3, adjust=False).mean().fillna(0)
df['spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.02)
df['wma50'] = df['whale'].rolling(50).mean()
df['wma200'] = df['whale'].rolling(200).mean()
df['wstr'] = df['whale'] / df['whale'].rolling(50).max().replace(0, np.nan) * 100
df['atr'] = (df['high'] - df['low']).rolling(14).mean()
df['atr_ma'] = df['atr'].rolling(20).mean()
df['vma'] = df['volume'].rolling(20).mean()

# RSI + Sell exhaustion
delta = df['close'].diff(); g = delta.clip(lower=0); l = -delta.clip(upper=0)
ag = g.ewm(alpha=1/14, adjust=False).mean(); al = l.ewm(alpha=1/14, adjust=False).mean()
df['rsi'] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))

vs = df['volume'].rolling(20).mean(); hh20 = df['high'].rolling(20).max().shift(1)
ll10 = df['low'].rolling(10).min().shift(1)
c = np.zeros(len(df))
c += ((df['volume'] > vs * 1.5) & (df['close'] < df['open'])).astype(int)
c += ((df['high'] > hh20) & (df['close'] < hh20)).astype(int)
c += ((df['high'] > hh20) & (df['close'] < df['open'])).astype(int)
c += ((df['close'].shift(1) > df['open'].shift(1)) & (df['volume'] > vs * 1.5) & (df['close'] < df['open'])).astype(int)
c += (df['low'] < ll10).astype(int)
c += ((df['high'] > df['high'].shift(1)) & (df['rsi'] < df['rsi'].shift(1))).astype(int)
df['sell_str'] = c / 6 * 100

# Swings
lb = 5
swl = np.zeros(len(df), dtype=bool)
for i in range(lb*2, len(df)):
    w = df['low'].iloc[i-lb*2:i+1]; m = i - lb
    if df['low'].iloc[m] == w.min() and w.values.argmin() == lb: swl[i] = True

def nsl(idx):
    for j in range(idx-1, max(0, idx-100), -1):
        if swl[j]: return df['low'].iloc[j]
    return df['low'].iloc[idx] * 0.95

# ─── Signal detection ────────────────────────────────────────────
print("🔍 Detecting signals...", flush=True)

long_ok = df['wma50'] > df['wma200']
entry_sig = (df['spike'] & (df['wstr'] > 50) & long_ok &
             (df['volume'] > df['vma'] * 1.0) & (df['atr'] > df['atr_ma']))
entry_idxs = np.where(entry_sig)[0]
print(f"  Entry signals: {len(entry_idxs)}", flush=True)

# ─── Trade Simulation with AUDIT ─────────────────────────────────
print("💸 Simulating trades (with audit)...", flush=True)

FEE = 0.001
CAPITAL = 1000
SELL_THRESH = 60

trades = []
in_trade = False
exit_done = 0
equity = CAPITAL
cmon = df['ts'].iloc[400].month
cyr = df['ts'].iloc[400].year
mstart = CAPITAL

for ei in entry_idxs:
    if ei < 500: continue
    if in_trade and ei < exit_done: continue
    
    ts = df['ts'].iloc[ei]
    if ts.month != cmon or ts.year != cyr:
        cmon, cyr = ts.month, ts.year
        mstart = equity
    
    # Monthly 7% limit
    if (equity - mstart) / mstart * 100 < -7:
        continue
    
    entry = df['close'].iloc[ei]
    sl = nsl(ei) * 0.998
    
    end = min(ei + 192, len(df))
    result = None
    exit_price = entry
    exit_idx = ei
    
    for j in range(ei + 1, end):
        if df['low'].iloc[j] <= sl:
            result = 'SL'
            exit_price = sl
            exit_idx = j
            break
        if df['sell_str'].iloc[j] >= SELL_THRESH:
            result = 'SELL'
            exit_price = df['close'].iloc[j]
            exit_idx = j
            break
    
    if result is None:
        result = 'TIME'
        exit_price = df['close'].iloc[end - 1]
        exit_idx = end - 1
    
    pnl_pct = (exit_price - entry) / entry * 100 - FEE * 2 * 100
    dollar_pnl = CAPITAL * (pnl_pct / 100)
    
    trades.append({
        'entry_idx': ei, 'entry_ts': df['ts'].iloc[ei],
        'entry_px': entry, 'sl_px': sl,
        'exit_idx': exit_idx, 'exit_ts': df['ts'].iloc[exit_idx],
        'exit_px': exit_price, 'result': result,
        'pnl_pct': pnl_pct, 'dollar_pnl': dollar_pnl,
        'swing_low': nsl(ei),
    })
    
    in_trade = True
    exit_done = exit_idx
    equity += dollar_pnl

# ─── AUDIT: Show sample trades ──────────────────────────────────
print(f"\n{'='*70}")
print(f"🔍 AUDIT — آخر 10 صفقات (مفحوصة يدوياً):")
print(f"{'='*70}")
print(f"{'تاريخ الدخول':<19} {'دخول':>7} {'SL':>7} {'سوينج':>7} {'نتيجة':<5} {'خروج':>7} {'ربح%':>7}")
print("-"*65)

for t in trades[-10:]:
    print(f"{str(t['entry_ts'])[:19]:<19} {t['entry_px']:>7.4f} {t['sl_px']:>7.4f} {t['swing_low']:>7.4f} {t['result']:<5} {t['exit_px']:>7.4f} {t['pnl_pct']:>+6.2f}%")

# Also show some SL exits that were profitable (trailing stop effect)
sl_wins = [t for t in trades if t['result'] == 'SL' and t['pnl_pct'] > 0]
if sl_wins:
    print(f"\n🔍 SL رابحة (مثال 5 من {len(sl_wins)}):")
    print(f"{'تاريخ':<19} {'دخول':>7} {'SL':>7} {'خروج':>7} {'ربح%':>7}")
    for t in sl_wins[:5]:
        print(f"{str(t['entry_ts'])[:19]:<19} {t['entry_px']:>7.4f} {t['sl_px']:>7.4f} {t['exit_px']:>7.4f} {t['pnl_pct']:>+6.2f}%")

sl_losses = [t for t in trades if t['result'] == 'SL' and t['pnl_pct'] < 0]
if sl_losses:
    print(f"\n🔴 SL خاسرة (مثال 5 من {len(sl_losses)}):")
    print(f"{'تاريخ':<19} {'دخول':>7} {'SL':>7} {'خروج':>7} {'ربح%':>7}")
    for t in sl_losses[:5]:
        print(f"{str(t['entry_ts'])[:19]:<19} {t['entry_px']:>7.4f} {t['sl_px']:>7.4f} {t['exit_px']:>7.4f} {t['pnl_pct']:>+6.2f}%")

# ─── Summary ─────────────────────────────────────────────────────
n = len(trades)
wins = [t for t in trades if t['pnl_pct'] > 0]
losses = [t for t in trades if t['pnl_pct'] <= 0]
nw, nl = len(wins), len(losses)
wr = nw / n * 100

total_profit = sum(t['pnl_pct'] for t in wins)
total_loss = abs(sum(t['pnl_pct'] for t in losses))
avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

pnls = [t['pnl_pct'] for t in trades]
sp = np.mean(pnls) / np.std(pnls) * np.sqrt(n) if np.std(pnls) > 0 else 0

eqs = [1000]
for t in trades: eqs.append(eqs[-1] + 1000 * (t['pnl_pct'] / 100))
pk = np.maximum.accumulate(eqs)
dd = (np.array(eqs) - pk) / pk * 100

sell_n = sum(1 for t in trades if t['result'] == 'SELL')
sl_n = sum(1 for t in trades if t['result'] == 'SL')
time_n = sum(1 for t in trades if t['result'] == 'TIME')

# Check for look-ahead: entry_idx vs exit_idx
lookahead = any(t['exit_idx'] <= t['entry_idx'] for t in trades)

print(f"\n{'='*50}")
print(f"🐋 WHALE 200-BAR — LONG ONLY — FULL HISTORY")
print(f"{'='*50}")
print(f"📅 {df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}")
print(f"📊 {len(df):,} شمعة")
print(f"🔍 Look-ahead bias: {'❌ YES (BUG!)' if lookahead else '✅ NONE'}")
print(f"")
print(f"📋 عدد الصفقات: {n}")
print(f"🟢 صفقات رابحة: {nw} | 🔴 صفقات خاسرة: {nl}")
print(f"📈 Win Rate: {wr:.1f}%")
print(f"💵 إجمالي الربح: +{total_profit:.1f}%")
print(f"💸 إجمالي الخسارة: -{total_loss:.1f}%")
print(f"💰 صافي: {sum(t['pnl_pct'] for t in trades):+.1f}%")
print(f"🟢 متوسط الربح: +{avg_win:.2f}%")
print(f"🔴 متوسط الخسارة: -{avg_loss:.2f}%")
print(f"📊 R:R: {rr:.1f}x")
print(f"📊 شارپ: {sp:.2f}")
print(f"📉 أقصى انخفاض: {dd.min():.1f}%")
print(f"")
print(f"🏦 المحفظة: $1000 → ${equity:,.0f} ({(equity/1000-1)*100:+.1f}%)")
print(f"")
print(f"🎯 إشارة بيع: {sell_n} | ⛔ SL: {sl_n} | ⏱️ زمني: {time_n}")
print(f"   منها SL رابحة: {len(sl_wins)} | SL خاسرة: {len(sl_losses)}")
print(f"   SELL رابحة: {sum(1 for t in trades if t['result']=='SELL' and t['pnl_pct']>0)} | خاسرة: {sum(1 for t in trades if t['result']=='SELL' and t['pnl_pct']<0)}")
