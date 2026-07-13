#!/usr/bin/env python3
"""Whale 200-bar LONG — FULL backtest with REAL compounding & DD"""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv', parse_dates=['ts'])
print(f"📊 {len(df):,} candles | {df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}", flush=True)

FEE = 0.001
BARS = 200

# Whale
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

# RSI + Sell
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
lb = 5; swl = np.zeros(len(df), dtype=bool)
for i in range(lb*2, len(df)):
    w = df['low'].iloc[i-lb*2:i+1]; m = i - lb
    if df['low'].iloc[m] == w.min() and w.values.argmax() == lb: swl[i] = True
def nsl(idx):
    for j in range(idx-1, max(0, idx-100), -1):
        if swl[j]: return df['low'].iloc[j]
    return df['low'].iloc[idx] * 0.95

# Entry
long_ok = df['wma50'] > df['wma200']
entry_sig = (df['spike'] & (df['wstr'] > 50) & long_ok &
             (df['volume'] > df['vma']) & (df['atr'] > df['atr_ma']))
entry_idxs = np.where(entry_sig)[0]
print(f"🐋 Signals: {len(entry_idxs)}", flush=True)

# Simulate with REAL compounding
trades = []
in_trade = False; exit_done = 0
equity = 1000  # starting capital
cmon = df['ts'].iloc[500].month; cyr = df['ts'].iloc[500].year
month_start = 1000
equity_curve = [(df['ts'].iloc[500], 1000)]

for ei in entry_idxs:
    if ei < 500: continue
    if in_trade and ei < exit_done: continue
    
    ts = df['ts'].iloc[ei]
    if ts.month != cmon or ts.year != cyr:
        cmon, cyr = ts.month, ts.year
        month_start = equity
    
    # Monthly 7% limit
    monthly_loss = (equity - month_start) / month_start * 100
    if monthly_loss <= -7:
        continue
    
    entry = df['close'].iloc[ei]
    sl = nsl(ei) * 0.998
    
    end = min(ei + 192, len(df))
    result = None; exit_price = entry; exit_idx = ei
    
    for j in range(ei + 1, end):
        if df['low'].iloc[j] <= sl:
            result = 'SL'; exit_price = sl; exit_idx = j; break
        if df['sell_str'].iloc[j] >= 60:
            result = 'SELL'; exit_price = df['close'].iloc[j]; exit_idx = j; break
    
    if result is None:
        result = 'TIME'; exit_price = df['close'].iloc[end-1]; exit_idx = end-1
    
    pnl_pct = (exit_price - entry) / entry * 100 - FEE * 2 * 100
    
    # REAL compounding: equity changes by pnl_pct
    trades.append({
        'entry_ts': ts, 'entry_px': entry,
        'exit_ts': df['ts'].iloc[exit_idx], 'exit_px': exit_price,
        'result': result, 'pnl_pct': pnl_pct,
        'eq_before': round(equity),
    })
    
    equity = equity * (1 + pnl_pct / 100)
    equity_curve.append((df['ts'].iloc[exit_idx], equity))
    
    in_trade = True
    exit_done = exit_idx

# Metrics
n = len(trades)
wins = [t for t in trades if t['pnl_pct'] > 0]
losses = [t for t in trades if t['pnl_pct'] <= 0]
nw, nl = len(wins), len(losses)
wr = nw / n * 100

total_profit = sum(t['pnl_pct'] for t in wins)
total_loss = abs(sum(t['pnl_pct'] for t in losses))
net_pnl = sum(t['pnl_pct'] for t in trades)
avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

pnls = [t['pnl_pct'] for t in trades]
sp = np.mean(pnls) / np.std(pnls) * np.sqrt(n) if np.std(pnls) > 0 else 0

# TRUE DD (based on equity curve, not capital)
eqs = np.array([e for _, e in equity_curve])
peak = np.maximum.accumulate(eqs)
dd_pct = (eqs - peak) / peak * 100
max_dd = dd_pct.min()

# Find worst DD period
worst_dd_idx = dd_pct.argmin()

# Monthly performance
trades_df = pd.DataFrame(trades)
trades_df['month'] = trades_df['entry_ts'].dt.to_period('M')
monthly = trades_df.groupby('month')['pnl_pct'].sum()
worst_month = monthly.min()
best_month = monthly.max()

print(f"\n{'='*60}")
print(f"🐋 WHALE 200-BAR LONG — COMPOUNDING (REAL DD)")
print(f"{'='*60}")
print(f"📅 {df['ts'].iloc[500].date()} → {df['ts'].iloc[-1].date()}")
print(f"")
print(f"📋 عدد الصفقات: {n}")
print(f"🟢 صفقات رابحة: {nw} | 🔴 صفقات خاسرة: {nl}")
print(f"📈 Win Rate: {wr:.1f}%")
print(f"💵 إجمالي الربح: +{total_profit:.1f}%")
print(f"💸 إجمالي الخسارة: -{total_loss:.1f}%")
print(f"💰 صافي: {net_pnl:+.1f}%")
print(f"🟢 متوسط الربح: +{avg_win:.2f}%")
print(f"🔴 متوسط الخسارة: -{avg_loss:.2f}%")
print(f"📊 R:R: {rr:.1f}x")
print(f"📊 شارپ: {sp:.2f}")
print(f"")
print(f"📉 أقصى انخفاض (DD): {max_dd:.1f}%")
print(f"   عند: {equity_curve[worst_dd_idx][0]}")
print(f"   المحفظة: ${equity_curve[worst_dd_idx][1]:,.0f}")
print(f"🏆 أفضل شهر: +{best_month:.1f}%")
print(f"💀 أسوأ شهر: {worst_month:.1f}%")
print(f"")
print(f"🏦 المحفظة: $1000 → ${equity:,.0f} ({(equity/1000-1)*100:+.1f}%)")
print(f"")
print(f"📈 أقصى صفقات رابحة متتالية: {max((sum(1 for _ in g) for k,g in __import__('itertools').groupby([t['pnl_pct']>0 for t in trades]) if k), default=0)}")
print(f"📉 أقصى صفقات خاسرة متتالية: {max((sum(1 for _ in g) for k,g in __import__('itertools').groupby([t['pnl_pct']<=0 for t in trades]) if k), default=0)}")

# Show worst month detail
worst_month_period = monthly.idxmin()
print(f"\n🔴 تفاصيل أسوأ شهر ({worst_month_period}):")
for t in trades:
    if t['entry_ts'].to_period('M') == worst_month_period:
        e = "🟢" if t['pnl_pct']>0 else "🔴"
        print(f"  {e} {str(t['entry_ts'])[:16]} → {str(t['exit_ts'])[:16]} | {t['entry_px']:.4f}→{t['exit_px']:.4f} | {t['result']:>4} | {t['pnl_pct']:+.2f}% | محفظة قبل: ${t['eq_before']:,.0f}")
