#!/usr/bin/env python3
"""Plot 10 QQE+SSL+EMA trades on FET/USDT 1h"""
import ccxt, pandas as pd, numpy as np, sys
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.insert(0, '/data/trading28')

COMM = 0.002; DAYS = 180; CAP = 1000

def fetch(symbol, tf, days):
    ex = ccxt.binance({'timeout': 15000})
    since = ex.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    all_c = []
    while True:
        batch = ex.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        if not batch: break
        all_c.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000: break
    df = pd.DataFrame(all_c, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True); df.sort_index(inplace=True)
    return df

def rsi_s(s, p):
    d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    return 100 - 100/(1 + g.ewm(alpha=1/p, adjust=False).mean()/l.ewm(alpha=1/p, adjust=False).mean())

def ema(s, p): return s.ewm(span=p, adjust=False).mean()

def hma(s, l):
    half = int(max(l/2, 2)); sq = int(max(np.sqrt(l), 1))
    w1 = s.rolling(half).apply(lambda x: np.average(x, weights=np.arange(1,half+1)), raw=True)
    w2 = s.rolling(l).apply(lambda x: np.average(x, weights=np.arange(1,l+1)), raw=True)
    return (2*w1 - w2).rolling(sq).apply(lambda x: np.average(x, weights=np.arange(1,sq+1)), raw=True)

def compute_qqe(close, rsi_len=6, smooth=5, factor=2.0):
    wilders_len = rsi_len * 2 - 1
    rsi_val = rsi_s(close, rsi_len)
    smoothed_rsi = ema(rsi_val, smooth)
    atr_rsi = (smoothed_rsi - smoothed_rsi.shift(1)).abs()
    smoothed_atr_rsi = ema(atr_rsi, wilders_len)
    dynamic_atr = smoothed_atr_rsi * factor
    n = len(close)
    long_band = np.full(n, np.nan); short_band = np.full(n, np.nan)
    warm = max(wilders_len + 10, 50)
    for i in range(warm, n):
        new_short = smoothed_rsi.iloc[i] + dynamic_atr.iloc[i]
        new_long = smoothed_rsi.iloc[i] - dynamic_atr.iloc[i]
        if not np.isnan(long_band[i-1]) and smoothed_rsi.iloc[i-1] > long_band[i-1] and smoothed_rsi.iloc[i] > long_band[i-1]:
            long_band[i] = max(long_band[i-1], new_long)
        else: long_band[i] = new_long
        if not np.isnan(short_band[i-1]) and smoothed_rsi.iloc[i-1] < short_band[i-1] and smoothed_rsi.iloc[i] < short_band[i-1]:
            short_band[i] = min(short_band[i-1], new_short)
        else: short_band[i] = new_short
    return smoothed_rsi.values

print("Fetching FET/USDT 1h...")
df = fetch('FET/USDT', '1h', DAYS)
print(f"  {len(df)} candles")

c = df['close'].values; h = df['high'].values; l = df['low'].values
n = len(c); warmup = 200

# QQE + SSL + EMA200
primary_rsi = compute_qqe(df['close'], 6, 5, 2.0)
secondary_rsi = compute_qqe(df['close'], 6, 5, 1.61)
primary_zero = primary_rsi - 50; secondary_zero = secondary_rsi - 50

bb_basis = pd.Series(primary_zero).rolling(30).mean().values
bb_std = pd.Series(primary_zero).rolling(30).std().values
bb_upper = bb_basis + 0.5 * bb_std; bb_lower = bb_basis - 0.5 * bb_std

qqe_blue = (secondary_zero > 2.0) & (primary_zero > bb_upper)
qqe_red = (secondary_zero < -2.0) & (primary_zero < bb_lower)

exit_high = hma(df['high'], 10).values
exit_low = hma(df['low'], 10).values
hlv3 = np.zeros(n); ssl_exit_val = np.full(n, np.nan)
for i in range(1, n):
    if np.isnan(exit_high[i]): hlv3[i] = hlv3[i-1]
    elif c[i] > exit_high[i]: hlv3[i] = 1
    elif c[i] < exit_low[i]: hlv3[i] = -1
    else: hlv3[i] = hlv3[i-1]
    ssl_exit_val[i] = exit_high[i] if hlv3[i] < 0 else exit_low[i]

ssl_bull = np.zeros(n, dtype=bool); ssl_bear = np.zeros(n, dtype=bool)
for i in range(2, n):
    if not np.isnan(ssl_exit_val[i]):
        ssl_bull[i] = c[i] > ssl_exit_val[i] and c[i-1] <= ssl_exit_val[i-1]
        ssl_bear[i] = c[i] < ssl_exit_val[i] and c[i-1] >= ssl_exit_val[i-1]

ema_line = ema(df['close'], 200).values

long_entry = np.zeros(n, dtype=bool); short_entry = np.zeros(n, dtype=bool)
for i in range(warmup, n):
    if np.isnan(ema_line[i]): continue
    if qqe_blue[i] and ssl_bull[i] and c[i] > ema_line[i]: long_entry[i] = True
    elif qqe_red[i] and ssl_bear[i] and c[i] < ema_line[i]: short_entry[i] = True

# Simulate
trades = []; pos = 0; ep = 0
for i in range(warmup, n):
    if pos == 0:
        if long_entry[i]: pos=1; ep=c[i]; trades.append({'type':'L','entry_i':i,'entry_px':c[i]})
        elif short_entry[i]: pos=-1; ep=c[i]; trades.append({'type':'S','entry_i':i,'entry_px':c[i]})
    elif pos == 1:
        if short_entry[i]:
            trades[-1]['exit_i']=i; trades[-1]['exit_px']=c[i]; trades[-1]['pnl']=(c[i]/ep-1)*100-COMM*100
            pos=-1; ep=c[i]; trades.append({'type':'S','entry_i':i,'entry_px':c[i]})
    elif pos == -1:
        if long_entry[i]:
            trades[-1]['exit_i']=i; trades[-1]['exit_px']=c[i]; trades[-1]['pnl']=(1-c[i]/ep)*100-COMM*100
            pos=1; ep=c[i]; trades.append({'type':'L','entry_i':i,'entry_px':c[i]})

# Take last 10 completed trades
completed = [t for t in trades if 'exit_i' in t]
plot_trades = completed[-10:]

# ═══════════ PLOT ═══════════
idx = df.index
fig, ax = plt.subplots(figsize=(20, 12))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Find the range covering all 10 trades
start_i = max(0, plot_trades[0]['entry_i'] - 20)
end_i = min(n-1, plot_trades[-1]['exit_i'] + 20)

# Plot candles
for i in range(start_i, end_i+1):
    clr = '#00ff88' if c[i] >= df['open'].iloc[i] else '#ff4466'
    ax.plot([i, i], [l[i], h[i]], color=clr, linewidth=0.8)
    ax.plot([i, i], [df['open'].iloc[i], c[i]], color=clr, linewidth=4)

# Plot EMA200
ax.plot(range(start_i, end_i+1), ema_line[start_i:end_i+1], color='orange', linewidth=1.5, alpha=0.7, label='EMA200')

# Plot trades
for t in plot_trades:
    ei = t['entry_i']; xi = t['exit_i']
    clr = '#00ff88' if t['type'] == 'L' else '#ff4466'
    # Entry arrow
    ax.scatter(ei, t['entry_px'], color=clr, s=120, marker='^' if t['type']=='L' else 'v', zorder=5, edgecolors='white', linewidths=1)
    # Exit arrow
    ax.scatter(xi, t['exit_px'], color='white', s=80, marker='o' if t['type']=='L' else 'o', zorder=5, edgecolors=clr, linewidths=2)
    # Line connecting
    ax.plot([ei, xi], [t['entry_px'], t['exit_px']], color=clr, linewidth=1.5, alpha=0.5, linestyle='--')
    # PnL label
    pnl = t['pnl']
    mid_i = (ei + xi) // 2
    mid_px = max(t['entry_px'], t['exit_px']) + 0.02
    ax.annotate(f"{pnl:+.1f}%", (mid_i, mid_px), color='white' if pnl > 0 else '#ff4466',
                fontsize=8, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=clr, alpha=0.8))

# X-axis dates
date_labels = [idx[i].strftime('%m/%d %H:%M') for i in range(start_i, end_i+1)]
step = max(1, (end_i - start_i) // 12)
tick_positions = list(range(start_i, end_i+1, step))
tick_labels = [date_labels[i-start_i] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, fontsize=8, color='white')

ax.tick_params(axis='y', colors='white')
ax.grid(alpha=0.15, color='white')
ax.set_ylabel('FET/USDT', color='white', fontsize=12)
ax.set_title(f'QQE+SSL+EMA200 — FET/USDT 1h — Last 10 Trades', color='white', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)

# Win/loss summary
wins = sum(1 for t in plot_trades if t['pnl'] > 0)
losses = len(plot_trades) - wins
win_avg = np.mean([t['pnl'] for t in plot_trades if t['pnl'] > 0]) if wins else 0
loss_avg = np.mean([t['pnl'] for t in plot_trades if t['pnl'] <= 0]) if losses else 0
summary = f"Trades: {len(plot_trades)} | Wins: {wins} | Losses: {losses} | WinRate: {wins/len(plot_trades)*100:.0f}% | AvgWin: {win_avg:+.1f}% | AvgLoss: {loss_avg:+.1f}%"
ax.text(0.5, 1.02, summary, transform=ax.transAxes, ha='center', color='white', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#2a2a4e', alpha=0.8))

plt.tight_layout()
path = '/data/trading28/charts/qqe_ssl_10trades_fet.png'
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"\nSaved: {path}")
print(f"\nTrade details:")
for i, t in enumerate(plot_trades):
    print(f"  {i+1}. {t['type']} | {idx[t['entry_i']].strftime('%m/%d %H:%M')} → {idx[t['exit_i']].strftime('%m/%d %H:%M')} | {t['pnl']:+.2f}%")
