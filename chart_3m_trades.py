#!/usr/bin/env python3
"""مخططات صفقات 3m — 5 ناجحة + 5 فاشلة"""
import json, numpy as np, pandas as pd, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta

COMM = 0.20
DATA_DIR = '/data/trading28/data/3m_4months'

def compute_3m_indicators(df):
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    w = (df['low'].values - df['low_raw'].values) / np.where(df['low_raw'].values != 0, df['low_raw'].values, np.nan) * 100
    df['whale'] = np.clip(w, 0, None)
    vm = df['volume'].rolling(20).mean().values
    df['spike'] = df['volume'].values / np.where(vm != 0, vm, np.nan)
    delta = df['close'].diff().values
    gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(14).mean().values
    loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(14).mean().values
    df['rsi'] = 100 - 100 / (1 + gain / np.where(loss != 0, loss, np.nan))
    return df

def chart_trade(symbol, entry_ts_ms, exit_ts_ms, entry_price, exit_price, pnl_pct, 
                exit_type, whale_val, rsi_val, tp_price, sl_price, output_path, candles=80):
    
    # Load data
    fpath = f'{DATA_DIR}/{symbol.upper()}.json'
    if not os.path.exists(fpath):
        print(f'  ❌ {symbol}: file not found')
        return None
    with open(fpath) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw).rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    
    # Compute indicators
    df = compute_3m_indicators(df)
    
    # Find entry/exit
    entry_dt = pd.to_datetime(entry_ts_ms, unit='ms', utc=True)
    exit_dt = pd.to_datetime(exit_ts_ms, unit='ms', utc=True)
    
    idx_loc = df.index.get_indexer([entry_dt], method='nearest')[0]
    start = max(0, idx_loc - candles)
    end = min(len(df), idx_loc + candles // 2)
    df_chart = df.iloc[start:end].copy()
    
    # Convert index to matplotlib dates
    x_dates = mdates.date2num(df_chart.index.to_pydatetime())
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), 
                                          gridspec_kw={'height_ratios': [2.5, 1.2, 1]},
                                          facecolor='#0d1117')
    fig.subplots_adjust(hspace=0.05)
    
    # Panel 1: Candlestick price
    ax1.set_facecolor('#0d1117')
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_chart['close'], df_chart['open'])]
    body_w = (x_dates[1] - x_dates[0]) * 0.6 if len(x_dates) > 1 else 0.0002
    
    for i in range(len(df_chart)):
        ax1.plot([x_dates[i]]*2, [df_chart['low'].iloc[i], df_chart['high'].iloc[i]], 
                color=colors[i], lw=1.2)
        body_b = min(df_chart['open'].iloc[i], df_chart['close'].iloc[i])
        body_h = abs(df_chart['close'].iloc[i] - df_chart['open'].iloc[i]) or 0.00005
        ax1.add_patch(plt.Rectangle((x_dates[i]-body_w/2, body_b), body_w, body_h,
                                     facecolor=colors[i], edgecolor=colors[i], lw=0.5))
    
    # Entry/Exit markers
    is_win = pnl_pct > 0
    marker_color = '#00ff41' if is_win else '#ff1744'
    ax1.scatter(mdates.date2num(entry_dt), entry_price, color=marker_color, s=250, zorder=10, 
               marker='^', edgecolors='white', lw=2)
    ax1.scatter(mdates.date2num(exit_dt), exit_price, color='#ff9800', s=250, zorder=10, 
               marker='v', edgecolors='white', lw=2)
    
    # TP/SL lines
    ax1.axhline(y=tp_price, color='#00ff41', ls=':', lw=1, alpha=0.4)
    ax1.axhline(y=sl_price, color='#ff1744', ls=':', lw=1, alpha=0.4)
    
    # Info box
    status = '🟢 ربح' if is_win else '🔴 خسارة'
    info = (f'{status} | {exit_type} | ⏱️ {(exit_dt-entry_dt).total_seconds()/60:.0f}د\n'
            f'دخول: {entry_price:.6f} | خروج: {exit_price:.6f}\n'
            f'🐋 حوت: {whale_val:.2f} | 📉 RSI: {rsi_val:.1f}\n'
            f'🎯 TP: {tp_price:.6f} | 🛑 SL: {sl_price:.6f}')
    ax1.text(0.02, 0.97, info, transform=ax1.transAxes, fontsize=10, va='top',
             color='#c9d1d9', family='monospace',
             bbox=dict(boxstyle='round', fc='#161b22', ec='#30363d', alpha=0.95))
    
    ax1.set_ylabel('USDT', color='white', fontsize=11)
    ax1.tick_params(colors='white', labelsize=9)
    ax1.grid(alpha=0.12, color='white')
    for spine in ax1.spines.values(): spine.set_edgecolor('#333')
    
    # Panel 2: Whale Indicator
    ax2.set_facecolor('#0d1117')
    ax2.fill_between(x_dates, df_chart['whale'], alpha=0.2, color='#7c4dff')
    ax2.plot(x_dates, df_chart['whale'], color='#7c4dff', lw=2)
    ax2.axhline(y=0.10, color='orange', ls='--', lw=1, alpha=0.5)
    ax2.scatter(mdates.date2num(entry_dt), whale_val, color='#7c4dff', s=120, zorder=10, edgecolors='white', lw=1.5)
    ax2.set_ylabel('Whale', color='white', fontsize=11)
    ax2.tick_params(colors='white', labelsize=9)
    ax2.grid(alpha=0.12, color='white')
    for spine in ax2.spines.values(): spine.set_edgecolor('#333')
    
    # Panel 3: RSI
    ax3.set_facecolor('#0d1117')
    ax3.plot(x_dates, df_chart['rsi'], color='#2196f3', lw=2)
    ax3.fill_between(x_dates, 35, df_chart['rsi'], where=(df_chart['rsi']<35), alpha=0.15, color='#ff1744')
    ax3.axhline(y=35, color='#ff1744', ls='--', lw=1, alpha=0.5)
    ax3.scatter(mdates.date2num(entry_dt), rsi_val, color='#ff9800', s=80, zorder=10, edgecolors='white', lw=1)
    ax3.set_ylabel('RSI(14)', color='white', fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.tick_params(colors='white', labelsize=9)
    ax3.grid(alpha=0.12, color='white')
    for spine in ax3.spines.values(): spine.set_edgecolor('#333')
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    title_emoji = '🟢' if is_win else '🔴'
    fig.suptitle(f'{title_emoji} {symbol.upper()}/USDT — 3m | {exit_type} | {pnl_pct:+.2f}%',
                 color='white', fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    return output_path

# ═══════════ الصفقات ═══════════
TRADES = [
    # 🟢 ناجحة
    ('0G', 1776155400000, 1776155760000, 0.5830, 0.5894, +1.10, 'TP', 0.17, 29.4, 0.5906, 0.5801),
    ('0G', 1780649820000, 1780650720000, 0.3140, 0.3174, +1.10, 'TP', 0.32, 33.3, 0.3181, 0.3124),
    ('0G', 1780687800000, 1780688700000, 0.2940, 0.2978, +1.10, 'TP', 1.03, 33.3, 0.2978, 0.2925),
    ('0G', 1782869580000, 1782870120000, 0.2000, 0.2025, +1.10, 'TP', 1.52, 25.0, 0.2026, 0.1990),
    ('2Z', 1775543220000, 1775543580000, 0.08331, 0.08439, +1.10, 'TP', 0.23, 15.3, 0.0844, 0.0829),
    # 🔴 فاشلة
    ('ONG', 1785116160000, 1785130560000, 0.0458, 0.0454, -0.79, 'TIME', 0.11, 23.5, 0.0464, 0.0456),
    ('SOL', 1777652460000, 1777666860000, 84.19, 83.56, -0.75, 'TIME', 0.18, 33.3, 85.28, 83.77),
    ('0G', 1776468780000, 1776470400000, 0.6210, 0.6167, -0.70, 'SL', 0.16, 30.0, 0.6291, 0.6179),
    ('0G', 1778333220000, 1778336640000, 0.5700, 0.5660, -0.70, 'SL', 0.18, 14.3, 0.5774, 0.5672),
    ('0G', 1779060600000, 1779061140000, 0.4900, 0.4866, -0.70, 'SL', 0.20, 33.3, 0.4964, 0.4876),
]

os.makedirs('/tmp/trade_charts', exist_ok=True)

for i, (sym, e_ts, x_ts, ep, xp, pnl, xt, wv, rv, tp_p, sl_p) in enumerate(TRADES):
    out = f'/tmp/trade_charts/{i+1:02d}_{sym}_{xt}_{"win" if pnl>0 else "loss"}.png'
    result = chart_trade(sym, e_ts, x_ts, ep, xp, pnl, xt, wv, rv, tp_p, sl_p, out)
    if result:
        print(f'✅ {result}')

print('\n📁 الملفات في /tmp/trade_charts/')
