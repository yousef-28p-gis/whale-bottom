#!/usr/bin/env python3
"""Compare Old vs Current Hunter Whale configs on July 2026."""

import json, numpy as np, pandas as pd, os
from datetime import datetime, timedelta
from collections import defaultdict

CACHE = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'
MIN_VOL = 200000
STR = 50

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
BLOCKED = {'SUPER','ORCA','VANA','W','DOGS','MET','XLM','BB','COS','LUNA','S'}

# ─── CONFIGS ───
OLD = {'tp': 3.0, 'sl': 2.0, 'pl': 50, 'trail': 0.20, 'max_h': 2, 'whale_min': 0.0, 'blocked': set(), 'time_filter': True}
NEW = {'tp': 2.5, 'sl': 2.0, 'pl': 40, 'trail': 0.10, 'max_h': 2, 'whale_min': 0.35, 'blocked': BLOCKED, 'time_filter': False}

BAD_HOURS = {21,22,23}  # UTC
BAD_DAY = 2  # Wednesday (0=Mon)


def load_cached(sym, mon):
    fpath = f'{CACHE}/{sym}_{mon}.json'
    if not os.path.exists(fpath): return None
    with open(fpath) as f: data = json.load(f)
    df = pd.DataFrame(data)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    return df.sort_values('ts').reset_index(drop=True)


def whale_indicator(df):
    df = df.copy()
    LB, WF, WS, VM = 30, 2, 5, 1.0
    df['lo'] = df['low'].rolling(LB).min()
    df['lc'] = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
    df['sm'] = df['lc'].ewm(span=3, adjust=False).mean()
    df['hi'] = df['sm'].rolling(LB).max()
    df['raw'] = np.where(df['low'] <= df['lo'], (df['sm'] + df['hi'] * 2) / 3, 0)
    df['whale'] = df['raw'].ewm(span=3, adjust=False).mean().fillna(0)
    df['spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.03)
    df['wf'] = df['whale'].rolling(WF).mean()
    df['ws'] = df['whale'].rolling(WS).mean()
    df['wp'] = df['whale'].rolling(50).max()
    df['str'] = (df['whale'] / df['wp'].replace(0, np.nan) * 100).fillna(0)
    df['vma'] = df['volume'].rolling(20).mean()
    df['entry'] = (df['spike'] & (df['wf'] > df['ws']) &
                   (df['str'] > STR) & (df['volume'] > df['vma'] * VM))
    return df


def find_entry(df_w, signal_dt):
    df_w = df_w.copy()
    df_w['td'] = abs((df_w['ts'] - signal_dt).dt.total_seconds())
    nearest = df_w['td'].idxmin()
    for j in range(min(len(df_w) - nearest, 96)):
        idx = nearest + j
        if idx < len(df_w) and df_w.iloc[idx]['entry']:
            return idx, float(df_w.iloc[idx]['whale']), float(df_w.iloc[idx]['close']), df_w.iloc[idx]['ts']
    return None


def simulate(df_w, entry_idx, cfg):
    tp_p = df_w.iloc[entry_idx]['close'] * (1 + cfg['tp']/100)
    sl_p = df_w.iloc[entry_idx]['close'] * (1 - cfg['sl']/100)
    pl_p = df_w.iloc[entry_idx]['close'] + (tp_p - df_w.iloc[entry_idx]['close']) * (cfg['pl']/100)
    entry_price = df_w.iloc[entry_idx]['close']

    pl_trig = False; peak = entry_price; trail_p = 0
    for j in range(entry_idx + 1, min(len(df_w), entry_idx + int(cfg['max_h']/0.25) + 1)):
        c = df_w.iloc[j]
        h = (j - entry_idx) * 0.25
        if h > cfg['max_h']:
            return round((c['close'] - entry_price) / entry_price * 100, 4), 'timeout'

        if not pl_trig and c['high'] >= pl_p:
            pl_trig = True; peak = c['high']; trail_p = c['high'] * (1 - cfg['trail']/100)

        if pl_trig:
            if c['high'] > peak: peak = c['high']; trail_p = c['high'] * (1 - cfg['trail']/100)
            if c['low'] <= trail_p:
                return round((trail_p - entry_price) / entry_price * 100, 4), 'trail'

        if c['high'] >= tp_p: return round(cfg['tp'], 4), 'tp'
        if c['low'] <= sl_p: return round(-cfg['sl'], 4), 'sl'

    j = min(len(df_w) - 1, entry_idx + int(cfg['max_h']/0.25))
    return round((df_w.iloc[j]['close'] - entry_price) / entry_price * 100, 4), 'eod'


def run_backtest(cfg, label):
    with open(SIGNALS_FILE) as f:
        raw = json.load(f)

    signals = []
    for s in raw:
        if s['symbol'] in STABLES or s['symbol'] in cfg['blocked']: continue
        if s.get('direction', 'LONG') != 'LONG': continue
        if s.get('volume_usdt', 0) < MIN_VOL: continue
        dt = datetime.fromisoformat(s['dt'])
        if dt.year != 2026 or dt.month < 1 or dt.month > 6: continue
        signals.append({'symbol': s['symbol'], 'dt': dt, 'month': dt.strftime('%Y-%m')})

    trades = []
    no_cache = no_entry = time_skip = 0

    for sig in signals:
        if cfg['time_filter']:
            if sig['dt'].hour in BAD_HOURS or sig['dt'].weekday() == BAD_DAY:
                time_skip += 1
                continue

        df = load_cached(sig['symbol'], sig['month'])
        if df is None: no_cache += 1; continue
        df_w = whale_indicator(df)
        r = find_entry(df_w, sig['dt'])
        if r is None: no_entry += 1; continue

        entry_idx, whale_val, entry_price, confirm_ts = r
        if whale_val < cfg['whale_min']: no_entry += 1; continue

        pnl, reason = simulate(df_w, entry_idx, cfg)
        trades.append({'pnl': pnl, 'reason': reason})

    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = sum(1 for t in trades if t['pnl'] <= 0)
    total_pnl = sum(t['pnl'] for t in trades)
    tp_count = sum(1 for t in trades if t['reason'] == 'tp')
    sl_count = sum(1 for t in trades if t['reason'] == 'sl')

    # Simple compounding approximation: each trade uses 33% of capital
    capital = 1000.0
    peak = capital; max_dd = 0
    for t in trades:
        pos_ret = t['pnl'] * 0.33
        capital *= (1 + pos_ret / 100)
        if capital > peak: peak = capital
        dd = (capital - peak) / peak * 100
        if dd < max_dd: max_dd = dd

    monthly_ret = (capital / 1000) ** (1/6) - 1

    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    print(f'  Trades:      {len(trades)}')
    if trades:
        print(f'  Wins:        {wins} ({wins/len(trades)*100:.1f}%)')
        print(f'  Losses:      {losses} ({losses/len(trades)*100:.1f}%)')
    print(f'  TP exits:    {tp_count}')
    print(f'  SL exits:    {sl_count}')
    print(f'  Total PnL:   {total_pnl:+.2f}%')
    print(f'  Final Cap:   ${capital:.0f} (from $1,000)')
    print(f'  Max DD:      {max_dd:.1f}%')
    print(f'  Monthly:     {monthly_ret*100:+.1f}%')
    print(f'  No Cache:    {no_cache} | No Entry: {no_entry} | Time Skip: {time_skip}')

    return {'trades': len(trades), 'wins': wins, 'losses': losses,
            'wr': wins/len(trades)*100 if trades else 0, 'total_pnl': total_pnl,
            'capital': capital, 'tp': tp_count, 'sl': sl_count}


# ── RUN ──
print('🐋 Jan-Jun 2026 — Old vs New Hunter Whale')
print(f'   OLD: TP3/2 PL50 Tr0.20 whale≥0 time-filter')
print(f'   NEW: TP2.5/2 PL40 Tr0.10 whale≥0.35 smart-block')

old_r = run_backtest(OLD, '🕰️  OLD (before optimization)')
new_r = run_backtest(NEW, '🚀  NEW (current)')

print(f'\n{"="*60}')
print(f'  📊 COMPARISON')
print(f'{"="*60}')
print(f'  {"Metric":<20} {"OLD":>10} {"NEW":>10} {"Δ":>10}')
print(f'  {"-"*50}')
print(f'  {"Trades":<20} {old_r["trades"]:>10} {new_r["trades"]:>10} {new_r["trades"]-old_r["trades"]:>+10}')
print(f'  {"Win Rate":<20} {old_r["wr"]:>9.1f}% {new_r["wr"]:>9.1f}% {new_r["wr"]-old_r["wr"]:>+9.1f}%')
print(f'  {"Total PnL":<20} {old_r["total_pnl"]:>+9.2f}% {new_r["total_pnl"]:>+9.2f}% {new_r["total_pnl"]-old_r["total_pnl"]:>+9.2f}%')
print(f'  {"Final Capital":<20} ${old_r["capital"]:>9.0f} ${new_r["capital"]:>9.0f} ${new_r["capital"]-old_r["capital"]:>+9.0f}')
print(f'  {"TP exits":<20} {old_r["tp"]:>10} {new_r["tp"]:>10} {new_r["tp"]-old_r["tp"]:>+10}')
print(f'  {"SL exits":<20} {old_r["sl"]:>10} {new_r["sl"]:>10} {new_r["sl"]-old_r["sl"]:>+10}')
