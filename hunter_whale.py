#!/usr/bin/env python3
"""
🐋 الحوت الصياد (Hunter Whale) — Trading Bot v1.0
=====================================================
Final config: TP2.5/2 | PL40 | Tr0.10 | max 2h | 3×33%
Filters: whale≥0.35 | no time filter | block loss>65% coins

Results (Jan-Jun 2026): 1,833 trades | WR 64.6% | $7,264 | DD -4.5% | +39.2%/mo
July 2026 (OOS): 73 trades | WR 69.9% | +19% in 12 days

Usage:
  python3 hunter_whale.py backtest   # Run 6-month backtest
  python3 hunter_whale.py live       # Monitor signals live (needs signals feed)
  python3 hunter_whale.py analyze    # Analyze losing trades
"""

import json, numpy as np, pandas as pd, os, sys
from datetime import datetime, timedelta
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
TP = 2.5          # Take Profit %
SL = 2.0          # Stop Loss %
PL = 40           # Profit Lock (% of TP)
TRAIL = 0.10      # Trailing stop %
MAX_HOURS = 2     # Max hold time
MAX_POS = 3       # Max concurrent positions
POS_PCT = 33      # Position size %
MIN_VOL = 200000  # Min signal volume
STR = 50          # Whale strength threshold
WHALE_MIN = 0.35  # Min whale value at entry

# Blocked coins (loss rate > 65%)
BLOCKED_COINS = {
    'SUPER', 'ORCA', 'VANA', 'W', 'DOGS', 'MET',
    'XLM', 'BB', 'COS', 'LUNA', 'S'
}

STABLES = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDE', 'XUSD',
    'BFUSD', 'FDUSD', 'USDD', 'FRAX', 'LUSD', 'PYUSD',
    'USDJ', 'RLUSD', 'XAUT', 'USD1', 'EUR'
}

CACHE_DIR = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'


# ═══════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def load_cached(symbol, month):
    """Load cached OHLCV data for a symbol+month pair."""
    fpath = f'{CACHE_DIR}/{symbol}_{month}.json'
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.rename(columns={
        'o': 'open', 'h': 'high', 'l': 'low',
        'c': 'close', 'v': 'volume'
    })
    return df.sort_values('ts').reset_index(drop=True)


def whale_indicator(df, STR=50):
    """Compute whale pump indicator on OHLCV DataFrame."""
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
    df['entry'] = (
        df['spike'] & (df['wf'] > df['ws']) &
        (df['str'] > STR) & (df['volume'] > df['vma'] * VM)
    )
    return df


def simulate(df, entry_idx):
    """Simulate a trade from entry_idx. Returns (pnl%, exit_reason)."""
    entry_price = df.iloc[entry_idx]['close']
    tp_price = entry_price * (1 + TP / 100)
    sl_price = entry_price * (1 - SL / 100)
    pl_price = entry_price + (tp_price - entry_price) * (PL / 100)

    pl_triggered = False
    peak = entry_price
    trail_price = 0

    for j in range(entry_idx + 1, len(df)):
        candle = df.iloc[j]
        hours = (j - entry_idx) * 0.25

        if hours > MAX_HOURS:
            return round((candle['close'] - entry_price) / entry_price * 100, 4), 'timeout'

        if not pl_triggered and candle['high'] >= pl_price:
            pl_triggered = True
            peak = candle['high']
            trail_price = candle['high'] * (1 - TRAIL / 100)

        if pl_triggered:
            if candle['high'] > peak:
                peak = candle['high']
                trail_price = candle['high'] * (1 - TRAIL / 100)
            if candle['low'] <= trail_price:
                return round((trail_price - entry_price) / entry_price * 100, 4), 'trail'

        if candle['high'] >= tp_price:
            return round(TP, 4), 'tp'

        if candle['low'] <= sl_price:
            return round(-SL, 4), 'sl'

    last_close = df.iloc[-1]['close']
    return round((last_close - entry_price) / entry_price * 100, 4), 'eod'


def find_entry(df_w, signal_dt):
    """Find whale entry point nearest to signal datetime.
    Returns (forward_df, entry_idx, whale_val) or None."""
    df_w['td'] = abs((df_w['ts'] - signal_dt).dt.total_seconds())
    nearest = df_w['td'].idxmin()
    forward = df_w.iloc[nearest:].reset_index(drop=True)

    for j, row in forward.iterrows():
        if j * 0.25 > 24:  # Max 24h look-forward
            break
        if row['entry']:
            return forward, j, float(row['whale'])
    return None


def load_signals(start_month=None, end_month=None):
    """Load and filter signals from the raw file."""
    with open(SIGNALS_FILE) as f:
        raw = json.load(f)

    signals = []
    for s in raw:
        if s['symbol'] in STABLES:
            continue
        if s.get('direction', 'LONG') != 'LONG':
            continue
        if s.get('volume_usdt', 0) < MIN_VOL:
            continue
        if s['symbol'] in BLOCKED_COINS:
            continue

        dt = datetime.fromisoformat(s['dt'])
        if dt.year != 2026:
            continue
        if start_month and dt.month < start_month:
            continue
        if end_month and dt.month > end_month:
            continue

        signals.append({
            'symbol': s['symbol'],
            'dt': dt,
            'month': dt.strftime('%Y-%m'),
            'volume_usdt': s.get('volume_usdt', 0)
        })

    return signals


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════════════

def run_portfolio(trades):
    """Simulate portfolio compounding with position sizing."""
    if not trades:
        return 1000, 0

    trades = sorted(trades, key=lambda t: t['dt'])
    capital = 1000.0
    initial_capital = capital
    active = []
    peak = capital
    max_dd = 0.0

    current_day = None
    current_month = None
    day_loss = 0.0
    month_loss = 0.0
    DAILY_LIMIT = 3.0   # Max daily loss %
    MONTHLY_LIMIT = 7.0  # Max monthly loss %

    for t in trades:
        dt = t['dt']

        # Close expired positions
        still_active = []
        for exit_time, position_size, pnl_dollar in active:
            if exit_time < dt:
                capital += position_size + pnl_dollar
            else:
                still_active.append((exit_time, position_size, pnl_dollar))
        active = still_active

        # Position limits
        if len(active) >= MAX_POS:
            continue

        position_size = capital * POS_PCT / 100
        if capital < position_size:
            continue

        # Daily/Monthly loss limits
        day_key = dt.strftime('%Y-%m-%d')
        if day_key != current_day:
            day_loss = 0.0
            current_day = day_key

        month_key = dt.strftime('%Y-%m')
        if month_key != current_month:
            month_loss = 0.0
            current_month = month_key

        if day_loss / initial_capital * 100 <= -DAILY_LIMIT:
            continue
        if month_loss / initial_capital * 100 <= -MONTHLY_LIMIT:
            continue

        pnl_dollar = position_size * t['pnl'] / 100
        capital -= position_size
        active.append((dt + timedelta(hours=MAX_HOURS), position_size, pnl_dollar))

        if pnl_dollar < 0:
            day_loss += abs(pnl_dollar)
            month_loss += abs(pnl_dollar)

        equity = capital + sum(ps + pd for _, ps, pd in active)
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    # Close remaining
    for _, ps, pd in active:
        capital += ps + pd

    return capital, max_dd


# ═══════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_backtest(months=range(1, 7)):
    """Run full backtest on specified months. Default: Jan-Jun 2026."""
    signals = load_signals(start_month=min(months), end_month=max(months))

    # Group by (symbol, month) for efficient cache loading
    by_pair = defaultdict(list)
    for sig in signals:
        by_pair[(sig['symbol'], sig['month'])].append(sig)

    trades = []
    pairs_done = 0
    total_pairs = len(by_pair)

    print(f'Running backtest on {len(signals)} signals across {total_pairs} pairs...')
    print()

    for (sym, mon), sigs in sorted(by_pair.items()):
        df = load_cached(sym, mon)
        if df is None:
            continue

        df_w = whale_indicator(df, STR)

        for sig in sigs:
            result = find_entry(df_w, sig['dt'])
            if result is None:
                continue

            forward, entry_idx, whale_val = result
            if whale_val < WHALE_MIN:
                continue

            pnl, reason = simulate(forward, entry_idx)
            trades.append({
                'symbol': sig['symbol'],
                'dt': sig['dt'],
                'pnl': pnl,
                'win': pnl > 0,
                'reason': reason
            })

        pairs_done += 1
        if pairs_done % 200 == 0:
            print(f'  {pairs_done}/{total_pairs} pairs, {len(trades)} trades')

    if not trades:
        print('No trades found!')
        return

    # Statistics
    wins = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    wr = len(wins) / len(trades) * 100

    final_capital, max_drawdown = run_portfolio(trades)
    monthly_return = (final_capital / 1000) ** (1 / len(months)) - 1

    print()
    print('=' * 60)
    print('🐋 HUNTER WHALE — BACKTEST RESULTS')
    print('=' * 60)
    print(f'  Period: {len(months)} months')
    print(f'  Trades: {len(trades)}')
    print(f'  Wins: {len(wins)} | Losses: {len(losses)}')
    print(f'  Win Rate: {wr:.1f}%')
    print(f'  Sum Wins: {sum(t["pnl"] for t in wins):.2f}%')
    print(f'  Sum Losses: {sum(t["pnl"] for t in losses):.2f}%')
    print(f'  Avg Win: {np.mean([t["pnl"] for t in wins]):.4f}%')
    print(f'  Avg Loss: {np.mean([t["pnl"] for t in losses]):.4f}%')
    print(f'  Final Capital: ${final_capital:,.0f}')
    print(f'  Max DD: {max_drawdown:.1f}%')
    print(f'  Monthly Return: {monthly_return * 100:+.1f}%')
    print('=' * 60)

    # By month
    print()
    print('By Month:')
    month_data = defaultdict(list)
    for t in trades:
        month_data[t['dt'].strftime('%b')].append(t)
    for m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']:
        if m not in month_data:
            continue
        mt = month_data[m]
        mw = sum(1 for t in mt if t['win'])
        print(f'  {m}: {len(mt):>4}T | WR {mw/len(mt)*100:.1f}% | Net {sum(t["pnl"] for t in mt):+.1f}%')


# ═══════════════════════════════════════════════════════════════
# LOSS ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_losses():
    """Analyze losing trades patterns."""
    signals = load_signals(start_month=1, end_month=6)
    by_pair = defaultdict(list)
    for sig in signals:
        by_pair[(sig['symbol'], sig['month'])].append(sig)

    trades = []
    for (sym, mon), sigs in sorted(by_pair.items()):
        df = load_cached(sym, mon)
        if df is None:
            continue
        df_w = whale_indicator(df, STR)
        for sig in sigs:
            result = find_entry(df_w, sig['dt'])
            if result is None:
                continue
            forward, ei, wv = result
            if wv < WHALE_MIN:
                continue
            pnl, reason = simulate(forward, ei)
            trades.append({
                'symbol': sig['symbol'],
                'dt': sig['dt'],
                'pnl': pnl,
                'win': pnl > 0,
                'reason': reason,
                'hour': sig['dt'].hour,
                'weekday': sig['dt'].strftime('%A')
            })

    losses = [t for t in trades if not t['win']]
    wins = [t for t in trades if t['win']]

    print('=' * 60)
    print('🔍 LOSS ANALYSIS')
    print('=' * 60)
    print(f'  Total: {len(trades)} | Wins: {len(wins)} | Losses: {len(losses)}')
    print(f'  WR: {len(wins)/len(trades)*100:.1f}%')
    print()

    # Exit reasons
    reasons = defaultdict(int)
    for t in losses:
        reasons[t['reason']] += 1
    print('Exit Reasons:')
    for r, c in reasons.most_common():
        print(f'  {r}: {c} ({c/len(losses)*100:.1f}%)')

    # By hour
    print()
    print('Losses by Hour:')
    hour_loss = defaultdict(list)
    for t in losses:
        hour_loss[t['hour']].append(t)
    hour_all = defaultdict(list)
    for t in trades:
        hour_all[t['hour']].append(t)
    for h in sorted(hour_loss):
        if len(hour_all[h]) < 10:
            continue
        wr_h = (len(hour_all[h]) - len(hour_loss[h])) / len(hour_all[h]) * 100
        print(f'  {h:02d}:00 | {len(hour_loss[h]):>3}L/{len(hour_all[h]):>3}T | WR {wr_h:.1f}%')

    # By weekday
    print()
    print('Losses by Weekday:')
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days:
        day_trades = [t for t in trades if t['weekday'] == day]
        day_losses = [t for t in losses if t['weekday'] == day]
        if not day_trades:
            continue
        wr_d = (len(day_trades) - len(day_losses)) / len(day_trades) * 100
        print(f'  {day:<10} | {len(day_losses):>3}L/{len(day_trades):>3}T | WR {wr_d:.1f}%')


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'backtest'

    if cmd == 'backtest':
        run_backtest()
    elif cmd == 'analyze':
        analyze_losses()
    elif cmd == 'live':
        print('Live mode: waiting for signals...')
        print('(Connect to signals feed to activate)')
    else:
        print(f'Usage: python3 hunter_whale.py [backtest|analyze|live]')
        print(f'  backtest  — Run 6-month backtest (Jan-Jun 2026)')
        print(f'  analyze   — Analyze losing trades')
        print(f'  live      — Monitor signals live')
