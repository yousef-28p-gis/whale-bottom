#!/usr/bin/env python3
"""
اختبار شهر يوليو 2026 — تفاصيل كل صفقة
=========================================
Signal time | Whale confirm time | Entry price | Result | TP hit first? | SL hit?
"""

import json, numpy as np, pandas as pd, os, sys
from datetime import datetime, timedelta
from collections import defaultdict

TP = 2.5
SL = 2.0
PL = 40
TRAIL = 0.10
MAX_HOURS = 2
MIN_VOL = 200000
STR = 50
WHALE_MIN = 0.35

STABLES = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDE', 'XUSD',
    'BFUSD', 'FDUSD', 'USDD', 'FRAX', 'LUSD', 'PYUSD',
    'USDJ', 'RLUSD', 'XAUT', 'USD1', 'EUR'
}

BLOCKED = {'SUPER', 'ORCA', 'VANA', 'W', 'DOGS', 'MET', 'XLM', 'BB', 'COS', 'LUNA', 'S'}

CACHE_DIR = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'


def load_cached(symbol, month):
    fpath = f'{CACHE_DIR}/{symbol}_{month}.json'
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
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
    df['entry'] = (
        df['spike'] & (df['wf'] > df['ws']) &
        (df['str'] > STR) & (df['volume'] > df['vma'] * VM)
    )
    return df


def simulate_trade(df, entry_idx, entry_price, entry_dt):
    """Simulate trade from entry. Returns detailed result."""
    tp_price = entry_price * (1 + TP / 100)
    sl_price = entry_price * (1 - SL / 100)
    pl_price = entry_price + (tp_price - entry_price) * (PL / 100)

    pl_triggered = False
    peak = entry_price
    trail_price = 0

    # Find which hits first: TP or SL
    first_hit = None
    first_hit_time = None
    first_hit_price = None

    for j in range(entry_idx + 1, len(df)):
        candle = df.iloc[j]
        hours = (j - entry_idx) * 0.25

        if hours > MAX_HOURS:
            return {
                'exit': 'timeout',
                'pnl': round((candle['close'] - entry_price) / entry_price * 100, 4),
                'exit_price': round(candle['close'], 8),
                'exit_dt': candle['ts'],
                'first_hit': first_hit,
                'first_hit_time': first_hit_time,
            }

        # Check TP
        if first_hit is None and candle['high'] >= tp_price:
            first_hit = 'tp'
            first_hit_time = candle['ts']

        # Check SL
        if first_hit is None and candle['low'] <= sl_price:
            first_hit = 'sl'
            first_hit_time = candle['ts']

        # PL logic
        if not pl_triggered and candle['high'] >= pl_price:
            pl_triggered = True
            peak = candle['high']
            trail_price = candle['high'] * (1 - TRAIL / 100)

        if pl_triggered:
            if candle['high'] > peak:
                peak = candle['high']
                trail_price = candle['high'] * (1 - TRAIL / 100)
            if candle['low'] <= trail_price:
                return {
                    'exit': 'trail',
                    'pnl': round((trail_price - entry_price) / entry_price * 100, 4),
                    'exit_price': round(trail_price, 8),
                    'exit_dt': candle['ts'],
                    'first_hit': first_hit,
                    'first_hit_time': first_hit_time,
                }

        if candle['high'] >= tp_price:  # TP hit before SL
            return {
                'exit': 'tp',
                'pnl': round(TP, 4),
                'exit_price': round(tp_price, 8),
                'exit_dt': candle['ts'],
                'first_hit': 'tp',
                'first_hit_time': first_hit_time or candle['ts'],
            }

        if candle['low'] <= sl_price:
            return {
                'exit': 'sl',
                'pnl': round(-SL, 4),
                'exit_price': round(sl_price, 8),
                'exit_dt': candle['ts'],
                'first_hit': first_hit,
                'first_hit_time': first_hit_time,
            }

    # End of data
    return {
        'exit': 'eod',
        'pnl': round((df.iloc[-1]['close'] - entry_price) / entry_price * 100, 4),
        'exit_price': round(df.iloc[-1]['close'], 8),
        'exit_dt': df.iloc[-1]['ts'],
        'first_hit': first_hit,
        'first_hit_time': first_hit_time,
    }


def find_entry(df_w, signal_dt):
    """Find whale entry nearest to signal datetime."""
    df_w['td'] = abs((df_w['ts'] - signal_dt).dt.total_seconds())
    nearest = df_w['td'].idxmin()
    forward = df_w.iloc[nearest:].reset_index(drop=True)

    for j in range(len(forward)):
        if j * 0.25 > 24:
            break
        if forward.iloc[j]['entry']:
            whale_val = float(forward.iloc[j]['whale'])
            if whale_val >= WHALE_MIN:
                return j, whale_val, forward.iloc[j]['ts'], float(forward.iloc[j]['close'])
    return None


def main():
    with open(SIGNALS_FILE) as f:
        raw = json.load(f)

    # Filter July 2026 LONG signals
    signals = []
    for s in raw:
        if s['symbol'] in STABLES or s['symbol'] in BLOCKED:
            continue
        if s.get('direction', 'LONG') != 'LONG':
            continue
        if s.get('volume_usdt', 0) < MIN_VOL:
            continue
        dt = datetime.fromisoformat(s['dt'])
        if dt.year == 2026 and dt.month == 7:
            signals.append({
                'symbol': s['symbol'],
                'dt': dt,
                'month': dt.strftime('%Y-%m'),
                'volume_usdt': s.get('volume_usdt', 0),
                'price': s.get('price', 0)
            })

    print(f'📊 إشارات يوليو 2026: {len(signals)}')
    print()

    trades = []
    no_cache = 0
    no_entry = 0

    for i, sig in enumerate(signals):
        df = load_cached(sig['symbol'], sig['month'])
        if df is None:
            no_cache += 1
            continue

        df_w = whale_indicator(df)

        # Find entry
        result = find_entry(df_w, sig['dt'])
        if result is None:
            no_entry += 1
            continue

        entry_offset, whale_val, confirm_dt, entry_price = result

        # Find the global index in df_w for simulation
        df_w['td2'] = abs((df_w['ts'] - confirm_dt).dt.total_seconds())
        entry_idx = df_w['td2'].idxmin()

        # Simulate
        sim = simulate_trade(df_w, entry_idx, entry_price, confirm_dt)

        trades.append({
            'symbol': sig['symbol'],
            'signal_dt': sig['dt'],
            'signal_price': sig['price'],
            'confirm_dt': confirm_dt,
            'entry_price': entry_price,
            'whale_val': whale_val,
            **sim
        })

    print(f'✅ صفقات مكتملة: {len(trades)}')
    print(f'❌ بدون كاش: {no_cache}')
    print(f'⏳ بدون دخول حوت: {no_entry}')
    print()
    print('=' * 80)

    # Show each trade
    wins = 0
    losses = 0
    tp_hits = 0
    sl_hits = 0
    total_pnl = 0

    for t in sorted(trades, key=lambda x: x['signal_dt']):
        pnl = t['pnl']
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        if t['exit'] == 'tp':
            tp_hits += 1
        elif t['exit'] == 'sl':
            sl_hits += 1

        emoji = '🟢' if pnl > 0 else '🔴'

        # Time difference
        diff = (t['confirm_dt'] - t['signal_dt']).total_seconds() / 60

        print(f'{emoji} {t["symbol"]:<12} | إشارة: {t["signal_dt"].strftime("%m/%d %H:%M")}')
        print(f'    تأكيد: {t["confirm_dt"].strftime("%m/%d %H:%M")} | فرق: {diff:.0f}د')
        print(f'    سعر الإشارة: {t["signal_price"]} | سعر الدخول: {t["entry_price"]}')
        print(f'    حوت: {t["whale_val"]:.3f} | خروج: {t["exit"]} | PnL: {pnl:+.2f}%')
        if t['first_hit']:
            print(f'    ⚡ أول اصطدام: {t["first_hit"]} عند {t["first_hit_time"].strftime("%m/%d %H:%M") if t["first_hit_time"] else "N/A"}')
        print()

    print('=' * 80)
    print(f'📊 الإجمالي: {len(trades)} صفقة')
    print(f'   ربحان: {wins} ({wins/len(trades)*100:.1f}%)')
    print(f'   خسران: {losses} ({losses/len(trades)*100:.1f}%)')
    print(f'   TP: {tp_hits} | SL: {sl_hits} | غيرهم: {len(trades)-tp_hits-sl_hits}')
    print(f'   مجموع PnL: {total_pnl:+.2f}%')
    if 'w' in locals() and wins > 0:
        print(f'   متوسط ربح: {sum(t["pnl"] for t in trades if t["pnl"]>0)/wins:+.2f}%')
        print(f'   متوسط خسارة: {sum(t["pnl"] for t in trades if t["pnl"]<0)/losses:+.2f}%')


if __name__ == '__main__':
    main()
