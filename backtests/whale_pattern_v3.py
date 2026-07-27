#!/usr/bin/env python3
"""
WHALE PATTERN v3 — Daily timeframe with tight filters
Matching pattern discovery: multiple whales + low RSI + prev day red
"""
import ccxt
import pandas as pd
import numpy as np
import json, os, time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
os.makedirs(DATA_DIR, exist_ok=True)

BACKTEST_DAYS = 90
TIMEFRAME = '1d'

# ── PARAMETERS ──────────────────────────────────────────
WHALE_STD = 2.0
MIN_WHALE_BARS = 1         # whale bars on daily
RSI_MAX = 35               # from analysis: <30 gave +11.4%, 30-40 gave +10.8%
REQUIRE_PREV_DAY_RED = True
REQUIRE_GREEN_CONFIRM = True  # next candle must be green (Filter 3)

TP_PCT = 0.07
SL_PCT = 0.03
MAX_HOLD_DAYS = 7
COMMISSION = 0.002
INITIAL_CAPITAL = 1000

# ── Load coins ──────────────────────────────────────────
with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
coins_raw = config['halal'] + config['halal2']
seen = set()
coins_raw = [c for c in coins_raw if not (c in seen or seen.add(c))]
blacklist = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','ROBO','PYTH','ANKR'}
coins = [c for c in coins_raw if c not in blacklist]

print(f"🎯 Whale Pattern v3 — Daily, tight filters")
print(f"   Whale≥{MIN_WHALE_BARS} | RSI<{RSI_MAX} | PrevRed={REQUIRE_PREV_DAY_RED} | GreenConfirm={REQUIRE_GREEN_CONFIRM}")
print(f"   TP:+{TP_PCT*100:.0f}% SL:-{SL_PCT*100:.0f}% | {BACKTEST_DAYS}d backtest")

_EXCHANGE = None
def get_exchange():
    global _EXCHANGE
    if _EXCHANGE is None:
        _EXCHANGE = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})
    return _EXCHANGE

def fetch_data():
    cache_file = os.path.join(DATA_DIR, 'daily_90d.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    
    exchange = get_exchange()
    since = exchange.parse8601((datetime.now() - timedelta(days=BACKTEST_DAYS)).isoformat())
    all_data = {}
    
    for i, coin in enumerate(coins):
        try:
            ohlcv = exchange.fetch_ohlcv(f"{coin}/USDT", TIMEFRAME, since=since, limit=BACKTEST_DAYS)
            if len(ohlcv) >= 40:
                all_data[coin] = {
                    'ts': [int(o[0]) for o in ohlcv],
                    'open': [float(o[1]) for o in ohlcv],
                    'high': [float(o[2]) for o in ohlcv],
                    'low': [float(o[3]) for o in ohlcv],
                    'close': [float(o[4]) for o in ohlcv],
                    'volume': [float(o[5]) for o in ohlcv],
                }
            if (i+1) % 30 == 0:
                print(f"  📊 {i+1}/{len(coins)}")
        except:
            pass
        time.sleep(0.05)
    
    with open(cache_file, 'w') as f:
        json.dump(all_data, f)
    print(f"✅ {len(all_data)} coins")
    return all_data

def backtest(all_data):
    all_signals = []
    
    for coin, data in all_data.items():
        close = np.array(data['close'])
        volume = np.array(data['volume'])
        n = len(close)
        if n < 40:
            continue
        
        # Whale detection (daily)
        vol_mean = pd.Series(volume).rolling(20).mean().values
        vol_std = pd.Series(volume).rolling(20).std().values
        
        # RSI(14)
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = (100 - (100 / (1 + rs))).values
        
        # Daily % change
        pct_chg = pd.Series(close).pct_change().values
        
        for i in range(30, n - 2):  # -2 for next-candle confirmation
            # Whale check
            if vol_std[i] <= 0 or np.isnan(vol_std[i]):
                continue
            if volume[i] <= vol_mean[i] + WHALE_STD * vol_std[i]:
                continue
            
            # RSI filter
            if np.isnan(rsi[i]) or rsi[i] > RSI_MAX:
                continue
            
            # Previous day red
            if REQUIRE_PREV_DAY_RED:
                if pct_chg[i] is None or np.isnan(pct_chg[i]) or pct_chg[i] >= 0:
                    continue
            
            # Green candle confirmation (next candle close > open)
            if REQUIRE_GREEN_CONFIRM:
                if close[i+1] <= data['open'][i+1]:
                    continue
            
            # Entry at close of confirmation candle
            entry_idx = i + 1
            entry_price = close[entry_idx]
            
            all_signals.append({
                'coin': coin,
                'idx': entry_idx,
                'entry_price': entry_price,
                'ts': data['ts'][entry_idx],
                'rsi': float(rsi[i]),
            })
    
    print(f"\n🎯 Signals: {len(all_signals)}")
    all_signals.sort(key=lambda s: s['ts'])
    
    trades = []
    capital = INITIAL_CAPITAL
    active = {}
    
    for sig in all_signals:
        coin = sig['coin']
        entry_idx = sig['idx']
        
        if coin in active and active[coin] > entry_idx:
            continue
        
        data = all_data[coin]
        close_arr = np.array(data['close'])
        high_arr = np.array(data['high'])
        low_arr = np.array(data['low'])
        n = len(close_arr)
        
        tp_price = sig['entry_price'] * (1 + TP_PCT)
        sl_price = sig['entry_price'] * (1 - SL_PCT)
        
        exit_price = None
        exit_type = None
        exit_idx = None
        
        for j in range(entry_idx + 1, min(entry_idx + MAX_HOLD_DAYS, n)):
            if low_arr[j] <= sl_price:
                exit_price = sl_price
                exit_type = 'SL'
                exit_idx = j
                break
            elif high_arr[j] >= tp_price:
                exit_price = tp_price
                exit_type = 'TP'
                exit_idx = j
                break
        
        if exit_price is None:
            end = min(entry_idx + MAX_HOLD_DAYS, n - 1)
            exit_price = close_arr[end]
            exit_type = 'TIME' if end == entry_idx + MAX_HOLD_DAYS else 'EOD'
            exit_idx = end
        
        pnl_pct = (exit_price / sig['entry_price'] - 1) * 100 - COMMISSION * 100
        size = capital * 0.10
        pnl_usd = size * pnl_pct / 100
        capital += pnl_usd
        
        trades.append({
            'coin': coin,
            'entry': round(sig['entry_price'], 8),
            'exit': round(exit_price, 8),
            'entry_time': datetime.fromtimestamp(sig['ts']/1000).strftime('%Y-%m-%d'),
            'type': exit_type,
            'pnl_pct': round(pnl_pct, 2),
            'pnl_usd': round(pnl_usd, 2),
            'rsi': sig['rsi'],
            'capital': round(capital, 2),
            'hold_d': exit_idx - entry_idx,
        })
        
        active[coin] = exit_idx
        active = {k: v for k, v in active.items() if v > entry_idx}
    
    return trades, capital

def report(trades, final_capital):
    if not trades:
        print("❌ No trades")
        return
    
    df = pd.DataFrame(trades)
    wins = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]
    wr = len(wins) / len(df) * 100 if len(df) > 0 else 0
    
    eq = [INITIAL_CAPITAL]
    for t in trades:
        eq.append(t['capital'])
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd_pct = (eq - peak) / peak * 100
    max_dd = dd_pct.min()
    
    total_ret = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    print(f"\n{'='*60}")
    print(f"📊 BACKTEST RESULTS (v3 — Daily)")
    print(f"{'='*60}")
    print(f"   Trades:     {len(df)}")
    print(f"   Win Rate:   {wr:.1f}%")
    print(f"   Wins:       {len(wins)} | Losses: {len(losses)}")
    if len(wins) > 0:
        print(f"   Avg Win:    +{wins['pnl_pct'].mean():.2f}%")
    if len(losses) > 0:
        print(f"   Avg Loss:   {losses['pnl_pct'].mean():.2f}%")
    print(f"   Avg Trade:  {df['pnl_pct'].mean():.2f}%")
    if len(losses) > 0:
        pf = abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum())
        print(f"   PF:         {pf:.2f}")
    print(f"   Return:     {total_ret:+.1f}%")
    print(f"   Final:      ${final_capital:,.2f}")
    print(f"   Max DD:     {max_dd:.2f}%")
    print(f"   Avg Hold:   {df['hold_d'].mean():.1f}d")
    
    print(f"\n## By Exit:")
    for et in ['TP', 'SL', 'TIME', 'EOD']:
        s = df[df['type'] == et]
        if len(s) > 0:
            print(f"   {et:5s}: {len(s)} trades, avg {s['pnl_pct'].mean():+.2f}%")
    
    print(f"\n## By RSI:")
    for r_range, (lo, hi) in [('<20', (0,20)), ('20-25', (20,25)), ('25-30', (25,30)), ('30-35', (30,35))]:
        s = df[(df['rsi'] >= lo) & (df['rsi'] < hi)]
        if len(s) > 0:
            swr = len(s[s['pnl_pct'] > 0]) / len(s) * 100
            print(f"   RSI {r_range}: {len(s)} trades, WR {swr:.0f}%, avg {s['pnl_pct'].mean():+.2f}%")
    
    # Equity curve summary
    monthly_trades = {}
    for t in trades:
        month = t['entry_time'][:7]
        if month not in monthly_trades:
            monthly_trades[month] = {'trades': 0, 'wins': 0, 'pnl': 0}
        monthly_trades[month]['trades'] += 1
        if t['pnl_pct'] > 0:
            monthly_trades[month]['wins'] += 1
        monthly_trades[month]['pnl'] += t['pnl_usd']
    
    print(f"\n## Monthly:")
    for month in sorted(monthly_trades):
        m = monthly_trades[month]
        mwr = m['wins'] / m['trades'] * 100 if m['trades'] > 0 else 0
        print(f"   {month}: {m['trades']} trades, WR {mwr:.0f}%, PnL ${m['pnl']:+.2f}")

# ── Main ─────────────────────────────────────────────────
print("=" * 60)
print("🐋 WHALE PATTERN v3 — Daily Timeframe")
print("=" * 60)

all_data = fetch_data()
trades, final_cap = backtest(all_data)
report(trades, final_cap)
print("\n✅ Done!")
