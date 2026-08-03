#!/usr/bin/env python3
"""
Sniper Strategy Backtest — FET/USDT multi-timeframe
Pine Script v6 → Python port
Close-only simulation, always-in-market reversal
"""

import ccxt, pandas as pd, numpy as np, sys, os
from datetime import datetime, timedelta

sys.path.insert(0, '/data/trading28')
os.environ['PYTHONUNBUFFERED'] = '1'

# ─── Config ───
SYMBOL = 'FET/USDT'
COMM = 0.002  # 0.2%
DAYS = 180
INITIAL_CAPITAL = 1000

# Chart TFs to test + their signal TFs
TF_PAIRS = [
    ('3m',  '15m'),
    ('5m',  '30m'),
    ('15m', '1h'),
    ('15m', '4h'),
    ('30m', '4h'),
    ('1h',  '4h'),
    ('1h',  '1d'),
]

# ─── Helpers ───
def fetch_ohlcv(symbol, tf, days):
    exchange = ccxt.binance({'timeout': 15000})
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    all_candles = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(all_candles, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    df.sort_index(inplace=True)
    return df

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def supertrend(df, factor, period):
    atr = df['high'].sub(df['low']).rolling(period).mean()  # simplified ATR
    hl2 = (df['high'] + df['low']) / 2
    up = hl2 - factor * atr
    dn = hl2 + factor * atr
    
    trend = pd.Series(0, index=df.index)
    trend_up = pd.Series(np.nan, index=df.index)
    trend_dn = pd.Series(np.nan, index=df.index)
    
    for i in range(period, len(df)):
        prev_close = df['close'].iloc[i-1]
        if pd.notna(trend_dn.iloc[i-1]) and prev_close < trend_dn.iloc[i-1]:
            trend.iloc[i] = -1
            trend_dn.iloc[i] = min(dn.iloc[i], trend_dn.iloc[i-1]) if pd.notna(trend_dn.iloc[i-1]) else dn.iloc[i]
        elif pd.notna(trend_up.iloc[i-1]) and prev_close > trend_up.iloc[i-1]:
            trend.iloc[i] = 1
            trend_up.iloc[i] = max(up.iloc[i], trend_up.iloc[i-1]) if pd.notna(trend_up.iloc[i-1]) else up.iloc[i]
        elif prev_close > trend_dn.iloc[i-1] if pd.notna(trend_dn.iloc[i-1]) else False:
            trend.iloc[i] = 1
            trend_up.iloc[i] = up.iloc[i]
        else:
            trend.iloc[i] = -1
            trend_dn.iloc[i] = dn.iloc[i]
    
    return trend

def compute_indicators(df):
    df = df.copy()
    df['ema13'] = ema(df['close'], 13)
    df['ema21'] = ema(df['close'], 21)
    df['ema200'] = ema(df['close'], 200)
    df['rsi14'] = rsi(df['close'], 14)
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['st_trend'] = supertrend(df, 2.0, 7)
    df['ema13_rising'] = df['ema13'].diff(2) > 0
    df['ema21_rising'] = df['ema21'].diff(2) > 0
    return df

def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ─── Main Backtest ───
print("Fetching FET/USDT data (180 days)...")

# Fetch base data at highest resolution needed
df_3m = fetch_ohlcv(SYMBOL, '3m', DAYS)
df_15m = fetch_ohlcv(SYMBOL, '15m', DAYS)
df_1h = fetch_ohlcv(SYMBOL, '1h', DAYS)

print(f"Data fetched: 3m={len(df_3m)} candles, 15m={len(df_15m)} candles, 1h={len(df_1h)} candles")

# ─── Run each TF pair ───
results = []

for chart_tf, signal_tf in TF_PAIRS:
    print(f"\n{'='*60}")
    print(f"Testing: Chart={chart_tf}, Signal={signal_tf}")
    
    # Load chart TF data
    if chart_tf in ('3m', '5m'):
        df_chart = df_3m.resample('5min' if chart_tf == '5m' else '3min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna() if chart_tf == '5m' else df_3m.copy()
    elif chart_tf == '15m':
        df_chart = df_15m.copy()
    elif chart_tf == '30m':
        df_chart = df_15m.resample('30min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
    elif chart_tf == '1h':
        df_chart = df_1h.copy()
    else:
        continue
    
    # Compute indicators on chart TF
    df_chart = compute_indicators(df_chart)
    
    # Build HTF signal data
    if signal_tf == '15m':
        df_htf = df_15m.copy()
    elif signal_tf == '30m':
        df_htf = df_15m.resample('30min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
    elif signal_tf == '1h':
        df_htf = df_1h.copy()
    elif signal_tf == '4h':
        df_htf = df_1h.resample('4h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
    elif signal_tf == '1d':
        df_htf = df_1h.resample('1d').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
    else:
        continue
    
    # HTF crossover detection: close crosses over open
    htf_bullish = df_htf['close'] > df_htf['open']
    htf_cross_up = htf_bullish & (~htf_bullish.shift(1).fillna(False))
    htf_cross_down = (~htf_bullish) & (htf_bullish.shift(1).fillna(False))
    
    # Forward-fill HTF signals to chart TF
    htf_cross_up_ff = htf_cross_up.reindex(df_chart.index, method='ffill').fillna(False).astype(bool)
    htf_cross_down_ff = htf_cross_down.reindex(df_chart.index, method='ffill').fillna(False).astype(bool)
    htf_bullish_ff = htf_bullish.reindex(df_chart.index, method='ffill').fillna(False).astype(bool)
    htf_bearish_ff = (~htf_bullish).reindex(df_chart.index, method='ffill').fillna(False).astype(bool)
    
    # Entry conditions on chart TF
    st_up = df_chart['st_trend'] == 1
    st_dn = df_chart['st_trend'] == -1
    
    long_signal = (
        htf_cross_up_ff & htf_bullish_ff &
        (df_chart['ema13'] > df_chart['ema21']) &
        df_chart['ema13_rising'] &
        st_up &
        (df_chart['close'] > df_chart['ema200']) &
        (df_chart['volume'] > df_chart['vol_sma20']) &
        (df_chart['rsi14'] < 70)
    )
    
    short_signal = (
        htf_cross_down_ff & htf_bearish_ff &
        (df_chart['ema13'] < df_chart['ema21']) &
        (~df_chart['ema13_rising']) &
        st_dn &
        (df_chart['close'] < df_chart['ema200']) &
        (df_chart['volume'] > df_chart['vol_sma20']) &
        (df_chart['rsi14'] > 30)
    )
    
    # Trade simulation: always-in-market reversal
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0
    entry_bar = None
    trades = []
    equity = INITIAL_CAPITAL
    equity_curve = [INITIAL_CAPITAL]
    
    closes = df_chart['close'].values
    idx = df_chart.index
    
    for i in range(200, len(df_chart)):  # skip first 200 bars for EMA warmup
        if position == 0:
            if long_signal.iloc[i]:
                position = 1
                entry_price = closes[i]
                entry_bar = idx[i]
            elif short_signal.iloc[i]:
                position = -1
                entry_price = closes[i]
                entry_bar = idx[i]
        elif position == 1:
            # Exit on short signal (reversal)
            if short_signal.iloc[i]:
                exit_price = closes[i]
                pnl_pct = (exit_price / entry_price - 1) * 100 - COMM * 100
                trades.append({'entry_bar': entry_bar, 'exit_bar': idx[i], 'side': 'LONG',
                              'entry': entry_price, 'exit': exit_price, 'pnl_pct': pnl_pct})
                equity *= (1 + pnl_pct / 100)
                
                # Flip to short
                position = -1
                entry_price = closes[i]
                entry_bar = idx[i]
        elif position == -1:
            # Exit on long signal (reversal)
            if long_signal.iloc[i]:
                exit_price = closes[i]
                pnl_pct = (1 - exit_price / entry_price) * 100 - COMM * 100
                trades.append({'entry_bar': entry_bar, 'exit_bar': idx[i], 'side': 'SHORT',
                              'entry': entry_price, 'exit': exit_price, 'pnl_pct': pnl_pct})
                equity *= (1 + pnl_pct / 100)
                
                # Flip to long
                position = 1
                entry_price = closes[i]
                entry_bar = idx[i]
        
        equity_curve.append(equity)
    
    # Close any open position at end
    if position != 0:
        exit_price = closes[-1]
        if position == 1:
            pnl_pct = (exit_price / entry_price - 1) * 100 - COMM * 100
        else:
            pnl_pct = (1 - exit_price / entry_price) * 100 - COMM * 100
        trades.append({'entry_bar': entry_bar, 'exit_bar': idx[-1], 'side': 'LONG' if position == 1 else 'SHORT',
                      'entry': entry_price, 'exit': exit_price, 'pnl_pct': pnl_pct})
        equity *= (1 + pnl_pct / 100)
    
    if not trades:
        print(f"  ❌ Zero trades — skipping")
        continue
    
    # Metrics
    tdf = pd.DataFrame(trades)
    n_trades = len(tdf)
    wins = tdf[tdf['pnl_pct'] > 0]
    losses = tdf[tdf['pnl_pct'] <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    wr = n_wins / n_trades * 100 if n_trades > 0 else 0
    avg_win = wins['pnl_pct'].mean() if n_wins > 0 else 0
    avg_loss = losses['pnl_pct'].mean() if n_losses > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    total_profit = wins['pnl_pct'].sum() if n_wins > 0 else 0
    total_loss = losses['pnl_pct'].sum() if n_losses > 0 else 0
    net_pnl = total_profit + total_loss
    
    # Drawdown
    eq = pd.Series(equity_curve)
    peak = eq.expanding().max()
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()
    
    # Sharpe
    daily_rets = pd.Series(eq).pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(365)) if daily_rets.std() > 0 else 0
    annual_return = (equity / INITIAL_CAPITAL) ** (365 / DAYS) - 1
    
    # Signal counts
    n_long_signals = long_signal.sum()
    n_short_signals = short_signal.sum()
    
    results.append({
        'chart_tf': chart_tf,
        'signal_tf': signal_tf,
        'n_trades': n_trades,
        'n_wins': n_wins,
        'n_losses': n_losses,
        'wr': wr,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr': rr,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'net_pnl': net_pnl,
        'equity': equity,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'annual_return': annual_return * 100,
        'n_long_signals': n_long_signals,
        'n_short_signals': n_short_signals,
        'candles': len(df_chart),
    })

# ─── Report ───
print("\n" + "=" * 70)
print("SNIPER STRATEGY — FET/USDT Multi-TF Backtest Results")
print(f"Period: Last {DAYS} days | Commission: 0.2%")
print("=" * 70)

for r in results:
    print(f"""
📊 Chart={r['chart_tf']}, Signal HTF={r['signal_tf']} ({r['candles']} candles)
─────────────────────────────────────────────
📋 صفقات: {r['n_trades']} | 🟢 {r['n_wins']} | 🔴 {r['n_losses']} | 📈 WR: {r['wr']:.1f}%
💵 إجمالي الربح: +{r['total_profit']:.2f}% | 💸 إجمالي الخسارة: {r['total_loss']:.2f}% | 💰 صافي: {r['net_pnl']:.2f}%
🟢 متوسط الربح: +{r['avg_win']:.3f}% | 🔴 متوسط الخسارة: {r['avg_loss']:.3f}% | 📊 R:R: {r['rr']:.2f}x
📊 شارپ: {r['sharpe']:.2f} | 📉 أقصى سحب: {r['max_dd']:.1f}%
🏦 المحفظة: ${INITIAL_CAPITAL} → ${r['equity']:.2f} (+{(r['equity']/INITIAL_CAPITAL - 1)*100:.1f}%)
📈 عائد سنوي: {r['annual_return']:.1f}%
🔔 إشارات شراء: {r['n_long_signals']} | بيع: {r['n_short_signals']}
""")

# ─── Summary table ───
print("\n📊 SUMMARY TABLE")
print(f"{'Chart':>6} | {'Signal':>6} | {'Trades':>6} | {'WR':>7} | {'AvgWin':>7} | {'AvgLoss':>7} | {'R:R':>5} | {'DD':>7} | {'PnL':>8} | {'Sharpe':>6}")
print("-" * 95)
for r in results:
    print(f"{r['chart_tf']:>6} | {r['signal_tf']:>6} | {r['n_trades']:>6} | {r['wr']:>6.1f}% | {r['avg_win']:>+6.3f}% | {r['avg_loss']:>+6.3f}% | {r['rr']:>4.2f}x | {r['max_dd']:>6.1f}% | {r['equity']-1000:>+8.0f} | {r['sharpe']:>6.2f}")
