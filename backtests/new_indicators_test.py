#!/usr/bin/env python3
"""
EXTENDED INDICATOR TEST — 120 days, new indicators
Testing: MACD, Stochastic, ADX, CCI, Williams%R, OBV, MFI, Aroon
Combined with filters to find 10%+ pumps
"""
import ccxt
import numpy as np
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
os.makedirs(DATA_DIR, exist_ok=True)
BACKTEST_DAYS = 120
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

print(f"📊 Extended Indicator Test — {BACKTEST_DAYS} days, {len(coins)} coins")
print(f"   Testing: MACD, Stochastic, ADX, CCI, Williams%R, OBV, MFI")

_EXCHANGE = None
def get_exchange():
    global _EXCHANGE
    if _EXCHANGE is None:
        _EXCHANGE = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})
    return _EXCHANGE

# ── Fetch 120-day data ──────────────────────────────────
def fetch_120d():
    cache_file = os.path.join(DATA_DIR, 'daily_120d.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    
    exchange = get_exchange()
    since = exchange.parse8601((datetime.now() - timedelta(days=BACKTEST_DAYS)).isoformat())
    all_data = {}
    
    for i, coin in enumerate(coins):
        try:
            ohlcv = exchange.fetch_ohlcv(f"{coin}/USDT", '1d', since=since, limit=BACKTEST_DAYS+10)
            if len(ohlcv) >= 60:
                all_data[coin] = {
                    'ts': [int(o[0]) for o in ohlcv],
                    'open': [float(o[1]) for o in ohlcv],
                    'high': [float(o[2]) for o in ohlcv],
                    'low': [float(o[3]) for o in ohlcv],
                    'close': [float(o[4]) for o in ohlcv],
                    'volume': [float(o[5]) for o in ohlcv],
                }
            if (i+1) % 25 == 0:
                print(f"  📊 {i+1}/{len(coins)}")
        except:
            pass
        time.sleep(0.05)
    
    with open(cache_file, 'w') as f:
        json.dump(all_data, f)
    print(f"✅ {len(all_data)} coins fetched")
    return all_data

# ── Compute ALL indicators ──────────────────────────────
def compute_all_indicators(close, high, low, volume, open_):
    """Returns dict of all indicator Series."""
    ind = {}
    
    # ── Basic ──
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    ind['rsi'] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
    ind['pct'] = close.pct_change() * 100
    
    # ── MACD ──
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    ind['macd'] = ema12 - ema26
    ind['macd_signal'] = ind['macd'].ewm(span=9).mean()
    ind['macd_hist'] = ind['macd'] - ind['macd_signal']
    ind['macd_cross_up'] = (ind['macd'] > ind['macd_signal']) & (ind['macd'].shift() <= ind['macd_signal'].shift())
    ind['macd_below_zero'] = ind['macd'] < 0
    
    # ── Stochastic ──
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    ind['stoch_k'] = (close - low14) / (high14 - low14) * 100
    ind['stoch_d'] = ind['stoch_k'].rolling(3).mean()
    ind['stoch_oversold'] = ind['stoch_k'] < 20
    ind['stoch_cross_up'] = (ind['stoch_k'] > ind['stoch_d']) & (ind['stoch_k'].shift() <= ind['stoch_d'].shift())
    
    # ── CCI (Commodity Channel Index) ──
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    ind['cci'] = (tp - sma_tp) / (0.015 * mad)
    ind['cci_oversold'] = ind['cci'] < -100
    
    # ── Williams %R ──
    ind['willr'] = (high14 - close) / (high14 - low14) * -100
    ind['willr_oversold'] = ind['willr'] < -80
    
    # ── ADX (trend strength) ──
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift()),
        'lc': abs(low - close.shift()),
    }).max(axis=1)
    atr = tr.rolling(14).mean()
    up = high - high.shift()
    down = low.shift() - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=close.index)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    ind['adx'] = dx.rolling(14).mean()
    ind['adx_strong'] = ind['adx'] > 25
    
    # ── OBV (On-Balance Volume) ──
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    ind['obv'] = pd.Series(obv, index=close.index)
    ind['obv_sma'] = ind['obv'].rolling(20).mean()
    ind['obv_rising'] = ind['obv'] > ind['obv_sma']
    
    # ── MFI (Money Flow Index) ──
    typical = tp
    raw_mf = typical * volume
    pos_mf = pd.Series(np.where(typical > typical.shift(), raw_mf, 0), index=close.index)
    neg_mf = pd.Series(np.where(typical < typical.shift(), raw_mf, 0), index=close.index)
    mf_ratio = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum()
    ind['mfi'] = 100 - (100 / (1 + mf_ratio))
    ind['mfi_oversold'] = ind['mfi'] < 20
    
    # ── Price Position ──
    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    ind['range_pos'] = (close - low20) / (high20 - low20)
    
    # ── Volume ──
    ind['vol_ratio'] = volume / volume.rolling(20).mean()
    ind['vol_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()
    
    # ── Red streak ──
    red = (ind['pct'] < 0).astype(int)
    streak = [0]
    for i in range(1, len(red)):
        streak.append(streak[-1] + 1 if red.iloc[i] else 0)
    ind['red_streak'] = pd.Series(streak, index=red.index)
    
    # ── Bollinger ──
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    ind['bb_pos'] = (close - (sma20 - 2*std20)) / (4*std20)
    
    return ind

# ── Strategy Definitions (combining new indicators) ─────
STRATEGIES = []

def make_entry(name, fn):
    STRATEGIES.append((name, fn))

# 1: RSI<30 + Stoch oversold + cross up
make_entry("RSI30+StochOS+Cross", lambda ind, i: (
    ind['rsi'].iloc[i] < 30 and ind['stoch_oversold'].iloc[i] and 
    ind['stoch_cross_up'].iloc[i] and not np.isnan(ind['rsi'].iloc[i])))

# 2: RSI<30 + CCI<-100 (extreme oversold)
make_entry("RSI30+CCI-100", lambda ind, i: (
    ind['rsi'].iloc[i] < 30 and ind['cci_oversold'].iloc[i] and
    not np.isnan(ind['rsi'].iloc[i])))

# 3: RSI<30 + Williams%R<-80
make_entry("RSI30+WillR-80", lambda ind, i: (
    ind['rsi'].iloc[i] < 30 and ind['willr_oversold'].iloc[i] and
    not np.isnan(ind['rsi'].iloc[i])))

# 4: RSI<30 + MFI<20 (money flow confirms oversold)
make_entry("RSI30+MFI<20", lambda ind, i: (
    ind['rsi'].iloc[i] < 30 and ind['mfi_oversold'].iloc[i] and
    not np.isnan(ind['rsi'].iloc[i])))

# 5: MACD cross up + RSI<40 + OBV rising
make_entry("MACDcross+RSI40+OBVup", lambda ind, i: (
    ind['macd_cross_up'].iloc[i] and ind['rsi'].iloc[i] < 40 and
    ind['obv_rising'].iloc[i] and not np.isnan(ind['rsi'].iloc[i])))

# 6: MACD<0 + RSI<25 + Stoch<20 (triple oversold)
make_entry("MACDneg+RSI25+Stoch20", lambda ind, i: (
    ind['macd_below_zero'].iloc[i] and ind['rsi'].iloc[i] < 25 and
    ind['stoch_oversold'].iloc[i] and not np.isnan(ind['rsi'].iloc[i])))

# 7: RSI<25 + CCI<-200 (extreme) + PrevRed
make_entry("RSI25+CCI-200+Red", lambda ind, i: (
    ind['rsi'].iloc[i] < 25 and ind['cci'].iloc[i] < -200 and
    ind['pct'].iloc[i] < 0 and not np.isnan(ind['rsi'].iloc[i])))

# 8: Stoch<10 + WillR<-90 + PrevRed (deep oversold combo)
make_entry("Stoch10+WillR90+Red", lambda ind, i: (
    ind['stoch_k'].iloc[i] < 10 and ind['willr'].iloc[i] < -90 and
    ind['pct'].iloc[i] < 0 and not np.isnan(ind['rsi'].iloc[i])))

# 9: RSI<30 + MFI<20 + OBV divergence (price down, OBV up)
make_entry("RSI30+MFI20+OBVdiv", lambda ind, i: (
    ind['rsi'].iloc[i] < 30 and ind['mfi_oversold'].iloc[i] and
    ind['obv_rising'].iloc[i] and ind['pct'].iloc[i] < 0 and
    not np.isnan(ind['rsi'].iloc[i])))

# 10: RSI<20 + Stoch<5 + CCI<-300 (nuclear oversold)
make_entry("RSI20+Stoch5+CCI-300", lambda ind, i: (
    ind['rsi'].iloc[i] < 20 and ind['stoch_k'].iloc[i] < 5 and
    ind['cci'].iloc[i] < -300 and not np.isnan(ind['rsi'].iloc[i])))

# 11: All 4 oversold signals agree (RSI+Stoch+WillR+CCI)
make_entry("4xOversold", lambda ind, i: (
    ind['rsi'].iloc[i] < 30 and ind['stoch_oversold'].iloc[i] and
    ind['willr_oversold'].iloc[i] and ind['cci_oversold'].iloc[i] and
    not np.isnan(ind['rsi'].iloc[i])))

# 12: RSI<30 + ADX>25 + PrevRed (oversold in strong trend)
make_entry("RSI30+ADX25+Red", lambda ind, i: (
    ind['rsi'].iloc[i] < 30 and ind['adx_strong'].iloc[i] and
    ind['pct'].iloc[i] < 0 and not np.isnan(ind['rsi'].iloc[i])))

# 13: MACD hist turning positive + RSI<35 + Vol
make_entry("MACDhistUp+RSI35+Vol", lambda ind, i: (
    ind['macd_hist'].iloc[i] > ind['macd_hist'].iloc[i-1] and
    ind['macd_hist'].iloc[i-1] < 0 and ind['rsi'].iloc[i] < 35 and
    ind['vol_ratio'].iloc[i] > 1.2 and not np.isnan(ind['rsi'].iloc[i])))

# ── Backtest Engine ─────────────────────────────────────
TP_SL = [(0.10, 0.05, "TP10/SL5"), (0.15, 0.06, "TP15/SL6")]

def backtest(all_data, entry_fn, tp, sl, max_hold=7):
    all_signals = []
    for coin, data in all_data.items():
        df = pd.DataFrame({
            'open': data['open'], 'high': data['high'],
            'low': data['low'], 'close': data['close'], 'volume': data['volume'],
        })
        if len(df) < 80: continue
        
        ind = compute_all_indicators(df['close'], df['high'], df['low'], df['volume'], df['open'])
        n = len(df)
        
        for i in range(50, n - 2):
            try:
                ok = entry_fn(ind, i)
            except:
                ok = False
            if not ok: continue
            
            # Confirmation: next candle must be green
            if df['close'].iloc[i+1] <= df['open'].iloc[i+1]:
                continue
            
            entry_idx = i + 1
            if entry_idx >= n: continue
            
            all_signals.append({
                'coin': coin, 'idx': entry_idx,
                'entry_price': df['close'].iloc[entry_idx],
                'date': str(df.index[entry_idx]),
            })
    
    all_signals.sort(key=lambda s: s['date'])
    trades = []
    capital = INITIAL_CAPITAL
    active = {}
    
    for sig in all_signals:
        coin = sig['coin']; entry_idx = sig['idx']
        if coin in active and active[coin] > entry_idx: continue
        
        data = all_data[coin]
        close_arr = np.array(data['close'])
        high_arr = np.array(data['high'])
        low_arr = np.array(data['low'])
        n = len(close_arr)
        
        tp_p = sig['entry_price'] * (1 + tp)
        sl_p = sig['entry_price'] * (1 - sl)
        
        exit_p = None; exit_t = None; exit_i = None
        for j in range(entry_idx + 1, min(entry_idx + max_hold, n)):
            if low_arr[j] <= sl_p:
                exit_p = sl_p; exit_t = 'SL'; exit_i = j; break
            elif high_arr[j] >= tp_p:
                exit_p = tp_p; exit_t = 'TP'; exit_i = j; break
        
        if exit_p is None:
            end = min(entry_idx + max_hold, n - 1)
            exit_p = close_arr[end]; exit_t = 'TIME'; exit_i = end
        
        pnl_pct = (exit_p / sig['entry_price'] - 1) * 100 - COMMISSION * 100
        size = capital * 0.10
        pnl_usd = size * pnl_pct / 100
        capital += pnl_usd
        
        trades.append({'pnl_pct': pnl_pct, 'pnl_usd': pnl_usd, 'type': exit_t, 'capital': capital})
        active[coin] = exit_i
        active = {k: v for k, v in active.items() if v > entry_idx}
    
    return trades, capital

# ── Run ─────────────────────────────────────────────────
print("\n── Fetching 120-day data ──")
all_data = fetch_120d()

print(f"\n{'='*100}")
print(f"📊 NEW INDICATOR STRATEGIES — {BACKTEST_DAYS}-day Backtest ({len(all_data)} coins)")
print(f"{'='*100}")

for tp, sl, label in TP_SL:
    print(f"\n{'─'*100}")
    print(f"📐 {label}")
    print(f"{'─'*100}")
    print(f"{'Strategy':<30s} {'Trades':>6s} {'WR':>6s} {'Return':>8s} {'MaxDD':>7s} {'PF':>6s} {'TP#':>5s} {'Avg':>7s}")
    print(f"{'-'*80}")
    
    for name, fn in STRATEGIES:
        trades, final_cap = backtest(all_data, fn, tp, sl)
        
        if not trades:
            print(f"{name:<30s} {'0':>6s} {'-':>6s} {'-':>8s} {'-':>7s} {'-':>6s} {'-':>5s} {'-':>7s}")
            continue
        
        df = pd.DataFrame(trades)
        wins = df[df['pnl_pct'] > 0]; losses = df[df['pnl_pct'] <= 0]
        wr = len(wins) / len(df) * 100
        eq = np.array([INITIAL_CAPITAL] + [t['capital'] for t in trades])
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        ret = (final_cap / INITIAL_CAPITAL - 1) * 100
        pf = abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum()) if len(losses) > 0 else 999
        
        ret_s = f"+{ret:.1f}%" if ret > 0 else f"{ret:.1f}%"
        print(f"{name:<30s} {len(df):>6d} {wr:>5.1f}% {ret_s:>8s} {dd.min():>6.2f}% {pf:>5.2f} {len(df[df['type']=='TP']):>5d} {df['pnl_pct'].mean():>+6.2f}%")

print(f"\n✅ Done! All new indicators tested on {BACKTEST_DAYS} days.")
