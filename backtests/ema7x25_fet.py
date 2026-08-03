#!/usr/bin/env python3
"""
EMA7 × EMA25 Crossover — FET/USDT — All Timeframes
Buy: EMA7 crosses ABOVE EMA25
Sell: EMA7 crosses BELOW EMA25
"""
import ccxt, numpy as np, pandas as pd, json, os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

COMMISSION = 0.002
INITIAL_CAPITAL = 1000
SYMBOL = 'FET/USDT'
DATA_DIR = '/data/trading28/data'
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_fet(tf, days):
    cache_key = f'fet_{tf}_{days}d'
    cache_file = os.path.join(DATA_DIR, f'{cache_key}.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    exchange = ccxt.binance({'timeout': 15000})
    since = exchange.parse8601((datetime.now() - timedelta(days=days+3)).isoformat())
    ohlcv = exchange.fetch_ohlcv(SYMBOL, tf, since=since, limit=10000)
    data = {
        'ts': [int(o[0]) for o in ohlcv],
        'open': [float(o[1]) for o in ohlcv],
        'high': [float(o[2]) for o in ohlcv],
        'low': [float(o[3]) for o in ohlcv],
        'close': [float(o[4]) for o in ohlcv],
        'volume': [float(o[5]) for o in ohlcv],
    }
    with open(cache_file, 'w') as f:
        json.dump(data, f)
    return data

def backtest_cross(data, ema_fast=7, ema_slow=25, tp=None, sl=None, max_hold=None):
    c = np.array(data['close'])
    h = np.array(data['high'])
    l = np.array(data['low'])
    n = len(c)
    
    ema_f = pd.Series(c).ewm(span=ema_fast).mean().values
    ema_s = pd.Series(c).ewm(span=ema_slow).mean().values
    
    trades = []
    in_trade = False
    entry_idx = entry_px = 0
    
    for i in range(1, n):
        buy_sig = ema_f[i] > ema_s[i] and ema_f[i-1] <= ema_s[i-1]
        sell_sig = ema_f[i] < ema_s[i] and ema_f[i-1] >= ema_s[i-1]
        
        if buy_sig and not in_trade:
            in_trade = True
            entry_idx = i
            entry_px = c[i]
            continue
        
        if not in_trade:
            continue
        
        exit_px = exit_type = None
        exit_idx = i
        
        if tp and h[i] >= entry_px * (1+tp):
            exit_px = entry_px * (1+tp)
            exit_type = 'TP'
        elif sl and l[i] <= entry_px * (1-sl):
            exit_px = entry_px * (1-sl)
            exit_type = 'SL'
        elif sell_sig:
            exit_px = c[i]
            exit_type = 'CROSS'
        elif max_hold and (i - entry_idx >= max_hold):
            exit_px = c[i]
            exit_type = 'TIME'
        
        if exit_px is not None:
            pnl_pct = (exit_px / entry_px - 1) * 100 - COMMISSION * 100
            trades.append({
                'entry_idx': entry_idx, 'exit_idx': exit_idx,
                'entry': entry_px, 'exit': exit_px,
                'pnl_pct': pnl_pct, 'type': exit_type, 'bars': exit_idx - entry_idx,
            })
            in_trade = False
    
    return trades

def metrics(trades, days=365):
    if not trades:
        return {'trades':0,'wins':0,'losses':0,'wr':0,'pnl_net':0,'sharpe':0,'max_dd':0,
                'avg_win':0,'avg_loss':0,'rr':0,'annual':0,'final_cap':INITIAL_CAPITAL,
                'tp_n':0,'sl_n':0,'time_n':0,'cross_n':0,'bars_avg':0}
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    curve = [INITIAL_CAPITAL]
    for t in trades:
        sz = curve[-1] * 0.10
        curve.append(curve[-1] + sz * t['pnl_pct'] / 100)
    final = curve[-1]
    net = (final / INITIAL_CAPITAL - 1) * 100
    dr = np.diff(curve) / curve[:-1] if len(curve) > 1 else [0]
    sh = np.mean(dr)/np.std(dr)*np.sqrt(252) if len(dr)>1 and np.std(dr)>0 else 0
    peak = np.maximum.accumulate(curve)
    dd = np.min((curve - peak) / peak * 100)
    avg_w = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_l = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    rr = abs(avg_w/avg_l) if avg_l!=0 else 0
    ann = ((final/INITIAL_CAPITAL)**(365/days)-1)*100 if days>0 else 0
    return {
        'trades':len(trades), 'wins':len(wins), 'losses':len(losses),
        'wr':len(wins)/len(trades)*100, 'pnl_net':net, 'sharpe':sh, 'max_dd':dd,
        'avg_win':avg_w, 'avg_loss':avg_l, 'rr':rr, 'annual':ann, 'final_cap':final,
        'tp_n':sum(1 for t in trades if t['type']=='TP'),
        'sl_n':sum(1 for t in trades if t['type']=='SL'),
        'time_n':sum(1 for t in trades if t['type']=='TIME'),
        'cross_n':sum(1 for t in trades if t['type']=='CROSS'),
        'bars_avg':np.mean([t['bars'] for t in trades]),
    }

# ── RUN ────────────────────────────────────────────
timeframes = [
    ('15m', 96*4, '15 دقيقة'),
    ('1h', 24*14, '1 ساعة'),
    ('4h', 6*30, '4 ساعات'),
    ('1d', 365, 'يومي'),
]
periods = [30, 90, 365]

print("="*70)
print(f"🧪 EMA7 × EMA25 Crossover — {SYMBOL}")
print("="*70)

for tf, max_hold, tf_ar in timeframes:
    print(f"\n{'─'*70}")
    print(f"⏱️ {tf_ar} ({tf}) — أقصى صبر {max_hold} شمعة")
    print(f"{'─'*70}")
    
    for days in periods:
        data = fetch_fet(tf, days)
        
        # 3 variants
        variants = [
            ("EMA7×25 تقاطع فقط", None, None),
            ("+ TP 2% / SL 1.5%", 0.02, 0.015),
            ("+ TP 3% / SL 2%", 0.03, 0.02),
        ]
        
        for label, tp, sl in variants:
            trades = backtest_cross(data, 7, 25, tp=tp, sl=sl, max_hold=max_hold)
            m = metrics(trades, days)
            
            if m['trades'] > 0:
                print(f"  {label:<20s} {days:>3d}d | {m['trades']:>3d}T | "
                      f"🟢{m['wr']:.0f}% | 💰${m['final_cap']:.0f} | "
                      f"📉{m['max_dd']:.1f}% | ⚡{m['sharpe']:+.2f} | "
                      f"🎯{m['tp_n']} 🛑{m['sl_n']} ✂{m['cross_n']} ⏰{m['time_n']} | "
                      f"W{m['avg_win']:+.2f}% L{m['avg_loss']:+.2f}%")
            else:
                print(f"  {label:<20s} {days:>3d}d | 0 trades")

print("\n✅ Done")
