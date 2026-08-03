#!/usr/bin/env python3
"""
S/R Strategies — All Timeframes — FET/USDT
Tests 6 S/R strategies with realistic ATR-based stops
"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime
import warnings, json, os
warnings.filterwarnings('ignore')

COMM = 0.002
CAP = 1000
SYM = 'FET/USDT'
DATA_DIR = '/data/trading28/data'
os.makedirs(DATA_DIR, exist_ok=True)

def fetch(tf, days):
    cache_file = os.path.join(DATA_DIR, f'fet_{tf}_{days}d_sr.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            d = json.load(f)
        return np.array(d['c']), np.array(d['h']), np.array(d['l']), [datetime.fromtimestamp(x/1000) for x in d['ts']], np.array(d['v'])
    exchange = ccxt.binance({'timeout': 15000})
    since = exchange.parse8601((datetime.now() - pd.Timedelta(days=days+3)).isoformat())
    ohlcv = exchange.fetch_ohlcv(SYM, tf, since=since, limit=10000)
    d = {'ts':[int(o[0]) for o in ohlcv], 'c':[float(o[4]) for o in ohlcv],
         'h':[float(o[2]) for o in ohlcv], 'l':[float(o[3]) for o in ohlcv],
         'v':[float(o[5]) for o in ohlcv]}
    with open(cache_file,'w') as f: json.dump(d,f)
    return np.array(d['c']), np.array(d['h']), np.array(d['l']), [datetime.fromtimestamp(x/1000) for x in d['ts']], np.array(d['v'])

def swings(h, l, depth=5):
    n = len(h)
    highs, lows = [], []
    for i in range(depth, n-depth):
        is_h = all(h[j] <= h[i] for j in range(i-depth, i+depth+1) if j != i)
        is_l = all(l[j] >= l[i] for j in range(i-depth, i+depth+1) if j != i)
        if is_h: highs.append((i, h[i]))
        if is_l: lows.append((i, l[i]))
    return highs, lows

def atr_n(c, h, l, period=14):
    n = len(c)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    return pd.Series(tr).rolling(period).mean().values

# ── Strategies ───────────────────────────────────────

def s1_breakout(i, c, h, l, sh, sl):
    """Break above last swing high with volume confirmation"""
    for si, sp in reversed(sh):
        if si < i:
            vol_ok = True  # simplified
            return c[i] > sp and c[i-1] <= sp and vol_ok
    return False

def s2_double_bottom(i, c, h, l, sh, sl):
    """Two swing lows at same level + green candle bounce"""
    recent = [x for x in sl if x[0] < i]
    if len(recent) < 2: return False
    l1, l2 = recent[-2], recent[-1]
    if abs(l1[1]-l2[1])/l1[1] < 0.03:
        return c[i] > c[i-1] and abs(c[i]-l1[1])/l1[1] < 0.04 and c[i] > l1[1]
    return False

def s3_support_bounce_confirmed(i, c, h, l, sh, sl):
    """Touch support + wait for bullish confirmation bar (not just any green)"""
    for si, sp in reversed(sl):
        if si < i - 2:  # swing low at least 2 bars ago
            # Previous bar touched support
            prev_touch = abs(l[i-1] - sp)/sp < 0.025
            # This bar confirms: strong green, closed near high
            bullish = c[i] > c[i-1] and c[i] > (h[i]+l[i])/2 and (c[i]-l[i]) > (h[i]-c[i])*1.2
            if prev_touch and bullish:
                return True
    return False

def s4_range_breakout(i, c, h, l, sh, sl):
    """Break above consolidation — price was ranging for 20+ bars then breaks high"""
    if i < 25: return False
    lookback = min(30, i-1)
    recent_high = max(h[i-lookback:i])
    recent_low = min(l[i-lookback:i])
    range_pct = (recent_high/recent_low - 1) * 100
    # Range must be tight (<15%)
    if range_pct > 15 or range_pct < 3:
        return False
    # Break above range high
    return c[i] > recent_high and c[i-1] <= recent_high

def s5_pullback_to_broken_resistance(i, c, h, l, sh, sl):
    """Price broke above old resistance, now pulling back to it (support flip)"""
    if len(sh) < 2: return False
    # Find a swing high that was broken (price is above it now)
    for si, sp in reversed(sh):
        if si < i - 15:  # old resistance
            if c[i-5] > sp * 1.02:  # was broken 5 bars ago
                # Now pulling back to it
                pullback = abs(l[i] - sp)/sp < 0.025
                bounce = c[i] > c[i-1]
                if pullback and bounce:
                    return True
    return False

def s6_trendline_bounce(i, c, h, l, sh, sl):
    """Bounce off trendline connecting 2+ swing lows (uptrend support)"""
    if len(sl) < 2: return False
    recent = [x for x in sl if x[0] < i]
    if len(recent) < 2: return False
    # Take last 2 swing lows, project trendline to current bar
    l1, l2 = recent[-2], recent[-1]
    if l2[1] <= l1[1]: return False  # must be uptrend
    # Linear projection
    slope = (l2[1] - l1[1]) / (l2[0] - l1[0])
    projected = l2[1] + slope * (i - l2[0])
    near = abs(l[i] - projected)/projected < 0.03
    bounce = c[i] > c[i-1] and c[i] > projected
    return near and bounce

# ── Backtest ─────────────────────────────────────────

def backtest(c, h, l, atr, entry_fn, max_hold, atr_sl=2.0, tp_atr=4.0, sh=None, sl_list=None):
    n = len(c)
    trades = []
    in_trade = False
    ei = ep = sl_px = tp_px = 0
    
    for i in range(10, n):
        if not in_trade and entry_fn(i, c, h, l, sh, sl_list):
            in_trade = True
            ei = i
            ep = c[i]
            sl_px = ep - atr[i] * atr_sl  # ATR-based stop
            tp_px = ep + atr[i] * tp_atr  # ATR-based target
            continue
        
        if not in_trade:
            continue
        
        ex_px = ex_type = None
        
        if l[i] <= sl_px:
            ex_px = sl_px; ex_type = 'SL'
        elif h[i] >= tp_px:
            ex_px = tp_px; ex_type = 'TP'
        elif i - ei >= max_hold:
            ex_px = c[i]; ex_type = 'TIME'
        
        if ex_px is not None:
            pnl = (ex_px/ep - 1)*100 - COMM*100
            trades.append({'pnl':pnl, 'type':ex_type, 'bars':i-ei, 'entry':ep, 'exit':ex_px})
            in_trade = False
    
    if not trades:
        return {'trades':0,'wins':0,'losses':0,'wr':0,'final':CAP,'dd':0,'sh':0,'avg_w':0,'avg_l':0,'rr':0,'net':0,'tp':0,'sl':0,'time':0}
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    curve = [CAP]
    for t in trades:
        sz = curve[-1] * 0.10
        curve.append(curve[-1] + sz * t['pnl']/100)
    
    final = curve[-1]
    net = (final/CAP - 1)*100
    wr = len(wins)/len(trades)*100
    dr = np.diff(curve)/curve[:-1]
    sh_val = np.mean(dr)/np.std(dr)*np.sqrt(252) if len(dr)>1 and np.std(dr)>0 else 0
    peak = np.maximum.accumulate(curve)
    dd = np.min((curve-peak)/peak*100)
    avg_w = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_l = np.mean([t['pnl'] for t in losses]) if losses else 0
    rr = abs(avg_w/avg_l) if avg_l!=0 else 0
    
    tp_n = sum(1 for t in trades if t['type']=='TP')
    sl_n = sum(1 for t in trades if t['type']=='SL')
    time_n = sum(1 for t in trades if t['type']=='TIME')
    
    return {'trades':len(trades),'wins':len(wins),'losses':len(losses),'wr':wr,
            'final':final,'dd':dd,'sh':sh_val,'avg_w':avg_w,'avg_l':avg_l,'rr':rr,'net':net,
            'tp':tp_n,'sl':sl_n,'time':time_n}

# ── Run ──────────────────────────────────────────────

strategies = [
    ("1️⃣ اختراق قمة متأرجحة", s1_breakout, 2.0, 4.0),
    ("2️⃣ قاع مزدوج", s2_double_bottom, 2.0, 4.0),
    ("3️⃣ ارتداد مؤكد من دعم", s3_support_bounce_confirmed, 2.0, 4.0),
    ("4️⃣ اختراق نطاق عرضي", s4_range_breakout, 2.0, 4.0),
    ("5️⃣ ارتداد لمقاومة مكسورة", s5_pullback_to_broken_resistance, 2.0, 4.0),
    ("6️⃣ ارتداد من خط اتجاه", s6_trendline_bounce, 2.0, 4.0),
]

timeframes = [
    ('15m', 30, 200, '15 دقيقة'),
    ('1h', 90, 80, '1 ساعة'),
    ('4h', 365, 40, '4 ساعات'),
    ('1d', 1095, 40, 'يومي'),
]

print("="*80)
print(f"🧪 6 استراتيجيات دعم ومقاومة — FET/USDT — مع SL/TP بنسبة ATR")
print("="*80)

for tf, days, max_h, tf_ar in timeframes:
    print(f"\n{'─'*80}")
    print(f"⏱️ {tf_ar} ({tf}) | {days} يوم | أقصى صبر {max_h} شمعة | SL=2ATR | TP=4ATR")
    print(f"{'─'*80}")
    print(f"{'الاستراتيجية':<30s} {'T':>3s} {'WR':>4s} {'💰 محفظة':>8s} {'سحب':>5s} {'شارپ':>6s} {'R:R':>5s} {'🎯TP':>4s} {'🛑SL':>4s} {'⏰وقت':>4s}")
    print("-"*80)
    
    c, h, l, ts, v = fetch(tf, days)
    atr = atr_n(c, h, l, 14)
    sh_sw, sl_sw = swings(h, l, depth=5)
    
    for sname, sfn, atr_sl, atr_tp in strategies:
        m = backtest(c, h, l, atr, sfn, max_h, atr_sl, atr_tp, sh_sw, sl_sw)
        if m['trades'] > 0:
            print(f"  {sname:<28s} {m['trades']:>3d} {m['wr']:>3.0f}% ${m['final']:>7.0f} {m['dd']:>+4.1f}% {m['sh']:>+5.2f} {m['rr']:>4.2f} {m['tp']:>4d} {m['sl']:>4d} {m['time']:>4d}")
        else:
            print(f"  {sname:<28s} 0 trades")

print(f"\n✅ Done")
