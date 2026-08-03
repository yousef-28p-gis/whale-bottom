#!/usr/bin/env python3
"""
Bull Flag Pattern Strategy — FET/USDT — All Timeframes
Structure: Pole (strong up move) → Flag (consolidation) → Breakout entry
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
    cache_file = os.path.join(DATA_DIR, f'fet_{tf}_{days}d_flag.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            d = json.load(f)
        return (np.array(d['c']), np.array(d['h']), np.array(d['l']),
                np.array(d['o']), np.array(d['v']),
                [datetime.fromtimestamp(x/1000) for x in d['ts']])
    exchange = ccxt.binance({'timeout': 15000})
    since = exchange.parse8601((datetime.now() - pd.Timedelta(days=days+5)).isoformat())
    ohlcv = exchange.fetch_ohlcv(SYM, tf, since=since, limit=10000)
    d = {'ts':[int(o[0]) for o in ohlcv],
         'o':[float(o[1]) for o in ohlcv], 'h':[float(o[2]) for o in ohlcv],
         'l':[float(o[3]) for o in ohlcv], 'c':[float(o[4]) for o in ohlcv],
         'v':[float(o[5]) for o in ohlcv]}
    with open(cache_file,'w') as f: json.dump(d,f)
    return (np.array(d['c']), np.array(d['h']), np.array(d['l']),
            np.array(d['o']), np.array(d['v']),
            [datetime.fromtimestamp(x/1000) for x in d['ts']])

def atr_n(c, h, l, period=14):
    n = len(c)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    return pd.Series(tr).rolling(period).mean().values

def detect_bull_flag(i, c, h, l, o, v):
    """
    Detect bull flag pattern ending at bar i-1.
    Returns (pole_start, pole_end, flag_high, flag_low, flag_top_line) or None.
    
    Bull flag:
    1. Pole: strong up move (5-20 bars, >5% gain, mostly bullish candles)
    2. Flag: 3-15 bars consolidation, lower highs, contained in channel
    3. Flag channel slopes slightly down or sideways
    """
    if i < 15:
        return None
    
    # Look for a pole in the last 5-20 bars before the flag
    for pole_len in range(5, min(21, i-5)):
        pole_end = i - 4  # flag starts after pole
        pole_start = pole_end - pole_len
        
        if pole_start < 0:
            continue
        
        # Pole must be a strong up move
        pole_move = (c[pole_end] / c[pole_start] - 1) * 100
        if pole_move < 5:  # minimum 5% pole
            continue
        
        # Pole should have mostly bullish candles (60%+)
        bullish_bars = sum(1 for j in range(pole_start, pole_end+1) if c[j] > o[j])
        if bullish_bars / pole_len < 0.55:
            continue
        
        # Pole volume should be above average
        pole_vol = np.mean(v[pole_start:pole_end+1])
        avg_vol = np.mean(v[max(0,pole_start-20):pole_start])
        if avg_vol > 0 and pole_vol < avg_vol * 1.1:
            continue
        
        # Now check the flag (pole_end+1 to i-1)
        flag_start = pole_end + 1
        flag_end = i - 1
        flag_len = flag_end - flag_start + 1
        
        if flag_len < 3 or flag_len > 15:
            continue
        
        # Flag: find the channel (lower highs, contained)
        flag_highs = h[flag_start:flag_end+1]
        flag_lows = l[flag_start:flag_end+1]
        flag_closes = c[flag_start:flag_end+1]
        
        # Upper trendline: connect pole peak to flag highs (should be descending or flat)
        pole_peak = max(h[pole_start:pole_end+1])
        
        # Fit upper trendline: from pole_peak to flag highs
        # Flag should have lower highs
        flag_max_h = max(flag_highs)
        if flag_max_h > pole_peak * 1.01:  # flag shouldn't exceed pole peak much
            continue
        
        # Lower boundary: horizontal or slightly descending support
        flag_min_l = min(flag_lows)
        
        # Flag channel width should be reasonable (2-8% for crypto)
        channel_pct = (flag_max_h / flag_min_l - 1) * 100
        if channel_pct < 1.5 or channel_pct > 12:
            continue
        
        # Flag volume should be declining (consolidation)
        flag_vol = np.mean(v[flag_start:flag_end+1])
        if pole_vol > 0 and flag_vol > pole_vol * 0.85:
            continue
        
        # Retracement should be < 50% of pole (preferably < 38.2%)
        retrace = (pole_peak - flag_min_l) / (pole_peak - min(l[pole_start:pole_end+1])) * 100
        if retrace > 50:
            continue
        
        # Check that flag has at least 2 touches on the upper boundary
        upper_touches = sum(1 for j in range(flag_start, flag_end+1)
                          if abs(h[j] - flag_max_h)/flag_max_h < 0.015)
        if upper_touches < 2:
            continue
        
        return {
            'pole_start': pole_start,
            'pole_end': pole_end,
            'pole_move': pole_move,
            'flag_high': flag_max_h,
            'flag_low': flag_min_l,
            'pole_peak': pole_peak,
            'pole_low': min(l[pole_start:pole_end+1]),
            'flag_len': flag_len,
            'channel_pct': channel_pct,
        }
    
    return None

def backtest_flag(c, h, l, o, v, atr, max_hold, atr_sl=1.5, use_target=True):
    n = len(c)
    trades = []
    in_trade = False
    ei = ep = sl_px = tp_px = 0
    target_pct = 0
    cooldown = 0
    
    for i in range(20, n):
        if cooldown > 0:
            cooldown -= 1
        
        if not in_trade and cooldown == 0:
            flag = detect_bull_flag(i, c, h, l, o, v)
            if flag is None:
                continue
            
            # Entry: breakout above flag high
            if c[i] > flag['flag_high'] and c[i-1] <= flag['flag_high']:
                in_trade = True
                ei = i
                ep = c[i]
                sl_px = flag['flag_low'] - atr[i] * 0.5  # below flag low
                # Measured move target: pole height projected from breakout
                pole_height = flag['pole_peak'] - flag['pole_low']
                if use_target:
                    tp_px = ep + pole_height
                else:
                    tp_px = ep + atr[i] * 5
            continue
        
        if not in_trade:
            continue
        
        ex_px = ex_type = None
        
        if l[i] <= sl_px:
            ex_px = sl_px; ex_type = 'SL'
        elif use_target and h[i] >= tp_px:
            ex_px = tp_px; ex_type = 'TP'
        elif i - ei >= max_hold:
            ex_px = c[i]; ex_type = 'TIME'
        
        if ex_px is not None:
            pnl = (ex_px/ep - 1)*100 - COMM*100
            trades.append({
                'pnl': pnl, 'type': ex_type, 'bars': i-ei,
                'entry': ep, 'exit': ex_px, 'date': datetime.fromtimestamp(0) if len(trades)==0 else None
            })
            in_trade = False
            cooldown = 3  # prevent immediate re-entry
    
    if not trades:
        return {'trades':0,'wins':0,'losses':0,'wr':0,'final':CAP,'dd':0,'sh':0,'avg_w':0,'avg_l':0,'rr':0,'net':0,'tp_n':0,'sl_n':0,'time_n':0}
    
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
    
    return {'trades':len(trades),'wins':len(wins),'losses':len(losses),'wr':wr,
            'final':final,'dd':dd,'sh':sh_val,'avg_w':avg_w,'avg_l':avg_l,'rr':rr,'net':net,
            'tp_n':sum(1 for t in trades if t['type']=='TP'),
            'sl_n':sum(1 for t in trades if t['type']=='SL'),
            'time_n':sum(1 for t in trades if t['type']=='TIME')}

# ── Run ──────────────────────────────────────────────

timeframes = [
    ('15m', 30, 96, '15 دقيقة'),
    ('1h', 90, 48, '1 ساعة'),
    ('4h', 365, 24, '4 ساعات'),
    ('1d', 1095, 15, 'يومي'),
]

print("="*85)
print(f"🚩 Bull Flag Strategy — FET/USDT")
print(f"   SL: تحت قاع العلم | TP: قياس القطب (measured move)")
print("="*85)

for tf, days, max_h, tf_ar in timeframes:
    print(f"\n{'─'*85}")
    print(f"⏱️ {tf_ar} ({tf}) | {days} يوم | أقصى صبر {max_h} شمعة")
    print(f"{'─'*85}")
    
    c, h_arr, l_arr, o, v, ts = fetch(tf, days)
    atr = atr_n(c, h_arr, l_arr, 14)
    
    # 3 variants
    variants = [
        ("قياس القطب (measured move)", True, 1.5),
        ("TP=3ATR / SL=1.5ATR", False, 1.5),
    ]
    
    print(f"  {'المتغير':<30s} {'صفقات':>4s} {'WR':>5s} {'💰 محفظة':>8s} {'سحب':>6s} {'شارپ':>6s} {'R:R':>5s} {'🎯TP':>4s} {'🛑SL':>4s} {'⏰':>3s} {'W':>7s} {'L':>7s}")
    print(f"  {'─'*30} {'─'*4} {'─'*5} {'─'*8} {'─'*6} {'─'*6} {'─'*5} {'─'*4} {'─'*4} {'─'*3} {'─'*7} {'─'*7}")
    
    for vname, use_target, sl_atr in variants:
        m = backtest_flag(c, h_arr, l_arr, o, v, atr, max_h, sl_atr, use_target)
        if m['trades'] > 0:
            print(f"  {vname:<30s} {m['trades']:>4d} {m['wr']:>4.0f}% ${m['final']:>7.0f} {m['dd']:>+5.1f}% {m['sh']:>+5.2f} {m['rr']:>4.2f} {m['tp_n']:>4d} {m['sl_n']:>4d} {m['time_n']:>3d} {m['avg_w']:>+6.1f}% {m['avg_l']:>+6.1f}%")
        else:
            print(f"  {vname:<30s} 0 trades")

# ── Deep dive on best timeframe ──
print(f"\n\n{'='*85}")
print(f"🔍 تفاصيل الصفقات — أفضل فريم")
print(f"{'='*85}")

# Use 4h with measured move
c, h_arr, l_arr, o, v, ts = fetch('4h', 365)
atr = atr_n(c, h_arr, l_arr, 14)

trades_detail = []
in_trade = False
ei = ep = sl_px = tp_px = 0
cooldown = 0

for i in range(20, len(c)):
    if cooldown > 0: cooldown -= 1
    
    if not in_trade and cooldown == 0:
        flag = detect_bull_flag(i, c, h_arr, l_arr, o, v)
        if flag and c[i] > flag['flag_high'] and c[i-1] <= flag['flag_high']:
            in_trade = True
            ei = i; ep = c[i]
            sl_px = flag['flag_low'] - atr[i]*0.5
            pole_h = flag['pole_peak'] - flag['pole_low']
            tp_px = ep + pole_h
        continue
    
    if not in_trade: continue
    
    ex_px = ex_type = None
    if l_arr[i] <= sl_px: ex_px = sl_px; ex_type = 'SL'
    elif h_arr[i] >= tp_px: ex_px = tp_px; ex_type = 'TP'
    elif i - ei >= 24: ex_px = c[i]; ex_type = 'TIME'
    
    if ex_px:
        pnl = (ex_px/ep - 1)*100 - COMM*100
        trades_detail.append({'entry':ep,'exit':ex_px,'pnl':pnl,'type':ex_type,'bars':i-ei,'date_in':ts[ei],'date_out':ts[i]})
        in_trade = False
        cooldown = 3

if trades_detail:
    print(f"\n⏱️ 4H — Measured Move Target")
    print(f"{'دخول':>12s}  {'خروج':>12s}  {'أيام':>4s}  {'دخول':>8s}  {'خروج':>8s}  {'ربح%':>8s}  {'نوع'}")
    print("-"*70)
    for t in trades_detail:
        icon = "🟢" if t['pnl'] > 0 else "🔴"
        print(f"{icon} {t['date_in'].strftime('%Y-%m-%d'):>10s}  {t['date_out'].strftime('%Y-%m-%d'):>10s}  {t['bars']:>3d}d  {t['entry']:>7.4f}  {t['exit']:>7.4f}  {t['pnl']:>+7.1f}%  {t['type']}")
    
    w = [t for t in trades_detail if t['pnl']>0]
    l = [t for t in trades_detail if t['pnl']<=0]
    print(f"\n🟢{len(w)} ربح | 🔴{len(l)} خسارة | WR {len(w)/len(trades_detail)*100:.0f}%")
    print(f"مجموع الربح: {sum(t['pnl'] for t in w):+.1f}% | مجموع الخسارة: {sum(t['pnl'] for t in l):+.1f}%")

print(f"\n✅ Done")
