#!/usr/bin/env python3
"""
نسخ استراتيجية EA Free Signals
TP1=0.7%, TP2=1.6%, TP3=2.8%, SL=1.8%
اختبار عدة طرق دخول لاكتشاف المشغل
"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

COMM = 0.002; CAP = 1000; SYM = 'FET/USDT'

exchange = ccxt.binance({'timeout': 15000})
since = exchange.parse8601('2026-05-01T00:00:00Z')
ohlcv = exchange.fetch_ohlcv(SYM, '15m', since=since, limit=10000)

c = np.array([float(o[4]) for o in ohlcv])
h = np.array([float(o[2]) for o in ohlcv])
l = np.array([float(o[3]) for o in ohlcv])
v = np.array([float(o[5]) for o in ohlcv])
n = len(c)

print(f"📊 {SYM} 15m | {n} شمعة | {datetime.fromtimestamp(ohlcv[0][0]/1000).date()} → {datetime.fromtimestamp(ohlcv[-1][0]/1000).date()}")
print(f"   السعر: {c[0]:.4f} → {c[-1]:.4f}")

# Fixed TP/SL percentages (from reverse engineering)
TP1_PCT = 0.7 / 100
TP2_PCT = 1.6 / 100
TP3_PCT = 2.8 / 100
SL_PCT = 1.8 / 100
MAX_BARS = 48  # 12 hours

# Precompute indicators
ema9 = pd.Series(c).ewm(span=9).mean().values
ema21 = pd.Series(c).ewm(span=21).mean().values
avg_vol = pd.Series(v).rolling(20).mean().values
rsi_arr = np.zeros(n)
for i in range(14, n):
    delta = np.diff(c[i-14:i+1])
    gain = np.mean(delta[delta > 0]) if any(delta > 0) else 0
    loss = -np.mean(delta[delta < 0]) if any(delta < 0) else 0.0001
    rsi_arr[i] = 100 - 100/(1 + gain/loss)

def backtest(label, entry_fn, direction='LONG'):
    trades = []
    for i in range(20, n-1):
        if not entry_fn(i):
            continue
        
        ep = c[i]
        tp1 = ep * (1 + TP1_PCT)
        tp2 = ep * (1 + TP2_PCT)
        tp3 = ep * (1 + TP3_PCT)
        sl = ep * (1 - SL_PCT)
        
        ex = et = None
        for j in range(i+1, min(i+MAX_BARS, n)):
            if l[j] <= sl: ex = sl; et = 'SL'; break
            elif h[j] >= tp3: ex = tp3; et = 'TP3'; break
            elif h[j] >= tp2: ex = tp2; et = 'TP2'; break
            elif h[j] >= tp1: ex = tp1; et = 'TP1'; break
        if not ex:
            ex = c[min(i+MAX_BARS, n-1)]; et = 'TIME'
        
        pnl = (ex/ep - 1)*100 - COMM*100
        trades.append({'pnl': pnl, 'type': et, 'bars': j-i if ex else MAX_BARS})
    
    if not trades: return None
    
    w = [t for t in trades if t['pnl']>0]; lo = [t for t in trades if t['pnl']<=0]
    curve = [CAP]
    for t in trades:
        sz = curve[-1]*0.10; curve.append(curve[-1]+sz*t['pnl']/100)
    final = curve[-1]; wr = len(w)/len(trades)*100
    dr = np.diff(curve)/curve[:-1]
    sh = np.mean(dr)/np.std(dr)*np.sqrt(252*24*4) if len(dr)>1 and np.std(dr)>0 else 0
    peak = np.maximum.accumulate(curve); dd = np.min((curve-peak)/peak*100)
    aw = np.mean([t['pnl'] for t in w]) if w else 0
    al = np.mean([t['pnl'] for t in lo]) if lo else 0
    
    tp1_n = sum(1 for t in trades if t['type']=='TP1')
    tp2_n = sum(1 for t in trades if t['type']=='TP2')
    tp3_n = sum(1 for t in trades if t['type']=='TP3')
    sl_n = sum(1 for t in trades if t['type']=='SL')
    tm_n = sum(1 for t in trades if t['type']=='TIME')
    
    return {'t':len(trades),'wr':wr,'final':final,'dd':dd,'sh':sh,'aw':aw,'al':al,
            'tp1':tp1_n,'tp2':tp2_n,'tp3':tp3_n,'sl':sl_n,'tm':tm_n}

# ── Entry strategies to test ──
entries = []

# 1: Volume spike + price breakout
entries.append(("Vol>2x + اختراق قمة 5", lambda i: 
    v[i] > avg_vol[i]*2.0 and c[i] > max(h[max(0,i-5):i]) and c[i] > c[i-1]))

# 2: Volume spike + EMA9 cross
entries.append(("Vol>1.5x + تقاطع EMA9", lambda i:
    v[i] > avg_vol[i]*1.5 and c[i] > ema9[i] and c[i-1] <= ema9[i-1]))

# 3: RSI oversold bounce
entries.append(("RSI < 30 + ارتداد", lambda i:
    rsi_arr[i-1] < 30 and c[i] > c[i-1] and c[i] > ema9[i]))

# 4: Simple momentum (3 green candles + volume)
entries.append(("3 شمعات خضراء + حجم", lambda i:
    c[i] > c[i-1] and c[i-1] > c[i-2] and c[i-2] > c[i-3] and v[i] > avg_vol[i]*1.2))

# 5: Price near day low + bounce
entries.append(("قرب قاع 24h + ارتداد", lambda i:
    c[i] > c[i-1] and c[i] < max(h[max(0,i-96):i])*0.95 and l[i] <= min(l[max(0,i-96):i])*1.02))

# 6: EMA9 > EMA21 + pullback to EMA9
entries.append(("EMA9>21 + ارتداد من EMA9", lambda i:
    ema9[i] > ema21[i] and abs(c[i]-ema9[i])/ema9[i] < 0.005 and c[i] > c[i-1]))

# 7: Multi-coin scanner style — any volume burst
entries.append(("أي انفجار حجم > 3x", lambda i:
    v[i] > avg_vol[i]*3.0 and c[i] > c[i-1]))

print(f"\n{'─'*80}")
print(f"{'استراتيجية الدخول':<35s} {'T':>4s} {'WR':>5s} {'💰':>7s} {'DD':>5s} {'TP1':>4s} {'TP2':>3s} {'TP3':>3s} {'SL':>4s} {'⏰':>3s}")
print(f"{'─'*80}")

for label, fn in entries:
    m = backtest(label, fn)
    if m and m['t'] > 0:
        print(f"  {label:<33s} {m['t']:>4d} {m['wr']:>4.0f}% ${m['final']:>6.0f} {m['dd']:>+4.1f}% {m['tp1']:>4d} {m['tp2']:>3d} {m['tp3']:>3d} {m['sl']:>4d} {m['tm']:>3d}")
    else:
        print(f"  {label:<33s} 0 trades")

# Show best candidate in detail
print(f"\n{'─'*80}")
print(f"🔍 تفاصيل أفضل مرشح — Vol>2x + اختراق قمة 5")
print(f"{'─'*80}")

# Run detailed version
trades_detailed = []
for i in range(20, n-1):
    if not (v[i] > avg_vol[i]*2.0 and c[i] > max(h[max(0,i-5):i]) and c[i] > c[i-1]):
        continue
    
    ep = c[i]
    tp1 = ep * (1 + TP1_PCT)
    tp2 = ep * (1 + TP2_PCT)
    tp3 = ep * (1 + TP3_PCT)
    sl = ep * (1 - SL_PCT)
    
    ex = et = None
    for j in range(i+1, min(i+MAX_BARS, n)):
        if l[j] <= sl: ex = sl; et = 'SL'; break
        elif h[j] >= tp3: ex = tp3; et = 'TP3'; break
        elif h[j] >= tp2: ex = tp2; et = 'TP2'; break
        elif h[j] >= tp1: ex = tp1; et = 'TP1'; break
    if not ex:
        ex = c[min(i+MAX_BARS, n-1)]; et = 'TIME'
    
    pnl = (ex/ep - 1)*100 - COMM*100
    ts_i = datetime.fromtimestamp(ohlcv[i][0]/1000)
    trades_detailed.append({'pnl':pnl,'type':et,'bars':j-i if ex else MAX_BARS,'ts':ts_i,'entry':ep,'exit':ex})

if trades_detailed:
    w = [t for t in trades_detailed if t['pnl']>0]
    l = [t for t in trades_detailed if t['pnl']<=0]
    print(f"  {len(trades_detailed)} صفقة | WR {len(w)/len(trades_detailed)*100:.0f}%")
    print(f"  TP1={sum(1 for t in trades_detailed if t['type']=='TP1')} TP2={sum(1 for t in trades_detailed if t['type']=='TP2')} TP3={sum(1 for t in trades_detailed if t['type']=='TP3')} SL={sum(1 for t in trades_detailed if t['type']=='SL')} TIME={sum(1 for t in trades_detailed if t['type']=='TIME')}")
    
    print(f"\n  {'#':>3s} {'تاريخ':>16s} {'ربح%':>7s} {'نوع'}")
    for idx, t in enumerate(trades_detailed, 1):
        icon = "🟢" if t['pnl'] > 0 else "🔴"
        print(f"  {icon}{idx:>2d} {t['ts'].strftime('%m-%d %H:%M'):>14s} {t['pnl']:>+7.2f}% {t['type']}")

print(f"\n✅ Done")
