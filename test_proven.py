"""Test proven high-WR strategies from memory"""
import json, os, numpy as np, pandas as pd
COMM, DATA = 0.002, 'data/whale_15m_1y'

def load(sym):
    p = os.path.join(DATA, f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d = json.load(f)
    return {'c': np.array(d['c'],float), 'h': np.array(d['h'],float),
            'l': np.array(d['l'],float), 'o': np.array(d['o'],float)}

coins = sorted([f.replace('.json','') for f in os.listdir(DATA) 
                if f.endswith('.json') and f!='_manifest.json'])[:60]

# ── Strategy 1: Steep angle + pullback on 1h equivalent ──
# Detect steep EMA20 slope (>X% over N bars) then buy pullback to EMA
def steep_angle_pullback(c,h,l,o,n):
    ema20 = pd.Series(c).ewm(span=20*4, adjust=False).mean().values  # 1h EMA20
    ema50 = pd.Series(c).ewm(span=50*4, adjust=False).mean().values  # 1h EMA50
    entries = []
    for i in range(500, n):
        slope_5 = (ema20[i] - ema20[i-5*4]) / ema20[i-5*4] * 100  # 5 bars on 1h
        steep = slope_5 > 0.3  # 0.3% per 5h
        above_ema50 = ema20[i] > ema50[i]
        # Pullback: price dipped below EMA20 but above EMA50
        pullback = c[i] < ema20[i] and c[i] > ema50[i]
        reversal = c[i] > o[i]  # green candle
        if steep and above_ema50 and pullback and reversal:
            entries.append(i)
    return entries

# ── Strategy 2: Pump over 24 candles + Spike confirmation ──
def pump24_spike(c,h,l,o,n):
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
    atr = pd.Series(h-l).ewm(span=14, adjust=False).mean().values
    entries = []
    for i in range(500, n):
        # Pump: price gained >3% over 24 candles
        pump = c[i] > c[i-24] * 1.03
        # Pullback to EMA20
        near_ema = abs(c[i] - ema20[i]) / ema20[i] < 0.01
        # Spike: last candle range > 1.5x ATR
        spike = (h[i] - l[i]) > atr[i] * 1.5
        green = c[i] > o[i]
        if pump and near_ema and spike and green:
            entries.append(i)
    return entries

# ── Strategy 3: Multi-TF trend alignment + pullback ──
def mtf_pullback(c,h,l,o,n):
    e4h_20 = pd.Series(c).ewm(span=20*16, adjust=False).mean().values
    e4h_50 = pd.Series(c).ewm(span=50*16, adjust=False).mean().values
    e1h_20 = pd.Series(c).ewm(span=20*4, adjust=False).mean().values
    e1h_50 = pd.Series(c).ewm(span=50*4, adjust=False).mean().values
    e15m_20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
    
    entries = []
    for i in range(500, n):
        # 4h: EMA20 > EMA50
        tf4 = e4h_20[i] > e4h_50[i]
        # 1h: EMA20 > EMA50
        tf1 = e1h_20[i] > e1h_50[i]
        # 15m: pullback to EMA20
        pb_15 = c[i] < e15m_20[i] and c[i] > e15m_20[i] * 0.99
        green = c[i] > o[i]
        if tf4 and tf1 and pb_15 and green:
            entries.append(i)
    return entries

# ── Strategy 4: SSL Channel break ──
def ssl_breakout(c,h,l,o,n):
    ssl_p = 20
    sup = pd.Series(h).rolling(ssl_p).mean().values
    sdn = pd.Series(l).rolling(ssl_p).mean().values
    
    # Higher TF trend
    e4h_50 = pd.Series(c).ewm(span=50*16, adjust=False).mean().values
    e4h_200 = pd.Series(c).ewm(span=200*16, adjust=False).mean().values
    
    entries = []
    for i in range(500, n):
        # SSL just turned bullish (was red, now green)
        ssl_cross = c[i] > sup[i] and c[i-1] <= sup[i-1]
        # 4h trend up
        trend = e4h_50[i] > e4h_200[i]
        # Price breaking above recent swing high
        sw_high = max(h[i-20:i])
        breakout = c[i] > sw_high * 0.995
        if ssl_cross and trend and breakout:
            entries.append(i)
    return entries

# ── Strategy 5: Deep pullback in uptrend ──
def deep_pullback(c,h,l,o,n):
    e20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
    e50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    e200 = pd.Series(c).ewm(span=200, adjust=False).mean().values
    rsi14 = np.zeros(n)
    delta = pd.Series(c).diff()
    gain = np.where(delta>0, delta, 0)
    loss = np.where(delta<0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False).mean().values
    rs = np.where(avg_loss>0, avg_gain/avg_loss, 100)
    rsi14 = 100 - 100/(1+rs)
    
    entries = []
    for i in range(500, n):
        # Uptrend: EMA20 > EMA50 > EMA200
        uptrend = e20[i] > e50[i] and e50[i] > e200[i]
        # Deep pullback: price below EMA20 but above EMA200
        deep = c[i] < e20[i] and c[i] > e200[i]
        # RSI oversold on pullback
        rsi_low = rsi14[i] < 40
        # Recovery candle
        recov = c[i] > o[i] and c[i] > c[i-1]
        if uptrend and deep and rsi_low and recov:
            entries.append(i)
    return entries

strategies = [
    ('SteepAngle+PB(1h)', steep_angle_pullback),
    ('Pump24+Spike(15m)', pump24_spike),
    ('MTF Pullback', mtf_pullback),
    ('SSL Breakout+4h', ssl_breakout),
    ('DeepPB+RSI', deep_pullback),
]

TP_SL_OPTIONS = [(1,2),(1.5,2.5),(2,3),(1,3),(2,2),(1.5,3)]

print(f"{'Strategy':<25} {'Best TP/SL':>10} {'T':>5} {'WR':>7} {'PnL$':>9} {'$/T':>7} {'C':>4}")
print("-"*73)

for sname, sfn in strategies:
    best_wr = 0; best_info = None
    for tp,sl in TP_SL_OPTIONS:
        t=0; w=0; l=0; pnl=0.0; cc=0
        for sym in coins:
            d = load(sym)
            if d is None or len(d['c'])<500: continue
            c,h,l_,o = d['c'], d['h'], d['l'], d['o']; n=len(c)
            entries = sfn(c,h,l_,o,n)
            if len(entries) < 3: continue
            cc += 1
            for ei in entries:
                ep=c[ei]; end=min(ei+48,n)
                th=sh=False; tj=sj=99999
                for j in range(ei+1,end):
                    if not th and h[j]>=ep*(1+tp/100): th=True; tj=j
                    if not sh and l_[j]<=ep*(1-sl/100): sh=True; sj=j
                    if th and sh: break
                t+=1
                if th and not sh: w+=1; pnl+=tp-COMM*100
                elif sh and not th: l+=1; pnl+=-sl-COMM*100
                else: pnl+=(c[end-1]/ep-1)*100-COMM*100
        if t >= 10:
            wr = w/t*100
            if wr > best_wr:
                best_wr = wr; best_info = (tp,sl,t,wr,w,l,pnl,pnl/t,cc)
    if best_info:
        tp,sl,t,wr,w,l,pnl,avg,cc = best_info
        print(f"{sname:<25} TP{tp}/SL{sl} {t:>5} {wr:>6.1f}% ${pnl:>+8.1f} ${avg:>+6.2f} {cc:>4}")
