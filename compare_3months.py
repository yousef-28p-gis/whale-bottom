#!/usr/bin/env python3
"""Compare Hunter Whale backtest vs live-style simulation — 3 months (Apr-Jun 2026)."""
import json, numpy as np, pandas as pd, os
from datetime import datetime, timedelta
from collections import defaultdict

CACHE = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'

TP = 2.5; SL = 2.0; PL = 40; TRAIL = 0.10; MAX_H = 2
STR = 50; WHALE_MIN = 0.35; MIN_VOL = 200000; COMMISSION = 0.20

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
BLOCKED = {'SUPER','ORCA','VANA','W','DOGS','MET','XLM','BB','COS','LUNA','S'}

def load_cached(sym, mon):
    fpath = f'{CACHE}/{sym}_{mon}.json'
    if not os.path.exists(fpath): return None
    with open(fpath) as f: data = json.load(f)
    df = pd.DataFrame(data)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    return df.sort_values('ts').reset_index(drop=True)

def whale_indicator(df):
    df = df.copy()
    LB = 30
    df['lo'] = df['low'].rolling(LB).min()
    df['lc'] = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
    df['sm'] = df['lc'].ewm(span=3, adjust=False).mean()
    df['hi'] = df['sm'].rolling(LB).max()
    df['raw'] = np.where(df['low'] <= df['lo'], (df['sm'] + df['hi'] * 2) / 3, 0)
    df['whale'] = df['raw'].ewm(span=3, adjust=False).mean().fillna(0)
    df['spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.03)
    df['wf'] = df['whale'].rolling(2).mean()
    df['ws'] = df['whale'].rolling(5).mean()
    df['wp'] = df['whale'].rolling(50).max()
    df['str'] = (df['whale'] / df['wp'].replace(0, np.nan) * 100).fillna(0)
    df['vma'] = df['volume'].rolling(20).mean()
    df['entry'] = (df['spike'] & (df['wf'] > df['ws']) &
                   (df['str'] > STR) & (df['volume'] > df['vma'] * 1.0))
    return df

def simulate_backtest(df, entry_idx):
    """Original: check high/low of each candle"""
    entry_price = df.iloc[entry_idx]['close']
    tp_price = entry_price * (1 + TP/100)
    sl_price = entry_price * (1 - SL/100)
    pl_price = entry_price + (tp_price - entry_price) * (PL/100)
    pl_triggered = False; peak = entry_price; trail_price = 0
    
    for j in range(entry_idx + 1, len(df)):
        row = df.iloc[j]
        hours = (j - entry_idx) * 0.25
        if hours > MAX_H:
            return ('TIMEOUT', round((row['close'] - entry_price)/entry_price*100, 4))
        
        if not pl_triggered and row['high'] >= pl_price:
            pl_triggered = True; peak = row['high']
            trail_price = row['high'] * (1 - TRAIL/100)
        if pl_triggered:
            if row['high'] > peak:
                peak = row['high']; trail_price = row['high'] * (1 - TRAIL/100)
            if row['low'] <= trail_price:
                return ('TRAIL', round((trail_price - entry_price)/entry_price*100, 4))
        if row['high'] >= tp_price:
            return ('TP', round(TP, 4))
        if row['low'] <= sl_price:
            return ('SL', round(-SL, 4))
    
    return ('EOD', round((df.iloc[-1]['close'] - entry_price)/entry_price*100, 4))

def simulate_live(df, entry_idx):
    """Live-style: check close price only"""
    entry_price = df.iloc[entry_idx]['close']
    tp_price = entry_price * (1 + TP/100)
    sl_price = entry_price * (1 - SL/100)
    pl_price = entry_price + (tp_price - entry_price) * (PL/100)
    pl_triggered = False; peak = entry_price; trail_price = 0
    
    for j in range(entry_idx + 1, len(df)):
        row = df.iloc[j]
        hours = (j - entry_idx) * 0.25
        current = row['close']
        if hours > MAX_H:
            return ('TIMEOUT', round((current - entry_price)/entry_price*100, 4))
        
        if not pl_triggered and current >= pl_price:
            pl_triggered = True; peak = current
            trail_price = current * (1 - TRAIL/100)
        if pl_triggered:
            if current > peak:
                peak = current; trail_price = current * (1 - TRAIL/100)
            if current <= trail_price:
                return ('TRAIL', round((trail_price - entry_price)/entry_price*100, 4))
        if current >= tp_price:
            return ('TP', round(TP, 4))
        if current <= sl_price:
            return ('SL', round(-SL, 4))
    
    return ('EOD', round((df.iloc[-1]['close'] - entry_price)/entry_price*100, 4))

# ── Load signals ──
with open(SIGNALS_FILE) as f:
    raw = json.load(f)

TARGET_MONTHS = {4, 5, 6}
signals = []
for s in raw:
    if s['symbol'] in STABLES or s['symbol'] in BLOCKED: continue
    if s.get('direction', 'LONG') != 'LONG': continue
    if s.get('volume_usdt', 0) < MIN_VOL: continue
    dt = datetime.fromisoformat(s['dt'])
    if dt.month not in TARGET_MONTHS or dt.year != 2026: continue
    signals.append({'symbol': s['symbol'], 'dt': dt, 'month': dt.strftime('%Y-%m')})

print(f'Signals: {len(signals)} ({len(set(s["month"] for s in signals))} months)')

# ── Run ──
by_pair = defaultdict(list)
for sig in signals:
    by_pair[(sig['symbol'], sig['month'])].append(sig)

trades_bt = []; trades_live = []
pairs_ok = 0; pairs_total = len(by_pair)

for (sym, mon), sigs in sorted(by_pair.items()):
    df = load_cached(sym, mon)
    if df is None: continue
    pairs_ok += 1
    df_w = whale_indicator(df)
    
    for sig in sigs:
        df_w['td'] = abs((df_w['ts'] - sig['dt']).dt.total_seconds())
        nearest = df_w['td'].idxmin()
        forward = df_w.iloc[nearest:].reset_index(drop=True)
        
        wi = None; wv = 0
        for j, row in forward.iterrows():
            if j * 0.25 > 24: break
            if row['entry']: wi = j; wv = float(row['whale']); break
        
        if wi is None or wv < WHALE_MIN: continue
        
        st_bt, pnl_bt = simulate_backtest(forward, wi)
        trades_bt.append({'symbol': sym, 'dt': sig['dt'], 'month': mon,
            'pnl': pnl_bt, 'net': round(pnl_bt - COMMISSION, 4), 'exit': st_bt})
        
        st_lv, pnl_lv = simulate_live(forward, wi)
        trades_live.append({'symbol': sym, 'dt': sig['dt'], 'month': mon,
            'pnl': pnl_lv, 'net': round(pnl_lv - COMMISSION, 4), 'exit': st_lv})

# ── Stats ──
def stats(trades, name):
    if not trades: return
    wins = [t for t in trades if t['net'] > 0]
    losses = [t for t in trades if t['net'] <= 0]
    wr = len(wins) / len(trades) * 100
    net = sum(t['net'] for t in trades)
    avg_win = np.mean([t['net'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['net'] for t in losses]) if losses else 0
    
    exits = defaultdict(int)
    for t in trades: exits[t['exit']] += 1
    
    print(f'\n{"="*55}')
    print(f'📊 {name}')
    print(f'{"="*55}')
    print(f'  صفقات: {len(trades)}')
    print(f'  رابحة: {len(wins)} | خاسرة: {len(losses)}')
    print(f'  WR: {wr:.1f}%')
    print(f'  صافي: {net:+.2f}%')
    print(f'  متوسط ربح: {avg_win:+.2f}% | متوسط خسارة: {avg_loss:+.2f}%')
    print(f'  TP={exits.get("TP",0)} | SL={exits.get("SL",0)} | TRAIL={exits.get("TRAIL",0)} | TIME={exits.get("TIMEOUT",0)} | EOD={exits.get("EOD",0)}')
    
    # By month
    print()
    months_ar = {'2026-04': 'أبريل', '2026-05': 'مايو', '2026-06': 'يونيو'}
    for mon in sorted(set(t['month'] for t in trades)):
        mt = [t for t in trades if t['month'] == mon]
        mw = sum(1 for t in mt if t['net'] > 0)
        mn = sum(t['net'] for t in mt)
        print(f'  {months_ar.get(mon, mon)}: {len(mt)} صفقة | WR {mw/len(mt)*100:.0f}% | صافي {mn:+.1f}%')
    
    # Monthly portfolio
    cap = 1000; active = []; peak = 1000; max_dd = 0; mcap = {}
    for t in sorted(trades, key=lambda t: t['dt']):
        exit_t = t['dt'] + timedelta(hours=MAX_H)
        still = []
        for et, ps, pd in active:
            if et < t['dt']: cap += ps + pd
            else: still.append((et, ps, pd))
        active = still
        if len(active) >= 3: continue
        ps = cap * 0.33
        if cap < ps: continue
        pd = ps * t['net'] / 100
        active.append((exit_t, ps, pd))
        cap -= ps
        eq = cap + sum(s + d for _, s, d in active)
        if eq > peak: peak = eq
        dd = (eq - peak) / peak * 100
        if dd < max_dd: max_dd = dd
        mcap[t['month']] = eq
    for _, s, d in active: cap += s + d
    
    print(f'\n  💼 محفظة: $1000 → ${cap:.0f} | DD: {max_dd:.1f}%')
    return wr, net, cap, max_dd

print(f'\n🔄 جاري تحليل {len(signals)} إشارة...')
print(f'  الأزواج المحملة: {pairs_ok}/{pairs_total}')

wr_bt, net_bt, cap_bt, dd_bt = stats(trades_bt, '🐋 الأصلي — هاي/لو داخل الشمعة')
wr_lv, net_lv, cap_lv, dd_lv = stats(trades_live, '📡 الحي — سعر الإغلاق فقط')

# ── Diff ──
diff = 0
for bt, lv in zip(trades_bt, trades_live):
    if bt['exit'] != lv['exit']: diff += 1

print(f'\n{"="*55}')
print(f'🔍 ملخص الفروقات')
print(f'{"="*55}')
print(f'  صفقات مختلفة النتيجة: {diff}/{len(trades_bt)} ({diff/len(trades_bt)*100:.0f}%)')
print(f'  فرق WR: {wr_bt - wr_lv:+.1f}%')
print(f'  فرق صافي: {net_bt - net_lv:+.1f}%')
print(f'  فرق محفظة: ${cap_bt - cap_lv:.0f}')
print(f'  فرق سحب: {dd_bt - dd_lv:+.1f}%')
