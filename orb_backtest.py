#!/usr/bin/env python3
"""Opening Range Breakout (ORB) — first 15m candle after anchor time"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
COMM = 0.002; CAP = 1000; MAX_SLIPPAGE = 1.5; COOLDOWN = 48

ANCHOR_HOUR = 0  # UTC hour for daily open (0 = midnight). 9:30 AM Jordan = 6:30 UTC → use 6
DATA_PREV = '/data/trading28/data/whale_15m_prev'
DATA_CUR = '/data/trading28/data/whale_15m_1y'

def load_data(dir_path):
    data = {}
    for f in os.listdir(dir_path):
        if not f.endswith('.json') or f == '_manifest.json': continue
        sym = f.replace('.json', '')
        try:
            with open(os.path.join(dir_path, f)) as fh:
                d = json.load(fh)
            if len(d.get('c', [])) < 500: continue
            data[sym] = {
                'c': np.array(d['c'], float), 'h': np.array(d['h'], float),
                'l': np.array(d['l'], float), 'o': np.array(d['o'], float),
                'ts': d.get('ts', [])
            }
        except: pass
    return data

def ema(s, p):
    return pd.Series(s).ewm(span=p, adjust=False).mean().values

def backtest_orb(data, tp, sl, trend_filter='none'):
    """ORB strategy: first 15m candle after anchor hour sets range"""
    c, h, l, o, ts = data['c'], data['h'], data['l'], data['o'], data['ts']
    n = len(c)

    # ── Build datetime index ──
    try:
        idx = pd.to_datetime(np.array(ts), unit='ms')
        df = pd.DataFrame({'c': c, 'h': h, 'l': l, 'o': o}, index=idx)

        # ── Trend filters ──
        filt = np.ones(n, bool)
        if trend_filter in ('4h', '4h+1h'):
            c4h = df['c'].resample('4h').last().dropna().values
            e50 = ema(c4h, 50); e200 = ema(c4h, 200)
            e50a = np.zeros(n); e200a = np.zeros(n)
            for i in range(n):
                j = i // 16
                if j < len(e50): e50a[i] = e50[j]; e200a[i] = e200[j]
            filt = e50a > e200a
            if trend_filter == '4h+1h':
                c1h = df['c'].resample('1h').last().dropna().values
                e20 = ema(c1h, 20); e50h = ema(c1h, 50)
                e20a = np.zeros(n); e50h_a = np.zeros(n)
                for i in range(n):
                    j = i // 4
                    if j < len(e20): e20a[i] = e20[j]; e50h_a[i] = e50h[j]
                filt = filt & (e20a > e50h_a)

        # ── Identify ORB candle (first 15m of each UTC day) ──
        orb_high = np.full(n, np.nan)
        orb_low = np.full(n, np.nan)
        orb_active = np.zeros(n, bool)  # True after ORB candle closes

        current_day = -1
        orb_h = orb_l_val = None
        for i in range(n):
            day = idx[i].day if hasattr(idx[i], 'day') else idx[i].day
            hour = idx[i].hour
            minute = idx[i].minute
            # Find first 15m candle (hour=ANCHOR_HOUR, minute=0)
            if hour == ANCHOR_HOUR and minute == 0:
                orb_h = h[i]
                orb_l_val = l[i]
                current_day = idx[i].day

            if orb_h is not None and idx[i].day == current_day:
                orb_high[i] = orb_h
                orb_low[i] = orb_l_val
                # Active after the ORB candle closes (i.e., next candle onwards)
                if i > 0 and idx[i-1].hour == ANCHOR_HOUR and idx[i-1].minute == 0:
                    orb_active[i] = True
                elif orb_active[i-1] if i > 0 else False:
                    orb_active[i] = True

    except Exception as e:
        return None

    # ── Generate entries: break above ORB high or below ORB low ──
    long_entries = np.zeros(n, bool)
    short_entries = np.zeros(n, bool)

    for i in range(200, n):
        if not orb_active[i] or np.isnan(orb_high[i]):
            continue
        # Long: price breaks above ORB high
        if h[i] >= orb_high[i] * 1.001 and c[i-1] <= orb_high[i]:
            long_entries[i] = True
        # Short: price breaks below ORB low
        if l[i] <= orb_low[i] * 0.999 and c[i-1] >= orb_low[i]:
            short_entries[i] = True

    # ── Simulate trades ──
    trades = []; eq = CAP; cv = [CAP]; pos = 0; ep = 0; cool = 0; side = 0
    
    for i in range(200, n):
        if pos:
            if side == 1:  # Long
                if h[i] >= ep * (1 + tp / 100):
                    pnl = tp - COMM * 100; trades.append(pnl)
                    eq *= (1 + pnl / 100); pos = 0; cool = COOLDOWN
                elif l[i] <= ep * (1 - sl / 100):
                    raw = (c[i] / ep - 1) * 100 - COMM * 100
                    pnl = max(raw, -sl * MAX_SLIPPAGE - COMM * 100)
                    trades.append(pnl); eq *= (1 + pnl / 100)
                    pos = 0; cool = COOLDOWN
            else:  # Short
                if l[i] <= ep * (1 - tp / 100):
                    pnl = tp - COMM * 100; trades.append(pnl)
                    eq *= (1 + pnl / 100); pos = 0; cool = COOLDOWN
                elif h[i] >= ep * (1 + sl / 100):
                    raw = (1 - c[i] / ep) * 100 - COMM * 100
                    pnl = max(raw, -sl * MAX_SLIPPAGE - COMM * 100)
                    trades.append(pnl); eq *= (1 + pnl / 100)
                    pos = 0; cool = COOLDOWN

        if not pos and cool == 0 and filt[i]:
            if long_entries[i]:
                pos = 1; ep = c[i]; side = 1; cool = 0
            elif short_entries[i]:
                pos = 1; ep = c[i]; side = -1; cool = 0

        if not pos and cool > 0:
            cool -= 1
        cv.append(eq)

    if pos:
        if side == 1:
            pnl = (c[-1] / ep - 1) * 100 - COMM * 100
        else:
            pnl = (1 - c[-1] / ep) * 100 - COMM * 100
        trades.append(pnl); eq *= (1 + pnl / 100)

    if len(trades) < 5:
        return None

    wins = sum(1 for p in trades if p > 0)
    wr = wins / len(trades) * 100
    dd = ((pd.Series(cv) - pd.Series(cv).expanding().max()) / pd.Series(cv).expanding().max() * 100).min()
    return {'t': len(trades), 'wr': wr, 'dd': dd, 'pnl': eq - CAP, 'w': wins, 'l': len(trades) - wins}

# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
prev_data = load_data(DATA_PREV)
cur_data = load_data(DATA_CUR)
common = sorted(set(prev_data) & set(cur_data))
print(f"Common coins: {len(common)}")

tp_sl_grid = [(2, 1), (3, 1.5), (4, 2), (5, 2.5), (6, 3)]
filters = ['none']

best = {}
for tp, sl in tp_sl_grid:
    print(f"\n{'='*60}")
    print(f"TP={tp}% SL={sl}% | Anchor={ANCHOR_HOUR}:00 UTC")
    print(f"{'='*60}")
    
    for period_name, period_data in [('PREV', prev_data), ('CUR', cur_data)]:
        results = []
        for sym in common:
            r = backtest_orb(period_data[sym], tp, sl, 'none')
            if r:
                r['sym'] = sym
                results.append(r)
        
        if not results:
            continue
            
        total_t = sum(r['t'] for r in results)
        total_w = sum(r['w'] for r in results)
        total_l = sum(r['l'] for r in results)
        total_pnl = sum(r['pnl'] for r in results)
        avg_wr = total_w / total_t * 100 if total_t > 0 else 0
        avg_dd = np.mean([r['dd'] for r in results])
        green = sum(1 for r in results if r['pnl'] > 0)
        
        key = f"TP{tp}_SL{sl}"
        if key not in best: best[key] = {}
        best[key][period_name] = {
            'coins': len(results), 'trades': total_t, 'wins': total_w, 'losses': total_l,
            'wr': avg_wr, 'dd': avg_dd, 'pnl': total_pnl, 'green': green
        }
        
        print(f"  {period_name}: {len(results):3d} coins | {total_t:5d} trades | "
              f"🟢{total_w} 🔴{total_l} | WR={avg_wr:.1f}% | DD={avg_dd:.1f}% | "
              f"${total_pnl:+,.0f} | green={green}")

# Summary
print(f"\n{'='*60}")
print("BEST CONFIGURATIONS")
print(f"{'='*60}")
for key in sorted(best.keys()):
    prev = best[key].get('PREV', {})
    cur = best[key].get('CUR', {})
    prev_pnl = prev.get('pnl', 0)
    cur_pnl = cur.get('pnl', 0)
    combined = prev_pnl + cur_pnl
    prev_wr = prev.get('wr', 0)
    cur_wr = cur.get('wr', 0)
    print(f"  {key}: PREV WR={prev_wr:.1f}% ${prev_pnl:+,.0f} | CUR WR={cur_wr:.1f}% ${cur_pnl:+,.0f} | COMBINED ${combined:+,.0f}")

print("\nDone")
