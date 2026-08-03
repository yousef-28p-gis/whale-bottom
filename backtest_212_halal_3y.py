#!/usr/bin/env python3
"""Ichimoku 8h Ultra — backtest 3 years, per-period coins"""
import json, os, numpy as np, pandas as pd

COMM = 0.002; CAP = 1000; MAX_SLIPPAGE = 1.5; COOLDOWN = 2

DATA_DIRS = {
    '2023': '/data/trading28/data/whale_15m_2023',
    'PREV': '/data/trading28/data/whale_15m_prev',
    'CUR':  '/data/trading28/data/whale_15m_1y',
}

def load(sym, period):
    p = os.path.join(DATA_DIRS[period], f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f:
        j = json.load(f)
    return (np.array(j['c'],float), np.array(j['h'],float), np.array(j['l'],float),
            np.array(j['o'],float), j.get('ts',[]))

def resample_8h(c, h, l, o, ts):
    try:
        idx = pd.to_datetime(np.array(ts), unit='ms')
        df = pd.DataFrame({'o':o, 'h':h, 'l':l, 'c':c}, index=idx)
        r = df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
        return r['c'].values, r['h'].values, r['l'].values, r['o'].values
    except:
        return None

def ichimoku(c, h, l, o, tenkan=3, kijun=9, senkou=18, tp=5, sl=2.5, cooldown=2):
    n = len(c)
    if n < senkou + 30: return None
    
    h_t = pd.Series(h).rolling(tenkan).max().values
    l_t = pd.Series(l).rolling(tenkan).min().values
    t_arr = (h_t + l_t) / 2
    
    h_k = pd.Series(h).rolling(kijun).max().values
    l_k = pd.Series(l).rolling(kijun).min().values
    k_arr = (h_k + l_k) / 2
    
    h_s = pd.Series(h).rolling(senkou).max().values
    l_s = pd.Series(l).rolling(senkou).min().values
    sb_raw = (h_s + l_s) / 2
    sa_raw = (t_arr + k_arr) / 2
    
    shift = kijun
    sa = np.full(n, np.nan); sb = np.full(n, np.nan)
    for i in range(max(shift, senkou), n - shift):
        if i + shift < n:
            sa[i+shift] = sa_raw[i]
            sb[i+shift] = sb_raw[i]
    
    trades = []; eq = CAP; cv = [CAP]
    pos = 0; ep = 0; cool = 0; side = 0
    
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        
        cloud_top = max(sa[i], sb[i]); cloud_bot = min(sa[i], sb[i])
        above = c[i] > cloud_top; below = c[i] < cloud_bot
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        death = t_arr[i] < k_arr[i] and t_arr[i-1] >= k_arr[i-1]
        
        if pos:
            if side == 1:  # Long
                if h[i] >= ep * (1 + tp/100):
                    pnl = tp - COMM * 100
                    trades.append(pnl); eq *= (1 + pnl/100)
                    pos = 0; cool = cooldown
                elif l[i] <= ep * (1 - sl/100):
                    pnl = max((c[i]/ep - 1)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append(pnl); eq *= (1 + pnl/100)
                    pos = 0; cool = cooldown
            else:  # Short
                if l[i] <= ep * (1 - tp/100):
                    pnl = tp - COMM * 100
                    trades.append(pnl); eq *= (1 + pnl/100)
                    pos = 0; cool = cooldown
                elif h[i] >= ep * (1 + sl/100):
                    pnl = max((1 - c[i]/ep)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append(pnl); eq *= (1 + pnl/100)
                    pos = 0; cool = cooldown
        
        if not pos and cool == 0:
            if above and golden:
                pos = 1; ep = c[i]; side = 1
            elif below and death:
                pos = 1; ep = c[i]; side = -1
        
        if not pos and cool > 0:
            cool -= 1
        cv.append(eq)
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100 if side == 1 else (1 - c[-1]/ep)*100 - COMM*100
        trades.append(pnl); eq *= (1 + pnl/100)
    
    if len(trades) < 3: return None
    
    w = sum(1 for p in trades if p > 0)
    win_trades = [p for p in trades if p > 0]
    loss_trades = [p for p in trades if p < 0]
    avg_win = np.mean(win_trades) if win_trades else 0
    avg_loss = abs(np.mean(loss_trades)) if loss_trades else 0
    
    # Max drawdown
    cv_series = pd.Series(cv)
    dd = ((cv_series - cv_series.expanding().max()) / cv_series.expanding().max() * 100).min()
    
    return {
        'trades': len(trades), 'wins': w, 'losses': len(trades) - w,
        'wr': w / len(trades) * 100,
        'pnl': eq - CAP,
        'eq': eq,
        'dd': dd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }

# Load tradeable coins
with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = set(d['halal'] + d['halal2'])
print(f"🎯 Ichimoku 8h Ultra (3/9/18) TP5/SL2.5 | Cooldown=2 | Commission=0.2%")
print(f"📦 Tradeable coins (halal+halal2): {len(tradeable)}\n")

all_results = {}
grand_total = 0

for period_name in ['2023', 'PREV', 'CUR']:
    print(f"{'='*60}")
    print(f"📅 {period_name}")
    print(f"{'='*60}")
    
    period_results = []
    failed = 0
    no_trades = 0
    
    for sym in sorted(tradeable):
        data = load(sym, period_name)
        if data is None:
            failed += 1
            continue
        
        c, h, l, o, ts = data
        resampled = resample_8h(c, h, l, o, ts)
        if resampled is None:
            no_trades += 1
            continue
        
        c8, h8, l8, o8 = resampled
        r = ichimoku(c8, h8, l8, o8, tenkan=3, kijun=9, senkou=18, tp=5, sl=2.5, cooldown=2)
        if r is None:
            no_trades += 1
            continue
        
        r['sym'] = sym
        period_results.append(r)
    
    if not period_results:
        print("  No trades found!")
        continue
    
    total_trades = sum(r['trades'] for r in period_results)
    total_wins = sum(r['wins'] for r in period_results)
    total_pnl = sum(r['pnl'] for r in period_results)
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    green = sum(1 for r in period_results if r['pnl'] > 0)
    red = sum(1 for r in period_results if r['pnl'] <= 0)
    
    # Sharpe (approximate)
    all_trade_pnls = []
    for r in period_results:
        all_trade_pnls.extend([r['avg_win'] if i < r['wins'] else -r['avg_loss'] for i in range(r['trades'])])
    sharpe = np.mean(all_trade_pnls) / np.std(all_trade_pnls) * np.sqrt(len(all_trade_pnls)) / 10 if len(all_trade_pnls) > 1 else 0
    
    # Max DD across all
    worst_dd = min(r['dd'] for r in period_results)
    
    # Annualized return
    eq_final = CAP + total_pnl
    years = 1
    annual = ((eq_final / CAP) ** (1/years) - 1) * 100
    
    # Top/Bottom 5
    sorted_by_pnl = sorted(period_results, key=lambda x: x['pnl'], reverse=True)
    
    print(f"\n📊 إجمالي الصفقات: {total_trades} | 🟢 ربح: {total_wins} | 🔴 خسارة: {total_trades - total_wins}")
    print(f"📈 WR: {wr:.1f}% | 🟢 عملات خضرا: {green} | 🔴 حمرا: {red}")
    print(f"💵 إجمالي الربح: ${total_pnl:+,.0f} | 📊 R:R نظري: {5/2.5:.1f}")
    print(f"📉 أسوأ سحب: {worst_dd:.1f}% | شارپ: {sharpe:.2f}")
    print(f"🏦 القيمة النهائية: ${eq_final:,.0f} | 📈 سنوي: {annual:.1f}%")
    print(f"❌ فشل تحميل: {failed} | ⚠️ بدون صفقات: {no_trades}")
    
    print(f"\n🏆 أفضل 5:")
    for r in sorted_by_pnl[:5]:
        print(f"  {r['sym']:8s} | {r['trades']:4d} صفقة | WR={r['wr']:.0f}% | ${r['pnl']:+,.0f} | DD={r['dd']:.1f}%")
    
    print(f"\n👎 أسوأ 5:")
    for r in sorted_by_pnl[-5:]:
        print(f"  {r['sym']:8s} | {r['trades']:4d} صفقة | WR={r['wr']:.0f}% | ${r['pnl']:+,.0f} | DD={r['dd']:.1f}%")
    
    all_results[period_name] = {
        'trades': total_trades, 'wins': total_wins, 'wr': wr,
        'pnl': total_pnl, 'green': green, 'red': red,
        'dd': worst_dd, 'sharpe': sharpe, 'annual': annual,
        'eq': eq_final, 'coins': len(period_results),
        'per_coin': period_results,
    }
    grand_total += total_pnl

# Final summary
print(f"\n{'='*60}")
print(f"🔥 الملخص النهائي — 3 سنوات")
print(f"{'='*60}")
for p in ['2023', 'PREV', 'CUR']:
    r = all_results.get(p)
    if r:
        print(f"📅 {p:5s} | {r['coins']:3d} عملة | {r['trades']:5d} صفقة | WR={r['wr']:.1f}% | ${r['pnl']:+,.0f} | DD={r['dd']:.1f}% | سنوي={r['annual']:.1f}%")
print(f"{'─'*60}")
print(f"💰 المجموع الكلي: ${grand_total:+,.0f}")
print(f"📊 متوسط WR: {np.mean([r['wr'] for r in all_results.values()]):.1f}%")
