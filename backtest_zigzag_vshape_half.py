#!/usr/bin/env python3
"""ZigZag V-Shape Correction + Half TP 0.5% + BE — 212 coins full backtest"""
import json, numpy as np, pandas as pd, os, gc, sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag

COMM = 0.20
CAPITAL = 1000
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

DEPTH = 10
DEV = 1.0
D = DEPTH // 2  # 5 bars each side
CONFIRM = D      # 5 bars after pivot to confirm
MAX_POS = 2
TIME_BARS = 120

# Exit params
TP_PCT = 1.0
SL_ABOVE_L2_PCT = -0.5  # SL = L2 - 0.5%
HALF_TP_PCT = 0.5      # Half position exits at 0.5%
DIST_FILTER = 0.5       # Entry must be within 0.5% of L2

def find_zpatterns(pv):
    """Find H1→L1→H2→L2 patterns"""
    pats = []
    for i in range(len(pv) - 3):
        p0, p1, p2, p3 = pv[i], pv[i+1], pv[i+2], pv[i+3]
        if p0[2] == 'H' and p1[2] == 'L' and p2[2] == 'H' and p3[2] == 'L':
            A = p0[1] - p1[1]  # wave A
            B = p2[1] - p1[1]  # wave B
            C = p2[1] - p3[1]  # wave C
            if A > 0 and B > 0 and C > 0:
                ret_B = B / A
                if 0.38 <= ret_B <= 0.79 and p3[1] < p1[1]:  # L2 < L1 (new low)
                    pats.append((p0, p1, p2, p3, A, B, C, ret_B))
    return pats

def simulate_one_coin(close, high, low, coin_name):
    """Generate trades for one coin"""
    n = len(close)
    
    # Compute zigzag
    pivots = zigzag(high, low, depth=DEPTH, dev=DEV)
    if len(pivots) < 4:
        return []
    
    # Find V-Shape patterns
    patterns = find_zpatterns(pivots)
    
    trades = []
    for H1, L1, H2, L2, A, B, C, ret_B in patterns:
        # Entry bar: L2 confirmed after D bars
        entry_bar = L2[0] + CONFIRM
        if entry_bar >= n:
            continue
        
        entry_price = close[entry_bar]
        
        # Distance filter: entry must be close to L2
        dist_pct = (entry_price - L2[1]) / L2[1] * 100
        if dist_pct > DIST_FILTER:
            continue
        
        # SL below L2
        sl_price = L2[1] * (1 + SL_ABOVE_L2_PCT / 100)
        if sl_price >= entry_price:
            continue  # SL must be below entry
        
        tp_price_full = entry_price * (1 + TP_PCT / 100)
        tp_price_half = entry_price * (1 + HALF_TP_PCT / 100)
        be_price = entry_price  # Breakeven after half TP
        
        # Simulate exit: walk forward from entry
        half_exited = False
        half1_pnl = 0.0  # initialized
        exit_idx = entry_bar
        exit_price = entry_price
        exit_type = 'TIME'
        exit_pnl_net = 0.0
        
        # Half TP price (for checking)
        # Full position PnL will be:
        #   Half1: exits at tp_half or portion of SL
        #   Half2: exits at tp_full or BE or TIME
        
        for j in range(entry_bar + 1, min(n, entry_bar + TIME_BARS + 1)):
            bar_low = low[j]
            bar_high = high[j]
            bar_close = close[j]
            
            if not half_exited:
                # Both halves active
                if bar_high >= tp_price_half:
                    # Half exits at TP 0.5%
                    half1_pnl = (tp_price_half / entry_price - 1) * 100 - COMM / 2  # half commission
                    half_exited = True
                    # Check if also hits full TP in same bar
                    if bar_high >= tp_price_full:
                        half2_pnl = (tp_price_full / entry_price - 1) * 100 - COMM / 2
                        exit_idx = j
                        exit_price = tp_price_full
                        exit_type = 'TP'
                        exit_pnl_net = (half1_pnl + half2_pnl) / 2
                        break
                    # Otherwise continue with BE for half2
                    continue
                
                # Check SL on full position (close-only)
                if j == entry_bar + 1:
                    continue  # First bar: check high first then low
                if bar_close <= sl_price:
                    # Full position hits SL
                    exit_idx = j
                    exit_price = bar_close
                    exit_type = 'SL'
                    exit_pnl_net = (bar_close / entry_price - 1) * 100 - COMM
                    half_exited = True  # mark done
                    break
            
            if half_exited:
                # Second half only, SL at BE
                if bar_high >= tp_price_full:
                    half2_pnl = (tp_price_full / entry_price - 1) * 100 - COMM / 2
                    exit_idx = j
                    exit_price = tp_price_full
                    exit_type = 'TP'
                    exit_pnl_net = (half1_pnl + half2_pnl) / 2
                    break
                
                # BE check (close-only): price closes at or below entry
                if bar_close <= be_price:
                    half2_pnl = (be_price / entry_price - 1) * 100 - COMM / 2
                    exit_idx = j
                    exit_price = bar_close
                    exit_type = 'BE'
                    exit_pnl_net = (half1_pnl + half2_pnl) / 2
                    break
        
        else:
            # TIMEOUT - didn't exit within TIME_BARS
            exit_idx = min(entry_bar + TIME_BARS, n - 1)
            exit_price = close[exit_idx]
            exit_type = 'TIME'
            if half_exited:
                # Half1 already won, half2 exits at market
                half2_pnl = (exit_price / entry_price - 1) * 100 - COMM / 2
                exit_pnl_net = (half1_pnl + half2_pnl) / 2
            else:
                # Neither half hit TP 0.5%
                exit_pnl_net = (exit_price / entry_price - 1) * 100 - COMM
        
        trades.append({
            'coin': coin_name,
            'entry_bar': entry_bar,
            'exit_bar': exit_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_type': exit_type,
            'pnl_net': round(exit_pnl_net, 4),
            'bars': exit_idx - entry_bar,
        })
    
    return trades

# ── Load coins ──
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal'] + shariah['halal2'] if c not in STABLES]

print(f'⏳ جاري معالجة {len(COINS)} عملة...', flush=True)

all_trades = []
skipped = 0
for ci, coin in enumerate(COINS):
    fpath = f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        raw = json.load(f)
    if len(raw) < 200:
        continue
    
    df = pd.DataFrame(raw)
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    coin_trades = simulate_one_coin(close, high, low, coin)
    all_trades.extend(coin_trades)
    
    del df
    gc.collect()
    
    if (ci + 1) % 40 == 0:
        print(f'  ⏳ {ci+1}/{len(COINS)} — {len(all_trades)} trades so far', flush=True)

print(f'\n✅ {len(COINS)} عملة — {len(all_trades)} إشارات قبل MAX_POS', flush=True)

# ── Apply global MAX_POS=2 ──
all_trades.sort(key=lambda t: t['entry_bar'])
executed = []
active_slots = []  # list of exit_bar values

for t in all_trades:
    # Free completed slots
    active_slots = [s for s in active_slots if s > t['entry_bar']]
    if len(active_slots) >= MAX_POS:
        skipped += 1
        continue
    active_slots.append(t['exit_bar'])
    executed.append(t)

print(f'✅ منفذة: {len(executed)} | ⏭️ متخطية: {skipped}', flush=True)

# ── Stats ──
pnls = [t['pnl_net'] for t in executed]
wins = [p for p in pnls if p > 0]
losses = [p for p in pnls if p <= 0]

wr = len(wins) / len(pnls) * 100 if pnls else 0
avg_win = np.mean(wins) if wins else 0
avg_loss = np.mean(losses) if losses else 0
net_pct = sum(pnls)

# Exit breakdown
tp_count = sum(1 for t in executed if t['exit_type'] == 'TP')
sl_count = sum(1 for t in executed if t['exit_type'] == 'SL')
be_count = sum(1 for t in executed if t['exit_type'] == 'BE')
time_count = sum(1 for t in executed if t['exit_type'] == 'TIME')

# Per-coin stats
coin_stats = {}
for t in executed:
    cn = t['coin']
    if cn not in coin_stats:
        coin_stats[cn] = {'wins': 0, 'losses': 0, 'net': 0.0}
    if t['pnl_net'] > 0:
        coin_stats[cn]['wins'] += 1
    else:
        coin_stats[cn]['losses'] += 1
    coin_stats[cn]['net'] += t['pnl_net']

winning_coins = sum(1 for c in coin_stats.values() if c['net'] > 0)
losing_coins = sum(1 for c in coin_stats.values() if c['net'] <= 0)

# Compounding with 10% risk
eq = CAPITAL
peq = CAPITAL
mdd = 0
consecutive_losses = 0
max_consecutive = 0
for p in pnls:
    eq *= (1 + 0.10 * p / 100)
    peq = max(peq, eq)
    dd = (eq - peq) / peq * 100
    mdd = min(mdd, dd)
    if p <= 0:
        consecutive_losses += 1
        max_consecutive = max(max_consecutive, consecutive_losses)
    else:
        consecutive_losses = 0

print(f'''
═══ ZigZag V-Shape + Half TP + BE ═══
depth={DEPTH} dev={DEV}% confirm={CONFIRM}
TP={TP_PCT}% HalfTP={HALF_TP_PCT}% SL=L2{SL_ABOVE_L2_PCT:+.1f}% TIME={TIME_BARS}
Dist<{DIST_FILTER}% MAX_POS={MAX_POS} COMM={COMM}%

📋 إشارات: {len(all_trades):,} | ✅ منفذة: {len(executed):,} | ⏭️ متخطية: {skipped:,}
🟢 ربح: {len(wins):,} | 🔴 خسارة: {len(losses):,}
📈 WR: {wr:.1f}%
🟢 متوسط ربح: {avg_win:+.2f}% | 🔴 متوسط خسارة: {avg_loss:+.2f}%
💰 صافي (جمع): {net_pct:+.1f}%
📉 سحب (10%): {mdd:.1f}% | أطول خسائر: {max_consecutive}
🎯 TP: {tp_count} | 🛑 SL: {sl_count} | 🟰 BE: {be_count} | ⏱️ TIME: {time_count}
🏦 محفظة 10%: ${CAPITAL:,.0f} → ${eq:,.0f} ({((eq/CAPITAL-1)*100):+.1f}%)
🟢 عملات رابحة: {winning_coins} | 🔴 خاسرة: {losing_coins}
''')

# Show exit type detail
print('تفاصيل الخروج:')
for t in executed[:5]:
    print(f"  {t['coin']} {t['exit_type']} {t['pnl_net']:+.2f}% {t['bars']}b entry={t['entry_price']:.6f}")
