#!/usr/bin/env python3
"""Dragon + Whale backtest — June 2026. Dragon signals → wait for whale → enter with Dragon TP/SL ladder."""
import json, os, ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from collections import defaultdict

CACHE = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_2026.json'
os.makedirs(CACHE, exist_ok=True)

# ── Hunter Whale params ──
LB, WF, WS, VM = 30, 2, 5, 1.0
STR = 50
WAIT_H = 24       # wait up to 24h for whale
MAX_LOSS_H = 4    # max time for losing trades
MAX_CONCURRENT = 2
POSITION = 500
CAPITAL = 1000
MAX_TP = 10       # filter TP > 10%
MAX_SL = 15       # filter SL > 15%
COMMISSION = 0.20

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# ── 1. Load & filter Dragon signals (June 2026) ──
with open(SIGNALS_FILE) as f:
    raw = json.load(f)

signals = []
for s in raw:
    if not s['dt'].startswith('2026-06'):
        continue
    if s['symbol'] in STABLES:
        continue
    if s['direction'] != 'LONG':
        continue
    de = s['entry']
    tp1_pct = (s['tp1'] - de) / de * 100 if s['tp1'] > de else 999
    sl_pct = (de - s['sl']) / de * 100 if s['sl'] < de else 999
    if not (1 <= tp1_pct <= MAX_TP and 1 <= sl_pct <= MAX_SL):
        continue
    s['tp1_pct'] = tp1_pct
    s['sl_pct'] = sl_pct
    s['dt_obj'] = datetime.strptime(s['dt'], '%Y-%m-%d %H:%M:%S')
    signals.append(s)

print(f'📋 Dragon June signals (LONG, filtered): {len(signals)}')

# ── 2. Fetch missing OHLCV ──
exchange = ccxt.binance()
needed = set()
for s in signals:
    mon = s['dt_obj'].strftime('%Y-%m')
    needed.add((s['symbol'], mon))

to_fetch = []
for sym, mon in needed:
    fpath = f'{CACHE}/{sym}_{mon}.json'
    if not os.path.exists(fpath):
        to_fetch.append((sym, mon))

print(f'Cache: {len(needed)-len(to_fetch)}/{len(needed)} present, fetching {len(to_fetch)}...')

done = 0; fail = 0
for sym, mon in sorted(to_fetch):
    fpath = f'{CACHE}/{sym}_{mon}.json'
    try:
        st = datetime.strptime(mon, '%Y-%m')
        if st.month == 12:
            end_st = datetime(st.year+1, 1, 1)
        else:
            end_st = datetime(st.year, st.month+1, 1)
        since = int(st.timestamp()*1000) - 5*24*60*60*1000
        end_ts = int(end_st.timestamp()*1000) + 7*24*60*60*1000
        max_since = end_ts - 60*24*60*60*1000
        if since < max_since:
            since = max_since
        
        all_c = []
        while since < end_ts:
            candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=since, limit=1000)
            if not candles:
                break
            all_c.extend(candles)
            since = candles[-1][0] + 1
            if len(candles) < 1000:
                break
        
        data = [{'ts':c[0], 'o':c[1], 'h':c[2], 'l':c[3], 'c':c[4], 'v':c[5]} for c in all_c]
        with open(fpath, 'w') as f:
            json.dump(data, f)
        done += 1
        if done % 20 == 0:
            print(f'  {done}/{len(to_fetch)}')
    except Exception as e:
        fail += 1

print(f'Fetched: {done}, failed: {fail}')

# ── 3. Whale indicator ──
def whale_indicator(df):
    df = df.copy()
    df['lo'] = df['low'].rolling(LB).min()
    df['lc'] = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
    df['sm'] = df['lc'].ewm(span=3, adjust=False).mean()
    df['hi'] = df['sm'].rolling(LB).max()
    df['raw'] = np.where(df['low'] <= df['lo'], (df['sm'] + df['hi'] * 2) / 3, 0)
    df['whale'] = df['raw'].ewm(span=3, adjust=False).mean().fillna(0)
    df['spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.03)
    df['wf'] = df['whale'].rolling(WF).mean()
    df['ws'] = df['whale'].rolling(WS).mean()
    df['wp'] = df['whale'].rolling(50).max()
    df['wstr'] = (df['whale'] / df['wp'].replace(0, np.nan) * 100).fillna(0)
    df['vma'] = df['volume'].rolling(20).mean()
    df['entry'] = (df['spike'] & (df['wf'] > df['ws']) &
                   (df['wstr'] > STR) & (df['volume'] > df['vma'] * VM))
    return df

# ── 4. Simulate trade (Dragon progressive SL ladder) ──
def simulate(df, entry_idx, sig):
    entry_price = df.iloc[entry_idx]['close']
    
    # Build Dragon TP/% targets
    tps_pct = []
    for i in range(1, 6):
        tp_key = f'tp{i}'
        if tp_key in sig and sig[tp_key] > sig['entry']:
            p = (sig[tp_key] - sig['entry']) / sig['entry'] * 100
            if 1 <= p <= MAX_TP:
                tps_pct.append(p)
    if not tps_pct:
        tps_pct = [sig['tp1_pct']]
    
    targets = [entry_price * (1 + p/100) for p in tps_pct]
    sl_price = entry_price * (1 - sig['sl_pct']/100)
    
    # Progressive SL ladder: [SL, BE, TP1, TP2, ...]
    sl_ladder = [sl_price, entry_price] + targets[:-1]
    
    ct_idx = cs_idx = 0
    sl_level = sl_ladder[0]
    highest_tp = 0
    ever_profitable = False
    highest_high = entry_price
    
    remaining = df.iloc[entry_idx + 1:]
    if len(remaining) == 0:
        return ('NO_DATA', 0.0)
    
    for j, (_, row) in enumerate(remaining.iterrows()):
        hours = j * 0.25
        
        if row['high'] > highest_high:
            highest_high = row['high']
        if highest_high > entry_price:
            ever_profitable = True
        
        # Time exit for losers only
        if not ever_profitable and hours > MAX_LOSS_H:
            pnl = round((row['close'] - entry_price) / entry_price * 100, 4)
            return ('⏰ وقت', pnl)
        
        # SL hit
        if row['low'] <= sl_level:
            pnl = round((sl_level - entry_price) / entry_price * 100, 4)
            if highest_tp > 0:
                return (f'🎯 TP{highest_tp}_SL', pnl)
            return ('🛑 ستوب', pnl)
        
        # TP hit → advance ladder
        if ct_idx < len(targets) and row['high'] >= targets[ct_idx]:
            highest_tp = ct_idx + 1
            while ct_idx + 1 < len(targets) and row['high'] >= targets[ct_idx + 1]:
                ct_idx += 1
                highest_tp = ct_idx + 1
            cs_idx = ct_idx + 1
            sl_level = sl_ladder[cs_idx] if cs_idx < len(sl_ladder) else sl_ladder[-1]
            ct_idx += 1
            if ct_idx >= len(targets) and highest_tp == len(targets):
                pnl = round((targets[-1] - entry_price) / entry_price * 100, 4)
                return ('🏆 كل الأهداف', pnl)
    
    # End of data
    pnl = round((remaining.iloc[-1]['close'] - entry_price) / entry_price * 100, 4)
    return ('📦 إغلاق', pnl)

# ── 5. Run backtest ──
trades = []
no_cache = 0
no_whale = 0

for sig in signals:
    sym = sig['symbol']
    mon = sig['dt_obj'].strftime('%Y-%m')
    fpath = f'{CACHE}/{sym}_{mon}.json'
    
    if not os.path.exists(fpath):
        no_cache += 1
        continue
    
    with open(fpath) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    df = df.sort_values('ts').reset_index(drop=True)
    
    df_w = whale_indicator(df)
    
    # Find candle at Dragon signal time
    sig_ts = sig['dt_obj'].replace(tzinfo=None)
    df_w['td'] = abs((df_w['ts'] - sig_ts).dt.total_seconds())
    nearest = df_w['td'].idxmin()
    
    # Look forward for whale entry within wait window
    fwd = df_w.iloc[nearest:].reset_index(drop=True)
    wi = None
    for j, row in fwd.iterrows():
        if j * 0.25 > WAIT_H:
            break
        if row['entry']:
            wi = j
            break
    
    if wi is None:
        no_whale += 1
        continue
    
    status, pnl = simulate(fwd, wi, sig)
    trades.append({
        'symbol': sym,
        'dragon_dt': sig['dt'],
        'entry_price': round(float(fwd.iloc[wi]['close']), 8),
        'exit_status': status,
        'pnl': pnl,
        'net': round(pnl - COMMISSION, 4),
    })

# ── 6. Results ──
print(f'\n{"="*60}')
print(f'🐉 دراجون + 🐋 حوت — باك تيست يونيو 2026')
print(f'{"="*60}')
print(f'إشارات دراجون: {len(signals)}')
print(f'بدون كاش: {no_cache}')
print(f'بدون تأكيد حوت: {no_whale}')
print(f'صفقات منفذة: {len(trades)}')
print()

if trades:
    wins = [t for t in trades if t['net'] > 0]
    losses = [t for t in trades if t['net'] <= 0]
    wr = len(wins) / len(trades) * 100
    
    # Exit status breakdown
    status_count = defaultdict(lambda: {'count': 0, 'pnl_sum': 0.0})
    for t in trades:
        status_count[t['exit_status']]['count'] += 1
        status_count[t['exit_status']]['pnl_sum'] += t['net']
    
    total_net = sum(t['net'] for t in trades)
    avg_net = total_net / len(trades)
    avg_win = sum(t['net'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['net'] for t in losses) / len(losses) if losses else 0
    
    print(f'📈 ملخص:')
    print(f'  رابحة: {len(wins)} 🟢 | خاسرة: {len(losses)} 🔴 | WR: {wr:.1f}%')
    print(f'  صافي: {total_net:+.2f}%')
    print(f'  متوسط الربح: {avg_win:+.2f}% | متوسط الخسارة: {avg_loss:+.2f}%')
    print(f'  R:R: {abs(avg_win/avg_loss) if avg_loss != 0 else 999:.1f}')
    print()
    print(f'📋 الإغلاقات:')
    for st, info in sorted(status_count.items(), key=lambda x: -x[1]['count']):
        avg = info['pnl_sum'] / info['count']
        print(f'  {st}: {info["count"]} | مجموع: {info["pnl_sum"]:+.2f}% | متوسط: {avg:+.2f}%')
    
    # Portfolio sim
    port = CAPITAL
    active = []
    for t in sorted(trades, key=lambda x: x['dragon_dt']):
        et = datetime.strptime(t['dragon_dt'], '%Y-%m-%d %H:%M:%S') + timedelta(hours=MAX_LOSS_H+2)
        still = []
        for exit_t, pnl_d in active:
            if et >= exit_t:
                port += POSITION + pnl_d
            else:
                still.append((exit_t, pnl_d))
        active = still
        if len(active) >= MAX_CONCURRENT or port < POSITION:
            continue
        pnl_d = POSITION * (t['net'] / 100)
        active.append((et, pnl_d))
        port -= POSITION
    for _, pnl_d in active:
        port += POSITION + pnl_d
    
    roi = (port - CAPITAL) / CAPITAL * 100
    print(f'\n💼 محفظة: ${CAPITAL} → ${port:.0f} ({roi:+.1f}%)')
    
    # Best/worst
    print(f'\n🟢 أفضل 5:')
    for t in sorted(trades, key=lambda x: -x['net'])[:5]:
        print(f'  {t["symbol"]:<10} | {t["net"]:+.2f}% | {t["exit_status"]} | {t["dragon_dt"][:10]}')
    print(f'\n🔴 أسوأ 5:')
    for t in sorted(trades, key=lambda x: x['net'])[:5]:
        print(f'  {t["symbol"]:<10} | {t["net"]:+.2f}% | {t["exit_status"]} | {t["dragon_dt"][:10]}')

out = '/data/trading28/dragon_whale_june2026.json'
with open(out, 'w') as f:
    json.dump({'trades': trades, 'signals': len(signals), 'no_cache': no_cache, 'no_whale': no_whale}, f, default=str)
print(f'\n✅ محفوظ: {out}')
