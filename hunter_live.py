#!/usr/bin/env python3
"""
🐋 Hunter Whale — Live Confirmation + Position Tracking
=========================================================
Monitors pending signals, checks whale confirmation,
tracks active positions, and reports results.
"""
import json, numpy as np, pandas as pd, os, ccxt
from datetime import datetime, timedelta, timezone

CACHE = '/data/trading28/cache/ohlcv'
PENDING_FILE = '/data/trading28/live_pending.json'
STATE_FILE = '/data/trading28/live_confirmed.json'
LIVE_CACHE = '/data/trading28/cache/live'

TP = 2.5; SL = 2.0; PL = 40; TRAIL = 0.10; MAX_H = 2; STR = 50; WHALE_MIN = 0.35
COMMISSION = 0.20  # 0.1% buy + 0.1% sell (Binance spot)

STABLES = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDE', 'XUSD',
    'BFUSD', 'FDUSD', 'USDD', 'FRAX', 'LUSD', 'PYUSD',
    'USDJ', 'RLUSD', 'XAUT', 'USD1', 'EUR'
}


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
    LB, WF, WS, VM = 30, 2, 5, 1.0
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
    df['str'] = (df['whale'] / df['wp'].replace(0, np.nan) * 100).fillna(0)
    df['vma'] = df['volume'].rolling(20).mean()
    df['entry'] = (df['spike'] & (df['wf'] > df['ws']) &
                   (df['str'] > STR) & (df['volume'] > df['vma'] * VM))
    return df


def get_live_ohlcv(symbol):
    os.makedirs(LIVE_CACHE, exist_ok=True)
    cache_path = f'{LIVE_CACHE}/{symbol}.json'
    df = None; last_ts_ms = 0
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f: data = json.load(f)
            if data:
                df = pd.DataFrame(data)
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
                df = df.sort_values('ts').reset_index(drop=True)
                last_ts_ms = int(df['ts'].iloc[-1].timestamp() * 1000) + 1
        except: pass

    exchange = ccxt.binance()
    if df is None:
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=5)).timestamp() * 1000)
    else:
        since_ms = last_ts_ms
    try:
        new_candles = exchange.fetch_ohlcv(f'{symbol}/USDT', '15m', since=since_ms, limit=500)
    except:
        new_candles = []
    if new_candles:
        new_df = pd.DataFrame(new_candles, columns=['ts','open','high','low','close','volume'])
        new_df['ts'] = pd.to_datetime(new_df['ts'], unit='ms')
        if df is not None:
            df = pd.concat([df, new_df]).drop_duplicates(subset=['ts']).sort_values('ts').reset_index(drop=True)
        else:
            df = new_df
        cache_data = [{'ts': int(r['ts'].timestamp()*1000), 'o': r['open'], 'h': r['high'],
                        'l': r['low'], 'c': r['close'], 'v': r['volume']} for _, r in df.iterrows()]
        with open(cache_path, 'w') as f: json.dump(cache_data, f)
    if df is None or len(df) < 200: return None
    return df


def check_signal(signal):
    sym = signal['symbol']
    dt = datetime.fromisoformat(signal['dt'])
    df = get_live_ohlcv(sym)
    if df is None or len(df) < 200: return None
    df = df.iloc[:-1]  # drop incomplete candle
    df_w = whale_indicator(df)
    dt_naive = dt.replace(tzinfo=None)
    df_w['td'] = abs((df_w['ts'] - dt_naive).dt.total_seconds())
    nearest = df_w['td'].idxmin()
    forward = df_w.iloc[nearest:].reset_index(drop=True)
    for j, row in forward.iterrows():
        if j * 0.25 > 24: break
        if row['entry']:
            whale_val = float(row['whale'])
            if whale_val >= WHALE_MIN:
                return {
                    'symbol': sym, 'confirmed_at': datetime.now(timezone.utc).isoformat(),
                    'whale_val': round(whale_val, 4), 'whale_str': round(float(row['str']), 1),
                    'entry_price': round(float(row['close']), 8),
                    'tp_price': round(float(row['close']) * (1+TP/100), 8),
                    'sl_price': round(float(row['close']) * (1-SL/100), 8),
                }
    return None


def get_live_price(symbol):
    try:
        t = ccxt.binance().fetch_ticker(f'{symbol}/USDT')
        return t['last']
    except:
        return None


def check_position(pos):
    """Check if active position should close. Returns (status, pnl%, detail) or None."""
    entry = pos['entry_price']
    tp = pos['tp_price']; sl = pos['sl_price']
    pl_price = entry + (tp - entry) * (PL / 100)
    current = get_live_price(pos['symbol'])
    if current is None: return None
    
    pnl = round((current - entry) / entry * 100, 4)
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(pos['entered_at'])).total_seconds() / 3600
    
    # Timeout check
    if elapsed >= MAX_H:
        return ('⏰ وقت', pnl, f'انتهت المدة ({MAX_H}h) | إغلاق بسعر {current}')
    
    # TP hit
    if current >= tp:
        return ('🎯 هدف', round(TP, 4), f'وصل الهدف +{TP}% | سعر {current}')
    
    # SL hit
    if current <= sl:
        return ('🛑 ستوب', round(-SL, 4), f'ضرب الستوب -{SL}% | سعر {current}')
    
    # PL + trail logic
    if 'pl_triggered' in pos and pos['pl_triggered']:
        if current > pos.get('peak', entry):
            pos['peak'] = current
            pos['trail_price'] = current * (1 - TRAIL / 100)
        if current <= pos.get('trail_price', 0):
            trail_pnl = round((pos['trail_price'] - entry) / entry * 100, 4)
            return ('🐌 تريل', trail_pnl, f'ارتد من القمة | إغلاق تريل {current}')
    else:
        if current >= pl_price:
            pos['pl_triggered'] = True
            pos['peak'] = current
            pos['trail_price'] = current * (1 - TRAIL / 100)
    
    return None


def load_pending():
    if os.path.exists(PENDING_FILE):
        try:
            with open(PENDING_FILE) as f:
                data = f.read().strip()
                if not data:
                    return []
                return json.loads(data)
        except (json.JSONDecodeError, Exception):
            return []
    return []


def save_pending(pending):
    tmp = PENDING_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(pending, f, default=str, indent=2)
    os.replace(tmp, PENDING_FILE)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                data = f.read().strip()
                if not data:
                    return {'confirmed': {}, 'active_positions': [], 'closed_positions': []}
                return json.loads(data)
        except (json.JSONDecodeError, Exception):
            return {'confirmed': {}, 'active_positions': [], 'closed_positions': []}
    return {'confirmed': {}, 'active_positions': [], 'closed_positions': []}


def save_state(state):
    tmp = STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, default=str, indent=2)
    os.replace(tmp, STATE_FILE)


def main():
    # ── Load new signals ──
    live_signals_path = '/data/trading28/live_signals.json'
    new_signals = []
    if os.path.exists(live_signals_path):
        with open(live_signals_path) as f:
            try: new_signals = json.load(f)
            except: pass

    pending = load_pending()
    existing_ids = {s['msg_id'] for s in pending}
    for s in new_signals:
        if s['msg_id'] not in existing_ids:
            s['added_at'] = datetime.now(timezone.utc).isoformat()
            s['direction'] = 'LONG'
            pending.append(s)

    # ── Check for whale confirmations ──
    state = load_state()
    confirmed_now = []

    for sig in pending[:]:
        result = check_signal(sig)
        if result:
            result['msg_id'] = sig['msg_id']
            result['signal_dt'] = sig['dt']
            result['volume_usdt'] = sig.get('volume_usdt', 0)
            cid = str(sig['msg_id'])
            if cid not in state['confirmed']:
                result['entered_at'] = datetime.now(timezone.utc).isoformat()
                result['pl_triggered'] = False
                result['peak'] = result['entry_price']
                state['confirmed'][cid] = result
                state['active_positions'].append(result)
                confirmed_now.append(result)
            pending.remove(sig)

    # ── Check active positions ──
    closed_now = []
    for pos in state['active_positions'][:]:
        result = check_position(pos)
        if result:
            status, pnl, detail = result
            pos['exit_status'] = status
            pos['exit_pnl'] = pnl
            pos['exit_detail'] = detail
            pos['closed_at'] = datetime.now(timezone.utc).isoformat()
            state['active_positions'].remove(pos)
            state.setdefault('closed_positions', []).append(pos)
            closed_now.append(pos)

    # ── Clean stale pending ──
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    pending = [s for s in pending if datetime.fromisoformat(s['added_at']) > cutoff]
    save_pending(pending)
    save_state(state)

    # ── REPORT ──
    has_content = False

    # 1. New confirmations
    if confirmed_now:
        has_content = True
        print('=' * 60)
        print(f'🐋 تأكيد حوت — إشارات دخول ({len(confirmed_now)})')
        print('=' * 60)
        for c in confirmed_now:
            print(f'  {c["symbol"]:<12} | حوت: {c["whale_val"]:.3f} | قوة: {c["whale_str"]:.0f}%')
            print(f'    دخول: {c["entry_price"]} | هدف: {c["tp_price"]} | ستوب: {c["sl_price"]}')
            print()
        print(f'✅ الهدف: +{TP}% | الستوب: -{SL}% | التريل: {TRAIL}% بعد PL{PL} | المدة: {MAX_H}h')

    # 2. Closed positions
    if closed_now:
        has_content = True
        print(f'\n📢 إغلاق صفقات ({len(closed_now)})')
        print('-' * 40)
        total_net = 0
        for p in closed_now:
            gross = p['exit_pnl']
            net = round(gross - COMMISSION, 4)
            p['exit_net'] = net
            emoji = '🟢' if net > 0 else '🔴'
            print(f'  {emoji} {p["symbol"]:<10} | {p["exit_status"]}')
            print(f'    الإجمالي: {gross:+.2f}% | العمولة: -{COMMISSION}% | الصافي: {net:+.2f}%')
            print(f'    {p["exit_detail"]}')
            total_net += net

        # Cumulative from all closed positions (including history)
        all_closed = state.get('closed_positions', [])
        if len(all_closed) > 0:
            cum_net = sum(p.get('exit_net', round(p.get('exit_pnl', 0) - COMMISSION, 4)) for p in all_closed)
            cum_gross = sum(p.get('exit_pnl', 0) for p in all_closed)
            wins = sum(1 for p in all_closed if p.get('exit_net', p.get('exit_pnl', 0) - COMMISSION) > 0)
            total_trades = len(all_closed)
            wr = round(wins / total_trades * 100, 1)
            cum_emoji = '🟢' if cum_net > 0 else '🔴'
            print(f'  📊 الإجمالي التراكمي ({total_trades} صفقة):')
            print(f'     الإجمالي الخام: {cum_gross:+.2f}% | الصافي: {cum_emoji} {cum_net:+.2f}%')
            print(f'     الرابحة: {wins} 🟢 | الخاسرة: {total_trades - wins} 🔴 | النسبة: {wr}%')

    # 3. Active positions
    active = state['active_positions']
    if active:
        has_content = True
        print(f'\n📊 صفقات نشطة ({len(active)})')
        print('-' * 40)
        for p in active:
            current = get_live_price(p['symbol'])
            if current:
                pnl = round((current - p['entry_price']) / p['entry_price'] * 100, 4)
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(p['entered_at'])).total_seconds() / 60
                pl_status = '🔒 PL' if p.get('pl_triggered') else ''
                emoji = '🟢' if pnl > 0 else '🔴'
                print(f'  {emoji} {p["symbol"]:<10} | {pnl:+.2f}% | {int(elapsed)}د | {pl_status}')
            else:
                print(f'  ⚠️ {p["symbol"]:<10} | تعذر جلب السعر')

    # 4. Pending signals
    if pending:
        has_content = True
        symbols = ', '.join(p['symbol'] for p in pending[:5])
        more = f' +{len(pending)-5}' if len(pending) > 5 else ''
        print(f'\n🔄 جاري المراقبة: {len(pending)} إشارات')
        print(f'   {symbols}{more}')

    # 5. Nothing happening
    if not has_content:
        print('😴 لا توجد إشارات معلقة ولا صفقات نشطة')


if __name__ == '__main__':
    main()
