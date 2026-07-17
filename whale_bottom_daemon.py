#!/usr/bin/env python3
"""🐋 حوت القاع — بوت مستمر (دايمن)
يمسح جميع أزواج بايننس كل 15 دقيقة
يبقى شغال 24/7
"""
import ccxt, json, os, numpy as np, pandas as pd, time as _time
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# ═══════════════════════ CONFIG ═══════════════════════
TP = 3.5; SL = 1.5; PL = 30; TRAIL = 0.10; MAX_H = 6
STR = 50; WHALE_MIN = 0.50; COMM = 0.20
MAX_POS = 2; POS_PCT = 50
SCAN_INTERVAL = 900  # 15 minutes

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
BLOCK_HOURS = {1, 3, 6, 12, 0, 4}
BLOCK_WEEKDAY = 3

STATE_FILE = '/data/trading28/whale_bottom_state.json'
LIVE_CACHE = '/data/trading28/cache/live'
REPORT_FILE = '/data/trading28/whale_bottom_report.txt'
LOG_FILE = '/data/trading28/whale_bottom_log.txt'

_exchange = None

def get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})
    return _exchange

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except:
            pass
    return {'active': [], 'closed': [], 'scanned': {}}

def save_state(state):
    tmp = STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, default=str, indent=2)
    os.replace(tmp, STATE_FILE)

def get_live_ohlcv(symbol):
    os.makedirs(LIVE_CACHE, exist_ok=True)
    cache_path = f'{LIVE_CACHE}/{symbol}.json'
    df = None
    last_ts_ms = 0
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                data = json.load(f)
            if data:
                df = pd.DataFrame(data)
                if 'ts' in df.columns:
                    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                if 'o' in df.columns:
                    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
                df = df.sort_values('ts').reset_index(drop=True)
                last_ts_ms = int(df['ts'].iloc[-1].timestamp() * 1000) + 1
        except:
            pass
    
    exchange = get_exchange()
    if df is None:
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=5)).timestamp() * 1000)
    else:
        since_ms = last_ts_ms
    
    try:
        new_candles = exchange.fetch_ohlcv(f'{symbol}/USDT', '15m', since=since_ms, limit=200)
    except:
        return df  # return existing data if fetch fails
    
    if new_candles:
        new_df = pd.DataFrame(new_candles, columns=['ts','open','high','low','close','volume'])
        new_df['ts'] = pd.to_datetime(new_df['ts'], unit='ms')
        if df is not None:
            df = pd.concat([df, new_df]).drop_duplicates(subset=['ts']).sort_values('ts').reset_index(drop=True)
        else:
            df = new_df
        
        # 🧹 نحتفظ بآخر 500 شمعة فقط
        MAX_CANDLES = 500
        if len(df) > MAX_CANDLES:
            df = df.iloc[-MAX_CANDLES:].reset_index(drop=True)
        
        cache_data = [{'ts':int(r['ts'].timestamp()*1000), 'o':r['open'], 'h':r['high'],
                        'l':r['low'], 'c':r['close'], 'v':r['volume']} for _, r in df.iterrows()]
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
    
    if df is None or len(df) < 200:
        return None
    return df

def compute_indicators(df):
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
    df['entry'] = (df['spike'] & (df['wf'] > df['ws']) & (df['str'] > STR) & (df['volume'] > df['vma'] * 1.0))
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def get_live_price(symbol):
    try:
        t = get_exchange().fetch_ticker(f'{symbol}/USDT')
        return t['last']
    except:
        return None

def check_entry(symbol, df_w):
    if df_w is None or len(df_w) < 100:
        return None
    
    last_idx = len(df_w) - 1
    for i in range(max(50, last_idx - 3), last_idx + 1):
        row = df_w.iloc[i]
        if not row['entry']:
            continue
        whale_val = float(row['whale'])
        if whale_val < WHALE_MIN:
            continue
        if i + 1 < len(df_w):
            if float(df_w.iloc[i + 1]['whale']) >= 0.35:
                continue
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi >= 25:
            continue
        ts = row['ts']
        if ts.weekday() == BLOCK_WEEKDAY:
            continue
        if ts.hour in BLOCK_HOURS:
            continue
        ps = max(0, i - 96)
        pb = float(df_w.iloc[ps]['close'])
        ep = float(row['close'])
        pump24 = (ep - pb) / pb * 100 if pb != 0 else 0
        if pump24 >= 0:
            continue
        
        return {
            'symbol': symbol,
            'entry_price': round(ep, 8),
            'tp_price': round(ep * (1 + TP / 100), 8),
            'sl_price': round(ep * (1 - SL / 100), 8),
            'whale_val': round(whale_val, 4),
            'rsi': round(rsi, 1),
            'pump24': round(pump24, 2),
            'signal_ts': str(ts),
            'entered_at': datetime.now(timezone.utc).isoformat()
        }
    return None

def check_position(pos):
    entry = pos['entry_price']
    tp = pos['tp_price']
    sl = pos['sl_price']
    pl_price = entry + (tp - entry) * (PL / 100)
    current = get_live_price(pos['symbol'])
    if current is None:
        return None
    
    pnl = round((current - entry) / entry * 100, 4)
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(pos['entered_at'])).total_seconds() / 3600
    
    if elapsed >= MAX_H:
        return ('⏰ وقت', pnl, f'انتهت المدة ({MAX_H}h) | إغلاق {current}')
    if current >= tp:
        return ('🎯 هدف', round(TP - COMM, 4), f'وصل الهدف +{TP}% | سعر {current}')
    if current <= sl:
        return ('🛑 ستوب', round(-SL - COMM, 4), f'ضرب الستوب -{SL}% | سعر {current}')
    
    if pos.get('pl_triggered'):
        if current > pos.get('peak', entry):
            pos['peak'] = current
            pos['trail_price'] = current * (1 - TRAIL / 100)
        if current <= pos.get('trail_price', 0):
            trail_pnl = round((pos['trail_price'] - entry) / entry * 100 - COMM, 4)
            return ('🐌 تريل', trail_pnl, f'ارتد من القمة | إغلاق تريل {current}')
    else:
        if current >= pl_price:
            pos['pl_triggered'] = True
            pos['peak'] = current
            pos['trail_price'] = current * (1 - TRAIL / 100)
    
    return None

def scan_all(state, coins):
    """Scan all coins for entry signals."""
    new_signals = []
    
    for sym in coins:
        if any(p['symbol'] == sym for p in state['active']):
            continue
        if len(state['active']) >= MAX_POS:
            break
        
        try:
            df = get_live_ohlcv(sym)
            if df is None:
                continue
            df_w = compute_indicators(df)
            result = check_entry(sym, df_w)
            if result:
                result['pl_triggered'] = False
                result['peak'] = result['entry_price']
                state['active'].append(result)
                new_signals.append(result)
        except:
            pass
    
    return new_signals

def check_all_positions(state):
    """Check all active positions for exits."""
    closed_now = []
    for pos in state['active'][:]:
        result = check_position(pos)
        if result:
            status, pnl, detail = result
            pos['exit_status'] = status
            pos['exit_pnl'] = pnl
            pos['exit_detail'] = detail
            pos['closed_at'] = datetime.now(timezone.utc).isoformat()
            pos['exit_net'] = round(pnl, 4)
            state['active'].remove(pos)
            state.setdefault('closed', []).append(pos)
            closed_now.append(pos)
    return closed_now

def report_to_buf(new_signals, closed_now, state, scanned, buf):
    has_content = False
    
    if new_signals:
        has_content = True
        buf.write('=' * 50 + '\n')
        buf.write(f'🐋🔥 حوت القاع — إشارات جديدة ({len(new_signals)})\n')
        buf.write('=' * 50 + '\n')
        for s in new_signals:
            buf.write(f'  ✅ {s["symbol"]:<10} | سعر: {s["entry_price"]}\n')
            buf.write(f'     🐋 حوت: {s["whale_val"]:.3f} | 📉 RSI: {s["rsi"]:.0f} | 📊 Pump24: {s["pump24"]:+.1f}%\n')
            buf.write(f'     🎯 هدف: {s["tp_price"]} | 🛑 ستوب: {s["sl_price"]}\n')
            buf.write('\n')
    
    if closed_now:
        has_content = True
        buf.write(f'📢 إغلاق صفقات ({len(closed_now)})\n')
        buf.write('-' * 40 + '\n')
        total_net = 0
        for p in closed_now:
            net = p.get('exit_net', 0)
            total_net += net
            emoji = '🟢' if net > 0 else '🔴'
            buf.write(f'  {emoji} {p["symbol"]:<10} | {p["exit_status"]} | {net:+.2f}%\n')
        
        all_closed = state.get('closed', [])
        if all_closed:
            cum_net = sum(p.get('exit_net', 0) for p in all_closed)
            wins = sum(1 for p in all_closed if p.get('exit_net', 0) > 0)
            total_t = len(all_closed)
            wr = round(wins / total_t * 100, 1) if total_t > 0 else 0
            cum_emoji = '🟢' if cum_net > 0 else '🔴'
            buf.write(f'  📊 تراكمي ({total_t} صفقة): {cum_emoji} {cum_net:+.2f}% | WR {wr}%\n')
    
    active = state['active']
    if active:
        has_content = True
        buf.write(f'📊 صفقات نشطة ({len(active)})\n')
        buf.write('-' * 40 + '\n')
        for p in active:
            current = get_live_price(p['symbol'])
            if current:
                pnl = round((current - p['entry_price']) / p['entry_price'] * 100, 4)
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(p['entered_at'])).total_seconds() / 60
                pl_status = '🔒PL' if p.get('pl_triggered') else ''
                emoji = '🟢' if pnl > 0 else '🔴'
                buf.write(f'  {emoji} {p["symbol"]:<10} | {pnl:+.2f}% | {int(elapsed)}د | {pl_status}\n')
    
    buf.write('\n')
    return has_content

def main():
    state = load_state()
    exchange = get_exchange()
    
    # 🕌 تحميل العملات المسموحة مباشرة — بدون تحميل كل الأسواق
    with open('config/shariah_coins.json') as f:
        shariah = json.load(f)
    coins = [c for c in shariah['halal'] + shariah['halal2'] if c not in STABLES]
    
    print(f'🐋🔥 حوت القاع — بدء التشغيل')
    print(f'🕌 {len(coins)} عملة (حلال + حلال2) | مسح كل {SCAN_INTERVAL//60} دقيقة')
    print(f'⚙️ TP=+{TP}% SL=-{SL}% PL={PL}% تريل={TRAIL}% مدة={MAX_H}h | صفقتين×{POS_PCT}%')
    print(f'🐋 حوت≥{WHALE_MIN} | 📉 RSI<25')
    print('=' * 50)
    
    while True:
        cycle_start = _time.time()
        
        new_signals = scan_all(state, coins)
        closed_now = check_all_positions(state)
        save_state(state)
        
        # Write report to file for cron delivery
        import io
        buf = io.StringIO()
        
        if new_signals or closed_now or state['active']:
            has = report_to_buf(new_signals, closed_now, state, len(coins), buf)
        else:
            buf.write(f'😴 لا جديد | تم مسح {len(coins)} عملة\n')
        
        with open(REPORT_FILE, 'w') as f:
            f.write(buf.getvalue())
        
        # Also print to stdout for logging
        print(buf.getvalue(), end='')
        
        elapsed = _time.time() - cycle_start
        sleep_time = max(10, SCAN_INTERVAL - elapsed)
        _time.sleep(sleep_time)

if __name__ == '__main__':
    main()
