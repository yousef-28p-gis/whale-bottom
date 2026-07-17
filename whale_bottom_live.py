#!/usr/bin/env python3
"""🐋 حوت القاع — بوت تداول حي
يمسح جميع أزواج بايننس كل 15 دقيقة
استراتيجية: حوت ≥ 0.50 + شمعة وحدة + pump24 سالب + RSI < 25 + حظر الخميس/ساعات
"""
import ccxt, json, os, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# ═══════════════════════ CONFIG ═══════════════════════
TP = 3.5; SL = 1.5; PL = 30; TRAIL = 0.10; MAX_H = 6
STR = 50; WHALE_MIN = 0.50; COMM = 0.20
MAX_POS = 2; POS_PCT = 50  # صفقتين × 50%

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
BLOCK_HOURS = {1, 3, 6, 12, 0, 4}  # حظر الساعات السيئة
BLOCK_WEEKDAY = 3  # الخميس

STATE_FILE = '/data/trading28/whale_bottom_state.json'
LIVE_CACHE = '/data/trading28/cache/live'

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
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
                df = df.sort_values('ts').reset_index(drop=True)
                last_ts_ms = int(df['ts'].iloc[-1].timestamp() * 1000) + 1
        except:
            pass
    
    exchange = get_exchange()
    if df is None:
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp() * 1000)
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
        cache_data = [{'ts':int(r['ts'].timestamp()*1000), 'o':r['open'], 'h':r['high'],
                        'l':r['low'], 'c':r['close'], 'v':r['volume']} for _, r in df.iterrows()]
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
    
    if df is None or len(df) < 200:
        return None
    return df

def compute_indicators(df):
    """Compute whale + RSI on DataFrame."""
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
    """Check latest candles for entry signal. Returns entry dict or None."""
    if df_w is None or len(df_w) < 100:
        return None
    
    # Look at last ~20 candles for new entry
    last_idx = len(df_w) - 1
    for i in range(max(50, last_idx - 3), last_idx + 1):
        row = df_w.iloc[i]
        if not row['entry']:
            continue
        
        whale_val = float(row['whale'])
        if whale_val < WHALE_MIN:
            continue
        
        # Single candle check
        if i + 1 < len(df_w):
            whale_next = float(df_w.iloc[i + 1]['whale'])
            if whale_next >= 0.35:
                continue
        
        # RSI < 25
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi >= 25:
            continue
        
        # Day/hour filters
        ts = row['ts']
        if ts.weekday() == BLOCK_WEEKDAY:
            continue
        if ts.hour in BLOCK_HOURS:
            continue
        
        # Pump24
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
    """Check if position should close."""
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

def main():
    state = load_state()
    exchange = get_exchange()
    
    # 🕌 تحميل العملات المسموحة مباشرة — بدون تحميل كل الأسواق
    with open('config/shariah_coins.json') as f:
        shariah = json.load(f)
    coins = [c for c in shariah['halal'] + shariah['halal2'] if c not in STABLES]
    
    new_signals = []
    scanned = 0
    
    for sym in coins:
        # Skip if already active
        if any(p['symbol'] == sym for p in state['active']):
            continue
        
        try:
            df = get_live_ohlcv(sym)
            if df is None:
                continue
            df_w = compute_indicators(df)
            result = check_entry(sym, df_w)
            if result:
                # Check position limits
                if len(state['active']) >= MAX_POS:
                    continue
                
                result['pl_triggered'] = False
                result['peak'] = result['entry_price']
                state['active'].append(result)
                new_signals.append(result)
        except:
            pass
        
        scanned += 1
        if scanned % 100 == 0:
            # Save state periodically
            save_state(state)
    
    # Check active positions
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
    
    save_state(state)
    
    # ═══════════ REPORT ═══════════
    has_content = False
    
    if new_signals:
        has_content = True
        print('=' * 50)
        print(f'🐋🔥 حوت القاع — إشارات جديدة ({len(new_signals)})')
        print('=' * 50)
        for s in new_signals:
            print(f'  ✅ {s["symbol"]:<10} | سعر: {s["entry_price"]}')
            print(f'     🐋 حوت: {s["whale_val"]:.3f} | 📉 RSI: {s["rsi"]:.0f} | 📊 Pump24: {s["pump24"]:+.1f}%')
            print(f'     🎯 هدف: {s["tp_price"]} | 🛑 ستوب: {s["sl_price"]}')
            print()
        print(f'⚙️ TP=+{TP}% | SL=-{SL}% | PL={PL}% | تريل={TRAIL}% | مدة={MAX_H}h | صفقتين×50%')
    
    if closed_now:
        has_content = True
        print(f'\n📢 إغلاق صفقات ({len(closed_now)})')
        print('-' * 40)
        total_net = 0
        for p in closed_now:
            net = p.get('exit_net', 0)
            total_net += net
            emoji = '🟢' if net > 0 else '🔴'
            print(f'  {emoji} {p["symbol"]:<10} | {p["exit_status"]} | {net:+.2f}%')
            print(f'     {p["exit_detail"]}')
        
        all_closed = state.get('closed', [])
        if all_closed:
            cum_net = sum(p.get('exit_net', 0) for p in all_closed)
            wins = sum(1 for p in all_closed if p.get('exit_net', 0) > 0)
            total_t = len(all_closed)
            wr = round(wins / total_t * 100, 1) if total_t > 0 else 0
            cum_emoji = '🟢' if cum_net > 0 else '🔴'
            print(f'  📊 تراكمي ({total_t} صفقة): {cum_emoji} {cum_net:+.2f}% | WR {wr}%')
    
    # Active positions
    active = state['active']
    if active:
        has_content = True
        print(f'\n📊 صفقات نشطة ({len(active)})')
        print('-' * 40)
        for p in active:
            current = get_live_price(p['symbol'])
            if current:
                pnl = round((current - p['entry_price']) / p['entry_price'] * 100, 4)
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(p['entered_at'])).total_seconds() / 60
                pl_status = '🔒PL' if p.get('pl_triggered') else ''
                emoji = '🟢' if pnl > 0 else '🔴'
                print(f'  {emoji} {p["symbol"]:<10} | {pnl:+.2f}% | {int(elapsed)}د | {pl_status}')
            else:
                print(f'  ⚠️ {p["symbol"]:<10} | تعذر جلب السعر')
    
    if not has_content:
        print('😴 لا إشارات جديدة ولا صفقات نشطة')
        print(f'🔍 تم مسح {scanned} عملة')

if __name__ == '__main__':
    main()
