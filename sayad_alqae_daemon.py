#!/usr/bin/env python3
"""
🐋 صياد القاع — بوت حي 3m
يمسح 212 عملة حلال كل 3 دقائق (متزامن مع حدود 3-دقائق UTC)
استراتيجية: WHALE≥0.25 + RSI<25 + SPK≥2.0 | NoSL + TRAIL 0.05% | TIME=8h

╔══════════════════════════════════════════════════════════════╗
║  ⛔ قوانين ثابتة — ممنوع التعديل بدون تأكيد صريح من يوسف:  ║
║                                                              ║
║  1. CLOSE-ONLY: دخول بسعر إغلاق الشمعة                       ║
║  2. NoSL: بدون وقف خسارة — الخروج بالتريل أو الوقت فقط       ║
║  3. MAX_POS=2 عالمي — ليس لكل عملة                            ║
║  4. دخول بسعر التيكر الحي (get_live_price)                    ║
║  5. المؤشرات: whale(2,3,5,7) + spike + RSI(14)               ║
║  6. الدخول: WHALE≥0.25, SPIKE≥2.0, RSI<25, بدون تأكيد       ║
║  7. الخروج: TP(تيكر)/TRAIL(تيكر,PL=30%)/TIME(8h)             ║
║  8. منع تكرار الدخول: (symbol, bar_ts) مرة واحدة              ║
║  9. تباعد 3 شمعات بين إشارات نفس العملة                       ║
║ 10. عمولة 0.2% (0.1% شراء + 0.1% بيع)                         ║
║                                                              ║
║  🔄 متغيرات مسموح تحديثها:                                    ║
║    - قائمة العملات (EXCLUDE)                                  ║
║    - TP, TRAIL, MAX_H (فقط بطلب يوسف)                         ║
╚══════════════════════════════════════════════════════════════╝
"""
import ccxt, json, os, time, sys
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta

# ═══════════════ الإعدادات ═══════════════
SYMBOLS_CACHE_FILE = '/data/trading28/cache/sayad_alqae_live/'
STATE_FILE = '/data/trading28/sayad_alqae_state.json'
REPORT_FILE = '/data/trading28/sayad_alqae_report.txt'
LOG_FILE = '/data/trading28/sayad_alqae_log.txt'
SHARIAH_FILE = '/data/trading28/config/shariah_coins.json'

SCAN_INTERVAL = 180  # 3 دقائق
COMM = 0.20
TF = '3m'
MAX_CANDLES = 500
MIN_CANDLES = 100

TP = 2.5
PL = 30
TRAIL = 0.05
MAX_HOLD_H = 8
WHALE_MIN = 0.25
RSI_MAX = 25
SPIKE_MIN = 2.0
MAX_POS = 2

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# ═══════════════ مؤشر الحوت 3m ═══════════════
def compute_indicators(df):
    df = df.copy()
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    w = (df['low'].values - df['low_raw'].values) / np.where(df['low_raw'].values != 0, df['low_raw'].values, np.nan) * 100
    df['whale'] = np.clip(w, 0, None)
    vm = df['volume'].rolling(20).mean().values
    df['spike'] = df['volume'].values / np.where(vm != 0, vm, np.nan)
    delta = df['close'].diff().values
    gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(14).mean().values
    loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(14).mean().values
    df['rsi'] = 100 - 100 / (1 + gain / np.where(loss != 0, loss, np.nan))
    return df

def check_entry(df, i):
    """Check if candle i is a valid entry — C2 strong filter"""
    if i < 50 or i >= len(df) - 1:
        return False
    row = df.iloc[i]
    whale_val = float(row['whale'])
    spike_val = float(row['spike'])
    rsi_val = float(row['rsi'])
    
    if np.isnan(whale_val) or np.isnan(spike_val) or np.isnan(rsi_val):
        return False
    if whale_val < WHALE_MIN or spike_val < SPIKE_MIN or rsi_val >= RSI_MAX:
        return False
    
    # تباعد 3 شمعات (يتحقق في المسح الرئيسي)
    return True

# ═══════════════ نظام التخزين ═══════════════
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {'active_positions': [], 'closed_positions': [], 'cumulative_pnl': 0.0, 'total_trades': 0}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, default=str, indent=2)

def write_report(text):
    with open(REPORT_FILE, 'w') as f:
        f.write(text)

def log(msg):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f'[{ts}] {msg}\n')

# ═══════════════ الكاش الحي ═══════════════
def get_live_ohlcv(exchange, symbol):
    """تحميل تراكمي: أول مرة 500 شمعة، بعدين شمعة جديدة"""
    os.makedirs(SYMBOLS_CACHE_FILE, exist_ok=True)
    cache_path = f'{SYMBOLS_CACHE_FILE}/{symbol}.json'
    
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        if raw:
            last_ts = raw[-1]['ts']
        else:
            last_ts = 0
    else:
        raw = []
        last_ts = 0
    
    since_ms = last_ts + 1 if last_ts > 0 else int((datetime.now(timezone.utc) - timedelta(days=2)).timestamp() * 1000)
    
    try:
        new_candles = exchange.fetch_ohlcv(f'{symbol}/USDT', TF, since=since_ms, limit=MAX_CANDLES)
    except Exception as e:
        return None
    
    if not new_candles:
        if raw:
            return raw
        return None
    
    for c in new_candles:
        raw.append({'ts': c[0], 'o': c[1], 'h': c[2], 'l': c[3], 'c': c[4], 'v': c[5]})
    
    seen = set()
    deduped = []
    for r in raw:
        if r['ts'] not in seen:
            seen.add(r['ts'])
            deduped.append(r)
    deduped.sort(key=lambda x: x['ts'])
    
    if len(deduped) > MAX_CANDLES:
        deduped = deduped[-MAX_CANDLES:]
    
    with open(cache_path, 'w') as f:
        json.dump(deduped, f)
    
    return deduped

def get_live_price(exchange, symbol):
    try:
        ticker = exchange.fetch_ticker(f'{symbol}/USDT')
        return ticker['last']
    except:
        return None

# ═══════════════ تزامن مع إغلاق الشموع ═══════════════
def sleep_until_next_3m_boundary():
    now = datetime.now(timezone.utc)
    seconds_since_hour = now.minute * 60 + now.second + now.microsecond / 1_000_000
    next_boundary = ((int(seconds_since_hour) // 180) + 1) * 180
    wait = next_boundary - seconds_since_hour
    if wait > 0:
        time.sleep(wait)

# ═══════════════ MAIN ═══════════════
def main():
    log('🐋 صياد القاع — بدء التشغيل')
    
    with open(SHARIAH_FILE) as f:
        shariah = json.load(f)
    # No exclude list — all 212 coins active
    COINS = [c for c in shariah['halal'] + shariah['halal2'] if c not in STABLES]
    log(f'📋 {len(COINS)} عملة حلال')
    
    exchange = ccxt.binance({'timeout': 10000, 'enableRateLimit': True})
    state = load_state()
    max_bars = int(MAX_HOLD_H * 60 / 3)
    
    scan_count = 0
    
    sleep_until_next_3m_boundary()
    
    while True:
        scan_count += 1
        scan_start = time.time()
        active_symbols = {p['symbol'] for p in state['active_positions']}
        
        report_lines = []
        new_entries = 0
        
        # منع الدخولات المكررة
        used_entries = set()
        for p in state['active_positions'] + state['closed_positions']:
            used_entries.add((p['symbol'], p.get('entry_bar_ts', 0)))
        
        for coin in COINS:
            if coin in active_symbols:
                continue
            
            raw_data = get_live_ohlcv(exchange, coin)
            if not raw_data or len(raw_data) < MIN_CANDLES:
                continue
            
            df = pd.DataFrame(raw_data)
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
            df = compute_indicators(df)
            
            # Check last 3 candles + spacing filter
            last_signals = []  # track signal indices for spacing
            for i in range(max(50, len(df) - 4), len(df) - 1):
                if check_entry(df, i):
                    # تباعد 3 شمعات
                    if last_signals and i - last_signals[-1] <= 3:
                        continue
                    last_signals.append(i)
                    
                    live_price = get_live_price(exchange, coin)
                    if live_price is None:
                        continue
                    entry_price = live_price
                    entry_bar_ts = int(df.iloc[i + 1]['ts'].timestamp() * 1000)
                    
                    if (coin, entry_bar_ts) in used_entries:
                        continue
                    
                    pos = {
                        'symbol': coin,
                        'entry_price': entry_price,
                        'entry_time': datetime.now(timezone.utc).isoformat(),
                        'tp': entry_price * (1 + TP/100),
                        'pl_triggered': False,
                        'peak': entry_price,
                        'trail_price': entry_price,
                        'entry_bar_ts': int(df.iloc[i + 1]['ts'].timestamp() * 1000),
                    }
                    
                    if len(state['active_positions']) < MAX_POS:
                        state['active_positions'].append(pos)
                        active_symbols.add(coin)
                        new_entries += 1
                        
                        report_lines.append(
                            f'🟢 {coin} LONG @ {entry_price:.6f}\n'
                            f'   🎯 TP={pos["tp"]:.6f} | 🐌 TRAIL={TRAIL}% | ⏰ {MAX_HOLD_H}h'
                        )
                        log(f'ENTRY: {coin} @ {entry_price:.6f}')
                    break  # One entry per coin per scan
        
        # Check active positions
        closed_this_scan = []
        for pos in state['active_positions'][:]:
            symbol = pos['symbol']
            current_price = get_live_price(exchange, symbol)
            
            if current_price is None:
                continue
            
            entry = pos['entry_price']
            hold_minutes = (datetime.now(timezone.utc) - datetime.fromisoformat(pos['entry_time'])).total_seconds() / 60
            hold_bars = int(hold_minutes / 3)
            
            exit_reason = None
            exit_pnl = 0
            
            # Timeout
            if hold_bars >= max_bars:
                exit_reason = 'TIME'
                exit_pnl = round((current_price / entry - 1) * 100 - COMM, 4)
            # TP
            elif current_price >= pos['tp']:
                exit_reason = 'TP'
                exit_pnl = round(TP - COMM, 4)
            # PL + Trail — تأكيد بإغلاق شمعة (close-only)
            elif pos['pl_triggered']:
                if current_price > pos['peak']:
                    pos['peak'] = current_price
                    pos['trail_price'] = current_price * (1 - TRAIL/100)
                if current_price <= pos['trail_price']:
                    # انتظر إغلاق الشمعة تحت التريل
                    raw_data = get_live_ohlcv(exchange, symbol)
                    if raw_data and len(raw_data) >= 2:
                        last_close = float(raw_data[-2]['c'])
                        if last_close <= pos['trail_price']:
                            exit_reason = 'TRAIL'
                            exit_pnl = round((last_close / entry - 1) * 100 - COMM, 4)
                            current_price = last_close
            else:
                pl_price = entry + (pos['tp'] - entry) * (PL / 100)
                if current_price >= pl_price:
                    pos['pl_triggered'] = True
                    pos['peak'] = current_price
                    pos['trail_price'] = current_price * (1 - TRAIL/100)
            
            if exit_reason:
                pos['exit_reason'] = exit_reason
                pos['exit_pnl'] = exit_pnl
                pos['exit_price'] = current_price
                pos['exit_time'] = datetime.now(timezone.utc).isoformat()
                
                state['closed_positions'].append(pos)
                state['active_positions'].remove(pos)
                state['total_trades'] += 1
                state['cumulative_pnl'] += exit_pnl
                
                closed_this_scan.append(
                    f'{symbol} | {exit_reason} | {exit_pnl:+.2f}% | ⏱️ {hold_minutes:.0f}د'
                )
                log(f'EXIT: {symbol} {exit_reason} {exit_pnl:+.2f}%')
        
        # Build report
        if new_entries > 0 or closed_this_scan:
            report = '🐋 صياد القاع — تحديث\n'
            report += '━' * 35 + '\n'
            
            if new_entries > 0:
                report += f'\n🟢 دخول جديد ({new_entries}):\n'
                for line in report_lines:
                    report += f'  {line}\n'
            
            if closed_this_scan:
                report += f'\n📊 صفقات مغلقة:\n'
                for line in closed_this_scan:
                    report += f'  {line}\n'
            
            if state['active_positions']:
                report += f'\n📌 صفقات مفتوحة ({len(state["active_positions"])}/{MAX_POS}):\n'
                for pos in state['active_positions']:
                    sym = pos['symbol']
                    cur = get_live_price(exchange, sym)
                    if cur:
                        pnl_live = round((cur / pos['entry_price'] - 1) * 100, 2)
                        report += f'  {sym}: {pos["entry_price"]:.6f} → {cur:.6f} ({pnl_live:+.2f}%)\n'
            
            wins = sum(1 for p in state['closed_positions'] if p.get('exit_pnl', 0) > 0)
            losses = sum(1 for p in state['closed_positions'] if p.get('exit_pnl', 0) <= 0)
            total = wins + losses
            wr = wins / total * 100 if total > 0 else 0
            cumulative = state['cumulative_pnl']
            report += f'\n📊 الإجمالي: {total} صفقة | 🟢{wins} 🔴{losses} | WR {wr:.1f}% | صافي {cumulative:+.2f}%'
            
            write_report(report)
        else:
            active_count = len(state['active_positions'])
            if active_count > 0:
                report = f'🐋 صياد القاع | {active_count}/{MAX_POS} صفقات مفتوحة\n'
                report += '━' * 35 + '\n'
                report += f'\n📌 صفقات مفتوحة:\n'
                for pos in state['active_positions']:
                    sym = pos['symbol']
                    cur = get_live_price(exchange, sym)
                    if cur:
                        pnl_live = round((cur / pos['entry_price'] - 1) * 100, 2)
                        report += f'  {sym}: {pos["entry_price"]:.6f} → {cur:.6f} ({pnl_live:+.2f}%)\n'
                wins = sum(1 for p in state['closed_positions'] if p.get('exit_pnl', 0) > 0)
                losses = sum(1 for p in state['closed_positions'] if p.get('exit_pnl', 0) <= 0)
                total = wins + losses
                wr = wins / total * 100 if total > 0 else 0
                cumulative = state['cumulative_pnl']
                report += f'\n📊 الإجمالي: {total} صفقة | 🟢{wins} 🔴{losses} | WR {wr:.1f}% | صافي {cumulative:+.2f}%'
                write_report(report)
            else:
                write_report(f'😴 لا جديد | 🐋 صياد القاع | {len(COINS)} عملة')
        
        save_state(state)
        
        elapsed = time.time() - scan_start
        log(f'📡 مسح #{scan_count} | {len(COINS)} عملة | {elapsed:.0f}ث | نشط: {len(state["active_positions"])}')
        sleep_until_next_3m_boundary()

if __name__ == '__main__':
    main()
