#!/usr/bin/env python3
"""
🐋 Whale Sniper Monitor — Live Signal Feed
============================================
Monitors @WhaleSniper Telegram channel via web scraping.
Reports new LONG signals with volume >= 200K.
Filters: blocked coins, stablecoins.
"""
import json, os, re, sys
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import subprocess

CHANNEL = 'https://t.me/s/WhaleSniper'
STATE_FILE = '/data/trading28/live_state.json'
SIGNALS_FILE = '/data/trading28/live_signals.json'

STABLES = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDE', 'XUSD',
    'BFUSD', 'FDUSD', 'USDD', 'FRAX', 'LUSD', 'PYUSD',
    'USDJ', 'RLUSD', 'XAUT', 'USD1', 'EUR'
}

BLOCKED = {
    'SUPER', 'ORCA', 'VANA', 'W', 'DOGS', 'MET',
    'XLM', 'BB', 'COS', 'LUNA', 'S'
}

MIN_VOL = 200000  # Minimum USDT volume

def parse_volume(text):
    """Parse volume like '225K' or '1.66M' to number"""
    text = text.strip().upper().replace('USDT', '').replace('$', '').strip()
    if 'K' in text:
        return float(text.replace('K', '')) * 1000
    if 'M' in text:
        return float(text.replace('M', '')) * 1000000
    return float(text)

def parse_message(text, dt):
    """Parse a WhaleSniper message. Returns dict or None."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    full_text = ' '.join(lines)

    # Extract symbol
    symbol_match = re.search(r'#(\w+)', full_text)
    if not symbol_match:
        return None
    symbol = symbol_match.group(1)

    # Skip non-LONG signals
    if 'selling' in full_text.lower():
        return {'symbol': symbol, 'dt': dt, 'direction': 'SHORT', 'skip': True}
    if 'buying' not in full_text.lower():
        return None

    # Skip blocked and stablecoins
    if symbol in STABLES or symbol in BLOCKED:
        return {'symbol': symbol, 'dt': dt, 'direction': 'LONG', 'skip': True}

    # Extract volume
    vol_match = re.search(r'([\d.]+[KM])\s*(?:USDT)?\s*in', full_text)
    volume = parse_volume(vol_match.group(1)) if vol_match else 0

    # Extract price
    price_match = re.search(r'P:\s*([\d.]+)', full_text)
    price = float(price_match.group(1)) if price_match else 0

    # Extract change %
    change_match = re.search(r'\(([+-]?[\d.]+)%\)', full_text)
    change_pct = float(change_match.group(1)) if change_match else 0

    return {
        'symbol': symbol,
        'dt': dt,
        'direction': 'LONG',
        'volume_usdt': volume,
        'price': price,
        'change_pct': change_pct,
        'skip': volume < MIN_VOL
    }


def fetch_signals():
    """Fetch and parse signals from WhaleSniper channel."""
    try:
        result = subprocess.run(
            ['curl', '-s', CHANNEL],
            capture_output=True, text=True, timeout=30
        )
        soup = BeautifulSoup(result.stdout, 'html.parser')
    except Exception as e:
        print(f'Fetch error: {e}')
        return []

    signals = []
    containers = soup.find_all('div', class_='tgme_widget_message_wrap')

    for c in containers:
        # Get message ID
        view_link = c.find('a', class_='tgme_widget_message_date')
        if not view_link:
            continue
        href = view_link.get('href', '')
        mid_match = re.search(r'/(\d+)', href)
        msg_id = int(mid_match.group(1)) if mid_match else 0

        # Get timestamp
        time_elem = c.find('time')
        if not time_elem or not time_elem.get('datetime'):
            continue
        dt = datetime.fromisoformat(time_elem.get('datetime'))

        # Get text
        text_elem = c.find('div', class_='tgme_widget_message_text')
        if not text_elem:
            continue
        text = text_elem.get_text(separator='\n')

        parsed = parse_message(text, dt)
        if parsed:
            parsed['msg_id'] = msg_id
            signals.append(parsed)

    return signals


def load_state():
    """Load last processed message ID."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {'last_msg_id': 0, 'history': []}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, default=str, indent=2)


def main():
    state = load_state()
    all_signals = fetch_signals()

    new_signals = [s for s in all_signals if s['msg_id'] > state['last_msg_id']]
    new_signals.sort(key=lambda s: s['msg_id'])

    if not new_signals:
        # Silent — no new signals
        return

    print(f'🔔 إشارات جديدة: {len(new_signals)}')
    print()

    active = []
    skipped = []
    for s in new_signals:
        if s['skip']:
            skipped.append(s)
        else:
            active.append(s)

    if skipped:
        print(f'🚫 تم التخطي ({len(skipped)}):')
        for s in skipped:
            vol = s.get('volume_usdt', 0)
            reason = 'شورت' if s.get('direction') == 'SHORT' else f'حجم=${vol:,.0f}' if vol < MIN_VOL else 'محظور/مستقر'
            print(f'  {s["symbol"]:<12} | {s["dt"].strftime("%H:%M")} | {reason}')

    if active:
        print()
        print('=' * 60)
        print(f'🔔 إشارات نشطة ({len(active)})')
        print('=' * 60)
        for s in active:
            print(f'  {s["symbol"]:<12} | {s["dt"].strftime("%m/%d %H:%M")} UTC')
            print(f'    حجم: ${s["volume_usdt"]:,.0f} | سعر: {s["price"]} | تغير: {s["change_pct"]:+.2f}%')
            print(f'    الحالة: ⏳ بانتظار تأكيد الحوت')
        print()
        print(f'المجموع: {len(active)} إشارات بانتظار تأكيد الحوت.')

    # Update state
    state['last_msg_id'] = max(s['msg_id'] for s in all_signals)

    # Add to history (keep last 200)
    for s in new_signals:
        state['history'].append({
            'msg_id': s['msg_id'],
            'symbol': s['symbol'],
            'dt': s['dt'].isoformat(),
            'direction': s['direction'],
            'volume_usdt': s.get('volume_usdt', 0),
            'price': s.get('price', 0),
            'skip': s['skip']
        })
    state['history'] = state['history'][-200:]

    save_state(state)

    # Save active signals for Hunter Whale processing (APPEND, don't overwrite)
    existing_signals = []
    if os.path.exists(SIGNALS_FILE):
        try:
            with open(SIGNALS_FILE) as f:
                existing_signals = json.load(f)
        except:
            pass

    existing_ids = {s['msg_id'] for s in existing_signals}
    safe_active = []
    for s in active:
        if s['msg_id'] not in existing_ids:
            safe_active.append({
                'symbol': s['symbol'],
                'dt': s['dt'].isoformat(),
                'volume_usdt': s.get('volume_usdt', 0),
                'price': s.get('price', 0),
                'msg_id': s['msg_id']
            })

    if safe_active:
        existing_signals.extend(safe_active)
        # Keep only last 50 to prevent file bloat
        existing_signals = existing_signals[-50:]
        with open(SIGNALS_FILE, 'w') as f:
            json.dump(existing_signals, f, default=str)


if __name__ == '__main__':
    main()
