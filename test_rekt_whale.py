#!/usr/bin/env python3
"""Check REKTbinance old liquidations + whale confirmation"""
import re, subprocess, json, os, numpy as np, pandas as pd
from html import unescape
from datetime import datetime, timedelta

# Get older messages (before a specific ID)
# First get recent page to find a message ID
result = subprocess.run(['curl', '-s', 'https://t.me/s/REKTbinance?embed=1'], 
                       capture_output=True, text=True, timeout=30)
html = unescape(result.stdout)

# Find a message ID to paginate back
id_match = re.search(r'data-post="REKTbinance/(\d+)"', html)
if id_match:
    msg_id = int(id_match.group(1))
else:
    print('Could not find message ID')
    exit()

print(f'Latest msg ID: {msg_id}')

# Go back a few hours by paginating
all_buys = []
target_id = msg_id - 500  # ~2-3 hours back

for _ in range(5):  # 5 pages
    result = subprocess.run(['curl', '-s', f'https://t.me/s/REKTbinance?embed=1&before={target_id}'],
                           capture_output=True, text=True, timeout=30)
    html = unescape(result.stdout)
    
    msgs = re.findall(r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL)
    
    for msg in msgs:
        text = re.sub(r'<[^>]+>', ' ', msg)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Find buys - format: 🟢😾 SYMUSDT BUY QTY @ PRICE
        for m in re.finditer(r'(\w+USDT)\s+BUY\s+([\d,.]+)\s+@\s+([\d.]+)', text):
            sym = m.group(1)
            qty = float(m.group(2).replace(',', ''))
            price = float(m.group(3))
            all_buys.append({'symbol': sym, 'qty': qty, 'price': price})
    
    # Get next page ID
    id_matches = re.findall(r'data-post="REKTbinance/(\d+)"', html)
    if id_matches:
        target_id = min(int(x) for x in id_matches) - 1
    else:
        break

print(f'Total BUY liquidations found: {len(all_buys)}')

# Group by symbol, count frequency
from collections import Counter
sym_counts = Counter(b['symbol'] for b in all_buys)
top_syms = sym_counts.most_common(15)

print(f'\n🔝 أعلى ١٥ رمز تصفية شراء:')
for sym, count in top_syms:
    print(f'  {sym:<15} {count}x')

# Now check whale indicator for these symbols
CACHE = '/data/trading28/cache/ohlcv'
STR=50; WHALE_MIN=0.40

def load_cached(sym, mon):
    fpath = f'{CACHE}/{sym}_{mon}.json'
    if not os.path.exists(fpath): return None
    try:
        with open(fpath) as f: data = json.load(f)
        df = pd.DataFrame(data); df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'}).sort_values('ts').reset_index(drop=True)
    except: return None

def whale_indicator(df):
    df = df.copy(); LB = 30
    df['lo'] = df['low'].rolling(LB).min()
    df['lc'] = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
    df['sm'] = df['lc'].ewm(span=3, adjust=False).mean()
    df['hi'] = df['sm'].rolling(LB).max()
    df['raw'] = np.where(df['low'] <= df['lo'], (df['sm'] + df['hi'] * 2) / 3, 0)
    df['whale'] = df['raw'].ewm(span=3, adjust=False).mean().fillna(0)
    df['spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.03)
    df['wf'] = df['whale'].rolling(2).mean(); df['ws'] = df['whale'].rolling(5).mean()
    df['wp'] = df['whale'].rolling(50).max()
    df['str'] = (df['whale'] / df['wp'].replace(0, np.nan) * 100).fillna(0)
    df['vma'] = df['volume'].rolling(20).mean()
    df['entry'] = (df['spike'] & (df['wf'] > df['ws']) & (df['str'] > STR) & (df['volume'] > df['vma'] * 1.0))
    return df

print(f'\n{"="*55}')
print(f'🔍 فحص الحوت للرموز الأكثر تصفية (يوليو)')
print(f'{"="*55}')
print(f'{"الرمز":<15} {"تصفيات":>6} {"حوت":>6} {"W_val":>6} {"آخر سعر":>10}')
print('-' * 55)

hits = 0; checked = 0
for sym, count in top_syms:
    df = load_cached(sym, '2026-07')
    if df is None:
        print(f'{sym:<15} {count:>6} {"❌كاش":>6}')
        continue
    checked += 1
    df_w = whale_indicator(df)
    last10 = df_w.iloc[-10:]
    found = False
    for i in range(len(last10)):
        row = last10.iloc[i]
        if row['entry'] and float(row['whale']) >= WHALE_MIN:
            found = True; break
    if found: hits += 1
    wv = float(df_w.iloc[-1]['whale'])
    last_price = float(df_w.iloc[-1]['close'])
    icon = '🐋✅' if found else '  ❌'
    print(f'{sym:<15} {count:>6} {icon:>6} {wv:.3f}   ${last_price:.4f}')

print(f'\n🐋 تأكيد الحوت: {hits}/{checked}')
print(f'⚠️ فحص مباشر (آخر ١٠ شمعات) — للفحص الكامل نحتاج أرشيف القناة')
