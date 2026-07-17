#!/usr/bin/env python3
"""Test BinanceWhaleVolumeAlerts + whale confirmation"""
import re, subprocess, json, os, numpy as np, pandas as pd
from html import unescape

result = subprocess.run(['curl', '-s', 'https://t.me/s/BinanceWhaleVolumeAlerts?embed=1'], 
                       capture_output=True, text=True, timeout=30)
html = unescape(result.stdout)

# Extract message divs
msgs = re.findall(r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL)

print(f'Messages found: {len(msgs)}')

signals = []
for msg in msgs:
    # Clean HTML tags
    text = re.sub(r'<[^>]+>', ' ', msg)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Extract symbol
    sym_match = re.search(r'#(\w+USDT)', text)
    if not sym_match: continue
    sym = sym_match.group(1)
    
    direction = 'LONG' if 'LONG' in text else 'SHORT'
    
    # Volume
    vol_match = re.search(r'(?:Long|Short)\s+Volume\s*:\s*\$(\d+\.?\d*)([kK])', text)
    if not vol_match: continue
    vol = float(vol_match.group(1)) * 1000
    
    # Sequence
    seq_match = re.search(r'Sequence\s*:\s*(\d+)', text)
    seq = int(seq_match.group(1)) if seq_match else 0
    
    signals.append({'symbol': sym, 'direction': direction, 'volume': vol, 'seq': seq})

unique_longs = list({s['symbol']: s for s in signals if s['direction'] == 'LONG'}.values())
unique_longs.sort(key=lambda s: -s['volume'])

print(f'Unique LONGs: {len(unique_longs)}')

if not unique_longs:
    print('No signals found! Raw first 500 chars:')
    print(html[:500])
    exit()

for s in unique_longs[:5]:
    print(f'  {s["symbol"]:<15} ${s["volume"]:>10,.0f}  seq={s["seq"]}')

# Check whale
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
print(f'🔍 فحص الحوت لآخر ٥ شمعات (يوليو)')
print(f'{"="*55}')
print(f'{"الرمز":<15} {"حجم":>10} {"Seq":>4} {"حوت":>6} {"W_val":>6}')
print('-' * 48)

hits = 0; checked = 0
for s in unique_longs[:15]:
    df = load_cached(s['symbol'], '2026-07')
    if df is None: continue
    checked += 1
    df_w = whale_indicator(df)
    last5 = df_w.iloc[-5:]
    found = False
    for i in range(len(last5)):
        row = last5.iloc[i]
        if row['entry'] and float(row['whale']) >= WHALE_MIN:
            found = True; break
    if found: hits += 1
    wv = float(df_w.iloc[-1]['whale'])
    icon = '🐋✅' if found else '  ❌'
    print(f'{s["symbol"]:<15} ${s["volume"]:>9,.0f} {s["seq"]:>4} {icon:>6} {wv:.3f}')

print(f'\n🐋 تأكيد حوت (آخر ٥ شمعات): {hits}/{checked}')
print(f'⚠️ هذا فحص مباشر — مش باك تيست تاريخي')
