#!/usr/bin/env python3
"""REKTbinance: 6 days BUY liquidations + whale check (spot only)"""
import re, subprocess, json, os, numpy as np, pandas as pd
from html import unescape
from collections import Counter, defaultdict

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

# ── Scrape REKTbinance — paginate back ──
print('⏳ جاري سحب بيانات REKTbinance...')
all_buys = []
target_id = None

for page in range(30):  # try up to 30 pages (~few hours)
    if page == 0:
        url = 'https://t.me/s/REKTbinance?embed=1'
    else:
        url = f'https://t.me/s/REKTbinance?embed=1&before={target_id}'
    
    result = subprocess.run(['curl', '-s', url], capture_output=True, text=True, timeout=30)
    html = unescape(result.stdout)
    
    msgs = re.findall(r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL)
    
    for msg in msgs:
        text = re.sub(r'<[^>]+>', ' ', msg)
        text = re.sub(r'\s+', ' ', text).strip()
        
        for m in re.finditer(r'(\w+USDT)\s+BUY\s+([\d,.]+)\s+@\s+([\d.]+)', text):
            sym = m.group(1)
            qty = float(m.group(2).replace(',', ''))
            price = float(m.group(3))
            all_buys.append({'symbol': sym, 'qty': qty, 'price': price})
    
    id_matches = re.findall(r'data-post="REKTbinance/(\d+)"', html)
    if not id_matches: break
    target_id = min(int(x) for x in id_matches) - 1

print(f'تم سحب {len(all_buys)} إشارة BUY liquidation')

# ── Group by symbol ──
sym_counts = Counter(b['symbol'] for b in all_buys)
print(f'رموز فريدة: {len(sym_counts)}')

# ── Check which have spot cache ──
print(f'\n🔍 فحص الكاش والتصفيات:')
print(f'{"الرمز":<15} {"#تصفية":>6} {"كاش":>6} {"حوت":>6}')
print('-' * 40)

results = []
for sym, count in sym_counts.most_common(30):
    df = load_cached(sym, '2026-07')
    has_cache = df is not None
    
    whale_confirmed = False
    whale_val = 0
    
    if has_cache:
        df_w = whale_indicator(df)
        last20 = df_w.iloc[-20:]  # last 5 hours
        for i in range(len(last20)):
            row = last20.iloc[i]
            if row['entry'] and float(row['whale']) >= WHALE_MIN:
                whale_confirmed = True
                break
        whale_val = float(df_w.iloc[-1]['whale'])
    
    cache_icon = '✅' if has_cache else '❌'
    whale_icon = '🐋✅' if whale_confirmed else ('  ❌' if has_cache else '  -')
    
    print(f'{sym:<15} {count:>6} {cache_icon:>6} {whale_icon:>6}')
    
    if has_cache:
        results.append({
            'symbol': sym, 'count': count, 'whale': whale_confirmed,
            'whale_val': whale_val
        })

# ── Summary ──
spot_hits = [r for r in results if r['whale']]
print(f'\n{"="*50}')
print(f'📊 النتيجة:')
print(f'  إجمالي BUY: {len(all_buys)}')
print(f'  رموز فريدة: {len(sym_counts)}')
print(f'  موجودة سبوت: {len(results)}')
print(f'  🐋 تأكيد حوت: {len(spot_hits)}/{len(results)}')
if spot_hits:
    print(f'\n  🐋 الرموز المؤكدة:')
    for r in sorted(spot_hits, key=lambda x: -x['count']):
        print(f'     {r["symbol"]}: {r["count"]} تصفية, حوت={r["whale_val"]:.3f}')