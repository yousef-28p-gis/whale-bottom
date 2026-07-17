#!/usr/bin/env python3
"""REKTbinance top 10 BUY liquidations → quick whale backtest (last 3 days)"""
import re, subprocess, json, ccxt, numpy as np, pandas as pd
from html import unescape
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict

# ── Scrape REKTbinance ──
print('⏳ سحب إشارات REKTbinance...')
all_buys = []
target_id = None

for page in range(20):
    url = 'https://t.me/s/REKTbinance' if page == 0 else f'https://t.me/s/REKTbinance?embed=1&before={target_id}'
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

print(f'تم سحب {len(all_buys)} إشارة BUY')

# ── Top 10 by count ──
sym_counts = Counter(b['symbol'] for b in all_buys)
top10 = sym_counts.most_common(10)
print(f'\n🔝 أعلى ١٠ رموز تصفية شراء:')
for sym, count in top10:
    total_qty = sum(b['qty'] for b in all_buys if b['symbol'] == sym)
    print(f'  {sym:<15} {count}x  حجم=${total_qty:,.0f}')

# ── Fetch OHLCV and check whale ──
print(f'\n🔍 جاري فحص الحوت لآخر ٣ أيام...')
exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})
STR=50; WHALE_MIN=0.40
results = []

for sym, count in top10:
    try:
        # Fetch last 5 days of 15m candles
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%SZ'))
        candles = exchange.fetch_ohlcv(f'{sym}', '15m', since=since, limit=500)
        if len(candles) < 200:
            print(f'  {sym}: ❌ بيانات غير كافية ({len(candles)} شمعة)')
            continue
        
        df = pd.DataFrame(candles, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.sort_values('ts').reset_index(drop=True)
        
        # Whale indicator
        LB = 30
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
        
        # Check last 48h for whale entries
        cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=48)).tz_localize(None)
        recent = df[df['ts'] > cutoff]
        entries_found = 0
        last_whale = 0
        
        for i, row in recent.iterrows():
            if row['entry'] and float(row['whale']) >= WHALE_MIN:
                # Check single candle
                if i+1 < len(recent):
                    wn = float(recent.iloc[list(recent.index).index(i)+1]['whale']) if list(recent.index).index(i)+1 < len(recent) else 0
                    if wn >= 0.35:
                        continue
                entries_found += 1
            last_whale = float(row['whale'])
        
        status = '🐋✅' if entries_found > 0 else '❌'
        print(f'  {sym:<15} {count}x تصفية | {status} {entries_found} تأكيد | حوت={last_whale:.3f}')
        results.append({'symbol': sym, 'count': count, 'entries': entries_found, 'whale': last_whale})
        
    except Exception as e:
        print(f'  {sym}: ❌ خطأ - {e}')

# Summary
print(f'\n{"="*50}')
hits = [r for r in results if r['entries'] > 0]
print(f'🐋 تأكيد حوت: {len(hits)}/{len(results)} رموز')
for r in sorted(hits, key=lambda x: -x['count']):
    print(f'   {r["symbol"]}: {r["count"]} تصفية, {r["entries"]} تأكيد, حوت={r["whale"]:.3f}')
