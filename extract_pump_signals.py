#!/usr/bin/env python3
"""Extract all pump signals from Telegram HTML export"""
import re, json, os
from datetime import datetime
from html.parser import HTMLParser

class SignalExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_text = False
        self.in_date = False
        self.current_date = None
        self.current_text = None
        self.signals = []
        self.date_attrs = {}
        
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'div' and 'text' in attrs.get('class', ''):
            self.in_text = True
            self.current_text = ''
        elif tag == 'div' and 'pull_right date details' in attrs.get('class', ''):
            self.in_date = True
            if 'title' in attrs:
                self.current_date = attrs['title']
    
    def handle_endtag(self, tag):
        if tag == 'div' and self.in_text:
            self.in_text = False
            if self.current_text and self.current_date:
                self.parse_signal()
        elif tag == 'div' and self.in_date:
            self.in_date = False
    
    def handle_data(self, data):
        if self.in_text:
            self.current_text += data
        elif self.in_date and not self.current_date:
            # fallback: just time like "00:36"
            d = data.strip()
            if d:
                self.current_date = d
    
    def parse_signal(self):
        text = self.current_text.strip()
        # Pattern: 🚀 Pump - SYM/USDT [Binance]
        m = re.search(r'🚀\s*(?:Pump|Dump)\s*-\s*(\w+)/USDT', text)
        if not m:
            return
        symbol = m.group(1)
        # Parse date
        try:
            dt = datetime.strptime(self.current_date, '%d.%m.%Y %H:%M:%S UTC%z')
        except:
            # Try just time with date from context
            return
        
        # Volume
        vol_m = re.search(r'Volume:\s*\$([\d.]+)([KMB])', text)
        volume_usdt = 0
        if vol_m:
            val = float(vol_m.group(1))
            unit = vol_m.group(2)
            if unit == 'K': val *= 1000
            elif unit == 'M': val *= 1000000
            elif unit == 'B': val *= 1000000000
            volume_usdt = val
        
        self.signals.append({
            'symbol': symbol,
            'dt': dt.isoformat(),
            'volume_usdt': volume_usdt,
            'raw': text[:200]
        })

# Extract from all HTML files
import zipfile
zip_path = '/data/profiles/trading28pbot/cache/documents/doc_986aef31d6cb_ChatExport_2026-07-13.zip'
z = zipfile.ZipFile(zip_path)

all_signals = []
for fname in z.namelist():
    if not fname.endswith('.html'):
        continue
    print(f'Parsing {fname}...')
    html = z.read(fname).decode('utf-8', errors='replace')
    extractor = SignalExtractor()
    extractor.feed(html)
    all_signals.extend(extractor.signals)
    print(f'  Found {len(extractor.signals)} signals')

z.close()

# Sort by date
all_signals.sort(key=lambda s: s['dt'])
print(f'\nTotal signals: {len(all_signals)}')
print(f'Date range: {all_signals[0]["dt"]} → {all_signals[-1]["dt"]}')

# Save
out = '/data/trading28/signals_pumpdetector.json'
with open(out, 'w') as f:
    json.dump(all_signals, f, default=str, indent=2)
print(f'Saved to {out}')

# Show sample
for s in all_signals[:5]:
    print(f'  {s["symbol"]:<10} | {s["dt"]} | ${s["volume_usdt"]:,.0f}')
