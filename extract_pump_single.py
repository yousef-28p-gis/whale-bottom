#!/usr/bin/env python3
"""Extract pump signals from a single Telegram HTML export file"""
import re, json
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
            self.current_text = None
        elif tag == 'div' and self.in_date:
            self.in_date = False
    
    def handle_data(self, data):
        if self.in_text:
            self.current_text += data
        elif self.in_date and not self.current_date:
            d = data.strip()
            if d:
                self.current_date = d
    
    def parse_signal(self):
        text = self.current_text.strip()
        m = re.search(r'🚀\s*(?:Pump|Dump)\s*-\s*(\w+)/USDT', text)
        if not m:
            return
        symbol = m.group(1)
        try:
            dt = datetime.strptime(self.current_date, '%d.%m.%Y %H:%M:%S UTC%z')
        except:
            try:
                dt = datetime.strptime(self.current_date, '%d.%m.%Y %H:%M:%S')
            except:
                return
        
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

html_path = '/data/profiles/trading28pbot/cache/documents/doc_d62a144015a6_messages8.JSON'

with open(html_path, encoding='utf-8', errors='replace') as f:
    html = f.read()

extractor = SignalExtractor()
extractor.feed(html)

all_signals = extractor.signals
all_signals.sort(key=lambda s: s['dt'])

print(f'Total signals: {len(all_signals)}')
if all_signals:
    print(f'Date range: {all_signals[0]["dt"]} → {all_signals[-1]["dt"]}')

out = '/data/trading28/signals_pump_single.json'
with open(out, 'w') as f:
    json.dump(all_signals, f, default=str, indent=2)
print(f'Saved to {out}')

for s in all_signals[:5]:
    print(f'  {s["symbol"]:<10} | {s["dt"]} | ${s["volume_usdt"]:,.0f}')
