#!/usr/bin/env python3
"""Extract Whale Sniper signals from single HTML export"""
import re, json
from datetime import datetime
from html.parser import HTMLParser

class WhaleSniperExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_body = False
        self.in_text = False
        self.in_date = False
        self.current_date = None
        self.current_text_parts = []
        self.in_from_name = False
        self.body_depth = 0
        self.body_div_open = False
        self.signals = []
        self._date_title = None
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = attrs_dict.get('class', '')
        
        if tag == 'div' and 'body' in classes and 'details' not in classes:
            self.body_div_open = True
            self.body_depth += 1
        elif tag == 'div' and 'pull_right date details' in classes:
            self.in_date = True
            if 'title' in attrs_dict:
                self._date_title = attrs_dict['title']
        elif tag == 'div' and classes == 'text':
            self.in_text = True
            self._text_buf = []
        elif tag == 'br' and self.in_text:
            self._text_buf.append('\n')
        
    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_date:
                self.in_date = False
                if self._date_title:
                    self.current_date = self._date_title
                    self._date_title = None
            if self.in_text:
                self.in_text = False
                raw = ''.join(self._text_buf)
                self.current_text_parts = raw
            if self.body_div_open:
                self.body_depth -= 1
                if self.body_depth == 0:
                    self.body_div_open = False
                    self._process_message()
        
    def handle_data(self, data):
        if self.in_text:
            self._text_buf.append(data)
        elif self.in_date and not self._date_title:
            pass  # timestamp already captured from title
    
    def _process_message(self):
        if not self.current_date or not self.current_text_parts:
            return
        
        text = self.current_text_parts
        # Reset for next message
        dt_str = self.current_date
        self.current_date = None
        self.current_text_parts = None
        
        # Extract symbol: #SYMBOL
        sym_m = re.search(r'#(\w+)', text)
        if not sym_m:
            return
        symbol = sym_m.group(1)
        
        # Determine direction: buying or selling
        is_buy = 'buying' in text.lower()
        is_sell = 'selling' in text.lower()
        direction = 'LONG' if is_buy else ('SHORT' if is_sell else None)
        if direction is None:
            return
        
        # Extract volume: e.g. "2.54M USDT in 14 minutes"
        vol_m = re.search(r'([\d.]+)([KMB])\s*USDT', text)
        volume_usdt = 0
        if vol_m:
            val = float(vol_m.group(1))
            unit = vol_m.group(2)
            if unit == 'K': val *= 1000
            elif unit == 'M': val *= 1000000
            elif unit == 'B': val *= 1000000000
            volume_usdt = val
        
        # Parse date: "24.06.2026 16:03:56 UTC+02:00"
        try:
            dt = datetime.strptime(dt_str, '%d.%m.%Y %H:%M:%S UTC%z')
        except:
            return
        
        self.signals.append({
            'symbol': symbol,
            'dt': dt.isoformat(),
            'direction': direction,
            'volume_usdt': volume_usdt,
            'raw': text[:300]
        })

html_path = '/data/profiles/trading28pbot/cache/documents/doc_d62a144015a6_messages8.JSON'

with open(html_path, encoding='utf-8', errors='replace') as f:
    html = f.read()

extractor = WhaleSniperExtractor()
extractor.feed(html)

all_signals = extractor.signals
all_signals.sort(key=lambda s: s['dt'])

print(f'Total signals: {len(all_signals)}')
longs = [s for s in all_signals if s['direction'] == 'LONG']
shorts = [s for s in all_signals if s['direction'] == 'SHORT']
print(f'LONG: {len(longs)}, SHORT: {len(shorts)}')

if all_signals:
    print(f'Date range: {all_signals[0]["dt"]} → {all_signals[-1]["dt"]}')

out = '/data/trading28/signals_ws_single.json'
with open(out, 'w') as f:
    json.dump(all_signals, f, default=str, indent=2)
print(f'Saved to {out}')

for s in all_signals[:5]:
    print(f'  {s["symbol"]:<10} | {s["direction"]:<5} | {s["dt"]} | ${s["volume_usdt"]:,.0f}')
