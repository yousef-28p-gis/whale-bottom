#!/usr/bin/env python3
"""
Pattern Discovery: Find common factors among daily top gainers.
For each of the last 30 days, take top 10 pumping coins,
analyze their pre-pump data, and find what they share.
"""
import ccxt
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────
DATA_DIR = '/data/trading28/backtests/pattern_data'
os.makedirs(DATA_DIR, exist_ok=True)

DAYS_TO_ANALYZE = 30      # how many recent days to analyze
TOP_N_PER_DAY = 10         # top gainers per day
LOOKBACK_DAYS = 60         # fetch this many days of data
PRE_PUMP_HOURS = 48        # hours of pre-pump data to analyze
COMMISSION = 0.002         # 0.2%

# ── Load halal coins ────────────────────────────────────
with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
halal_coins = config['halal'] + config['halal2']
# Remove duplicates while preserving order
seen = set()
halal_coins = [c for c in halal_coins if not (c in seen or seen.add(c))]
# Filter out blacklist
blacklist = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','ROBO','PYTH','ANKR'}
halal_coins = [c for c in halal_coins if c not in blacklist]

print(f"🔍 Analyzing {len(halal_coins)} halal coins over {DAYS_TO_ANALYZE} days")

# ── Exchange singleton ──────────────────────────────────
_EXCHANGE = None
def get_exchange():
    global _EXCHANGE
    if _EXCHANGE is None:
        _EXCHANGE = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})
    return _EXCHANGE

# ── Step 1: Fetch daily OHLCV for all coins ────────────
def fetch_all_daily_data(coins):
    """Fetch daily OHLCV for all coins, cache to disk."""
    cache_file = os.path.join(DATA_DIR, 'daily_all.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        print(f"📦 Loaded daily cache: {len(data)} coins")
        return data
    
    exchange = get_exchange()
    since = exchange.parse8601((datetime.now() - timedelta(days=LOOKBACK_DAYS)).isoformat())
    
    all_data = {}
    errors = 0
    
    for i, coin in enumerate(coins):
        symbol = f"{coin}/USDT"
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=LOOKBACK_DAYS)
            if len(ohlcv) >= 30:  # need at least 30 days
                df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                df.set_index('ts', inplace=True)
                # Daily % change
                df['pct'] = df['close'].pct_change() * 100
                all_data[coin] = {
                    'dates': [d.strftime('%Y-%m-%d') for d in df.index],
                    'open': df['open'].tolist(),
                    'high': df['high'].tolist(),
                    'low': df['low'].tolist(),
                    'close': df['close'].tolist(),
                    'volume': df['volume'].tolist(),
                    'pct': df['pct'].tolist(),
                }
            if (i+1) % 20 == 0:
                print(f"  📊 Fetched {i+1}/{len(coins)} coins...")
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ⚠️ {symbol}: {e}")
        time.sleep(0.05)  # rate limit
    
    with open(cache_file, 'w') as f:
        json.dump(all_data, f)
    print(f"✅ Fetched {len(all_data)} coins ({errors} errors), saved to cache")
    return all_data

# ── Step 2: Find top gainers per day ────────────────────
def find_top_gainers(all_data):
    """For each day, find top N coins by % gain."""
    cache_file = os.path.join(DATA_DIR, 'top_gainers.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    
    # Build a dict: date -> [(coin, pct), ...]
    date_gainers = defaultdict(list)
    
    for coin, data in all_data.items():
        for i, (date_str, pct) in enumerate(zip(data['dates'], data['pct'])):
            if pct is not None and not np.isnan(pct) and pct > 0:
                date_gainers[date_str].append((coin, round(pct, 2)))
    
    # Sort dates, take last DAYS_TO_ANALYZE
    sorted_dates = sorted(date_gainers.keys())[-DAYS_TO_ANALYZE:]
    
    top_per_day = {}
    for date_str in sorted_dates:
        gainers = sorted(date_gainers[date_str], key=lambda x: -x[1])[:TOP_N_PER_DAY]
        top_per_day[date_str] = gainers
    
    with open(cache_file, 'w') as f:
        json.dump(top_per_day, f)
    
    print(f"\n📈 Top {TOP_N_PER_DAY} gainers per day ({len(top_per_day)} days):")
    for date_str, gainers in list(top_per_day.items())[-5:]:  # show last 5 days
        coins_str = ', '.join([f"{c}(+{p}%)" for c, p in gainers[:5]])
        print(f"  {date_str}: {coins_str}...")
    
    return top_per_day

# ── Step 3: Fetch pre-pump hourly data ──────────────────
def fetch_pre_pump_hourly(top_per_day, all_data):
    """For each top gainer, fetch 1h candles before the pump day."""
    cache_file = os.path.join(DATA_DIR, 'pre_pump_data.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    
    exchange = get_exchange()
    pre_pump = {}  # date -> coin -> pre-pump indicators
    
    total = sum(len(coins) for coins in top_per_day.values())
    done = 0
    
    for date_str, gainers in top_per_day.items():
        pump_date = datetime.strptime(date_str, '%Y-%m-%d')
        since = exchange.parse8601((pump_date - timedelta(hours=PRE_PUMP_HOURS)).isoformat())
        
        pre_pump[date_str] = {}
        
        for coin, pct in gainers:
            symbol = f"{coin}/USDT"
            try:
                # Fetch 1h candles for PRE_PUMP_HOURS before the pump
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=PRE_PUMP_HOURS)
                
                if len(ohlcv) < 12:  # need at least 12 hours
                    done += 1
                    continue
                
                df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                df.set_index('ts', inplace=True)
                
                # ── Compute indicators ──────────────────────
                close = df['close']
                volume = df['volume']
                
                # RSI(14) on hourly
                delta = close.diff()
                gain = delta.where(delta > 0, 0)
                loss = (-delta).where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                last_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
                
                # Volume ratio (last 4h avg vs 24h avg)
                vol_4h = volume.iloc[-4:].mean() if len(volume) >= 4 else volume.mean()
                vol_24h = volume.iloc[-24:].mean() if len(volume) >= 24 else volume.mean()
                vol_ratio = float(vol_4h / vol_24h) if vol_24h > 0 else 1.0
                
                # Volume trend: last 8h vs previous 16h
                vol_recent_8h = volume.iloc[-8:].mean() if len(volume) >= 8 else volume.mean()
                vol_prev_16h = volume.iloc[-24:-8].mean() if len(volume) >= 24 else volume.mean()
                vol_trend = float(vol_recent_8h / vol_prev_16h) if vol_prev_16h > 0 else 1.0
                
                # Price position in pre-pump range
                price_range = close.max() - close.min()
                price_pos = float((close.iloc[-1] - close.min()) / price_range) if price_range > 0 else 0.5
                
                # Price trend: last 8h vs 24h ago
                price_8h_change = float((close.iloc[-1] / close.iloc[-8] - 1) * 100) if len(close) >= 8 else 0
                
                # Whale detection: any 1h candle with volume > 3x average
                vol_mean = volume.mean()
                vol_std = volume.std()
                whale_bars = int((volume > vol_mean + 2 * vol_std).sum())
                whale_ratio = float(whale_bars / len(volume)) if len(volume) > 0 else 0
                
                # Consecutive red/green hours before pump
                last_6h = close.iloc[-6:]
                green_count = int((last_6h.diff() > 0).sum())
                red_count = int((last_6h.diff() < 0).sum())
                
                # Daily data: previous day's indicators
                # Find this coin's daily data
                coin_daily = all_data.get(coin, {})
                prev_day_pct = None
                prev_day_vol_ratio = None
                days_red_before = 0
                
                if coin_daily and 'dates' in coin_daily:
                    dates_list = coin_daily['dates']
                    pct_list = coin_daily['pct']
                    vol_list = coin_daily['volume']
                    
                    # Find index of pump day
                    try:
                        idx = dates_list.index(date_str)
                        # Previous day change
                        if idx > 0:
                            prev_day_pct = pct_list[idx-1] if pct_list[idx-1] is not None and not np.isnan(pct_list[idx-1]) else None
                        # Count consecutive red days before pump
                        for j in range(idx-1, -1, -1):
                            if pct_list[j] is not None and not np.isnan(pct_list[j]) and pct_list[j] < 0:
                                days_red_before += 1
                            else:
                                break
                        # Volume ratio vs previous day
                        if idx > 0:
                            prev_day_vol_ratio = float(vol_list[idx-1] / vol_list[idx]) if vol_list[idx] > 0 else None
                    except (ValueError, IndexError):
                        pass
                
                pre_pump[date_str][coin] = {
                    'pump_pct': pct,
                    'last_rsi': round(last_rsi, 1) if last_rsi else None,
                    'vol_ratio_4h': round(vol_ratio, 2),
                    'vol_trend': round(vol_trend, 2),
                    'price_position': round(price_pos, 2),
                    'price_8h_change': round(price_8h_change, 2),
                    'whale_bars_count': whale_bars,
                    'whale_ratio': round(whale_ratio, 3),
                    'green_of_last_6h': green_count,
                    'red_of_last_6h': red_count,
                    'prev_day_pct': round(prev_day_pct, 1) if prev_day_pct else None,
                    'days_red_before': days_red_before,
                    'prev_day_vol_ratio': round(prev_day_vol_ratio, 2) if prev_day_vol_ratio else None,
                    'data_hours': len(df),
                }
                
                done += 1
                if done % 30 == 0:
                    print(f"  🔄 Pre-pump: {done}/{total}")
                    
            except Exception as e:
                done += 1
                if done <= 5:
                    print(f"  ⚠️ {symbol} pre-pump: {e}")
            
            time.sleep(0.05)
    
    # Convert to serializable format
    with open(cache_file, 'w') as f:
        json.dump(pre_pump, f, default=str)
    
    print(f"✅ Pre-pump data: {sum(len(v) for v in pre_pump.values())} entries")
    return pre_pump

# ── Step 4: Analyze common factors ──────────────────────
def analyze_patterns(top_per_day, pre_pump):
    """Find common factors across all top gainers."""
    
    all_entries = []
    for date_str, coins in pre_pump.items():
        for coin, indicators in coins.items():
            indicators['date'] = date_str
            indicators['coin'] = coin
            all_entries.append(indicators)
    
    print(f"\n{'='*60}")
    print(f"📊 PATTERN ANALYSIS: {len(all_entries)} top gainers across {len(top_per_day)} days")
    print(f"{'='*60}")
    
    # ── Aggregate stats ──────────────────────────────────
    df = pd.DataFrame(all_entries)
    
    print(f"\n## 1️⃣ RSI Before Pump (hourly RSI-14)")
    rsi_vals = df['last_rsi'].dropna()
    if len(rsi_vals) > 0:
        print(f"   Mean RSI: {rsi_vals.mean():.1f}")
        print(f"   Median RSI: {rsi_vals.median():.1f}")
        print(f"   Min/Max: {rsi_vals.min():.1f} / {rsi_vals.max():.1f}")
        oversold = (rsi_vals < 30).sum()
        oversold_pct = oversold / len(rsi_vals) * 100
        near_oversold = ((rsi_vals >= 30) & (rsi_vals < 40)).sum()
        neutral = ((rsi_vals >= 40) & (rsi_vals <= 60)).sum()
        overbought = (rsi_vals > 70).sum()
        print(f"   RSI < 30 (oversold):  {oversold} ({oversold_pct:.0f}%) ⭐")
        print(f"   RSI 30-40 (near OS):   {near_oversold} ({near_oversold/len(rsi_vals)*100:.0f}%)")
        print(f"   RSI 40-60 (neutral):   {neutral} ({neutral/len(rsi_vals)*100:.0f}%)")
        print(f"   RSI > 70 (overbought): {overbought} ({overbought/len(rsi_vals)*100:.0f}%)")
    
    print(f"\n## 2️⃣ Volume Spike Before Pump")
    vol_ratio = df['vol_ratio_4h'].dropna()
    if len(vol_ratio) > 0:
        print(f"   Mean vol ratio (4h/24h): {vol_ratio.mean():.2f}")
        high_vol = (vol_ratio > 1.5).sum()
        print(f"   High volume (>1.5x):     {high_vol} ({high_vol/len(vol_ratio)*100:.0f}%) ⭐")
        very_high_vol = (vol_ratio > 2.0).sum()
        print(f"   Very high volume (>2x):  {very_high_vol} ({very_high_vol/len(vol_ratio)*100:.0f}%)")
    
    vol_trend = df['vol_trend'].dropna()
    if len(vol_trend) > 0:
        print(f"   Volume trend (recent/older): {vol_trend.mean():.2f}")
        rising_vol = (vol_trend > 1.2).sum()
        print(f"   Rising volume trend:         {rising_vol} ({rising_vol/len(vol_trend)*100:.0f}%)")
    
    print(f"\n## 3️⃣ Whale Activity (2σ volume bars)")
    whale = df['whale_bars_count'].dropna()
    if len(whale) > 0:
        print(f"   Mean whale bars: {whale.mean():.1f}")
        has_whale = (whale > 0).sum()
        print(f"   Has whale bars:  {has_whale} ({has_whale/len(whale)*100:.0f}%) ⭐")
        multi_whale = (whale >= 2).sum()
        print(f"   Multiple whales: {multi_whale} ({multi_whale/len(whale)*100:.0f}%)")
    
    print(f"\n## 4️⃣ Pre-Pump Price Position")
    price_pos = df['price_position'].dropna()
    if len(price_pos) > 0:
        print(f"   Mean position (0=bottom, 1=top): {price_pos.mean():.2f}")
        at_bottom = (price_pos < 0.3).sum()
        print(f"   Near bottom (<0.3): {at_bottom} ({at_bottom/len(price_pos)*100:.0f}%) ⭐")
        at_mid = ((price_pos >= 0.3) & (price_pos < 0.7)).sum()
        print(f"   Mid range (0.3-0.7): {at_mid} ({at_mid/len(price_pos)*100:.0f}%)")
        at_top = (price_pos >= 0.7).sum()
        print(f"   Near top (>0.7): {at_top} ({at_top/len(price_pos)*100:.0f}%)")
    
    print(f"\n## 5️⃣ Price Action Last 8 Hours")
    p8h = df['price_8h_change'].dropna()
    if len(p8h) > 0:
        print(f"   Mean 8h change: {p8h.mean():.2f}%")
        falling = (p8h < -1).sum()
        flat = ((p8h >= -1) & (p8h <= 1)).sum()
        rising = (p8h > 1).sum()
        print(f"   Falling (<-1%):  {falling} ({falling/len(p8h)*100:.0f}%)")
        print(f"   Flat (-1% to 1%): {flat} ({flat/len(p8h)*100:.0f}%)")
        print(f"   Rising (>1%):    {rising} ({rising/len(p8h)*100:.0f}%)")
    
    print(f"\n## 6️⃣ Previous Day Behavior")
    prev_pct = df['prev_day_pct'].dropna()
    if len(prev_pct) > 0:
        print(f"   Mean prev day: {prev_pct.mean():.2f}%")
        prev_red = (prev_pct < 0).sum()
        prev_green = (prev_pct > 0).sum()
        print(f"   Previous day RED:   {prev_red} ({prev_red/len(prev_pct)*100:.0f}%) ⭐")
        print(f"   Previous day GREEN: {prev_green} ({prev_green/len(prev_pct)*100:.0f}%)")
    
    days_red = df['days_red_before'].dropna()
    if len(days_red) > 0:
        print(f"   Mean red days before: {days_red.mean():.1f}")
        red_2plus = (days_red >= 2).sum()
        print(f"   2+ red days before:   {red_2plus} ({red_2plus/len(days_red)*100:.0f}%) ⭐")
        red_3plus = (days_red >= 3).sum()
        print(f"   3+ red days before:   {red_3plus} ({red_3plus/len(days_red)*100:.0f}%)")
    
    print(f"\n## 7️⃣ Candle Color Before Pump (last 6 hours)")
    green = df['green_of_last_6h'].dropna()
    red = df['red_of_last_6h'].dropna()
    if len(green) > 0:
        print(f"   Green candles (mean): {green.mean():.1f}/6")
        print(f"   Red candles (mean):   {red.mean():.1f}/6")
        mostly_red = ((red - green) >= 2).sum()
        mostly_green = ((green - red) >= 2).sum()
        print(f"   Mostly RED (red-green>=2):   {mostly_red} ({mostly_red/len(green)*100:.0f}%)")
        print(f"   Mostly GREEN (green-red>=2): {mostly_green} ({mostly_green/len(green)*100:.0f}%)")
    
    # ── Top recurring coins ──────────────────────────────
    print(f"\n## 8️⃣ Most Frequent Gainers")
    coin_counts = Counter(e['coin'] for e in all_entries)
    for coin, count in coin_counts.most_common(15):
        avg_pump = np.mean([e['pump_pct'] for e in all_entries if e['coin'] == coin])
        print(f"   {coin}: {count}x top 10, avg pump +{avg_pump:.1f}%")
    
    # ── Correlation with pump size ───────────────────────
    print(f"\n## 9️⃣ What Correlates with BIGGER Pumps?")
    pump_sizes = df['pump_pct'].dropna()
    median_pump = pump_sizes.median()
    print(f"   Median pump: +{median_pump:.1f}%")
    
    big = df[df['pump_pct'] >= median_pump]
    small = df[df['pump_pct'] < median_pump]
    
    comparisons = [
        ('RSI', 'last_rsi', 'lower'),
        ('Volume Ratio', 'vol_ratio_4h', 'higher'),
        ('Whale Bars', 'whale_bars_count', 'higher'),
        ('Price Position', 'price_position', 'lower'),
        ('8h Change', 'price_8h_change', 'lower'),
        ('Days Red Before', 'days_red_before', 'higher'),
        ('Prev Day %', 'prev_day_pct', 'lower'),
        ('Red Candles Last 6h', 'red_of_last_6h', 'higher'),
    ]
    
    for name, col, preferred in comparisons:
        big_val = big[col].dropna().mean()
        small_val = small[col].dropna().mean()
        if not np.isnan(big_val) and not np.isnan(small_val):
            diff = big_val - small_val
            direction = '✅' if (preferred == 'higher' and diff > 0) or (preferred == 'lower' and diff < 0) else '❌'
            print(f"   {name:20s}: Big pumps {big_val:.2f} vs Small {small_val:.2f} (diff: {diff:+.2f}) {direction}")
    
    # ── Save report ──────────────────────────────────────
    report_path = os.path.join(DATA_DIR, 'pattern_report.txt')
    report_lines = []
    
    # (recompute for file)
    report_lines.append(f"PATTERN DISCOVERY REPORT")
    report_lines.append(f"{'='*60}")
    report_lines.append(f"Period: {len(top_per_day)} days, {len(all_entries)} top gainers analyzed")
    report_lines.append(f"")
    
    if len(rsi_vals) > 0:
        report_lines.append(f"RSI Before Pump: mean={rsi_vals.mean():.1f}, median={rsi_vals.median():.1f}")
        report_lines.append(f"  Oversold (<30): {oversold_pct:.0f}%")
    
    if len(vol_ratio) > 0:
        report_lines.append(f"Volume Ratio (4h/24h): mean={vol_ratio.mean():.2f}")
        report_lines.append(f"  High vol (>1.5x): {high_vol/len(vol_ratio)*100:.0f}%")
    
    if len(whale) > 0:
        report_lines.append(f"Whale Activity: {has_whale/len(whale)*100:.0f}% have whale bars")
    
    if len(prev_pct) > 0:
        report_lines.append(f"Previous Day RED: {prev_red/len(prev_pct)*100:.0f}%")
    
    if len(days_red) > 0:
        report_lines.append(f"2+ Red Days Before: {red_2plus/len(days_red)*100:.0f}%")
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n📄 Report saved: {report_path}")
    return df

# ── Main ─────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🔍 PATTERN DISCOVERY: What do top gainers share?")
    print("=" * 60)
    
    # Step 1: Get daily data
    print("\n── Step 1: Fetching daily data ──")
    all_data = fetch_all_daily_data(halal_coins)
    
    # Step 2: Find top gainers
    print("\n── Step 2: Finding top gainers ──")
    top_per_day = find_top_gainers(all_data)
    
    # Step 3: Fetch pre-pump hourly data
    print(f"\n── Step 3: Fetching pre-pump hourly data ──")
    pre_pump = fetch_pre_pump_hourly(top_per_day, all_data)
    
    # Step 4: Analyze
    print(f"\n── Step 4: Analyzing patterns ──")
    df = analyze_patterns(top_per_day, pre_pump)
    
    print(f"\n✅ DONE! Analysis complete.")
    return df

if __name__ == '__main__':
    df = main()
