#!/usr/bin/env python3
"""Build OHLCV cache — memory-efficient version. Limits to ~60 days per pair."""
import json, ccxt, os
from datetime import datetime, timedelta

STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
CACHE='/data/trading28/cache/ohlcv'
os.makedirs(CACHE, exist_ok=True)

with open('/data/trading28/signals_whalesniper_all.json') as f: raw=json.load(f)

# Collect unique (symbol, month) with earliest signal timestamp
pairs={}
for s in raw:
    if s['symbol'] in STABLES or s['direction']!='LONG' or s['volume_usdt']<200000: continue
    dt=datetime.fromisoformat(s['dt'])
    if dt.year!=2026 or dt.month>6: continue
    key=f"{s['symbol']}_{dt.strftime('%Y-%m')}"
    ts=int(dt.timestamp()*1000)
    if key not in pairs or ts<pairs[key][2]:
        pairs[key]=(s['symbol'],dt.strftime('%Y-%m'),ts)

# Remove already-cached pairs
to_fetch={}
for key,(sym,mon,ts) in pairs.items():
    fpath=f'{CACHE}/{key}.json'
    if not os.path.exists(fpath):
        to_fetch[key]=(sym,mon,ts)

print(f"Total pairs: {len(pairs)}, already cached: {len(pairs)-len(to_fetch)}, to fetch: {len(to_fetch)}")

exchange=ccxt.binance()
done=0; fail=0
MAX_DAYS=60  # Max 60 days of data per pair

for key,(sym,mon,sig_ts) in sorted(to_fetch.items()):
    fpath=f'{CACHE}/{key}.json'
    try:
        # Fetch from 5 days before earliest signal to end of month+7 days
        st=datetime.strptime(mon,'%Y-%m')
        # Next month 1st
        if st.month==12: end_st=datetime(st.year+1,1,1)
        else: end_st=datetime(st.year,st.month+1,1)
        
        since=int(st.timestamp()*1000)-5*24*60*60*1000
        end_ts=int(end_st.timestamp()*1000)+7*24*60*60*1000
        
        # Limit: max 60 days = ~5760 candles at 15m
        max_since=end_ts-MAX_DAYS*24*60*60*1000
        if since<max_since: since=max_since
        
        all_candles=[]
        fetch_since=since
        max_iters=6  # 6*1000=6000 candles max
        for _ in range(max_iters):
            candles=exchange.fetch_ohlcv(f'{sym}/USDT','15m',since=fetch_since,limit=1000)
            if not candles: break
            all_candles.extend(candles)
            if candles[-1][0]>=end_ts or len(candles)<1000: break
            fetch_since=candles[-1][0]+1
        
        if all_candles:
            out=[{'ts':c[0],'o':c[1],'h':c[2],'l':c[3],'c':c[4],'v':c[5]} for c in all_candles]
            with open(fpath,'w') as f: json.dump(out,f)
            done+=1
    except Exception as e:
        fail+=1
    
    if (done+fail)%50==0:
        print(f"  {done} done, {fail} fail, {done+fail}/{len(to_fetch)}", flush=True)

print(f"\nDone: {done} fetched, {fail} failed")
print(f"Cache files: {len(os.listdir(CACHE))}")
