#!/usr/bin/env python3
"""🧪 زجزاج + فيبو — موجة صعود + تصحيح ≤61.8% + اختراق = شراء"""
import json, numpy as np, pandas as pd, os, gc

COMM=0.20; CAPITAL=1000
DATA_DIR='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# Test with different ZigZag sensitivity (N) and Fib levels
TESTS=[
    ('ZZ5_Fib618',   5,  0.618),
    ('ZZ5_Fib50',    5,  0.500),
    ('ZZ5_Fib382',   5,  0.382),
    ('ZZ8_Fib618',   8,  0.618),
    ('ZZ8_Fib50',    8,  0.500),
    ('ZZ10_Fib618', 10,  0.618),
    ('ZZ10_Fib50',  10,  0.500),
    ('ZZ3_Fib618',   3,  0.618),
]

def find_pivots(high, low, N):
    """Find zigzag pivot highs and lows"""
    n=len(high)
    pivots=[]  # (index, type, price) type='H' or 'L'
    for i in range(N, n-N):
        is_high=all(high[i]>=high[i-N:i]) and all(high[i]>=high[i+1:i+N+1])
        if is_high:
            # Check not too close to previous
            if not pivots or abs(i-pivots[-1][0])>N:
                pivots.append((i,'H',high[i]))
        is_low=all(low[i]<=low[i-N:i]) and all(low[i]<=low[i+1:i+N+1])
        if is_low:
            if not pivots or abs(i-pivots[-1][0])>N:
                pivots.append((i,'L',low[i]))
    return pivots

def simulate(close, high, low, N, fib_max, max_pos=2):
    n=len(close)
    pivots=find_pivots(high, low, N)
    if len(pivots)<4: return [],[]
    
    # Walk through pivots, find LOW→HIGH→LOW patterns
    potential=[]  # (entry_idx, entry_price, target_price)
    i=0
    while i<len(pivots)-2:
        if pivots[i][1]=='L' and pivots[i+1][1]=='H':
            # Upward leg: L→H
            leg_low_idx,leg_low_price=pivots[i][0],pivots[i][2]
            leg_high_idx,leg_high_price=pivots[i+1][0],pivots[i+1][2]
            leg_height=leg_high_price-leg_low_price
            if leg_height<=0:
                i+=1; continue
            
            # Find lowest close between leg_high and end of data
            # This is the retracement
            search_end=min(leg_high_idx+200, n)  # look up to 200 candles ahead
            retrace_low=min(close[leg_high_idx:search_end])
            retrace_pct=(leg_high_price-retrace_low)/leg_height
            
            if retrace_pct<=fib_max:
                # Valid retracement — now look for breakout
                for j in range(leg_high_idx+1, search_end):
                    if close[j]>leg_high_price:
                        entry_idx=j
                        entry_price=close[j]
                        target_price=entry_price+leg_height
                        potential.append((entry_idx, entry_price, target_price))
                        break
        i+=1
    
    # Global MAX_POS simulation
    potential.sort(key=lambda x:x[0])
    active=[]; executed=[]; skipped=0
    for eidx,ep,target in potential:
        active=[a for a in active if a[2]>eidx]  # remove expired
        if len(active)>=max_pos: skipped+=1; continue
        
        # Find exit
        exit_idx=eidx; exit_price=ep; exit_type='TIME'
        for j in range(eidx+1, n):
            if close[j]>=target:
                exit_idx=j; exit_price=target; exit_type='TP'; break
        
        pnl=round((exit_price/ep-1)*100-COMM,4)
        active.append((eidx,exit_idx,pnl))
        executed.append(pnl)
    
    return executed, [skipped]

print('⏳ زجزاج + فيبو...', flush=True)
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah=json.load(f)
COINS=[c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]

all_res={}
for ci,coin in enumerate(COINS):
    fpath=f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw); df=df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    close=df['close'].values; high=df['high'].values; low=df['low'].values
    n=len(close)
    
    for name,N,fib in TESTS:
        if name not in all_res: all_res[name]=[]
        executed,skipped=simulate(close,high,low,N,fib)
        all_res[name].extend(executed)
    
    del df; gc.collect()
    if (ci+1)%40==0: print(f'  ⏳ {ci+1}/{len(COINS)}', flush=True)

print(f'\n✅ {len(COINS)} عملة\n')

for name,N,fib in TESTS:
    pnls=all_res[name]
    if not pnls: print(f'{name}: 0 صفقات'); continue
    wins=sum(1 for p in pnls if p>0); loss=sum(1 for p in pnls if p<=0)
    wr=wins/len(pnls)*100 if pnls else 0
    aw=np.mean([p for p in pnls if p>0]) if wins else 0
    al=np.mean([p for p in pnls if p<=0]) if loss else 0
    rr=aw/abs(al) if al!=0 else 0
    net=sum(pnls)
    
    # Simple compounding
    eq=CAPITAL; peq=CAPITAL; mdd=0
    for p in pnls:
        eq+=eq/2*(p/100)
        if eq>peq: peq=eq
        dd=(eq-peq)/peq*100
        if dd<mdd: mdd=dd
    ar=((eq/1000)**(365/122)-1)*100
    
    print(f'{name}: {len(pnls):>5} صفقة | WR={wr:.1f}% | R:R={rr:.1f}x | م.ربح={aw:+.3f}% | م.خسارة={al:+.3f}% | محفظة=${eq:,.0f} | سحب={mdd:.1f}%')
