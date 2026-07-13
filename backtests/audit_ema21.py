import pandas as pd, numpy as np
CACHE='/data/trading28/backtests/cache';FEE=0.001;B=200

ddf=pd.read_csv(f'{CACHE}/FET_USDT_1d.csv',parse_dates=['ts'])
ddf['date']=ddf['ts'].dt.date;ddf['sma50']=ddf['close'].rolling(50).mean().shift(1)
df=pd.read_csv(f'{CACHE}/FET_USDT_15m_FULL.csv',parse_dates=['ts'])
df['date']=df['ts'].dt.date;df['sma50d']=df['date'].map(ddf.set_index('date')['sma50'].to_dict())
lo=df['low'].rolling(B).min();al=(df['low']<=lo).astype(float)
lc=abs(df['low']-df['low'].shift(1))/df['low']*100;sm=lc.ewm(span=3,adjust=False).mean()
hi=sm.rolling(B).max();st=np.where(al>0,(sm+hi*2)/3,0)
df['w']=pd.Series(st).ewm(span=3,adjust=False).mean().fillna(0)
df['spike']=(df['w']>df['w'].shift(1))&(df['w'].shift(1)<=0.02)
df['w20']=df['w'].rolling(20).mean();df['w50']=df['w'].rolling(50).mean()
df['wstr']=df['w']/df['w'].rolling(50).max().replace(0,np.nan)*100
df['atr']=(df['high']-df['low']).rolling(14).mean();df['vma']=df['volume'].rolling(20).mean()
df['ema21']=df['close'].ewm(span=21,adjust=False).mean()
entry_sig=df['spike'] & (df['w20']>df['w50']) & (df['close']>df['sma50d']) & (df['wstr']>50) & (df['volume']>df['vma']*1.5)
eis=np.where(entry_sig)[0]

print("="*80)
print("تدقيق اول 5 صفقات - بيانات خام")
print("="*80)

count=0;it=False;ed=0;equity=1000
for ei in eis:
    if ei<500 or ei+1>=len(df):continue
    if it and ei<ed:continue
    if count>=5:break
    
    ts_sig=str(df['ts'].iloc[ei])[:19]
    close_sig=df['close'].iloc[ei]
    ema=df['ema21'].iloc[ei]
    entry=df['open'].iloc[ei+1]
    ts_entry=str(df['ts'].iloc[ei+1])[:19]
    sl_target=ema*0.999
    
    end=min(ei+48,len(df))
    found=False;exit_px=0;exit_ts=""
    for j in range(ei+1, end):
        if df['low'].iloc[j] <= sl_target:
            exit_px=sl_target;exit_ts=str(df['ts'].iloc[j])[:19]
            found=True;break
    if not found:
        exit_px=df['close'].iloc[end-1];exit_ts=str(df['ts'].iloc[end-1])[:19]
    
    pnl=(exit_px-entry)/entry*100-0.2
    
    print("\nصفقة %d:" % (count+1))
    print("  اشارة: %s | اغلاق: $%.4f" % (ts_sig, close_sig))
    print("  EMA21: $%.4f" % ema)
    print("  دخول: %s | OPEN: $%.4f" % (ts_entry, entry))
    print("  هدف: $%.4f" % sl_target)
    
    if found:
        low_at_exit=df['low'].iloc[j]
        print("  خروج: %s | LOW($%.4f) <= هدف($%.4f) ✓" % (exit_ts, low_at_exit, sl_target))
    else:
        print("  خروج زمني: %s | $%.4f" % (exit_ts, exit_px))
    
    print("  PnL: ($%.4f - $%.4f)/$%.4f - 0.2%% = %+.2f%%" % (exit_px, entry, entry, pnl))
    equity*=1+pnl/100
    print("  محفظة: $%.0f" % equity)
    count+=1;it=True;ed=j if found else end-1
