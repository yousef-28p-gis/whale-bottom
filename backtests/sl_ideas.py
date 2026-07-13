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
df['atr']=(df['high']-df['low']).rolling(14).mean()
df['vma']=df['volume'].rolling(20).mean()

# Other indicators
df['ema21']=df['close'].ewm(span=21,adjust=False).mean()
df['lowest20']=df['low'].rolling(20).min()
df['highest20']=df['high'].rolling(20).max()

# PSAR (simplified)
def psar(high,low,af_start=0.02,af_step=0.02,af_max=0.2):
    n=len(high);psar_arr=np.zeros(n);trend=np.ones(n)
    ep=np.zeros(n);af=np.zeros(n)
    for i in range(1,n):
        if trend[i-1]==1:
            psar_arr[i]=psar_arr[i-1]+af[i-1]*(ep[i-1]-psar_arr[i-1])
            psar_arr[i]=min(psar_arr[i],low[i-1],low[i-2]) if i>=2 else min(psar_arr[i],low[i-1])
            if high[i]>ep[i-1]:ep[i]=high[i];af[i]=min(af[i-1]+af_step,af_max)
            else:ep[i]=ep[i-1];af[i]=af[i-1]
            if low[i]<psar_arr[i]:trend[i]=-1;psar_arr[i]=ep[i];ep[i]=low[i];af[i]=af_start
        else:
            psar_arr[i]=psar_arr[i-1]-af[i-1]*(psar_arr[i-1]-ep[i-1])
            psar_arr[i]=max(psar_arr[i],high[i-1],high[i-2]) if i>=2 else max(psar_arr[i],high[i-1])
            if low[i]<ep[i-1]:ep[i]=low[i];af[i]=min(af[i-1]+af_step,af_max)
            else:ep[i]=ep[i-1];af[i]=af[i-1]
            if high[i]>psar_arr[i]:trend[i]=1;psar_arr[i]=ep[i];ep[i]=high[i];af[i]=af_start
    return psar_arr

df['psar']=psar(df['high'].values,df['low'].values)

# Sell signal
delta=df['close'].diff();g=delta.clip(lower=0);l=-delta.clip(upper=0)
ag=g.ewm(alpha=1/14,adjust=False).mean();al=l.ewm(alpha=1/14,adjust=False).mean()
df['rsi']=100-(100/(1+ag/al.replace(0,np.nan)))
vs=df['volume'].rolling(20).mean();hh20=df['high'].rolling(20).max().shift(1)
ll10=df['low'].rolling(10).min().shift(1)
c=np.zeros(len(df))
c+=((df['volume']>vs*1.5)&(df['close']<df['open'])).astype(int)
c+=((df['high']>hh20)&(df['close']<hh20)).astype(int)
c+=((df['high']>hh20)&(df['close']<df['open'])).astype(int)
c+=((df['close'].shift(1)>df['open'].shift(1))&(df['volume']>vs*1.5)&(df['close']<df['open'])).astype(int)
c+=(df['low']<ll10).astype(int);c+=((df['high']>df['high'].shift(1))&(df['rsi']<df['rsi'].shift(1))).astype(int)
df['sell']=c/6*100

entry_sig=df['spike'] & (df['w20']>df['w50']) & (df['close']>df['sma50d']) & (df['wstr']>50) & (df['volume']>df['vma']*1.5)
eis=np.where(entry_sig)[0]

def simulate(name, sl_mode):
    trades=[];it=False;ed=0;equity=1000;peak_eq=1000
    cmon=df['ts'].iloc[500].month;cyr=df['ts'].iloc[500].year;mstart=1000
    for ei in eis:
        if ei<500:continue
        if it and ei<ed:continue
        ts=df['ts'].iloc[ei]
        if ts.month!=cmon or ts.year!=cyr:cmon,cyr=ts.month,ts.year;mstart=equity
        if (equity-mstart)/mstart*100<=-7:continue
        e=df['close'].iloc[ei];tp=e+df['atr'].iloc[ei]*3
        
        if sl_mode=='chandelier': sl=e-df['atr'].iloc[ei]*3
        elif sl_mode=='ema21': sl=df['ema21'].iloc[ei]
        elif sl_mode=='psar': sl=df['psar'].iloc[ei]
        elif sl_mode=='donchian': sl=df['lowest20'].iloc[ei]
        elif sl_mode=='supertrend':
            atr_val=df['atr'].iloc[ei];med=(df['high'].iloc[ei]+df['low'].iloc[ei])/2
            sl=med-atr_val*2
        else: sl=0
        
        end=min(ei+192,len(df));r=None;ep=e;ex=ei
        if sl_mode!='none':
            for j in range(ei+1,end):
                if df['low'].iloc[j]<=sl:r='SL';ep=sl;ex=j;break
                if df['high'].iloc[j]>=tp:r='TP';ep=tp;ex=j;break
                if df['sell'].iloc[j]>=60:r='SELL';ep=df['close'].iloc[j];ex=j;break
                if sl_mode=='chandelier':sl=max(sl,df['high'].iloc[j]-df['atr'].iloc[j]*3)
                elif sl_mode=='psar':sl=df['psar'].iloc[j]
        else:
            for j in range(ei+1,end):
                if df['high'].iloc[j]>=tp:r='TP';ep=tp;ex=j;break
                if df['sell'].iloc[j]>=60:r='SELL';ep=df['close'].iloc[j];ex=j;break
        if not r:r='TIME';ep=df['close'].iloc[end-1];ex=end-1
        
        pnl=(ep-e)/e*100-0.2;equity+=equity*(pnl/100)
        if equity>peak_eq:peak_eq=equity
        trades.append(dict(pnl=pnl,r=r));it=True;ed=ex
    
    n=len(trades)
    if n<5:return None
    wins=[t for t in trades if t['pnl']>0];nw=len(wins)
    eqs=[1000]
    for t in trades:eqs.append(eqs[-1]+eqs[-1]*(t['pnl']/100))
    peak=np.maximum.accumulate(eqs);dd=(np.array(eqs)-peak)/peak*100
    sl_count=sum(1 for t in trades if t['r']=='SL')
    tp_count=sum(1 for t in trades if t['r']=='TP')
    sell_count=sum(1 for t in trades if t['r']=='SELL')
    aw=np.mean([t['pnl'] for t in wins]) if wins else 0
    aloss=np.mean([t['pnl'] for t in trades if t['pnl']<=0]) if (n-nw) else 0
    return dict(name=name,n=n,wr=nw/n*100,eq=equity,dd=dd.min(),sl=sl_count,tp=tp_count,sell=sell_count,aw=aw,al=aloss)

results=[simulate('بدون SL (الحالي)','none'),
         simulate('Chandelier Exit (3ATR)','chandelier'),
         simulate('EMA21 كسر','ema21'),
         simulate('Parabolic SAR','psar'),
         simulate('Donchian (lowest 20)','donchian'),
         simulate('SuperTrend (ATR*2)','supertrend')]

results=[r for r in results if r]
base_dd=results[0]['dd']

print(f"{'SL':<25} {'T':>4} {'WR':>4} {'محفظة':>8} {'DD':>6} {'ΔDD':>6} {'SL/TP/SELL':>14} {'W/L':>10}")
print("-"*83)
for r in sorted(results,key=lambda x:x['eq']-abs(x['dd'])*15,reverse=True):
    dd_diff = r['dd'] - base_dd
    dd_delta = f"+{dd_diff:.1f}%" if dd_diff > 0 else f"{dd_diff:.1f}%"
    cnt = str(r['sl']) + "/" + str(r['tp']) + "/" + str(r['sell'])
    wl = f"+{r['aw']:.1f}/{r['al']:.1f}"
    print(f"{r['name']:<25} {r['n']:>4} {r['wr']:>3.0f}% ${r['eq']:>7,.0f} {r['dd']:>5.1f}% {dd_delta:>6} {cnt:>14} {wl:>10}")
