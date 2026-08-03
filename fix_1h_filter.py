"""Option 3 + 1h filter using EXISTING config parameters"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; DATA='data/whale_15m_1y'
MAX_SLIPPAGE=1.5; COOLDOWN=48

def load(sym):
    with open(os.path.join(DATA, f'{sym}.json')) as f: d=json.load(f)
    return {'c':np.array(d['c'],float),'h':np.array(d['h'],float),
            'l':np.array(d['l'],float),'o':np.array(d['o'],float),
            'ts':pd.to_datetime(d['ts'],unit='ms')}

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

with open('final_bot_config.json') as f: configs = {r['sym']: r for r in json.load(f)}
coins=sorted([f.replace('.json','') for f in os.listdir(DATA) if f.endswith('.json') and f!='_manifest.json'])

results_4h = []  # original (4h only)
results_1h = []  # +1h on same config
results_both = [] # +4h AND +1h (same as +1h but explicit)

for sym in coins:
    if sym not in configs: continue
    cfg = configs[sym]
    d=load(sym)
    if d is None or len(d['c'])<2000: continue
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; n=len(c); idx=d['ts']
    
    # Crash filter
    skip=False
    for i in range(1,n):
        if abs(c[i]/c[i-1]-1)*100>40: skip=True; break
    if skip: continue
    
    LB=cfg['LB']; sp=cfg['ssl']; tp=cfg['tp']; sl=cfg['sl']
    
    # Trends
    try:
        df=pd.DataFrame({'c':c},index=idx)
        c4h=df['c'].resample('4h').last().dropna().values
        e50_4h=ema(c4h,50); e200_4h=ema(c4h,200)
        e50_a=np.zeros(n); e200_a=np.zeros(n)
        for i in range(n):
            j=i//16
            if j<len(e50_4h): e50_a[i]=e50_4h[j]; e200_a[i]=e200_4h[j]
        t4=e50_a>e200_a
        
        c1h=df['c'].resample('1h').last().dropna().values
        e20_1h=ema(c1h,20); e50_1h=ema(c1h,50)
        e20_a=np.zeros(n); e50_a2=np.zeros(n)
        for i in range(n):
            j=i//4
            if j<len(e20_1h): e20_a[i]=e20_1h[j]; e50_a2[i]=e50_1h[j]
        t1=e20_a>e50_a2
    except:
        t1=np.ones(n,bool); t4=np.ones(n,bool)
    
    green_4h=t4.sum()/n*100
    green_1h=t1.sum()/n*100
    
    # Whale
    sm=3
    ln=pd.Series(l_).shift(1).rolling(LB).min().values
    lc=np.zeros(n)
    for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
    sc=pd.Series(lc).ewm(span=sm,adjust=False).mean().values
    hc=pd.Series(sc).rolling(LB).max().values
    sr=np.where(l_<=ln,(sc+hc*2)/3,0)
    wp=pd.Series(sr).ewm(span=sm,adjust=False).mean().values
    wp_up=wp>np.roll(wp,1)
    
    # SSL
    sma_h=pd.Series(h).rolling(sp).mean().values
    ssl_c=np.zeros(n,int)
    for i in range(sp,n):
        if h[i-1]>sma_h[i-1]: ssl_c[i]=1
        else: ssl_c[i]=-1
    
    # Entry arrays
    le_4h=np.zeros(n,bool)
    le_1h=np.zeros(n,bool)
    for i in range(200,n):
        base = ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0
        if base and t4[i]: le_4h[i]=True
        if base and t4[i] and t1[i]: le_1h[i]=True
    
    # Backtest both
    for ename, le_arr in [('4h',le_4h),('1h',le_1h)]:
        if le_arr.sum()<3: continue
        t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; crash=0
        for i in range(200,n):
            if pos:
                if h[i]>=ep*(1+tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif l_[i]<=ep*(1-sl/100):
                    raw=(c[i]/ep-1)*100-COMM*100
                    pnl=max(raw,-sl*MAX_SLIPPAGE-COMM*100)
                    if raw<pnl: crash+=1
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            if not pos and cool==0 and le_arr[i]: pos=1; ep=c[i]
            if not pos and cool>0: cool-=1
            cv.append(eq)
        if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
        if len(t)<3: continue
        w=[p for p in t if p>0]; lo=[p for p in t if p<=0]
        wr=len(w)/len(t)*100
        dd_val=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
        r={'sym':sym,'entry':ename,'LB':LB,'ssl':sp,'tp':tp,'sl':sl,'t':len(t),
           'wr':wr,'dd':dd_val,'eq':eq,'crash':crash,'green':green_4h if ename=='4h' else green_1h,'sigs':le_arr.sum()}
        if ename=='4h': results_4h.append(r)
        else: results_1h.append(r)

# Filter profitable
def filter_coins(data, min_wr=40, max_dd=20, min_t=5, max_crash=3):
    return [d for d in data if d['eq']>CAP and d['wr']>=min_wr and abs(d['dd'])<=max_dd and d['t']>=min_t and d['crash']<=max_crash]

p4 = filter_coins(results_4h)
p1 = filter_coins(results_1h)

def summarize(name, passed):
    total=sum(d['eq']-CAP for d in passed)
    trades=sum(d['t'] for d in passed)
    wr=np.mean([d['wr'] for d in passed])
    dd=np.mean([d['dd'] for d in passed])
    return f'{name:<25} {len(passed):>5} {trades:>6} {wr:>5.1f}% {dd:>5.1f}% ${total:>+8.0f}'

print(f'{"Strategy":<25} {"عملات":>5} {"صفقات":>6} {"WR":>5} {"DD":>5} {"ربح$":>9}')
print('─'*60)
print(summarize('أصلي (نفس الكونفق، 4h)', p4))
print(summarize('+1h (نفس الكونفق)', p1))

# Coins that are in BOTH
both_syms = set(d['sym'] for d in p4) & set(d['sym'] for d in p1)
only_4h = set(d['sym'] for d in p4) - set(d['sym'] for d in p1)
only_1h = set(d['sym'] for d in p1) - set(d['sym'] for d in p4)
print(f'\nمشتركين: {len(both_syms)} | 4h فقط: {len(only_4h)} | 1h فقط: {len(only_1h)}')

# Detailed comparison for shared coins
shared_4h = [d for d in p4 if d['sym'] in both_syms]
shared_1h = [d for d in p1 if d['sym'] in both_syms]
s4_total = sum(d['eq']-CAP for d in shared_4h)
s1_total = sum(d['eq']-CAP for d in shared_1h)
s4_wr = np.mean([d['wr'] for d in shared_4h])
s1_wr = np.mean([d['wr'] for d in shared_1h])
s4_dd = np.mean([d['dd'] for d in shared_4h])
s1_dd = np.mean([d['dd'] for d in shared_1h])
s4_t = sum(d['t'] for d in shared_4h)
s1_t = sum(d['t'] for d in shared_1h)
print(f'\nالعملات المشتركة ({len(both_syms)}):')
print(f'  4h فقط: {s4_t} صفقة WR={s4_wr:.1f}% DD={s4_dd:.1f}% +${s4_total:.0f}')
print(f'  +1h:    {s1_t} صفقة WR={s1_wr:.1f}% DD={s1_dd:.1f}% +${s1_total:.0f}')
