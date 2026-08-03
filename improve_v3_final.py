"""Option 3 Improvements — test on ALL 198 coins, replicate original exactly + add enhancements"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; DATA='/data/trading28/data/whale_15m_1y'
MAX_SLIPPAGE=1.5; COOLDOWN=48

def load(sym):
    with open(os.path.join(DATA, f'{sym}.json')) as f: d=json.load(f)
    return {'c':np.array(d['c'],float),'h':np.array(d['h'],float),
            'l':np.array(d['l'],float),'o':np.array(d['o'],float),
            'ts':pd.to_datetime(d['ts'],unit='ms')}

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def sim(le, c, h, l_, n, tp, sl):
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
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    return t,cv,eq,crash

def make_sl_handling(tp,sl):
    """Standard SL handling function"""
    return lambda c,ep: max((c/ep-1)*100-COMM*100, -sl*MAX_SLIPPAGE-COMM*100)

def test_coin_base(sym):
    """Exact replica of original test_coin"""
    d=load(sym)
    if d is None or len(d['c'])<2000: return None
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; n=len(c); idx=d['ts']
    
    for i in range(1,n):
        if abs(c[i]/c[i-1]-1)*100>40: return None
    
    # 4h trend
    try:
        df=pd.DataFrame({'c':c},index=idx)
        c4h=df['c'].resample('4h').last().dropna().values
        e50_4h=ema(c4h,50); e200_4h=ema(c4h,200)
        e50_a=np.zeros(n); e200_a=np.zeros(n)
        for i in range(n):
            j=i//16
            if j<len(e50_4h): e50_a[i]=e50_4h[j]; e200_a[i]=e200_4h[j]
        trend_4h=e50_a>e200_a
        green_pct=trend_4h.sum()/n*100
    except:
        trend_4h=np.ones(n,bool); green_pct=100
    
    # 1h trend
    try:
        c1h=df['c'].resample('1h').last().dropna().values
        e20_1h=ema(c1h,20); e50_1h=ema(c1h,50)
        e20_1h_a=np.zeros(n); e50_1h_a=np.zeros(n)
        for i in range(n):
            j=i//4
            if j<len(e20_1h): e20_1h_a[i]=e20_1h[j]; e50_1h_a[i]=e50_1h[j]
        trend_1h=e20_1h_a>e50_1h_a
    except:
        trend_1h=np.ones(n,bool)
    
    best={'eq':0,'score':-999999}
    configs=[(30,3),(50,3),(70,3)]
    ssls=[5,10,20]
    tps=[(2,1),(3,1.5),(5,2.5)]
    
    for LB,sm in configs:
        ln=pd.Series(l_).shift(1).rolling(LB).min().values
        lc=np.zeros(n)
        for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
        sc=pd.Series(lc).ewm(span=sm,adjust=False).mean().values
        hc=pd.Series(sc).rolling(LB).max().values
        sr=np.where(l_<=ln,(sc+hc*2)/3,0)
        wp=pd.Series(sr).ewm(span=sm,adjust=False).mean().values
        wp_up=wp>np.roll(wp,1)
        
        for sp in ssls:
            sma_h=pd.Series(h).rolling(sp).mean().values
            sma_l=pd.Series(l_).rolling(sp).mean().values
            ssl_c=np.zeros(n,int)
            for i in range(sp,n):
                if h[i-1]>sma_h[i-1]: ssl_c[i]=1
                else: ssl_c[i]=-1
            
            # --- ORIGINAL entry ---
            le_orig=np.zeros(n,bool)
            for i in range(200,n):
                if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0 and trend_4h[i]:
                    le_orig[i]=True
            
            # --- ENHANCED: +1h trend ---
            le_1h=np.zeros(n,bool)
            for i in range(200,n):
                if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0 and trend_4h[i] and trend_1h[i]:
                    le_1h[i]=True
            
            # --- ENHANCED: +1h trend + stronger whale ---
            le_strong=np.zeros(n,bool)
            for i in range(200,n):
                if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*3 and wp[i]>0 and trend_4h[i] and trend_1h[i]:
                    le_strong[i]=True
            
            # --- ENHANCED: +1h + SSL just crossed ---
            le_cross=np.zeros(n,bool)
            for i in range(200,n):
                ssl_just_up = ssl_c[i]==1 and ssl_c[i-1]==-1
                if ssl_just_up and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0 and trend_4h[i] and trend_1h[i]:
                    le_cross[i]=True
            
            for tp,sl in tps:
                # ORIGINAL
                if le_orig.sum()>=5:
                    tr,cv,eq,cr=sim(le_orig,c,h,l_,n,tp,sl)
                    if len(tr)>=5:
                        w=[p for p in tr if p>0]; lo=[p for p in tr if p<=0]
                        wr=len(w)/len(tr)*100
                        dd_val=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
                        score=(eq-CAP)-abs(dd_val)*15-cr*100+wr*1.5
                        if score>best['score']:
                            best={'entry':'ORIG','LB':LB,'ssl':sp,'tp':tp,'sl':sl,
                                  't':len(tr),'wr':wr,'dd':dd_val,'eq':eq,'crash':cr,
                                  'score':score,'green':green_pct,'sigs':le_orig.sum()}
                
                # ENHANCEMENTS (store separately, compare later)
                for ename, le_arr in [('+1h', le_1h), ('+1h+Strong', le_strong), ('+1h+Cross', le_cross)]:
                    if le_arr.sum() < 5: continue
                    tr,cv,eq,cr=sim(le_arr,c,h,l_,n,tp,sl)
                    if len(tr)>=5:
                        w=[p for p in tr if p>0]; lo=[p for p in tr if p<=0]
                        wr=len(w)/len(tr)*100
                        dd_val=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
                        score=(eq-CAP)-abs(dd_val)*15-cr*100+wr*1.5
                        key = f'{ename}_LB{LB}_SSL{sp}_TP{tp}SL{sl}'
                        if key not in all_enhanced or score > all_enhanced[key]['score']:
                            all_enhanced[key] = {'sym':sym,'entry':ename,'LB':LB,'ssl':sp,'tp':tp,'sl':sl,
                                't':len(tr),'wr':wr,'dd':dd_val,'eq':eq,'crash':cr,'score':score,'green':green_pct}
    
    if best['eq']==0: return None
    return {'sym':sym,'best':best}

# ── Run ──
print('🔄 اختبار التحسينات على كل العملات...')
coins=sorted([f.replace('.json','') for f in os.listdir(DATA) if f.endswith('.json') and f!='_manifest.json'])

all_enhanced = {}  # global dict for enhanced results
all_data=[]
for i,sym in enumerate(coins):
    r=test_coin_base(sym)
    if r: all_data.append(r)
    if (i+1)%30==0: print(f'  {i+1}/{len(coins)}...')

# Original baseline
f={'min_wr':40,'max_dd':20,'min_trades':5,'max_crash':3}
orig_passed=[d for d in all_data if d['best']['eq']>CAP and d['best']['wr']>=f['min_wr']
    and abs(d['best']['dd'])<=f['max_dd'] and d['best']['t']>=f['min_trades']
    and d['best']['crash']<=f['max_crash']]

orig_total=sum(d['best']['eq']-CAP for d in orig_passed)
orig_trades=sum(d['best']['t'] for d in orig_passed)
orig_wr=np.mean([d['best']['wr'] for d in orig_passed])
orig_dd=np.mean([d['best']['dd'] for d in orig_passed])

# Enhanced: filter only profitable
enh_passed=[v for v in all_enhanced.values() if v['eq']>CAP and v['wr']>=40 and abs(v['dd'])<=20 and v['t']>=5 and v['crash']<=3]

print(f'\n{"="*70}')
print(f'📊 مقارنة: الأصلي vs التحسينات')
print(f'{"="*70}')
print(f'{"":20} {"عملات":>5} {"صفقات":>6} {"WR":>6} {"DD":>6} {"ربح$":>9}')
print(f'{"─"*55}')
print(f'{"أصلي (Option 3)":20} {len(orig_passed):>5} {orig_trades:>6} {orig_wr:>5.1f}% {orig_dd:>5.1f}% ${orig_total:>+8.0f}')

# Group enhanced by entry type
for ename in ['+1h','+1h+Strong','+1h+Cross']:
    ep=[v for v in enh_passed if v['entry']==ename]
    if not ep: continue
    et=sum(v['t'] for v in ep)
    eeq=sum(v['eq']-CAP for v in ep)
    ewr=np.mean([v['wr'] for v in ep])
    edd=np.mean([v['dd'] for v in ep])
    print(f'{ename:20} {len(ep):>5} {et:>6} {ewr:>5.1f}% {edd:>5.1f}% ${eeq:>+8.0f}')

# Detailed top coins for best enhancement
print(f'\n🏆 أفضل التحسينات:')
# Best by entry type
for ename in ['+1h','+1h+Strong','+1h+Cross']:
    ep=sorted([v for v in enh_passed if v['entry']==ename], key=lambda x:-x['eq'])[:5]
    if ep:
        print(f'\n{ename}:')
        for v in ep:
            print(f'  {v["sym"]:<10} LB{v["LB"]}/SSL{v["ssl"]} TP{v["tp"]}/SL{v["sl"]} T={v["t"]} WR={v["wr"]:.1f}% DD={v["dd"]:.1f}% +${v["eq"]-CAP:.0f}')
