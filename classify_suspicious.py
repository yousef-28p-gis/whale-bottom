#!/usr/bin/env python3
"""تصنيف العملات المشبوهة إلى فئات"""
import json

with open('/data/trading28/config/shariah_coins.json') as f:
    data = json.load(f)

suspicious = data['suspicious']

categories = {
    '🎭 عملات ميم (مضاربة بحتة)': {
        'coins': ['DOGE','SHIB','PEPE','FLOKI','WIF','BONK','BOME','MEME','TURBO','NEIRO','PNUT','PENGU','1MBABYDOGE','DOGS','TST','BANANAS31','BANANA','BROCCOLI714','TURTLE','1000CHEEMS','1000CAT','1000SATS','MUBARAK','MUB','KAT','CHIP','FOGO','TUT','MMT','GIGGLE','BABY','DOLO','PUMP','SLP','WIN','XPL'],
        'reason': 'لا قيمة حقيقية — مضاربة بحتة = ميسر'
    },
    '🔒 عملات خصوصية': {
        'coins': ['ZEC','DASH','PIVX','SCRT','XVG'],
        'reason': 'ميزات خصوصية تسهل أنشطة محرمة'
    },
    '🔄 منصات DEX': {
        'coins': ['UNI','SUSHI','CAKE','DYDX','JOE','QUICK','RAY','ORCA','VELODROME','JUP','1INCH','COW','DODO'],
        'reason': 'تسهل تداول عملات حلال وحرام معاً'
    },
    '🎮 ألعاب وميتافيرس': {
        'coins': ['SAND','MANA','GALA','AXS','ENJ','ALICE','ILV','TLM','MAGIC','YGG','PIXEL','BIGTIME','HMSTR','NOT','CATI','AGLD','ANIME'],
        'reason': 'قد تحتوي عناصر قمار أو محتوى غير لائق'
    },
    '⚽ عملات مشجعين': {
        'coins': ['PSG','BAR','CITY','JUV','ASR','ATM','ACM','LAZIO','PORTO','SANTOS','ALPINE','OG'],
        'reason': 'مرتبطة بمنصات مراهنات رياضية'
    },
    '🏦 DeFi متقدم': {
        'coins': ['AEVO','CRV','GMX','GNS','INJ','JTO','KMNO','ME','PENDLE','SNX','BEL','EIGEN','ENA','KERNEL','LDO','LQTY','RPL','SSV','STG','SUN'],
        'reason': 'إقراض ورهان ومشتقات — شبهة ربا'
    },
    '📦 Staking Tokens': {
        'coins': ['BNSOL','WBETH','SOLV'],
        'reason': 'إيصالات رهن — شبهة ربا'
    },
    '🏢 منصات مركزية': {
        'coins': ['BNB','FTT','NEXO','WBTC'],
        'reason': 'مرتبطة بمنصات تقدم خدمات ربوية'
    },
    '🟡 رمادي قديم': {
        'coins': ['ACE','AIXBT','APE','ARPA','ASTER','AUDIO','BAND','BEAMX','BLUR','BNT','CETUS','CFX','CGPT','CHZ','COTI','CYBER','DCR','DUSK','ENS','GNO','GMT','HFT','ICP','METIS','MTL','POL','POLYX','PORTAL','QNT','RLC','RONIN','ROSE','RUNE','SEI','SKL','SYN','TNSR','VANRY','WAXP','XAI','XAUT','ZAMA','ZEN','ZK','ZKP','ZRX','MET','RIF','GLMR','MOVR','CFG','ALT','AXL','BB','BICO','C98','CELR','CTK','CTSI','CVX','DEXE','EGLD','ETHFI','GTC','HEI','ID','JST','KNC','KSM','MASK','MAV','MINA','NEWT','OGN','ONDO','ONE','ONT','OP','ORDI','OSMO','PAXG','PEOPLE','PYR','QI','RAD','RARE','REZ','RSR','RVN','SFP','SPELL','SUPER','T','TWT','UMA','VANA','VIC','VIRTUAL','WOO','XVS','YFI','CELO','FLOW','KAVA'],
        'reason': 'خلاف بين العلماء — تحتاج فتوى فردية'
    },
    '🆕 جديدة غير محكمة': {
        'coins': ['ACX','BMT','EDEN','EPIC','ERA','FORM','GPS','GRAM','HAEDAL','HOME','HUMA','IQ','LAYER','LUMIA','MANTA','MANTRA','MITO','PLUME','RESOLV','SCR','SHELL','SKY','SPK','STO','SYRUP','THE','U','YB','ZKC','AT','GUN','HEMI','KAIA','KAITO','NIGHT','NIL','NXPC','OPG','PROVE','SOPH','TREE','W','ZBT','A','AVNT','AWE','BANK','BARD','C','F','G','GENIUS','KGST','NOM','OPN','RE','S','WCT'],
        'reason': 'عملات حديثة — لم تصدر فيها فتاوى بعد'
    },
    '💀 منهارة': {
        'coins': ['LUNA','LUNC'],
        'reason': 'مشاريع منهارة — خسارة شبه كاملة'
    },
}

print("=" * 70)
print("📊 تصنيف العملات المشبوهة (274 عملة)")
print("=" * 70)

total = 0
for cat_name, cat_data in categories.items():
    found = [c for c in cat_data['coins'] if c in suspicious]
    if not found:
        continue
    total += len(found)
    print(f"\n{cat_name} — {cat_data['reason']}")
    print(f"  🔢 {len(found)} عملة")
    # Print in rows of 8
    row = []
    for c in sorted(found):
        row.append(c)
        if len(row) == 8:
            print(f"  {', '.join(row)}")
            row = []
    if row:
        print(f"  {', '.join(row)}")

assigned = set()
for cat_data in categories.values():
    assigned.update(cat_data['coins'])
unassigned = [c for c in suspicious if c not in assigned]

print(f"\n{'='*70}")
print(f"✅ المجموع: {total} | ❓ غير مصنفة: {len(unassigned)}")
if unassigned:
    print(f"   {unassigned}")

# Save
output = {'categories': {}, 'total': total}
for cat_name, cat_data in categories.items():
    found = [c for c in cat_data['coins'] if c in suspicious]
    if found:
        output['categories'][cat_name] = {
            'count': len(found),
            'reason': cat_data['reason'],
            'coins': sorted(found)
        }

with open('/data/trading28/config/shariah_suspicious_categories.json', 'w') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n💾 تم الحفظ: /data/trading28/config/shariah_suspicious_categories.json")
