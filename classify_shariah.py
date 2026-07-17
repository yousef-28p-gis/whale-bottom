#!/usr/bin/env python3
"""
تصنيف جميع عملات بايننس (USDT) حسب الشريعة الإسلامية
المجموعات: حلال / مشبوهة / محرمة

قواعد يوسف:
- محرم = ربا + سياسة + عملات مستقرة فقط
- مشبوه = ميم + خصوصية + DEX + ألعاب + fan tokens + باقي الرمادي
- حلال = الباقي (L1, L2, Infra, AI, Payments...)

المصادر: SharifBot, HalalSignalz, CryptoHalal, AAOIFI Standard 17
"""

import json

# ═══════════════════════════════════════════════════════
# 1. تحميل قائمة بايننس
# ═══════════════════════════════════════════════════════
raw_pairs = """0G/USDT
1000CAT/USDT
1000CHEEMS/USDT
1000SATS/USDT
1INCH/USDT
1MBABYDOGE/USDT
2Z/USDT
A/USDT
AAOIB/USDT
AAVE/USDT
ACE/USDT
ACH/USDT
ACM/USDT
ACT/USDT
ACX/USDT
ADA/USDT
ADX/USDT
AEUR/USDT
AEVO/USDT
AGLD/USDT
AI/USDT
AIGENSYN/USDT
AIXBT/USDT
ALGO/USDT
ALICE/USDT
ALLO/USDT
ALPINE/USDT
ALT/USDT
AMDB/USDT
AMP/USDT
ANIME/USDT
ANKR/USDT
APE/USDT
API3/USDT
APT/USDT
AR/USDT
ARB/USDT
ARK/USDT
ARKM/USDT
ARMB/USDT
ARPA/USDT
ASR/USDT
ASTER/USDT
ASTR/USDT
AT/USDT
ATM/USDT
ATOM/USDT
AUCTION/USDT
AUDIO/USDT
AVA/USDT
AVAX/USDT
AVGOB/USDT
AVNT/USDT
AWE/USDT
AXL/USDT
AXS/USDT
BABAB/USDT
BABY/USDT
BANANA/USDT
BANANAS31/USDT
BAND/USDT
BANK/USDT
BAR/USDT
BARD/USDT
BAT/USDT
BB/USDT
BCH/USDT
BEAMX/USDT
BEL/USDT
BERA/USDT
BFUSD/USDT
BICO/USDT
BIGTIME/USDT
BIO/USDT
BLUR/USDT
BMT/USDT
BNB/USDT
BNSOL/USDT
BNT/USDT
BOME/USDT
BONK/USDT
BREV/USDT
BROCCOLI714/USDT
BTC/USDT
BTTC/USDT
C/USDT
C98/USDT
CAKE/USDT
CATI/USDT
CBRSB/USDT
CELO/USDT
CELR/USDT
CETUS/USDT
CFG/USDT
CFX/USDT
CGPT/USDT
CHIP/USDT
CHR/USDT
CHZ/USDT
CITY/USDT
CKB/USDT
COINB/USDT
COMP/USDT
COOKIE/USDT
COTI/USDT
COW/USDT
CRCLB/USDT
CRV/USDT
CTK/USDT
CTSI/USDT
CVC/USDT
CVX/USDT
CYBER/USDT
DASH/USDT
DCR/USDT
DEXE/USDT
DGB/USDT
DIA/USDT
DODO/USDT
DOGE/USDT
DOGS/USDT
DOLO/USDT
DOT/USDT
DRAMB/USDT
DUSK/USDT
DYDX/USDT
DYM/USDT
EDEN/USDT
EDU/USDT
EGLD/USDT
EIGEN/USDT
ENA/USDT
ENJ/USDT
ENS/USDT
ENSO/USDT
EPIC/USDT
ERA/USDT
ESP/USDT
ETC/USDT
ETH/USDT
ETHFI/USDT
EUL/USDT
EUR/USDT
EURI/USDT
EWYB/USDT
F/USDT
FDUSD/USDT
FET/USDT
FF/USDT
FIDA/USDT
FIL/USDT
FLOKI/USDT
FLOW/USDT
FLUX/USDT
FOGO/USDT
FORM/USDT
FRAX/USDT
FTT/USDT
G/USDT
GALA/USDT
GAS/USDT
GENIUS/USDT
GIGGLE/USDT
GLM/USDT
GLMR/USDT
GLWB/USDT
GMT/USDT
GMX/USDT
GNO/USDT
GNS/USDT
GOOGLB/USDT
GPS/USDT
GRAM/USDT
GRT/USDT
GTC/USDT
GUN/USDT
HAEDAL/USDT
HBAR/USDT
HEI/USDT
HEMI/USDT
HFT/USDT
HIVE/USDT
HMSTR/USDT
HOLO/USDT
HOME/USDT
HOODB/USDT
HOT/USDT
HUMA/USDT
HYPER/USDT
IBMB/USDT
ICP/USDT
ICX/USDT
ID/USDT
ILV/USDT
IMX/USDT
INIT/USDT
INJ/USDT
INTCB/USDT
IO/USDT
IOST/USDT
IOTA/USDT
IOTX/USDT
IQ/USDT
JASMY/USDT
JOE/USDT
JST/USDT
JTO/USDT
JUP/USDT
JUV/USDT
KAIA/USDT
KAITO/USDT
KAT/USDT
KAVA/USDT
KERNEL/USDT
KGST/USDT
KITE/USDT
KMNO/USDT
KNC/USDT
KSM/USDT
LA/USDT
LAYER/USDT
LAZIO/USDT
LDO/USDT
LINEA/USDT
LINK/USDT
LISTA/USDT
LITEB/USDT
LPT/USDT
LQTY/USDT
LSK/USDT
LTC/USDT
LUMIA/USDT
LUNA/USDT
LUNC/USDT
MAGIC/USDT
MANA/USDT
MANTA/USDT
MANTRA/USDT
MASK/USDT
MAV/USDT
MBL/USDT
ME/USDT
MEGA/USDT
MEME/USDT
MET/USDT
METAB/USDT
METIS/USDT
MINA/USDT
MIRA/USDT
MITO/USDT
MMT/USDT
MORPHO/USDT
MOVE/USDT
MOVR/USDT
MRVLB/USDT
MSFTB/USDT
MSTRB/USDT
MTL/USDT
MUB/USDT
MUBARAK/USDT
NBISB/USDT
NEAR/USDT
NEIRO/USDT
NEO/USDT
NEWT/USDT
NEXO/USDT
NIGHT/USDT
NIL/USDT
NMR/USDT
NOKB/USDT
NOM/USDT
NOT/USDT
NVDAB/USDT
NXPC/USDT
OG/USDT
OGN/USDT
ONDO/USDT
ONE/USDT
ONG/USDT
ONT/USDT
OP/USDT
OPEN/USDT
OPG/USDT
OPN/USDT
ORCA/USDT
ORDI/USDT
OSMO/USDT
PARTI/USDT
PAXG/USDT
PENDLE/USDT
PENGU/USDT
PEOPLE/USDT
PEPE/USDT
PHA/USDT
PIVX/USDT
PIXEL/USDT
PLTRB/USDT
PLUME/USDT
PNUT/USDT
POL/USDT
POLYX/USDT
PORTAL/USDT
PORTO/USDT
POWR/USDT
PROM/USDT
PROVE/USDT
PSG/USDT
PUMP/USDT
PUNDIX/USDT
PYR/USDT
PYTH/USDT
QCOMB/USDT
QI/USDT
QKC/USDT
QNT/USDT
QQQB/USDT
QTUM/USDT
QUICK/USDT
RAD/USDT
RARE/USDT
RAY/USDT
RE/USDT
RED/USDT
RENDER/USDT
REQ/USDT
RESOLV/USDT
REZ/USDT
RIF/USDT
RKLBB/USDT
RLC/USDT
RLUSD/USDT
ROBO/USDT
RONIN/USDT
ROSE/USDT
RPL/USDT
RSR/USDT
RUNE/USDT
RVN/USDT
S/USDT
SAGA/USDT
SAHARA/USDT
SAND/USDT
SANTOS/USDT
SAPIEN/USDT
SC/USDT
SCR/USDT
SCRT/USDT
SEI/USDT
SENT/USDT
SFP/USDT
SHELL/USDT
SHIB/USDT
SIGN/USDT
SKHYB/USDT
SKL/USDT
SKY/USDT
SLP/USDT
SNDKB/USDT
SNX/USDT
SOL/USDT
SOLV/USDT
SOMI/USDT
SOPH/USDT
SOXLB/USDT
SPCXB/USDT
SPELL/USDT
SPK/USDT
SPYB/USDT
SSV/USDT
STEEM/USDT
STG/USDT
STO/USDT
STORJ/USDT
STRAX/USDT
STRK/USDT
STX/USDT
SUI/USDT
SUN/USDT
SUPER/USDT
SUSHI/USDT
SXT/USDT
SYN/USDT
SYRUP/USDT
T/USDT
TAO/USDT
TFUEL/USDT
THE/USDT
THETA/USDT
TIA/USDT
TKO/USDT
TLM/USDT
TNSR/USDT
TOWNS/USDT
TRB/USDT
TREE/USDT
TRUMP/USDT
TRX/USDT
TSLAB/USDT
TSMB/USDT
TST/USDT
TURBO/USDT
TURTLE/USDT
TUSD/USDT
TUT/USDT
TWT/USDT
U/USDT
UMA/USDT
UNI/USDT
USD1/USDT
USDC/USDT
USDE/USDT
USDP/USDT
USDS/USDT
USTC/USDT
USUAL/USDT
VANA/USDT
VANRY/USDT
VELODROME/USDT
VET/USDT
VIC/USDT
VIRTUAL/USDT
VTHO/USDT
W/USDT
WAL/USDT
WAXP/USDT
WBETH/USDT
WBTC/USDT
WCT/USDT
WDCB/USDT
WIF/USDT
WIN/USDT
WLD/USDT
WLFI/USDT
WOO/USDT
XAI/USDT
XAUT/USDT
XEC/USDT
XLM/USDT
XNO/USDT
XPL/USDT
XRP/USDT
XTZ/USDT
XUSD/USDT
XVG/USDT
XVS/USDT
YB/USDT
YFI/USDT
YGG/USDT
ZAMA/USDT
ZBT/USDT
ZEC/USDT
ZEN/USDT
ZIL/USDT
ZK/USDT
ZKC/USDT
ZKP/USDT
ZRO/USDT
ZRX/USDT
币安人生/USDT"""

coins = sorted(set(p.split('/')[0] for p in raw_pairs.strip().split('\n')))

# ═══════════════════════════════════════════════════════
# 2. قواعد التصنيف حسب طلب يوسف
# ═══════════════════════════════════════════════════════

# ⛔ محرم = ربا + سياسة + عملات مستقرة فقط
HARAM = {
    # بروتوكولات الإقراض بالربا
    'AAVE', 'COMP', 'MORPHO', 'EUL', 'LISTA', 'FF',
    
    # عملات سياسية/شخصية
    'TRUMP', 'WLFI',
    
    # عملات مستقرة — أدوات مالية وليست استثمارية
    'USDT', 'USDC', 'FDUSD', 'TUSD', 'USDP', 'USDS', 'USD1', 'USDE',
    'AEUR', 'EUR', 'EURI', 'BFUSD', 'FRAX', 'RLUSD', 'USTC', 'XUSD', 'USUAL',
}

# ⚠️ مشبوه = ميم + خصوصية + DEX + ألعاب + fan tokens + باقي الرمادي
SUSPICIOUS = {
    # عملات الميم — مضاربة بحتة = ميسر
    'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'WIF', 'BONK', 'BOME', 'MEME',
    'TURBO', 'NEIRO', 'PNUT', 'PENGU', 'BABYDOGE', '1MBABYDOGE', 'DOGS',
    'TST', 'BANANAS31', 'BANANA', 'BROCCOLI714', 'TURTLE',
    '1000CHEEMS', '1000CAT', '1000SATS', 'MUBARAK', 'MUB',
    'KAT', 'CHIP', 'FOGO', 'TUT', 'MMT', 'GIGGLE', 'BABY',
    'DOLO', 'PUMP', 'SLP', 'WIN', 'XPL',
    
    # عملات خصوصية
    'XMR', 'ZEC', 'DASH', 'PIVX', 'SCRT', 'XVG',
    
    # منصات تداول لامركزية — تخدم حلال وحرام
    'UNI', 'SUSHI', 'CAKE', 'DYDX', 'JOE', 'QUICK', 'RAY',
    'ORCA', 'VELODROME', 'JUP', '1INCH', 'COW', 'DODO',
    
    # ألعاب وميتافيرس — خلاف على القمار
    'SAND', 'MANA', 'GALA', 'AXS', 'ENJ', 'ALICE',
    'ILV', 'TLM', 'MAGIC', 'YGG', 'PIXEL', 'BIGTIME',
    'HMSTR', 'NOT', 'CATI', 'AGLD', 'ANIME',
    
    # عملات مشجعين (fan tokens) — مرتبطة بمراهنات
    'PSG', 'BAR', 'CITY', 'JUV', 'ASR', 'ATM', 'ACM',
    'LAZIO', 'PORTO', 'SANTOS', 'ALPINE', 'OG',
    
    # DeFi/DEX مشبوهة
    'AEVO', 'CRV', 'GMX', 'GNS', 'INJ', 'JTO', 'KMNO',
    'ME', 'PENDLE', 'SNX', 'BEL', 'EIGEN', 'ENA', 'KERNEL',
    'LDO', 'LQTY', 'RPL', 'SSV', 'STG', 'SUN',
    
    # Staking tokens (ربا محتمل)
    'BNSOL', 'WBETH', 'SOLV',
    
    # BNB ومنصات مركزية
    'BNB', 'CRO', 'FTT', 'NEXO',
    
    # WBTC (إيصال إيداع)
    'WBTC',
    
    # عملات SharifBot Grey Area
    'ACE', 'AIXBT', 'APE', 'ARPA', 'ASTER', 'AUDIO',
    'BAND', 'BEAMX', 'BLUR', 'BNT', 'CETUS', 'CFX', 'CGPT',
    'CHZ', 'COTI', 'CYBER', 'DCR', 'DUSK',
    'ENS', 'GNO', 'GMT', 'HFT', 'ICP', 'METIS',
    'MTL', 'POL', 'POLYX', 'PORTAL', 'QNT', 'RLC', 'RONIN',
    'ROSE', 'RUNE', 'SEI', 'SKL', 'SYN',
    'TNSR', 'VANRY', 'WAXP', 'XAI', 'XAUT',
    'ZAMA', 'ZEN', 'ZK', 'ZKP', 'ZRX',
    'MET', 'RIF', 'GLMR', 'MOVR', 'CFG',
    
    # عملات جديدة/غير محكمة
    'ALT', 'AXL', 'BB', 'BICO', 'C98', 'CELR', 
    'CTK', 'CTSI', 'CVX', 'DEXE', 'EGLD',
    'ETHFI', 'GTC', 'HEI', 'ID', 'JST', 'KNC', 'KSM',
    'MASK', 'MAV', 'MINA',
    'NEWT', 'OGN', 'ONDO', 'ONE', 'ONT', 'OP', 'ORDI',
    'OSMO', 'PAXG', 'PEOPLE', 'PYR', 'QI',
    'RAD', 'RARE', 'REZ', 'RSR', 'RVN', 'SFP',
    'SPELL', 'SUPER', 'T', 'TWT',
    'UMA', 'VANA', 'VIC', 'VIRTUAL', 'WOO', 'XVS', 'YFI',
    'CELO', 'FLOW', 'ICP', 'KAVA',
    
    # عملات جديدة بالكامل
    'ACX', 'BMT', 'EDEN', 'EPIC', 'ERA', 'FORM',
    'GPS', 'GRAM', 'HAEDAL', 'HOME', 'HUMA', 'IQ',
    'LAYER', 'LUMIA', 'MANTA', 'MANTRA', 'MITO',
    'PLUME', 'RESOLV', 'SCR', 'SHELL', 'SKY', 'SPK', 'STO',
    'SYRUP', 'THE', 'U', 'YB', 'ZKC',
    'AT', 'GUN', 'HEMI', 'KAIA', 'KAITO', 'NIGHT',
    'NIL', 'NXPC', 'OPG', 'PROVE', 'SOPH',
    'TREE', 'W', 'ZBT',
    
    # باقي غير المصنفة سابقاً
    'A', 'AVNT', 'AWE', 'BANK', 'BARD',
    'C', 'F', 'G', 'GENIUS', 'KGST',
    'NOM', 'OPN', 'RE', 'S', 'WCT',
    'LUNA', 'LUNC',  # عملات منهارة
}

# ✅ حلال — L1, L2, Infra, AI, Payments, RWA
HALAL = {
    # Layer 1
    'BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX', 'NEAR', 'SUI', 'APT',
    'ATOM', 'TRX', 'ALGO', 'XTZ', 'HBAR', 'XRP', 'XLM', 'LTC', 'BCH',
    'ETC', 'ZIL', 'IOTA', 'IOTX', 'NEO', 'QTUM', 'ICX',
    'EOS', 'LSK', 'STEEM', 'XEC', 'DGB',
    '2Z', 'ARK', 'ASTR', 'BREV', 'DYM', 'MOVE',
    'ROBO', 'SAGA', 'SOMI', 'STRAX', 'TIA', 'BERA', 'KITE',
    
    # Layer 2 & Scaling
    'ARB', 'STRK', 'STX', 'MATIC',
    'IMX', 'LINEA', 'MEGA', 'TOWNS',
    
    # Infrastructure
    'LINK', 'GRT', 'FIL', 'AR', 'THETA', 'RENDER', 'TAO', 'FET',
    'VET', 'STORJ', 'SC', 'HOT', 'ANKR', 'CKB', 'JASMY', 'API3',
    'PYTH', 'GLM', 'FLUX', 'POWR', 'IOST', 'ONG', 'GAS', 'TFUEL',
    'HIVE', 'HOLO', 'BAT', 'CVC',
    'ADX', 'LPT', 'MBL', 'PHA', 'RED', 'SIGN',
    'PARTI', 'QKC', 'TRB', 'VTHO', 'XNO', 'ZRO',
    'HYPER', 'LA', 'TKO', 'WAL',
    'ENSO', 'ESP', 'OPEN',
    
    # AI
    'AI', 'AIGENSYN', 'ALLO', 'BIO', 'COOKIE',
    'INIT', 'MIRA', 'NMR', 'SAPIEN', 'SENT', 'SXT', 'WLD',
    'ACT', '0G', 'SAHARA', 'IO',
    
    # Gaming حلال
    'CHR', 'PROM', 'AUCTION',
    
    # Payments/Utility
    'ACH', 'AMP', 'REQ', 'PUNDIX', 'AVA', 'FIDA',
    
    # RWA/Data
    'BTTC', 'DIA', 'ARKM', 'EDU',
    
    # ذهب
    'PAXG', 'XAUT',
    
    # Oracles/Data
    'BAND',
    
    # QNT (Overledger — interop)
    'QNT',
}

# ═══════════════════════════════════════════════════════
# 3. تصحيح التعارضات (priorities: HARAM > SUSPICIOUS > HALAL)
# ═══════════════════════════════════════════════════════
for c in HARAM:
    SUSPICIOUS.discard(c)
    HALAL.discard(c)

for c in SUSPICIOUS:
    HALAL.discard(c)

# ═══════════════════════════════════════════════════════
# 4. تصنيف
# ═══════════════════════════════════════════════════════
halal_coins = []
suspicious_coins = []
haram_coins = []
stock_tokens = []
unknown_coins = []

# رموز الأسهم — تنتهي بـ B بعد اسم الشركة
STOCK_SUFFIXES = {
    'AMDB', 'ARMB', 'AVGOB', 'BABAB', 'CBRSB', 'COINB', 'CRCLB',
    'DRAMB', 'EWYB', 'GLWB', 'GOOGLB', 'HOODB', 'IBMB', 'INTCB',
    'LITEB', 'METAB', 'MRVLB', 'MSFTB', 'MSTRB', 'NBISB', 'NOKB',
    'NVDAB', 'PLTRB', 'QCOMB', 'QQQB', 'RKLBB', 'SKHYB', 'SNDKB',
    'SOXLB', 'SPCXB', 'SPYB', 'TSLAB', 'TSMB', 'WDCB', 'AAOIB',
}

for coin in coins:
    # تجاهل الرموز الصينية
    if any(ord(c) > 127 for c in coin):
        continue
    
    # رموز الأسهم
    if coin in STOCK_SUFFIXES:
        stock_tokens.append(coin)
        continue
    
    if coin in HARAM:
        haram_coins.append(coin)
    elif coin in SUSPICIOUS:
        suspicious_coins.append(coin)
    elif coin in HALAL:
        halal_coins.append(coin)
    else:
        unknown_coins.append(coin)

# ═══════════════════════════════════════════════════════
# 5. تقرير
# ═══════════════════════════════════════════════════════
print("=" * 70)
print("📊 تصنيف عملات بايننس (USDT) حسب الشريعة الإسلامية")
print("=" * 70)
print()
print(f"✅ حلال:        {len(halal_coins)} عملة")
print(f"⚠️  مشبوهة:     {len(suspicious_coins)} عملة")
print(f"⛔ محرمة:       {len(haram_coins)} عملة")
print(f"📦 أسهم/خاصة:   {len(stock_tokens)} رمز")
print(f"📊 المجموع:     {len(coins)}")

print()
print("=" * 70)
print("⛔ المحرمات (ربا + سياسة + عملات مستقرة)")
print("=" * 70)
for c in sorted(haram_coins):
    reason = ""
    if c in ('AAVE','COMP','MORPHO','EUL','LISTA','FF'):
        reason = " — ربا"
    elif c in ('TRUMP','WLFI'):
        reason = " — سياسة"
    else:
        reason = " — عملة مستقرة"
    print(f"  {c}{reason}")

print()
print("=" * 70)
print("⚠️  المشبوهات (ميم + خصوصية + DEX + ألعاب + Fan Tokens + رمادي)")
print("=" * 70)
for c in sorted(suspicious_coins):
    print(f"  {c}")

print()
print("=" * 70)
print("✅ الحلال")
print("=" * 70)
for c in sorted(halal_coins):
    print(f"  {c}")

if unknown_coins:
    print()
    print("=" * 70)
    print("❓ غير مصنفة")
    print("=" * 70)
    for c in sorted(unknown_coins):
        print(f"  {c}")

# ═══════════════════════════════════════════════════════
# 6. حفظ ملف JSON
# ═══════════════════════════════════════════════════════
output = {
    "halal": sorted(halal_coins),
    "suspicious": sorted(suspicious_coins),
    "haram": sorted(haram_coins),
    "stock_tokens": sorted(stock_tokens),
    "unknown": sorted(unknown_coins),
    "counts": {
        "halal": len(halal_coins),
        "suspicious": len(suspicious_coins),
        "haram": len(haram_coins),
        "stock_tokens": len(stock_tokens),
        "unknown": len(unknown_coins),
        "total": len(coins)
    },
    "rules": {
        "haram": ["بروتوكولات الإقراض بالربا (AAVE, COMP, MORPHO...)", "عملات سياسية (TRUMP, WLFI)", "عملات مستقرة (USDT, USDC...) — ليست للاستثمار"],
        "suspicious": ["عملات الميم — مضاربة بحتة = ميسر", "عملات خصوصية (ZEC, DASH, PIVX...)", "منصات DEX — تخدم حلال وحرام", "ألعاب وميتافيرس — خلاف على القمار", "عملات مشجعين — مرتبطة بمراهنات", "عملات غير محكمة شرعياً"],
        "halal": ["Layer 1 أصلية", "Layer 2 وبنية تحتية", "ذكاء اصطناعي وبيانات", "مدفوعات وتحويلات"]
    },
    "sources": [
        "SharifBot — AAOIFI Shariah Standard 17",
        "HalalSignalz — Mufti Faraz Adam's Crypto Shariah Framework",
        "CryptoHalal.cc"
    ],
    "_note": "هذا التصنيف اجتهادي. استشر عالم شرعي متخصص للفتوى النهائية."
}

with open('/data/trading28/config/shariah_coins.json', 'w') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n💾 تم الحفظ: /data/trading28/config/shariah_coins.json")
print("\n⚠️  هذا التصنيف اجتهادي — استشر عالم شرعي للفتوى النهائية.")
