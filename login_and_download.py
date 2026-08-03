#!/usr/bin/env python3
"""Login + download PDFs from channel — all in one go"""
import asyncio, os, sys
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

API_ID = 2040
API_HASH = "b18441a1ff607e10a989891a5462e627"
CHANNEL = "TRADING_FREE_KINGFTL_27"
SESSION = "trading_session"

def load_phone():
    with open('/data/trading28/phone.hex') as f:
        hex_str = f.read().strip()
    return ''.join(chr(int(h, 16)) for h in hex_str.split())

async def main():
    client = TelegramClient(SESSION, API_ID, API_HASH)
    await client.connect()
    
    if await client.is_user_authorized():
        print("ALREADY_LOGGED_IN")
    else:
        phone = load_phone()
        sent = await client.send_code_request(phone, force_sms=True)
        print(f"CODE_SENT|{sent.phone_code_hash}")
        
        # Poll for code.txt
        code_file = '/data/trading28/code.txt'
        code = None
        for i in range(120):
            await asyncio.sleep(0.5)
            if os.path.exists(code_file):
                with open(code_file) as f:
                    val = f.read().strip()
                if val and len(val) >= 5:
                    os.remove(code_file)
                    code = val
                    break
        
        if not code:
            print("TIMEOUT_NO_CODE")
            return
        
        try:
            await client.sign_in(phone, code, phone_code_hash=sent.phone_code_hash)
        except SessionPasswordNeededError:
            print("2FA_NEEDED")
            for i in range(120):
                await asyncio.sleep(0.5)
                if os.path.exists(code_file):
                    with open(code_file) as f:
                        pwd = f.read().strip()
                    if pwd:
                        os.remove(code_file)
                        await client.sign_in(password=pwd)
                        break
            else:
                print("TIMEOUT_2FA")
                return
    
    me = await client.get_me()
    print(f"LOGGED_IN|{me.first_name}")
    
    # Resolve channel and download PDFs
    entity = await client.get_entity(CHANNEL)
    print(f"CHANNEL|{entity.title}|{entity.id}")
    
    # Create PDF directory
    pdf_dir = '/data/trading28/pdfs'
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Scan for PDFs
    pdfs = []
    async for msg in client.iter_messages(entity, limit=500):
        if msg.media and hasattr(msg.media, 'document'):
            doc = msg.media.document
            for attr in doc.attributes:
                if hasattr(attr, 'file_name') and attr.file_name:
                    name = attr.file_name
                    if name.lower().endswith('.pdf'):
                        size_mb = doc.size / (1024*1024) if doc.size else 0
                        pdfs.append((msg.id, name, size_mb))
                        break
    
    print(f"FOUND_PDFS|{len(pdfs)}")
    for msg_id, name, size in pdfs:
        print(f"PDF|{msg_id}|{name}|{size:.1f}MB")
    
    # Download PDFs
    downloaded = 0
    for msg_id, name, size in pdfs:
        filepath = os.path.join(pdf_dir, name)
        if os.path.exists(filepath):
            print(f"SKIP|{name} (exists)")
            continue
        
        msg = await client.get_messages(entity, ids=msg_id)
        print(f"DOWNLOADING|{name}|{size:.1f}MB...")
        await msg.download_media(file=filepath)
        downloaded += 1
        print(f"DONE|{name}")
    
    print(f"DOWNLOADED|{downloaded}")
    await client.disconnect()

asyncio.run(main())
