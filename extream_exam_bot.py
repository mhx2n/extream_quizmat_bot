import os
import json
import asyncio
from pyrogram import Client, filters
from pyrogram.errors import FloodWait

API_ID = 36680379
API_HASH = "86bb52af9122d52bd16223114e3a52bb"
BOT_TOKEN = "8200161005:AAF_bgiFj7UYVtDGddi3yAT9GW7zFQzBr_U"
OWNER_ID = 8389621809

TEMP = "temp"
DB = "db.json"

os.makedirs(TEMP, exist_ok=True)

app = Client("bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

pending = {}

def load_db():
    if not os.path.exists(DB):
        return {}
    return json.load(open(DB))

def save_db(data):
    json.dump(data, open(DB, "w"))

def get_thumb():
    return load_db().get("thumb")

def set_thumb(path):
    data = load_db()
    data["thumb"] = path
    save_db(data)

def owner_only(func):
    async def wrapper(client, message):
        if message.from_user.id != OWNER_ID:
            return
        try:
            await func(client, message)
        except FloodWait as e:
            await asyncio.sleep(e.value)
            await func(client, message)
        except Exception as e:
            print("ERROR:", e)
            await message.reply(f"❌ {e}")
    return wrapper

@app.on_message(filters.command("start") & filters.private)
@owner_only
async def start(client, message):
    await message.reply("Send Photo → Save Thumbnail\nSend File → Rename")

@app.on_message(filters.photo & filters.private)
@owner_only
async def save_thumb(client, message):
    path = await message.download(f"{TEMP}/thumb.jpg")
    set_thumb(path)
    await message.reply("✅ Thumbnail Saved")

@app.on_message(filters.document & filters.private)
@owner_only
async def get_file(client, message):

    thumb = get_thumb()
    if not thumb:
        return await message.reply("❌ Set thumbnail first")

    file_path = await message.download(TEMP)
    pending[message.from_user.id] = file_path

    await message.reply("✏️ Now send new file name (only text)")

@app.on_message(filters.text & filters.private & ~filters.command(["start"]))
@owner_only
async def rename_file(client, message):

    user_id = message.from_user.id

    if user_id not in pending:
        return

    old_path = pending[user_id]
    new_name = message.text.strip()

    ext = os.path.splitext(old_path)[1]
    new_path = f"{TEMP}/{new_name}{ext}"

    os.rename(old_path, new_path)

    thumb = get_thumb()

    await message.reply("📤 Uploading...")

    await message.reply_document(
        document=new_path,
        thumb=thumb,
        caption="✅ Done"
    )

    os.remove(new_path)
    del pending[user_id]

print("Bot Running...")
app.run()
