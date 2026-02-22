import os
import json
import asyncio
from pyrogram import Client, filters
from pyrogram.errors import FloodWait

# ============== CONFIG ==============
API_ID = 36680379
API_HASH = "86bb52af9122d52bd16223114e3a52bb"
BOT_TOKEN = "8200161005:AAF_bgiFj7UYVtDGddi3yAT9GW7zFQzBr_U"
OWNER_ID = 8389621809

DB_FILE = "db.json"
TEMP_FOLDER = "temp"

os.makedirs(TEMP_FOLDER, exist_ok=True)

app = Client(
    "rename_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ============== DB ==============
def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f)

def get_thumb():
    return load_db().get("thumb")

def set_thumb(path):
    data = load_db()
    data["thumb"] = path
    save_db(data)

# ============== OWNER CHECK ==============
def owner_only(func):
    async def wrapper(client, message):
        if not message.from_user or message.from_user.id != OWNER_ID:
            return
        try:
            await func(client, message)
        except FloodWait as e:
            await asyncio.sleep(e.value)
            await func(client, message)
        except Exception as e:
            print("ERROR:", e)
            await message.reply(f"❌ Error:\n{e}")
    return wrapper

# ============== START ==============
@app.on_message(filters.command("start") & filters.private)
@owner_only
async def start(client, message):
    await message.reply(
        "🔥 Rename + Thumbnail Bot Ready!\n\n"
        "1️⃣ Send Photo → Save Thumbnail\n"
        "2️⃣ Send File → Rename + Apply Thumbnail"
    )

# ============== SAVE THUMB ==============
@app.on_message(filters.photo & filters.private)
@owner_only
async def save_thumb(client, message):
    thumb_path = await message.download(file_name=f"{TEMP_FOLDER}/thumb.jpg")
    set_thumb(thumb_path)
    await message.reply("✅ Thumbnail Saved")

# ============== RENAME + APPLY ==============
pending = {}

@app.on_message(filters.document & filters.private)
@owner_only
async def receive_file(client, message):

    thumb = get_thumb()
    if not thumb or not os.path.exists(thumb):
        return await message.reply("❌ Set thumbnail first.")

    file_path = await message.download(file_name=TEMP_FOLDER)
    pending[message.from_user.id] = file_path

    await message.reply("✏️ Send new file name (without extension):")


@app.on_message(filters.text & filters.private)
@owner_only
async def rename_process(client, message):

    user_id = message.from_user.id

    if user_id not in pending:
        return

    old_path = pending[user_id]
    new_name = message.text.strip()

    ext = os.path.splitext(old_path)[1]
    new_path = os.path.join(TEMP_FOLDER, f"{new_name}{ext}")

    os.rename(old_path, new_path)

    thumb = get_thumb()

    try:
        await message.reply_document(
            document=new_path,
            thumb=thumb,
            caption="✅ Renamed & Thumbnail Added"
        )
    except Exception as e:
        print("UPLOAD ERROR:", e)
        await message.reply("❌ Upload failed.")
        return

    try:
        os.remove(new_path)
    except:
        pass

    del pending[user_id]

# ============== RUN ==============
print("Bot running...")
app.run()
