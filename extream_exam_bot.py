import os
import json
import asyncio
from pyrogram import Client, filters
from pyrogram.errors import FloodWait

# ================== CONFIG ==================
API_ID = 36680379  # <-- my.telegram.org থেকে
API_HASH = "86bb52af9122d52bd16223114e3a52bb"  # <-- my.telegram.org থেকে
BOT_TOKEN = "8200161005:AAF_bgiFj7UYVtDGddi3yAT9GW7zFQzBr_U"  # <-- BotFather থেকে
OWNER_ID = 8389621809  # <-- তোমার Telegram user id

DB_FILE = "thumbnail_db.json"
TEMP_FOLDER = "downloads"

os.makedirs(TEMP_FOLDER, exist_ok=True)

# ================== APP ==================
app = Client(
    "rename_thumb_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ================== DATABASE ==================
def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f)

def get_thumb():
    return load_db().get("thumbnail")

def set_thumb(file_id):
    data = load_db()
    data["thumbnail"] = file_id
    save_db(data)

def delete_thumb():
    data = load_db()
    data.pop("thumbnail", None)
    save_db(data)

# ================== OWNER CHECK ==================
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
            await message.reply(f"⚠️ Error:\n{e}")
    return wrapper

# ================== START ==================
@app.on_message(filters.command("start") & filters.private)
@owner_only
async def start(client, message):
    await message.reply(
        "🔥 Rename + Thumbnail Bot Ready!\n\n"
        "📌 Send Photo → Set Thumbnail\n"
        "📌 Send Document → Rename + Apply Thumbnail\n\n"
        "/viewthumb - View thumbnail\n"
        "/delthumb - Delete thumbnail"
    )

# ================== SAVE THUMB ==================
@app.on_message(filters.photo & filters.private)
@owner_only
async def save_thumbnail(client, message):
    file_id = message.photo.file_id
    set_thumb(file_id)
    await message.reply("✅ Thumbnail Saved Successfully!")

# ================== VIEW THUMB ==================
@app.on_message(filters.command("viewthumb") & filters.private)
@owner_only
async def view_thumb(client, message):
    thumb = get_thumb()
    if not thumb:
        return await message.reply("❌ No thumbnail set.")
    await message.reply_photo(thumb, caption="📌 Current Thumbnail")

# ================== DELETE THUMB ==================
@app.on_message(filters.command("delthumb") & filters.private)
@owner_only
async def del_thumb(client, message):
    delete_thumb()
    await message.reply("🗑 Thumbnail Deleted Successfully!")

# ================== RENAME + APPLY ==================
@app.on_message(filters.document & filters.private)
@owner_only
async def rename_file(client, message):

    thumb_id = get_thumb()
    if not thumb_id:
        return await message.reply("❌ No thumbnail set. Send a photo first.")

    await message.reply("✏️ Send new file name (without extension):")

    try:
        name_msg = await client.listen(message.chat.id, timeout=60)
    except:
        return await message.reply("⏰ Time expired. Send file again.")

    new_name = name_msg.text.strip()

    # Download original file
    original_path = await message.download(file_name=TEMP_FOLDER)

    ext = os.path.splitext(original_path)[1]
    new_file_path = os.path.join(TEMP_FOLDER, f"{new_name}{ext}")

    os.rename(original_path, new_file_path)

    # Download thumbnail locally
    thumb_path = await client.download_media(thumb_id)

    try:
        await message.reply_document(
            document=new_file_path,
            thumb=thumb_path,
            caption="✅ Renamed & Thumbnail Applied"
        )
        await message.delete()
    except Exception as e:
        await message.reply(f"❌ Upload Failed:\n{e}")

    # Cleanup
    try:
        os.remove(new_file_path)
        os.remove(thumb_path)
    except:
        pass

# ================== RUN ==================
print("Bot is running...")
app.run()
