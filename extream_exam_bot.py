import os
import json
import asyncio
from pyrogram import Client, filters
from pyrogram.errors import FloodWait

# ================= CONFIG =================
API_ID = 36680379  # <-- এখানে তোমার API_ID বসাও
API_HASH = "8200161005:AAF_bgiFj7UYVtDGddi3yAT9GW7zFQzBr_U"
BOT_TOKEN = "8318888870:AAG_HjP0ucgmq4zDUKsXgEFjj5371LffnZI"
OWNER_ID = 8389621809  # <-- এখানে তোমার Telegram user id বসাও

DB_FILE = "thumbnail_db.json"

# ================= APP =================
app = Client(
    "thumbnail_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ================= DATABASE =================
def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f)

def get_thumb():
    data = load_db()
    return data.get("thumbnail")

def set_thumb(file_id):
    data = load_db()
    data["thumbnail"] = file_id
    save_db(data)

def delete_thumb():
    data = load_db()
    data.pop("thumbnail", None)
    save_db(data)

# ================= OWNER CHECK =================
def owner_only(func):
    async def wrapper(client, message):
        if message.from_user.id != OWNER_ID:
            return await message.reply("❌ You are not allowed to use this bot.")
        try:
            await func(client, message)
        except FloodWait as e:
            await asyncio.sleep(e.value)
            await func(client, message)
        except Exception as e:
            await message.reply(f"⚠️ Error: {e}")
    return wrapper

# ================= COMMANDS =================

@app.on_message(filters.command("start") & filters.private)
@owner_only
async def start(client, message):
    await message.reply(
        "✅ Thumbnail Bot Ready!\n\n"
        "📌 Send a photo to set thumbnail.\n"
        "📌 Send video/document to auto apply thumbnail.\n\n"
        "Commands:\n"
        "/viewthumb - View current thumbnail\n"
        "/delthumb - Delete thumbnail"
    )

@app.on_message(filters.photo & filters.private)
@owner_only
async def save_thumbnail(client, message):
    file_id = message.photo.file_id
    set_thumb(file_id)
    await message.reply("✅ Thumbnail Saved Successfully!")

@app.on_message(filters.command("viewthumb") & filters.private)
@owner_only
async def view_thumb(client, message):
    thumb = get_thumb()
    if not thumb:
        return await message.reply("❌ No thumbnail set.")
    await message.reply_photo(thumb, caption="📌 Current Thumbnail")

@app.on_message(filters.command("delthumb") & filters.private)
@owner_only
async def del_thumb(client, message):
    delete_thumb()
    await message.reply("🗑 Thumbnail Deleted Successfully!")

@app.on_message((filters.video | filters.document) & filters.private)
@owner_only
async def apply_thumbnail(client, message):
    thumb = get_thumb()
    if not thumb:
        return await message.reply("❌ No thumbnail set. Send a photo first.")

    try:
        await message.reply_document(
            document=message.document.file_id if message.document else message.video.file_id,
            thumb=thumb,
            caption=message.caption or ""
        )
        await message.delete()
    except Exception as e:
        await message.reply(f"⚠️ Failed to apply thumbnail:\n{e}")

# ================= RUN =================
print("Bot is running...")
app.run()    )

# ================= SAVE THUMB =================

@dp.message(F.photo)
async def save_thumb(message: Message):
    path = f"thumbs/{message.from_user.id}.jpg"
    await bot.download(message.photo[-1], destination=path)

    img = Image.open(path).convert("RGB")
    img.save(path, "JPEG")

    await message.answer("✅ Thumbnail Saved")

# ================= RECEIVE PDF =================

@dp.message(F.document)
async def receive_pdf(message: Message, state: FSMContext):
    if message.document.file_size > MAX_FILE_SIZE:
        await message.answer("⚠️ File too large (max 20MB).")
        return

    if not message.document.file_name.lower().endswith(".pdf"):
        await message.answer("⚠️ Only PDF supported.")
        return

    user_id = message.from_user.id

    if not await check_force_join(user_id):
        await message.answer("❌ Join required channel.")
        return

    thumb_path = f"thumbs/{user_id}.jpg"
    if not os.path.exists(thumb_path):
        await message.answer("⚠️ Send thumbnail first.")
        return

    file_path = f"files/{message.document.file_id}.pdf"
    await bot.download(message.document, destination=file_path)

    await state.update_data(file_path=file_path)
    await message.answer("✏️ Send new file name:")
    await state.set_state(RenameState.waiting_name)

# ================= COVER INJECTION =================

@dp.message(RenameState.waiting_name)
async def inject_cover(message: Message, state: FSMContext):
    async with LOCK:  # concurrency safe
        user_id = message.from_user.id
        ACTIVE_USERS.add(user_id)

        data = await state.get_data()
        original_pdf = data["file_path"]

        new_name = safe_name(message.text.strip())
        if not new_name.lower().endswith(".pdf"):
            new_name += ".pdf"

        new_path = f"files/{new_name}"
        thumb_path = f"thumbs/{user_id}.jpg"

        try:
            original = fitz.open(original_pdf)
            new_pdf = fitz.open()

            img = fitz.open(thumb_path)
            rect = img[0].rect

            cover = new_pdf.new_page(width=rect.width, height=rect.height)
            cover.insert_image(rect, filename=thumb_path)

            new_pdf.insert_pdf(original)
            new_pdf.save(new_path)

            new_pdf.close()
            original.close()

        except Exception as e:
            await message.answer("❌ PDF processing failed.")
            ACTIVE_USERS.remove(user_id)
            return

        await message.answer_document(
            document=FSInputFile(new_path),
            caption="✅ Done Successfully!"
        )

        async with aiosqlite.connect("bot.db") as db:
            await db.execute("UPDATE stats SET total = total + 1")
            await db.commit()

        try:
            os.remove(original_pdf)
            os.remove(new_path)
        except:
            pass

        ACTIVE_USERS.remove(user_id)
        await state.clear()

# ================= OWNER PANEL =================

@dp.message(Command("stats"))
async def stats(message: Message):
    if message.from_user.id != OWNER_ID:
        return

    async with aiosqlite.connect("bot.db") as db:
        async with db.execute("SELECT COUNT(*) FROM users") as cur:
            users = (await cur.fetchone())[0]
        async with db.execute("SELECT total FROM stats") as cur:
            total = (await cur.fetchone())[0]

    await message.answer(
        f"👥 Users: {users}\n"
        f"📂 Files Processed: {total}\n"
        f"⚡ Active Now: {len(ACTIVE_USERS)}"
    )

# ================= MAIN =================

async def main():
    await init_db()
    asyncio.create_task(auto_cleanup())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main()) 


