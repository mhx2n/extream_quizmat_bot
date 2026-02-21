import os
import asyncio
import aiosqlite
import fitz
from PIL import Image
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import CommandStart, Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext

# ================= CONFIG =================

BOT_TOKEN = "8200161005:AAF_bgiFj7UYVtDGddi3yAT9GW7zFQzBr_U"
OWNER_ID = 7231202058
FORCE_CHANNELS = ["@Amar_Channel_Amar_Post"]

CLEANUP_TIME = 3600  # 1 hour
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB safe limit

# ================= INIT =================

bot = Bot(BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

os.makedirs("files", exist_ok=True)
os.makedirs("thumbs", exist_ok=True)

ACTIVE_USERS = set()
LOCK = asyncio.Lock()

# ================= DATABASE =================

async def init_db():
    async with aiosqlite.connect("bot.db") as db:
        await db.execute("CREATE TABLE IF NOT EXISTS users(user_id INTEGER PRIMARY KEY)")
        await db.execute("CREATE TABLE IF NOT EXISTS stats(total INTEGER DEFAULT 0)")
        await db.execute("INSERT OR IGNORE INTO stats(rowid,total) VALUES(1,0)")
        await db.commit()

# ================= UTIL =================

async def check_force_join(user_id):
    for channel in FORCE_CHANNELS:
        try:
            member = await bot.get_chat_member(channel, user_id)
            if member.status not in ["member", "administrator", "creator"]:
                return False
        except:
            return False
    return True

def safe_name(name):
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)

# ================= AUTO CLEANUP =================

async def auto_cleanup():
    while True:
        now = datetime.now()
        for folder in ["files", "thumbs"]:
            for f in os.listdir(folder):
                path = os.path.join(folder, f)
                if os.path.isfile(path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(path))
                    if now - file_time > timedelta(seconds=CLEANUP_TIME):
                        try:
                            os.remove(path)
                        except:
                            pass
        await asyncio.sleep(CLEANUP_TIME)

# ================= STATES =================

class RenameState(StatesGroup):
    waiting_name = State()

# ================= START =================

@dp.message(CommandStart())
async def start(message: Message):
    async with aiosqlite.connect("bot.db") as db:
        await db.execute("INSERT OR IGNORE INTO users(user_id) VALUES(?)", (message.from_user.id,))
        await db.commit()

    if not await check_force_join(message.from_user.id):
        btn = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(
                    text="📢 Join Channel",
                    url=f"https://t.me/{FORCE_CHANNELS[0][1:]}"
                )]
            ]
        )
        await message.answer("⚠️ Please join required channel first!", reply_markup=btn)
        return

    await message.answer(
        "🔥 Ultra Pro PDF Rename Bot\n\n"
        "1️⃣ Send thumbnail image\n"
        "2️⃣ Send PDF file\n"
        "3️⃣ Send new name\n\n"
        "💎 Cover will be injected into PDF"
    )

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
    asyncio.run(main()) a 
