import asyncio
import logging
import os
import html
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Literal

from aiogram import Bot, Dispatcher, Router, F
from aiogram.enums import ChatType, ParseMode
from aiogram.filters import Command
from aiogram.types import (
    Message,
    CallbackQuery,
    PollAnswer,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ChatMember,
)
from aiogram.exceptions import TelegramBadRequest

# ===================== CONFIG =====================

BOT_TOKEN = os.getenv("BOT_TOKEN", "8318888870:AAER2X_Z2M7I9GOiA77tY9I46XlbvsXclos")

OWNER_ID = 8389621809
OWNER_USERNAME = "@Your_Himus"

DENY_TEXT = (
    "পড়তে Boy 😒 "
    f"অনুমতি নিতে যোগাযোগ করুন 👉 {OWNER_USERNAME}"
)

RIGHT_MARK = 1.0
MAX_LEADERBOARD = 10   # গ্রুপে শীর্ষ কতজন দেখাবে

# ===================== DATA MODELS =====================

@dataclass(slots=True)
class Question:
    text: str
    options: List[str]
    correct_id: int

@dataclass(slots=True)
class ExamTemplate:
    title: str = "Global Exam"
    time_per_question: int = 30
    negative_mark: float = 0.25
    finalized: bool = False

@dataclass(slots=True)
class UserResult:
    user_id: int
    full_name: str
    username: Optional[str]
    correct: int = 0
    wrong: int = 0
    skipped: int = 0
    score: float = 0.0

@dataclass(slots=True)
class ExamSession:
    chat_id: int
    questions: List[Question]
    template: ExamTemplate
    active: bool = False
    poll_map: Dict[str, int] = field(default_factory=dict)  # poll_id -> question index
    answered: Dict[int, Set[int]] = field(default_factory=dict)  # user_id -> set(question_index)
    results: Dict[int, UserResult] = field(default_factory=dict)
    pinned_message_id: Optional[int] = None

# ===================== GLOBAL STATE =====================

router = Router()

GLOBAL_QUESTION_BANK: List[Question] = []
EXAM_TEMPLATE = ExamTemplate()
EXAMS: Dict[int, ExamSession] = {}

EditField = Literal["title", "time", "negative"]
OWNER_EDIT_STATE: Optional[EditField] = None

# ===================== PERMISSION HELPERS =====================

async def is_group_admin(bot: Bot, chat_id: int, user_id: int) -> bool:
    try:
        member: ChatMember = await bot.get_chat_member(chat_id, user_id)
        return member.status in {"creator", "administrator"}
    except TelegramBadRequest:
        return False

async def deny(target: Message | CallbackQuery) -> None:
    if isinstance(target, CallbackQuery):
        await target.answer(DENY_TEXT, show_alert=True)
    else:
        await target.reply(DENY_TEXT)

async def owner_only(target: Message | CallbackQuery) -> bool:
    user_id = target.from_user.id
    if user_id != OWNER_ID:
        await deny(target)
        return False
    return True

async def admin_or_owner(message: Message, bot: Bot) -> bool:
    if message.from_user.id == OWNER_ID:
        return True
    if message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}:
        if await is_group_admin(bot, message.chat.id, message.from_user.id):
            return True
    await deny(message)
    return False

# ===================== UI BUILDERS =====================

def inbox_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="➕ Add Questions", callback_data="add_q")],
        [InlineKeyboardButton(text="📦 Question Bank", callback_data="bank")],
        [InlineKeyboardButton(text="🧩 Exam Builder", callback_data="builder")],
    ])

def builder_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✏️ Edit Title", callback_data="edit_title")],
        [InlineKeyboardButton(text="⏱ Time / Question", callback_data="edit_time")],
        [InlineKeyboardButton(text="➖ Negative Mark", callback_data="edit_negative")],
        [InlineKeyboardButton(text="✅ Finalize Exam", callback_data="finalize")],
        [InlineKeyboardButton(text="🗑 Clear Bank", callback_data="clear_bank")],
        [InlineKeyboardButton(text="⬅ Back", callback_data="back")],
    ])

def group_ready_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="▶️ Start Exam", callback_data="start_exam")]
    ])

# ===================== COMMANDS =====================

@router.message(Command("start"))
async def cmd_start(message: Message, bot: Bot):
    # Owner inbox
    if message.chat.type == ChatType.PRIVATE:
        if not await owner_only(message):
            return
        await message.answer("📘 Exam Control Panel", reply_markup=inbox_menu(), parse_mode=ParseMode.HTML)
        return

    # Group
    if not await admin_or_owner(message, bot):
        return

    if not EXAM_TEMPLATE.finalized:
        await message.reply("❌ Exam এখনও finalize করা হয়নি", parse_mode=ParseMode.HTML)
        return

    await message.reply(
        f"<blockquote>📣 <b>Exam Ready!</b></blockquote>\n"
        f"<b>{html.escape(EXAM_TEMPLATE.title)}</b>\n"
        f"<blockquote>"
        f"Questions: <b>{len(GLOBAL_QUESTION_BANK)}</b>\n"
        f"Time/Q: <b>{EXAM_TEMPLATE.time_per_question}s</b>"
        f"</blockquote>",
        reply_markup=group_ready_menu(),
        parse_mode=ParseMode.HTML,
    )

@router.message(Command("help"))
async def cmd_help(message: Message):
    help_text = (
        "📖 <b>Exam Bot Help</b>\n\n"
        "👤 <b>Owner Commands</b>\n"
        "/start - Exam Control Panel (Private)\n"
        "/help - এই সাহায্য মেনু দেখায়\n"
        "/clear - Question Bank ক্লিয়ার করুন\n\n"
        "👥 <b>Group Commands</b>\n"
        "/start - Exam Ready (Admin/Owner only)\n"
        "/finish - চলমান Exam শেষ করে Leaderboard দেখায়\n"
        "▶️ Start Exam - Exam শুরু করুন (Button)\n\n"
        "📊 <b>Exam Flow</b>\n"
        "- Owner Quiz Poll পাঠালে Question Bank এ সেভ হয়\n"
        "- Finalize করার পর Group এ Exam শুরু করা যায়\n"
        "- Leaderboard গ্রুপে দেখায় (Top 10)\n"
        "- Owner ইনবক্সে Full Leaderboard যায়\n"
        "- প্রত্যেক ইউজারকে DM এ Result পাঠানো হয়\n"
        "- Exam শেষে Question Bank auto-clear হয়\n"
    )
    await message.answer(help_text, parse_mode=ParseMode.HTML)

@router.message(Command("clear"))
async def cmd_clear(message: Message):
    if not await owner_only(message):
        return
    GLOBAL_QUESTION_BANK.clear()
    EXAM_TEMPLATE.finalized = False
    await message.answer("🗑 Question Bank ক্লিয়ার করা হয়েছে!\nনতুন Exam তৈরি করতে আবার প্রশ্ন যোগ করুন।", parse_mode=ParseMode.HTML)

@router.message(Command("finish"))
async def cmd_finish(message: Message, bot: Bot):
    if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return
    if not await admin_or_owner(message, bot):
        return
    session = EXAMS.get(message.chat.id)
    if not session or not session.active:
        await message.reply("ℹ️ কোনো active exam নেই", parse_mode=ParseMode.HTML)
        return
    session.active = False
    await finish_exam(session, bot)
    await message.reply("✅ Exam শেষ করা হয়েছে (partial results shown)", parse_mode=ParseMode.HTML)

# ===================== CALLBACKS =====================

@router.callback_query(F.data == "add_q")
async def cb_add_q(call: CallbackQuery):
    if not await owner_only(call):
        return
    await call.message.answer(
        "➕ Quiz Poll পাঠান (type=quiz, correct option সেট করুন)\n"
        "প্রতি Poll সেভ হয়ে Question Bank এ যোগ হবে।",
        parse_mode=ParseMode.HTML,
    )
    await call.answer()

@router.callback_query(F.data == "bank")
async def cb_bank(call: CallbackQuery):
    if not await owner_only(call):
        return
    total = len(GLOBAL_QUESTION_BANK)
    await call.message.answer(
        f"📦 Question Bank: <b>{total}</b> টি প্রশ্ন সেভ করা আছে।",
        parse_mode=ParseMode.HTML,
    )
    await call.answer()

@router.callback_query(F.data == "builder")
async def cb_builder(call: CallbackQuery):
    if not await owner_only(call):
        return
    await call.message.answer(
        "🧩 Exam Builder\n"
        f"Title: <b>{html.escape(EXAM_TEMPLATE.title)}</b>\n"
        f"Time/Q: <b>{EXAM_TEMPLATE.time_per_question}s</b>\n"
        f"Negative: <b>-{EXAM_TEMPLATE.negative_mark}</b>\n"
        f"Finalized: <b>{'Yes' if EXAM_TEMPLATE.finalized else 'No'}</b>",
        reply_markup=builder_menu(),
        parse_mode=ParseMode.HTML,
    )
    await call.answer()

@router.callback_query(F.data == "edit_title")
async def cb_edit_title(call: CallbackQuery):
    global OWNER_EDIT_STATE
    if not await owner_only(call):
        return
    OWNER_EDIT_STATE = "title"
    await call.message.answer("✏️ নতুন Exam Title পাঠান (৩+ অক্ষর)", parse_mode=ParseMode.HTML)
    await call.answer()

@router.callback_query(F.data == "edit_time")
async def cb_edit_time(call: CallbackQuery):
    global OWNER_EDIT_STATE
    if not await owner_only(call):
        return
    OWNER_EDIT_STATE = "time"
    await call.message.answer("⏱ প্রতি প্রশ্নের সময় (৫–৩০০ সেকেন্ড) পাঠান", parse_mode=ParseMode.HTML)
    await call.answer()

@router.callback_query(F.data == "edit_negative")
async def cb_edit_negative(call: CallbackQuery):
    global OWNER_EDIT_STATE
    if not await owner_only(call):
        return
    OWNER_EDIT_STATE = "negative"
    await call.message.answer("➖ Negative mark পাঠান (০–২, যেমন: 0.25)", parse_mode=ParseMode.HTML)
    await call.answer()

@router.callback_query(F.data == "finalize")
async def cb_finalize(call: CallbackQuery):
    if not await owner_only(call):
        return
    if len(GLOBAL_QUESTION_BANK) == 0:
        await call.message.answer("❌ Question Bank ফাঁকা। আগে প্রশ্ন যোগ করুন।", parse_mode=ParseMode.HTML)
        await call.answer()
        return
    EXAM_TEMPLATE.finalized = True
    await call.message.answer("✅ Exam finalized. এখন গ্রুপে /start দিয়ে শুরু করতে পারবেন।", parse_mode=ParseMode.HTML)
    await call.answer()

@router.callback_query(F.data == "clear_bank")
async def cb_clear_bank(call: CallbackQuery):
    if not await owner_only(call):
        return
    GLOBAL_QUESTION_BANK.clear()
    EXAM_TEMPLATE.finalized = False
    await call.message.answer("🗑 Question Bank ক্লিয়ার করা হয়েছে!\nনতুন Exam তৈরি করতে আবার প্রশ্ন যোগ করুন।", parse_mode=ParseMode.HTML)
    await call.answer()

@router.callback_query(F.data == "back")
async def cb_back(call: CallbackQuery):
    if not await owner_only(call):
        return
    await call.message.answer("📘 Exam Control Panel", reply_markup=inbox_menu(), parse_mode=ParseMode.HTML)
    await call.answer()

@router.callback_query(F.data == "start_exam")
async def cb_start_exam(call: CallbackQuery, bot: Bot):
    chat = call.message.chat
    if chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        await call.answer("শুধু গ্রুপে চালানো যায়", show_alert=True)
        return
    if call.from_user.id != OWNER_ID:
        if not await is_group_admin(bot, chat.id, call.from_user.id):
            await deny(call)
            return
    if not EXAM_TEMPLATE.finalized:
        await call.answer("Exam finalize করা হয়নি", show_alert=True)
        return
    if len(GLOBAL_QUESTION_BANK) == 0:
        await call.answer("Question Bank ফাঁকা", show_alert=True)
        return

    session = ExamSession(
        chat_id=chat.id,
        questions=list(GLOBAL_QUESTION_BANK),
        template=EXAM_TEMPLATE,
        active=True,
    )
    EXAMS[chat.id] = session

    announced = await call.message.answer(
        f"📌 <b>Exam Started</b>\n\n"
        f"<b>{html.escape(EXAM_TEMPLATE.title)}</b>\n\n"
        f"<blockquote>"
        f"Total: <b>{len(session.questions)}</b>\n"
        f"Time/Q: <b>{session.template.time_per_question}s</b>\n"
        f"</blockquote>"
        f"<blockquote>"
        "সবার জন্য শুভকামনা!"
        f"</blockquote>",
        parse_mode=ParseMode.HTML,
    )
    session.pinned_message_id = announced.message_id
    try:
        await bot.pin_chat_message(chat_id=session.chat_id, message_id=announced.message_id, disable_notification=True)
    except TelegramBadRequest:
        pass

    await call.answer()
    asyncio.create_task(run_exam(session, bot))

# ===================== TEXT INPUT HANDLER (OWNER EDITS) =====================

@router.message(F.text)
async def handle_builder_text(message: Message):
    global OWNER_EDIT_STATE
    if message.chat.type != ChatType.PRIVATE:
        return
    if message.from_user.id != OWNER_ID:
        await deny(message)
        return

    text = message.text.strip()

    if OWNER_EDIT_STATE == "title":
        if len(text) >= 3:
            EXAM_TEMPLATE.title = text
            OWNER_EDIT_STATE = None
            await message.answer(f"✅ Title updated to: <b>{html.escape(text)}</b>", reply_markup=builder_menu(), parse_mode=ParseMode.HTML)
        else:
            await message.answer("❌ Title কমপক্ষে ৩ অক্ষরের হতে হবে", parse_mode=ParseMode.HTML)
        return

    if OWNER_EDIT_STATE == "time":
        if text.isdigit():
            t = int(text)
            if 5 <= t <= 300:
                EXAM_TEMPLATE.time_per_question = t
                OWNER_EDIT_STATE = None
                await message.answer(f"✅ Time/Q set to: <b>{t}s</b>", reply_markup=builder_menu(), parse_mode=ParseMode.HTML)
            else:
                await message.answer("❌ Time ৫ থেকে ৩০০ সেকেন্ডের মধ্যে দিন", parse_mode=ParseMode.HTML)
        else:
            await message.answer("❌ একটি পূর্ণসংখ্যা দিন (৫–৩০০)", parse_mode=ParseMode.HTML)
        return

    if OWNER_EDIT_STATE == "negative":
        try:
            n = float(text)
            if 0 <= n <= 2:
                EXAM_TEMPLATE.negative_mark = n
                OWNER_EDIT_STATE = None
                await message.answer(f"✅ Negative mark set to: <b>-{n}</b>", reply_markup=builder_menu(), parse_mode=ParseMode.HTML)
            else:
                await message.answer("❌ Negative ০–২ এর মধ্যে দিন", parse_mode=ParseMode.HTML)
        except ValueError:
            await message.answer("❌ একটি দশমিক সংখ্যা দিন (যেমন 0.25)", parse_mode=ParseMode.HTML)
        return

    # No edit mode: show inbox
    await message.answer("📘 Exam Control Panel", reply_markup=inbox_menu(), parse_mode=ParseMode.HTML)

# ===================== POLL HANDLERS =====================

@router.message(F.poll)
async def handle_poll_save(message: Message):
    if message.chat.type != ChatType.PRIVATE:
        return
    if not await owner_only(message):
        return

    poll = message.poll
    if poll.type != "quiz" or poll.correct_option_id is None:
        await message.answer("❌ Quiz Poll (type=quiz) এবং correct option সেট করুন", parse_mode=ParseMode.HTML)
        return

    GLOBAL_QUESTION_BANK.append(
        Question(
            text=poll.question,
            options=[opt.text for opt in poll.options],
            correct_id=poll.correct_option_id,
        )
    )

    await message.answer(f"✅ Question saved | Total: <b>{len(GLOBAL_QUESTION_BANK)}</b>", parse_mode=ParseMode.HTML)

@router.poll_answer()
async def handle_poll_answer(poll_answer: PollAnswer, bot: Bot):
    for session in EXAMS.values():
        if not session.active:
            continue
        if poll_answer.poll_id not in session.poll_map:
            continue

        q_index = session.poll_map[poll_answer.poll_id]
        user = poll_answer.user

        res = session.results.setdefault(
            user.id,
            UserResult(user.id, user.full_name, user.username),
        )

        answered_q = session.answered.setdefault(user.id, set())
        if q_index in answered_q:
            continue
        answered_q.add(q_index)

        if poll_answer.option_ids and poll_answer.option_ids[0] == session.questions[q_index].correct_id:
            res.correct += 1
            res.score += RIGHT_MARK
        else:
            res.wrong += 1
            res.score -= session.template.negative_mark

# ===================== EXAM FLOW =====================

async def run_exam(session: ExamSession, bot: Bot):
    try:
        for idx, q in enumerate(session.questions, start=1):
            poll_message = await bot.send_poll(
                chat_id=session.chat_id,
                question=f"Q{idx}. {q.text}",
                options=q.options,
                type="quiz",
                correct_option_id=q.correct_id,
                is_anonymous=False,
                open_period=session.template.time_per_question,
            )
            session.poll_map[poll_message.poll.id] = idx - 1
            await asyncio.sleep(session.template.time_per_question + 2)

        session.active = False
        await finish_exam(session, bot)
    except Exception as e:
        logging.exception("Exam runtime error: %s", e)

async def finish_exam(session: ExamSession, bot: Bot):
    total_questions = len(session.questions)

    # Compute skipped
    for result in session.results.values():
        result.skipped = total_questions - (result.correct + result.wrong)

    # Sort leaderboard
    leaderboard = sorted(
        session.results.values(),
        key=lambda r: (-r.score, -r.correct),
    )

    if leaderboard:
        # Group: top MAX_LEADERBOARD
        lines = ["<blockquote>🏆 <b>Exam Leaderboard</b></blockquote>", ""]
        for idx, res in enumerate(leaderboard[:MAX_LEADERBOARD], start=1):
            safe_name = html.escape(res.full_name)
            lines.append(
                f"{idx}. {safe_name} — "
                f"<blockquote>"
                f"<b>🌟 Score: {res.score:.2f}</b> \n "
                f"<b>✅ Correct : {res.correct} \n ❌ Wrong : {res.wrong} \n ⏭ Skippped : {res.skipped}</b>"
                f"</blockquote>"
            )

        await bot.send_message(
            session.chat_id,
            "\n".join(lines),
            parse_mode=ParseMode.HTML,
        )

        # Owner inbox: full leaderboard
        full_lines = ["<blockquote>📥 <b>Full Exam Leaderboard</b></blockquote>", ""]
        for idx, res in enumerate(leaderboard, start=1):
            safe_name = html.escape(res.full_name)
            full_lines.append(
                f"{idx}. {safe_name} — "
                f"<blockquote>"
                f"<b>🌟 Score: {res.score:.2f}</b> \n "
                f"<b>✅ Correct : {res.correct} \n ❌ Wrong : {res.wrong} \n ⏭ Skipped : {res.skipped}</b>"
                f"</blockquote>"
            )
        try:
            await bot.send_message(
                OWNER_ID,
                "\n".join(full_lines),
                parse_mode=ParseMode.HTML,
            )
        except TelegramBadRequest:
            pass

    # DM results (only users who allow DM)
    for res in leaderboard:
        try:
            safe_name = html.escape(res.full_name)
            await bot.send_message(
                res.user_id,
                (
                    f"<blockquote>📊 <b>Your Exam Result</b></blockquote>\n"
                    f"Name: {safe_name}\n"
                    f"<blockquote>"
                    f"Score: <b>{res.score:.2f}</b>\n"
                    f"✅ Correct: <b>{res.correct}</b>\n"
                    f"❌ Wrong: <b>{res.wrong}</b>\n"
                    f"⏭ Skipped: <b>{res.skipped}</b>"
                    f"</blockquote>"
                ),
                parse_mode=ParseMode.HTML,
            )
        except TelegramBadRequest:
            # user might not have messaged the bot
            pass

    # Auto-clear after exam finish
    GLOBAL_QUESTION_BANK.clear()
    EXAM_TEMPLATE.finalized = False

# ===================== MAIN =====================

async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    if BOT_TOKEN == "PUT_YOUR_BOT_TOKEN_HERE":
        raise RuntimeError("Please set BOT_TOKEN before running the bot")

    bot = Bot(BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
