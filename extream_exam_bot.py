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
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter

# ===================== CONFIG =====================

BOT_TOKEN = os.getenv("BOT_TOKEN", "8318888870:AAG_HjP0ucgmq4zDUKsXgEFjj5371LffnZI")

OWNER_ID = 8389621809
OWNER_USERNAME = "@Your_Himus"

DENY_TEXT = (
    
    f"<blockquote><b> Please avoid unnecessary commands.</b></blockquote>\n"
    f"<blockquote><b>For any requirements or issues, kindly reach out to the Owner: {OWNER_USERNAME}</b></blockquote>"
    
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
    finalized : bool = False  


@dataclass(slots=True)
class UserResult:
    user_id: int
    full_name: str
    username: Optional[str]
    correct: int = 0
    wrong: int = 0
    answered: int = 0   # ✅ ADD THIS LINE
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
    task: Optional[asyncio.Task] = None

# ===================== GLOBAL STATE =====================

router = Router()

GLOBAL_QUESTION_BANK: List[Question] = []
EXAM_TEMPLATE = ExamTemplate()
EXAMS: Dict[int, ExamSession] = {}
ANNOUNCEMENT_DATA: dict | None = None
PENDING_QUESTIONS: List[Question] = []
BUILDER_MSG_ID: Optional[int] = None
READY_MSG_ID: Optional[int] = None
READY_CHAT_ID: Optional[int] = None

import time

LAST_BUILDER_UPDATE: float = 0.0
BUILDER_UPDATE_INTERVAL = 10.0
EditField = Literal["title", "time", "negative"]
OWNER_EDIT_STATE: Optional[EditField] = None

# ===================== PERMISSION HELPERS =====================

async def is_group_admin(bot: Bot, chat_id: int, user_id: int) -> bool:
    try:
        member: ChatMember = await bot.get_chat_member(chat_id, user_id)
        return member.status in {"creator", "administrator"}
    except TelegramBadRequest:
        return False

async def safe_edit(bot, chat_id, message_id, text, reply_markup):
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    except TelegramBadRequest as e:
        if "message is not modified" in str(e):
            pass
        else:
            raise

async def deny(target: Message | CallbackQuery) -> None:
    # If group & exam running → silent
    chat_id = None
    bot = None

    if isinstance(target, Message):
        chat_id = target.chat.id
        bot = target.bot
    else:
        chat_id = target.message.chat.id
        bot = target.message.bot

    if chat_id and await exam_running_in_chat(chat_id):
        await log_to_owner(
            f"🚫 <b>Permission denied during exam</b>\n"
            f"👤 {html.escape(target.from_user.full_name)}\n"
            f"💬 Chat ID: <code>{chat_id}</code>",
            bot
        )
        return

    # Normal behaviour
    if isinstance(target, CallbackQuery):
        await target.answer(DENY_TEXT, show_alert=True)
    else:
        await target.reply(DENY_TEXT, parse_mode=ParseMode.HTML)

async def exam_running_in_chat(chat_id: int) -> bool:
    session = EXAMS.get(chat_id)
    return bool(session and session.active)

async def log_to_owner(text: str, bot: Bot):
    try:
        await bot.send_message(
            OWNER_ID,
            text,
            parse_mode=ParseMode.HTML
        )
    except TelegramBadRequest:
        pass


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
def builder_text():
    return (
        "🧩 <b>Exam Builder Panel</b>\n\n"
        f"📘 Title: <b>{html.escape(EXAM_TEMPLATE.title)}</b>\n"
        f"⏱ Time/Q: <b>{EXAM_TEMPLATE.time_per_question}s</b>\n"
        f"➖ Negative: <b>-{EXAM_TEMPLATE.negative_mark}</b>\n"
        f"📦 Questions: <b>{len(GLOBAL_QUESTION_BANK) + len(PENDING_QUESTIONS)}</b>\n"
        f"🔒 Questions Locked: <b>{'Yes' if EXAM_TEMPLATE.finalized else 'No'}</b>"
    )


def inbox_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="➕ Add Q & Create E", callback_data="add_q")],
    ])

def builder_keyboard():
    lock_text = "🔓 Unlock Questions" if EXAM_TEMPLATE.finalized else "🔒 Lock Questions"

    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✏️ Edit Title", callback_data="edit_title")],
        [InlineKeyboardButton(text="⏱ Time / Question", callback_data="edit_time")],
        [InlineKeyboardButton(text="➖ Negative Mark", callback_data="edit_negative")],
        [InlineKeyboardButton(text=lock_text, callback_data="finalize")],
        [InlineKeyboardButton(text="🗑 Clear All Data", callback_data="clear_all")],
    ])



def group_ready_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="▶️ Start Exam", callback_data="start_exam")]
    ])

# ===================== COMMANDS =====================
@router.message(Command("startexam"))
async def cmd_start_exam(message: Message, bot: Bot):
    # Group only
    chat = message.chat

    if await exam_running_in_chat(chat.id):
        await log_to_owner(
            f"⚠️ <b>Start Exam button pressed again</b>\n"
            f"👤 {html.escape(message.from_user.full_name)}\n"
            f"💬 Chat ID: <code>{chat.id}</code>",
            bot
        )
        return

    if await exam_running_in_chat(message.chat.id):
        await log_to_owner(
            f"⚠️ <b>/startexam blocked (exam running)</b>\n"
            f"👤 {html.escape(message.from_user.full_name)}\n"
            f"💬 Chat ID: <code>{message.chat.id}</code>",
            bot
        )
        return

    if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return

    # Admin / Owner only
    if not await admin_or_owner(message, bot):
        return

    if not EXAM_TEMPLATE.finalized:
        GLOBAL_QUESTION_BANK.extend(PENDING_QUESTIONS)
        PENDING_QUESTIONS.clear()
        EXAM_TEMPLATE.finalized = True
    


    ready_msg =await message.reply(
        f"<blockquote>📣 <b>Exam Ready!</b></blockquote>\n"
        f"<b>{html.escape(EXAM_TEMPLATE.title)}</b>\n"
        f"<blockquote>"
        f"Total Questions: <b>{len(GLOBAL_QUESTION_BANK)}</b>\n"
        f"Time per Questio: <b>{EXAM_TEMPLATE.time_per_question}s</b>"
        f"</blockquote>",
        reply_markup=group_ready_menu(),
        parse_mode=ParseMode.HTML,
    )
    global READY_MSG_ID, READY_CHAT_ID
    READY_MSG_ID = ready_msg.message_id
    READY_CHAT_ID = message.chat.id

@router.message(Command("start"))
async def cmd_start(message: Message, bot: Bot):

    # Private → only owner
    if message.chat.type == ChatType.PRIVATE:
        if message.from_user.id != OWNER_ID:
            return
        await message.answer(
            "📘 Exam Control Panel",
            reply_markup=inbox_menu(),
            parse_mode=ParseMode.HTML
        )
        return

    # Group
    if message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}:
        # Non-admin → ignore silently
        if not await is_group_admin(bot, message.chat.id, message.from_user.id) \
           and message.from_user.id != OWNER_ID:
            return

        await message.reply(
            "ℹ️ Exam bot is ready.\n▶️ Use /startexam to start the exam.",
            parse_mode=ParseMode.HTML
        )




@router.message(Command("save_A"))
async def cmd_save_announcement(message: Message):
    global ANNOUNCEMENT_DATA

    # Only owner
    if not await owner_only(message):
        return

    # Only private
    if message.chat.type != ChatType.PRIVATE:
        await message.reply("❌ এই কমান্ড শুধু Private chat এ ব্যবহার করুন")
        return

    # Must be reply
    if not message.reply_to_message:
        await message.reply("❌ যে message save করতে চান, সেটার reply দিয়ে /save_A দিন")
        return

    src_msg = message.reply_to_message

    ANNOUNCEMENT_DATA = {
        "chat_id": src_msg.chat.id,
        "message_id": src_msg.message_id,
    }

    await message.reply(
        "✅ Announcement saved successfully\n"
        "👉 গ্রুপে গিয়ে /Announcement দিন",
        parse_mode=ParseMode.HTML
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
        "📢 Announcement System\n"
        "/save_A - (Private) Reply করে announcement save\n"
        "/Announcement - (Group) Announcement post & pin\n"
        "▶️ Start Exam - Exam শুরু করুন (Button)\n\n"
        "👥 Group Commands/startexam - Exam start (Admin/Owner)"
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
    if await exam_running_in_chat(message.chat.id):
    # finish is allowed only by admin/owner, but no group spam
        await log_to_owner(
            f"🛑 <b>Exam finish triggered</b>\n"
            f"👤 {html.escape(message.from_user.full_name)}\n"
            f"💬 Chat ID: <code>{message.chat.id}</code>",
            bot
        )

    if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return
    if not await admin_or_owner(message, bot):
        return
    session = EXAMS.get(message.chat.id)
    if not session or not session.active:
        await message.reply("ℹ️ কোনো active exam নেই", parse_mode=ParseMode.HTML)
        return
    session.active = False
    if session.task and not session.task.done():
        session.task.cancel()
    await finish_exam(session, bot)
    await message.reply("✅ The ongoing exam has been stopped.", parse_mode=ParseMode.HTML)

@router.message(Command("Announcement"))
async def cmd_announcement(message: Message, bot: Bot):
    global ANNOUNCEMENT_DATA
    if await exam_running_in_chat(message.chat.id):
        await log_to_owner(
            f"⚠️ <b>/Announcement blocked (exam running)</b>\n"
            f"👤 {html.escape(message.from_user.full_name)}\n"
            f"💬 Chat ID: <code>{message.chat.id}</code>",
            bot
        )
        return


    # Only group
    if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return

    # Admin or owner
    if not await admin_or_owner(message, bot):
        return

    if not ANNOUNCEMENT_DATA:
        await bot.send_message(OWNER_ID,"❌ কোনো saved announcement নেই")
        return

    try:
        copied = await bot.copy_message(
            chat_id=message.chat.id,
            from_chat_id=ANNOUNCEMENT_DATA["chat_id"],
            message_id=ANNOUNCEMENT_DATA["message_id"],
        )

        # Silent pin
        await bot.pin_chat_message(
            chat_id=message.chat.id,
            message_id=copied.message_id,
            disable_notification=True
        )

        await bot.send_message(OWNER_ID,"📌 Announcement posted & pinned")

        # Auto clear
        ANNOUNCEMENT_DATA = None

    except TelegramBadRequest as e:
        await message.reply("❌ Announcement post করতে সমস্যা হয়েছে")

# ===================== CALLBACKS =====================

@router.callback_query(F.data == "add_q")
async def cb_add_q(call: CallbackQuery):
    global BUILDER_MSG_ID

    if not await owner_only(call):
        return

    # ✅ CASE 1: builder নেই → নতুন পাঠাও + pin
    if BUILDER_MSG_ID is None:
        msg = await call.message.answer(
            builder_text(),
            reply_markup=builder_keyboard(),
            parse_mode=ParseMode.HTML
        )
        BUILDER_MSG_ID = msg.message_id

        await call.message.bot.pin_chat_message(
            chat_id=call.message.chat.id,
            message_id=BUILDER_MSG_ID,
            disable_notification=True
        )

    # ✅ CASE 2: builder আছে → শুধু edit
    else:
        try:
            await call.message.bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=BUILDER_MSG_ID,
                text=builder_text(),
                reply_markup=builder_keyboard(),
                parse_mode=ParseMode.HTML
            )
        except TelegramBadRequest:
            pass

    await call.answer("Quiz Poll পাঠান")



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


@router.callback_query(F.data == "edit_title")
async def cb_edit_title(call: CallbackQuery):
    global OWNER_EDIT_STATE
    if not await owner_only(call):
        return
    OWNER_EDIT_STATE = "title"
    await call.message.answer("✏️ Please enter a new exam title ( minimum 3 characters )", parse_mode=ParseMode.HTML)
    await call.answer()

@router.callback_query(F.data == "edit_time")
async def cb_edit_time(call: CallbackQuery):
    global OWNER_EDIT_STATE
    if not await owner_only(call):
        return
    OWNER_EDIT_STATE = "time"
    await call.message.answer("⏱ Enter the time per question ( 5 - 300 seconds )", parse_mode=ParseMode.HTML)
    await call.answer()

@router.callback_query(F.data == "edit_negative")
async def cb_edit_negative(call: CallbackQuery):
    global OWNER_EDIT_STATE
    if not await owner_only(call):
        return
    OWNER_EDIT_STATE = "negative"
    await call.message.answer("➖ Enter the negative mark ( 0 - 2 , e.g.: 0.25)", parse_mode=ParseMode.HTML)
    await call.answer()

@router.callback_query(F.data == "finalize")
async def cb_finalize(call: CallbackQuery):

    if not await owner_only(call):
        return
    
    if not EXAM_TEMPLATE.finalized:
        GLOBAL_QUESTION_BANK.extend(PENDING_QUESTIONS)
        PENDING_QUESTIONS.clear()
        EXAM_TEMPLATE.finalized = True
        await call.answer("Questions locked 🔒")
    else:
        EXAM_TEMPLATE.finalized = False
        await call.answer("Questions unlocked 🔓 ")

    if BUILDER_MSG_ID:
        try:
            await call.message.bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=BUILDER_MSG_ID,
                text=builder_text(),
                reply_markup=builder_keyboard(),
                parse_mode=ParseMode.HTML
            )
        except TelegramBadRequest:
            pass

    if not EXAM_TEMPLATE.finalized:
        GLOBAL_QUESTION_BANK.extend(PENDING_QUESTIONS)
        PENDING_QUESTIONS.clear()
        EXAM_TEMPLATE.finalized = True

    if BUILDER_MSG_ID:
        try:
            await call.message.bot.unpin_chat_message(
                chat_id = call.message.chat.id,
                message_id = BUILDER_MSG_ID
            )
        except TelegramBadRequest:
            pass

@router.callback_query(F.data == "clear_all")
async def cb_clear_all(call: CallbackQuery):
    global BUILDER_MSG_ID
    if not await owner_only(call):
        return

    GLOBAL_QUESTION_BANK.clear()
    PENDING_QUESTIONS.clear()
    EXAM_TEMPLATE.finalized = False

    try:
        await call.message.bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=BUILDER_MSG_ID,
            text=builder_text(),
            reply_markup=builder_keyboard(),
            parse_mode=ParseMode.HTML
        )
    except TelegramBadRequest:
        pass

    await call.answer("All data cleared")


@router.callback_query(F.data == "start_exam")
async def cb_start_exam(call: CallbackQuery, bot: Bot):

    chat = call.message.chat

    # ✅ Group check
    if chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        await call.answer("শুধু গ্রুপে চালানো যায়", show_alert=True)
        return

    # ✅ Permission check FIRST
    if call.from_user.id != OWNER_ID:
        if not await is_group_admin(bot, chat.id, call.from_user.id):
            await call.answer(
                "❌ Access denied - Only exam administratore can start the exam 🔒",
                show_alert=True
            )
            return

    # ✅ Exam finalized?
    if not EXAM_TEMPLATE.finalized:
        await call.answer("❌ Exam finalize করা হয়নি", show_alert=True)
        return

    if len(GLOBAL_QUESTION_BANK) == 0:
        await call.answer("❌ Question Bank ফাঁকা", show_alert=True)
        return

    # 🔥 এখন safe → delete Ready message
    try:
        await call.message.delete()
    except:
        pass

    global READY_MSG_ID, READY_CHAT_ID
    if READY_MSG_ID and READY_CHAT_ID:
        try:
            await bot.delete_message(READY_CHAT_ID, READY_MSG_ID)
        except:
            pass
        READY_MSG_ID = None
        READY_CHAT_ID = None

    # ✅ Start exam
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
        f"Total Questions: <b>{len(session.questions)}</b>\n"
        f"Time per Question: <b>{session.template.time_per_question}s</b>\n"
        f"Negative Mark: <b>{session.template.negative_mark}</b>\n"
        f"</blockquote>"
        f"<blockquote><b>সবার জন্য শুভকামনা!</b></blockquote>",
        parse_mode=ParseMode.HTML,
    )

    session.pinned_message_id = announced.message_id
    try:
        await bot.pin_chat_message(chat.id, announced.message_id)
    except:
        pass

    await call.answer()
    session.task = asyncio.create_task(run_exam(session, bot))


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
        if len(text) < 3:
            await message.answer("❌ Title কমপক্ষে ৩ অক্ষরের হতে হবে", parse_mode=ParseMode.HTML)
            return

        EXAM_TEMPLATE.title = text
        OWNER_EDIT_STATE = None

        if BUILDER_MSG_ID:
            await safe_edit(
                message.bot,
                message.chat.id,
                BUILDER_MSG_ID,
                builder_text(),
                builder_keyboard()
            )
        return


    if OWNER_EDIT_STATE == "time":
        if text.isdigit():
            t = int(text)
            if 5 <= t <= 300:
                EXAM_TEMPLATE.time_per_question = t
                OWNER_EDIT_STATE = None
                await message.bot.edit_message_text(chat_id=message.chat.id,message_id=BUILDER_MSG_ID,text=builder_text(),reply_markup=builder_keyboard(),parse_mode=ParseMode.HTML)
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
                await message.bot.edit_message_text(chat_id=message.chat.id,message_id=BUILDER_MSG_ID,text=builder_text(),reply_markup=builder_keyboard(),parse_mode=ParseMode.HTML)
            else:
                await message.answer("❌ Negative ০–২ এর মধ্যে দিন", parse_mode=ParseMode.HTML)
        except ValueError:
            await message.answer("❌ একটি দশমিক সংখ্যা দিন (যেমন 0.25)", parse_mode=ParseMode.HTML)
        return

    # No edit mode: show inbox
    await message.answer("📘 Exam Control Panel", reply_markup=inbox_menu(), parse_mode=ParseMode.HTML)

# ===================== POLL HANDLERS =====================

@router.message(F.poll)
async def handle_poll_save(message: Message, bot: Bot):
    
    if EXAM_TEMPLATE.finalized:
        return  # ❌ new quiz reject

    if message.chat.type != ChatType.PRIVATE:
        return
    if not await owner_only(message):
        return

    poll = message.poll
    if poll.type != "quiz" or poll.correct_option_id is None:
        return

    for q in PENDING_QUESTIONS:
        if q.text == poll.question:
            return
        
    PENDING_QUESTIONS.append(
        Question(
            text=poll.question,
            options=[opt.text for opt in poll.options],
            correct_id=poll.correct_option_id,
        )
    )

    global LAST_BUILDER_UPDATE
    now = time.time()

    if BUILDER_MSG_ID and (now - LAST_BUILDER_UPDATE)>= BUILDER_UPDATE_INTERVAL:
        try:
            await bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=BUILDER_MSG_ID,
                text=builder_text(),
                reply_markup=builder_keyboard(),
                parse_mode=ParseMode.HTML
            )
            LAST_BUILDER_UPDATE = now
        except TelegramRetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except TelegramBadRequest:
            pass

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
        res.answered += 1   # ✅ ADD THIS LINE


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
            if not session.active:
                break
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
        result.skipped = total_questions - result.answered

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

    global BUILDER_MSG_ID

    if BUILDER_MSG_ID:
        try:
            await bot.delete_message(
            chat_id=OWNER_ID,
            message_id=BUILDER_MSG_ID
            )
        except:
            pass

    BUILDER_MSG_ID = None


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


