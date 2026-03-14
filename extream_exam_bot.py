#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
প্রবাহ — Professional Ultra Quiz Bot (Single File)

Core features preserved:
- Admin/Owner can send Text blocks or Polls/Quizzes -> parsed -> buffered
- /done exports CSV (utf-8-sig) then clears buffer
- /filter adds per-admin filters
- /clear clears buffer

Enhanced (new additions, without breaking existing behavior):
- Professional English UI + role-based /help (polished)
- Reply-aware commands:
  - /ask, /reply, /broadcast work either inline OR by replying to a message
- Channel privacy / access control:
  - Owner sees all channels
  - Admin sees ONLY channels they added
  - Owner can grant/revoke “view all channels” access to selected admins
- Per-admin visibility (owner can view all):
  - /adminpanel: admins see own stats; owner sees all
  - /banned: admins see only bans they issued; owner sees all
"""

import asyncio
import contextlib
import datetime as dt
import json
import logging
from multiprocessing import context

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import os
import re
import sqlite3
import sys
import tempfile
import time
import uuid
from bs4 import BeautifulSoup
from datetime import datetime
import base64
import html as html_escape
import requests
from concurrent.futures import ThreadPoolExecutor
#from openai import OpenAI
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
#from openai import OpenAI
import pandas as pd
from telegram import Update, Poll, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode, ChatAction
from telegram.error import RetryAfter, Forbidden, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# =========================================================
# ✅ HARD-CODED CONFIG
# =========================================================
BOT_TOKEN = "8286585007:AAHz1NIOXIbkBATy9qdcrQtHNr0DauL325U"  # set in Pella Env Vars
OWNER_ID = 8389621809  # your Telegram numeric user id

OWNER_CONTACT = "@Your_Himus"
BOT_BRAND = "প্রবাহ"

DB_PATH = "probaho_bot.sqlite3"
MAX_BUFFERED_QUESTIONS = 500
POST_DELAY_SECONDS = 0.8
BROADCAST_DELAY_SECONDS = 0.05

START_TIME = time.time()  # process start time (uptime)

# ---------------------------
# GEMINI (Google AI Studio) — Image→Quiz extraction (HARDCODED)
# ---------------------------
# ⚠️ Security note: If you share this file, your keys can leak.
GEMINI3_HTTP_URL = "http://127.0.0.1:5000/api/ask"  # optional
GEMINI3_HTTP_TIMEOUT = 60
GEMINI_API_KEY = "AIzaSyCdfVvpLA3ynViRjqw8mM1kDQRY2jLVCF8"  # set in Pella Env Vars
# Free & stable vision model
# ---------------------------
# MODEL CONFIGURATION (Switch here easily)
# ---------------------------

# অপশন ১: ফাস্ট এবং ফ্রি (Flash) - বর্তমানে লাল দাগ বা লিমিট শেষ হলে এটি বন্ধ রাখুন
# GEMINI_MODEL_VISION = "models/gemini-2.5-flash"
# GEMINI_MODEL_TEXT = "models/gemini-2.5-flash"

# অপশন ২: পাওয়ারফুল এবং হাই লিমিট (Pro) - আপনার স্ক্রিনশট অনুযায়ী এটি এখন ব্যবহার করা উচিত
GEMINI_MODEL_VISION = "models/gemini-2.5-flash"
GEMINI_MODEL_TEXT = "models/gemini-2.5-flash"
########-------------------------------------------
GEMINI_TIMEOUT_SECONDS = 60
GEMINI_TEXT_TIMEOUT_SECONDS = 25  # faster text responses





# ---------------------------
# ✅ Solver backend preference
# ---------------------------
# If you want NO Google API key usage for /solve_on (users), keep this False.
# When False, the bot will use only Gemini3 (Gemini3.py / web session) and will NOT call Google AI Studio REST.
USE_OFFICIAL_GEMINI_REST_FALLBACK = True

# Use official Gemini REST for Generate Quiz JSON (recommended). Works even if solve REST fallback is disabled.
USE_GEMINI_REST_FOR_GENQUIZ = True
# ---------------------------
# ✅ Perplexity (HTTP) — Text/MCQ solving fallback (from main.py)
# ---------------------------
# Used ONLY when Gemini3 fails (prevents "REST fallback disabled" error for math/solve).
PERPLEXITY_API = "https://pplxtyai.vercel.app/api/ask"
USE_PERPLEXITY_FALLBACK = True


# ---------------------------
# ✅ DeepSeek (OpenAI-compatible) — optional third AI
# ---------------------------
# NOTE: Keep empty if you don't want DeepSeek button to work.
#DEEPSEEK_API_KEY = "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"  # set in Pella Env Vars
#DEEPSEEK_BASE_URL = "https://openrouter.ai/api/v1"
#DEEPSEEK_MODEL_TEXT = "deepseek/deepseek-r1-0528:free"

SHOW_DEEPSEEK_BUTTON = False 

# ---------------------------

if not BOT_TOKEN:
    raise SystemExit("Please set BOT_TOKEN inside the code first.")
if not isinstance(OWNER_ID, int) or OWNER_ID <= 0:
    raise SystemExit("Please set OWNER_ID (numeric) inside the code first.")


# =========================================================
# Render Free Web Service Health Server
# =========================================================
def _run_render_health_server():
    port = int(os.getenv("PORT", "10000"))

    class _HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, format, *args):
            return

    try:
        server = HTTPServer(("0.0.0.0", port), _HealthHandler)
        server.serve_forever()
    except Exception as e:
        logging.exception("Health server failed: %s", e)


# ---------------------------
# LOGGING
# ---------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("probaho")



# =========================================================
# ✅ Concurrency: separate pools so USER কাজ করলে ADMIN/OWNER আটকে না যায়
# =========================================================
from concurrent.futures import ThreadPoolExecutor

# Pella / low-RAM safe defaults (tunable)
_OWNER_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="owner")
_ADMIN_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="admin")
_USER_EXECUTOR  = ThreadPoolExecutor(max_workers=8, thread_name_prefix="user")

# Limit how many heavy jobs can run at once per group
_OWNER_SEM = asyncio.Semaphore(2)
_ADMIN_SEM = asyncio.Semaphore(3)
_USER_SEM  = asyncio.Semaphore(6)

def _pick_executor_and_sem(role: str):
    r = (role or "").upper()
    if r == "OWNER":
        return _OWNER_EXECUTOR, _OWNER_SEM
    if r == "ADMIN":
        return _ADMIN_EXECUTOR, _ADMIN_SEM
    # default USER
    return _USER_EXECUTOR, _USER_SEM

async def _run_blocking(role: str, fn, *args, timeout: float | None = None, **kwargs):
    """Run a blocking function in a role-based thread pool.

    Why: Python-telegram-bot v20 runs handlers in asyncio. If we call blocking
    I/O (requests, AI calls, heavy parsing) directly, it blocks the event loop.
    This helper offloads the work while also preventing USER workload from
    starving ADMIN/OWNER tasks.
    """
    executor, sem = _pick_executor_and_sem(role)
    loop = asyncio.get_running_loop()

    async with sem:
        fut = loop.run_in_executor(executor, lambda: fn(*args, **kwargs))
        if timeout is not None:
            return await asyncio.wait_for(fut, timeout=timeout)
        return await fut

# =========================================================
# ✅ HTTP / Rate-limit helpers
# =========================================================
class RateLimitError(RuntimeError):
    """Raised when a backend is rate-limited / quota exhausted."""
    pass

def _is_gemini_quota_error(status_code: int, body_text: str) -> bool:
    t = (body_text or "").lower()
    if status_code in (429,):
        return True
    # Gemini sometimes returns 403 for quota/project billing issues
    if status_code in (403,) and ("quota" in t or "rate" in t or "exhaust" in t or "billing" in t):
        return True
    if "resource_exhausted" in t or "rate limit" in t or "quota" in t:
        return True
    return False

def _requests_with_retries(method, url: str, *, json_payload=None, params=None, timeout=25, max_tries=3):
    """requests.* wrapper with small retries + backoff for transient network/rate errors."""
    import requests as _rq
    last_err = None
    for i in range(max_tries):
        try:
            r = method(url, json=json_payload, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            # Rate limit / quota
            if _is_gemini_quota_error(r.status_code, r.text):
                raise RateLimitError(f"Gemini rate-limited/quota exhausted (HTTP {r.status_code}).")
            # transient server errors
            if r.status_code in (500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            else:
                # non-retryable
                r.raise_for_status()
                return r
        except RateLimitError:
            raise
        except Exception as e:
            last_err = e
        time.sleep(0.8 * (2 ** i))
    if last_err:
        raise last_err
    raise RuntimeError("Request failed.")




# =========================================================
# ✅ Perplexity fallback client (merged from main.py)
# =========================================================
def query_ai(prompt: str) -> str | None:
    """HTTP fallback solver. Returns plain text answer or None."""
    if not USE_PERPLEXITY_FALLBACK:
        return None
    try:
        r = requests.get(PERPLEXITY_API, params={"prompt": prompt}, timeout=60)
        if r.status_code != 200:
            logger.error("Perplexity HTTP %s: %s", r.status_code, (r.text or "")[:2000])
            return None
        data = r.json()
        if data.get("status") == "success" and "answer" in data:
            return str(data["answer"]).strip()
        logger.error("Perplexity bad response: %s", str(data)[:2000])
        return None
    except Exception as e:
        logger.exception("Perplexity error: %s", e)
        return None


# =========================================================
# ✅ Multi-AI Router (Gemini3 / Perplexity / DeepSeek) — Inline Buttons
# =========================================================

_PENDING_KEY = "pending_solve_requests"

def _pending_store(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    d = context.application.bot_data.get(_PENDING_KEY)
    if not isinstance(d, dict):
        d = {}
        context.application.bot_data[_PENDING_KEY] = d
    return d

def _make_token() -> str:
    return uuid.uuid4().hex[:10]

def _solver_picker_kb(token: str) -> InlineKeyboardMarkup:
    kb = [
        [
            InlineKeyboardButton("✨Gemini 3 Flash", callback_data=f"solve:G:{token}"),
            InlineKeyboardButton("֎Perplexity (GPT-5.1)", callback_data=f"solve:P:{token}"),
        ],
        #[
           # InlineKeyboardButton("🐳 DeepSeek", callback_data=f"solve:D:{token}"),
        #],
    ]
    return InlineKeyboardMarkup(kb)

def _verify_kb(token: str, used: str, kind: str = "text") -> InlineKeyboardMarkup:
    alt = []
    if used != "P":
        alt.append(InlineKeyboardButton("⚛ Perplexity", callback_data=f"solve:P:{token}"))
    if used != "G":
        alt.append(InlineKeyboardButton("✨ Gemini", callback_data=f"solve:G:{token}"))

    rows = [alt[i:i+2] for i in range(0, len(alt), 2)]

    # Show Generate Quiz ONLY for quiz/poll based solutions
    if str(kind or "") == "poll":
        rows.append([InlineKeyboardButton("📊 Generate Quiz", callback_data=f"genquiz:{token}")])

    return InlineKeyboardMarkup(rows)

# def _deepseek_client() -> OpenAI:
#     if not DEEPSEEK_API_KEY or "sk-" not in str(DEEPSEEK_API_KEY):
#         raise RuntimeError("DeepSeek API Key সেট করা নেই।")
#     return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

def deepseek_solve_text(problem_text: str) -> str:
    prompt = (STRICT_SYSTEM_PROMPT + "\n\nUser Message:\n" + (problem_text or "").strip()).strip()
    client = _deepseek_client()
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL_TEXT,
        messages=[
            {"role": "system", "content": "You are a strict academic problem-solving assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip() or "..."

def perplexity_solve_text(problem_text: str) -> str:
    prompt = (STRICT_SYSTEM_PROMPT + "\n\nUser Message:\n" + (problem_text or "").strip()).strip()
    alt = query_ai(prompt)
    if alt:
        return alt.strip()
    raise RuntimeError("Perplexity unavailable.")

def perplexity_solve_mcq_json(question: str, options: List[str]) -> Dict[str, Any]:
    # Ask Perplexity proxy to return strict JSON
    q = (question or "").strip()
    opts = [(o or "").strip() for o in (options or []) if (o or "").strip()][:5]
    opt_lines = "\n".join([f"{_safe_letter(i+1)}. {opts[i]}" for i in range(len(opts))])
    p2 = (
        "Return STRICT JSON only (no markdown).\n"
        "Solve the MCQ and respond in this JSON format exactly:\n"
        "{\"answer\":1,\"confidence\":0,\"explanation\":\"...\",\"why_not\":{\"A\":\"..\",\"B\":\"..\",\"C\":\"..\",\"D\":\"..\",\"E\":\"..\"}}\n\n"
        f"Question:\n{q}\n\nOptions:\n{opt_lines}\n"
    )
    alt = query_ai(p2)
    if not alt:
        raise RuntimeError("Perplexity unavailable.")
    try:
        data = _extract_json_strict(alt)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"answer": 0, "confidence": 0, "explanation": (alt[:1800] if isinstance(alt, str) else str(alt)[:1800]), "why_not": {}}

def deepseek_solve_mcq_json(question: str, options: List[str]) -> Dict[str, Any]:
    """Solve an MCQ using DeepSeek and return strict JSON dict.

    This function must NEVER raise due to minor JSON formatting issues; it will attempt repair.
    """
    q = (question or "").strip()
    opts = [(o or "").strip() for o in (options or []) if (o or "").strip()][:5]
    opt_lines = "\n".join([f"{_safe_letter(i+1)}. {opts[i]}" for i in range(len(opts))])

    prompt = (
        "Return STRICT JSON only. No markdown. No extra text.\n"
        "Solve the MCQ and respond in this JSON format exactly:\n"
        "{\"answer\":1,\"confidence\":0,\"explanation\":\"Bangla...\\nEnglish...\",\"why_not\":{\"A\":\"..\",\"B\":\"..\",\"C\":\"..\",\"D\":\"..\",\"E\":\"..\"}}\n\n"
        f"Question:\n{q}\n\nOptions:\n{opt_lines}\n"
    )

    client = _deepseek_client()
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL_TEXT,
        messages=[
            {"role": "system", "content": "You are a strict academic problem-solving assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    raw = (resp.choices[0].message.content or "").strip()

    schema_hint = '{"answer":1,"confidence":0.0,"explanation":"Bangla...\\nEnglish...","why_not":{"A":"..","B":"..","C":"..","D":"..","E":".."}}'
    try:
        data = _extract_json_strict(raw)
    except Exception:
        repaired = _repair_to_json(raw, schema_hint=schema_hint, timeout_seconds=18)
        if not repaired:
            # graceful fallback
            return {"answer": 0, "confidence": 0, "explanation": (raw[:1800] or ""), "why_not": {}}
        data = repaired

    if isinstance(data, dict):
        return data
    return {"answer": 0, "confidence": 0, "explanation": (raw[:1800] or ""), "why_not": {}}


# Regex to detect Bangla characters
_BN_CHAR_RE = re.compile(r"[\u0980-\u09FF]")

def _is_bangla_text(s: str) -> bool:
    return bool(_BN_CHAR_RE.search(s or ""))

def _normalize_options(options: List[str], max_n: int = 4) -> List[str]:
    opts = [(o or "").strip() for o in (options or []) if (o or "").strip()]
    if len(opts) < 2:
        return ["Option A", "Option B", "Option C", "Option D"][:max_n]
    if len(opts) >= max_n:
        return opts[:max_n]
    while len(opts) < max_n:
        opts.append(f"Option {chr(65+len(opts))}")
    return opts[:max_n]

def _trim_expl_for_poll(expl: str, link: str = "") -> str:
    # Keep explanation short enough for Telegram quiz explanation field.
    # Telegram allows ~200 chars, but we keep it smaller to avoid errors.
    t = (expl or "").strip()

    # Prefer only first 2 lines if many lines
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        t = "\n".join(lines[:2])

    if link:
        t = (t + "\n" if t else "") + f" {link}".strip()

    t = t.strip()
    if len(t) > 160:
        t = t[:157] + "..."
    return t


def generate_quiz_items_gemini_then_verify(seed_question: str, seed_options: List[str]) -> List[Dict[str, Any]]:
    """Generate 3 MCQs on the same topic using Gemini3, then verify each with Perplexity."""
    sq = (seed_question or "").strip()
    so = _normalize_options(seed_options or [], max_n=4)

    is_bn = _is_bangla_text(sq + " " + " ".join(so))
    allowed = [
        "পদার্থবিজ্ঞান","রসায়ন","ম্যাথ","বাইলোজি","বাংলা","ইংরেজি","স্বাধারণজ্ঞান","মানবিক দক্ষতা"
    ] if is_bn else [
        "Physics","Chemistry","Math","Biology","Bangla","English","General Knowledge","Mental Ability"
    ]

    prompt = (
        "Return STRICT JSON only (no markdown, no extra text).\n"
        "Task: You are given a SEED quiz question (MCQ) with options.\n"
        "1) Infer the *MICRO-TOPIC / chapter concept* strictly from the seed (e.g., 'Kinematics: acceleration from velocity-position relation', 'Myelinated neuron: saltatory conduction', etc.).\n"
        "2) Generate exactly 3 NEW MCQs ONLY from that same micro-topic (same concept family).\n"
        "   - Do NOT generate from the whole subject/book.\n"
        "   - Do NOT repeat the seed question or trivially rephrase it.\n"
        "   - Keep difficulty similar to admission-style questions.\n"
        "3) Each MCQ must have 4 options and exactly one correct answer.\n"
        "4) The explanation must be SHORT and bilingual: Bangla + English (1-2 lines each).\n"
        "5) Keep the question language consistent with the seed question language.\n\n"
        "Allowed major topics (for labeling only): Physics, Chemistry, Math, Biology, Bangla, English, General Knowledge, Humanities Skills.\n"
        "JSON format:\n"
        "{\n"
        "  \"topic\": \"<major topic>\",\n"
        "  \"microtopic\": \"<micro-topic inferred from seed>\",\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"question\": \"...\",\n"
        "      \"options\": [\"...\",\"...\",\"...\",\"...\"],\n"
        "      \"answer\": 1,\n"
        "      \"explanation\": \"Bangla line(s)\nEnglish line(s)\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Seed Question:\n{sq}\n\n"
        "Seed Options:\n" + "\n".join([f"{_safe_letter(i+1)}. {so[i]}" for i in range(len(so))])
    )

    raw = None
    last_err = None

    # 1) Prefer official Gemini REST for GenQuiz (fast, stable)
    if USE_GEMINI_REST_FOR_GENQUIZ and GEMINI_API_KEY:
        try:
            raw = call_gemini_text_rest(prompt, timeout_seconds=18, force_json=True)
        except Exception as e:
            last_err = e
            raw = None

    # 2) Fallback: Perplexity proxy (fast). If Perplexity fails, then try Gemini3 web.
    if not raw and USE_PERPLEXITY_FALLBACK:
        try:
            raw = query_ai(prompt)
        except Exception as e:
            last_err = e
            raw = None

    # 3) Final fallback: Gemini3 web session (can be slow/blocked)
    if not raw:
        try:
            raw = gemini3_solve(prompt)
        except Exception as e:
            last_err = e
            raw = None

    if not raw:
        raise RuntimeError(f"Quiz generation failed: {last_err or 'all backends unavailable'}")


    schema_hint = '{"microtopic":"<micro>","items":[{"question":"...","options":["...","...","...","..."],"answer":1,"explanation":"Bangla...\\nEnglish..."}]}'
    try:
        data = _extract_json_strict(raw)
    except Exception:
        repaired = _repair_to_json(raw, schema_hint=schema_hint, timeout_seconds=18)
        if not repaired:
            raise
        data = repaired

    if not isinstance(data, dict):
        raise RuntimeError("Quiz generation failed.")

    items = data.get("items", []) or []
    out: List[Dict[str, Any]] = []
    for it in items[:3]:
        q = str(it.get("question", "")).strip()
        opts = _normalize_options([str(x) for x in (it.get("options", []) or [])], max_n=4)
        ans = int(it.get("answer", 0) or 0)
        expl = str(it.get("explanation", "")).strip()

        # Verify with Perplexity (best-effort)
        try:
            ver = perplexity_solve_mcq_json(q, opts)
            vans = int((ver or {}).get("answer", 0) or 0)
            vexpl = str((ver or {}).get("explanation", "") or "").strip()
            if 1 <= vans <= 4:
                ans = vans
            if vexpl:
                expl = vexpl
        except Exception:
            pass

        if q and opts and 1 <= ans <= 4:
            out.append({"question": q, "options": opts, "answer": ans, "explanation": expl})

    return out[:3]


async def on_solver_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle solver button callbacks: solve:G/P/D:<token>"""
    if not update.callback_query:
        return
    q = update.callback_query
    await q.answer("Processing…", show_alert=False)

    data = (q.data or "").strip()
    m = re.match(r"^solve:([GPD]):([0-9a-f]{6,16})$", data)
    if not m:
        return
    model = m.group(1)  # G=Gemini, P=Perplexity, D=DeepSeek
    token = m.group(2)

    store = _pending_store(context)
    req = store.get(token)
    if not isinstance(req, dict):
        with contextlib.suppress(Exception):
            await q.edit_message_text("⚠️ This request has expired. Please send your question again.")
        return

    uid = int(req.get("uid") or 0)
    if q.from_user and q.from_user.id != uid:
        with contextlib.suppress(Exception):
            await q.answer("This is not your request.", show_alert=True)
        return

    payload = req.get("payload") or {}
    problem_text = str(payload.get("text") or "").strip()
    kind = str(req.get("kind") or "text").lower()

    # Show processing message
    with contextlib.suppress(Exception):
        await q.edit_message_text(
            ui_box_text("Solving", "Please wait… Processing your request.", emoji="⏳"),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

    try:
        if kind == "poll" and payload.get("question"):
            # MCQ solve
            question = str(payload.get("question", "")).strip()
            options = payload.get("options", [])
            
            if model == "G":
                result = await _run_blocking(_role_of(uid), gemini_solve_mcq_json, question, options)
            elif model == "P":
                result = await _run_blocking(_role_of(uid), perplexity_solve_mcq_json, question, options)
            elif model == "D":
                result = await _run_blocking(_role_of(uid), deepseek_solve_mcq_json, question, options)
            else:
                result = {"answer": 0, "confidence": 0, "explanation": "Unknown model", "why_not": {}}

            # --- NEW CODE START ---
            raw_expl = str(result.get('explanation', '') or "")
            clean_expl = clean_latex(raw_expl)  # এখানে ক্লিন করা হচ্ছে

            # Why not অপশনগুলোও ক্লিন করা দরকার
            raw_why_not = result.get("why_not", {}) or {}
            clean_why_not = {k: clean_latex(v) for k, v in raw_why_not.items()}

            msg_html = _format_user_poll_solution(
                question=question,
                options=options,
                model_ans=int(result.get("answer", 0) or 0),
                official_ans=int(payload.get("official_ans", 0) or 0),
                # এখানে ক্লিন করা টেক্সট পাঠানো হচ্ছে
                model_expl=f"[{['Gemini', 'Perplexity', 'DeepSeek'][['G','P','D'].index(model)]}]\n{clean_expl}".strip(),
                official_expl=str(payload.get("official_expl", "")).strip(),
                why_not=clean_why_not,
                conf=int(result.get("confidence", 0) or 0),
            )
            # --- NEW CODE END ---
            kb = _verify_kb(token, model, "poll")
        else:
            # Text solve
            if model == "G":
                answer = await _run_blocking(_role_of(uid), gemini_solve_text, problem_text)
            elif model == "P":
                answer = await _run_blocking(_role_of(uid), perplexity_solve_text, problem_text)
            elif model == "D":
                answer = await _run_blocking(_role_of(uid), deepseek_solve_text, problem_text)
            else:
                answer = "Unknown model"

            if is_admin(uid) or is_owner(uid):
                src_text = problem_text
                if looks_like_programming_request(src_text) or looks_like_programming_request(answer):
                    msg_html = f"<pre>{h(answer)}</pre>"
                else:
                    msg_html = h(answer)
            else:
                msg_html = h(answer)
            kb = _verify_kb(token, model, "text")

        with contextlib.suppress(Exception):
            await q.edit_message_text(msg_html, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        if q.message and getattr(q.message.chat, "type", "") in ("group", "supergroup"):
            asyncio.create_task(_auto_delete_after(context.bot, q.message.chat_id, [q.message.message_id], 300))

    except Exception as e:
        db_log("ERROR", "solver_callback_failed", {"user_id": uid, "model": model, "error": str(e)})
        with contextlib.suppress(Exception):
            await q.edit_message_text(
                ui_box_text("Solve Failed", str(e)[:180], emoji="❌"),
                parse_mode=ParseMode.HTML,
            )


async def on_genquiz_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    q = update.callback_query
    await q.answer("Processing…", show_alert=False)

    data = (q.data or "").strip()
    m = re.match(r"^genquiz:([0-9a-f]{6,16})$", data)
    if not m:
        return
    token = m.group(1)

    store = _pending_store(context)
    req = store.get(token)
    if not isinstance(req, dict):
        with contextlib.suppress(Exception):
            await q.edit_message_text("⚠️ This request has expired. Please send the quiz again.")
        return

    uid = int(req.get("uid") or 0)
    if q.from_user and q.from_user.id != uid:
        with contextlib.suppress(Exception):
            await q.answer("This is not your request.", show_alert=True)
        return

    # ONLY allow when the original content was a Poll/Quiz
    if str(req.get("kind") or "") != "poll":
        with contextlib.suppress(Exception):
            await q.answer("Generate Quiz is available only for quiz questions.", show_alert=True)
        return

    payload = req.get("payload") or {}
    seed_question = str(payload.get("question") or "").strip()
    seed_options = payload.get("options") or []

    qpfx = (get_setting("quiz_prefix", "প্রবাহ") or "প্রবাহ").strip()
    qlink = (get_setting("quiz_expl_link", "") or "").strip()

    # UI feedback
    with contextlib.suppress(Exception):
        await q.edit_message_text(
            ui_box_text("Generating Quizzes", "Please wait… Creating quizzes...", emoji="⏳"),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )


    try:
        chat_id = int(req.get("chat_id") or (q.message.chat_id if q.message else uid))
    
        # Generate until we have 3 quizzes (best effort, no feature loss)
        items = []
        seen_q = set()
        for _attempt in range(3):
            new_items = await _run_blocking(_role_of(uid), generate_quiz_items_gemini_then_verify, seed_question, seed_options)
            for it in (new_items or []):
                qt = str(it.get("question","") or "").strip()
                if not qt:
                    continue
                key = re.sub(r"\s+", " ", qt).lower()
                if key in seen_q:
                    continue
                seen_q.add(key)
                items.append(it)
                if len(items) >= 3:
                    break
            if len(items) >= 3:
                break
    
        if not items:
            raise RuntimeError("Quiz generation returned empty items.")
    
        items = items[:3]
    
        # Serialize sending per-chat to avoid flood-control cutting off the batch
        lock = _get_chat_lock(context, chat_id)
        async with lock:
            SEP = "\n\u200b"
            for it in items:
                qq = str(it["question"]).strip()
                opts = [str(x).strip() for x in it["options"]]
                ans = int(it["answer"])
                expl = _trim_expl_for_poll(str(it.get("explanation", "")), qlink)
    
                q_final = f"{qpfx}{SEP}{qq}".strip() if qpfx else qq
                if len(q_final) > 300:
                    q_final = q_final[:297] + "..."
    
                await _send_poll_with_retry(
                    context.bot,
                    chat_id=chat_id,
                    question=q_final,
                    options=opts,
                    is_anonymous=True,
                    type=Poll.QUIZ,
                    correct_option_id=ans - 1,
                    explanation=expl if expl else None,
                )
                await asyncio.sleep(0.35)
    
        done_msg = ui_box_text("Quizzes Generated", "Quizzes have been generated ✅", emoji="📊")
        with contextlib.suppress(Exception):
            await q.edit_message_text(done_msg, parse_mode=ParseMode.HTML)
    
    except Exception as e:
        db_log("ERROR", "generate_quiz_failed", {"user_id": uid, "error": str(e)})
        with contextlib.suppress(Exception):
            await q.edit_message_text(
                ui_box_text("Generate Quiz Failed", str(e)[:180], emoji="❌"),
                parse_mode=ParseMode.HTML,
            )


# ---------------------------
# UTIL
# ---------------------------
from datetime import timezone


def now_iso() -> str:
    return dt.datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def chunk_text(s: str, size: int = 3500) -> Iterable[str]:
    if not s:
        return []
    return (s[i:i + size] for i in range(0, len(s), size))


# ---------------------------
# Process / System stats helpers (owner dashboard)
# ---------------------------
def process_rss_mb() -> float:
    """Approximate RSS memory usage (MB) for this process. Works on Linux; graceful fallback."""
    try:
        # Linux: /proc/self/statm
        with open("/proc/self/statm", "r") as f:
            parts = f.read().strip().split()
        if len(parts) >= 2:
            rss_pages = int(parts[1])
            page_size = os.sysconf("SC_PAGE_SIZE")  # bytes
            return (rss_pages * page_size) / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource  # stdlib
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is KB on Linux, bytes on macOS. We assume Linux here.
        return float(rusage.ru_maxrss) / 1024.0
    except Exception:
        return 0.0

def fmt_mb(x: float) -> str:
    try:
        return f"{x:.1f} MB"
    except Exception:
        return "N/A"

def fmt_uptime() -> str:
    try:
        secs = int(time.time() - START_TIME)
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        if h > 0:
            return f"{h}h {m}m {s}s"
        if m > 0:
            return f"{m}m {s}s"
        return f"{s}s"
    except Exception:
        return "N/A"


# ---------------------------
# HTML helpers (safer + cleaner formatting)
# ---------------------------
def h(s: Any) -> str:
    """Escape text for Telegram HTML parse mode."""
    return html_escape.escape(str(s if s is not None else ""), quote=False)

def b(s: Any) -> str:
    return f"<b>{h(s)}</b>"

def code(s: Any) -> str:
    return f"<code>{h(s)}</code>"

def md_to_html_basic(s: str) -> str:
    """Convert a small subset of Markdown (**bold**, `code`) to Telegram-safe HTML."""
    if not s:
        return ""
    s = re.sub(r"`([^`]+)`", lambda m: f"<code>{html_escape.escape(m.group(1), quote=False)}</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: f"<b>{html_escape.escape(m.group(1), quote=False)}</b>", s)
    return s

def to_int(s: str) -> Optional[int]:
    try:
        return int(str(s).strip())
    except Exception:
        return None


def looks_like_programming_request(text: str) -> bool:
    s = (text or "").lower()
    keys = [
        "python", "javascript", "js", "java", "c++", "cpp", "c#", "php", "sql", "html", "css",
        "program", "code", "bug", "error", "traceback", "exception", "api", "function", "class",
        "loop", "array", "dict", "json", "regex", "algorithm", "query", "database", "telegram bot"
    ]
    return any(k in s for k in keys)


# ---------------------------
# DB
# ---------------------------
def db_connect() -> sqlite3.Connection:
    # SQLite tuning for multi-user / multi-update concurrency.
    # - WAL allows concurrent readers + a writer
    # - busy_timeout avoids 'database is locked' spikes under load
    # - longer connect timeout helps on slower disks (e.g., Pella)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        # If PRAGMA fails for any reason, continue with defaults (do not break bot).
        pass
    return conn


def _table_has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r["name"] for r in cur.fetchall()]
    return col in cols


def db_init() -> None:
    conn = db_connect()
    cur = conn.cursor()

    # Users: includes role + banned
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        role TEXT NOT NULL DEFAULT 'USER',
        first_name TEXT,
        username TEXT,
        is_banned INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """)

    # Migration: optional access flag
    if not _table_has_column(conn, "users", "can_view_all"):
        cur.execute("ALTER TABLE users ADD COLUMN can_view_all INTEGER NOT NULL DEFAULT 0")

    # Migration: optional vision (image→quiz) access flag
    if not _table_has_column(conn, "users", "can_use_vision"):
        cur.execute("ALTER TABLE users ADD COLUMN can_use_vision INTEGER NOT NULL DEFAULT 0")


    # Migration: per-user feature toggles (command-based)
    if not _table_has_column(conn, "users", "vision_mode_on"):
        cur.execute("ALTER TABLE users ADD COLUMN vision_mode_on INTEGER NOT NULL DEFAULT 0")
    if not _table_has_column(conn, "users", "solver_mode_on"):
        cur.execute("ALTER TABLE users ADD COLUMN solver_mode_on INTEGER NOT NULL DEFAULT 0")
    if not _table_has_column(conn, "users", "explain_mode_on"):
        cur.execute("ALTER TABLE users ADD COLUMN explain_mode_on INTEGER NOT NULL DEFAULT 0")

    # Migration: last seen timestamp (for active user stats)
    if not _table_has_column(conn, "users", "last_seen_at"):
        cur.execute("ALTER TABLE users ADD COLUMN last_seen_at TEXT")

    # Filters (per admin)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS filters (
        user_id INTEGER NOT NULL,
        phrase TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (user_id, phrase)
    )
    """)

    # Quiz buffer (per admin)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS quiz_buffer (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # Channels (added_by indicates who added it)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS channels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_chat_id INTEGER NOT NULL UNIQUE,
        title TEXT,
        prefix TEXT DEFAULT '',
        expl_link TEXT DEFAULT '',
        added_by INTEGER,
        created_at TEXT NOT NULL
    )
    """)

    # Admin post stats
    cur.execute("""
    CREATE TABLE IF NOT EXISTS admin_post_stats (
        admin_id INTEGER PRIMARY KEY,
        total_posts INTEGER NOT NULL DEFAULT 0,
        last_post_at TEXT
    )
    """)

    # Inbox / Tickets
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        student_name TEXT,
        status TEXT NOT NULL DEFAULT 'OPEN',
        created_at TEXT NOT NULL,
        last_update_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ticket_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id INTEGER NOT NULL,
        from_role TEXT NOT NULL, -- STUDENT or STAFF
        from_id INTEGER NOT NULL,
        message_text TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # Ban audit (who banned whom)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ban_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_user_id INTEGER NOT NULL,
        action TEXT NOT NULL, -- BAN or UNBAN
        by_user_id INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # Logs (lightweight)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS bot_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level TEXT NOT NULL,
        event TEXT NOT NULL,
        meta_json TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # Global settings defaults (non-breaking)
    settings_init_defaults(conn)


    conn.commit()
    conn.close()


def db_log(level: str, event: str, meta: Optional[Dict[str, Any]] = None) -> None:
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO bot_logs(level, event, meta_json, created_at) VALUES (?,?,?,?)",
            (level.upper(), event, json.dumps(meta or {}, ensure_ascii=False), now_iso()),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("db_log failed (ignored)")


# ---------------------------
# GLOBAL SETTINGS (Generate Quiz prefix / explanation link)
# ---------------------------
def settings_init_defaults(conn: sqlite3.Connection) -> None:
    """Ensure settings table exists + defaults (non-breaking)."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    ts = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    cur.execute("INSERT OR IGNORE INTO settings(key,value,updated_at) VALUES (?,?,?)", ("quiz_prefix", "প্রবাহ", ts))
    cur.execute("INSERT OR IGNORE INTO settings(key,value,updated_at) VALUES (?,?,?)", ("quiz_expl_link", "", ts))

def get_setting(key: str, default: str = "") -> str:
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        conn.close()
        if row and row["value"] is not None:
            return str(row["value"])
    except Exception:
        pass
    return default

def set_setting(key: str, value: str) -> None:
    conn = db_connect()
    cur = conn.cursor()
    ts = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    cur.execute(
        "INSERT INTO settings(key,value,updated_at) VALUES (?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value or "", ts),
    )
    conn.commit()
    conn.close()


# ---------------------------
# ROLES / PERMISSIONS
# ---------------------------
ROLE_OWNER = "OWNER"
ROLE_ADMIN = "ADMIN"
ROLE_USER = "USER"


def normalize_role(role: str) -> str:
    r = (role or "").upper().strip()
    return r if r in (ROLE_OWNER, ROLE_ADMIN, ROLE_USER) else ROLE_USER


def ensure_user(update: Update) -> None:
    if not update.effective_user:
        return
    u = update.effective_user
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT role FROM users WHERE user_id=?", (u.id,))
    row = cur.fetchone()
    if row is None:
        role = ROLE_OWNER if u.id == OWNER_ID else ROLE_USER
        cur.execute(
            "INSERT INTO users(user_id, role, first_name, username, is_banned, created_at, can_view_all, can_use_vision, last_seen_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (u.id, role, u.first_name, u.username, 0, now_iso(), 0, 0, now_iso()),
        )
    else:
        cur.execute(
            "UPDATE users SET first_name=?, username=?, last_seen_at=? WHERE user_id=?",
            (u.first_name, u.username, now_iso(), u.id),
        )
    conn.commit()
    conn.close()


def get_role(user_id: int) -> str:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT role FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row and row["role"]:
        return normalize_role(row["role"])
    return ROLE_OWNER if user_id == OWNER_ID else ROLE_USER


def _role_of(user_id: int) -> str:
    """Return role label for concurrency pools (OWNER/ADMIN/USER)."""
    try:
        return get_role(int(user_id or 0))
    except Exception:
        return ROLE_USER


# ---------------------------
# Per-chat locks (avoid flood + keep per chat ordering)
# ---------------------------
def _get_chat_lock(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> asyncio.Lock:
    """Get/create an asyncio.Lock for a chat_id stored in application bot_data."""
    try:
        locks = context.application.bot_data.get("_chat_locks")
        if not isinstance(locks, dict):
            locks = {}
            context.application.bot_data["_chat_locks"] = locks
        lock = locks.get(int(chat_id))
        if not isinstance(lock, asyncio.Lock):
            lock = asyncio.Lock()
            locks[int(chat_id)] = lock
        return lock
    except Exception:
        # last resort: a new lock (won't be shared)
        return asyncio.Lock()


async def _send_poll_with_retry(
    bot,
    *,
    chat_id: int,
    question: str,
    options: List[str],
    is_anonymous: bool = True,
    type: str = Poll.QUIZ,
    correct_option_id: int | None = None,
    explanation: str | None = None,
    allows_multiple_answers: bool = False,
    protect_content: bool = False,
    max_tries: int = 5,
):
    """send_poll wrapper with RetryAfter handling + small backoff."""
    last_err = None
    for i in range(max_tries):
        try:
            return await bot.send_poll(
                chat_id=chat_id,
                question=question,
                options=options,
                is_anonymous=is_anonymous,
                type=type,
                correct_option_id=correct_option_id,
                explanation=explanation,
                allows_multiple_answers=allows_multiple_answers,
                protect_content=protect_content,
            )
        except RetryAfter as e:
            await asyncio.sleep(float(getattr(e, "retry_after", 1.0)) + 0.2)
            last_err = e
        except TelegramError as e:
            # transient errors: retry a bit
            last_err = e
            await asyncio.sleep(0.4 * (2 ** i))
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.4 * (2 ** i))
    raise RuntimeError(str(last_err) if last_err else "send_poll failed")


def _deepseek_client():
    """Lazy DeepSeek client (OpenAI-compatible). Only used if DeepSeek is enabled."""
    if not globals().get("DEEPSEEK_API_KEY") or "sk-" not in str(globals().get("DEEPSEEK_API_KEY")):
        raise RuntimeError("DeepSeek API Key সেট করা নেই।")
    try:
        from openai import OpenAI  # optional dependency
    except Exception as e:
        raise RuntimeError("openai package missing for DeepSeek.") from e
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def is_owner(user_id: int) -> bool:
    return user_id == OWNER_ID or get_role(user_id) == ROLE_OWNER


def is_admin(user_id: int) -> bool:
    return get_role(user_id) in (ROLE_OWNER, ROLE_ADMIN)


def is_banned(user_id: int) -> bool:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT is_banned FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    return int(row["is_banned"] or 0) == 1


def set_ban(user_id: int, banned: bool) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET is_banned=? WHERE user_id=?", (1 if banned else 0, user_id))
    conn.commit()
    conn.close()


def audit_ban(by_user_id: int, target_user_id: int, action: str) -> None:
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ban_audit(target_user_id, action, by_user_id, created_at) VALUES (?,?,?,?)",
            (target_user_id, action, by_user_id, now_iso()),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("audit_ban failed (ignored)")


def can_view_all(user_id: int) -> bool:
    if is_owner(user_id):
        return True
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT can_view_all FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    return int(row["can_view_all"] or 0) == 1


def set_can_view_all(user_id: int, value: bool) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET can_view_all=? WHERE user_id=?", (1 if value else 0, user_id))
    conn.commit()
    conn.close()



def can_use_vision(user_id: int) -> bool:
    """Owner always can. Others need explicit grant (can_use_vision=1)."""
    if is_owner(user_id):
        return True
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT can_use_vision FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    return int(row["can_use_vision"] or 0) == 1


def set_can_use_vision(user_id: int, value: bool) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET can_use_vision=? WHERE user_id=?", (1 if value else 0, user_id))
    conn.commit()
    conn.close()



def vision_mode_on(user_id: int) -> bool:
    """Command-based toggle: if OFF, image→quiz handler ignores images."""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT vision_mode_on FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return int(row["vision_mode_on"] or 0) == 1 if row else False


def set_vision_mode_on(user_id: int, value: bool) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET vision_mode_on=? WHERE user_id=?", (1 if value else 0, user_id))
    conn.commit()
    conn.close()


def solver_mode_on(user_id: int) -> bool:
    """Command-based toggle: if ON (USER role), bot will solve incoming text."""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT solver_mode_on FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return int(row["solver_mode_on"] or 0) == 1 if row else False


def set_solver_mode_on(user_id: int, value: bool) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET solver_mode_on=? WHERE user_id=?", (1 if value else 0, user_id))
    conn.commit()
    conn.close()




def himusai_mode_on(user_id: int) -> bool:
    """Alias for admin/owner inbox AI-only mode.

    Historical builds used a separate HimusAI toggle name, but the current
    database stores this state in users.solver_mode_on. Keeping this alias
    prevents NameError in the active handlers and restores the previous flow
    where private admin/owner chats skip poll/text buffering when HimusAI is on.
    """
    return solver_mode_on(user_id)


def set_himusai_mode_on(user_id: int, value: bool) -> None:
    """Persist HimusAI mode using the existing solver_mode_on column."""
    set_solver_mode_on(user_id, value)

def explain_mode_on(user_id: int) -> bool:
    """Command-based toggle: if ON, quizzes include explanation; if OFF, quizzes are posted without explanation."""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT explain_mode_on FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return int(row["explain_mode_on"] or 0) == 1 if row else False


def set_explain_mode_on(user_id: int, value: bool) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET explain_mode_on=? WHERE user_id=?", (1 if value else 0, user_id))
    conn.commit()
    conn.close()



async def warn_unauthorized(update: Update, reason: str = "This action is not allowed for your role.") -> None:
    body = f"{h(reason)}\n\nIf you genuinely need access, contact the owner: {h(OWNER_CONTACT)}"
    await warn(update, "Unauthorized", body)


def require_admin(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        ensure_user(update)
        uid = update.effective_user.id if update.effective_user else 0
        if is_banned(uid):
            await safe_reply(update, f"🚫 Access denied. You are banned.\nContact: {OWNER_CONTACT}")
            return
        if not is_admin(uid):
            await warn_unauthorized(update, "Only Admin/Owner can use this feature.")
            return
        return await func(update, context)
    return wrapper


def require_owner(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        ensure_user(update)
        uid = update.effective_user.id if update.effective_user else 0
        if is_banned(uid):
            await safe_reply(update, f"🚫 Access denied. You are banned.\nContact: {OWNER_CONTACT}")
            return
        if not is_owner(uid):
            return
        return await func(update, context)
    return wrapper


# For message handlers: silently ignore non-admins (prevents double warnings)
def require_admin_silent(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        ensure_user(update)
        uid = update.effective_user.id if update.effective_user else 0
        if is_banned(uid):
            return
        if not is_admin(uid):
            return
        return await func(update, context)
    return wrapper



def require_vision(func):
    """Owner or granted users can use image→quiz feature."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        ensure_user(update)
        uid = update.effective_user.id if update.effective_user else 0
        if is_banned(uid):
            await safe_reply(update, f"🚫 Access denied. You are banned.\nContact: {OWNER_CONTACT}")
            return
        if not can_use_vision(uid):
            await warn_unauthorized(update, "Only the Owner (or explicitly granted staff) can use Image→Quiz.")
            return
        return await func(update, context)
    return wrapper


def require_vision_silent(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        ensure_user(update)
        uid = update.effective_user.id if update.effective_user else 0
        if is_banned(uid):
            return
        if not can_use_vision(uid):
            return
        return await func(update, context)
    return wrapper


# ---------------------------
# TELEGRAM SAFE SEND
# ---------------------------
async def safe_reply(update: Update, text: str) -> None:
    if not update.message:
        return
    for part in chunk_text(text, 3500):
        try:
            await update.message.reply_text(
                part,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except TelegramError as e:
            logger.exception("HTML parse error in safe_reply: %s", e)
            # Send plain text if HTML formatting fails
            with contextlib.suppress(Exception):
                await update.message.reply_text(
                    part,
                    disable_web_page_preview=True,
                )


async def safe_send_text(bot, chat_id: int, text: str, protect: bool = False) -> None:
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            protect_content=protect,
        )
    except RetryAfter as e:
        await asyncio.sleep(float(e.retry_after) + 0.2)
        with contextlib.suppress(Exception):
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                protect_content=protect,
            )
    except (Forbidden, TelegramError):
        pass
    except Exception:
        pass


async def safe_copy_message(bot, chat_id: int, from_chat_id: int, message_id: int, protect: bool = False) -> bool:
    """
    Copies a message without forward header.
    protect_content=True restricts forwarding/saving (Telegram feature).
    """
    try:
        await bot.copy_message(
            chat_id=chat_id,
            from_chat_id=from_chat_id,
            message_id=message_id,
            protect_content=protect,
        )
        return True
    except RetryAfter as e:
        await asyncio.sleep(float(e.retry_after) + 0.2)
        with contextlib.suppress(Exception):
            await bot.copy_message(
                chat_id=chat_id,
                from_chat_id=from_chat_id,
                message_id=message_id,
                protect_content=protect,
            )
            return True
    except (Forbidden, TelegramError):
        return False
    except Exception:
        return False



# ---------------------------
# Solver "searching" animation (Telegram-friendly)
# ---------------------------
async def _spinner_task(bot, chat_id: int, message_id: int) -> None:
    frames = [
        "🔎 Searching",
        "🔎 Searching.",
        "🔎 Searching..",
        "🔎 Searching...",
        "⏳ Preparing solution...",
    ]
    i = 0
    while True:
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=frames[i % len(frames)],
            )
        except Exception:
            pass
        i += 1
        await asyncio.sleep(0.9)

# -------------------------------
# Gemini3 single-file core (NO Flask)
# -------------------------------

def extract_snlm0e_token(html):
    snlm0e_patterns = [
        r'"SNlM0e":"([^"]+)"',
        r"'SNlM0e':'([^']+)'",
        r'SNlM0e["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'"FdrFJe":"([^"]+)"',
        r"'FdrFJe':'([^']+)'",
        r'FdrFJe["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'"cfb2h":"([^"]+)"',
        r"'cfb2h':'([^']+)'",
        r'cfb2h["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'at["\']?\s*[:=]\s*["\']([^"\']{50,})["\']',
        r'"at":"([^"]+)"',
        r'"token":"([^"]+)"',
        r'data-token["\']?\s*=\s*["\']([^"\']+)["\']',
    ]

    for pattern in snlm0e_patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            token = match.group(1)
            if len(token) > 20:
                return token
    return None


def extract_from_script_tags(html):
    soup = BeautifulSoup(html, 'html.parser')
    script_tags = soup.find_all('script')

    for script in script_tags:
        if script.string:
            script_content = script.string

            if 'SNlM0e' in script_content or 'FdrFJe' in script_content:
                token = extract_snlm0e_token(script_content)
                if token:
                    return token

            json_patterns = [
                r'\{[^}]*"[^"]*token[^"]*"[^}]*\}',
                r'\{[^}]*SNlM0e[^}]*\}',
                r'\{[^}]*FdrFJe[^}]*\}',
            ]

            for pattern in json_patterns:
                for match in re.finditer(pattern, script_content, re.IGNORECASE):
                    try:
                        json_obj = json.loads(match.group(0))
                        for _k, v in json_obj.items():
                            if isinstance(v, str) and len(v) > 50:
                                return v
                    except Exception:
                        continue
    return None


def extract_build_and_session_params(html):
    params = {}

    bl_patterns = [
        r'bl["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'"bl":"([^"]+)"',
        r'buildLabel["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        r'boq[_-]assistant[^"\']*_(\d+\.\d+[^"\']*)',
        r'/_/BardChatUi.*?bl=([^&"\']+)',
    ]
    for pattern in bl_patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            params['bl'] = match.group(1)
            break

    fsid_patterns = [
        r'f\.sid["\']?\s*[:=]\s*["\']?([^"\'&\s]+)',
        r'"fsid":"([^"]+)"',
        r'f\.sid=([^&"\']+)',
        r'sessionId["\']?\s*[:=]\s*["\']([^"\']+)["\']',
    ]
    for pattern in fsid_patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            params['fsid'] = match.group(1)
            break

    reqid_match = re.search(r'_reqid["\']?\s*[:=]\s*["\']?(\d+)', html)
    if reqid_match:
        params['reqid'] = int(reqid_match.group(1))

    if not params.get('bl'):
        params['bl'] = 'boq_assistant-bard-web-server_20251217.07_p5'
    if not params.get('fsid'):
        params['fsid'] = str(-1 * int(time.time() * 1000))
    if not params.get('reqid'):
        params['reqid'] = int(time.time() * 1000) % 1000000

    return params


# -------------------------------
# Gemini3 session cache (reduces latency)
# -------------------------------
_G3_CACHE = {"data": None, "ts": 0.0}
_G3_CACHE_TTL_SECONDS = 900  # 15 minutes


def scrape_fresh_session():
    session = requests.Session()
    url = 'https://gemini.google.com/app'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-site': 'none',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-dest': 'document',
        'upgrade-insecure-requests': '1',
        'cache-control': 'no-cache',
        'pragma': 'no-cache'
    }

    try:
        response = session.get(url, headers=headers, timeout=30)
        html = response.text

        cookies = {c.name: c.value for c in session.cookies}

        snlm0e = extract_snlm0e_token(html) or extract_from_script_tags(html)
        if not snlm0e:
            return None

        params = extract_build_and_session_params(html)

        return {
            'session': session,
            'cookies': cookies,
            'snlm0e': snlm0e,
            'bl': params['bl'],
            'fsid': params['fsid'],
            'reqid': params['reqid'],
            'html': html
        }
    except Exception:
        return None


def build_payload(prompt, snlm0e):
    escaped_prompt = prompt.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    session_id = uuid.uuid4().hex
    request_uuid = str(uuid.uuid4()).upper()

    payload_data = [
        [escaped_prompt, 0, None, None, None, None, 0],
        ["en-US"],
        ["", "", "", None, None, None, None, None, None, ""],
        snlm0e,
        session_id,
        None,
        [0],
        1,
        None,
        None,
        1,
        0,
        None,
        None,
        None,
        None,
        None,
        [[0]],
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        1,
        None,
        None,
        [4],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        [2],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        0,
        None,
        None,
        None,
        None,
        None,
        request_uuid,
        None,
        []
    ]

    payload_str = json.dumps(payload_data, separators=(',', ':'))
    escaped_payload = payload_str.replace('\\', '\\\\').replace('"', '\\"')

    return {'f.req': f'[null,"{escaped_payload}"]', '': ''}


def parse_streaming_response(response_text):
    lines = response_text.strip().split('\n')
    full_text = ""

    for line in lines:
        if not line or line.startswith(')]}'):
            continue
        try:
            if line.isdigit():
                continue
            data = json.loads(line)
            if isinstance(data, list) and len(data) > 0:
                if data[0][0] == "wrb.fr" and len(data[0]) > 2:
                    inner_json = data[0][2]
                    if inner_json:
                        parsed = json.loads(inner_json)
                        if isinstance(parsed, list) and len(parsed) > 4:
                            content_array = parsed[4]
                            if isinstance(content_array, list) and len(content_array) > 0:
                                first_item = content_array[0]
                                if isinstance(first_item, list) and len(first_item) > 0:
                                    response_id = first_item[0]
                                    if isinstance(response_id, str) and response_id.startswith('rc_'):
                                        if len(first_item) > 1 and isinstance(first_item[1], list):
                                            text_array = first_item[1]
                                            if len(text_array) > 0:
                                                text_content = text_array[0]
                                                if isinstance(text_content, str) and len(text_content) > len(full_text):
                                                    full_text = text_content
        except Exception:
            continue

    if full_text:
        full_text = full_text.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
    return full_text if full_text else None


def chat_with_gemini(prompt):
    start_time = time.time()
    # Reuse a cached Gemini session to reduce latency.
    scraped = None
    now_ts = time.time()
    try:
        cached = _G3_CACHE.get("data")
        if cached and (now_ts - float(_G3_CACHE.get("ts") or 0.0) < _G3_CACHE_TTL_SECONDS):
            scraped = cached
    except Exception:
        scraped = None

    if not scraped:
        scraped = scrape_fresh_session()
        if not scraped:
            return {'success': False, 'error': 'Failed to establish session with Gemini'}
        _G3_CACHE["data"] = scraped
        _G3_CACHE["ts"] = now_ts

    session = scraped['session']

    cookies = scraped['cookies']
    snlm0e = scraped['snlm0e']
    bl = scraped['bl']
    fsid = scraped['fsid']
    reqid = scraped['reqid']

    # refresh _reqid each request to avoid stale sessions
    reqid = int(time.time() * 1000) % 1000000
    base_url = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
    url = f"{base_url}?bl={bl}&f.sid={fsid}&hl=en-US&_reqid={reqid}&rt=c"

    payload = build_payload(prompt, snlm0e)
    cookie_str = '; '.join([f"{k}={v}" for k, v in cookies.items()])

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
        'x-same-domain': '1',
        'origin': 'https://gemini.google.com',
        'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'referer': 'https://gemini.google.com/',
        'Cookie': cookie_str
    }

    try:
        response = session.post(url, data=payload, headers=headers, timeout=20)
        if response.status_code != 200:
            return {'success': False, 'error': f'HTTP {response.status_code}'}

        result = parse_streaming_response(response.text)

        response_time = round(time.time() - start_time, 2)
        if result:
            return {
                'success': True,
                'response': result,
                'metadata': {
                    'response_time': f'{response_time}s',
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'model': 'gemini',
                    'character_count': len(result),
                    'word_count': len(result.split())
                }
            }
        return {'success': False, 'error': 'No response received from Gemini'}

    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': str(e)}

def gemini3_solve(prompt: str) -> str:
    """
    Single-file Gemini3 wrapper.
    Returns only the response text (same as your old usage).
    """
    res = chat_with_gemini(prompt)
    if isinstance(res, dict) and res.get("success") and res.get("response"):
        return str(res["response"]).strip()
    err = res.get("error") if isinstance(res, dict) else None
    raise RuntimeError(err or "Gemini3 solve failed.")


# ---------------------------
# MESSAGE HELPER FUNCTIONS
# ---------------------------
async def ok(update: Update, title: str, body: str) -> None:
    """Send success message using plain text."""
    msg = ui_box_text(title, body, emoji="✅")
    await safe_reply(update, msg)


async def ok_html(update: Update, title: str, body_html: str, emoji: str = "✅", footer_html: str = "") -> None:
    """Send success message with HTML formatting."""
    msg = ui_box_html(title, body_html, emoji=emoji, footer_html=footer_html)
    await safe_reply(update, msg)


async def warn(update: Update, title: str, body: str) -> None:
    """Send warning message using plain text."""
    msg = ui_box_text(title, body, emoji="⚠️")
    await safe_reply(update, msg)


async def warn_html(update: Update, title: str, body_html: str, emoji: str = "⚠️", footer_html: str = "") -> None:
    """Send warning message with HTML formatting."""
    msg = ui_box_html(title, body_html, emoji=emoji, footer_html=footer_html)
    await safe_reply(update, msg)


async def err(update: Update, title: str, body: str) -> None:
    """Send error message using plain text."""
    msg = ui_box_text(title, body, emoji="❌")
    await safe_reply(update, msg)


async def err_html(update: Update, title: str, body_html: str, emoji: str = "❌", footer_html: str = "") -> None:
    """Send error message with HTML formatting."""
    msg = ui_box_html(title, body_html, emoji=emoji, footer_html=footer_html)
    await safe_reply(update, msg)


async def info_html(update: Update, title: str, body_html: str, emoji: str = "ℹ️", footer_html: str = "") -> None:
    """Send informational message with HTML formatting."""
    msg = ui_box_html(title, body_html, emoji=emoji, footer_html=footer_html)
    await safe_reply(update, msg)


def reply_text_or_caption(update: Update) -> str:
    """
    Returns text from the replied message if present; otherwise empty string.
    """
    if not update.message or not update.message.reply_to_message:
        return ""
    m = update.message.reply_to_message
    return (m.text or m.caption or "").strip()


def parse_ticket_id_from_any_message(msg) -> Optional[int]:
    if not msg:
        return None
    text = "\n".join([
        str(getattr(msg, "text", "") or ""),
        str(getattr(msg, "caption", "") or ""),
    ]).strip()
    if not text:
        return None
    patterns = [
        r"(?:^|\n)\s*Ticket\s*[:#-]\s*(\d+)",
        r"(?:^|\n)\s*Ticket ID\s*[:#-]\s*(\d+)",
        r"/reply\s+(\d+)(?:\s|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


# ---------------------------
# CLEANER / PARSER
# ---------------------------
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)", re.IGNORECASE)
BRACKET_ANY_RE = re.compile(r"\[[^\]]*\]")  # removes [ ... ] anywhere
OPT_LINE_RE = re.compile(r"^\s*[\(\[]?[a-zA-Z0-9\u0980-\u09ff]+[\)\]\.]+\s+")


def get_user_filters(user_id: int) -> List[str]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT phrase FROM filters WHERE user_id=?", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [r["phrase"] for r in rows]


def clean_common(text: str, user_id: int) -> str:
    if not text:
        return ""

    for phrase in get_user_filters(user_id):
        if phrase:
            text = text.replace(phrase, "")

    text = BRACKET_ANY_RE.sub("", text)

    # Remove leading numbering: "62." "62)" "(62)" "৬২." "৬২)" "62।"
    text = re.sub(r"^\s*\(?[0-9\u09E6-\u09EF]+\)?\s*[\.\)\।]\s*", "", text)

    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def clean_explanation(text: str, user_id: int) -> str:
    if not text:
        return ""
    text = clean_common(text, user_id)
    # Remove common boilerplate headings
    text = re.sub(r"^\s*(Explanation\s*(for\s*question\s*\d+)?|Explain)\s*[:\-]*\s*", "", text, flags=re.IGNORECASE)
    text = MD_LINK_RE.sub("", text)
    text = BRACKET_ANY_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ==========================================
# ✅ LATEX CLEANER FUNCTION (Add this)
# ==========================================
def clean_latex(text: str) -> str:
    """LaTeX সিম্বল এবং ম্যাথ ফর্মেট ক্লিন করার শক্তিশালী ফাংশন"""
    if not text:
        return ""

    # ১. সাধারণ LaTeX ট্যাগ রিমুভ (যেমন \text{}, \mathrm{})
    text = re.sub(r"\\(text|mathrm|mathbf|it|bf)\{([^}]+)\}", r"\1", text)

    # ২. ভগ্নাংশ (Fractions) হ্যান্ডেল করা: \frac{a}{b} অথবা rac{a}{b} -> (a/b)
    # এটি \frac, frac, \rac, rac সব ধরবে
    text = re.sub(r"\\?f?rac\{([^}]+)\}\{([^}]+)\}", r"(\1/\2)", text)

    # ৩. গানিতিক চিহ্ন রিপ্লেস করা
    replacements = {
        r"\times": "×", r"\cdot": "·", r"\approx": "≈", r"\neq": "≠",
        r"\leq": "≤", r"\geq": "≥", r"\pm": "±", r"\mp": "∓",
        r"\rightarrow": "→", r"\leftarrow": "←", r"\infty": "∞",
        r"\degree": "°", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ",
        r"\theta": "θ", r"\pi": "π", r"\sigma": "σ", r"\Delta": "Δ",
        r"\omega": "ω", r"\lambda": "λ", r"\mu": "μ", r"\rho": "ρ"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # ৪. সুপারস্ক্রিপ্ট (Power) ঠিক করা (যেমন x^2 -> x²)
    superscripts = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
        "+": "⁺", "-": "⁻", "(": "⁽", ")": "⁾"
    }
    # ^2, ^3, ^-1 ইত্যাদি প্যাটার্ন খোঁজা
    def replace_sup(match):
        content = match.group(1).replace("{", "").replace("}", "")
        return "".join(superscripts.get(c, c) for c in content)
    
    text = re.sub(r"\^\{?([0-9+\-()]+)\}?", replace_sup, text)

    # ৫. সাবস্ক্রিপ্ট (Subscript) ঠিক করা (যেমন x_0 -> x₀)
    subscripts = {
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉"
    }
    def replace_sub(match):
        content = match.group(1).replace("{", "").replace("}", "")
        return "".join(subscripts.get(c, c) for c in content)

    text = re.sub(r"_\{?([0-9]+)\}?", replace_sub, text)

    # ৬. বাকি থাকা অদরকারী চিহ্ন মুছে ফেলা
    text = text.replace("$", "")   # ম্যাথ ডলার সাইন
    text = text.replace("\\", "")  # ব্যাকস্ল্যাশ
    
    # ৭. অতিরিক্ত স্পেস এবং লাইন ব্রেক ঠিক করা
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# ---------------------------
# INLINE EXPLANATION SPLITTER
# ---------------------------
INLINE_EXPL_RE = re.compile(
    r"""(?is)
    (.*?)                                   # question part (lazy)
    (?:\s*[\(\[]?\s*)?
    (?:explain|explanation|ব্যাখ্যা)         # marker
    (?:\s*[\)\]]?\s*)?
    \s*[:;\-–—]\s*                        # separator
    (.+)$                                   # explanation part
    """
)

def split_inline_explain(text: str) -> Tuple[str, str]:
    """
    If the question line contains something like:
      '... explain ; ...'  OR  '... Explanation: ...'  OR  '... ব্যাখ্যা: ...'
    then split it into (question, explanation).
    """
    t = (text or "").strip()
    if not t:
        return "", ""
    m = INLINE_EXPL_RE.match(t)
    if not m:
        return t, ""
    q = (m.group(1) or "").strip()
    e = (m.group(2) or "").strip()
    # Cleanup trailing separators
    q = re.sub(r"\s*[:;\-–—]+\s*$", "", q).strip()
    return q, e


def clean_option_text(line: str) -> str:
    line = re.sub(r"^\s*[\(\[]?[a-zA-Z0-9\u0980-\u09ff]+[\)\]\.]+\s+", "", line)
    return line.strip()


def split_blocks(text: str) -> List[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    parts = re.split(r"\n\s*\n+|\n\s*n\s*\n", text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p and p.strip()]


def parse_text_block(block: str, user_id: int) -> Optional[Dict[str, Any]]:
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    if not lines:
        return None

    # Explanation marker
    expl_idx = -1
    for i, ln in enumerate(lines):
        if re.match(r"^(Explanation|Note|ব্যাখ্যা)[:\-]", ln, re.IGNORECASE):
            expl_idx = i
            break

    explanation = ""
    if expl_idx != -1:
        raw_expl = "\n".join(lines[expl_idx:])
        raw_expl = re.sub(r"^(Explanation|Note|ব্যাখ্যা)[:\-]\s*", "", raw_expl, flags=re.IGNORECASE).strip()
        explanation = clean_explanation(raw_expl, user_id)
        lines = lines[:expl_idx]

    if not lines:
        return None

    question_parts: List[str] = []
    options: List[str] = []
    correct_answer = 0

    q0 = clean_common(lines[0], user_id)
    if q0:
        question_parts.append(q0)

    for ln in lines[1:]:
        ln = clean_common(ln, user_id)
        if not ln:
            continue

        if OPT_LINE_RE.match(ln):
            is_correct = False
            if ln.endswith("*"):
                is_correct = True
                ln = ln[:-1].strip()

            opt = clean_option_text(ln)
            options.append(opt)
            if is_correct:
                correct_answer = len(options)
        else:
            question_parts.append(ln)

    final_question = " ".join([p for p in question_parts if p]).strip()
    final_question = clean_common(final_question, user_id)  # ensure serial/brackets removed fully

    # If user wrote explanation inline inside the question (e.g. "... explain ; ..."),
    # move it to the explanation field (fixes Telegram quiz explanation showing in question).
    q2, expl2 = split_inline_explain(final_question)
    if expl2:
        final_question = q2.strip()
        cleaned = clean_explanation(expl2, user_id)
        if cleaned:
            explanation = (explanation + "\n" + cleaned).strip() if explanation else cleaned

    if not final_question:
        return None

    opts = options + [""] * (5 - len(options))
    return {
        "questions": final_question,
        "option1": opts[0], "option2": opts[1], "option3": opts[2],
        "option4": opts[3], "option5": opts[4],
        "answer": int(correct_answer) if correct_answer else 0,
        "explanation": explanation,
        "type": 1, "section": 1,
    }


# ---------------------------
# GEMINI VISION (REST) — Image → MCQ JSON → Buffer payloads
# ---------------------------
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json_strict(text: str) -> Dict[str, Any]:
    """Strict JSON parser with a safe fallback."""
    raw = (text or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        m = _JSON_OBJ_RE.search(raw)
        if not m:
            raise ValueError("Model did not return valid JSON.")
        return json.loads(m.group(0))

def _repair_to_json(raw_text: str, schema_hint: str = "", timeout_seconds: int = 18) -> Optional[Dict[str, Any]]:
    """Best-effort: ask a backend to convert a messy output into strict JSON."""
    raw = (raw_text or "").strip()
    if not raw:
        return None
    repair_prompt = (
        "Return STRICT JSON only (no markdown, no extra text).\n"
        "Your job: convert the following content into VALID JSON that matches this schema hint.\n"
        "Schema hint (must follow):\n"
        f"{schema_hint.strip()}\n\n"
        "Content to convert:\n"
        f"{raw}\n"
    )
    # Prefer Gemini REST with JSON mime
    if GEMINI_API_KEY:
        try:
            fixed = call_gemini_text_rest(repair_prompt, timeout_seconds=timeout_seconds, force_json=True)
            data = json.loads(fixed.strip())
            return data if isinstance(data, dict) else None
        except Exception:
            pass
    # Fallback: Perplexity proxy (best effort)
    if USE_PERPLEXITY_FALLBACK:
        try:
            fixed = query_ai(repair_prompt)
            if fixed:
                data = _extract_json_strict(fixed)
                return data if isinstance(data, dict) else None
        except Exception:
            pass
    return None





def list_gemini_models() -> Dict[str, Any]:
    """Return the raw ListModels response."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set inside the code.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    r = _requests_with_retries(requests.get, url, timeout=GEMINI_TIMEOUT_SECONDS, max_tries=2)
    if r.status_code != 200:
        raise RuntimeError(f"ListModels failed {r.status_code}: {r.text[:400]}")
    return r.json()

def pick_working_model(preferred: str) -> str:
    """Pick a model that supports generateContent. Prefer Flash then Pro."""
    pref = (preferred or "").strip()
    if pref and not pref.startswith("models/"):
        pref = "models/" + pref
    data = list_gemini_models()
    models = data.get("models", []) or []

    def supports_generate(m: Dict[str, Any]) -> bool:
        methods = m.get("supportedGenerationMethods", []) or []
        return any(str(x).lower() == "generatecontent" for x in methods)

    candidates = [m for m in models if supports_generate(m)]
    names = [m.get("name","") for m in candidates if m.get("name")]
    if pref and pref in names:
        return pref

    flash = [n for n in names if "flash" in n.lower()]
    pro = [n for n in names if "pro" in n.lower()]
    if flash:
        return flash[0]
    if pro:
        return pro[0]
    if names:
        return names[0]

    raise RuntimeError("No generateContent-capable models found for this API key/project.")

def call_gemini_vision_rest(image_path: str, prompt: str, force_json: bool = True) -> str:
    """Calls Gemini Vision model using AI Studio API key. Returns model text (expected JSON)."""
    
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set inside the code.")

    model = (GEMINI_MODEL_VISION or "").strip()
    if not model:
        raise RuntimeError("GEMINI_MODEL_VISION is empty.")

    if not model.startswith("models/"):
        model = "models/" + model

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"{model}:generateContent?key={GEMINI_API_KEY}"
    )

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_b64
                    }
                },
            ],
        }],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": 2048,
            **({"responseMimeType": "application/json"} if force_json else {}),
        },
    }

    r = _requests_with_retries(
        requests.post,
        url,
        json_payload=payload,
        timeout=GEMINI_TEXT_TIMEOUT_SECONDS,
        max_tries=3,
    )

    if r.status_code == 404:
        # Model fallback
        picked = pick_working_model(model)
        model = picked
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"{model}:generateContent?key={GEMINI_API_KEY}"
        )

        r = _requests_with_retries(
            requests.post,
            url,
            json_payload=payload,
            timeout=GEMINI_TEXT_TIMEOUT_SECONDS,
            max_tries=3,
        )

    if r.status_code != 200:
        raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:400]}")

    data = r.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise RuntimeError("Unexpected Gemini response format (no candidates/content/parts/text).")

def gemini_extract_mcq_from_image_rest(image_path: str) -> List[Dict[str, Any]]:
    """Returns a list of buffer payload dicts."""
    prompt = (
        "You are an exam question extractor.\n"
        "From the given image, extract ALL MCQ questions.\n"
        "Return STRICT JSON only (no markdown, no commentary, no extra text).\n\n"
        "Output format:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "questions": "...",\n'
        '      "option1": "...",\n'
        '      "option2": "...",\n'
        '      "option3": "...",\n'
        '      "option4": "...",\n'
        '      "option5": "",\n'
        '      "answer": 1,\n'
        '      "explanation": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Keep Bengali text exactly as-is.\n"
        "- If an option is missing, keep it \"\".\n"
        "- answer must be 1-5. If unknown, set 0.\n"
        "- explanation must be short, exam-style (1-3 short lines).\n"
        "- Do NOT invent questions that are not present.\n""- If the image shows the correct option (e.g., marked/ticked/underlined or written), you MUST set answer accordingly (1-5).\n""- Never output generic placeholders like 'Explanation for question X'.\n"
    )

    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            raw = call_gemini_vision_rest(image_path, prompt)
            data = _extract_json_strict(raw)
            items = data.get("items", []) or []
            out: List[Dict[str, Any]] = []
            for it in items:
                out.append({
                    "questions": str(it.get("questions", "")).strip(),
                    "option1": str(it.get("option1", "")).strip(),
                    "option2": str(it.get("option2", "")).strip(),
                    "option3": str(it.get("option3", "")).strip(),
                    "option4": str(it.get("option4", "")).strip(),
                    "option5": str(it.get("option5", "")).strip(),
                    "answer": int(it.get("answer", 0) or 0),
                    "explanation": str(it.get("explanation", "")).strip(),
                    "type": 1,
                    "section": 1,
                })
            out = [x for x in out if x.get("questions")]
            return out
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Image extraction failed: {last_err}")



# ---------------------------
# GEMINI TEXT (REST) — Problem Solving Chat
# ---------------------------
def call_gemini_text_rest(prompt: str, timeout_seconds: int = GEMINI_TEXT_TIMEOUT_SECONDS, *, force_json: bool = False) -> str:
    """Calls Gemini text model. Returns plain text."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set inside the code.")

    model = (GEMINI_MODEL_TEXT or "").strip()
    if not model:
        raise RuntimeError("GEMINI_MODEL_TEXT is empty.")
    if not model.startswith("models/"):
        model = "models/" + model

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"{model}:generateContent?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}],
        }],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": 2048,
        },
    }
    if force_json:
        payload.setdefault("generationConfig", {})["responseMimeType"] = "application/json"


    r = _requests_with_retries(requests.post, url, json_payload=payload, timeout=timeout_seconds, max_tries=3)
    if r.status_code == 404:
        # Model not found / not supported: pick a working model dynamically and retry once
        picked = pick_working_model(model)
        model = picked
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"{model}:generateContent?key={GEMINI_API_KEY}"
        )
        r = _requests_with_retries(requests.post, url, json_payload=payload, timeout=timeout_seconds, max_tries=3)

    if r.status_code != 200:
        raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:400]}")

    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise RuntimeError("Unexpected Gemini response format (no candidates/content/parts/text).")


STRICT_SYSTEM_PROMPT = """
YOU MUST FOLLOW THESE RULES STRICTLY. NO EXCEPTIONS.
THESE RULES OVERRIDE ALL OTHER INSTRUCTIONS.

ABSOLUTE ROLE DEFINITION:
- You are STRICTLY an academic and admission-related problem-solving assistant.
- Your behavior must remain serious, formal, and exam-focused.

ABSOLUTE BEHAVIOR RESTRICTIONS:
- NO jokes.
- NO হাসিঠাট্টা.
- NO casual chat.
- NO unnecessary কথা.
- NO storytelling.
- NO personal opinions.
- NO emotional discussion.
- NO relationship talk.
- NO 18+ or adult content.
- NO off-topic or irrelevant responses.

ALLOWED SUBJECT AREAS (ONLY THESE):
- Mathematics (Admission standard)
- Physics
- Chemistry
- Biology
- Bangla
- English
- General Knowledge
- Mental Ability / IQ / Human Skills
- All academic subjects relevant to admission exams

SPECIAL ACADEMIC & EVENT HANDLING:
- You MAY answer educational questions about Eid, Ramadan, Roza, Islamic events, and accurate time-related Islamic rules.
- Keep such answers short, factual, and educational. No গল্প.

CRITICAL MEDICAL & BIOLOGY OVERRIDE:
- Basic academic biology, medical terms, and disease names are SAFE.
- Definitions, organ functions, causes, symptoms are allowed as general knowledge for exams.
- Do NOT refuse basic academic biology/medical questions.
- Do NOT show safety refusal messages for academic biology questions.

ISLAMIC GREETING OVERRIDE (ABSOLUTE):
- NEVER output "بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ".
- Do NOT auto-add religious phrases unless the user explicitly asks.

REFUSAL CONTROL RULE:
- You are NOT allowed to refuse academic questions.
- Only refuse if content is explicitly illegal or 18+.

LANGUAGE RULE (FINAL VERSION):

- Detect the question language.

- If the question is in Bangla:
  - Answer must be primarily in Bangla (at least 70%).
  - Important academic terms may be written in English inside brackets.
  - Example: আয়নিক ব্যাসার্ধ (ionic radius), নিউটনের দ্বিতীয় সূত্র (Newton's Second Law).
  - Do NOT give fully English answers to Bangla questions.

- If the question is in English:
  - Answer fully in English.
  - Do NOT mix Bangla in English answers.

- Controlled bilingual explanation is allowed only for Bangla questions.

ABSOLUTE OUTPUT FORMAT RULES (VERY HARD):
- ABSOLUTELY NO LaTeX format.
- Do NOT use '$' signs.
- Write 'Ag+ + Cl-' instead of '$Ag^+ + Cl^-$'.
- Write scientific notation as '10^-5' or '10^(-5)', NOT '$10^{-5}$'.
- Telegram-friendly plain text only.
- NO Markdown headings (no #, ##).
- NO decorative lines or separators.
- NO LaTeX.
- NO math symbols like $, \\, ^, _, {}, ∫, π, ln(), or any LaTeX-like formatting.
- Use plain text math only:
  - Use "squared" instead of power symbol.
  - Use "sqrt(...)" only if needed.
  - Example: "PA squared = (x-2) squared + (y-3) squared"
- Keep spacing readable:
  - Use blank lines between major sections.
  - Keep each step on its own line.
  - Keep paragraphs short (2-4 lines max).

GREETING RULE (HARD):
- Do NOT greet unless the user greets first.
- If user says only "Hi/Hello" or no academic question, reply ONLY:
  "অনুগ্রহ করে আপনার প্রশ্নটি পাঠান।"

BOT INTRODUCTION RULE (VERY HARD):
- NEVER introduce yourself.
- NEVER talk about the bot, assistant, system, AI, mission, or background.

EXCEPTION (ONLY ONE CASE):
IF AND ONLY IF the user explicitly asks about the bot (who are you / about bot / developer / প্রবাহ বট / এই বটটা কি / তোমার ডেভেলপার কে):
- Give a VERY SHORT introduction, then answer the question.
- Bangla query -> Bangla intro:
  "এটি Probaho বট সহকারী। এটি ভর্তি পরীক্ষার সমস্যা সমাধানে সহায়তা করে। Developer: @Your_Himus। শিক্ষামূলক উদ্দেশ্যে তৈরি।"
- English query -> English intro (short).
- Do NOT repeat the intro again in the same conversation.

QUESTION GENERATION MODE (VERY IMPORTANT):
If user asks for questions only (প্রশ্ন দাও / generate questions / practice questions / এডমিশন প্রশ্ন বানাও):
- ONLY generate questions.
- NO answers, NO explanations.

SOLVING MODE (MANDATORY FORMAT):
When solving any problem, the answer MUST follow EXACTLY this structure with blank lines:

1) Answer:
- One line: the correct option/value only.

(blank line)

2) Explanation:
- Step 1:
- Step 2:
- Step 3:
(Show hand-calculation steps when needed, exam style, no unnecessary text.)

(blank line)

3) Final Answer:
- One line repeating the final answer.

QUIZ/MCQ DISPLAY RULE (IF OPTIONS GIVEN):
- If the user included options, repeat them in a clean list.
- Keep one blank line between Question and Options.
- After giving Answer, show Explanation, then Final Answer.
- Do NOT show "Confidence" or percentages.

SECOND-OPINION TEXT (PROFESSIONAL):
- If asked to suggest verification, write ONE short line only:
  "যাচাই করতে চাইলে নিচের বাটন থেকে অন্য মডেল ব্যবহার করুন।"
(English version: "For verification, use another model button below.")

STRICT CONSISTENCY RULE:
- Do NOT contradict yourself in the same response.
- If you correct an earlier mistake, acknowledge it clearly and provide the correct answer.
"""

def gemini_solve_text(problem_text: str) -> str:
    prompt = (
        STRICT_SYSTEM_PROMPT
        + "\n\nUser Message:\n"
        + (problem_text or "").strip()
    )

    # 1) Official Gemini REST (fast & stable when quota allows)
    if USE_OFFICIAL_GEMINI_REST_FALLBACK and GEMINI_API_KEY:
        try:
            return call_gemini_text_rest(prompt, timeout_seconds=18).strip()
        except RateLimitError:
            # quota/rate-limited → immediately fallback to other backends
            pass
        except Exception:
            pass

    # 2) Perplexity (usually fast)
    if USE_PERPLEXITY_FALLBACK:
        try:
            alt = query_ai(prompt)
            if alt:
                return alt.strip()
        except Exception:
            pass

    # 3) Gemini3 web session (last resort; can be slow/blocked)
    try:
        return gemini3_solve(prompt)
    except Exception:
        pass

    raise RuntimeError("Solver failed: all backends unavailable.")


def _safe_letter(i: int) -> str:
    return {1:"A", 2:"B", 3:"C", 4:"D", 5:"E"}.get(int(i or 0), "")

def _poll_official_answer(poll: Poll) -> int:
    """
    Returns 1-10 if official correct_option_id exists, else 0.
    Note: forwarded quizzes often hide correct_option_id (Telegram limitation).
    """
    try:
        if poll and poll.type == "quiz" and poll.correct_option_id is not None:
            return int(poll.correct_option_id) + 1
    except Exception:
        pass
    return 0

def gemini_solve_mcq_json(question: str, options: List[str]) -> Dict[str, Any]:
    """
    Robust MCQ solver that returns strict JSON:
      {"answer": 1-5, "confidence": 0-100, "explanation": "...", "why_not": {"A":"..","B":"..",...}}
    Uses Gemini3 first, then Gemini REST fallback.
    """
    q = (question or "").strip()
    opts = [(o or "").strip() for o in (options or []) if (o or "").strip()]
    opts = opts[:5]  # consistent A-E mapping
    if len(opts) < 2:
        raise ValueError("Not enough options to solve.")
    opt_lines = "\n".join([f"{_safe_letter(i+1)}. {opts[i]}" for i in range(len(opts))])

    prompt = (
        "Return STRICT JSON only. No markdown. No extra text.\n\n"
        "Task: Solve the following MCQ and pick the correct option.\n"
        "Rules:\n"
        "- answer must be 1-5 (A=1,B=2,C=3,D=4,E=5). If unsure, pick the best option.\n"
        "- explanation: detailed step-by-step (8–12 lines) (Bangla if question is Bangla).\n"
        "- why_not: short 1-5 line reason for each wrong option that exists.\n"
        "- confidence: 0-100 integer.\n\n"
        f"Question:\n{q}\n\nOptions:\n{opt_lines}\n\n"
        "JSON format:\n"
        "{\n"
        "  \"answer\": 1,\n"
        "  \"confidence\": 0,\n"
        "  \"explanation\": \"....\",\n"
        "  \"why_not\": {\"A\":\"..\",\"B\":\"..\",\"C\":\"..\",\"D\":\"..\",\"E\":\"..\"\"}\n"
        "}"
    )

    # 1) Gemini3 (may return JSON as text)
    try:
        raw = gemini3_solve(prompt)
        data = _extract_json_strict(raw)
        return data if isinstance(data, dict) else {"answer": 0, "confidence": 0, "explanation": str(raw)[:400], "why_not": {}}
    except Exception:
        pass

    # 2) Optional REST fallback (disabled by default)
    if USE_OFFICIAL_GEMINI_REST_FALLBACK and GEMINI_API_KEY:
        raw2 = call_gemini_text_rest(prompt)
        data2 = _extract_json_strict(raw2)
        if not isinstance(data2, dict):
            raise RuntimeError("MCQ solver returned non-JSON response.")
        return data2

    # 3) Fallback: Perplexity (ask it to return STRICT JSON)
    if USE_PERPLEXITY_FALLBACK:
        p2 = (
            "Return STRICT JSON only (no markdown).\n"
            "Solve the MCQ and respond in this JSON format exactly:\n"
            "{\"answer\":1,\"confidence\":0,\"explanation\":\"...\",\"why_not\":{\"A\":\"..\",\"B\":\"..\",\"C\":\"..\",\"D\":\"..\",\"E\":\"..\"}}\n\n"
            f"Question:\n{q}\n\nOptions:\n{opt_lines}\n"
        )
        alt = query_ai(p2)
        if alt:
            try:
                data3 = _extract_json_strict(alt)
                if isinstance(data3, dict) and int(data3.get("answer",0) or 0) > 0:
                    return data3
            except Exception:
                # If it doesn't return JSON, provide a safe wrapper
                return {"answer": 0, "confidence": 0, "explanation": (alt[:1800] if isinstance(alt,str) else str(alt)[:1800]), "why_not": {}}
    raise RuntimeError("MCQ solver failed: Gemini3 unavailable and REST fallback disabled.")

def _format_user_poll_solution(
    question: str,
    options: List[str],
    model_ans: int,
    official_ans: int,
    model_expl: str,
    official_expl: str,
    why_not: Dict[str, str],
    conf: int
) -> str:
    """
    Telegram-HTML safe formatted output.
    """
    q = h(question or "")
    opts = [(o or "").strip() for o in (options or []) if (o or "").strip()]
    opts = opts[:5]

    lines = []
    lines.append("<b>📊 Quiz Solution</b>")
    lines.append(f"\n<b>Question:</b>\n{q}")
    lines.append("\n<b>Options:</b>")
    for i, o in enumerate(opts, start=1):
        lines.append(f"• <b>{_safe_letter(i)}</b>) {h(o)}")

    if 1 <= int(model_ans or 0) <= len(opts):
        lines.append(f"\n<b>✅ Ai Response:</b> <b>{_safe_letter(model_ans)}</b>) {h(opts[model_ans-1])}")

    else:
        lines.append(f"\n<b>✅ Ai Response:</b> <b>{h(_safe_letter(model_ans)) or 'N/A'}</b>")


    if official_ans > 0 and official_ans <= len(opts):
        match = (official_ans == model_ans)
        tag = "✅ Match" if match else "❌ Mismatch"
        lines.append(f"<b>📌 Given Answer:</b> <b>{_safe_letter(official_ans)}</b>) {h(opts[official_ans-1])}  <i>({tag})</i>")
    else:
        lines.append("<b>📌 Given Answer:</b> <i>Not available (forwarded quizzes often hide the correct answer).</i>")

    if model_expl:
        lines.append(f"\n<b>Explanation (Solved):</b>\n{h(model_expl)}")
    if official_expl:
        lines.append(f"\n<b>Explanation (From Quiz):</b>\n{h(official_expl)}")

    if why_not:
        wn_lines = []
        for k in ["A","B","C","D","E"]:
            v = (why_not or {}).get(k)
            if v:
                wn_lines.append(f"• <b>{h(k)}</b>: {h(v)}")
        if wn_lines:
            lines.append("\n<b>Why other options are wrong:</b>\n" + "\n".join(wn_lines))

    return "\n".join(lines).strip()





# ---------------------------
# BUFFER
# ---------------------------
def buffer_count(user_id: int) -> int:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM quiz_buffer WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return int(row["c"]) if row else 0


def buffer_add(user_id: int, payload: Dict[str, Any]) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO quiz_buffer(user_id, payload_json, created_at) VALUES (?,?,?)",
        (user_id, json.dumps(payload, ensure_ascii=False), now_iso()),
    )
    conn.commit()
    conn.close()


def buffer_list(user_id: int, limit: int = 9999) -> List[Tuple[int, Dict[str, Any]]]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, payload_json FROM quiz_buffer WHERE user_id=? ORDER BY id ASC LIMIT ?",
        (user_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append((int(r["id"]), json.loads(r["payload_json"])))
    return out


def buffer_clear(user_id: int) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM quiz_buffer WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def buffer_remove_ids(user_id: int, ids: List[int]) -> None:
    if not ids:
        return
    conn = db_connect()
    cur = conn.cursor()
    q = ",".join("?" for _ in ids)
    cur.execute(f"DELETE FROM quiz_buffer WHERE user_id=? AND id IN ({q})", [user_id, *ids])
    conn.commit()
    conn.close()


# ---------------------------
# CHANNELS
# ---------------------------
@dataclass
class ChannelRow:
    id: int
    channel_chat_id: int
    title: str
    prefix: str
    expl_link: str
    added_by: int


def channel_add(channel_chat_id: int, title: str, added_by: int) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO channels(channel_chat_id, title, prefix, expl_link, added_by, created_at)
        VALUES (?,?,?,?,?,?)
        """,
        (channel_chat_id, title or "", "", "", added_by, now_iso()),
    )
    conn.commit()
    conn.close()


def channel_list_for_user(requester_id: int) -> List[ChannelRow]:
    conn = db_connect()
    cur = conn.cursor()

    if can_view_all(requester_id):
        cur.execute("SELECT id, channel_chat_id, title, prefix, expl_link, added_by FROM channels ORDER BY id ASC")
    else:
        cur.execute(
            "SELECT id, channel_chat_id, title, prefix, expl_link, added_by FROM channels WHERE added_by=? ORDER BY id ASC",
            (requester_id,),
        )

    rows = cur.fetchall()
    conn.close()
    return [
        ChannelRow(
            id=int(r["id"]),
            channel_chat_id=int(r["channel_chat_id"]),
            title=r["title"] or "",
            prefix=r["prefix"] or "",
            expl_link=r["expl_link"] or "",
            added_by=int(r["added_by"] or 0),
        )
        for r in rows
    ]


def channel_get_by_id(cid: int) -> Optional[ChannelRow]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT id, channel_chat_id, title, prefix, expl_link, added_by FROM channels WHERE id=?", (cid,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return ChannelRow(
        id=int(r["id"]),
        channel_chat_id=int(r["channel_chat_id"]),
        title=r["title"] or "",
        prefix=r["prefix"] or "",
        expl_link=r["expl_link"] or "",
        added_by=int(r["added_by"] or 0),
    )


def channel_get_by_id_for_user(requester_id: int, cid: int) -> Optional[ChannelRow]:
    ch = channel_get_by_id(cid)
    if not ch:
        return None
    if can_view_all(requester_id):
        return ch
    return ch if ch.added_by == requester_id else None


def channel_remove(cid: int) -> bool:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM channels WHERE id=?", (cid,))
    ok = cur.rowcount > 0
    conn.commit()
    conn.close()
    return ok


def channel_set_prefix(cid: int, prefix: str) -> bool:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE channels SET prefix=? WHERE id=?", (prefix or "", cid))
    ok = cur.rowcount > 0
    conn.commit()
    conn.close()
    return ok


def channel_set_expl_link(cid: int, link: str) -> bool:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE channels SET expl_link=? WHERE id=?", (link or "", cid))
    ok = cur.rowcount > 0
    conn.commit()
    conn.close()
    return ok


# ---------------------------
# ADMIN POST STATS
# ---------------------------
def inc_admin_post(admin_id: int, count: int) -> None:
    if count <= 0:
        return
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT total_posts FROM admin_post_stats WHERE admin_id=?", (admin_id,))
    r = cur.fetchone()
    if r is None:
        cur.execute(
            "INSERT INTO admin_post_stats(admin_id, total_posts, last_post_at) VALUES (?,?,?)",
            (admin_id, count, now_iso()),
        )
    else:
        cur.execute(
            "UPDATE admin_post_stats SET total_posts=total_posts+?, last_post_at=? WHERE admin_id=?",
            (count, now_iso(), admin_id),
        )
    conn.commit()
    conn.close()


# ---------------------------
# INBOX / TICKETS
# ---------------------------
def ticket_open(student_id: int, student_name: str) -> int:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tickets(student_id, student_name, status, created_at, last_update_at) VALUES (?,?,?,?,?)",
        (student_id, student_name or "", "OPEN", now_iso(), now_iso()),
    )
    tid = cur.lastrowid
    conn.commit()
    conn.close()
    return int(tid)


def ticket_find_open_by_student(student_id: int) -> Optional[int]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM tickets WHERE student_id=? AND status='OPEN' ORDER BY id DESC LIMIT 1",
        (student_id,),
    )
    r = cur.fetchone()
    conn.close()
    return int(r["id"]) if r else None


def ticket_add_msg(ticket_id: int, from_role: str, from_id: int, text: str) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO ticket_messages(ticket_id, from_role, from_id, message_text, created_at) VALUES (?,?,?,?,?)",
        (ticket_id, from_role, from_id, text, now_iso()),
    )
    cur.execute("UPDATE tickets SET last_update_at=? WHERE id=?", (now_iso(), ticket_id))
    conn.commit()
    conn.close()


def ticket_get(ticket_id: int) -> Optional[sqlite3.Row]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tickets WHERE id=?", (ticket_id,))
    r = cur.fetchone()
    conn.close()
    return r


def ticket_close(ticket_id: int) -> bool:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE tickets SET status='CLOSED', last_update_at=? WHERE id=?", (now_iso(), ticket_id))
    ok = cur.rowcount > 0
    conn.commit()
    conn.close()
    return ok


def list_staff_ids() -> List[int]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE role IN ('OWNER','ADMIN')")
    rows = cur.fetchall()
    conn.close()
    ids = [int(r["user_id"]) for r in rows]
    if OWNER_ID not in ids:
        ids.append(OWNER_ID)
    return sorted(set(ids))


# ---------------------------
# HELP TEXT (ROLE-BASED, polished)
# ---------------------------
def help_for_role(role: str, requester_id: int) -> str:
    """
    Generate help text for a user role. Pure Telegram-HTML (no Markdown, no <blockquote>).
    """
    role = normalize_role(role)

    access_note = ""
    if role == ROLE_ADMIN and can_view_all(requester_id):
        access_note = "\n<b>✅ Special Access:</b> You can view/manage ALL channels."

    intro_html = (
        "This bot helps staff parse quizzes, export CSV files, and post anonymous quizzes to channels."
        f"\n\n<b>📌 Owner:</b> {h(OWNER_CONTACT)}{access_note}"
    )
    header = ui_box_html(f"{BOT_BRAND} — Quiz Management Bot", intro_html, emoji="📚")

    user_cmds_html = (
        "<code>/start</code> — Welcome message\n"
        "<code>/help</code> — Show this guide\n"
        "<code>/commands</code> — List commands (filtered by your role)\n"
        "<code>/ask</code> — Contact support (send text or reply to a message/file)\n"
        "\n<i>Staff tools are restricted. Contact the owner for access.</i>"
    )
    user_section = ui_box_html("User Commands", user_cmds_html, emoji="👤")

    staff_cmds_html = (
        "<b>Quiz & Export</b>\n"
        "• Send text message → Auto-parsed into buffer\n"
        "• Forward Poll/Quiz → Auto-saved to buffer\n"
        "• <code>/filter &lt;text&gt;</code> — Remove text during parsing\n"
        "• <code>/done</code> — Export CSV + JSON, clear buffer\n"
        "• <code>/clear</code> — Clear buffer without exporting\n"
        "\n<b>Channels</b>\n"
        "• <code>/addchannel &lt;@channel | -100...&gt;</code> — Add a channel\n"
        "• <code>/listchannels</code> — List your channels\n"
        "• <code>/removechannel &lt;DB-ID&gt;</code> — Remove a channel\n"
        "• <code>/setprefix &lt;DB-ID&gt; &lt;text&gt;</code> — Set prefix\n"
        "• <code>/setexplink &lt;DB-ID&gt; &lt;link&gt;</code> — Set explanation link\n"
        "• <code>/post &lt;DB-ID&gt;</code> — Post quizzes to channel\n"
        "• <code>/post &lt;DB-ID&gt; keep</code> — Post without clearing\n"
        "\n<b>Inbox & Moderation</b>\n"
        "• <code>/reply &lt;ticket_id&gt; [msg]</code> — Reply to ticket (or reply to a message)\n"
        "• <code>/close &lt;ticket_id&gt;</code> — Close ticket\n"
        "• <code>/ban &lt;user_id&gt;</code> — Ban user\n"
        "• <code>/unban &lt;user_id&gt;</code> — Unban user\n"
        "• <code>/banned</code> — List banned users\n"
        "\n<b>Broadcast & Content</b>\n"
        "• <code>/broadcast [message]</code> — Send to all users (or reply to broadcast media)\n"
        "• <code>/private_send &lt;id|all&gt; [text]</code> — Protected content (or reply to send media)\n"
        "\n<b>Analytics</b>\n"
        "• <code>/adminpanel</code> — Posting leaderboard"
    )
    staff_section = ui_box_html("Staff Commands (Admin / Owner)", staff_cmds_html, emoji="🛠")

    owner_cmds_html = (
        "• <code>/addadmin &lt;user_id&gt;</code> — Promote to Admin\n"
        "• <code>/removeadmin &lt;user_id&gt;</code> — Demote to User\n"
        "• <code>/grantall &lt;admin_id&gt;</code> — Grant full channel access\n"
        "• <code>/revokeall &lt;admin_id&gt;</code> — Revoke full access\n"
        "• <code>/grantvision &lt;user_id&gt;</code> — Grant Image→Quiz access\n"
        "• <code>/revokevision &lt;user_id&gt;</code> — Revoke Image→Quiz access"
    )
    owner_section = ui_box_html("Owner Controls", owner_cmds_html, emoji="👑")

    if role == ROLE_OWNER:
        return "\n\n".join([header, user_section, staff_section, owner_section])
    if role == ROLE_ADMIN:
        return "\n\n".join([header, user_section, staff_section])
    return "\n\n".join([header, user_section])


# ---------------------------
# UI STYLING HELPERS (100% HTML, zero Markdown)
# ---------------------------

def _quote_html(body_html: str) -> str:
    """
    Create a Telegram-compatible 'quote' look WITHOUT <blockquote> (Telegram HTML doesn't support it).
    We prefix each line with a light vertical bar.
    body_html may already contain \n and inline tags (<b>, <code>, <i>).
    """
    if not body_html:
        return ""
    parts = body_html.split("\n")
    parts = [p.strip() for p in parts]
    parts = [p for p in parts if p != ""]
    if not parts:
        return ""
    return "\n".join([f"│ {p}" for p in parts])

def ui_box_text(title: str, body_text: str, emoji: str = "✅", footer_text: str = "") -> str:
    """
    Use when body/footer are PLAIN TEXT. We escape them safely.
    """
    body_html = h(body_text)
    body_html = _quote_html(body_html)
    out = f"<b>{emoji} {h(title)}</b>"
    if body_html:
        out += f"\n{body_html}"
    if footer_text:
        out += f"\n<i>{h(footer_text)}</i>"
    return out

def ui_box_html(title: str, body_html: str, emoji: str = "✅", footer_html: str = "") -> str:
    """
    Use when body already contains HTML tags (<b>, <code>, <br>, etc).
    IMPORTANT: Caller must escape any user-provided data using h().
    """
    body_q = _quote_html(body_html)
    out = f"<b>{emoji} {h(title)}</b>"
    if body_q:
        out += f"\n{body_q}"
    if footer_html:
        out += f"\n<i>{footer_html}</i>"
    return out

def usage_box(command: str, args: str = "", description: str = "") -> str:
    """
    Consistent usage message in HTML (no Markdown).
    """
    cmd = command.lstrip("/")
    body = f"<code>/{h(cmd)} {h(args)}</code>"
    if description:
        body += f"\n\n{h(description)}"
    return ui_box_html("Usage", body, emoji="ℹ️")


# ---------------------------
# COMMANDS REGISTRY
# ---------------------------
COMMANDS_REGISTRY = {
    "public": {
        "description": "👤 User Commands",
        "commands": {
            "start": "Welcome / membership check",
            "help": "Show detailed command guide",
            "commands": "Show all available commands",
            "ask": "Contact support (text or reply to file/photo)",
            "solve_on": "Enable user AI solving",
            "solve_off": "Disable user AI solving",
            "scanhelp": "Image→Quiz tutorial (if vision granted)",
            "vision_on": "Enable Image→Quiz mode",
            "vision_off": "Disable Image→Quiz mode"
        }
    },
    "workflow": {
        "description": "🛠 Core Workflow (Admin/Owner)",
        "items": [
            "Send text message → Auto-parsed into buffer",
            "Forward Poll/Quiz → Auto-saved to buffer",
            "Send photo → (Enable with /vision_on) → Extract MCQs → Buffer",
            "/done → Export CSV + JSON, clear buffer",
            "/post <DB-ID> → Publish buffered quizzes to channel",
            "/filter <text> → Remove text during parsing",
            "/clear → Clear buffer without exporting",
        ]
    },
    "admin": {
        "description": "🛠 Staff Commands (Admin + Owner)",
        "commands": {
            "filter": "Remove specific text during parsing",
            "done": "Export CSV + JSON, clear buffer",
            "clear": "Clear buffer without exporting",
            "addchannel": "Add a target channel",
            "listchannels": "List channels (visible scope)",
            "removechannel": "Remove a channel",
            "setprefix": "Set channel prefix",
            "setexplink": "Set explanation link",
            "post": "Post buffered quizzes to channel",
            "postemoji": "Post buffered emoji quizzes to channel",
            "broadcast": "Send message to all users",
            "adminpanel": "View posting leaderboard",
            "reply": "Reply to support ticket",
            "close": "Close support ticket",
            "ban": "Ban a user",
            "unban": "Unban a user",
            "banned": "Show ban log / banned users",
            "private_send": "Send private message to a user",
            "send_private": "Alias of /private_send",
            "himusai_on": "Enable admin/owner inbox AI-only mode",
            "himusai_off": "Disable admin/owner inbox AI-only mode",
            "probaho_on": "Enable user AI in current group",
            "probaho_off": "Disable user AI in current group",
            "explain_on": "Enable explanation in quiz/csv/json exports",
            "explain_off": "Disable explanation in quiz/csv/json exports",
            "quizprefix": "Set global generated-quiz prefix",
            "quizlink": "Set global generated-quiz link"
        }
    },
    "owner": {
        "description": "👑 Owner-Only Commands",
        "commands": {
            "addadmin": "Promote user to Admin",
            "removeadmin": "Demote Admin to User",
            "grantall": "Grant admin full channel access",
            "revokeall": "Revoke admin full access",
            "grantvision": "Grant Image→Quiz access to admin",
            "revokevision": "Revoke Image→Quiz access",
            "ownerstats": "Owner dashboard (users/admins/active/memory/errors)"
        }
    },
    "vision": {
        "description": "📷 Image → Quiz (Owner + Granted Admins)",
        "items": [
            "Send clear photo/scan of question page",
            "Bot extracts MCQs + generates explanations",
            "Questions saved to your buffer",
            "Use /done to export, /post to publish",
            "Owner can use /grantvision <user_id> to enable for admins",
            "Use /vision_on to start image extraction, /vision_off to stop",
        ]
    },
}


# ---------------------------
# COMMANDS
# ---------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id
    if not await enforce_required_memberships(update, context):
        return
    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\n\nContact: {OWNER_CONTACT}")
        return

    role = get_role(uid)

    body_html = (
        f"<b>Your Role:</b> <code>{h(role)}</code>"
        f"\n\nUse <code>/help</code> for commands or <code>/commands</code> for a quick list."
    )
    msg = ui_box_html(f"Welcome to {BOT_BRAND}", body_html, emoji="👋")
    await safe_reply(update, msg)



async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id

    if not await enforce_required_memberships(update, context):
        return

    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\n\nContact: {OWNER_CONTACT}")
        return

    role = get_role(uid)
    help_text = help_for_role(role, uid)
    await safe_reply(update, help_text)


async def cmd_commands(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all available commands in a categorized list."""
    ensure_user(update)
    uid = update.effective_user.id

    if not await enforce_required_memberships(update, context):
        return

    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\n\nContact: {OWNER_CONTACT}")
        return

    role = get_role(uid)

    # Build sections (HTML)
    sections = []

    # Public commands (USER always sees)
    pub = COMMANDS_REGISTRY.get("public", {})
    if pub:
        cmds = dict(pub.get("commands", {}))

        # Hide vision-related commands unless user actually has vision access
        if not can_use_vision(uid):
            cmds.pop("scanhelp", None)
            cmds.pop("vision_on", None)
            cmds.pop("vision_off", None)

        # Hide staff-only toggles from normal users
        if not (is_admin(uid) or is_owner(uid)):
            cmds.pop("explain_on", None)
            cmds.pop("explain_off", None)

        body = "\n".join([f"<code>/{h(c)}</code> — {h(d)}" for c, d in cmds.items()])
        sections.append(ui_box_html(pub["description"], body, emoji="👤"))

    # Workflow: ONLY Admin/Owner (never USER)
    if is_admin(uid) or is_owner(uid):
        workflow = COMMANDS_REGISTRY.get("workflow", {})
        if workflow:
            body = "\n".join([f"• {h(item)}" for item in workflow.get("items", [])])
            sections.append(ui_box_html(workflow["description"], body, emoji="🛠"))

    # Admin commands: Admin+Owner
    if is_admin(uid) or is_owner(uid):
        admin_cmds = COMMANDS_REGISTRY.get("admin", {})
        if admin_cmds:
            body = "\n".join([f"<code>/{h(c)}</code> — {h(d)}" for c, d in admin_cmds.get("commands", {}).items()])
            sections.append(ui_box_html(admin_cmds["description"], body, emoji="🛠"))

    # Owner commands: Owner only
    if is_owner(uid):
        owner_cmds = COMMANDS_REGISTRY.get("owner", {})
        if owner_cmds:
            body = "\n".join([f"<code>/{h(c)}</code> — {h(d)}" for c, d in owner_cmds.get("commands", {}).items()])
            sections.append(ui_box_html(owner_cmds["description"], body, emoji="👑"))

    # Vision section: ONLY if can_use_vision(uid)
    if can_use_vision(uid):
        vision = COMMANDS_REGISTRY.get("vision", {})
        if vision:
            body = "\n".join([f"• {h(item)}" for item in vision.get("items", [])])
            sections.append(ui_box_html(vision["description"], body, emoji="📷"))

    header = ui_box_html("All Available Commands", "Choose a command below.", emoji="📋")
    msg = header + "\n\n" + "\n\n".join(sections)
    await safe_reply(update, msg)



async def cmd_features(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Alias for /commands."""
    await cmd_commands(update, context)


@require_vision
async def cmd_scanhelp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    body_html = (
        "<b>Steps</b>\n"
        "1) Send a clear photo/scan of the question page\n"
        "2) Bot extracts MCQs + explanations → saves to your buffer\n"
        "3) Use <code>/done</code> to export CSV/JSON\n"
        "4) Use <code>/post</code> to publish to channel\n\n"
        "<b>Tips for best results</b>\n"
        "• Crop tightly (avoid background)\n"
        "• Good lighting, no blur\n"
        "• For 2-column pages: crop section-by-section"
    )
    await ok_html(update, "Image → Quiz Tutorial", body_html, emoji="📷")



@require_vision
async def cmd_vision_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enable Image→Quiz processing for the current user (until turned off)."""
    uid = update.effective_user.id
    set_vision_mode_on(uid, True)
    await ok_html(update, "Image→Quiz Enabled", "Now you can send images and the bot will extract MCQs into your buffer.\n\nDisable anytime using <code>/vision_off</code>.", emoji="📷")


@require_vision
async def cmd_vision_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Disable Image→Quiz processing for the current user."""
    uid = update.effective_user.id
    set_vision_mode_on(uid, False)
    await ok_html(update, "Image→Quiz Disabled", "Image messages will be ignored by the extractor until you enable it again using <code>/vision_on</code>.", emoji="📷")


async def cmd_solve_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enable problem-solving chat for USER role."""
    ensure_user(update)
    uid = update.effective_user.id
    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\n\nContact: {OWNER_CONTACT}")
        return
    if get_role(uid) != ROLE_USER:
        await warn(update, "Not Available", "Problem-solving chat is intended for normal users. Admin/Owner workflow should remain unchanged.")
        return
    set_solver_mode_on(uid, True)
    await ok_html(update, "Solver Enabled", "Now just send your question as text and the bot will reply with a solved explanation.\n\nTurn off anytime using <code>/solve_off</code>.", emoji="🧠")


async def cmd_solve_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Disable problem-solving chat for USER role."""
    ensure_user(update)
    uid = update.effective_user.id
    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\n\nContact: {OWNER_CONTACT}")
        return
    if get_role(uid) != ROLE_USER:
        await warn(update, "Not Available", "Problem-solving chat is intended for normal users.")
        return
    set_solver_mode_on(uid, False)
    await ok_html(update, "Solver Disabled", "The bot will no longer auto-solve your text messages.", emoji="🧠")


@require_admin
async def cmd_explain_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enable explanation posting for staff (Admin/Owner)."""
    uid = update.effective_user.id
    set_explain_mode_on(uid, True)
    await ok_html(
        update,
        "Explanation Enabled",
        "ইনশাআল্লাহ, এখন থেকে কুইজ পোস্ট করার সময় <b>Explanation</b> যুক্ত হবে।\n\nবন্ধ করতে <code>/explain_off</code> ব্যবহার করুন।",
        emoji="📖",
    )


@require_admin
async def cmd_explain_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Disable explanation posting for staff (Admin/Owner)."""
    uid = update.effective_user.id
    set_explain_mode_on(uid, False)
    await ok_html(
        update,
        "Explanation Disabled",
        "এখন থেকে কুইজ পোস্ট হবে <b>শুধু প্রশ্ন + অপশন</b> দিয়ে (Explanation ছাড়া)।\n\nচালু করতে <code>/explain_on</code> ব্যবহার করুন।",
        emoji="🧾",
    )


@require_owner
async def cmd_ownerstats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner-only: show bot usage + health stats."""
    uid = update.effective_user.id
    # Totals
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM users")
    total_users = int(cur.fetchone()["c"] or 0)

    cur.execute("SELECT COUNT(*) AS c FROM users WHERE role IN ('OWNER','ADMIN')")
    staff_count = int(cur.fetchone()["c"] or 0)

    # Active users in last 24h
    since_dt = dt.datetime.now(timezone.utc) - dt.timedelta(hours=24)
    since_iso = since_dt.replace(microsecond=0).isoformat()
    cur.execute("SELECT COUNT(*) AS c FROM users WHERE last_seen_at IS NOT NULL AND last_seen_at >= ?", (since_iso,))
    active_24h = int(cur.fetchone()["c"] or 0)

    # Errors in last 24h + last few errors
    cur.execute("SELECT COUNT(*) AS c FROM bot_logs WHERE level='ERROR' AND created_at >= ?", (since_iso,))
    err_24h = int(cur.fetchone()["c"] or 0)

    cur.execute("SELECT created_at, event, meta_json FROM bot_logs WHERE level='ERROR' ORDER BY id DESC LIMIT 5")
    last_errors = cur.fetchall()
    conn.close()

    # DB size on disk
    db_mb = 0.0
    try:
        if os.path.exists(DB_PATH):
            db_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    except Exception:
        db_mb = 0.0

    # Process memory (RSS)
    rss_mb = process_rss_mb()

    # Compose message (Telegram-friendly, HTML safe)
    lines = []
    lines.append(f"<b>👑 Owner Dashboard</b>")
    lines.append(f"⏱ Uptime: <code>{h(fmt_uptime())}</code>")
    lines.append("")
    lines.append(f"👥 Total Users: <b>{h(total_users)}</b>")
    lines.append(f"🛠 (Owner+Admin): <b>{h(staff_count)}</b>")
    lines.append(f"✅ Active (last 24 hours): <b>{h(active_24h)}</b>")
    lines.append("")
    lines.append(f"💾 DB Size: <code>{h(fmt_mb(db_mb))}</code>")
    lines.append(f"🧠 RAM (RSS): <code>{h(fmt_mb(rss_mb))}</code>")
    lines.append("")
    if err_24h == 0:
        lines.append("🟢 Error (24 hours): <b>0</b> — Chill bro 🌝")
    else:
        lines.append(f"🔴 Error (24 hours): <b>{h(err_24h)}</b>")
        if last_errors:
            lines.append("")
            lines.append("<b>last 5 Error:</b>")
            for r in last_errors:
                ts = str(r["created_at"] or "")
                ev = str(r["event"] or "")
                meta = ""
                try:
                    meta_obj = json.loads(r["meta_json"] or "{}")
                    meta = str(meta_obj.get("error") or "")[:80]
                except Exception:
                    meta = ""
                if meta:
                    lines.append(f"• <code>{h(ts)}</code> — {h(ev)} — <i>{h(meta)}</i>")
                else:
                    lines.append(f"• <code>{h(ts)}</code> — {h(ev)}")

    msg = "\n".join(lines)
    await safe_reply(update, msg)




@require_owner
async def cmd_quizprefix(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set prefix used for generated quizzes."""
    if not update.message:
        return
    val = " ".join(context.args).strip() if context.args else ""
    if not val:
        cur = get_setting("quiz_prefix", "প্রবাহ")
        await safe_reply(update, ui_box_text("Generate Quiz Prefix", f"Current prefix: {cur}", emoji="📝"))
        return
    set_setting("quiz_prefix", val)
    await safe_reply(update, ui_box_text("Updated", f"Generate Quiz prefix set to: {val}", emoji="✅"))


@require_owner
async def cmd_quizlink(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set explanation link appended to generated quiz explanations."""
    if not update.message:
        return
    val = " ".join(context.args).strip() if context.args else ""
    # allow clearing
    set_setting("quiz_expl_link", val)
    if val:
        await safe_reply(update, ui_box_text("Updated", f"Generate Quiz explanation link set.", emoji="✅"))
    else:
        await safe_reply(update, ui_box_text("Updated", f"Generate Quiz explanation link cleared.", emoji="✅"))


async def cmd_addadmin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await err(update, "Usage", "/addadmin <user_id>")
        return
    target = to_int(context.args[0])
    if not target:
        await err(update, "Invalid Input", f"Invalid user_id: {context.args[0]}")
        return

    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT role FROM users WHERE user_id=?", (target,))
    r = cur.fetchone()
    if r is None:
        cur.execute(
            "INSERT INTO users(user_id, role, first_name, username, is_banned, created_at, can_view_all, can_use_vision, last_seen_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (target, ROLE_ADMIN, "", "", 0, now_iso(), 0, 0),
        )
    else:
        cur.execute("UPDATE users SET role=? WHERE user_id=?", (ROLE_ADMIN, target))
    conn.commit()
    conn.close()

    db_log("INFO", "add_admin", {"by": update.effective_user.id, "target": target})
    await ok_html(update, "Admin Promoted", f"User <code>{h(target)}</code> is now an <b>ADMIN</b>.")


@require_owner
async def cmd_removeadmin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await err(update, "Usage", "/removeadmin <user_id>")
        return
    target = to_int(context.args[0])
    if not target:
        await err(update, "Invalid Input", f"Invalid user_id: {context.args[0]}")
        return
    if target == OWNER_ID:
        await warn(update, "Cannot Demote", "Owner cannot be demoted.")
        return

    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE users SET role=?, can_view_all=0 WHERE user_id=?", (ROLE_USER, target))
    conn.commit()
    conn.close()

    db_log("INFO", "remove_admin", {"by": update.effective_user.id, "target": target})
    await ok_html(update, "Admin Demoted", f"User <code>{h(target)}</code> is now a <b>USER</b>.")


@require_owner
async def cmd_grantall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await err(update, "Usage", "/grantall <admin_id>")
        return
    target = to_int(context.args[0])
    if not target:
        await err(update, "Invalid Input", "Invalid admin_id.")
        return
    if target == OWNER_ID:
        await warn(update, "Already Granted", "Owner already has full access.")
        return
    if get_role(target) != ROLE_ADMIN:
        await err(update, "Invalid Role", "Target user is not an Admin.")
        return
    set_can_view_all(target, True)
    await ok_html(update, "Full Access Granted", f"User <code>{h(target)}</code> can now manage <b>all channels</b>.")


@require_owner
async def cmd_revokeall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await err(update, "Usage", "/revokeall <admin_id>")
        return
    target = to_int(context.args[0])
    if not target:
        await err(update, "Invalid Input", "Invalid admin_id.")
        return
    if target == OWNER_ID:
        await warn(update, "Cannot Revoke", "Owner access cannot be revoked.")
        return
    set_can_view_all(target, False)
    await ok_html(update, "Access Revoked", f"User <code>{h(target)}</code> channel access revoked.")


@require_owner
async def cmd_grantvision(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await err(update, "Usage", "/grantvision <user_id>")
        return
    target = to_int(context.args[0])
    if not target:
        await err(update, "Invalid Input", "Invalid user_id.")
        return
    if target == OWNER_ID:
        await warn(update, "Already Granted", "Owner already has Image→Quiz access.")
        return
    if get_role(target) != ROLE_ADMIN:
        await err(update, "Invalid Role", "Target must be an ADMIN first. Use /addadmin to promote.")
        return
    set_can_use_vision(target, True)
    await ok_html(update, "Vision Access Granted", f"User <code>{h(target)}</code> can now use <b>Image→Quiz</b>.")

@require_owner
async def cmd_revokevision(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await err(update, "Usage", "/revokevision <user_id>")
        return
    target = to_int(context.args[0])
    if not target:
        await err(update, "Invalid Input", "Invalid user_id.")
        return
    if target == OWNER_ID:
        await warn(update, "Cannot Revoke", "Owner access cannot be revoked.")
        return
    set_can_use_vision(target, False)
    await ok_html(update, "Vision Access Revoked", f"User <code>{h(target)}</code> Image→Quiz access revoked.")


@require_admin
async def cmd_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    phrase = " ".join(context.args).strip()
    if not phrase:
        await safe_reply(update, usage_box("filter", "<text to remove>", "Remove this text from parsed questions"))
        return
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO filters(user_id, phrase, created_at) VALUES (?,?,?)",
        (uid, phrase, now_iso()),
    )
    conn.commit()
    conn.close()
    body = f"<b>Filter Added:</b> <code>{h(phrase)}</code>"
    await ok_html(update, "Filter Configured", body)


@require_admin
async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    buffer_clear(uid)
    await ok(update, "Buffer Cleared", "Your buffer is now empty.")


@require_admin


async def on_image_react_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fallback callback for image reaction quizzes.

    Some builds register an image-specific callback handler using the pattern
    ``imgreact:<quiz_id>:<selected>``. Earlier code paths reuse the normal
    emoji-quiz keyboard and never emit this callback. To keep startup robust and
    avoid NameError crashes, this handler gracefully supports the image callback
    format and stores the answer in the same emoji_quizzes tables.
    """
    if not update.callback_query:
        return
    q = update.callback_query
    data = (q.data or '').strip()
    m = re.match(r'^imgreact:([0-9a-f]{6,16}):(\d+)$', data)
    if not m:
        return
    quiz_id = m.group(1)
    selected = int(m.group(2))
    uid = q.from_user.id if q.from_user else 0
    if not uid:
        return
    quiz = emoji_quiz_get(quiz_id)
    if not quiz:
        await q.answer('Quiz expired or not found.', show_alert=True)
        return
    if emoji_quiz_has_answered(quiz_id, uid):
        prev = emoji_quiz_user_choice(quiz_id, uid)
        counts = emoji_quiz_counts(quiz_id)
        expl = str(quiz.get('explanation', '') or '').strip()
        if expl:
            expl = clean_latex(expl)
        stat_parts = []
        for i in range(1, 5):
            stat_parts.append(f"{EMOJI_BUTTONS[i-1]}={counts.get(i,0)}")
        msg = f"You already answered: {EMOJI_BUTTONS[max(0, prev-1)] if 1 <= prev <= 4 else '-'}\n"
        correct = int(quiz.get('correct_answer', 0) or 0)
        if correct > 0:
            msg += f"Correct: {EMOJI_BUTTONS[max(0, correct-1)]}\n"
        msg += ' | '.join(stat_parts)
        if expl:
            msg += f"\n\n{expl}"
        await q.answer(msg[:180], show_alert=True)
        return
    correct = int(quiz.get('correct_answer', 0) or 0)
    is_correct = (selected == correct and correct > 0)
    emoji_quiz_record_answer(quiz_id, uid, selected, is_correct)
    counts = emoji_quiz_counts(quiz_id)
    expl = str(quiz.get('explanation', '') or '').strip()
    if expl:
        expl = clean_latex(expl)
    stat_parts = []
    for i in range(1, 5):
        stat_parts.append(f"{EMOJI_BUTTONS[i-1]}={counts.get(i,0)}")
    if is_correct:
        msg = '🎉 Congratulations!'
    else:
        msg = f"❌ Wrong\n✅ Correct: {EMOJI_BUTTONS[max(0, correct-1)] if correct > 0 else '?'}"
    msg += f"\n{' | '.join(stat_parts)}"
    if expl:
        msg += f"\n\n{expl}"
    await q.answer(msg[:180], show_alert=True)
async def cmd_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    items = buffer_list(uid, limit=99999)
    if not items:
        await warn(update, "Buffer Empty", "No questions to export. Use /add or send quizzes first.")
        return

    rows = [payload for (_id, payload) in items]
    df = pd.DataFrame(rows)

    # Normalize inline explanation inside question across buffer (fixes CSV/JSON export where explanation appears in question)
    norm_rows = []
    for r in rows:
        q = str(r.get("questions", "") or "")
        e = str(r.get("explanation", "") or "")
        q2, expl2 = split_inline_explain(q)
        if expl2 and not e.strip():
            e = expl2
        # overwrite
        rr = dict(r)
        rr["questions"] = q2.strip()
        rr["explanation"] = (e.strip() if explain_mode_on(uid) else "")
        norm_rows.append(rr)
    rows = norm_rows
    df = pd.DataFrame(rows)
    cols = ["questions", "option1", "option2", "option3", "option4", "option5", "answer", "explanation", "type", "section"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]

    with tempfile.NamedTemporaryFile("w+b", suffix=".csv", delete=False) as f:
        path = f.name
    df.to_csv(path, index=False, encoding="utf-8-sig")

    # Also export JSON in quiz format (as requested)
    def _ans_to_letter(n: int) -> str:
        return {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}.get(int(n or 0), "")

    quiz_json = []
    for r in rows:
        opts_map = {"A": r.get("option1", ""), "B": r.get("option2", ""), "C": r.get("option3", ""), "D": r.get("option4", "")}
        # Include E only if present (keeps UI format similar to screenshot)
        if str(r.get("option5", "")).strip():
            opts_map["E"] = r.get("option5", "")
        quiz_json.append({
            "question": r.get("questions", ""),
            "options": opts_map,
            "correct_answer": _ans_to_letter(r.get("answer", 0)),
            "explanation": r.get("explanation", ""),
        })

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as jf:
        json_path = jf.name
        json.dump(quiz_json, jf, ensure_ascii=False, indent=2)

    try:
        await update.message.reply_document(
            document=open(path, "rb"),
            caption=f"<b>✅ CSV Export</b>\n<i>{len(df)} questions exported</i>",
            parse_mode=ParseMode.HTML,
        )
        await update.message.reply_document(
            document=open(json_path, "rb"),
            caption="<b>✅ JSON Export</b>\n<i>Quiz format (question/options/correct_answer/explanation)</i>",
            parse_mode=ParseMode.HTML,
        )
        await ok_html(update, "Export Complete", f"CSV + JSON ready. <code>{h(len(df))}</code> questions exported.")
    finally:
        with contextlib.suppress(Exception):
            os.remove(path)
        with contextlib.suppress(Exception):
            os.remove(json_path)

    buffer_clear(uid)


# Channel commands
@require_admin
async def cmd_addchannel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /addchannel -1001234567890
    /addchannel @channelusername
    """
    uid = update.effective_user.id
    if not context.args:
        await safe_reply(update, usage_box("addchannel", "<@channel | -100...>", "Add a new target channel"))
        return

    ref = context.args[0].strip()
    try:
        if ref.lstrip("-").isdigit():
            chat = await context.bot.get_chat(int(ref))
        else:
            chat = await context.bot.get_chat(ref)

        channel_add(chat.id, chat.title or chat.username or "", uid)
        body = (
            f"ChatID: {h(str(chat.id))}\n"
            f"Title: {h(chat.title or chat.username or 'N/A')}\n"
            f"\nUse /listchannels to get the DB-ID."
        )
        await ok(update, "Channel Added", body)
    except TelegramError as e:
        await err(update, "Failed to Add Channel", f"Error: {h(str(e)[:100])}\n\nMake sure the bot is an Admin in that channel.")


@require_admin
async def cmd_listchannels(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    rows = channel_list_for_user(uid)
    if not rows:
        await warn(update, "No Channels", "No channels found for your access level.")
        return

    lines = []
    for r in rows:
        owner_tag = ""
        if can_view_all(uid):
            owner_tag = f" | Added by <code>{h(str(r.added_by))}</code>"
        lines.append(
            f"<b>DB-ID:</b> <code>{h(str(r.id))}</code> | <b>ChatID:</b> <code>{h(str(r.channel_chat_id))}</code>{owner_tag}\n"
            f"<b>Title:</b> {h(r.title)}\n"
            f"<b>Prefix:</b> <code>{h(r.prefix or '')}</code>\n"
            f"<b>Link:</b> <code>{h(r.expl_link or '')}</code>\n"
        )
    
    body = "\n".join(lines)
    msg = ui_box_html("Your Channels", body, emoji="📋")
    await safe_reply(update, msg)


@require_admin
async def cmd_removechannel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("removechannel", "<DB-ID>", "Remove a channel"))
        return
    cid = int(context.args[0])

    ch = channel_get_by_id_for_user(uid, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or you don't have access.")
        return

    result = channel_remove(cid)
    if result:
        await ok_html(update, "Channel Removed", f"<code>{h(ch.title)}</code> has been removed.")
    else:
        await err(update, "Removal Failed", "Could not remove the channel. Try again.")


@require_admin
async def cmd_setprefix(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if len(context.args) < 2 or not context.args[0].isdigit():
        await safe_reply(update, usage_box("setprefix", "<DB-ID> <text>", "Set the prefix for a channel"))
        return

    cid = int(context.args[0])
    new_prefix = " ".join(context.args[1:]).strip()

    ch = channel_get_by_id_for_user(uid, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or you don't have access.")
        return

    old_prefix = ch.prefix or "(none)"
    result = channel_set_prefix(cid, new_prefix)

    if result:
        body = (
            f"Channel: {h(ch.title)}\n"
            f"DB-ID: {h(str(cid))}\n"
            f"Old Prefix: {h(old_prefix)}\n"
            f"New Prefix: {h(new_prefix)}"
        )
        await ok(update, "Prefix Updated", body)
    else:
        await err(update, "Update Failed", "Could not update the prefix. Try again.")


@require_admin
async def cmd_setexplink(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if len(context.args) < 2 or not context.args[0].isdigit():
        await safe_reply(update, usage_box("setexplink", "<DB-ID> <https://...>", "Set explanation link for a channel"))
        return

    cid = int(context.args[0])
    new_link = " ".join(context.args[1:]).strip()

    ch = channel_get_by_id_for_user(uid, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or you don't have access.")
        return

    old_link = ch.expl_link or "(none)"
    result = channel_set_expl_link(cid, new_link)

    if result:
        body = (
            f"Channel: {h(ch.title)}\n"
            f"DB-ID: {h(str(cid))}\n"
            f"Old Link: {h(old_link)}\n"
            f"New Link: {h(new_link)}"
        )
        await ok(update, "Link Updated", body)
    else:
        await err(update, "Update Failed", "Could not update the link. Try again.")


def quiz_to_poll_parts(payload: Dict[str, Any]) -> Tuple[str, List[str], int, str]:
    q = str(payload.get("questions", "")).strip()
    # Normalize: if explanation was mistakenly stored inside question text (e.g. "... explain ; ...")
    q2, expl2 = split_inline_explain(q)
    if expl2 and not str(payload.get("explanation", "")).strip():
        q = q2.strip()
        payload = dict(payload)
        payload["explanation"] = expl2.strip()
    else:
        q = q2.strip()
    opts = [
        str(payload.get("option1", "")).strip(),
        str(payload.get("option2", "")).strip(),
        str(payload.get("option3", "")).strip(),
        str(payload.get("option4", "")).strip(),
        str(payload.get("option5", "")).strip(),
    ]
    opts = [o for o in opts if o]
    if len(opts) < 2:
        if len(opts) == 0:
            opts = ["Option A", "Option B"]
        else:  # len(opts) == 1
            opts = opts + ["Option B"]
    if len(opts) > 10:
        opts = opts[:10]
    ans = int(payload.get("answer", 0) or 0)  # 1-based
    correct_option_id = ans - 1 if 1 <= ans <= len(opts) else None
    explanation = str(payload.get("explanation", "")).strip()
    return q, opts, (correct_option_id if correct_option_id is not None else -1), explanation


@require_admin
async def cmd_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /post <DB-ID> [keep]
    """
    admin_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("post", "<DB-ID> [keep]", "Post buffered quizzes to a channel. Use 'keep' to keep buffer."))
        return

    cid = int(context.args[0])
    keep = (len(context.args) > 1 and context.args[1].strip().lower() == "keep")

    ch = channel_get_by_id_for_user(admin_id, cid)
    if not ch:
        await warn_html(update, "Channel Not Found", f"No access to that channel. Use <code>/listchannels</code> to view yours.")
        return

    items = buffer_list(admin_id, limit=MAX_BUFFERED_QUESTIONS)
    if not items:
        await warn(update, "Buffer Empty", "No quizzes to post. Send text or forward polls first.")
        return

    await info_html(update, "Posting to Channel", f"<code>{h(ch.title)}</code> — <code>{h(str(ch.channel_chat_id))}</code>\n\nPosting <code>{h(len(items))}</code> question(s)...")

    posted_ids: List[int] = []
    ok_count, fail_count = 0, 0

    for (row_id, payload) in items:
        try:
            q, opts, correct_option_id, expl = quiz_to_poll_parts(payload)

            prefix = (ch.prefix or "")
            prefix = prefix.strip(" ")
            expl_link = (ch.expl_link or "").strip()

            SEP = "\n\u200b"
            q_final = f"{prefix}{SEP}{q}".strip() if prefix else q
            if len(q_final) > 300:
                q_final = q_final[:297] + "..."

            expl_final = expl.strip()
            # Explanation toggle (staff): if OFF, never attach explanation
            if not explain_mode_on(admin_id):
                expl_final = ""
            if expl_link:
                expl_final = (expl_final + "\n\n" if expl_final else "") + f"🔗 {expl_link}"
            expl_final = expl_final.strip()
            if len(expl_final) > 200:
                expl_final = expl_final[:197] + "..."

            if correct_option_id >= 0:
                await context.bot.send_poll(
                    chat_id=ch.channel_chat_id,
                    question=q_final,
                    options=opts,
                    is_anonymous=True,
                    type=Poll.QUIZ,
                    correct_option_id=correct_option_id,
                    explanation=expl_final if expl_final else None,
                )
            else:
                await context.bot.send_poll(
                    chat_id=ch.channel_chat_id,
                    question=q_final,
                    options=opts,
                    is_anonymous=True,
                    type=Poll.REGULAR,
                )
                if expl_final:
                    await context.bot.send_message(
                        chat_id=ch.channel_chat_id,
                        text=f"📖 {expl_final}",
                        disable_web_page_preview=True,
                    )

            ok_count += 1
            posted_ids.append(row_id)
            await asyncio.sleep(POST_DELAY_SECONDS)

        except RetryAfter as e:
            await asyncio.sleep(float(e.retry_after) + 0.5)
            fail_count += 1
        except TelegramError as e:
            fail_count += 1
            db_log("ERROR", "post_failed", {"admin_id": admin_id, "channel": ch.channel_chat_id, "error": str(e)})
        except Exception as e:
            fail_count += 1
            db_log("ERROR", "post_failed_unknown", {"admin_id": admin_id, "error": str(e)})

    inc_admin_post(admin_id, ok_count)

    if posted_ids and not keep:
        buffer_remove_ids(admin_id, posted_ids)

    body = (
        f"Posted: {ok_count}\n"
        f"Failed: {fail_count}\n"
        f"Remaining in Buffer: {buffer_count(admin_id)}"
    )
    await ok(update, "Posting Complete", body)


@require_admin
async def cmd_adminpanel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id

    conn = db_connect()
    cur = conn.cursor()

    if can_view_all(uid):  # owner or granted
        cur.execute("""
            SELECT u.user_id, u.first_name, u.username, u.role,
                   COALESCE(s.total_posts, 0) AS total_posts,
                   COALESCE(s.last_post_at, '') AS last_post_at
            FROM users u
            LEFT JOIN admin_post_stats s ON s.admin_id = u.user_id
            WHERE u.role IN ('OWNER','ADMIN')
            ORDER BY total_posts DESC, u.user_id ASC
        """)
    else:
        cur.execute("""
            SELECT u.user_id, u.first_name, u.username, u.role,
                   COALESCE(s.total_posts, 0) AS total_posts,
                   COALESCE(s.last_post_at, '') AS last_post_at
            FROM users u
            LEFT JOIN admin_post_stats s ON s.admin_id = u.user_id
            WHERE u.user_id=?
            LIMIT 1
        """, (uid,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        await safe_reply(update, "No stats available.")
        return

    title = "Staff Posting Leaderboard" if can_view_all(uid) else "Your Posting Stats"
    msg = f"<b>{h(title)}</b>\n\n"
    for r in rows:
        name = (r["first_name"] or "").strip()
        uname = ("@" + r["username"]) if r["username"] else ""
        msg += (
            f"<code>{r['user_id']}</code> {h(name)} {h(uname)}\n"
            f"  Role: <b>{h(r['role'])}</b> | Posted: <b>{r['total_posts']}</b>\n"
            f"  Last: <code>{h(r['last_post_at'])}</code>\n\n"
        )
    await safe_reply(update, msg)


@require_admin
async def cmd_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /broadcast <message>
    OR reply to any message with /broadcast (broadcasts the replied message)
    """
    text = " ".join(context.args).strip()
    replied = update.message.reply_to_message if update.message else None

    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE is_banned=0")
    rows = cur.fetchall()
    conn.close()
    targets = [int(r["user_id"]) for r in rows]

    if not text and not replied:
        await safe_reply(update, usage_box("broadcast", "<message>", "Send message to all users, or reply to forward a message"))
        return

    await info_html(update, "Broadcasting", f"Sending to <code>{h(len(targets))}</code> user(s)...")

    sent, failed = 0, 0

    if replied and not text:
        # Broadcast by copying the replied message (supports media too)
        for tid in targets:
            result = await safe_copy_message(
                context.bot,
                chat_id=tid,
                from_chat_id=replied.chat_id,
                message_id=replied.message_id,
                protect=False,
            )
            if result:
                sent += 1
            else:
                failed += 1
            await asyncio.sleep(BROADCAST_DELAY_SECONDS)
    else:
        # Text broadcast
        for tid in targets:
            try:
                await context.bot.send_message(chat_id=tid, text=text, disable_web_page_preview=True)
                sent += 1
                await asyncio.sleep(BROADCAST_DELAY_SECONDS)
            except Exception:
                failed += 1

    body = (
        f"Sent: {sent}\n"
        f"Failed: {failed}"
    )
    await ok(update, "Broadcast Complete", body)


# Protected content sending:
# Reply to any message: /private_send <user_id|all>
# Protected content sending:
# Reply to any message: /private_send <user_id|all>
# Or send protected text inline: /private_send <user_id|all> <text>
@require_admin
async def cmd_private_send(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await safe_reply(update, usage_box("private_send", "<user_id|all> [text]", "Send protected message (no forward/save). Reply to message or provide text."))
        return

    target = context.args[0].strip().lower()
    reply_msg = update.message.reply_to_message if update.message else None
    inline_text = " ".join(context.args[1:]).strip()

    if target == "all":
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM users WHERE is_banned=0")
        rows = cur.fetchall()
        conn.close()
        targets = [int(r["user_id"]) for r in rows]
    else:
        if not target.isdigit():
            await err_html(update, "Invalid Target", f"Use numeric user_id or <code>all</code>")
            return
        targets = [int(target)]

    # If no reply message, allow protected text send
    if not reply_msg:
        if not inline_text:
            await warn(update, "No Content", "Reply to a message/file/photo or provide text inline")
            return

        ok, fail = 0, 0
        for tid in targets:
            try:
                await context.bot.send_message(
                    chat_id=tid,
                    text=inline_text,
                    disable_web_page_preview=True,
                    protect_content=True,
                )
                ok += 1
            except Exception:
                fail += 1
            await asyncio.sleep(BROADCAST_DELAY_SECONDS)

        body = f"<b>Delivered:</b> <code>{ok}</code>\n<b>Failed:</b> <code>{fail}</code>"
        await ok(update, "Protected Text Delivery Complete", body)
        return

    # Otherwise: copy replied message as protected content (supports all media)
    ok, fail = 0, 0
    for tid in targets:
        success = await safe_copy_message(
            context.bot,
            chat_id=tid,
            from_chat_id=reply_msg.chat_id,
            message_id=reply_msg.message_id,
            protect=True,
        )
        if success:
            ok += 1
        else:
            fail += 1
        await asyncio.sleep(BROADCAST_DELAY_SECONDS)

    body = f"<b>Delivered:</b> <code>{ok}</code>\n<b>Failed:</b> <code>{fail}</code>"
    await ok(update, "Protected Delivery Complete", body)


# Inbox/Tickets
async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id
    if not await enforce_required_memberships(update, context):
        return
    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\nContact: {OWNER_CONTACT}")
        return

    replied = update.message.reply_to_message if update.message else None
    text = " ".join(context.args).strip()

    if not text:
        text = reply_text_or_caption(update)

    # If still empty but we have a replied message (media without caption), allow it
    if not text and not replied:
        await safe_reply(update, usage_box("ask", "<message>", "Ask a support question (or reply to message/file/photo)"))
        return

    tid = ticket_find_open_by_student(uid)
    if tid is None:
        tid = ticket_open(uid, update.effective_user.first_name or "")
        db_log("INFO", "ticket_open", {"ticket_id": tid, "student_id": uid})

    # Save ticket message
    if text:
        ticket_add_msg(tid, "STUDENT", uid, text)
    elif replied:
        ticket_add_msg(tid, "STUDENT", uid, "[MEDIA MESSAGE]")

    staff_ids = list_staff_ids()

    header = (
        f"📩 New Support Message\n"
        f"Ticket: {tid}\n"
        f"From: {uid} ({update.effective_user.first_name or ''})"
    )

    if text:
        for sid in staff_ids:
            await safe_send_text(context.bot, sid, f"{header}\n\n{text}", protect=False)
    else:
        for sid in staff_ids:
            await safe_send_text(context.bot, sid, f"{header}\n\n[MEDIA MESSAGE RECEIVED]", protect=False)

    # Copy replied content to staff (supports all media)
    if replied:
        for sid in staff_ids:
            await safe_copy_message(
                context.bot,
                chat_id=sid,
                from_chat_id=replied.chat_id,
                message_id=replied.message_id,
                protect=False,
            )

    body = f"Ticket ID: {tid}\nA staff member will respond soon."
    await ok(update, "Message Received", body)


@require_admin
async def cmd_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /reply <ticket_id> <message>
    OR reply to any support message/card and run /reply <message>
    Supports text + media/files/photos (by replying).
    """
    replied = update.message.reply_to_message if update.message else None
    tid = None
    text = ""
    if context.args and str(context.args[0]).isdigit():
        tid = int(context.args[0])
        text = " ".join(context.args[1:]).strip()
    else:
        tid = parse_ticket_id_from_any_message(replied)
        text = " ".join(context.args).strip()
    if not tid:
        await safe_reply(update, usage_box("reply", "<ticket_id> [message]", "Reply to support ticket (or reply to support card/media)"))
        return

    if not text:
        text = reply_text_or_caption(update)

    tr = ticket_get(tid)
    if not tr:
        await warn_html(update, "Ticket Not Found", f"No ticket with ID <code>{h(tid)}</code> found")
        return
    if tr["status"] != "OPEN":
        await err_html(update, "Ticket Closed", f"Ticket <code>{h(tid)}</code> is already <b>CLOSED</b>")
        return

    student_id = int(tr["student_id"])
    if is_banned(student_id):
        await warn(update, "User Banned", "The user is currently banned. Unban them first if needed.")
        return

    sent_any = False

    if text:
        ticket_add_msg(tid, "STAFF", update.effective_user.id, text)
        await safe_send_text(context.bot, student_id, f"💬 Support Reply (Ticket {tid})\n\n{text}", protect=False)
        sent_any = True

    if replied:
        ok = await safe_copy_message(
            context.bot,
            chat_id=student_id,
            from_chat_id=replied.chat_id,
            message_id=replied.message_id,
            protect=False,
        )
        if ok:
            ticket_add_msg(tid, "STAFF", update.effective_user.id, "[MEDIA MESSAGE]")
            sent_any = True

    if sent_any:
        await ok_html(update, "Reply Sent", f"<b>Ticket:</b> <code>{h(tid)}</code>\nMessage(s) sent to user.")
        return

    await warn(update, "No Content", "Reply to a message/file/photo or provide text inline")


@require_admin
async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("close", "<ticket_id>", "Close a support ticket"))
        return
    tid = int(context.args[0])
    ok = ticket_close(tid)
    if ok:
        await ok_html(update, "Ticket Closed", f"<b>Ticket:</b> <code>{h(tid)}</code> is now closed.")
    else:
        await warn_html(update, "Ticket Not Found", f"No ticket with ID <code>{h(tid)}</code> found")


# Ban/Unban
@require_admin
async def cmd_ban(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("ban", "<user_id>", "Ban a user from the bot"))
        return
    target = int(context.args[0])
    if target == OWNER_ID:
        await err(update, "Cannot Ban Owner", "The owner cannot be banned.")
        return

    set_ban(target, True)
    audit_ban(update.effective_user.id, target, "BAN")
    db_log("INFO", "ban", {"by": update.effective_user.id, "target": target})

    body = f"User Banned: {target}"
    await ok(update, "User Banned", body)
    await safe_send_text(context.bot, target, f"🚫 You have been banned from <b>{h(BOT_BRAND)}</b>.\nContact: {h(OWNER_CONTACT)}", protect=False)


@require_admin
async def cmd_unban(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("unban", "<user_id>", "Unban a user on the bot"))
        return
    target = int(context.args[0])
    set_ban(target, False)
    audit_ban(update.effective_user.id, target, "UNBAN")
    db_log("INFO", "unban", {"by": update.effective_user.id, "target": target})

    body = f"User Unbanned: {target}"
    await ok(update, "User Unbanned", body)
    await safe_send_text(context.bot, target, f"✅ You have been unbanned. You may use <b>{h(BOT_BRAND)}</b> again.", protect=False)


@require_admin
async def cmd_banned(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id

    conn = db_connect()
    cur = conn.cursor()

    if can_view_all(uid):
        cur.execute("""
            SELECT u.user_id, u.first_name, u.username
            FROM users u
            WHERE u.is_banned=1
            ORDER BY u.user_id ASC
        """)
        rows = cur.fetchall()
        conn.close()
        if not rows:
            await safe_reply(update, "No banned users.")
            return

        msg = "<b>Banned Users (All)</b>\n\n"
        for r in rows:
            uname = ("@" + r["username"]) if r["username"] else ""
            msg += f"<code>{r['user_id']}</code> {h(r['first_name'] or '')} {h(uname)}\n"
        await safe_reply(update, msg)
        return

    # Admin: show only users they banned (currently banned)
    cur.execute("""
        SELECT DISTINCT u.user_id, u.first_name, u.username
        FROM ban_audit b
        JOIN users u ON u.user_id = b.target_user_id
        WHERE b.by_user_id=? AND b.action='BAN' AND u.is_banned=1
        ORDER BY u.user_id ASC
    """, (uid,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        await safe_reply(update, "No banned users (by you).")
        return

    msg = "<b>Banned Users (By You)</b>\n\n"
    for r in rows:
        uname = ("@" + r["username"]) if r["username"] else ""
        msg += f"<code>{r['user_id']}</code> {h(r['first_name'] or '')} {h(uname)}\n"
    await safe_reply(update, msg)


# ---------------------------
# MESSAGE HANDLERS (Core workflow preserved)
# ---------------------------
@require_admin_silent
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin/Owner: any plain text (non-command) gets parsed into buffer.
    In private chat, if HimusAI mode is ON for admin/owner, buffering is skipped.
    """
    uid = update.effective_user.id
    if is_private_chat(update) and get_role(uid) in (ROLE_ADMIN, ROLE_OWNER) and solver_mode_on(uid):
        return
    text = update.message.text or ""
    if not text.strip():
        return

    if buffer_count(uid) >= MAX_BUFFERED_QUESTIONS:
        await warn(update, "Buffer Limit Reached", f"You have {MAX_BUFFERED_QUESTIONS} questions buffered.\n\nUse /done to export or /clear to reset.")
        return

    blocks = split_blocks(text)
    added = 0
    for b in blocks:
        if buffer_count(uid) >= MAX_BUFFERED_QUESTIONS:
            break
        try:
            payload = parse_text_block(b, uid)
            if payload:
                buffer_add(uid, payload)
                added += 1
        except Exception as e:
            db_log("ERROR", "parse_text_failed", {"admin_id": uid, "error": str(e)})

    if added:
        await ok_html(update, "Added to Buffer", f"<code>{h(added)}</code> question(s) added.\n\nTotal buffered: <code>{h(buffer_count(uid))}</code>", footer_html="Use <code>/done</code> to export")
    else:
        await warn(update, "No Questions Found", "No valid quiz blocks detected. Check formatting.")


@require_admin_silent
async def handle_poll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin/Owner: poll/quiz forwarded/sent -> buffer.
    """
    uid = update.effective_user.id
    poll = update.message.poll

    question = clean_common(poll.question or "", uid)
    options = [o.text for o in poll.options]
    opts = options + [""] * (5 - len(options))

    explanation = ""
    if hasattr(poll, "explanation") and poll.explanation:
        explanation = clean_explanation(poll.explanation, uid)

    correct_answer_id = 0
    if poll.type == "quiz" and poll.correct_option_id is not None:
        correct_answer_id = int(poll.correct_option_id) + 1

    payload = {
        "questions": question,
        "option1": (opts[0] or "").strip(),
        "option2": (opts[1] or "").strip(),
        "option3": (opts[2] or "").strip(),
        "option4": (opts[3] or "").strip(),
        "option5": (opts[4] or "").strip(),
        "answer": correct_answer_id,
        "explanation": explanation,
        "type": 1, "section": 1,
    }

    if buffer_count(uid) >= MAX_BUFFERED_QUESTIONS:
        await warn_html(update, "Buffer Limit Reached", f"You have <code>{h(MAX_BUFFERED_QUESTIONS)}</code> questions buffered.\n\nUse <code>/done</code> to export or <code>/clear</code> to reset.")
        return

    buffer_add(uid, payload)

    note = ""
    if correct_answer_id == 0 and poll.type == "quiz":
        note = "\n\n⚠️ Telegram may hide the correct answer in forwarded quizzes. CSV will store <code>answer=0</code>."
    body = f"Total buffered: <code>{buffer_count(uid)}</code>{note}"
    await ok_html(update, "Poll Saved", body)



# ---------------------------
# USER: Forwarded Quiz/Poll → Solve (only when /solve_on)
# ---------------------------
async def handle_user_poll_solver(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    USER mode:
    - If /solve_on is enabled, and user forwards/sends a quiz/poll,
      then bot extracts question + options (+ quiz explanation if any),
      solves it with Gemini first, and provides verify buttons (Perplexity / DeepSeek).
    """
    ensure_user(update)
    if not update.effective_user or not update.message or not update.message.poll:
        return
    uid = update.effective_user.id
    if is_banned(uid):
        return
    if get_role(uid) != ROLE_USER:
        return
    if not solver_mode_on(uid):
        return

    poll = update.message.poll
    qtext = (poll.question or "").strip()
    options = [o.text for o in (poll.options or [])]
    options = [x.strip() for x in options if (x or "").strip()]

    official_expl = ""
    try:
        if getattr(poll, "explanation", None):
            official_expl = str(poll.explanation or "").strip()
    except Exception:
        official_expl = ""

    official_ans = _poll_official_answer(poll)

    spinner_msg = None
    spinner_task = None
    try:
        spinner_msg = await update.message.reply_text("🔎 Searching")
        spinner_task = asyncio.create_task(_spinner_task(context.bot, spinner_msg.chat_id, spinner_msg.message_id))

        data = await _run_blocking('user', gemini_solve_mcq_json, qtext, options)
        model_ans = int(data.get("answer", 0) or 0)
        conf = int(data.get("confidence", 0) or 0)
        
        # --- FIX START: Apply clean_latex ---
        raw_expl = str(data.get("explanation", "") or "").strip()
        model_expl = clean_latex(raw_expl)  # এখানে ক্লিন করা হচ্ছে

        raw_why_not = data.get("why_not", {}) or {}
        why_not = {k: clean_latex(v) for k, v in raw_why_not.items()}
        # --- FIX END ---

        spinner_task.cancel()
        with contextlib.suppress(Exception):
            await context.bot.delete_message(chat_id=spinner_msg.chat_id, message_id=spinner_msg.message_id)

        msg_html = _format_user_poll_solution(
            question=qtext,
            options=options,
            model_ans=model_ans,
            official_ans=official_ans,
            model_expl=f"[Gemini 3 Flash]\n{model_expl}".strip(),
            official_expl=official_expl,
            why_not=why_not if isinstance(why_not, dict) else {},
            conf=conf,
        )

        poll_payload = {
            "question": qtext,
            "options": options,
            "official_ans": official_ans,
            "official_expl": official_expl,
        }
        await send_poll_verify_buttons(update, context, poll_payload, msg_html)

    except Exception as e:
        if spinner_task:
            spinner_task.cancel()
        if spinner_msg:
            with contextlib.suppress(Exception):
                await context.bot.delete_message(chat_id=spinner_msg.chat_id, message_id=spinner_msg.message_id)
        db_log("ERROR", "poll_solver_failed", {"user_id": uid, "error": str(e)})
        await err(update, "Solve Failed", f"{h(str(e)[:160])}")


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin/Owner: photo/image -> extract MCQs + explanations -> buffer."""
    uid = update.effective_user.id

    # Command-based toggle: ignore images unless enabled
    if not vision_mode_on(uid):
        return

    if buffer_count(uid) >= MAX_BUFFERED_QUESTIONS:
        await warn(update, "Buffer Limit Reached", f"You have {MAX_BUFFERED_QUESTIONS} questions buffered.\n\nUse /done to export or /clear to reset.")
        return

    msg = update.message
    tg_file = None
    if msg.photo:
        tg_file = await msg.photo[-1].get_file()
    elif msg.document:
        tg_file = await msg.document.get_file()
    else:
        return

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        local_path = f.name

    await tg_file.download_to_drive(local_path)

    try:
        if not GEMINI_API_KEY or "PASTE_YOUR_GOOGLE_AI_STUDIO_API_KEY_HERE" in GEMINI_API_KEY:
            await safe_reply(
                update,
                "❌ Gemini API Key সেট করা হয়নি।\n\nফাইলের শুরুর দিকে <b>GEMINI_API_KEY</b> এর জায়গায় তোমার key বসাও, তারপর বট restart দাও।"
            )
            return

        items = await _run_blocking(_role_of(uid), gemini_extract_mcq_from_image_rest, local_path)

        added = 0
        for payload in items:
            if buffer_count(uid) >= MAX_BUFFERED_QUESTIONS:
                break
            if not explain_mode_on(uid):
                payload["explanation"] = ""
            buffer_add(uid, payload)
            added += 1

        if added:
            await ok_html(update, "Image Processed", f"<code>{h(added)}</code> question(s) extracted.\n\nTotal buffered: <code>{h(buffer_count(uid))}</code>", footer_html="Use <code>/done</code> to export")
        else:
            await warn(update, "No Questions Found", "No MCQs detected in image. Try a clearer scan or different crop.")

    except Exception as e:
        db_log("ERROR", "image_extract_failed", {"admin_id": uid, "error": str(e)})
        await err(update, "Image Extraction Failed", f"{h(str(e)[:100])}")

    finally:
        with contextlib.suppress(Exception):
            os.remove(local_path)

# For normal users doing unusual things: warn in English + contact owner
async def send_solver_picker(update: Update, context: ContextTypes.DEFAULT_TYPE, problem_text: str) -> None:
    """Send model picker buttons for problem solving."""
    token = _make_token()
    store = _pending_store(context)
    uid = update.effective_user.id
    
    store[token] = {
        "uid": uid,
        "chat_id": update.effective_chat.id if update.effective_chat else uid,
        "kind": "text",
        "payload": {"text": problem_text},
    }
    
    kb = _solver_picker_kb(token)
    msg = ui_box_html("Which AI model?", f"<code>{h(problem_text[:100])}</code>", emoji="🧠")
    await update.message.reply_text(msg, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


async def send_poll_verify_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE, poll_payload: Dict[str, Any], msg_html: str) -> None:
    """Send solved MCQ with verify buttons."""
    token = _make_token()
    store = _pending_store(context)
    uid = update.effective_user.id
    
    store[token] = {
        "uid": uid,
        "chat_id": update.effective_chat.id if update.effective_chat else uid,
        "kind": "poll",
        "payload": poll_payload,
    }
    
    kb = _verify_kb(token, "G", "poll")
    await update.message.reply_text(msg_html, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


async def handle_user_text_unusual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id
    if is_banned(uid):
        await safe_reply(update, f"🚫 Access denied. You are banned.\nContact: {OWNER_CONTACT}")
        return

    
    if get_role(uid) == ROLE_USER:
        # Problem-solving chat (command-based)
        if solver_mode_on(uid):
            user_text = (update.message.text or "").strip()
            if not user_text:
                return
            # Show model picker (Gemini / Perplexity / DeepSeek)
            await send_solver_picker(update, context, user_text)
            return


        # Users can use /ask; random texts without /ask are considered unusual
        await warn_unauthorized(update, "This bot is currently restricted for staff operations. Please use /ask [message] for support.")


# ---------------------------
# ERROR HANDLER
# ---------------------------
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled error: %s", context.error)
    db_log("ERROR", "unhandled_exception", {"error": str(context.error)})


def _cmdh(command, callback, *args, **kwargs):
    """CommandHandler wrapper that supports both /command and .command."""
    try:
        return CommandHandler(command, callback, *args, prefixes=("/", "."), **kwargs)
    except TypeError:
        return CommandHandler(command, callback, *args, **kwargs)


# ---------------------------
# BUILD APP
# ---------------------------


@require_owner
async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT user_id, role, first_name, username, is_banned, created_at, last_seen_at FROM users ORDER BY created_at ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    if not rows:
        await warn(update, "No Users", "No users found.")
        return
    import csv, tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['user_id','role','first_name','username','is_banned','created_at','last_seen_at'])
        w.writeheader(); w.writerows(rows)
        path = f.name
    with open(path, 'rb') as rf:
        await context.bot.send_document(chat_id=update.effective_user.id, document=rf, filename='probaho_users.csv', caption='All started users')
    with contextlib.suppress(Exception):
        os.unlink(path)


def _required_join_kb() -> InlineKeyboardMarkup:
    rows = []
    for r in required_chat_list():
        title = str(r["title"] or r["chat_id"])
        cid = int(r["chat_id"])
        url = None
        try:
            if title.startswith("@"):
                url = f"https://t.me/{title.lstrip('@')}"
        except Exception:
            url = None
        if url:
            rows.append([InlineKeyboardButton(f"Join {title}", url=url)])
    rows.append([InlineKeyboardButton("Verify", callback_data="req:verify")])
    return InlineKeyboardMarkup(rows)


async def on_required_verify_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    q = update.callback_query
    uid = q.from_user.id if q.from_user else 0
    if not uid:
        with contextlib.suppress(Exception):
            await q.answer("User not found.", show_alert=True)
        return
    if is_owner(uid) or is_admin(uid):
        with contextlib.suppress(Exception):
            await q.answer("Verified.", show_alert=False)
        return

    ok, missing = await user_meets_required_memberships(context, uid)
    if ok:
        reset_warn_count(uid)
        with contextlib.suppress(Exception):
            await q.answer("Verification successful.", show_alert=True)
        with contextlib.suppress(Exception):
            if q.message:
                await q.message.delete()
        try:
            chat = q.message.chat_id if q.message else uid
            body_html = (
                f"<b>Your Role:</b> <code>{h(get_role(uid))}</code>"
                f"\n\nUse <code>/help</code> for commands or <code>/commands</code> for a quick list."
            )
            msg = ui_box_html(f"Welcome to {BOT_BRAND}", body_html, emoji="👋")
            await safe_send_text(context.bot, chat, msg)
        except Exception:
            pass
        return

    count = inc_warn_count(uid)
    if count >= 5:
        set_ban(uid, True)
        audit_ban(OWNER_ID, uid, "BAN")
        with contextlib.suppress(Exception):
            await q.answer("You are banned for repeated membership violations.", show_alert=True)
        with contextlib.suppress(Exception):
            if q.message:
                await q.message.edit_text(
                    f"🚫 You are banned from {BOT_BRAND}. Contact: {OWNER_CONTACT}"
                )
        return

    names = ", ".join(missing[:10]) if missing else "required channel/group"
    with contextlib.suppress(Exception):
        await q.answer(f"Still missing: {names}", show_alert=True)
    with contextlib.suppress(Exception):
        if q.message:
            await q.message.edit_text(
                f"⚠️ Please join required chats first.\n\nMissing: {names}\nWarning: {count}/5",
                reply_markup=_required_join_kb()
            )

def build_app() -> Application:
    db_init()
    builder = ApplicationBuilder().token(BOT_TOKEN)
    try:
        builder = builder.concurrent_updates(64)
    except Exception:
        # Older PTB versions may not support concurrent_updates; ignore.
        pass
    app = builder.build()

    # Public
    app.add_handler(_cmdh("start", cmd_start))
    app.add_handler(_cmdh("help", cmd_help))
    app.add_handler(_cmdh("commands", cmd_commands))
    app.add_handler(_cmdh("features", cmd_features))
    app.add_handler(CallbackQueryHandler(on_solver_callback, pattern=r"^solve:"))
    app.add_handler(CallbackQueryHandler(on_genquiz_callback, pattern=r"^genquiz:"))
    app.add_handler(_cmdh("ask", cmd_ask))
    app.add_handler(_cmdh("scanhelp", cmd_scanhelp))
    app.add_handler(_cmdh("vision_on", cmd_vision_on))
    app.add_handler(_cmdh("vision_off", cmd_vision_off))
    app.add_handler(_cmdh("solve_on", cmd_solve_on))
    app.add_handler(_cmdh("solve_off", cmd_solve_off))
    app.add_handler(_cmdh("explain_on", cmd_explain_on))
    app.add_handler(_cmdh("explain_off", cmd_explain_off))

    # Owner only
    app.add_handler(_cmdh("quizprefix", cmd_quizprefix))
    app.add_handler(_cmdh("quizlink", cmd_quizlink))
    app.add_handler(_cmdh("addadmin", cmd_addadmin))
    app.add_handler(_cmdh("removeadmin", cmd_removeadmin))
    app.add_handler(_cmdh("grantall", cmd_grantall))
    app.add_handler(_cmdh("revokeall", cmd_revokeall))
    app.add_handler(_cmdh("grantvision", cmd_grantvision))
    app.add_handler(_cmdh("revokevision", cmd_revokevision))

    # Owner dashboard
    app.add_handler(_cmdh("ownerstats", cmd_ownerstats))
    app.add_handler(_cmdh("users", cmd_users))

    # Admin/Owner
    app.add_handler(_cmdh("filter", cmd_filter))
    app.add_handler(_cmdh("done", cmd_done))
    app.add_handler(_cmdh("clear", cmd_clear))

    app.add_handler(_cmdh("addchannel", cmd_addchannel))
    app.add_handler(_cmdh("listchannels", cmd_listchannels))
    app.add_handler(_cmdh("removechannel", cmd_removechannel))
    app.add_handler(_cmdh("setprefix", cmd_setprefix))
    app.add_handler(_cmdh("setexplink", cmd_setexplink))
    app.add_handler(_cmdh("post", cmd_post))

    app.add_handler(_cmdh("broadcast", cmd_broadcast))
    app.add_handler(_cmdh("adminpanel", cmd_adminpanel))

    app.add_handler(_cmdh("reply", cmd_reply))
    app.add_handler(_cmdh("close", cmd_close))

    app.add_handler(_cmdh("ban", cmd_ban))
    app.add_handler(_cmdh("unban", cmd_unban))
    app.add_handler(_cmdh("banned", cmd_banned))

    app.add_handler(_cmdh("private_send", cmd_private_send))
    app.add_handler(_cmdh("send_private", cmd_private_send))

    # Polls, Images & admin parsing (silent for non-admins)
    app.add_handler(MessageHandler(filters.POLL, handle_poll))
    # USER quiz solver (works only when /solve_on)
    app.add_handler(MessageHandler(filters.POLL, handle_user_poll_solver), group=1)
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))

    # If a USER sends text (non-command), warn them (professional)
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_user_text_unusual), group=1)

    app.add_error_handler(on_error)
    return app



# ===========================
# ADVANCED PATCH ADDON
# ===========================

def mention_user(uid: int, name: str = "User") -> str:
    return f'<a href="tg://user?id={int(uid)}">{h(name or "User")}</a>'

EMOJI_BUTTONS = ["❤️", "😮", "😢", "🥳", "🔥"]


def extra_db_init() -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS required_memberships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL UNIQUE,
        title TEXT,
        chat_type TEXT,
        added_by INTEGER,
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_warnings (
        user_id INTEGER PRIMARY KEY,
        warn_count INTEGER NOT NULL DEFAULT 0,
        last_warn_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS emoji_quizzes (
        quiz_id TEXT PRIMARY KEY,
        channel_chat_id INTEGER NOT NULL,
        message_id INTEGER,
        payload_json TEXT NOT NULL,
        created_by INTEGER,
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS emoji_quiz_responses (
        quiz_id TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        selected_option INTEGER NOT NULL,
        is_correct INTEGER NOT NULL DEFAULT 0,
        clicked_at TEXT NOT NULL,
        PRIMARY KEY (quiz_id, user_id)
    )
    """)
    conn.commit()
    conn.close()


def required_chat_add(chat_id: int, title: str, chat_type: str, added_by: int) -> None:
    conn = db_connect(); cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO required_memberships(chat_id,title,chat_type,added_by,created_at) VALUES (?,?,?,?,?)",
        (int(chat_id), title or "", chat_type or "", int(added_by), now_iso()),
    )
    conn.commit(); conn.close()


def required_chat_remove(chat_id: int) -> bool:
    conn = db_connect(); cur = conn.cursor()
    cur.execute("DELETE FROM required_memberships WHERE chat_id=?", (int(chat_id),))
    ok = cur.rowcount > 0
    conn.commit(); conn.close()
    return ok


def required_chat_list() -> List[sqlite3.Row]:
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT * FROM required_memberships ORDER BY id ASC")
    rows = cur.fetchall(); conn.close()
    return rows


def get_warn_count(user_id: int) -> int:
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT warn_count FROM user_warnings WHERE user_id=?", (int(user_id),))
    row = cur.fetchone(); conn.close()
    return int(row["warn_count"] or 0) if row else 0


def inc_warn_count(user_id: int) -> int:
    conn = db_connect(); cur = conn.cursor()
    current = get_warn_count(user_id)
    new_count = current + 1
    cur.execute(
        "INSERT INTO user_warnings(user_id,warn_count,last_warn_at) VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET warn_count=excluded.warn_count,last_warn_at=excluded.last_warn_at",
        (int(user_id), new_count, now_iso()),
    )
    conn.commit(); conn.close()
    return new_count


def reset_warn_count(user_id: int) -> None:
    conn = db_connect(); cur = conn.cursor()
    cur.execute("DELETE FROM user_warnings WHERE user_id=?", (int(user_id),))
    conn.commit(); conn.close()


def set_group_ai_enabled(chat_id: int, value: bool) -> None:
    set_setting(f"group_ai_enabled:{int(chat_id)}", "1" if value else "0")


def is_group_ai_enabled(chat_id: int) -> bool:
    return get_setting(f"group_ai_enabled:{int(chat_id)}", "0") == "1"


def is_private_chat(update: Update) -> bool:
    try:
        return (update.effective_chat.type == "private")
    except Exception:
        return False


async def user_meets_required_memberships(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> Tuple[bool, List[str]]:
    rows = required_chat_list()
    if not rows:
        return True, []
    missing = []
    for r in rows:
        cid = int(r["chat_id"])
        title = str(r["title"] or cid)
        try:
            member = await context.bot.get_chat_member(cid, user_id)
            status = str(getattr(member, "status", ""))
            if status in ("left", "kicked"):
                missing.append(title)
        except Exception:
            missing.append(title)
    return (len(missing) == 0), missing


async def enforce_required_memberships(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    ensure_user(update)
    uid = update.effective_user.id if update.effective_user else 0
    if not uid or is_owner(uid) or is_admin(uid):
        return True
    ok, missing = await user_meets_required_memberships(context, uid)
    if ok:
        reset_warn_count(uid)
        return True
    count = inc_warn_count(uid)
    if count >= 5:
        set_ban(uid, True)
        audit_ban(OWNER_ID, uid, "BAN")
        with contextlib.suppress(Exception):
            await safe_send_text(context.bot, uid, f"🚫 You are banned from <b>{h(BOT_BRAND)}</b> for leaving required channel/group. Contact: {h(OWNER_CONTACT)}")
        return False
    names = ", ".join(missing[:10]) if missing else "required channel/group"
    if update.message:
        await warn(update, "Join Required", f"Please join: {names}\n\nWarning: {count}/5")
    return False


def _copyable_quiz_block(question: str, options: List[str]) -> str:
    parts = [question.strip(), ""]
    for i, o in enumerate(options, start=1):
        parts.append(f"{_safe_letter(i)}) {o}")
    raw = "\n".join(parts).strip()
    return f"<pre>{h(raw)}</pre>"


def _remember_quiz_context(context: ContextTypes.DEFAULT_TYPE, message_id: int, payload: Dict[str, Any]) -> None:
    store = context.application.bot_data.get("_quiz_context")
    if not isinstance(store, dict):
        store = {}
        context.application.bot_data["_quiz_context"] = store
    store[int(message_id)] = dict(payload)
    if len(store) > 2000:
        for k in list(store.keys())[:500]:
            store.pop(k, None)


def _get_quiz_context(context: ContextTypes.DEFAULT_TYPE, message_id: int) -> Optional[Dict[str, Any]]:
    store = context.application.bot_data.get("_quiz_context")
    if not isinstance(store, dict):
        return None
    item = store.get(int(message_id))
    return item if isinstance(item, dict) else None


def emoji_quiz_save(quiz_id: str, channel_chat_id: int, message_id: int, payload: Dict[str, Any], created_by: int) -> None:
    conn = db_connect(); cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO emoji_quizzes(quiz_id,channel_chat_id,message_id,payload_json,created_by,created_at) VALUES (?,?,?,?,?,?)",
        (quiz_id, int(channel_chat_id), int(message_id), json.dumps(payload, ensure_ascii=False), int(created_by), now_iso()),
    )
    conn.commit(); conn.close()


def emoji_quiz_get(quiz_id: str) -> Optional[Dict[str, Any]]:
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT * FROM emoji_quizzes WHERE quiz_id=?", (str(quiz_id),))
    row = cur.fetchone(); conn.close()
    if not row:
        return None
    data = json.loads(row["payload_json"])
    data["quiz_id"] = quiz_id
    data["channel_chat_id"] = int(row["channel_chat_id"])
    data["message_id"] = int(row["message_id"] or 0)
    return data


def emoji_quiz_has_answered(quiz_id: str, user_id: int) -> bool:
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT 1 FROM emoji_quiz_responses WHERE quiz_id=? AND user_id=?", (str(quiz_id), int(user_id)))
    row = cur.fetchone(); conn.close()
    return bool(row)


def emoji_quiz_user_choice(quiz_id: str, user_id: int) -> int:
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT selected_option FROM emoji_quiz_responses WHERE quiz_id=? AND user_id=?", (str(quiz_id), int(user_id)))
    row = cur.fetchone(); conn.close()
    return int(row["selected_option"] or 0) if row else 0


def emoji_quiz_record_answer(quiz_id: str, user_id: int, selected_option: int, is_correct: bool) -> None:
    conn = db_connect(); cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO emoji_quiz_responses(quiz_id,user_id,selected_option,is_correct,clicked_at) VALUES (?,?,?,?,?)",
        (str(quiz_id), int(user_id), int(selected_option), 1 if is_correct else 0, now_iso()),
    )
    conn.commit(); conn.close()


def emoji_quiz_counts(quiz_id: str) -> Dict[int, int]:
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT selected_option, COUNT(*) AS c FROM emoji_quiz_responses WHERE quiz_id=? GROUP BY selected_option", (str(quiz_id),))
    rows = cur.fetchall(); conn.close()
    return {int(r["selected_option"]): int(r["c"]) for r in rows}


def emoji_quiz_keyboard(num_options: int, quiz_id: str) -> InlineKeyboardMarkup:
    buttons = []
    row = []
    for i in range(num_options):
        row.append(InlineKeyboardButton(EMOJI_BUTTONS[i], callback_data=f"eq:{quiz_id}:{i+1}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(buttons)


@require_owner
async def cmd_addrequired(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await safe_reply(update, usage_box("addrequired", "<@channel|group|-100...>", "Add required channel/group for all normal users"))
        return
    ref = context.args[0].strip()
    try:
        chat = await context.bot.get_chat(int(ref) if ref.lstrip('-').isdigit() else ref)
        required_chat_add(chat.id, chat.title or chat.username or str(chat.id), getattr(chat, "type", ""), update.effective_user.id)
        await ok_html(update, "Required Chat Added", f"{h(chat.title or chat.username or chat.id)}\n<code>{h(chat.id)}</code>")
    except Exception as e:
        await err(update, "Failed", str(e)[:180])


@require_owner
async def cmd_delrequired(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await safe_reply(update, usage_box("delrequired", "<chat_id>", "Remove required channel/group"))
        return
    cid = to_int(context.args[0])
    if not cid:
        await err(update, "Invalid Input", "Invalid chat id")
        return
    if required_chat_remove(cid):
        await ok(update, "Removed", f"Required chat removed: {cid}")
    else:
        await warn(update, "Not Found", f"Required chat not found: {cid}")


@require_owner
async def cmd_listrequired(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = required_chat_list()
    if not rows:
        await warn(update, "No Required Chats", "No required membership configured.")
        return
    body = "\n\n".join([f"<b>{h(r['title'] or '')}</b>\n<code>{h(r['chat_id'])}</code>\nType: {h(r['chat_type'] or '')}" for r in rows])
    await ok_html(update, "Required Memberships", body, emoji="📌")


@require_admin
async def cmd_himusai_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_private_chat(update):
        await warn(update, "Private Only", "This command works in inbox/private chat only.")
        return
    set_himusai_mode_on(update.effective_user.id, True)
    await ok_html(update, "HimusAI Enabled", "Admin/Owner inbox auto-response enabled.", emoji="🧠")


@require_admin
async def cmd_himusai_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_private_chat(update):
        await warn(update, "Private Only", "This command works in inbox/private chat only.")
        return
    set_himusai_mode_on(update.effective_user.id, False)
    await ok_html(update, "HimusAI Disabled", "Admin/Owner inbox auto-response disabled.", emoji="🧠")


@require_admin
async def cmd_probaho_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if not chat or chat.type not in ("group", "supergroup"):
        await warn(update, "Group Only", "Use this command inside a group/supergroup.")
        return
    set_group_ai_enabled(chat.id, True)
    await ok(update, "Group AI Enabled", f"Users can now get AI responses in this group: {chat.id}")


@require_admin
async def cmd_probaho_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if not chat or chat.type not in ("group", "supergroup"):
        await warn(update, "Group Only", "Use this command inside a group/supergroup.")
        return
    set_group_ai_enabled(chat.id, False)
    await ok(update, "Group AI Disabled", f"Users will no longer get AI responses in this group: {chat.id}")


@require_admin
async def cmd_postemoji(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("postemoji", "<DB-ID> [keep]", "Post buffered questions as emoji quiz to a channel"))
        return
    cid = int(context.args[0])
    keep = (len(context.args) > 1 and context.args[1].strip().lower() == "keep")
    ch = channel_get_by_id_for_user(admin_id, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or no access.")
        return
    items = buffer_list(admin_id, limit=MAX_BUFFERED_QUESTIONS)
    if not items:
        await warn(update, "Buffer Empty", "No buffered questions found.")
        return
    sent = 0
    sent_ids = []
    for bid, payload in items:
        q, opts, corr_idx0, explanation = quiz_to_poll_parts(payload)
        block = _copyable_quiz_block(q, opts)
        msg_html = f"<b>📚 Emoji Quiz</b>\n\n{block}"
        try:
            m = await context.bot.send_message(chat_id=ch.channel_chat_id, text=msg_html, parse_mode=ParseMode.HTML, reply_markup=emoji_quiz_keyboard(len(opts), uuid.uuid4().hex[:10]), disable_web_page_preview=True)
            sent += 1
            sent_ids.append(bid)
            # fix quiz_id from keyboard callback data
            quiz_id = None
            try:
                quiz_id = m.reply_markup.inline_keyboard[0][0].callback_data.split(":")[1]
            except Exception:
                quiz_id = uuid.uuid4().hex[:10]
            emoji_quiz_save(quiz_id, ch.channel_chat_id, m.message_id, {"question": q, "options": opts, "correct_answer": corr_idx0 + 1 if corr_idx0 >= 0 else 0, "explanation": explanation}, admin_id)
            await asyncio.sleep(0.25)
        except Exception as e:
            db_log("ERROR", "postemoji_failed", {"admin_id": admin_id, "channel": ch.channel_chat_id, "error": str(e)})
    if sent and not keep:
        buffer_remove_ids(admin_id, sent_ids)
    await ok_html(update, "Emoji Quiz Posted", f"Sent: <code>{h(sent)}</code>\nChannel: <code>{h(ch.title)}</code>")


async def on_emoji_quiz_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    q = update.callback_query
    data = (q.data or "").strip()
    m = re.match(r"^eq:([0-9a-f]{6,16}):(\d+)$", data)
    if not m:
        return
    quiz_id = m.group(1)
    selected = int(m.group(2))
    uid = q.from_user.id if q.from_user else 0
    if not uid:
        return
    if emoji_quiz_has_answered(quiz_id, uid):
        await q.answer("আপনি ইতোমধ্যে উত্তর দিয়েছেন।", show_alert=True)
        return
    quiz = emoji_quiz_get(quiz_id)
    if not quiz:
        await q.answer("Quiz expired or not found.", show_alert=True)
        return
    correct = int(quiz.get("correct_answer", 0) or 0)
    is_correct = (selected == correct and correct > 0)
    emoji_quiz_record_answer(quiz_id, uid, selected, is_correct)
    counts = emoji_quiz_counts(quiz_id)
    opts = quiz.get("options", []) or []
    stats = []
    for i, opt in enumerate(opts, start=1):
        stats.append(f"{_safe_letter(i)}) {opt} — {counts.get(i,0)}")
    stats_text = "\n".join(stats)
    expl = str(quiz.get("explanation", "") or "").strip()
    if expl:
        expl = clean_latex(expl)
    if is_correct:
        msg = f"🎉 Congratulations!\n\n✅ Correct answer: {_safe_letter(correct)}\n\n{expl}\n\nStats:\n{stats_text}".strip()
    else:
        msg = f"❌ Wrong answer\n✅ Correct: {_safe_letter(correct)}\n\n{expl}\n\nStats:\n{stats_text}".strip()
    await q.answer("Answer recorded.", show_alert=False)
    # Try DM first so only that user sees analysis
    delivered = False
    with contextlib.suppress(Exception):
        await context.bot.send_message(chat_id=uid, text=msg)
        delivered = True
    if not delivered:
        with contextlib.suppress(Exception):
            await q.answer(msg[:180], show_alert=True)


async def cmd_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    items = buffer_list(uid, limit=99999)
    if not items:
        await warn(update, "Buffer Empty", "No questions to export. Use /add or send quizzes first.")
        return
    rows = [payload for (_id, payload) in items]
    norm_rows = []
    explanations_enabled = explain_mode_on(uid)
    for r in rows:
        q = str(r.get("questions", "") or "")
        e = str(r.get("explanation", "") or "")
        q2, expl2 = split_inline_explain(q)
        if expl2 and not e.strip():
            e = expl2
        rr = dict(r)
        rr["questions"] = q2.strip()
        rr["explanation"] = e.strip() if explanations_enabled else ""
        norm_rows.append(rr)
    rows = norm_rows
    df = pd.DataFrame(rows)
    cols = ["questions", "option1", "option2", "option3", "option4", "option5", "answer", "explanation", "type", "section"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    with tempfile.NamedTemporaryFile("w+b", suffix=".csv", delete=False) as f:
        path = f.name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    def _ans_to_letter(n: int) -> str:
        return {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}.get(int(n or 0), "")
    quiz_json = []
    for r in rows:
        opts_map = {"A": r.get("option1", ""), "B": r.get("option2", ""), "C": r.get("option3", ""), "D": r.get("option4", "")}
        if str(r.get("option5", "")).strip():
            opts_map["E"] = r.get("option5", "")
        quiz_json.append({
            "question": r.get("questions", ""),
            "options": opts_map,
            "correct_answer": _ans_to_letter(r.get("answer", 0)),
            "explanation": r.get("explanation", "") if explanations_enabled else "",
        })
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as jf:
        json_path = jf.name
        json.dump(quiz_json, jf, ensure_ascii=False, indent=2)
    try:
        await update.message.reply_document(document=open(path, "rb"), caption=f"<b>✅ CSV Export</b>\n<i>{len(df)} questions exported</i>", parse_mode=ParseMode.HTML)
        await update.message.reply_document(document=open(json_path, "rb"), caption="<b>✅ JSON Export</b>", parse_mode=ParseMode.HTML)
        await ok_html(update, "Export Complete", f"CSV + JSON ready. <code>{h(len(df))}</code> questions exported.")
    finally:
        with contextlib.suppress(Exception): os.remove(path)
        with contextlib.suppress(Exception): os.remove(json_path)
    buffer_clear(uid)


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id
    if not await enforce_required_memberships(update, context):
        return
    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\nContact: {OWNER_CONTACT}")
        return
    if not await enforce_required_memberships(update, context):
        return
    replied = update.message.reply_to_message if update.message else None
    text = " ".join(context.args).strip()
    if not text:
        text = reply_text_or_caption(update)
    if not text and not replied:
        await safe_reply(update, usage_box("ask", "<message>", "Ask a support question (or reply to message/file/photo)"))
        return
    tid = ticket_find_open_by_student(uid)
    if tid is None:
        tid = ticket_open(uid, update.effective_user.first_name or "")
        db_log("INFO", "ticket_open", {"ticket_id": tid, "student_id": uid})
    if text:
        ticket_add_msg(tid, "STUDENT", uid, text)
    elif replied:
        ticket_add_msg(tid, "STUDENT", uid, "[MEDIA MESSAGE]")
    staff_ids = list_staff_ids()
    profile = mention_user(uid, update.effective_user.first_name or str(uid))
    uname = f"@{update.effective_user.username}" if getattr(update.effective_user, 'username', None) else ""
    header = f"📩 New Support Message\nTicket: <code>{tid}</code>\nFrom: {profile} <code>{uid}</code> {h(uname)}"
    if text:
        for sid in staff_ids:
            await safe_send_text(context.bot, sid, f"{header}\n\n<pre>{h(text)}</pre>")
    else:
        for sid in staff_ids:
            await safe_send_text(context.bot, sid, f"{header}\n\n[MEDIA MESSAGE RECEIVED]")
    if replied:
        for sid in staff_ids:
            await safe_copy_message(context.bot, chat_id=sid, from_chat_id=replied.chat_id, message_id=replied.message_id, protect=False)
    await ok(update, "Message Received", f"Ticket ID: {tid}\nA staff member will respond soon.")


@require_admin
async def cmd_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("reply", "<ticket_id> [message]", "Reply to support ticket (or reply to message/file/photo)"))
        return
    tid = int(context.args[0]); text = " ".join(context.args[1:]).strip(); replied = update.message.reply_to_message if update.message else None
    if not text:
        text = reply_text_or_caption(update)
    tr = ticket_get(tid)
    if not tr:
        await warn_html(update, "Ticket Not Found", f"No ticket with ID <code>{h(tid)}</code> found")
        return
    if tr["status"] != "OPEN":
        await err_html(update, "Ticket Closed", f"Ticket <code>{h(tid)}</code> is already <b>CLOSED</b>")
        return
    student_id = int(tr["student_id"])
    if is_banned(student_id):
        await warn(update, "User Banned", "The user is currently banned.")
        return
    sent_any = False
    if text:
        ticket_add_msg(tid, "STAFF", update.effective_user.id, text)
        await safe_send_text(context.bot, student_id, f"💬 Support Reply (Ticket <code>{tid}</code>)\n\n<pre>{h(text)}</pre>")
        sent_any = True
    if replied:
        okc = await safe_copy_message(context.bot, chat_id=student_id, from_chat_id=replied.chat_id, message_id=replied.message_id, protect=False)
        if okc:
            ticket_add_msg(tid, "STAFF", update.effective_user.id, "[MEDIA MESSAGE]")
            sent_any = True
    if sent_any:
        await ok_html(update, "Reply Sent", f"<b>Ticket:</b> <code>{h(tid)}</code>\nMessage(s) sent to user.")
    else:
        await warn(update, "No Content", "Reply to a message/file/photo or provide text inline")


def _format_user_poll_solution(question: str, options: List[str], model_ans: int, official_ans: int, model_expl: str, official_expl: str, why_not: Dict[str, str], conf: int) -> str:
    opts = [(o or "").strip() for o in (options or []) if (o or "").strip()][:5]
    copy_block = _copyable_quiz_block(question or "", opts)
    lines = ["<b>📊 Quiz Solution</b>", "", "<b>Question + Options (copyable):</b>", copy_block]
    if 1 <= int(model_ans or 0) <= len(opts):
        lines.append(f"\n<b>✅ AI Response:</b> <b>{_safe_letter(model_ans)}</b>) {h(opts[model_ans-1])}")
    if official_ans > 0 and official_ans <= len(opts):
        tag = "✅ Match" if official_ans == model_ans else "❌ Mismatch"
        lines.append(f"<b>📌 Given Answer:</b> <b>{_safe_letter(official_ans)}</b>) {h(opts[official_ans-1])} <i>({tag})</i>")
    if model_expl:
        lines.append(f"\n<b>Explanation (Solved):</b>\n<pre>{h(model_expl)}</pre>")
    if official_expl:
        lines.append(f"\n<b>Explanation (From Quiz):</b>\n<pre>{h(official_expl)}</pre>")
    if why_not:
        wn = []
        for k in ["A","B","C","D","E"]:
            v = (why_not or {}).get(k)
            if v:
                wn.append(f"• <b>{h(k)}</b>: {h(v)}")
        if wn:
            lines.append("\n<b>Why other options are wrong:</b>\n" + "\n".join(wn))
    return "\n".join(lines).strip()


async def on_solver_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    q = update.callback_query
    await q.answer("Processing…", show_alert=False)
    data = (q.data or "").strip()
    m = re.match(r"^solve:([GPD]):([0-9a-f]{6,16})$", data)
    if not m:
        return
    model = m.group(1); token = m.group(2)
    store = _pending_store(context); req = store.get(token)
    if not isinstance(req, dict):
        with contextlib.suppress(Exception):
            await q.edit_message_text("⚠️ This request has expired. Please send your question again.")
        return
    uid = int(req.get("uid") or 0)
    if q.from_user and q.from_user.id != uid:
        with contextlib.suppress(Exception):
            await q.answer("This is not your request.", show_alert=True)
        return
    payload = req.get("payload") or {}
    problem_text = str(payload.get("text") or "").strip(); kind = str(req.get("kind") or "text").lower()
    with contextlib.suppress(Exception):
        await q.edit_message_text(ui_box_text("Solving", "Please wait… Processing your request.", emoji="⏳"), parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    try:
        if kind == "poll" and payload.get("question"):
            question = str(payload.get("question", "")).strip(); options = payload.get("options", [])
            if model == "G": result = await _run_blocking(_role_of(uid), gemini_solve_mcq_json, question, options)
            elif model == "P": result = await _run_blocking(_role_of(uid), perplexity_solve_mcq_json, question, options)
            else: result = await _run_blocking(_role_of(uid), deepseek_solve_mcq_json, question, options)
            raw_expl = str(result.get('explanation', '') or ""); clean_expl = clean_latex(raw_expl)
            raw_why_not = result.get("why_not", {}) or {}; clean_why_not = {k: clean_latex(v) for k, v in raw_why_not.items()}
            msg_html = _format_user_poll_solution(question=question, options=options, model_ans=int(result.get("answer", 0) or 0), official_ans=int(payload.get("official_ans", 0) or 0), model_expl=f"[{['Gemini', 'Perplexity', 'DeepSeek'][['G','P','D'].index(model)]}]\n{clean_expl}".strip(), official_expl=str(payload.get("official_expl", "")).strip(), why_not=clean_why_not, conf=int(result.get("confidence", 0) or 0))
            kb = _verify_kb(token, model, "poll")
        else:
            if model == "G": answer = await _run_blocking(_role_of(uid), gemini_solve_text, problem_text)
            elif model == "P": answer = await _run_blocking(_role_of(uid), perplexity_solve_text, problem_text)
            else: answer = await _run_blocking(_role_of(uid), deepseek_solve_text, problem_text)
            msg_html = f"<pre>{h(answer)}</pre>"
            kb = _verify_kb(token, model, "text")
        with contextlib.suppress(Exception):
            await q.edit_message_text(msg_html, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            if q.message and kind == "poll":
                _remember_quiz_context(context, q.message.message_id, payload)
    except Exception as e:
        db_log("ERROR", "solver_callback_failed", {"user_id": uid, "model": model, "error": str(e)})
        with contextlib.suppress(Exception):
            await q.edit_message_text(ui_box_text("Solve Failed", str(e)[:180], emoji="❌"), parse_mode=ParseMode.HTML)


async def send_poll_verify_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE, poll_payload: Dict[str, Any], msg_html: str) -> None:
    token = _make_token(); store = _pending_store(context); uid = update.effective_user.id
    store[token] = {"uid": uid, "chat_id": update.effective_chat.id if update.effective_chat else uid, "kind": "poll", "payload": poll_payload}
    kb = _verify_kb(token, "G", "poll")
    sent = await update.message.reply_text(msg_html, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    _remember_quiz_context(context, sent.message_id, poll_payload)


async def handle_user_poll_solver(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    if not update.effective_user or not update.message or not update.message.poll:
        return
    uid = update.effective_user.id
    if is_banned(uid):
        return
    if not await enforce_required_memberships(update, context):
        return
    role = get_role(uid)
    private = is_private_chat(update)
    if role == ROLE_USER:
        if not solver_mode_on(uid):
            return
        if not private and not is_group_ai_enabled(update.effective_chat.id):
            return
    elif role in (ROLE_ADMIN, ROLE_OWNER):
        if not private or not solver_mode_on(uid):
            return
    else:
        return
    poll = update.message.poll
    qtext = (poll.question or "").strip(); options = [o.text for o in (poll.options or [])]; options = [x.strip() for x in options if (x or "").strip()]
    official_expl = str(getattr(poll, "explanation", "") or "").strip(); official_ans = _poll_official_answer(poll)
    spinner_msg = None; spinner_task = None
    try:
        spinner_msg = await update.message.reply_text("🔎 Searching")
        spinner_task = asyncio.create_task(_spinner_task(context.bot, spinner_msg.chat_id, spinner_msg.message_id))
        data = await _run_blocking('user', gemini_solve_mcq_json, qtext, options)
        model_ans = int(data.get("answer", 0) or 0); conf = int(data.get("confidence", 0) or 0)
        raw_expl = str(data.get("explanation", "") or "").strip(); model_expl = clean_latex(raw_expl)
        raw_why_not = data.get("why_not", {}) or {}; why_not = {k: clean_latex(v) for k, v in raw_why_not.items()}
        spinner_task.cancel()
        with contextlib.suppress(Exception): await context.bot.delete_message(chat_id=spinner_msg.chat_id, message_id=spinner_msg.message_id)
        msg_html = _format_user_poll_solution(question=qtext, options=options, model_ans=model_ans, official_ans=official_ans, model_expl=f"[Gemini 3 Flash]\n{model_expl}".strip(), official_expl=official_expl, why_not=why_not if isinstance(why_not, dict) else {}, conf=conf)
        poll_payload = {"question": qtext, "options": options, "official_ans": official_ans, "official_expl": official_expl}
        await send_poll_verify_buttons(update, context, poll_payload, msg_html)
    except Exception as e:
        if spinner_task: spinner_task.cancel()
        if spinner_msg:
            with contextlib.suppress(Exception): await context.bot.delete_message(chat_id=spinner_msg.chat_id, message_id=spinner_msg.message_id)
        db_log("ERROR", "poll_solver_failed", {"user_id": uid, "error": str(e)})
        await err(update, "Solve Failed", f"{h(str(e)[:160])}")


async def handle_user_text_unusual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id
    if is_banned(uid):
        return
    if not await enforce_required_memberships(update, context):
        return
    role = get_role(uid)
    private = is_private_chat(update)
    if role == ROLE_USER:
        if not solver_mode_on(uid):
            if private:
                await warn_unauthorized(update, "This bot is currently restricted for staff operations. Please use /ask [message] for support.")
            return
        if not private and not is_group_ai_enabled(update.effective_chat.id):
            return
    elif role in (ROLE_ADMIN, ROLE_OWNER):
        if not private or not solver_mode_on(uid):
            return
    else:
        return
    user_text = (update.message.text or "").strip()
    if not user_text:
        return
    # If user replied to a previous quiz solution, ask with that quiz context
    reply_msg = update.message.reply_to_message
    if reply_msg:
        ctx = _get_quiz_context(context, reply_msg.message_id)
        if ctx:
            qtext = str(ctx.get("question", "") or "").strip()
            opts = ctx.get("options", []) or []
            prompt = f"Question:\n{qtext}\n\nOptions:\n" + "\n".join([f"{_safe_letter(i+1)}. {o}" for i,o in enumerate(opts)]) + f"\n\nUser follow-up:\n{user_text}"
            await send_solver_picker(update, context, prompt)
            return
    await send_solver_picker(update, context, user_text)


def build_app() -> Application:
    db_init(); extra_db_init()
    builder = ApplicationBuilder().token(BOT_TOKEN)
    try:
        builder = builder.concurrent_updates(64)
    except Exception:
        pass
    app = builder.build()
    app.add_handler(_cmdh("start", cmd_start))
    app.add_handler(_cmdh("help", cmd_help))
    app.add_handler(_cmdh("commands", cmd_commands))
    app.add_handler(_cmdh("features", cmd_features))
    app.add_handler(CallbackQueryHandler(on_solver_callback, pattern=r"^solve:"))
    app.add_handler(CallbackQueryHandler(on_genquiz_callback, pattern=r"^genquiz:"))
    app.add_handler(CallbackQueryHandler(on_emoji_quiz_callback, pattern=r"^eq:"))
    app.add_handler(CallbackQueryHandler(on_required_verify_callback, pattern=r"^req:verify$"))
    app.add_handler(_cmdh("ask", cmd_ask))
    app.add_handler(_cmdh("scanhelp", cmd_scanhelp))
    app.add_handler(_cmdh("vision_on", cmd_vision_on))
    app.add_handler(_cmdh("vision_off", cmd_vision_off))
    app.add_handler(_cmdh("solve_on", cmd_solve_on))
    app.add_handler(_cmdh("solve_off", cmd_solve_off))
    app.add_handler(_cmdh("himusai_on", cmd_himusai_on))
    app.add_handler(_cmdh("himusai_off", cmd_himusai_off))
    app.add_handler(_cmdh("probaho_on", cmd_probaho_on))
    app.add_handler(_cmdh("probaho_off", cmd_probaho_off))
    app.add_handler(_cmdh("explain_on", cmd_explain_on))
    app.add_handler(_cmdh("explain_off", cmd_explain_off))
    app.add_handler(_cmdh("quizprefix", cmd_quizprefix))
    app.add_handler(_cmdh("quizlink", cmd_quizlink))
    app.add_handler(_cmdh("addadmin", cmd_addadmin))
    app.add_handler(_cmdh("removeadmin", cmd_removeadmin))
    app.add_handler(_cmdh("grantall", cmd_grantall))
    app.add_handler(_cmdh("revokeall", cmd_revokeall))
    app.add_handler(_cmdh("grantvision", cmd_grantvision))
    app.add_handler(_cmdh("revokevision", cmd_revokevision))
    app.add_handler(_cmdh("addrequired", cmd_addrequired))
    app.add_handler(_cmdh("delrequired", cmd_delrequired))
    app.add_handler(_cmdh("listrequired", cmd_listrequired))
    app.add_handler(_cmdh("ownerstats", cmd_ownerstats))
    app.add_handler(_cmdh("users", cmd_users))
    app.add_handler(_cmdh("filter", cmd_filter))
    app.add_handler(_cmdh("done", cmd_done))
    app.add_handler(_cmdh("clear", cmd_clear))
    app.add_handler(_cmdh("addchannel", cmd_addchannel))
    app.add_handler(_cmdh("listchannels", cmd_listchannels))
    app.add_handler(_cmdh("removechannel", cmd_removechannel))
    app.add_handler(_cmdh("setprefix", cmd_setprefix))
    app.add_handler(_cmdh("setexplink", cmd_setexplink))
    app.add_handler(_cmdh("post", cmd_post))
    app.add_handler(_cmdh("postemoji", cmd_postemoji))
    app.add_handler(_cmdh("broadcast", cmd_broadcast))
    app.add_handler(_cmdh("adminpanel", cmd_adminpanel))
    app.add_handler(_cmdh("reply", cmd_reply))
    app.add_handler(_cmdh("close", cmd_close))
    app.add_handler(_cmdh("ban", cmd_ban))
    app.add_handler(_cmdh("unban", cmd_unban))
    app.add_handler(_cmdh("banned", cmd_banned))
    app.add_handler(_cmdh("private_send", cmd_private_send))
    app.add_handler(_cmdh("send_private", cmd_private_send))
    app.add_handler(MessageHandler(filters.POLL, handle_poll))
    app.add_handler(MessageHandler(filters.POLL, handle_user_poll_solver), group=1)
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_user_text_unusual), group=1)
    app.add_error_handler(on_error)
    return app


# ===========================
# FINAL PATCH OVERRIDES
# ===========================
REQUIRED_DEFAULT_JOIN_URL = "https://t.me/FX_Ur_Target"

async def _is_group_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
    try:
        member = await context.bot.get_chat_member(chat_id, user_id)
        return str(getattr(member, "status", "")) in ("administrator", "creator")
    except Exception:
        return False

async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT user_id, role, first_name, username, is_banned, created_at, last_seen_at FROM users ORDER BY created_at ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    if not rows:
        await warn(update, "No Users", "No users found.")
        return
    import tempfile, json
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        path = f.name
    with open(path, "rb") as rf:
        await context.bot.send_document(chat_id=update.effective_user.id, document=rf, filename="probaho_users.json", caption="All started users")
    with contextlib.suppress(Exception):
        os.unlink(path)

def _required_join_kb() -> InlineKeyboardMarkup:
    rows = []
    for r in required_chat_list():
        title = str(r["title"] or r["chat_id"])
        if title.startswith("@"):
            url = f"https://t.me/{title.lstrip('@')}"
        elif "t.me/" in title:
            url = title if title.startswith("http") else ("https://" + title.lstrip("/"))
        else:
            url = REQUIRED_DEFAULT_JOIN_URL
        rows.append([InlineKeyboardButton(f"Join {title}", url=url)])
    rows.append([InlineKeyboardButton("Verify", callback_data="req:verify")])
    return InlineKeyboardMarkup(rows)

async def enforce_required_memberships(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    ensure_user(update)
    uid = update.effective_user.id if update.effective_user else 0
    if not uid or is_owner(uid) or is_admin(uid):
        return True
    ok, missing = await user_meets_required_memberships(context, uid)
    if ok:
        reset_warn_count(uid)
        return True
    count = inc_warn_count(uid)
    if count >= 5:
        set_ban(uid, True)
        audit_ban(OWNER_ID, uid, "BAN")
        with contextlib.suppress(Exception):
            await safe_send_text(context.bot, uid, f"🚫 You are banned from <b>{h(BOT_BRAND)}</b> for leaving required channel/group. Contact: {h(OWNER_CONTACT)}")
        return False
    names = ", ".join(missing[:10]) if missing else "required channel/group"
    if update.message:
        try:
            await update.message.reply_text(f"⚠️ Join Required\nPlease join: {names}\nWarning: {count}/5", reply_markup=_required_join_kb())
        except Exception:
            await warn(update, "Join Required", f"Please join: {names}\n\nWarning: {count}/5")
    return False

@require_owner
async def cmd_addrequired(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await safe_reply(update, usage_box("addrequired", "<@channel|group|-100...>", "Add required channel/group for all normal users"))
        return
    ref = context.args[0].strip()
    try:
        chat = await context.bot.get_chat(int(ref) if ref.lstrip("-").isdigit() else ref)
        title = ("@" + chat.username) if getattr(chat, "username", None) else (chat.title or str(chat.id))
        required_chat_add(chat.id, title, getattr(chat, "type", ""), update.effective_user.id)
        await ok_html(update, "Required Chat Added", f"{h(title)}\n<code>{h(chat.id)}</code>")
    except Exception as e:
        await err(update, "Failed", str(e)[:180])

async def cmd_probaho_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    uid = update.effective_user.id if update.effective_user else 0
    if not chat or chat.type not in ("group", "supergroup"):
        await warn(update, "Group Only", "Use this command inside a group/supergroup.")
        return
    if not await _is_group_admin(context, chat.id, uid):
        await warn(update, "Unauthorized", "Only a group admin can use this command.")
        return
    set_group_ai_enabled(chat.id, True)
    await ok(update, "Group AI Enabled", f"Users can now get AI responses in this group: {chat.id}")

async def cmd_probaho_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    uid = update.effective_user.id if update.effective_user else 0
    if not chat or chat.type not in ("group", "supergroup"):
        await warn(update, "Group Only", "Use this command inside a group/supergroup.")
        return
    if not await _is_group_admin(context, chat.id, uid):
        await warn(update, "Unauthorized", "Only a group admin can use this command.")
        return
    set_group_ai_enabled(chat.id, False)
    await ok(update, "Group AI Disabled", f"Users will no longer get AI responses in this group: {chat.id}")

def _copyable_quiz_block(question: str, options: List[str], labels: Optional[List[str]] = None) -> str:
    parts = [question.strip(), ""]
    labs = labels or []
    for i, o in enumerate(options, start=1):
        label = labs[i-1] if i-1 < len(labs) else f"{_safe_letter(i)})"
        parts.append(f"{label} {o}")
    raw = "\n".join(parts).strip()
    return f"<pre>{h(raw)}</pre>"

@require_admin
async def cmd_postemoji(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("postemoji", "<DB-ID> [keep]", "Post buffered questions as emoji quiz to a channel"))
        return
    cid = int(context.args[0])
    keep = (len(context.args) > 1 and context.args[1].strip().lower() == "keep")
    ch = channel_get_by_id_for_user(admin_id, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or no access.")
        return
    items = buffer_list(admin_id, limit=MAX_BUFFERED_QUESTIONS)
    if not items:
        await warn(update, "Buffer Empty", "No buffered questions found.")
        return
    sent = 0; sent_ids = []
    try:
        prefix = str(ch["prefix"] or "").strip()
    except Exception:
        prefix = ""
    for bid, payload in items:
        q, opts, corr_idx0, explanation = quiz_to_poll_parts(payload)
        labels = EMOJI_BUTTONS[:len(opts)]
        block = _copyable_quiz_block(q, opts, labels=labels)
        title = prefix if prefix else "Emoji Quiz"
        msg_html = f"<b>{h(title)}</b>\n\n{block}"
        quiz_id = uuid.uuid4().hex[:10]
        try:
            m = await context.bot.send_message(chat_id=ch.channel_chat_id, text=msg_html, parse_mode=ParseMode.HTML, reply_markup=emoji_quiz_keyboard(len(opts), quiz_id), disable_web_page_preview=True)
            sent += 1; sent_ids.append(bid)
            emoji_quiz_save(quiz_id, ch.channel_chat_id, m.message_id, {"question": q, "options": opts, "correct_answer": corr_idx0 + 1 if corr_idx0 >= 0 else 0, "explanation": explanation, "prefix": title}, admin_id)
            await asyncio.sleep(0.25)
        except Exception as e:
            db_log("ERROR", "postemoji_failed", {"admin_id": admin_id, "channel": ch.channel_chat_id, "error": str(e)})
    if sent and not keep:
        buffer_remove_ids(admin_id, sent_ids)
    await ok_html(update, "Emoji Quiz Posted", f"Sent: <code>{h(sent)}</code>\nChannel: <code>{h(ch.title)}</code>")

async def on_emoji_quiz_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    q = update.callback_query
    data = (q.data or "").strip()
    m = re.match(r"^eq:([0-9a-f]{6,16}):(\d+)$", data)
    if not m:
        return
    quiz_id = m.group(1)
    selected = int(m.group(2))
    uid = q.from_user.id if q.from_user else 0
    if not uid:
        return
    ok, _missing = await user_meets_required_memberships(context, uid)
    if not ok:
        await q.answer("Join required channel first.", show_alert=True)
        return
    quiz = emoji_quiz_get(quiz_id)
    if not quiz:
        await q.answer("Quiz expired or not found.", show_alert=True)
        return
    if not emoji_quiz_has_answered(quiz_id, uid):
        correct = int(quiz.get("correct_answer", 0) or 0)
        emoji_quiz_record_answer(quiz_id, uid, selected, (selected == correct and correct > 0))
    selected = emoji_quiz_user_choice(quiz_id, uid) or selected
    correct = int(quiz.get("correct_answer", 0) or 0)
    counts = emoji_quiz_counts(quiz_id)
    opts = quiz.get("options", []) or []
    stats = []
    for i, opt in enumerate(opts, start=1):
        label = EMOJI_BUTTONS[i-1] if i-1 < len(EMOJI_BUTTONS) else _safe_letter(i)
        stats.append(f"{label} {opt} — {counts.get(i,0)}")
    stats_text = "\n".join(stats)
    expl = clean_latex(str(quiz.get("explanation", "") or "").strip())
    if selected == correct and correct > 0:
        msg = f"✅ Correct\n{expl}\n\nStats:\n{stats_text}".strip()
    else:
        corr_label = EMOJI_BUTTONS[correct-1] if 0 < correct <= len(EMOJI_BUTTONS) else _safe_letter(correct)
        msg = f"❌ Wrong\n✅ Correct: {corr_label}\n{expl}\n\nStats:\n{stats_text}".strip()
    await q.answer(msg[:190], show_alert=True)


# ===========================
# STRONG FIX OVERRIDES (GPT)
# ===========================
USE_OFFICIAL_GEMINI_REST_FALLBACK = False
USE_GEMINI_REST_FOR_GENQUIZ = False
REQUIRED_DEFAULT_JOIN_URL = "https://t.me/FX_Ur_Target"
REQUIRED_DEFAULT_CHAT_USERNAME = "@FX_Ur_Target"
REQUIRED_DEFAULT_CHAT_TITLE = "✨TARGET🎯"


def _effective_required_targets() -> List[Dict[str, Any]]:
    rows = required_chat_list()
    targets: List[Dict[str, Any]] = []
    has_default = False
    for r in rows:
        try:
            cid = int(r["chat_id"])
        except Exception:
            continue
        title = str(r["title"] or cid)
        tl = title.lower()
        if "fx_ur_target" in tl:
            has_default = True
        if title.startswith("@"):
            url = f"https://t.me/{title.lstrip('@')}"
        elif "t.me/" in title:
            url = title if title.startswith("http") else ("https://" + title.lstrip("/"))
        else:
            url = REQUIRED_DEFAULT_JOIN_URL
        targets.append({"chat_id": cid, "title": title, "url": url})
    if not has_default:
        targets.insert(0, {
            "chat_id": REQUIRED_DEFAULT_CHAT_USERNAME,
            "title": REQUIRED_DEFAULT_CHAT_TITLE,
            "url": REQUIRED_DEFAULT_JOIN_URL,
        })
    return targets


def _required_join_kb() -> InlineKeyboardMarkup:
    rows = []
    targets = _effective_required_targets()
    for i, t in enumerate(targets[:8]):
        title = str(t.get("title") or "Channel")
        label = "Join Channel" if i == 0 else f"Join {title}"
        rows.append([InlineKeyboardButton(label, url=str(t.get("url") or REQUIRED_DEFAULT_JOIN_URL))])
    rows.append([InlineKeyboardButton("✅ Verify", callback_data="req:verify")])
    return InlineKeyboardMarkup(rows)


def _warn_count_or_increment(user_id: int, *, throttle_seconds: int = 45) -> int:
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT warn_count, last_warn_at FROM user_warnings WHERE user_id=?", (int(user_id),))
    row = cur.fetchone(); conn.close()
    if row and row["last_warn_at"]:
        try:
            last = dt.datetime.fromisoformat(str(row["last_warn_at"]))
            now = dt.datetime.now(last.tzinfo or dt.timezone.utc)
            if abs((now - last).total_seconds()) <= throttle_seconds:
                return int(row["warn_count"] or 0)
        except Exception:
            pass
    return inc_warn_count(user_id)


async def _send_join_required_message(update: Update, context: ContextTypes.DEFAULT_TYPE, missing: List[str], count: int) -> None:
    names = ", ".join(missing[:10]) if missing else REQUIRED_DEFAULT_CHAT_TITLE
    msg = ui_box_text("Join Required", f"Please join: {names}\nWarning: {count}/5", emoji="⚠️")
    if update.message:
        old_mid = None
        try:
            old_mid = context.user_data.get("_req_prompt_mid")
        except Exception:
            old_mid = None
        if old_mid:
            with contextlib.suppress(Exception):
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=int(old_mid))
        try:
            sent = await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=_required_join_kb(), disable_web_page_preview=True)
            try:
                context.user_data["_req_prompt_mid"] = sent.message_id
            except Exception:
                pass
            return
        except Exception:
            pass
    if update.callback_query and update.callback_query.message:
        with contextlib.suppress(Exception):
            await update.callback_query.message.edit_text(msg, parse_mode=ParseMode.HTML, reply_markup=_required_join_kb(), disable_web_page_preview=True)
            return
    if update.effective_user:
        await safe_send_text(context.bot, update.effective_user.id, msg, reply_markup=_required_join_kb())


async def user_meets_required_memberships(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> Tuple[bool, List[str]]:
    targets = _effective_required_targets()
    if not targets:
        return True, []
    missing: List[str] = []
    for t in targets:
        cid = t.get("chat_id")
        title = str(t.get("title") or cid)
        try:
            member = await context.bot.get_chat_member(cid, int(user_id))
            status = str(getattr(member, "status", "")).lower()
            if status in ("left", "kicked"):
                missing.append(title)
        except Exception:
            missing.append(title)
    return (len(missing) == 0), missing


async def enforce_required_memberships(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    ensure_user(update)
    uid = update.effective_user.id if update.effective_user else 0
    if not uid or is_owner(uid) or is_admin(uid):
        return True
    ok, missing = await user_meets_required_memberships(context, uid)
    if ok:
        reset_warn_count(uid)
        return True
    count = _warn_count_or_increment(uid)
    if count >= 5:
        set_ban(uid, True)
        audit_ban(OWNER_ID, uid, "BAN")
        with contextlib.suppress(Exception):
            await safe_send_text(context.bot, uid, f"🚫 You are banned from <b>{h(BOT_BRAND)}</b> for leaving required channel/group. Contact: {h(OWNER_CONTACT)}")
        return False
    await _send_join_required_message(update, context, missing, count)
    return False


async def on_required_verify_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    q = update.callback_query
    uid = q.from_user.id if q.from_user else 0
    if not uid:
        with contextlib.suppress(Exception):
            await q.answer("User not found.", show_alert=True)
        return
    if is_owner(uid) or is_admin(uid):
        with contextlib.suppress(Exception):
            await q.answer("Verified.", show_alert=False)
        return
    ok, missing = await user_meets_required_memberships(context, uid)
    if ok:
        reset_warn_count(uid)
        with contextlib.suppress(Exception):
            await q.answer("Verification successful.", show_alert=True)
        with contextlib.suppress(Exception):
            if q.message:
                await q.message.delete()
        try:
            body_html = (
                f"<b>Your Role:</b> <code>{h(get_role(uid))}</code>"
                f"\n\nUse <code>/help</code> for commands or <code>/commands</code> for a quick list."
            )
            msg = ui_box_html(f"Welcome to {BOT_BRAND}", body_html, emoji="👋")
            await safe_send_text(context.bot, uid, msg)
        except Exception:
            pass
        return
    count = _warn_count_or_increment(uid)
    with contextlib.suppress(Exception):
        await q.answer("Still not joined. Please join first.", show_alert=True)
    if count >= 5:
        set_ban(uid, True)
        audit_ban(OWNER_ID, uid, "BAN")
        with contextlib.suppress(Exception):
            if q.message:
                await q.message.edit_text(f"🚫 You are banned from {BOT_BRAND}. Contact: {OWNER_CONTACT}")
        return
    await _send_join_required_message(update, context, missing, count)


def _all_commands_for(uid: int) -> List[Tuple[str, List[Tuple[str, str]]]]:
    role = get_role(uid)
    sections: List[Tuple[str, List[Tuple[str, str]]]] = []
    user_cmds = [
        ("/start", "Welcome / membership check"),
        ("/help", "Detailed guide"),
        ("/commands", "All commands list"),
        ("/ask", "Contact support"),
        ("/solve_on", "Enable AI solving"),
        ("/solve_off", "Disable AI solving"),
    ]
    if can_use_vision(uid):
        user_cmds += [
            ("/scanhelp", "Image-to-quiz guide"),
            ("/vision_on", "Enable image extraction"),
            ("/vision_off", "Disable image extraction"),
        ]
    sections.append(("👤 User Commands", user_cmds))
    if role in (ROLE_ADMIN, ROLE_OWNER):
        admin_cmds = [
            ("/filter", "Add parser filter text"),
            ("/done", "Export CSV + JSON, then clear buffer"),
            ("/clear", "Clear buffer"),
            ("/addchannel", "Add target channel"),
            ("/listchannels", "List available channels"),
            ("/removechannel", "Remove a channel"),
            ("/setprefix", "Set channel prefix"),
            ("/setexplink", "Set explanation link"),
            ("/post", "Post normal quizzes"),
            ("/postemoji", "Post emoji quizzes"),
            ("/broadcast", "Broadcast message to users"),
            ("/adminpanel", "Posting stats"),
            ("/reply", "Reply to support ticket"),
            ("/close", "Close support ticket"),
            ("/ban", "Ban a user"),
            ("/unban", "Unban a user"),
            ("/banned", "View bans"),
            ("/private_send", "Send private message to a user"),
            ("/send_private", "Alias of /private_send"),
            ("/himusai_on", "Enable admin inbox AI-only mode"),
            ("/himusai_off", "Disable admin inbox AI-only mode"),
            ("/probaho_on", "Enable group user AI"),
            ("/probaho_off", "Disable group user AI"),
            ("/explain_on", "Enable explanation in quiz + export"),
            ("/explain_off", "Disable explanation in quiz + export"),
            ("/quizprefix", "Set generated quiz prefix"),
            ("/quizlink", "Set generated quiz link"),
        ]
        sections.append(("🛠 Staff Commands", admin_cmds))
    if role == ROLE_OWNER:
        owner_cmds = [
            ("/addadmin", "Promote a user to admin"),
            ("/removeadmin", "Remove admin role"),
            ("/grantall", "Grant admin all-channel access"),
            ("/revokeall", "Revoke all-channel access"),
            ("/grantvision", "Grant image extraction access"),
            ("/revokevision", "Revoke image extraction access"),
            ("/addrequired", "Add required channel/group"),
            ("/delrequired", "Remove required channel/group"),
            ("/listrequired", "List required channels/groups"),
            ("/ownerstats", "Owner dashboard"),
            ("/users", "Export started users JSON"),
        ]
        sections.append(("👑 Owner Commands", owner_cmds))
    return sections


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id
    if not await enforce_required_memberships(update, context):
        return
    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\n\nContact: {OWNER_CONTACT}")
        return
    sections = _all_commands_for(uid)
    blocks = [ui_box_html(f"{BOT_BRAND} — Command Guide", f"Owner: {h(OWNER_CONTACT)}\nUse only the commands available for your role.", emoji="📚")]
    for title, items in sections:
        body = "\n".join([f"<code>{h(cmd)}</code> — {h(desc)}" for cmd, desc in items])
        blocks.append(ui_box_html(title, body, emoji="•"))
    await safe_reply(update, "\n\n".join(blocks))


async def cmd_commands(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id
    if not await enforce_required_memberships(update, context):
        return
    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\n\nContact: {OWNER_CONTACT}")
        return
    sections = _all_commands_for(uid)
    parts = [ui_box_html("All Available Commands", "Choose a command below.", emoji="📋")]
    for title, items in sections:
        body = "\n".join([f"<code>{h(cmd)}</code> — {h(desc)}" for cmd, desc in items])
        parts.append(ui_box_html(title, body, emoji="👤" if "User" in title else ("🛠" if "Staff" in title else "👑")))
    await safe_reply(update, "\n\n".join(parts))


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id if update.effective_user else 0
    if not await enforce_required_memberships(update, context):
        return
    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\n\nContact: {OWNER_CONTACT}")
        return
    role = get_role(uid)
    body_html = (
        f"<b>Your Role:</b> <code>{h(role)}</code>"
        f"\n\nUse <code>/help</code> for commands or <code>/commands</code> for a quick list."
    )
    msg = ui_box_html(f"Welcome to {BOT_BRAND}", body_html, emoji="👋")
    await safe_reply(update, msg)


async def cmd_solve_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id
    if not await enforce_required_memberships(update, context):
        return
    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\n\nContact: {OWNER_CONTACT}")
        return
    if get_role(uid) != ROLE_USER:
        await warn(update, "Not Available", "Problem-solving chat is intended for normal users. Admin/Owner workflow should remain unchanged.")
        return
    set_solver_mode_on(uid, True)
    await ok_html(update, "Solver Enabled", "Now just send your question as text and the bot will reply with a solved explanation.\n\nTurn off anytime using <code>/solve_off</code>.", emoji="🧠")


async def cmd_solve_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id
    if not await enforce_required_memberships(update, context):
        return
    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\n\nContact: {OWNER_CONTACT}")
        return
    if get_role(uid) != ROLE_USER:
        await warn(update, "Not Available", "Problem-solving chat is intended for normal users.")
        return
    set_solver_mode_on(uid, False)
    await ok_html(update, "Solver Disabled", "The bot will no longer auto-solve your text messages.", emoji="🧠")


def _extract_ticket_id_from_message(msg) -> Optional[int]:
    if not msg:
        return None
    texts = []
    for attr in ("text", "caption"):
        val = getattr(msg, attr, None)
        if val:
            texts.append(str(val))
    if not texts:
        return None
    blob = "\n".join(texts)
    m = re.search(r"Ticket\s*:?\s*(\d+)", blob, re.I)
    if not m:
        m = re.search(r"Ticket\s*ID\s*:?\s*(\d+)", blob, re.I)
    return int(m.group(1)) if m else None


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id
    if not await enforce_required_memberships(update, context):
        return
    if is_banned(uid):
        await err(update, "Access Denied", f"You are banned.\nContact: {OWNER_CONTACT}")
        return
    replied = update.message.reply_to_message if update.message else None
    text = " ".join(context.args).strip()
    if not text:
        text = reply_text_or_caption(update)
    if not text and not replied:
        await safe_reply(update, usage_box("ask", "<message>", "Ask a support question (or reply to message/file/photo)"))
        return
    tid = ticket_find_open_by_student(uid)
    if tid is None:
        tid = ticket_open(uid, update.effective_user.first_name or "")
        db_log("INFO", "ticket_open", {"ticket_id": tid, "student_id": uid})
    if text:
        ticket_add_msg(tid, "STUDENT", uid, text)
    elif replied:
        ticket_add_msg(tid, "STUDENT", uid, "[MEDIA MESSAGE]")
    staff_ids = list_staff_ids()
    profile = mention_user(uid, update.effective_user.first_name or str(uid))
    uname = f"@{update.effective_user.username}" if getattr(update.effective_user, 'username', None) else ""
    header = f"📩 New Support Message\nTicket: {tid}\nFrom: {profile} | <code>{uid}</code> {h(uname)}"
    if text:
        for sid in staff_ids:
            body = f"{header}\n\n{h(text)}"
            await safe_send_text(context.bot, sid, body)
    else:
        for sid in staff_ids:
            await safe_send_text(context.bot, sid, f"{header}\n\n[MEDIA MESSAGE RECEIVED]")
    if replied:
        for sid in staff_ids:
            await safe_copy_message(context.bot, chat_id=sid, from_chat_id=replied.chat_id, message_id=replied.message_id, protect=False)
    await ok(update, "Message Received", "A staff member will respond soon.")


@require_admin
async def cmd_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    replied = update.message.reply_to_message if update.message else None
    tid = None
    if context.args and str(context.args[0]).isdigit():
        tid = int(context.args[0])
        text = " ".join(context.args[1:]).strip()
    else:
        tid = _extract_ticket_id_from_message(replied)
        text = " ".join(context.args).strip()
    if tid is None:
        await safe_reply(update, usage_box("reply", "<ticket_id> [message]", "Reply to support ticket, or reply to the support card and use /reply [message]"))
        return
    if not text:
        text = reply_text_or_caption(update)
    tr = ticket_get(int(tid))
    if not tr:
        await warn_html(update, "Ticket Not Found", f"No ticket with ID <code>{h(tid)}</code> found")
        return
    if tr["status"] != "OPEN":
        await err_html(update, "Ticket Closed", f"Ticket <code>{h(tid)}</code> is already <b>CLOSED</b>")
        return
    student_id = int(tr["student_id"])
    if is_banned(student_id):
        await warn(update, "User Banned", "The user is currently banned.")
        return
    sent_any = False
    if text:
        ticket_add_msg(int(tid), "STAFF", update.effective_user.id, text)
        if looks_like_programming_request(text):
            await safe_send_text(context.bot, student_id, f"💬 Support Reply\n\n<pre>{h(text)}</pre>")
        else:
            await safe_send_text(context.bot, student_id, f"💬 Support Reply\n\n{h(text)}")
        sent_any = True
    if replied and getattr(replied, 'message_id', None):
        okc = await safe_copy_message(context.bot, chat_id=student_id, from_chat_id=replied.chat_id, message_id=replied.message_id, protect=False)
        if okc:
            ticket_add_msg(int(tid), "STAFF", update.effective_user.id, "[MEDIA MESSAGE]")
            sent_any = True
    if sent_any:
        await ok_html(update, "Reply Sent", f"<b>Ticket:</b> <code>{h(tid)}</code>\nMessage(s) sent to user.")
    else:
        await warn(update, "No Content", "Reply to a message/file/photo or provide text inline")


@require_owner
async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT user_id, role, first_name, username, is_banned, created_at, last_seen_at FROM users ORDER BY created_at ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    if not rows:
        await warn(update, "No Users", "No users found.")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        path = f.name
    try:
        with open(path, "rb") as rf:
            await context.bot.send_document(chat_id=update.effective_user.id, document=rf, filename="probaho_users.json", caption="All started users")
    finally:
        with contextlib.suppress(Exception):
            os.unlink(path)


def gemini_solve_text(problem_text: str) -> str:
    prompt = (STRICT_SYSTEM_PROMPT + "\n\nUser Message:\n" + (problem_text or "").strip()).strip()
    last_err = None
    try:
        out = gemini3_solve(prompt)
        if out and str(out).strip():
            return str(out).strip()
    except Exception as e:
        last_err = e
    if USE_PERPLEXITY_FALLBACK:
        try:
            alt = query_ai(prompt)
            if alt:
                return alt.strip()
        except Exception as e:
            last_err = e
    if USE_OFFICIAL_GEMINI_REST_FALLBACK and GEMINI_API_KEY:
        try:
            return call_gemini_text_rest(prompt, timeout_seconds=18).strip()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Solver failed: {str(last_err)[:120] if last_err else 'all backends unavailable'}")


def _infer_option_from_text(text: str, n: int) -> int:
    s = (text or "").upper()
    patterns = [
        r"FINAL ANSWER\s*[:\-]\s*([A-E])",
        r"CORRECT ANSWER\s*[:\-]\s*([A-E])",
        r"ANSWER\s*[:\-]\s*([A-E])",
        r"OPTION\s*([A-E])",
        r"\(([A-E])\)",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            idx = ord(m.group(1)) - 64
            if 1 <= idx <= n:
                return idx
    return 0


def gemini_solve_mcq_json(question: str, options: List[str]) -> Dict[str, Any]:
    q = (question or "").strip()
    opts = [(o or "").strip() for o in (options or []) if (o or "").strip()][:5]
    if len(opts) < 2:
        raise ValueError("Not enough options to solve.")
    opt_lines = "\n".join([f"{_safe_letter(i+1)}. {opts[i]}" for i in range(len(opts))])
    prompt = (
        "Return STRICT JSON only. No markdown. No extra text.\n\n"
        "Task: Solve the following MCQ and pick the correct option.\n"
        "Rules:\n"
        "- answer must be 1-5 (A=1,B=2,C=3,D=4,E=5). If unsure, pick the best option.\n"
        "- explanation: clear exam-style explanation.\n"
        "- why_not: short reason for wrong options.\n"
        "- confidence: 0-100 integer.\n\n"
        f"Question:\n{q}\n\nOptions:\n{opt_lines}\n\n"
        "JSON format:\n"
        "{\"answer\":1,\"confidence\":0,\"explanation\":\"...\",\"why_not\":{\"A\":\"..\",\"B\":\"..\",\"C\":\"..\",\"D\":\"..\",\"E\":\"..\"}}"
    )
    last_err = None
    try:
        raw = gemini3_solve(prompt)
        data = _extract_json_strict(raw)
        if isinstance(data, dict) and int(data.get("answer", 0) or 0) > 0:
            return data
    except Exception as e:
        last_err = e
    if USE_PERPLEXITY_FALLBACK:
        try:
            alt = query_ai(prompt)
            if alt:
                try:
                    data = _extract_json_strict(alt)
                    if isinstance(data, dict) and int(data.get("answer", 0) or 0) > 0:
                        return data
                except Exception:
                    pass
                inferred = _infer_option_from_text(alt, len(opts))
                return {
                    "answer": inferred,
                    "confidence": 0,
                    "explanation": (alt[:1800] if isinstance(alt, str) else str(alt)[:1800]),
                    "why_not": {},
                }
        except Exception as e:
            last_err = e
    if USE_OFFICIAL_GEMINI_REST_FALLBACK and GEMINI_API_KEY:
        try:
            raw2 = call_gemini_text_rest(prompt, timeout_seconds=18, force_json=True)
            data2 = _extract_json_strict(raw2)
            if isinstance(data2, dict):
                return data2
        except Exception as e:
            last_err = e
    raise RuntimeError(f"MCQ solver failed: {str(last_err)[:120] if last_err else 'all backends unavailable'}")


async def on_solver_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    q = update.callback_query
    await q.answer("Processing…", show_alert=False)
    data = (q.data or "").strip()
    m = re.match(r"^solve:([GPD]):([0-9a-f]{6,16})$", data)
    if not m:
        return
    model = m.group(1)
    token = m.group(2)
    store = _pending_store(context)
    req = store.get(token)
    if not isinstance(req, dict):
        with contextlib.suppress(Exception):
            await q.edit_message_text("⚠️ This request has expired. Please send your question again.")
        return
    uid = int(req.get("uid") or 0)
    if q.from_user and q.from_user.id != uid:
        with contextlib.suppress(Exception):
            await q.answer("This is not your request.", show_alert=True)
        return
    payload = req.get("payload") or {}
    problem_text = str(payload.get("text") or "").strip()
    kind = str(req.get("kind") or "text").lower()
    with contextlib.suppress(Exception):
        await q.edit_message_text(ui_box_text("Solving", "Please wait… Processing your request.", emoji="⏳"), parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    try:
        if kind == "poll" and payload.get("question"):
            question = str(payload.get("question", "")).strip()
            options = payload.get("options", [])
            if model == "G":
                result = await _run_blocking(_role_of(uid), gemini_solve_mcq_json, question, options)
                model_name = "Gemini"
            elif model == "P":
                result = await _run_blocking(_role_of(uid), perplexity_solve_mcq_json, question, options)
                model_name = "Perplexity"
            else:
                result = await _run_blocking(_role_of(uid), deepseek_solve_mcq_json, question, options)
                model_name = "DeepSeek"
            raw_expl = str(result.get('explanation', '') or "")
            clean_expl = clean_latex(raw_expl)
            raw_why_not = result.get("why_not", {}) or {}
            clean_why_not = {k: clean_latex(v) for k, v in raw_why_not.items()}
            msg_html = _format_user_poll_solution(
                question=question,
                options=options,
                model_ans=int(result.get("answer", 0) or 0),
                official_ans=int(payload.get("official_ans", 0) or 0),
                model_expl=f"[{model_name}]\n{clean_expl}".strip(),
                official_expl=str(payload.get("official_expl", "")).strip(),
                why_not=clean_why_not,
                conf=int(result.get("confidence", 0) or 0),
            )
            kb = _verify_kb(token, model, "poll")
        else:
            if model == "G":
                answer = await _run_blocking(_role_of(uid), gemini_solve_text, problem_text)
            elif model == "P":
                answer = await _run_blocking(_role_of(uid), perplexity_solve_text, problem_text)
            else:
                answer = await _run_blocking(_role_of(uid), deepseek_solve_text, problem_text)
            if (is_admin(uid) or is_owner(uid)) and (looks_like_programming_request(problem_text) or looks_like_programming_request(answer)):
                msg_html = f"<pre>{h(answer)}</pre>"
            else:
                msg_html = h(answer)
            kb = _verify_kb(token, model, "text")
        with contextlib.suppress(Exception):
            await q.edit_message_text(msg_html, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            if q.message and kind == "poll":
                _remember_quiz_context(context, q.message.message_id, payload)
    except Exception as e:
        db_log("ERROR", "solver_callback_failed", {"user_id": uid, "model": model, "error": str(e)})
        with contextlib.suppress(Exception):
            await q.edit_message_text(ui_box_text("Solve Failed", str(e)[:180], emoji="❌"), parse_mode=ParseMode.HTML)


async def handle_user_text_unusual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id
    if is_banned(uid):
        return
    if not await enforce_required_memberships(update, context):
        return
    role = get_role(uid)
    private = is_private_chat(update)
    if role == ROLE_USER:
        if not solver_mode_on(uid):
            if private:
                await warn_unauthorized(update, "This bot is currently restricted for staff operations. Please use /ask [message] for support.")
            return
        if not private and not is_group_ai_enabled(update.effective_chat.id):
            return
    elif role in (ROLE_ADMIN, ROLE_OWNER):
        if not private or not solver_mode_on(uid):
            return
    else:
        return
    user_text = (update.message.text or "").strip()
    if not user_text:
        return
    reply_msg = update.message.reply_to_message
    if reply_msg:
        ctx = _get_quiz_context(context, reply_msg.message_id)
        if not ctx and getattr(reply_msg, 'poll', None):
            poll = reply_msg.poll
            ctx = {
                "question": str(poll.question or "").strip(),
                "options": [str(o.text).strip() for o in (poll.options or []) if str(o.text or '').strip()],
                "official_ans": _poll_official_answer(poll),
                "official_expl": str(getattr(poll, 'explanation', '') or '').strip(),
            }
        if ctx:
            qtext = str(ctx.get("question", "") or "").strip()
            opts = ctx.get("options", []) or []
            prompt = f"Question:\n{qtext}\n\nOptions:\n" + "\n".join([f"{_safe_letter(i+1)}. {o}" for i, o in enumerate(opts)]) + f"\n\nUser follow-up:\n{user_text}"
            await send_solver_picker(update, context, prompt)
            return
    await send_solver_picker(update, context, user_text)


def _copyable_quiz_block(question: str, options: List[str], labels: Optional[List[str]] = None) -> str:
    parts = [question.strip(), ""]
    labs = labels or []
    for i, o in enumerate(options, start=1):
        label = labs[i-1] if i-1 < len(labs) else f"{_safe_letter(i)})"
        parts.append(f"{label} {o}")
    raw = "\n".join(parts).strip()
    return f"<pre>{h(raw)}</pre>"


@require_admin
async def cmd_postemoji(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("postemoji", "<DB-ID> [keep]", "Post buffered questions as emoji quiz to a channel"))
        return
    cid = int(context.args[0])
    keep = (len(context.args) > 1 and context.args[1].strip().lower() == "keep")
    ch = channel_get_by_id_for_user(admin_id, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or no access.")
        return
    items = buffer_list(admin_id, limit=MAX_BUFFERED_QUESTIONS)
    if not items:
        await warn(update, "Buffer Empty", "No buffered questions found.")
        return
    sent = 0
    sent_ids = []
    prefix = str(getattr(ch, "prefix", "") or "").strip()
    title = prefix if prefix else BOT_BRAND
    for bid, payload in items:
        qtext, opts, corr_idx0, explanation = quiz_to_poll_parts(payload)
        labels = EMOJI_BUTTONS[:len(opts)]
        block = _copyable_quiz_block(qtext, opts, labels=labels)
        msg_html = f"<b>{h(title)}</b>\n\n{block}"
        quiz_id = uuid.uuid4().hex[:10]
        try:
            m = await context.bot.send_message(
                chat_id=ch.channel_chat_id,
                text=msg_html,
                parse_mode=ParseMode.HTML,
                reply_markup=emoji_quiz_keyboard(len(opts), quiz_id),
                disable_web_page_preview=True,
            )
            sent += 1
            sent_ids.append(bid)
            emoji_quiz_save(
                quiz_id,
                ch.channel_chat_id,
                m.message_id,
                {
                    "question": qtext,
                    "options": opts,
                    "correct_answer": corr_idx0 + 1 if corr_idx0 >= 0 else 0,
                    "explanation": explanation,
                    "prefix": title,
                },
                admin_id,
            )
            await asyncio.sleep(0.25)
        except Exception as e:
            db_log("ERROR", "postemoji_failed", {"admin_id": admin_id, "channel": ch.channel_chat_id, "error": str(e)})
    if sent and not keep:
        buffer_remove_ids(admin_id, sent_ids)
    await ok_html(update, "Emoji Quiz Posted", f"Sent: <code>{h(sent)}</code>\nChannel: <code>{h(ch.title)}</code>")


async def on_emoji_quiz_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    q = update.callback_query
    data = (q.data or "").strip()
    m = re.match(r"^eq:([0-9a-f]{6,16}):(\d+)$", data)
    if not m:
        return
    quiz_id = m.group(1)
    selected = int(m.group(2))
    uid = q.from_user.id if q.from_user else 0
    if not uid:
        return
    ok, _missing = await user_meets_required_memberships(context, uid)
    if not ok:
        await q.answer("⚠️ আগে required channel-এ join করো, তারপর Verify দাও।", show_alert=True)
        return
    quiz = emoji_quiz_get(quiz_id)
    if not quiz:
        await q.answer("Quiz expired or not found.", show_alert=True)
        return
    saved_choice = emoji_quiz_user_choice(quiz_id, uid)
    if saved_choice and int(saved_choice) != int(selected):
        saved_label = EMOJI_BUTTONS[saved_choice - 1] if 0 < saved_choice <= len(EMOJI_BUTTONS) else str(saved_choice)
        await q.answer(f"⚠️ তুমি আগে {saved_label} দিয়ে answer দিয়েছো। Result দেখতে একই reaction-এ tap করো।", show_alert=True)
        return
    correct = int(quiz.get("correct_answer", 0) or 0)
    if not saved_choice:
        emoji_quiz_record_answer(quiz_id, uid, selected, (selected == correct and correct > 0))
        saved_choice = selected
    counts = emoji_quiz_counts(quiz_id)
    opts = quiz.get("options", []) or []
    stats_text = " | ".join([f"{EMOJI_BUTTONS[i-1]}={counts.get(i, 0)}" for i in range(1, len(opts) + 1)])
    expl = clean_latex(str(quiz.get("explanation", "") or "").strip())
    expl = re.sub(r"\s+", " ", expl).strip()
    if len(expl) > 90:
        expl = expl[:87] + "..."
    sel_label = EMOJI_BUTTONS[saved_choice - 1] if 0 < saved_choice <= len(EMOJI_BUTTONS) else str(saved_choice)
    corr_label = EMOJI_BUTTONS[correct - 1] if 0 < correct <= len(EMOJI_BUTTONS) else str(correct)
    if saved_choice == correct and correct > 0:
        msg = f"✅ Correct\nYour reaction: {sel_label}\n{stats_text}"
    else:
        msg = f"❌ Wrong\nYour reaction: {sel_label}\n✅ Correct: {corr_label}\n{stats_text}"
    if expl:
        msg += f"\n\n{expl}"
    await q.answer(msg[:190], show_alert=True)


_original_handle_image = handle_image


# ===========================
# FINAL STABLE OVERRIDES
# ===========================
USE_OFFICIAL_GEMINI_REST_FALLBACK = False
USE_GEMINI_REST_FOR_GENQUIZ = True
REQUIRED_DEFAULT_JOIN_URL = "https://t.me/FX_Ur_Target"
REQUIRED_DEFAULT_CHAT_USERNAME = "@FX_Ur_Target"
REQUIRED_DEFAULT_CHAT_TITLE = "✨TARGET🎯"


def _effective_required_targets() -> List[Dict[str, Any]]:
    rows = required_chat_list()
    targets: List[Dict[str, Any]] = []
    seen = set()
    for r in rows:
        try:
            cid = int(r["chat_id"])
        except Exception:
            cid = r["chat_id"]
        title = str(r["title"] or cid)
        if title.startswith("@"):
            url = f"https://t.me/{title.lstrip('@')}"
        elif "t.me/" in title:
            url = title if title.startswith("http") else ("https://" + title.lstrip("/"))
        else:
            url = REQUIRED_DEFAULT_JOIN_URL
        targets.append({"chat_id": cid, "title": title, "url": url})
        seen.add(str(cid))
        seen.add(title.lower())
    if REQUIRED_DEFAULT_CHAT_USERNAME.lower() not in seen:
        targets.insert(0, {
            "chat_id": REQUIRED_DEFAULT_CHAT_USERNAME,
            "title": REQUIRED_DEFAULT_CHAT_TITLE,
            "url": REQUIRED_DEFAULT_JOIN_URL,
        })
    return targets


def _required_join_kb() -> InlineKeyboardMarkup:
    rows = []
    targets = _effective_required_targets()
    primary = targets[0] if targets else {"url": REQUIRED_DEFAULT_JOIN_URL, "title": REQUIRED_DEFAULT_CHAT_TITLE}
    rows.append([InlineKeyboardButton("📢 Join Channel", url=str(primary.get("url") or REQUIRED_DEFAULT_JOIN_URL))])
    if len(targets) > 1:
        for t in targets[1:8]:
            rows.append([InlineKeyboardButton(f"Join {str(t.get('title') or 'Chat')}", url=str(t.get("url") or REQUIRED_DEFAULT_JOIN_URL))])
    rows.append([InlineKeyboardButton("✅ I Joined", callback_data="req:verify")])
    return InlineKeyboardMarkup(rows)


async def user_meets_required_memberships(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> Tuple[bool, List[str]]:
    targets = _effective_required_targets()
    if not targets:
        return True, []
    missing: List[str] = []
    for t in targets:
        cid = t.get("chat_id")
        title = str(t.get("title") or cid)
        try:
            member = await context.bot.get_chat_member(cid, int(user_id))
            status = str(getattr(member, "status", "")).lower()
            if status in ("left", "kicked"):
                missing.append(title)
        except Exception:
            missing.append(title)
    return (len(missing) == 0), missing


def _warn_count_or_increment(user_id: int, *, throttle_seconds: int = 45) -> int:
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT warn_count, last_warn_at FROM user_warnings WHERE user_id=?", (int(user_id),))
    row = cur.fetchone(); conn.close()
    if row and row["last_warn_at"]:
        try:
            last = dt.datetime.fromisoformat(str(row["last_warn_at"]))
            now = dt.datetime.now(last.tzinfo or dt.timezone.utc)
            if abs((now - last).total_seconds()) <= throttle_seconds:
                return int(row["warn_count"] or 0)
        except Exception:
            pass
    return inc_warn_count(user_id)


async def _send_join_required_message(update: Update, context: ContextTypes.DEFAULT_TYPE, missing: List[str]) -> None:
    names = ", ".join(missing[:3]) if missing else REQUIRED_DEFAULT_CHAT_TITLE
    body_html = (
        f"You must join <b>{h(names)}</b> before using this bot."
        f"\n\nTap <b>Join Channel</b>, then press <b>I Joined</b>."
    )
    msg = ui_box_html("Join Required", body_html, emoji="⚠️")
    if update.message:
        old_mid = None
        try:
            old_mid = context.user_data.get("_req_prompt_mid")
        except Exception:
            old_mid = None
        if old_mid:
            with contextlib.suppress(Exception):
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=int(old_mid))
        try:
            sent = await update.message.reply_text(
                msg,
                parse_mode=ParseMode.HTML,
                reply_markup=_required_join_kb(),
                disable_web_page_preview=True,
            )
            try:
                context.user_data["_req_prompt_mid"] = sent.message_id
            except Exception:
                pass
            return
        except Exception:
            pass
    if update.callback_query and update.callback_query.message:
        with contextlib.suppress(Exception):
            await update.callback_query.message.edit_text(
                msg,
                parse_mode=ParseMode.HTML,
                reply_markup=_required_join_kb(),
                disable_web_page_preview=True,
            )
            return
    if update.effective_user:
        await safe_send_text(context.bot, update.effective_user.id, msg, reply_markup=_required_join_kb())


async def enforce_required_memberships(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    ensure_user(update)
    uid = update.effective_user.id if update.effective_user else 0
    if not uid or is_owner(uid) or is_admin(uid):
        return True
    ok, missing = await user_meets_required_memberships(context, uid)
    if ok:
        reset_warn_count(uid)
        return True
    count = _warn_count_or_increment(uid)
    if count >= 5:
        set_ban(uid, True)
        audit_ban(OWNER_ID, uid, "BAN")
        with contextlib.suppress(Exception):
            await safe_send_text(context.bot, uid, f"🚫 You are banned from <b>{h(BOT_BRAND)}</b>. Contact: {h(OWNER_CONTACT)}")
        return False
    await _send_join_required_message(update, context, missing)
    return False


async def on_required_verify_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    q = update.callback_query
    uid = q.from_user.id if q.from_user else 0
    if not uid:
        with contextlib.suppress(Exception):
            await q.answer("User not found.", show_alert=True)
        return
    if is_owner(uid) or is_admin(uid):
        with contextlib.suppress(Exception):
            await q.answer("Verified.", show_alert=False)
        return
    ok, missing = await user_meets_required_memberships(context, uid)
    if ok:
        reset_warn_count(uid)
        with contextlib.suppress(Exception):
            await q.answer("Verification successful.", show_alert=True)
        with contextlib.suppress(Exception):
            if q.message:
                await q.message.delete()
        role = get_role(uid)
        body_html = (
            f"<b>Your Role:</b> <code>{h(role)}</code>"
            f"\n\nUse <code>/help</code> for commands or <code>/commands</code> for a quick list."
        )
        msg = ui_box_html(f"Welcome to {BOT_BRAND}", body_html, emoji="👋")
        await safe_send_text(context.bot, uid, msg)
        return
    count = _warn_count_or_increment(uid)
    with contextlib.suppress(Exception):
        await q.answer("Join the required channel first.", show_alert=True)
    if count >= 5:
        set_ban(uid, True)
        audit_ban(OWNER_ID, uid, "BAN")
        with contextlib.suppress(Exception):
            if q.message:
                await q.message.edit_text(f"🚫 You are banned from {BOT_BRAND}. Contact: {OWNER_CONTACT}")
        return
    await _send_join_required_message(update, context, missing)


@require_admin
async def cmd_setprefix(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args or not str(context.args[0]).isdigit():
        await safe_reply(update, usage_box("setprefix", "<DB-ID> [text]", "Set or clear the prefix for a channel"))
        return
    cid = int(context.args[0])
    new_prefix = " ".join(context.args[1:]).strip() if len(context.args) > 1 else ""
    ch = channel_get_by_id_for_user(uid, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or you don't have access.")
        return
    old_prefix = getattr(ch, "prefix", "") or "(empty)"
    ok2 = channel_set_prefix(cid, new_prefix)
    if ok2:
        shown = new_prefix if new_prefix else "(empty)"
        body = (
            f"Channel: {h(getattr(ch, 'title', cid))}\n"
            f"DB-ID: {h(cid)}\n"
            f"Old Prefix: {h(old_prefix)}\n"
            f"New Prefix: {h(shown)}"
        )
        await ok(update, "Prefix Updated", body)
    else:
        await err(update, "Update Failed", "Could not update the prefix.")


@require_admin
async def cmd_setexplink(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args or not str(context.args[0]).isdigit():
        await safe_reply(update, usage_box("setexplink", "<DB-ID> [link]", "Set or clear the explanation link for a channel"))
        return
    cid = int(context.args[0])
    new_link = " ".join(context.args[1:]).strip() if len(context.args) > 1 else ""
    ch = channel_get_by_id_for_user(uid, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or you don't have access.")
        return
    old_link = getattr(ch, "expl_link", "") or "(empty)"
    ok2 = channel_set_expl_link(cid, new_link)
    if ok2:
        shown = new_link if new_link else "(empty)"
        body = (
            f"Channel: {h(getattr(ch, 'title', cid))}\n"
            f"DB-ID: {h(cid)}\n"
            f"Old Link: {h(old_link)}\n"
            f"New Link: {h(shown)}"
        )
        await ok(update, "Link Updated", body)
    else:
        await err(update, "Update Failed", "Could not update the link.")


@require_owner
async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT user_id, role, first_name, username, is_banned, created_at, last_seen_at FROM users ORDER BY created_at ASC")
    rows = cur.fetchall(); conn.close()
    if not rows:
        await warn(update, "No Users", "No users found.")
        return
    data = []
    for i, r in enumerate(rows, start=1):
        item = dict(r)
        item["serial"] = i
        data.append(item)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        path = f.name
    try:
        with open(path, "rb") as rf:
            await context.bot.send_document(
                chat_id=update.effective_user.id,
                document=rf,
                filename="probaho_users.json",
                caption=f"All started users • Total: {len(data)}",
            )
    finally:
        with contextlib.suppress(Exception):
            os.unlink(path)


@require_admin
async def cmd_usersd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not str(context.args[0]).lstrip("-").isdigit():
        await safe_reply(update, usage_box("usersd", "<user_id>", "Show a clickable profile link for a user ID"))
        return
    target = int(context.args[0])
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT first_name, username, role, is_banned, created_at, last_seen_at FROM users WHERE user_id=?", (target,))
    row = cur.fetchone(); conn.close()
    if row:
        name = row["first_name"] or str(target)
        uname = ("@" + row["username"]) if row["username"] else "(none)"
        body = (
            f"Profile: {mention_user(target, name)}\n"
            f"User ID: <code>{h(target)}</code>\n"
            f"Username: {h(uname)}\n"
            f"Role: <code>{h(row['role'] or 'USER')}</code>\n"
            f"Banned: <code>{'Yes' if int(row['is_banned'] or 0) else 'No'}</code>\n"
            f"Created: <code>{h(row['created_at'] or '')}</code>\n"
            f"Last Seen: <code>{h(row['last_seen_at'] or '')}</code>"
        )
    else:
        body = f"Profile: {mention_user(target, str(target))}\nUser ID: <code>{h(target)}</code>"
    await ok_html(update, "User Profile Link", body, emoji="🔎")


def _all_commands_for(uid: int) -> List[Tuple[str, List[Tuple[str, str]]]]:
    role = get_role(uid)
    sections: List[Tuple[str, List[Tuple[str, str]]]] = []
    user_cmds = [
        ("/start", "Start / membership check"),
        ("/help", "Detailed command guide"),
        ("/commands", "All commands list"),
        ("/ask", "Contact support"),
        ("/solve_on", "Enable private AI solving"),
        ("/solve_off", "Disable private AI solving"),
    ]
    if can_use_vision(uid):
        user_cmds += [
            ("/scanhelp", "Image-to-quiz guide"),
            ("/vision_on", "Enable image extraction"),
            ("/vision_off", "Disable image extraction"),
        ]
    sections.append(("👤 User Commands", user_cmds))
    if role in (ROLE_ADMIN, ROLE_OWNER):
        staff_cmds = [
            ("/filter", "Add parser filter text"),
            ("/done", "Export CSV + JSON, then clear buffer"),
            ("/clear", "Clear buffer"),
            ("/addchannel", "Add target channel"),
            ("/listchannels", "List available channels"),
            ("/removechannel", "Remove a channel"),
            ("/setprefix", "Set or clear channel prefix"),
            ("/setexplink", "Set or clear explanation link"),
            ("/post", "Post normal quizzes"),
            ("/postemoji", "Post emoji quizzes"),
            ("/reply", "Reply to support ticket"),
            ("/close", "Close support ticket"),
            ("/ban", "Ban a user"),
            ("/unban", "Unban a user"),
            ("/banned", "View banned users"),
            ("/broadcast", "Broadcast message to users"),
            ("/private_send", "Send private message"),
            ("/send_private", "Alias of /private_send"),
            ("/adminpanel", "Posting stats"),
            ("/himusai_on", "Enable inbox AI-only mode"),
            ("/himusai_off", "Disable inbox AI-only mode"),
            ("/probaho_on", "Enable user AI in this group"),
            ("/probaho_off", "Disable user AI in this group"),
            ("/explain_on", "Enable explanation in quiz + export"),
            ("/explain_off", "Disable explanation in quiz + export"),
            ("/quizprefix", "Set generated quiz prefix"),
            ("/quizlink", "Set generated quiz link"),
            ("/usersd", "Open a user profile by ID"),
        ]
        sections.append(("🛠 Staff Commands", staff_cmds))
    if role == ROLE_OWNER:
        owner_cmds = [
            ("/addadmin", "Add admin"),
            ("/removeadmin", "Remove admin"),
            ("/grantall", "Grant all-channels access"),
            ("/revokeall", "Revoke all-channels access"),
            ("/grantvision", "Grant image extraction"),
            ("/revokevision", "Revoke image extraction"),
            ("/addrequired", "Add required channel/group"),
            ("/delrequired", "Remove required channel/group"),
            ("/listrequired", "List required memberships"),
            ("/ownerstats", "Owner dashboard"),
            ("/users", "Export started users JSON"),
        ]
        sections.append(("👑 Owner Commands", owner_cmds))
    return sections


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id if update.effective_user else 0
    if not uid or is_banned(uid):
        return
    role = get_role(uid)
    if role not in (ROLE_ADMIN, ROLE_OWNER):
        return
    if is_private_chat(update) and solver_mode_on(uid):
        return
    text = update.message.text or ""
    if not text.strip():
        return
    if buffer_count(uid) >= MAX_BUFFERED_QUESTIONS:
        await warn(update, "Buffer Limit Reached", f"You have {MAX_BUFFERED_QUESTIONS} questions buffered.\n\nUse /done to export or /clear to reset.")
        return
    blocks = split_blocks(text)
    added = 0
    for b in blocks:
        if buffer_count(uid) >= MAX_BUFFERED_QUESTIONS:
            break
        try:
            payload = parse_text_block(b, uid)
            if payload:
                buffer_add(uid, payload)
                added += 1
        except Exception as e:
            db_log("ERROR", "parse_text_failed", {"admin_id": uid, "error": str(e)})
    if added:
        await ok_html(update, "Added to Buffer", f"<code>{h(added)}</code> question(s) added.\n\nTotal buffered: <code>{h(buffer_count(uid))}</code>", footer_html="Use <code>/done</code> to export")
    else:
        await warn(update, "No Questions Found", "No valid quiz blocks detected. Check formatting.")


async def handle_poll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id if update.effective_user else 0
    if not uid or is_banned(uid):
        return
    role = get_role(uid)
    if role not in (ROLE_ADMIN, ROLE_OWNER):
        return
    if is_private_chat(update) and solver_mode_on(uid):
        return
    poll = update.message.poll
    question = clean_common(poll.question or "", uid)
    options = [o.text for o in poll.options]
    opts = options + [""] * (5 - len(options))
    explanation = ""
    if hasattr(poll, "explanation") and poll.explanation:
        explanation = clean_explanation(poll.explanation, uid)
    correct_answer_id = 0
    if poll.type == "quiz" and poll.correct_option_id is not None:
        correct_answer_id = int(poll.correct_option_id) + 1
    payload = {
        "questions": question,
        "option1": (opts[0] or "").strip(),
        "option2": (opts[1] or "").strip(),
        "option3": (opts[2] or "").strip(),
        "option4": (opts[3] or "").strip(),
        "option5": (opts[4] or "").strip(),
        "answer": correct_answer_id,
        "explanation": explanation,
        "type": 1,
        "section": 1,
    }
    if buffer_count(uid) >= MAX_BUFFERED_QUESTIONS:
        await warn_html(update, "Buffer Limit Reached", f"You have <code>{h(MAX_BUFFERED_QUESTIONS)}</code> questions buffered.\n\nUse <code>/done</code> to export or <code>/clear</code> to reset.")
        return
    buffer_add(uid, payload)
    note = ""
    if correct_answer_id == 0 and poll.type == "quiz":
        note = "\n\n⚠️ Telegram may hide the correct answer in forwarded quizzes. CSV will store <code>answer=0</code>."
    body = f"Total buffered: <code>{buffer_count(uid)}</code>{note}"
    await ok_html(update, "Poll Saved", body)


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else 0
    if uid and get_role(uid) in (ROLE_ADMIN, ROLE_OWNER) and is_private_chat(update) and solver_mode_on(uid):
        return
    # fall through to existing image extraction logic
    return await globals()["_original_handle_image"](update, context)


def gemini_solve_text(problem_text: str) -> str:
    prompt = (
        STRICT_SYSTEM_PROMPT
        + "\n\nUser Message:\n"
        + (problem_text or "").strip()
    )
    try:
        out = gemini3_solve(prompt)
        if out and str(out).strip():
            return str(out).strip()
    except Exception:
        pass
    if USE_PERPLEXITY_FALLBACK:
        try:
            alt = query_ai(prompt)
            if alt:
                return alt.strip()
        except Exception:
            pass
    if USE_OFFICIAL_GEMINI_REST_FALLBACK and GEMINI_API_KEY:
        try:
            return call_gemini_text_rest(prompt, timeout_seconds=18).strip()
        except Exception:
            pass
    raise RuntimeError("AI backend is temporarily unavailable. Please try again.")


def _infer_option_from_text(text: str, n: int) -> int:
    s = (text or "").upper()
    patterns = [
        r"FINAL ANSWER\s*[:\-]\s*([A-E])",
        r"CORRECT ANSWER\s*[:\-]\s*([A-E])",
        r"ANSWER\s*[:\-]\s*([A-E])",
        r"OPTION\s*([A-E])",
        r"\(([A-E])\)",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            idx = ord(m.group(1)) - 64
            if 1 <= idx <= n:
                return idx
    return 0


def gemini_solve_mcq_json(question: str, options: List[str]) -> Dict[str, Any]:
    q = (question or "").strip()
    opts = [(o or "").strip() for o in (options or []) if (o or "").strip()][:5]
    if len(opts) < 2:
        raise ValueError("Not enough options to solve.")
    opt_lines = "\n".join([f"{_safe_letter(i+1)}. {opts[i]}" for i in range(len(opts))])
    prompt = (
        "Return STRICT JSON only. No markdown. No extra text.\n\n"
        "Task: Solve the following MCQ and pick the correct option.\n"
        "Rules:\n"
        "- answer must be 1-5 (A=1,B=2,C=3,D=4,E=5). If unsure, pick the best option.\n"
        "- explanation: clear exam-style explanation.\n"
        "- why_not: short reason for wrong options.\n"
        "- confidence: 0-100 integer.\n\n"
        f"Question:\n{q}\n\nOptions:\n{opt_lines}\n\n"
        "JSON format:\n"
        "{\"answer\":1,\"confidence\":0,\"explanation\":\"...\",\"why_not\":{\"A\":\"..\",\"B\":\"..\",\"C\":\"..\",\"D\":\"..\",\"E\":\"..\"}}"
    )
    try:
        raw = gemini3_solve(prompt)
        data = _extract_json_strict(raw)
        if isinstance(data, dict) and int(data.get("answer", 0) or 0) > 0:
            return data
    except Exception:
        pass
    if USE_PERPLEXITY_FALLBACK:
        try:
            alt = query_ai(prompt)
            if alt:
                try:
                    data = _extract_json_strict(alt)
                    if isinstance(data, dict) and int(data.get("answer", 0) or 0) > 0:
                        return data
                except Exception:
                    pass
                inferred = _infer_option_from_text(alt, len(opts))
                return {
                    "answer": inferred,
                    "confidence": 0,
                    "explanation": (alt[:1800] if isinstance(alt, str) else str(alt)[:1800]),
                    "why_not": {},
                }
        except Exception:
            pass
    if USE_OFFICIAL_GEMINI_REST_FALLBACK and GEMINI_API_KEY:
        try:
            raw2 = call_gemini_text_rest(prompt, timeout_seconds=18, force_json=True)
            data2 = _extract_json_strict(raw2)
            if isinstance(data2, dict):
                return data2
        except Exception:
            pass
    raise RuntimeError("AI backend is temporarily unavailable. Please try again.")


def _solve_text_with_preference(model: str, problem_text: str) -> Tuple[str, str]:
    model = (model or "G").upper()
    if model == "P":
        try:
            return perplexity_solve_text(problem_text), "Perplexity"
        except Exception:
            return gemini_solve_text(problem_text), "Gemini"
    if model == "D":
        try:
            return deepseek_solve_text(problem_text), "DeepSeek"
        except Exception:
            return gemini_solve_text(problem_text), "Gemini"
    return gemini_solve_text(problem_text), "Gemini"


def _solve_mcq_with_preference(model: str, question: str, options: List[str]) -> Tuple[Dict[str, Any], str]:
    model = (model or "G").upper()
    if model == "P":
        try:
            return perplexity_solve_mcq_json(question, options), "Perplexity"
        except Exception:
            return gemini_solve_mcq_json(question, options), "Gemini"
    if model == "D":
        try:
            return deepseek_solve_mcq_json(question, options), "DeepSeek"
        except Exception:
            return gemini_solve_mcq_json(question, options), "Gemini"
    return gemini_solve_mcq_json(question, options), "Gemini"


async def handle_user_poll_solver(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    if not update.effective_user or not update.message or not update.message.poll:
        return
    uid = update.effective_user.id
    if is_banned(uid):
        return
    role = get_role(uid)
    private = is_private_chat(update)
    if role == ROLE_USER:
        if private:
            if not solver_mode_on(uid):
                return
        else:
            if not is_group_ai_enabled(update.effective_chat.id):
                return
        if not await enforce_required_memberships(update, context):
            return
    elif role in (ROLE_ADMIN, ROLE_OWNER):
        if not private or not solver_mode_on(uid):
            return
    else:
        return

    poll = update.message.poll
    qtext = (poll.question or "").strip()
    options = [str(o.text).strip() for o in (poll.options or []) if str(o.text or '').strip()]
    official_expl = str(getattr(poll, "explanation", "") or "").strip()
    official_ans = _poll_official_answer(poll)

    spinner_msg = None
    spinner_task = None
    try:
        spinner_msg = await update.message.reply_text("🔎 Searching")
        spinner_task = asyncio.create_task(_spinner_task(context.bot, spinner_msg.chat_id, spinner_msg.message_id))
        data = await _run_blocking(_role_of(uid), gemini_solve_mcq_json, qtext, options)
        model_ans = int(data.get("answer", 0) or 0)
        conf = int(data.get("confidence", 0) or 0)
        raw_expl = str(data.get("explanation", "") or "").strip()
        model_expl = clean_latex(raw_expl)
        raw_why_not = data.get("why_not", {}) or {}
        why_not = {k: clean_latex(v) for k, v in raw_why_not.items()}
        if spinner_task:
            spinner_task.cancel()
        if spinner_msg:
            with contextlib.suppress(Exception):
                await context.bot.delete_message(chat_id=spinner_msg.chat_id, message_id=spinner_msg.message_id)
        msg_html = _format_user_poll_solution(
            question=qtext,
            options=options,
            model_ans=model_ans,
            official_ans=official_ans,
            model_expl=f"[Gemini 3 Flash]\n{model_expl}".strip(),
            official_expl=official_expl,
            why_not=why_not if isinstance(why_not, dict) else {},
            conf=conf,
        )
        poll_payload = {
            "question": qtext,
            "options": options,
            "official_ans": official_ans,
            "official_expl": official_expl,
        }
        await send_poll_verify_buttons(update, context, poll_payload, msg_html)
    except Exception as e:
        if spinner_task:
            spinner_task.cancel()
        if spinner_msg:
            with contextlib.suppress(Exception):
                await context.bot.delete_message(chat_id=spinner_msg.chat_id, message_id=spinner_msg.message_id)
        db_log("ERROR", "poll_solver_failed", {"user_id": uid, "error": str(e)})
        await err(update, "Solve Failed", "AI backend is temporarily unavailable. Please try again.")


async def handle_user_text_unusual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id
    if is_banned(uid):
        return
    role = get_role(uid)
    private = is_private_chat(update)
    if role == ROLE_USER:
        if private:
            if not solver_mode_on(uid):
                await warn_unauthorized(update, "This bot is currently restricted for staff operations. Please use /ask [message] for support.")
                return
            if not await enforce_required_memberships(update, context):
                return
        else:
            if not is_group_ai_enabled(update.effective_chat.id):
                return
            if not await enforce_required_memberships(update, context):
                return
    elif role in (ROLE_ADMIN, ROLE_OWNER):
        if not private or not solver_mode_on(uid):
            return
    else:
        return
    user_text = (update.message.text or "").strip()
    if not user_text:
        return
    reply_msg = update.message.reply_to_message
    if reply_msg:
        ctx = _get_quiz_context(context, reply_msg.message_id)
        if not ctx and getattr(reply_msg, 'poll', None):
            poll = reply_msg.poll
            ctx = {
                "question": str(poll.question or "").strip(),
                "options": [str(o.text).strip() for o in (poll.options or []) if str(o.text or '').strip()],
                "official_ans": _poll_official_answer(poll),
                "official_expl": str(getattr(poll, 'explanation', '') or '').strip(),
            }
        if ctx:
            qtext = str(ctx.get("question", "") or "").strip()
            opts = ctx.get("options", []) or []
            prompt = f"Question:\n{qtext}\n\nOptions:\n" + "\n".join([f"{_safe_letter(i+1)}. {o}" for i, o in enumerate(opts)]) + f"\n\nUser follow-up:\n{user_text}"
            await send_solver_picker(update, context, prompt)
            return
    await send_solver_picker(update, context, user_text)


async def cmd_probaho_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    uid = update.effective_user.id if update.effective_user else 0
    if not chat or chat.type not in ("group", "supergroup"):
        await warn(update, "Group Only", "Use this command inside a group/supergroup.")
        return
    if not await _is_group_admin(context, chat.id, uid):
        await warn(update, "Unauthorized", "Only a group admin can use this command.")
        return
    set_group_ai_enabled(chat.id, True)
    await ok(update, "Group AI Enabled", f"Users can now get AI responses in this group: {chat.id}")


async def cmd_probaho_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    uid = update.effective_user.id if update.effective_user else 0
    if not chat or chat.type not in ("group", "supergroup"):
        await warn(update, "Group Only", "Use this command inside a group/supergroup.")
        return
    if not await _is_group_admin(context, chat.id, uid):
        await warn(update, "Unauthorized", "Only a group admin can use this command.")
        return
    set_group_ai_enabled(chat.id, False)
    await ok(update, "Group AI Disabled", f"Users will no longer get AI responses in this group: {chat.id}")


def _emoji_quiz_text(question: str, options: List[str], title: str) -> str:
    lines = [str(title or BOT_BRAND).strip(), "", str(question or "").strip(), ""]
    labels = EMOJI_BUTTONS[:len(options)]
    for i, opt in enumerate(options):
        label = labels[i] if i < len(labels) else f"{_safe_letter(i+1)})"
        lines.append(f"{label} {opt}")
    return "\n".join([x for x in lines if x is not None]).strip()


@require_admin
async def cmd_postemoji(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("postemoji", "<DB-ID> [keep]", "Post buffered questions as emoji quiz to a channel"))
        return
    cid = int(context.args[0])
    keep = (len(context.args) > 1 and context.args[1].strip().lower() == "keep")
    ch = channel_get_by_id_for_user(admin_id, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or no access.")
        return
    items = buffer_list(admin_id, limit=MAX_BUFFERED_QUESTIONS)
    if not items:
        await warn(update, "Buffer Empty", "No buffered questions found.")
        return
    prefix = str(getattr(ch, "prefix", "") or "").strip()
    title = prefix if prefix else BOT_BRAND
    sent = 0
    sent_ids = []
    for bid, payload in items:
        qtext, opts, corr_idx0, explanation = quiz_to_poll_parts(payload)
        if not opts:
            continue
        msg_text = _emoji_quiz_text(qtext, opts, title)
        quiz_id = uuid.uuid4().hex[:10]
        try:
            m = await context.bot.send_message(
                chat_id=ch.channel_chat_id,
                text=msg_text,
                reply_markup=emoji_quiz_keyboard(len(opts), quiz_id),
                disable_web_page_preview=True,
            )
            sent += 1
            sent_ids.append(bid)
            emoji_quiz_save(
                quiz_id,
                ch.channel_chat_id,
                m.message_id,
                {
                    "question": qtext,
                    "options": opts,
                    "correct_answer": corr_idx0 + 1 if corr_idx0 >= 0 else 0,
                    "explanation": explanation,
                    "prefix": title,
                },
                admin_id,
            )
            await asyncio.sleep(0.25)
        except Exception as e:
            db_log("ERROR", "postemoji_failed", {"admin_id": admin_id, "channel": getattr(ch, 'channel_chat_id', 0), "error": str(e)})
    if sent and not keep:
        buffer_remove_ids(admin_id, sent_ids)
    await ok_html(update, "Emoji Quiz Posted", f"Sent: <code>{h(sent)}</code>\nChannel: <code>{h(getattr(ch, 'title', cid))}</code>")


async def on_emoji_quiz_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    q = update.callback_query
    data = (q.data or "").strip()
    m = re.match(r"^eq:([0-9a-f]{6,16}):(\d+)$", data)
    if not m:
        return
    quiz_id = m.group(1)
    selected = int(m.group(2))
    uid = q.from_user.id if q.from_user else 0
    if not uid:
        return
    ok_member, _missing = await user_meets_required_memberships(context, uid)
    if not ok_member:
        await q.answer("⚠️ Join the required channel first, then press I Joined.", show_alert=True)
        return
    quiz = emoji_quiz_get(quiz_id)
    if not quiz:
        await q.answer("Quiz expired or not found.", show_alert=True)
        return
    saved_choice = emoji_quiz_user_choice(quiz_id, uid)
    correct = int(quiz.get("correct_answer", 0) or 0)
    opts = quiz.get("options", []) or []
    expl = clean_latex(str(quiz.get("explanation", "") or "").strip())
    expl = re.sub(r"\s+", " ", expl).strip()
    if len(expl) > 150:
        expl = expl[:147] + "..."
    corr_label = EMOJI_BUTTONS[correct - 1] if 0 < correct <= len(EMOJI_BUTTONS) else str(correct)

    if saved_choice and int(saved_choice) != int(selected):
        saved_label = EMOJI_BUTTONS[saved_choice - 1] if 0 < saved_choice <= len(EMOJI_BUTTONS) else str(saved_choice)
        await q.answer(f"⚠️ You already answered with {saved_label}. Tap the same reaction to view your result.", show_alert=True)
        return

    if not saved_choice:
        emoji_quiz_record_answer(quiz_id, uid, selected, (selected == correct and correct > 0))
        sel_label = EMOJI_BUTTONS[selected - 1] if 0 < selected <= len(EMOJI_BUTTONS) else str(selected)
        if selected == correct and correct > 0:
            first_msg = f"🎉🎊 Congratulations!\n✅ Correct: {corr_label}\nYour reaction: {sel_label}\n\nTap the same reaction again for explanation & stats."
        else:
            first_msg = f"❌ Wrong answer\n✅ Correct: {corr_label}\nYour reaction: {sel_label}\n\nTap the same reaction again for explanation & stats."
        await q.answer(first_msg[:190], show_alert=True)
        return

    counts = emoji_quiz_counts(quiz_id)
    sel_label = EMOJI_BUTTONS[saved_choice - 1] if 0 < saved_choice <= len(EMOJI_BUTTONS) else str(saved_choice)
    stats_text = " | ".join([f"{EMOJI_BUTTONS[i-1]}={counts.get(i, 0)}" for i in range(1, len(opts) + 1)])
    if saved_choice == correct and correct > 0:
        msg = f"🎉🎊 Correct\nYour reaction: {sel_label}\n✅ Correct: {corr_label}"
    else:
        msg = f"❌ Wrong\nYour reaction: {sel_label}\n✅ Correct: {corr_label}"
    if stats_text:
        msg += f"\n{stats_text}"
    if expl:
        msg += f"\n\n{expl}"
    await q.answer(msg[:190], show_alert=True)


async def on_solver_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    q = update.callback_query
    await q.answer("Processing…", show_alert=False)
    data = (q.data or "").strip()
    m = re.match(r"^solve:([GPD]):([0-9a-f]{6,16})$", data)
    if not m:
        return
    model = m.group(1)
    token = m.group(2)
    store = _pending_store(context)
    req = store.get(token)
    if not isinstance(req, dict):
        with contextlib.suppress(Exception):
            await q.edit_message_text("⚠️ This request has expired. Please send your question again.")
        return
    uid = int(req.get("uid") or 0)
    if q.from_user and q.from_user.id != uid:
        with contextlib.suppress(Exception):
            await q.answer("This is not your request.", show_alert=True)
        return
    payload = req.get("payload") or {}
    problem_text = str(payload.get("text") or "").strip()
    kind = str(req.get("kind") or "text").lower()
    with contextlib.suppress(Exception):
        await q.edit_message_text(ui_box_text("Solving", "Please wait… Processing your request.", emoji="⏳"), parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    try:
        if kind == "poll" and payload.get("question"):
            question = str(payload.get("question", "")).strip()
            options = payload.get("options", [])
            result, model_name = await _run_blocking(_role_of(uid), _solve_mcq_with_preference, model, question, options)
            raw_expl = str(result.get('explanation', '') or "")
            clean_expl = clean_latex(raw_expl)
            raw_why_not = result.get("why_not", {}) or {}
            clean_why_not = {k: clean_latex(v) for k, v in raw_why_not.items()}
            msg_html = _format_user_poll_solution(
                question=question,
                options=options,
                model_ans=int(result.get("answer", 0) or 0),
                official_ans=int(payload.get("official_ans", 0) or 0),
                model_expl=f"[{model_name}]\n{clean_expl}".strip(),
                official_expl=str(payload.get("official_expl", "")).strip(),
                why_not=clean_why_not,
                conf=int(result.get("confidence", 0) or 0),
            )
            kb = _verify_kb(token, model, "poll")
        else:
            answer, _used = await _run_blocking(_role_of(uid), _solve_text_with_preference, model, problem_text)
            if (is_admin(uid) or is_owner(uid)) and (looks_like_programming_request(problem_text) or looks_like_programming_request(answer)):
                msg_html = f"<pre>{h(answer)}</pre>"
            else:
                msg_html = h(answer)
            kb = _verify_kb(token, model, "text")
        with contextlib.suppress(Exception):
            await q.edit_message_text(msg_html, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            if q.message and kind == "poll":
                _remember_quiz_context(context, q.message.message_id, payload)
    except Exception as e:
        db_log("ERROR", "solver_callback_failed", {"user_id": uid, "model": model, "error": str(e)})
        with contextlib.suppress(Exception):
            await q.edit_message_text(ui_box_text("Solve Failed", "AI backend is temporarily unavailable. Please try again.", emoji="❌"), parse_mode=ParseMode.HTML)




# ===========================
# FINAL UX PATCHES (2026-03-11)
# ===========================

def _profile_link_keyboard(user_id: int, username: Optional[str] = None) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton("👤 Open Profile", url=f"tg://user?id={int(user_id)}")]]
    un = str(username or "").lstrip("@").strip()
    if un:
        rows.append([InlineKeyboardButton(f"🌐 @{un}", url=f"https://t.me/{un}")])
    return InlineKeyboardMarkup(rows)


@require_admin
async def cmd_usersd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not str(context.args[0]).lstrip("-").isdigit():
        await safe_reply(update, usage_box("usersd", "<user_id>", "Show a clickable profile button for a user ID"))
        return
    target = int(context.args[0])
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT first_name, username, role, is_banned, created_at, last_seen_at FROM users WHERE user_id=?", (target,))
    row = cur.fetchone(); conn.close()
    if row:
        name = row["first_name"] or str(target)
        username = row["username"] or ""
        uname = ("@" + username) if username else "(none)"
        body = (
            f"Profile: {mention_user(target, name)}\n"
            f"User ID: <code>{h(target)}</code>\n"
            f"Username: {h(uname)}\n"
            f"Role: <code>{h(row['role'] or 'USER')}</code>\n"
            f"Banned: <code>{'Yes' if int(row['is_banned'] or 0) else 'No'}</code>\n"
            f"Created: <code>{h(row['created_at'] or '')}</code>\n"
            f"Last Seen: <code>{h(row['last_seen_at'] or '')}</code>"
        )
        kb = _profile_link_keyboard(target, username)
    else:
        body = (
            f"Profile: {mention_user(target, str(target))}\n"
            f"User ID: <code>{h(target)}</code>\n"
            f"Stored info: <code>Not found in local users table</code>"
        )
        kb = _profile_link_keyboard(target, None)
    await update.message.reply_text(
        ui_box_html("User Profile Link", body, emoji="🔎"),
        parse_mode=ParseMode.HTML,
        reply_markup=kb,
        disable_web_page_preview=True,
    )


def _format_user_poll_solution(question: str, options: List[str], model_ans: int, official_ans: int, model_expl: str, official_expl: str, why_not: Dict[str, str], conf: int) -> str:
    opts = [(o or "").strip() for o in (options or []) if (o or "").strip()][:5]
    copy_block = _copyable_quiz_block(question or "", opts)
    lines = ["<b>📊 Quiz Solution</b>", "", "<b>Question + Options (copyable):</b>", copy_block]
    if 1 <= int(model_ans or 0) <= len(opts):
        lines.append(f"\n<b>✅ AI Response:</b> <b>{_safe_letter(model_ans)}</b>) {h(opts[model_ans-1])}")
    if official_ans > 0 and official_ans <= len(opts):
        tag = "✅ Match" if official_ans == model_ans else "❌ Mismatch"
        lines.append(f"<b>📌 Given Answer:</b> <b>{_safe_letter(official_ans)}</b>) {h(opts[official_ans-1])} <i>({tag})</i>")
    if model_expl:
        lines.append("\n<b>Explanation (Solved):</b>")
        lines.append(h(model_expl))
    if official_expl:
        lines.append("\n<b>Explanation (From Quiz):</b>")
        lines.append(h(official_expl))
    if why_not:
        wn = []
        for k in ["A", "B", "C", "D", "E"]:
            v = (why_not or {}).get(k)
            if v:
                wn.append(f"• <b>{h(k)}</b>: {h(v)}")
        if wn:
            lines.append("\n<b>Why other options are wrong:</b>\n" + "\n".join(wn))
    return "\n".join(lines).strip()


async def on_emoji_quiz_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    q = update.callback_query
    data = (q.data or "").strip()
    m = re.match(r"^eq:([0-9a-f]{6,16}):(\d+)$", data)
    if not m:
        return
    quiz_id = m.group(1)
    selected = int(m.group(2))
    uid = q.from_user.id if q.from_user else 0
    if not uid:
        return

    ok_member, _missing = await user_meets_required_memberships(context, uid)
    if not ok_member:
        await q.answer("⚠️ Join the required channel first, then press I Joined.", show_alert=True)
        return

    quiz = emoji_quiz_get(quiz_id)
    if not quiz:
        await q.answer("Quiz expired or not found.", show_alert=True)
        return

    saved_choice = emoji_quiz_user_choice(quiz_id, uid)
    correct = int(quiz.get("correct_answer", 0) or 0)
    opts = quiz.get("options", []) or []
    expl = clean_latex(str(quiz.get("explanation", "") or "").strip())
    expl = re.sub(r"\s+", " ", expl).strip()
    if len(expl) > 150:
        expl = expl[:147] + "..."

    corr_label = EMOJI_BUTTONS[correct - 1] if 0 < correct <= len(EMOJI_BUTTONS) else str(correct)

    if saved_choice and int(saved_choice) != int(selected):
        saved_label = EMOJI_BUTTONS[saved_choice - 1] if 0 < saved_choice <= len(EMOJI_BUTTONS) else str(saved_choice)
        await q.answer(f"⚠️ You already answered with {saved_label}. Tap the same reaction to view your result.", show_alert=True)
        return

    if not saved_choice:
        emoji_quiz_record_answer(quiz_id, uid, selected, (selected == correct and correct > 0))
        sel_label = EMOJI_BUTTONS[selected - 1] if 0 < selected <= len(EMOJI_BUTTONS) else str(selected)
        if selected == correct and correct > 0:
            # Bot API callback answers only support text/alert/url/cache; no native quiz-confetti trigger.
            toast = "🎉🎊 Congratulations! Tap the same reaction again for explanation & stats."
            await q.answer(toast[:190], show_alert=False)
        else:
            first_msg = f"❌ Wrong answer\n✅ Correct: {corr_label}\nYour reaction: {sel_label}\n\nTap the same reaction again for explanation & stats."
            await q.answer(first_msg[:190], show_alert=True)
        return

    counts = emoji_quiz_counts(quiz_id)
    sel_label = EMOJI_BUTTONS[saved_choice - 1] if 0 < saved_choice <= len(EMOJI_BUTTONS) else str(saved_choice)
    stats_text = " | ".join([f"{EMOJI_BUTTONS[i-1]}={counts.get(i, 0)}" for i in range(1, len(opts) + 1)])
    if saved_choice == correct and correct > 0:
        msg = f"🎉🎊 Congratulations!\nYour reaction: {sel_label}"
    else:
        msg = f"❌ Wrong\nYour reaction: {sel_label}\n✅ Correct: {corr_label}"
    if stats_text:
        msg += f"\n{stats_text}"
    if expl:
        msg += f"\n\n{expl}"
    await q.answer(msg[:190], show_alert=True)

# ===========================
# FINAL PATCHES (2026-03-13)
# ===========================

def _strip_leading_quiz_noise(q: str) -> str:
    s = str(q or '').strip()
    # Drop repeated source tags like [White Apron 🩺] at the beginning.
    while True:
        new_s = re.sub(r'^\s*\[[^\]]{1,80}\]\s*', '', s)
        if new_s == s:
            break
        s = new_s.strip()
    # Drop serials such as "124.", "১২৪।", "238)", repeated if needed.
    s = re.sub(r'^\s*(?:(?:\d+|[০-৯]+)\s*[\.)।:\-]+\s*)+', '', s).strip()
    return s


def _shuffle_quiz_payload(question: str, options: List[str], correct_option_id0: int) -> Tuple[str, List[str], int]:
    import random
    q = _strip_leading_quiz_noise(question)
    opts = [str(o).strip() for o in (options or []) if str(o).strip()]
    if len(opts) < 2:
        return q, opts, correct_option_id0
    order = list(range(len(opts)))
    random.shuffle(order)
    shuffled = [opts[i] for i in order]
    new_correct = -1
    if 0 <= int(correct_option_id0) < len(opts):
        try:
            new_correct = order.index(int(correct_option_id0))
        except Exception:
            new_correct = -1
    return q, shuffled, new_correct


def _score_reply_text(total_posted: int) -> str:
    return f"📝 Your score: ____ / {int(total_posted or 0)}"


def quiz_to_poll_parts(payload: Dict[str, Any]) -> Tuple[str, List[str], int, str]:
    q = str(payload.get("questions", "")).strip()
    q2, expl2 = split_inline_explain(q)
    if expl2 and not str(payload.get("explanation", "")).strip():
        q = q2.strip()
        payload = dict(payload)
        payload["explanation"] = expl2.strip()
    else:
        q = q2.strip()
    opts = [
        str(payload.get("option1", "")).strip(),
        str(payload.get("option2", "")).strip(),
        str(payload.get("option3", "")).strip(),
        str(payload.get("option4", "")).strip(),
        str(payload.get("option5", "")).strip(),
    ]
    opts = [o for o in opts if o]
    if len(opts) < 2:
        if len(opts) == 0:
            opts = ["Option A", "Option B"]
        else:
            opts = opts + ["Option B"]
    if len(opts) > 10:
        opts = opts[:10]
    ans = int(payload.get("answer", 0) or 0)
    correct_option_id = ans - 1 if 1 <= ans <= len(opts) else -1
    explanation = str(payload.get("explanation", "")).strip()
    q, opts, correct_option_id = _shuffle_quiz_payload(q, opts, correct_option_id)
    return q, opts, correct_option_id, explanation


@require_admin
async def cmd_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("post", "<DB-ID> [keep]", "Post buffered quizzes to a channel. Use 'keep' to keep buffer."))
        return

    cid = int(context.args[0])
    keep = (len(context.args) > 1 and context.args[1].strip().lower() == "keep")
    ch = channel_get_by_id_for_user(admin_id, cid)
    if not ch:
        await warn_html(update, "Channel Not Found", f"No access to that channel. Use <code>/listchannels</code> to view yours.")
        return

    items = buffer_list(admin_id, limit=MAX_BUFFERED_QUESTIONS)
    if not items:
        await warn(update, "Buffer Empty", "No quizzes to post. Send text or forward polls first.")
        return

    await info_html(update, "Posting to Channel", f"<code>{h(ch.title)}</code> — <code>{h(str(ch.channel_chat_id))}</code>\n\nPosting <code>{h(len(items))}</code> question(s)...")

    posted_ids: List[int] = []
    ok_count, fail_count = 0, 0
    first_post_message_id = None

    for (row_id, payload) in items:
        try:
            q, opts, correct_option_id, expl = quiz_to_poll_parts(payload)
            prefix = (ch.prefix or "").strip(" ")
            expl_link = (ch.expl_link or "").strip()
            SEP = "\n\u200b"
            q_final = f"{prefix}{SEP}{q}".strip() if prefix else q
            if len(q_final) > 300:
                q_final = q_final[:297] + "..."

            expl_final = expl.strip()
            if not explain_mode_on(admin_id):
                expl_final = ""
            if expl_link:
                expl_final = (expl_final + "\n\n" if expl_final else "") + f"🔗 {expl_link}"
            expl_final = expl_final.strip()
            if len(expl_final) > 200:
                expl_final = expl_final[:197] + "..."

            if correct_option_id >= 0:
                m = await context.bot.send_poll(
                    chat_id=ch.channel_chat_id,
                    question=q_final,
                    options=opts,
                    is_anonymous=True,
                    type=Poll.QUIZ,
                    correct_option_id=correct_option_id,
                    explanation=expl_final if expl_final else None,
                )
            else:
                m = await context.bot.send_poll(
                    chat_id=ch.channel_chat_id,
                    question=q_final,
                    options=opts,
                    is_anonymous=True,
                    type=Poll.REGULAR,
                )
                if expl_final:
                    await context.bot.send_message(chat_id=ch.channel_chat_id, text=f"📖 {expl_final}", disable_web_page_preview=True)

            if first_post_message_id is None and getattr(m, 'message_id', None):
                first_post_message_id = m.message_id
            ok_count += 1
            posted_ids.append(row_id)
            await asyncio.sleep(POST_DELAY_SECONDS)
        except RetryAfter as e:
            await asyncio.sleep(float(e.retry_after) + 0.5)
            fail_count += 1
        except TelegramError as e:
            fail_count += 1
            db_log("ERROR", "post_failed", {"admin_id": admin_id, "channel": ch.channel_chat_id, "error": str(e)})
        except Exception as e:
            fail_count += 1
            db_log("ERROR", "post_failed_unknown", {"admin_id": admin_id, "error": str(e)})

    if ok_count > 0 and first_post_message_id:
        with contextlib.suppress(Exception):
            await context.bot.send_message(
                chat_id=ch.channel_chat_id,
                text=_score_reply_text(ok_count),
                reply_to_message_id=first_post_message_id,
                allow_sending_without_reply=True,
            )

    inc_admin_post(admin_id, ok_count)
    if posted_ids and not keep:
        buffer_remove_ids(admin_id, posted_ids)
    body = f"Posted: {ok_count}\nFailed: {fail_count}\nRemaining in Buffer: {buffer_count(admin_id)}"
    await ok(update, "Posting Complete", body)


@require_admin
async def cmd_postemoji(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("postemoji", "<DB-ID> [keep]", "Post buffered questions as emoji quiz to a channel"))
        return
    cid = int(context.args[0])
    keep = (len(context.args) > 1 and context.args[1].strip().lower() == "keep")
    ch = channel_get_by_id_for_user(admin_id, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or no access.")
        return
    items = buffer_list(admin_id, limit=MAX_BUFFERED_QUESTIONS)
    if not items:
        await warn(update, "Buffer Empty", "No buffered questions found.")
        return
    prefix = str(getattr(ch, "prefix", "") or "").strip()
    title = prefix if prefix else BOT_BRAND
    sent = 0
    sent_ids = []
    first_post_message_id = None
    for bid, payload in items:
        qtext, opts, corr_idx0, explanation = quiz_to_poll_parts(payload)
        if not opts:
            continue
        msg_text = _emoji_quiz_text(qtext, opts, title)
        quiz_id = uuid.uuid4().hex[:10]
        try:
            m = await context.bot.send_message(
                chat_id=ch.channel_chat_id,
                text=msg_text,
                reply_markup=emoji_quiz_keyboard(len(opts), quiz_id),
                disable_web_page_preview=True,
            )
            if first_post_message_id is None:
                first_post_message_id = m.message_id
            sent += 1
            sent_ids.append(bid)
            emoji_quiz_save(
                quiz_id,
                ch.channel_chat_id,
                m.message_id,
                {
                    "question": qtext,
                    "options": opts,
                    "correct_answer": corr_idx0 + 1 if corr_idx0 >= 0 else 0,
                    "explanation": explanation,
                    "prefix": title,
                },
                admin_id,
            )
            await asyncio.sleep(0.25)
        except Exception as e:
            db_log("ERROR", "postemoji_failed", {"admin_id": admin_id, "channel": getattr(ch, 'channel_chat_id', 0), "error": str(e)})
    if sent > 0 and first_post_message_id:
        with contextlib.suppress(Exception):
            await context.bot.send_message(
                chat_id=ch.channel_chat_id,
                text=_score_reply_text(sent),
                reply_to_message_id=first_post_message_id,
                allow_sending_without_reply=True,
            )
    if sent and not keep:
        buffer_remove_ids(admin_id, sent_ids)
    await ok_html(update, "Emoji Quiz Posted", f"Sent: <code>{h(sent)}</code>\nChannel: <code>{h(getattr(ch, 'title', cid))}</code>")

def build_app() -> Application:
    db_init(); extra_db_init()
    # keep original image extraction around for override wrapper
    globals().setdefault("_original_handle_image", globals().get("handle_image"))
    builder = ApplicationBuilder().token(BOT_TOKEN)
    try:
        builder = builder.concurrent_updates(64)
    except Exception:
        pass
    app = builder.build()
    app.add_handler(_cmdh("start", cmd_start))
    app.add_handler(_cmdh("help", cmd_help))
    app.add_handler(_cmdh("commands", cmd_commands))
    app.add_handler(_cmdh("features", cmd_features))
    app.add_handler(CallbackQueryHandler(on_solver_callback, pattern=r"^solve:"))
    app.add_handler(CallbackQueryHandler(on_genquiz_callback, pattern=r"^genquiz:"))
    app.add_handler(CallbackQueryHandler(on_emoji_quiz_callback, pattern=r"^eq:"))
    app.add_handler(CallbackQueryHandler(on_required_verify_callback, pattern=r"^req:verify$"))
    app.add_handler(_cmdh("ask", cmd_ask))
    app.add_handler(_cmdh("scanhelp", cmd_scanhelp))
    app.add_handler(_cmdh("vision_on", cmd_vision_on))
    app.add_handler(_cmdh("vision_off", cmd_vision_off))
    app.add_handler(_cmdh("solve_on", cmd_solve_on))
    app.add_handler(_cmdh("solve_off", cmd_solve_off))
    app.add_handler(_cmdh("himusai_on", cmd_himusai_on))
    app.add_handler(_cmdh("himusai_off", cmd_himusai_off))
    app.add_handler(_cmdh("probaho_on", cmd_probaho_on))
    app.add_handler(_cmdh("probaho_off", cmd_probaho_off))
    app.add_handler(_cmdh("explain_on", cmd_explain_on))
    app.add_handler(_cmdh("explain_off", cmd_explain_off))
    app.add_handler(_cmdh("quizprefix", cmd_quizprefix))
    app.add_handler(_cmdh("quizlink", cmd_quizlink))
    app.add_handler(_cmdh("addadmin", cmd_addadmin))
    app.add_handler(_cmdh("removeadmin", cmd_removeadmin))
    app.add_handler(_cmdh("grantall", cmd_grantall))
    app.add_handler(_cmdh("revokeall", cmd_revokeall))
    app.add_handler(_cmdh("grantvision", cmd_grantvision))
    app.add_handler(_cmdh("revokevision", cmd_revokevision))
    app.add_handler(_cmdh("addrequired", cmd_addrequired))
    app.add_handler(_cmdh("delrequired", cmd_delrequired))
    app.add_handler(_cmdh("listrequired", cmd_listrequired))
    app.add_handler(_cmdh("ownerstats", cmd_ownerstats))
    app.add_handler(_cmdh("users", cmd_users))
    app.add_handler(_cmdh("usersd", cmd_usersd))
    app.add_handler(_cmdh("filter", cmd_filter))
    app.add_handler(_cmdh("done", cmd_done))
    app.add_handler(_cmdh("clear", cmd_clear))
    app.add_handler(_cmdh("addchannel", cmd_addchannel))
    app.add_handler(_cmdh("listchannels", cmd_listchannels))
    app.add_handler(_cmdh("removechannel", cmd_removechannel))
    app.add_handler(_cmdh("setprefix", cmd_setprefix))
    app.add_handler(_cmdh("setexplink", cmd_setexplink))
    app.add_handler(_cmdh("post", cmd_post))
    app.add_handler(_cmdh("postemoji", cmd_postemoji))
    app.add_handler(_cmdh("broadcast", cmd_broadcast))
    app.add_handler(_cmdh("adminpanel", cmd_adminpanel))
    app.add_handler(_cmdh("reply", cmd_reply))
    app.add_handler(_cmdh("close", cmd_close))
    app.add_handler(_cmdh("ban", cmd_ban))
    app.add_handler(_cmdh("unban", cmd_unban))
    app.add_handler(_cmdh("banned", cmd_banned))
    app.add_handler(_cmdh("private_send", cmd_private_send))
    app.add_handler(_cmdh("send_private", cmd_private_send))
    app.add_handler(MessageHandler(filters.POLL, handle_poll))
    app.add_handler(MessageHandler(filters.POLL, handle_user_poll_solver), group=1)
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_user_text_unusual), group=1)
    app.add_error_handler(on_error)
    return app

def main():
    app = build_app()
    # Render free web service port binding
    try:
        threading.Thread(target=_run_render_health_server, daemon=True).start()
    except Exception:
        logging.exception("Failed to start Render health server")

  
    try:
        # Attempt to reconfigure stdout to UTF-8 encoding for Windows compatibility
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        print(f"🤖 {BOT_BRAND} started. OWNER_ID={OWNER_ID} DB={DB_PATH}")
    except (UnicodeEncodeError, AttributeError, TypeError):
        # Fallback to ASCII-only message if encoding fails
        try:
            print("[BOT] Started. OWNER_ID={} DB={}".format(OWNER_ID, DB_PATH))
        except:
            # Final fallback - use logging instead
            logging.info(f"Bot started. OWNER_ID={OWNER_ID} DB={DB_PATH}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


# ===========================
# FINAL PATCHES (2026-03-13)
# ===========================

def _profile_link_keyboard(user_id: int, username: Optional[str] = None) -> Optional[InlineKeyboardMarkup]:
    """Safer profile keyboard: only public username links are used.
    tg://user buttons can fail with Button_user_privacy_restricted.
    """
    un = str(username or '').lstrip('@').strip()
    if not un:
        return None
    return InlineKeyboardMarkup([[InlineKeyboardButton(f"👤 Open @{un}", url=f"https://t.me/{un}")]])


@require_admin
async def cmd_usersd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not str(context.args[0]).lstrip('-').isdigit():
        await safe_reply(update, usage_box("usersd", "<user_id>", "Show user details and a public profile button if available"))
        return
    target = int(context.args[0])
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT first_name, username, role, is_banned, created_at, last_seen_at FROM users WHERE user_id=?", (target,))
    row = cur.fetchone(); conn.close()
    if row:
        name = row["first_name"] or str(target)
        username = row["username"] or ""
        uname = ("@" + username) if username else "(no public username)"
        body = (
            f"Profile: {mention_user(target, name)}\n"
            f"User ID: <code>{h(target)}</code>\n"
            f"Username: {h(uname)}\n"
            f"Role: <code>{h(row['role'] or 'USER')}</code>\n"
            f"Banned: <code>{'Yes' if int(row['is_banned'] or 0) else 'No'}</code>\n"
            f"Created: <code>{h(row['created_at'] or '')}</code>\n"
            f"Last Seen: <code>{h(row['last_seen_at'] or '')}</code>"
        )
        kb = _profile_link_keyboard(target, username)
        if not kb:
            body += "\n\n⚠️ Public profile button unavailable for this user."
    else:
        body = (
            f"Profile: {mention_user(target, str(target))}\n"
            f"User ID: <code>{h(target)}</code>\n"
            f"Stored info: <code>Not found in local users table</code>\n\n"
            f"⚠️ Public profile button unavailable unless the user has a username."
        )
        kb = None
    await update.message.reply_text(
        ui_box_html("User Details", body, emoji="🔎"),
        parse_mode=ParseMode.HTML,
        reply_markup=kb,
        disable_web_page_preview=True,
    )


def _buffer_feedback_key(chat_id: int, user_id: int) -> str:
    return f"_buffer_feedback:{int(chat_id)}:{int(user_id)}"


async def _show_buffer_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE, title: str, body_html: str, emoji: str = "✅") -> None:
    if not update.message or not update.effective_chat or not update.effective_user:
        return
    chat_id = int(update.effective_chat.id)
    uid = int(update.effective_user.id)
    lock = _get_chat_lock(context, chat_id)
    async with lock:
        key = _buffer_feedback_key(chat_id, uid)
        prev_mid = context.application.bot_data.get(key)
        if isinstance(prev_mid, int):
            with contextlib.suppress(Exception):
                await context.bot.delete_message(chat_id=chat_id, message_id=prev_mid)
        msg = await update.message.reply_text(
            ui_box_html(title, body_html, emoji=emoji),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        context.application.bot_data[key] = int(msg.message_id)


@require_admin_silent
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id if update.effective_user else 0
    if not uid or is_banned(uid):
        return
    role = get_role(uid)
    if role not in (ROLE_ADMIN, ROLE_OWNER):
        return
    if is_private_chat(update) and (solver_mode_on(uid) or himusai_mode_on(uid)):
        return
    text = update.message.text or ""
    if not text.strip():
        return
    if buffer_count(uid) >= MAX_BUFFERED_QUESTIONS:
        await warn(update, "Buffer Limit Reached", f"You have {MAX_BUFFERED_QUESTIONS} questions buffered.\n\nUse /done to export or /clear to reset.")
        return
    blocks = split_blocks(text)
    added = 0
    for b in blocks:
        if buffer_count(uid) >= MAX_BUFFERED_QUESTIONS:
            break
        try:
            payload = parse_text_block(b, uid)
            if payload:
                buffer_add(uid, payload)
                added += 1
        except Exception as e:
            db_log("ERROR", "parse_text_failed", {"admin_id": uid, "error": str(e)})
    if added:
        await _show_buffer_feedback(
            update,
            context,
            "Added to Buffer",
            f"<code>{h(added)}</code> question(s) added.\n\nTotal buffered: <code>{h(buffer_count(uid))}</code>",
        )
    else:
        await warn(update, "No Questions Found", "No valid quiz blocks detected. Check formatting.")


@require_admin_silent
async def handle_poll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    uid = update.effective_user.id if update.effective_user else 0
    if not uid or is_banned(uid):
        return
    role = get_role(uid)
    if role not in (ROLE_ADMIN, ROLE_OWNER):
        return
    if is_private_chat(update) and (solver_mode_on(uid) or himusai_mode_on(uid)):
        return
    poll = update.message.poll
    question = clean_common(poll.question or "", uid)
    options = [o.text for o in poll.options]
    opts = options + [""] * (5 - len(options))
    explanation = ""
    if hasattr(poll, "explanation") and poll.explanation:
        explanation = clean_explanation(poll.explanation, uid)
    correct_answer_id = 0
    if poll.type == "quiz" and poll.correct_option_id is not None:
        correct_answer_id = int(poll.correct_option_id) + 1
    payload = {
        "questions": question,
        "option1": (opts[0] or "").strip(),
        "option2": (opts[1] or "").strip(),
        "option3": (opts[2] or "").strip(),
        "option4": (opts[3] or "").strip(),
        "option5": (opts[4] or "").strip(),
        "answer": correct_answer_id,
        "explanation": explanation,
        "type": 1,
        "section": 1,
    }
    if buffer_count(uid) >= MAX_BUFFERED_QUESTIONS:
        await warn_html(update, "Buffer Limit Reached", f"You have <code>{h(MAX_BUFFERED_QUESTIONS)}</code> questions buffered.\n\nUse <code>/done</code> to export or <code>/clear</code> to reset.")
        return
    buffer_add(uid, payload)
    note = ""
    if correct_answer_id == 0 and poll.type == "quiz":
        note = "<br><br>⚠️ Telegram may hide the correct answer in forwarded quizzes. Export will store <code>answer=0</code>."
    await _show_buffer_feedback(
        update,
        context,
        "Poll Saved",
        f"Total buffered: <code>{buffer_count(uid)}</code>{note}",
    )


@require_admin
async def cmd_buffercount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else 0
    cnt = buffer_count(uid)
    items = buffer_list(uid, limit=3)
    preview = []
    for rid, payload in items:
        q = str(payload.get('questions', '') or '').strip()
        if q:
            preview.append(f"• <code>{rid}</code> — {h(q[:70])}{'...' if len(q) > 70 else ''}")
    body = f"Total buffered: <code>{cnt}</code>"
    if preview:
        body += "\n\nLatest items:\n" + "\n".join(preview)
    await info_html(update, "Buffer Status", body)


@require_admin
async def cmd_imgreact(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id if update.effective_user else 0
    if not context.args or len(context.args) < 2 or not str(context.args[0]).isdigit() or not str(context.args[1]).isdigit():
        await safe_reply(update, usage_box("imgreact", "<DB-ID> <correct_emoji_no 1-4> [explanation]", "Reply to a photo/image and post it as an image reaction quiz"))
        return
    cid = int(context.args[0])
    corr = int(context.args[1])
    if corr < 1 or corr > 4:
        await warn(update, "Invalid Answer", "correct_emoji_no must be between 1 and 4.")
        return
    ch = channel_get_by_id_for_user(admin_id, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or no access.")
        return
    if not update.message or not update.message.reply_to_message:
        await warn(update, "Reply Required", "Reply to a photo/image message with /imgreact <DB-ID> <correct_emoji_no> [explanation]")
        return
    src = update.message.reply_to_message
    photo_file_id = None
    if getattr(src, 'photo', None):
        photo_file_id = src.photo[-1].file_id
    elif getattr(src, 'document', None) and str(getattr(src.document, 'mime_type', '')).startswith('image/'):
        photo_file_id = src.document.file_id
    if not photo_file_id:
        await warn(update, "Image Required", "Reply to a photo or image document.")
        return
    explanation = " ".join(context.args[2:]).strip()
    prefix = str(getattr(ch, 'prefix', '') or '').strip() or BOT_BRAND
    caption_parts = [prefix]
    src_caption = str(getattr(src, 'caption', '') or '').strip()
    if src_caption:
        caption_parts.append(src_caption)
    caption = "\n".join([p for p in caption_parts if p]).strip()
    quiz_id = uuid.uuid4().hex[:10]
    try:
        m = await context.bot.send_photo(
            chat_id=ch.channel_chat_id,
            photo=photo_file_id,
            caption=caption[:1024] if caption else None,
            reply_markup=emoji_quiz_keyboard(4, quiz_id),
        )
        emoji_quiz_save(
            quiz_id,
            ch.channel_chat_id,
            m.message_id,
            {
                "question": src_caption,
                "options": EMOJI_BUTTONS[:4],
                "correct_answer": corr,
                "explanation": explanation,
                "prefix": prefix,
                "image_file_id": photo_file_id,
                "image_mode": 1,
            },
            admin_id,
        )
        await ok_html(update, "Image Reaction Quiz Posted", f"Channel: <code>{h(getattr(ch, 'title', cid))}</code>")
    except Exception as e:
        db_log("ERROR", "imgreact_failed", {"admin_id": admin_id, "channel": getattr(ch, 'channel_chat_id', 0), "error": str(e)})
        await err(update, "Post Failed", str(e)[:180])


_old_build_app = build_app

def build_app() -> Application:
    app = _old_build_app()
    app.add_handler(_cmdh("emojipost", cmd_postemoji))
    app.add_handler(_cmdh("buffercount", cmd_buffercount))
    app.add_handler(_cmdh("imgreact", cmd_imgreact))
    return app




# ===========================
# FINAL PATCHES V4 (2026-03-13)
# ===========================

def _final_user_command_set() -> set[str]:
    return {"/start", "/help", "/commands", "/ask", "/solve_on", "/solve_off"}


def _all_commands_for(user_id: int):
    role = get_role(user_id)
    sections = []
    user_cmds = [
        ("/start", "Welcome / membership check"),
        ("/help", "Show detailed command guide"),
        ("/commands", "Show all available commands"),
        ("/ask", "Contact support (text or reply to file/photo)"),
        ("/solve_on", "Enable user AI solving"),
        ("/solve_off", "Disable user AI solving"),
    ]
    sections.append(("👤 User Commands", user_cmds))
    if role in (ROLE_ADMIN, ROLE_OWNER):
        admin_cmds = [
            ("/himusai_on", "Enable admin/owner inbox AI mode"),
            ("/himusai_off", "Disable admin/owner inbox AI mode"),
            ("/probaho_on", "Enable AI in current group (group admin)"),
            ("/probaho_off", "Disable AI in current group (group admin)"),
            ("/filter", "Add parsing filter phrase"),
            ("/clear", "Clear your buffer"),
            ("/done", "Export your buffered quizzes"),
            ("/buffercount", "Show total buffered quizzes"),
            ("/addchannel", "Add a channel/group for posting"),
            ("/listchannels", "List your channels/groups"),
            ("/removechannel", "Remove a channel/group"),
            ("/setprefix", "Set or clear channel prefix"),
            ("/setexplink", "Set or clear explanation link"),
            ("/post", "Post buffered quizzes to a channel"),
            ("/postemoji", "Post buffered emoji quizzes to a channel"),
            ("/emojipost", "Alias of /postemoji"),
            ("/imgreact", "Post image-based reaction quiz by replying to a photo"),
            ("/broadcast", "Broadcast a message"),
            ("/adminpanel", "View posting/admin stats"),
            ("/reply", "Reply to a support ticket"),
            ("/close", "Close a support ticket"),
            ("/ban", "Ban a user"),
            ("/unban", "Unban a user"),
            ("/banned", "View banned users"),
            ("/private_send", "Send a private message to a user"),
            ("/usersd", "Show user details / open profile if public"),
            ("/vision_on", "Enable image extraction mode"),
            ("/vision_off", "Disable image extraction mode"),
            ("/scanhelp", "Show image extraction help"),
            ("/explain_on", "Enable explanation in quiz + export"),
            ("/explain_off", "Disable explanation in quiz + export"),
        ]
        sections.append(("🛠 Staff Commands", admin_cmds))
    if role == ROLE_OWNER:
        owner_cmds = [
            ("/addadmin", "Promote a user to admin"),
            ("/removeadmin", "Remove admin role"),
            ("/grantall", "Grant admin all-channel access"),
            ("/revokeall", "Revoke all-channel access"),
            ("/grantvision", "Grant image extraction access"),
            ("/revokevision", "Revoke image extraction access"),
            ("/addrequired", "Add required channel/group"),
            ("/delrequired", "Remove required channel/group"),
            ("/listrequired", "List required channels/groups"),
            ("/ownerstats", "Owner dashboard"),
            ("/users", "Export started users JSON"),
            ("/quizprefix", "Set generated quiz prefix"),
            ("/quizlink", "Set generated quiz link"),
        ]
        sections.append(("👑 Owner Commands", owner_cmds))
    return sections


@require_admin
async def cmd_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else 0
    items = buffer_list(uid, limit=99999)
    if not items:
        await warn(update, "Buffer Empty", "No questions to export. Use /add or send quizzes first.")
        return
    rows = [payload for (_id, payload) in items]
    norm_rows = []
    explanations_enabled = explain_mode_on(uid)
    for r in rows:
        q = str(r.get("questions", "") or "")
        e = str(r.get("explanation", "") or "")
        q2, expl2 = split_inline_explain(q)
        if expl2 and not e.strip():
            e = expl2
        rr = dict(r)
        rr["questions"] = q2.strip()
        rr["explanation"] = e.strip() if explanations_enabled else ""
        norm_rows.append(rr)
    rows = norm_rows
    df = pd.DataFrame(rows)
    cols = ["questions", "option1", "option2", "option3", "option4", "option5", "answer", "explanation", "type", "section"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    with tempfile.NamedTemporaryFile("w+b", suffix=".csv", delete=False) as f:
        path = f.name
    df.to_csv(path, index=False, encoding="utf-8-sig")

    def _ans_to_letter(n: int) -> str:
        return {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}.get(int(n or 0), "")

    quiz_json = []
    for idx, r in enumerate(rows, start=1):
        opts_map = {"A": r.get("option1", ""), "B": r.get("option2", ""), "C": r.get("option3", ""), "D": r.get("option4", "")}
        if str(r.get("option5", "")).strip():
            opts_map["E"] = r.get("option5", "")
        quiz_json.append({
            "serial": idx,
            "question": r.get("questions", ""),
            "options": opts_map,
            "correct_answer": _ans_to_letter(r.get("answer", 0)),
            "explanation": r.get("explanation", "") if explanations_enabled else "",
        })
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as jf:
        json_path = jf.name
        json.dump(quiz_json, jf, ensure_ascii=False, indent=2)
    try:
        await update.message.reply_document(document=open(path, "rb"), caption=f"<b>✅ CSV Export</b>\n<i>{len(df)} questions exported</i>", parse_mode=ParseMode.HTML)
        await update.message.reply_document(document=open(json_path, "rb"), caption="<b>✅ JSON Export</b>", parse_mode=ParseMode.HTML)
        await ok_html(update, "Export Complete", f"CSV + JSON ready. <code>{h(len(df))}</code> questions exported.")
    finally:
        with contextlib.suppress(Exception): os.remove(path)
        with contextlib.suppress(Exception): os.remove(json_path)
    buffer_clear(uid)


@require_owner
async def cmd_addadmin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not str(context.args[0]).lstrip('-').isdigit():
        await safe_reply(update, usage_box("addadmin", "<user_id>", "Promote a user to admin"))
        return
    target = int(context.args[0])
    if target == OWNER_ID:
        await warn(update, "Not Needed", "Owner already has full access.")
        return
    conn = db_connect(); cur = conn.cursor()
    cur.execute("UPDATE users SET role=? WHERE user_id=?", (ROLE_ADMIN, target))
    if cur.rowcount == 0:
        cur.execute(
            "INSERT OR REPLACE INTO users(user_id, role, first_name, username, is_banned, created_at, can_view_all, can_use_vision, last_seen_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (target, ROLE_ADMIN, "", None, 0, now_iso(), 0, 0, now_iso()),
        )
    conn.commit(); conn.close()
    await ok(update, "Admin Added", f"User <code>{h(target)}</code> promoted to ADMIN.")


_prev_build_app_v4 = build_app

def build_app() -> Application:
    app = _prev_build_app_v4()
    return app



# ===== FINAL OVERRIDES v5 =====

def _normalize_emoji_quiz_parts(payload: Dict[str, Any]) -> Tuple[str, List[str], int, str]:
    q, opts, correct_option_id, explanation = quiz_to_poll_parts(payload)
    opts = [str(o).strip() for o in (opts or []) if str(o).strip()]
    if len(opts) > len(EMOJI_BUTTONS):
        opts = opts[:len(EMOJI_BUTTONS)]
        if correct_option_id >= len(opts):
            correct_option_id = -1
    if len(opts) < 2:
        return q, [], -1, explanation
    return q, opts, correct_option_id, explanation


@require_admin
async def cmd_setexplink(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args or not str(context.args[0]).isdigit():
        await safe_reply(update, usage_box("setexplink", "<DB-ID> [text]", "Set or clear the explanation tail text for a channel"))
        return
    cid = int(context.args[0])
    new_link = " ".join(context.args[1:]).strip() if len(context.args) > 1 else ""
    ch = channel_get_by_id_for_user(uid, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or you don't have access.")
        return
    old_link = getattr(ch, "expl_link", "") or "(empty)"
    ok2 = channel_set_expl_link(cid, new_link)
    if ok2:
        shown = new_link if new_link else "(empty)"
        body = (
            f"Channel: {h(getattr(ch, 'title', cid))}\n"
            f"DB-ID: {h(cid)}\n"
            f"Old Text: {h(old_link)}\n"
            f"New Text: {h(shown)}"
        )
        await ok(update, "Explanation Text Updated", body)
    else:
        await err(update, "Update Failed", "Could not update the explanation text.")


@require_admin
async def cmd_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("post", "<DB-ID> [keep]", "Post buffered quizzes to a channel. Use 'keep' to keep buffer."))
        return

    cid = int(context.args[0])
    keep = (len(context.args) > 1 and context.args[1].strip().lower() == "keep")
    ch = channel_get_by_id_for_user(admin_id, cid)
    if not ch:
        await warn_html(update, "Channel Not Found", f"No access to that channel. Use <code>/listchannels</code> to view yours.")
        return

    items = buffer_list(admin_id, limit=MAX_BUFFERED_QUESTIONS)
    if not items:
        await warn(update, "Buffer Empty", "No quizzes to post. Send text or forward polls first.")
        return

    await info_html(update, "Posting to Channel", f"<code>{h(ch.title)}</code> — <code>{h(str(ch.channel_chat_id))}</code>\n\nPosting <code>{h(len(items))}</code> question(s)...")

    posted_ids: List[int] = []
    ok_count, fail_count = 0, 0
    first_post_message_id = None

    for (row_id, payload) in items:
        try:
            q, opts, correct_option_id, expl = quiz_to_poll_parts(payload)
            if len(opts) < 2:
                continue
            prefix = (ch.prefix or "").strip(" ")
            expl_tail = (ch.expl_link or "").strip()
            SEP = "\n\u200b"
            q_final = f"{prefix}{SEP}{q}".strip() if prefix else q
            if len(q_final) > 300:
                q_final = q_final[:297] + "..."

            expl_final = expl.strip()
            if not explain_mode_on(admin_id):
                expl_final = ""
            if expl_tail:
                expl_final = (expl_final + "\n\n" if expl_final else "") + expl_tail
            expl_final = expl_final.strip()
            if len(expl_final) > 200:
                expl_final = expl_final[:197] + "..."

            if correct_option_id >= 0:
                m = await context.bot.send_poll(
                    chat_id=ch.channel_chat_id,
                    question=q_final,
                    options=opts,
                    is_anonymous=True,
                    type=Poll.QUIZ,
                    correct_option_id=correct_option_id,
                    explanation=expl_final if expl_final else None,
                )
            else:
                m = await context.bot.send_poll(
                    chat_id=ch.channel_chat_id,
                    question=q_final,
                    options=opts,
                    is_anonymous=True,
                    type=Poll.REGULAR,
                )
                if expl_final:
                    await context.bot.send_message(chat_id=ch.channel_chat_id, text=expl_final, disable_web_page_preview=True)

            if first_post_message_id is None and getattr(m, 'message_id', None):
                first_post_message_id = m.message_id
            ok_count += 1
            posted_ids.append(row_id)
            await asyncio.sleep(POST_DELAY_SECONDS)
        except RetryAfter as e:
            await asyncio.sleep(float(e.retry_after) + 0.5)
            fail_count += 1
        except TelegramError as e:
            fail_count += 1
            db_log("ERROR", "post_failed", {"admin_id": admin_id, "channel": ch.channel_chat_id, "error": str(e)})
        except Exception as e:
            fail_count += 1
            db_log("ERROR", "post_failed_unknown", {"admin_id": admin_id, "error": str(e)})

    if ok_count > 0 and first_post_message_id:
        with contextlib.suppress(Exception):
            await context.bot.send_message(
                chat_id=ch.channel_chat_id,
                text=_score_reply_text(ok_count),
                reply_to_message_id=first_post_message_id,
                allow_sending_without_reply=True,
            )

    inc_admin_post(admin_id, ok_count)
    if posted_ids and not keep:
        buffer_remove_ids(admin_id, posted_ids)
    body = f"Posted: {ok_count}\nFailed: {fail_count}\nRemaining in Buffer: {buffer_count(admin_id)}"
    await ok(update, "Posting Complete", body)


@require_admin
async def cmd_postemoji(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, usage_box("postemoji", "<DB-ID> [keep]", "Post buffered questions as emoji quiz to a channel"))
        return
    cid = int(context.args[0])
    keep = (len(context.args) > 1 and context.args[1].strip().lower() == "keep")
    ch = channel_get_by_id_for_user(admin_id, cid)
    if not ch:
        await warn(update, "Not Found", "Channel not found or no access.")
        return
    items = buffer_list(admin_id, limit=MAX_BUFFERED_QUESTIONS)
    if not items:
        await warn(update, "Buffer Empty", "No buffered questions found.")
        return
    prefix = str(getattr(ch, "prefix", "") or "").strip()
    title = prefix if prefix else BOT_BRAND
    sent = 0
    sent_ids: List[int] = []
    first_post_message_id = None
    fail_count = 0
    for bid, payload in items:
        qtext, opts, corr_idx0, explanation = _normalize_emoji_quiz_parts(payload)
        if len(opts) < 2:
            fail_count += 1
            continue
        msg_text = _emoji_quiz_text(qtext, opts, title)
        quiz_id = uuid.uuid4().hex[:10]
        try:
            m = await context.bot.send_message(
                chat_id=ch.channel_chat_id,
                text=msg_text,
                reply_markup=emoji_quiz_keyboard(len(opts), quiz_id),
                disable_web_page_preview=True,
            )
            if first_post_message_id is None:
                first_post_message_id = m.message_id
            sent += 1
            sent_ids.append(bid)
            emoji_quiz_save(
                quiz_id,
                ch.channel_chat_id,
                m.message_id,
                {
                    "question": qtext,
                    "options": opts,
                    "correct_answer": corr_idx0 + 1 if corr_idx0 >= 0 else 0,
                    "explanation": explanation,
                    "prefix": title,
                },
                admin_id,
            )
            await asyncio.sleep(0.30)
        except RetryAfter as e:
            await asyncio.sleep(float(getattr(e, 'retry_after', 1.0)) + 0.5)
            try:
                m = await context.bot.send_message(
                    chat_id=ch.channel_chat_id,
                    text=msg_text,
                    reply_markup=emoji_quiz_keyboard(len(opts), quiz_id),
                    disable_web_page_preview=True,
                )
                if first_post_message_id is None:
                    first_post_message_id = m.message_id
                sent += 1
                sent_ids.append(bid)
                emoji_quiz_save(
                    quiz_id,
                    ch.channel_chat_id,
                    m.message_id,
                    {
                        "question": qtext,
                        "options": opts,
                        "correct_answer": corr_idx0 + 1 if corr_idx0 >= 0 else 0,
                        "explanation": explanation,
                        "prefix": title,
                    },
                    admin_id,
                )
            except Exception as e2:
                fail_count += 1
                db_log("ERROR", "postemoji_failed_retry", {"admin_id": admin_id, "channel": getattr(ch, 'channel_chat_id', 0), "error": str(e2), "buffer_id": bid})
        except Exception as e:
            fail_count += 1
            db_log("ERROR", "postemoji_failed", {"admin_id": admin_id, "channel": getattr(ch, 'channel_chat_id', 0), "error": str(e), "buffer_id": bid})

    if sent > 0 and first_post_message_id:
        with contextlib.suppress(Exception):
            await context.bot.send_message(
                chat_id=ch.channel_chat_id,
                text=_score_reply_text(sent),
                reply_to_message_id=first_post_message_id,
                allow_sending_without_reply=True,
            )
    if sent_ids and not keep:
        buffer_remove_ids(admin_id, sent_ids)
    await ok_html(update, "Emoji Quiz Posted", f"Sent: <code>{h(sent)}</code>\nFailed: <code>{h(fail_count)}</code>\nChannel: <code>{h(getattr(ch, 'title', cid))}</code>")

# ===== END FINAL OVERRIDES v5 =====



# ===== ULTRA GROUP/MAINTENANCE PATCH v6 =====
from telegram.ext import ApplicationHandlerStop

# Registry refresh
try:
    COMMANDS_REGISTRY.setdefault("public", {}).setdefault("commands", {}).update({
        "start": "Welcome / membership check (private only)",
        "help": "Show command guide (private only)",
        "commands": "Show available commands (private only)",
        "ask": "Contact support from inbox",
        "solve_on": "Enable private user AI solving",
        "solve_off": "Disable private user AI solving",
    })
    COMMANDS_REGISTRY.setdefault("admin", {}).setdefault("commands", {}).update({
        "buffercount": "Show total buffered quizzes",
        "postemoji": "Post buffered emoji quizzes to channel",
        "emojipost": "Alias of /postemoji",
        "imgreact": "Post image reaction quiz (reply to image)",
        "usersd": "Show stored user details",
        "probaho_on": "Enable /sh AI in current group",
        "probaho_off": "Disable /sh AI in current group",
        "porag": "Delete a replied message range in group",
        "tutorial": "Show group usage tutorial",
    })
    COMMANDS_REGISTRY.setdefault("owner", {}).setdefault("commands", {}).update({
        "maintenance_on": "Enable maintenance mode and notify users",
        "maintenance_off": "Disable maintenance mode and notify users",
    })
    if "workflow" in COMMANDS_REGISTRY:
        COMMANDS_REGISTRY["workflow"]["items"] = [
            "Private inbox only -> parse text / polls / images into buffer",
            "/done -> Export CSV and clear buffer",
            "/post <DB-ID> -> Publish normal quizzes to channel",
            "/postemoji <DB-ID> -> Publish emoji quizzes to channel",
            "Group mode: /probaho_on then members use /sh to ask AI",
        ]
except Exception:
    pass


def maintenance_mode_on() -> bool:
    return get_setting("maintenance_mode", "0") == "1"


def maintenance_message() -> str:
    return get_setting("maintenance_message", "Bot is under maintenance. Please try again later.")


def set_maintenance_mode(value: bool, message: str = "") -> None:
    set_setting("maintenance_mode", "1" if value else "0")
    if message is not None:
        set_setting("maintenance_message", message or "Bot is under maintenance. Please try again later.")


async def _dm_text(context: ContextTypes.DEFAULT_TYPE, user_id: int, text: str, reply_markup=None) -> bool:
    try:
        await context.bot.send_message(
            chat_id=int(user_id),
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=reply_markup,
        )
        return True
    except Exception:
        return False


async def _broadcast_private(context: ContextTypes.DEFAULT_TYPE, text: str) -> int:
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT user_id FROM users")
    ids = [int(r[0]) for r in cur.fetchall()]
    conn.close()
    sent = 0
    for uid in ids:
        if await _dm_text(context, uid, text):
            sent += 1
        await asyncio.sleep(0.03)
    return sent


async def _auto_delete_after(bot, chat_id: int, message_ids: list[int], delay_seconds: int = 300) -> None:
    await asyncio.sleep(delay_seconds)
    for mid in message_ids:
        with contextlib.suppress(Exception):
            await bot.delete_message(chat_id=chat_id, message_id=mid)


async def _is_group_admin_user(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
    if is_owner(user_id) or is_admin(user_id):
        return True
    try:
        cm = await context.bot.get_chat_member(chat_id, user_id)
        st = str(getattr(cm, "status", ""))
        return st in ("administrator", "creator")
    except Exception:
        return False


def _extract_command_name(text: str) -> str:
    t = (text or "").strip().split()[0] if (text or "").strip() else ""
    t = t.split("@")[0]
    return t.lstrip("/").lower()


async def global_maintenance_guard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not maintenance_mode_on():
        return
    uid = update.effective_user.id if update.effective_user else 0
    if not uid or is_owner(uid):
        return
    msg = ui_box_html("Maintenance Mode", h(maintenance_message()), emoji="🛠")
    if update.effective_chat and update.effective_chat.type == "private":
        with contextlib.suppress(Exception):
            await safe_reply(update, msg)
    else:
        await _dm_text(context, uid, msg)
    raise ApplicationHandlerStop


async def group_command_guard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat or update.effective_chat.type not in ("group", "supergroup"):
        return
    cmd = _extract_command_name(update.message.text or "")
    allowed = {"probaho_on", "probaho_off", "sh", "porag", "tutorial"}
    if cmd and cmd not in allowed:
        raise ApplicationHandlerStop


@require_owner
async def cmd_maintenance_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = " ".join(context.args).strip() or "Bot maintenance চলছে। কিছুক্ষণ পরে আবার চেষ্টা করুন।"
    set_maintenance_mode(True, msg)
    sent = await _broadcast_private(context, ui_box_html("Maintenance Mode", h(msg), emoji="🛠"))
    await ok(update, "Maintenance Enabled", f"Message sent to: {sent}")


@require_owner
async def cmd_maintenance_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_maintenance_mode(False, "")
    sent = await _broadcast_private(context, ui_box_html("Service Resumed", "Bot is now active again.", emoji="✅"))
    await ok(update, "Maintenance Disabled", f"Resume message sent to: {sent}")


@require_admin
async def cmd_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    items = buffer_list(uid, limit=99999)
    if not items:
        await warn(update, "Buffer Empty", "No questions to export. Use /add or send quizzes first.")
        return
    rows = []
    for _id, payload in items:
        q = str(payload.get("questions", "") or "")
        e = str(payload.get("explanation", "") or "")
        q2, expl2 = split_inline_explain(q)
        if expl2 and not e.strip():
            e = expl2
        rr = dict(payload)
        rr["questions"] = q2.strip()
        rr["explanation"] = (e.strip() if explain_mode_on(uid) else "")
        rows.append(rr)
    df = pd.DataFrame(rows)
    cols = ["questions", "option1", "option2", "option3", "option4", "option5", "answer", "explanation", "type", "section"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    with tempfile.NamedTemporaryFile("w+b", suffix=".csv", delete=False) as tf:
        csv_path = tf.name
    try:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        with open(csv_path, "rb") as rf:
            await context.bot.send_document(
                chat_id=uid,
                document=rf,
                filename="probaho_export.csv",
                caption=f"Exported {len(df)} question(s)",
            )
        buffer_clear(uid)
        await ok(update, "Export Complete", f"CSV exported successfully.\n\nExported: {len(df)}\nBuffer cleared.")
    finally:
        with contextlib.suppress(Exception):
            os.unlink(csv_path)


async def cmd_probaho_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    uid = update.effective_user.id if update.effective_user else 0
    if not chat or chat.type not in ("group", "supergroup"):
        if update.message:
            await warn(update, "Group Only", "Use this command inside a group/supergroup.")
        return
    if not await _is_group_admin_user(context, chat.id, uid):
        await _dm_text(context, uid, ui_box_html("Unauthorized", "Only a group admin or the bot owner can use this command.", emoji="⚠️"))
        with contextlib.suppress(Exception):
            await update.message.delete()
        return
    set_group_ai_enabled(chat.id, True)
    with contextlib.suppress(Exception):
        await update.message.delete()
    await _dm_text(context, uid, ui_box_html("Group AI Enabled", f"Group: <code>{h(chat.id)}</code>\nMode: members can use <code>/sh</code> in this group.", emoji="✅"))


async def cmd_probaho_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    uid = update.effective_user.id if update.effective_user else 0
    if not chat or chat.type not in ("group", "supergroup"):
        if update.message:
            await warn(update, "Group Only", "Use this command inside a group/supergroup.")
        return
    if not await _is_group_admin_user(context, chat.id, uid):
        await _dm_text(context, uid, ui_box_html("Unauthorized", "Only a group admin or the bot owner can use this command.", emoji="⚠️"))
        with contextlib.suppress(Exception):
            await update.message.delete()
        return
    set_group_ai_enabled(chat.id, False)
    with contextlib.suppress(Exception):
        await update.message.delete()
    await _dm_text(context, uid, ui_box_html("Group AI Disabled", f"Group: <code>{h(chat.id)}</code>\nThe <code>/sh</code> AI command is now off in this group.", emoji="✅"))


def _poll_text_for_sh(poll) -> tuple[str, list[str], str]:
    qtext = str(getattr(poll, 'question', '') or '').strip()
    options = [str(o.text).strip() for o in getattr(poll, 'options', [])]
    expl = str(getattr(poll, 'explanation', '') or '').strip()
    return qtext, options, expl


async def cmd_sh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat or update.effective_chat.type not in ("group", "supergroup"):
        return
    uid = update.effective_user.id if update.effective_user else 0
    chat_id = int(update.effective_chat.id)
    if not is_group_ai_enabled(chat_id):
        with contextlib.suppress(Exception):
            await update.message.delete()
        return
    if is_banned(uid):
        with contextlib.suppress(Exception):
            await update.message.delete()
        return
    ok, missing = await user_meets_required_memberships(context, uid)
    if not ok and not (is_owner(uid) or is_admin(uid) or await _is_group_admin_user(context, chat_id, uid)):
        names = ", ".join(missing[:10]) if missing else "required channel/group"
        await _dm_text(context, uid, ui_box_html("Join Required", f"Please join: {h(names)}", emoji="⚠️"), reply_markup=_required_join_kb())
        with contextlib.suppress(Exception):
            await update.message.delete()
        return

    reply = update.message.reply_to_message
    inline = " ".join(context.args).strip()

    token = _make_token()
    store = _pending_store(context)
    preview = inline or "Solve this message/quiz"

    if reply and getattr(reply, 'poll', None):
        qtext, options, qexpl = _poll_text_for_sh(reply.poll)
        official_ans = 0
        with contextlib.suppress(Exception):
            if getattr(reply.poll, 'type', '') == 'quiz' and getattr(reply.poll, 'correct_option_id', None) is not None:
                official_ans = int(reply.poll.correct_option_id) + 1
        store[token] = {
            "uid": uid,
            "chat_id": chat_id,
            "kind": "poll",
            "payload": {
                "question": qtext,
                "options": options,
                "official_ans": official_ans,
                "official_expl": qexpl,
            },
        }
        preview = qtext or preview
    else:
        prompt = inline
        if reply:
            base = (reply.text or reply.caption or "").strip()
            if inline and base:
                prompt = f"Context:\n{base}\n\nQuestion:\n{inline}"
            elif base:
                prompt = base
        if not (prompt or "").strip():
            await _dm_text(context, uid, ui_box_html("Usage", "Use <code>/sh your question</code> or reply to a message/quiz with <code>/sh</code>.", emoji="ℹ️"))
            with contextlib.suppress(Exception):
                await update.message.delete()
            return
        store[token] = {
            "uid": uid,
            "chat_id": chat_id,
            "kind": "text",
            "payload": {"text": prompt},
        }
        preview = prompt

    kb = _solver_picker_kb(token)
    sent = await context.bot.send_message(
        chat_id=chat_id,
        text=ui_box_html("Which AI model?", f"<code>{h(preview[:100])}</code>", emoji="🧠"),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
        reply_markup=kb,
        reply_to_message_id=(reply.message_id if reply else update.message.message_id),
        allow_sending_without_reply=True,
    )
    asyncio.create_task(_auto_delete_after(context.bot, chat_id, [sent.message_id], 300))


async def cmd_porag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat or update.effective_chat.type not in ("group", "supergroup"):
        return
    uid = update.effective_user.id if update.effective_user else 0
    if not await _is_group_admin_user(context, update.effective_chat.id, uid):
        await _dm_text(context, uid, ui_box_html("Unauthorized", "Only a group admin or the bot owner can use /porag.", emoji="⚠️"))
        with contextlib.suppress(Exception):
            await update.message.delete()
        return
    if not update.message.reply_to_message:
        await _dm_text(context, uid, ui_box_html("Usage", "Reply to the first message you want to delete, then send <code>/porag</code>.", emoji="ℹ️"))
        with contextlib.suppress(Exception):
            await update.message.delete()
        return
    start_id = int(update.message.reply_to_message.message_id)
    end_id = int(update.message.message_id)
    total = end_id - start_id + 1
    if total > 150:
        await _dm_text(context, uid, ui_box_html("Too Many Messages", "Please delete at most 150 messages at a time.", emoji="⚠️"))
        with contextlib.suppress(Exception):
            await update.message.delete()
        return
    deleted = 0
    for mid in range(start_id, end_id + 1):
        try:
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=mid)
            deleted += 1
        except Exception:
            pass
    await _dm_text(context, uid, ui_box_html("Messages Deleted", f"Deleted: <code>{deleted}</code>", emoji="🧹"))


async def cmd_tutorial(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat or update.effective_chat.type not in ("group", "supergroup"):
        return
    uid = update.effective_user.id if update.effective_user else 0
    if not await _is_group_admin_user(context, update.effective_chat.id, uid):
        with contextlib.suppress(Exception):
            await update.message.delete()
        return
    text = (
        "Group rules:\n"
        "1) Use /probaho_on to enable group AI.\n"
        "2) Members ask only with /sh.\n"
        "3) AI replies auto-delete after 5 minutes.\n"
        "4) Other bot commands work only in inbox.\n"
        "5) Reply to a start message with /porag to delete a range."
    )
    with contextlib.suppress(Exception):
        await update.message.reply_text(text)
    with contextlib.suppress(Exception):
        await update.message.delete()


async def on_tutorial_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.callback_query:
        return
    q = update.callback_query
    uid = q.from_user.id if q.from_user else 0
    chat_id = q.message.chat_id if q.message else 0
    if not await _is_group_admin_user(context, chat_id, uid):
        with contextlib.suppress(Exception):
            await q.answer("Admins only.", show_alert=True)
        return
    text = (
        "Use /probaho_on in this group.\n"
        "Members can then ask AI using /sh text অথবা reply + /sh.\n"
        "All other bot tools stay in inbox/private.\n"
        "Use /probaho_off to stop group AI."
    )
    with contextlib.suppress(Exception):
        await q.answer(text, show_alert=True)


async def on_my_chat_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cmu = getattr(update, 'my_chat_member', None)
    if not cmu:
        return
    try:
        old_status = cmu.old_chat_member.status
        new_status = cmu.new_chat_member.status
        chat = cmu.chat
        actor = cmu.from_user
    except Exception:
        return
    if new_status in ("member", "administrator") and old_status in ("left", "kicked") and chat.type in ("group", "supergroup"):
        actor_name = actor.first_name if actor else "Admin"
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("📘 Tutorial", callback_data="tutorial:show")]])
        with contextlib.suppress(Exception):
            await context.bot.send_message(
                chat_id=chat.id,
                text=f"ধন্যবাদ {h(actor_name)}, {h(BOT_BRAND)} বটটি group-এ add করার জন্য। Admin guide দেখতে নিচের button ব্যবহার করুন.",
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )


@require_admin
async def cmd_buffercount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else 0
    await info_html(update, "Buffer Status", f"Total buffered: <code>{buffer_count(uid)}</code>", emoji="ℹ️")


def _private_filter(base_filter):
    return filters.ChatType.PRIVATE & base_filter


def _group_filter(base_filter):
    return (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP) & base_filter


def build_app() -> Application:
    db_init()
    with contextlib.suppress(Exception):
        extra_db_init()
    from telegram.ext import ChatMemberHandler
    builder = ApplicationBuilder().token(BOT_TOKEN)
    try:
        builder = builder.concurrent_updates(64)
    except Exception:
        pass
    app = builder.build()

    # Global guards
    app.add_handler(MessageHandler(filters.ALL, global_maintenance_guard), group=-100)
    app.add_handler(MessageHandler(_group_filter(filters.COMMAND), group_command_guard), group=-90)

    # Callbacks
    app.add_handler(CallbackQueryHandler(on_solver_callback, pattern=r"^solve:"))
    app.add_handler(CallbackQueryHandler(on_genquiz_callback, pattern=r"^genquiz:"))
    app.add_handler(CallbackQueryHandler(on_required_verify_callback, pattern=r"^req:verify$"))
    app.add_handler(CallbackQueryHandler(on_emoji_quiz_callback, pattern=r"^eq:"))
    app.add_handler(CallbackQueryHandler(on_image_react_callback, pattern=r"^imgreact:"))
    app.add_handler(CallbackQueryHandler(on_tutorial_callback, pattern=r"^tutorial:show$"))

    # Private public commands
    app.add_handler(_cmdh("start", cmd_start, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("help", cmd_help, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("commands", cmd_commands, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("features", cmd_features, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("ask", cmd_ask, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("scanhelp", cmd_scanhelp, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("vision_on", cmd_vision_on, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("vision_off", cmd_vision_off, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("solve_on", cmd_solve_on, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("solve_off", cmd_solve_off, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("explain_on", cmd_explain_on, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("explain_off", cmd_explain_off, filters=filters.ChatType.PRIVATE))

    # Owner/private
    app.add_handler(_cmdh("quizprefix", cmd_quizprefix, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("quizlink", cmd_quizlink, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("addadmin", cmd_addadmin, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("removeadmin", cmd_removeadmin, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("grantall", cmd_grantall, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("revokeall", cmd_revokeall, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("grantvision", cmd_grantvision, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("revokevision", cmd_revokevision, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("ownerstats", cmd_ownerstats, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("users", cmd_users, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("maintenance_on", cmd_maintenance_on, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("maintenance_off", cmd_maintenance_off, filters=filters.ChatType.PRIVATE))

    # Admin/private
    app.add_handler(_cmdh("filter", cmd_filter, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("done", cmd_done, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("clear", cmd_clear, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("buffercount", cmd_buffercount, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("addchannel", cmd_addchannel, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("listchannels", cmd_listchannels, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("removechannel", cmd_removechannel, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("setprefix", cmd_setprefix, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("setexplink", cmd_setexplink, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("post", cmd_post, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("postemoji", cmd_postemoji, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("emojipost", cmd_postemoji, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("imgreact", cmd_imgreact, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("broadcast", cmd_broadcast, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("adminpanel", cmd_adminpanel, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("reply", cmd_reply, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("close", cmd_close, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("ban", cmd_ban, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("unban", cmd_unban, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("banned", cmd_banned, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("private_send", cmd_private_send, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("send_private", cmd_private_send, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("usersd", cmd_usersd, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("addrequired", cmd_addrequired, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("delrequired", cmd_delrequired, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("listrequired", cmd_listrequired, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("himusai_on", cmd_himusai_on, filters=filters.ChatType.PRIVATE))
    app.add_handler(_cmdh("himusai_off", cmd_himusai_off, filters=filters.ChatType.PRIVATE))

    # Private message handlers only
    app.add_handler(MessageHandler(_private_filter(filters.POLL), handle_poll))
    app.add_handler(MessageHandler(_private_filter(filters.POLL), handle_user_poll_solver), group=1)
    app.add_handler(MessageHandler(_private_filter(filters.PHOTO), handle_image))
    app.add_handler(MessageHandler(_private_filter(filters.Document.IMAGE), handle_image))
    app.add_handler(MessageHandler(_private_filter(filters.TEXT & (~filters.COMMAND)), handle_text))
    app.add_handler(MessageHandler(_private_filter(filters.TEXT & (~filters.COMMAND)), handle_user_text_unusual), group=1)

    # Group-only handlers
    app.add_handler(_cmdh("probaho_on", cmd_probaho_on, filters=(filters.ChatType.GROUP | filters.ChatType.SUPERGROUP)))
    app.add_handler(_cmdh("probaho_off", cmd_probaho_off, filters=(filters.ChatType.GROUP | filters.ChatType.SUPERGROUP)))
    app.add_handler(_cmdh("sh", cmd_sh, filters=(filters.ChatType.GROUP | filters.ChatType.SUPERGROUP)))
    app.add_handler(_cmdh("porag", cmd_porag, filters=(filters.ChatType.GROUP | filters.ChatType.SUPERGROUP)))
    app.add_handler(_cmdh("tutorial", cmd_tutorial, filters=(filters.ChatType.GROUP | filters.ChatType.SUPERGROUP)))
    app.add_handler(ChatMemberHandler(on_my_chat_member, chat_member_types=ChatMemberHandler.MY_CHAT_MEMBER))

    app.add_error_handler(on_error)
    return app

# ===== END ULTRA GROUP/MAINTENANCE PATCH v6 =====

if __name__ == "__main__":
    main()
