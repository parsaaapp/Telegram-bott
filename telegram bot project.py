import logging
import re
import os
import tempfile
import sqlite3
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sympy import symbols, sympify, diff, integrate, lambdify, S
from sympy.core.sympify import SympifyError

from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# =========================
# Configuration and logging
# =========================

BOT_TOKEN = "8506020763:AAGDCMt8orIE3Cn4ka3kwIPj07HuVo-py_E"
OWNER_USERNAME = "parsa7cr"  # without @

x = symbols('x')

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# =========================
# SQLite database
# =========================

DB_FILE = "codes.db"

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def ensure_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS codes (
            code TEXT PRIMARY KEY,
            status TEXT NOT NULL,    -- 'used' or 'new' (optional)
            used_by INTEGER,
            used_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            active INTEGER NOT NULL DEFAULT 0,
            expires_at TEXT
        )
    """)
    conn.commit()
    conn.close()

# =========================
# Helper functions
# =========================

def is_forward_from_owner(msg) -> bool:
    """
    Return True if the given Message is forwarded from OWNER_USERNAME.
    Checks forward_from (User) and forward_from_chat (Chat) usernames.
    """
    try:
        # Forwarded from a user with visible username
        if msg.forward_from and msg.forward_from.username:
            if msg.forward_from.username.lower() == OWNER_USERNAME.lower():
                return True
        # Forwarded from a private chat or channel with username
        if msg.forward_from_chat and msg.forward_from_chat.username:
            if msg.forward_from_chat.username.lower() == OWNER_USERNAME.lower():
                return True
    except Exception as e:
        logger.warning(f"is_forward_from_owner error: {e}")
    return False

def is_active(user_id: int) -> bool:
    """
    Check if user has active=1 and expires_at in the future.
    """
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT active, expires_at FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return False
        active_flag, exp_str = row
        if not active_flag:
            return False
        if not exp_str:
            return True
        try:
            exp_dt = datetime.fromisoformat(exp_str)
            return datetime.utcnow() <= exp_dt
        except Exception:
            return True
    except Exception as e:
        logger.error(f"is_active error: {e}")
        return False

def perform_activation(user_id: int, code: str) -> str:
    """
    Activate subscription based on code length:
    - 8 digits: 30 days
    - 9 digits: 90 days
    Stores activation in SQLite (codes, users).
    Returns a user-facing message (success or error).
    """
    if not re.fullmatch(r"\d{8,9}", code):
        return "کد نامعتبر است. باید ۸ یا ۹ رقمی باشد."

    duration_days = 30 if len(code) == 8 else 90
    now_iso = datetime.utcnow().isoformat()

    try:
        conn = get_db()
        cur = conn.cursor()

        # Check if code already used
        cur.execute("SELECT status, used_by FROM codes WHERE code = ?", (code,))
        row = cur.fetchone()
        if row:
            status, used_by = row
            if status == "used":
                conn.close()
                return "این کد قبلاً استفاده شده است."
            # If status is something else, consider unused (optional), but we'll still block reuse once used below.

        # Mark code as used
        cur.execute("""
            INSERT INTO codes (code, status, used_by, used_at)
            VALUES (?, 'used', ?, ?)
            ON CONFLICT(code) DO UPDATE SET status='used', used_by=excluded.used_by, used_at=excluded.used_at
        """, (code, user_id, now_iso))

        # Update user subscription
        cur.execute("SELECT expires_at FROM users WHERE user_id = ?", (user_id,))
        urow = cur.fetchone()
        start_time = datetime.utcnow()
        if urow and urow[0]:
            try:
                current_exp = datetime.fromisoformat(urow[0])
                if current_exp > start_time:
                    start_time = current_exp  # extend from current expiry
            except Exception:
                pass
        expires_at = (start_time + timedelta(days=duration_days)).isoformat()

        cur.execute("""
            INSERT INTO users (user_id, active, expires_at)
            VALUES (?, 1, ?)
            ON CONFLICT(user_id) DO UPDATE SET active=1, expires_at=excluded.expires_at
        """, (user_id, expires_at))

        conn.commit()
        conn.close()
        return f"اشتراک شما فعال شد! مدت اعتبار {duration_days} روز."
    except Exception as e:
        logger.error(f"perform_activation error: {e}")
        return "در فعال‌سازی مشکلی رخ داد. لطفاً بعداً تلاش کنید."

def parse_function(expr_text: str):
    """
    Replace ^ with ** and parse to sympy expression using variable x.
    """
    try:
        cleaned = expr_text.replace("^", "**")
        return sympify(cleaned, locals={"x": x})
    except SympifyError as e:
        raise SympifyError(f"Cannot parse expression: {expr_text}") from e

def plot_function(expr, filename=None, xmin=-5, xmax=5, num=400, title="f(x)"):
    """
    Plot sympy expression over [xmin, xmax] and save to a temporary PNG.
    Returns file path.
    """
    f = lambdify(x, expr, "numpy")
    import numpy as np
    X = np.linspace(xmin, xmax, num)
    try:
        Y = f(X)
    except Exception:
        Y = []
        for xv in X:
            try:
                Y.append(float(f(xv)))
            except Exception:
                Y.append(float("nan"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(X, Y, label=title)
    ax.set_xlabel("x")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if filename is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        filename = tmp.name
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return filename

# =========================
# Handlers
# =========================

def start_handler(update: Update, context: CallbackContext):
    text = (
        "سلام! خوش آمدی به بات محاسبات ریاضی.\n\n"
        "امکانات رایگان:\n"
        "- محاسبه مشتق‌های اول تا سوم\n"
        "- رسم نمودار تابع\n\n"
        "امکانات اشتراکی:\n"
        "- انتگرال نامعین (Symbolic)\n"
        "- انتگرال معین عددی (با حدود)\n\n"
        "دستورات:\n"
        "/derivative f(x)\n"
        "/integral f(x) [a b]\n"
        "/activate <CODE> (با قوانین امنیتی)\n"
        "/pricing\n"
        "/union"
    )
    update.message.reply_text(text)

def pricing_handler(update: Update, context: CallbackContext):
    text = (
        "طرح‌های اشتراک:\n"
        "- کد ۸ رقمی = ۱ ماه (۳۰ روز)\n"
        "- کد ۹ رقمی = ۳ ماه (۹۰ روز)\n\n"
        "نکته امنیتی: کد باید از پی‌وی ادمین (@parsa7cr) فوروارد شده باشد یا دستور /activate به همان پیام فوروارد شده ریپلای شود."
    )
    update.message.reply_text(text)

def derivative_handler(update: Update, context: CallbackContext):
    args = context.args
    if not args:
        update.message.reply_text("لطفاً تابع را وارد کنید. مثال: /derivative x^3 + 2*x")
        return

    expr_text = " ".join(args)
    try:
        f = parse_function(expr_text)
    except SympifyError:
        update.message.reply_text("تابع نامعتبر است. از x استفاده کنید و می‌توانید از ^ برای توان استفاده کنید.")
        return

    try:
        f1 = diff(f, x, 1)
        f2 = diff(f, x, 2)
        f3 = diff(f, x, 3)
    except Exception as e:
        update.message.reply_text(f"خطا در محاسبه مشتق‌ها: {e}")
        return

    msg = (
        f"تابع:\n{f}\n\n"
        f"مشتق اول:\n{f1}\n\n"
        f"مشتق دوم:\n{f2}\n\n"
        f"مشتق سوم:\n{f3}"
    )
    update.message.reply_text(msg)

    try:
        img_path = plot_function(f, title="f(x)")
        with open(img_path, "rb") as img:
            update.message.reply_photo(photo=img, caption="نمودار f(x)")
        os.unlink(img_path)
    except Exception as e:
        update.message.reply_text(f"امکان رسم نمودار نبود: {e}")

def integral_handler(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    args = context.args
    if not args:
        update.message.reply_text("لطفاً تابع را وارد کنید. مثال: /integral x^2 + 3*x یا /integral sin(x) 0 3.14")
        return

    if not is_active(user_id):
        update.message.reply_text("این قابلیت فقط برای کاربران فعال است. برای فعال‌سازی، قوانین امنیتی را رعایت کنید.")
        return

    expr_text = args[0]
    extra = args[1:]
    try:
        f = parse_function(expr_text)
    except SympifyError:
        update.message.reply_text("تابع نامعتبر است. از x استفاده کنید.")
        return

    # Indefinite integral
    try:
        F = integrate(f, x)
        verify = diff(F, x).simplify()
        parts = [
            f"تابع:\n{f}",
            f"انتگرال نامعین:\n∫ f(x) dx = {F} + C",
            f"بررسی سریع:\n d/dx({F}) = {verify}"
        ]
    except Exception as e:
        update.message.reply_text(f"خطا در انتگرال‌گیری: {e}")
        return

    # Definite integral if bounds provided
    if len(extra) == 2:
        a_text, b_text = extra
        try:
            a = float(S(a_text))
            b = float(S(b_text))
            f_num = lambdify(x, f, "numpy")
            import numpy as np
            N = 1000
            X = np.linspace(a, b, N + 1)
            Y = f_num(X)
            Y = np.where(np.isfinite(Y), Y, 0.0)
            h = (b - a) / N
            I = h/3 * (Y[0] + Y[-1] + 4 * Y[1:-1:2].sum() + 2 * Y[2:-2:2].sum())
            parts.append(f"انتگرال معین عددی در بازه [{a}, {b}] ≈ {I}")
        except Exception as e:
            parts.append(f"محاسبه عددی انتگرال معین با خطا مواجه شد: {e}")

    update.message.reply_text("\n\n".join(parts))

    try:
        img_path = plot_function(F, title="F(x) = ∫ f(x) dx")
        with open(img_path, "rb") as img:
            update.message.reply_photo(photo=img, caption="نمودار F(x)")
        os.unlink(img_path)
    except Exception as e:
        update.message.reply_text(f"امکان رسم نمودار نبود: {e}")

def activate_handler(update: Update, context: CallbackContext):
    """
    Only activate if:
    - The command is a reply to a forwarded message from @parsa7cr, OR
    - The current message itself is a forwarded message from @parsa7cr (rare for commands).
    If user types code directly without valid forward context -> error.
    """
    ensure_db()
    user_id = update.effective_user.id
    msg = update.message

    code_arg = context.args[0].strip() if context.args else None
    reply_msg = msg.reply_to_message

    # Case A: command is reply to a forwarded message from owner
    if reply_msg and is_forward_from_owner(reply_msg):
        if not code_arg:
            update.message.reply_text("کد را وارد کنید. مثال: /activate 12345678")
            return
        res = perform_activation(user_id, code_arg)
        update.message.reply_text(res)
        return

    # Case B: command message itself is forwarded from owner (less common)
    if is_forward_from_owner(msg):
        # Try to extract code from command args or the forwarded text
        code = code_arg
        if not code and msg.text:
            m = re.search(r"\b(\d{8,9})\b", msg.text)
            code = m.group(1) if m else None
        if not code:
            update.message.reply_text("کد در پیام یافت نشد. لطفاً کد را همراه دستور ارسال کنید.")
            return
        res = perform_activation(user_id, code)
        update.message.reply_text(res)
        return

    # Otherwise: reject direct typing
    update.message.reply_text("برای فعال‌سازی، پیام حاوی کد را از پی‌وی ادمین (@parsa7cr) فوروارد کنید یا دستور /activate را به همان پیام فوروارد شده ریپلای کنید.")

def handle_message(update: Update, context: CallbackContext):
    """
    - If user sends only digits and message is NOT a valid forward -> error.
    - If message is a valid forward from owner and contains an 8/9 digit code -> activate.
    - Otherwise provide gentle guidance.
    """
    ensure_db()
    user_id = update.effective_user.id
    msg = update.message
    text = msg.text or ""

    digits_only = re.fullmatch(r"\s*\d{8,9}\s*", text) is not None

    if digits_only and not is_forward_from_owner(msg):
        update.message.reply_text("ارسال مستقیم کد مجاز نیست. لطفاً پیام را از پی‌وی ادمین (@parsa7cr) فوروارد کنید.")
        return

    if is_forward_from_owner(msg):
        m = re.search(r"\b(\d{8,9})\b", text)
        if m:
            code = m.group(1)
            res = perform_activation(user_id, code)
            update.message.reply_text(res)
            return

    guidance = (
        "اگر می‌خواهی محاسبه انجام بدهم:\n"
        "- مشتق: /derivative f(x)\n"
        "- انتگرال: /integral f(x) [a b]\n\n"
        "برای فعال‌سازی:\n"
        "پیام حاوی کد را از پی‌وی ادمین (@parsa7cr) فوروارد کن یا دستور /activate را به همان پیام ریپلای کن."
    )
    update.message.reply_text(guidance)

def unknown_handler(update: Update, context: CallbackContext):
    text = (
        "دستور ناشناخته است.\n"
        "دستورات موجود:\n"
        "/start - شروع\n"
        "/pricing - طرح‌های اشتراک\n"
        "/derivative f(x) - مشتق‌ها و نمودار\n"
        "/integral f(x) [a b] - انتگرال‌ها (کاربران فعال)\n"
        "/activate CODE - فعال‌سازی (با قوانین امنیتی)\n"
        "/union - فهرست دستورات"
    )
    update.message.reply_text(text)

def union_handler(update: Update, context: CallbackContext):
    text = (
        "فهرست دستورات:\n"
        "- /start : شروع و معرفی\n"
        "- /pricing : طرح‌های اشتراک\n"
        "- /derivative f(x) : مشتق‌های اول، دوم و سوم + نمودار\n"
        "- /integral f(x) [a b] : انتگرال نامعین و معین (کاربران فعال)\n"
        "- /activate CODE : فعال‌سازی با فوروارد از @parsa7cr یا ریپلای به آن\n"
    )
    update.message.reply_text(text)

# =========================
# Main entry
# =========================

def main():
    ensure_db()

    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start_handler))
    dp.add_handler(CommandHandler("pricing", pricing_handler))
    dp.add_handler(CommandHandler("derivative", derivative_handler))
    dp.add_handler(CommandHandler("integral", integral_handler))
    dp.add_handler(CommandHandler("activate", activate_handler))
    dp.add_handler(CommandHandler("union", union_handler))

    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), handle_message))
    dp.add_handler(MessageHandler(Filters.command, unknown_handler))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
