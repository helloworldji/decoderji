import os
import tempfile
import logging
import asyncio
import base64
import zlib
import marshal
import codecs
import binascii
import re
import zipfile
import io
from contextlib import asynccontextmanager
from typing import Optional
import traceback

from fastapi import FastAPI, Request, Response
from telegram import Update, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# --- Logging Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Environment Variables ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
PORT = int(os.environ.get("PORT", 10000))

# --- Validation ---
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN not set!")
if not WEBHOOK_URL:
    raise ValueError("❌ WEBHOOK_URL not set!")

# --- Constants ---
CREDIT = "Dev: @aadi_io"
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_ITERATIONS = 100

# --- Global Bot Application ---
telegram_app: Optional[Application] = None


# --- ACTUAL DECODING FUNCTIONS ---

def extract_from_exec(code: str) -> str:
    """Extract code from exec() calls."""
    patterns = [
        r'exec\s*KATEX_INLINE_OPEN\s*(.+?)\s*KATEX_INLINE_CLOSE\s*$',
        r'exec\s*KATEX_INLINE_OPEN\s*(.+?)\s*,',
        r'eval\s*KATEX_INLINE_OPEN\s*(.+?)\s*KATEX_INLINE_CLOSE',
        r'compile\s*KATEX_INLINE_OPEN\s*(.+?)\s*,',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, code, re.DOTALL | re.MULTILINE)
        if match:
            inner = match.group(1).strip()
            logger.info(f"Extracted from exec/eval: {len(inner)} chars")
            return inner
    
    return code


def decode_base64_strings(code: str) -> str:
    """Decode all base64 strings in code."""
    try:
        # Pattern: base64.b64decode(b'...' or '...')
        pattern = r"base64\.b64decode\s*KATEX_INLINE_OPEN\s*[b]?['\"]([A-Za-z0-9+/=]+)['\"]\s*KATEX_INLINE_CLOSE"
        
        def replacer(match):
            encoded = match.group(1)
            try:
                decoded = base64.b64decode(encoded).decode('utf-8', errors='ignore')
                logger.info(f"Decoded base64 string: {len(decoded)} chars")
                return f'"""{decoded}"""'
            except:
                return match.group(0)
        
        result = re.sub(pattern, replacer, code)
        
        # Also try to find standalone base64 strings
        pattern2 = r"[b]?['\"]([A-Za-z0-9+/=]{100,})['\"]"
        matches = re.findall(pattern2, code)
        
        for match in matches:
            try:
                decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                if len(decoded) > 20 and decoded.isprintable():
                    logger.info(f"Decoded standalone base64: {len(decoded)} chars")
                    code = code.replace(match, decoded)
            except:
                pass
        
        return result
    except Exception as e:
        logger.error(f"Base64 decode error: {e}")
        return code


def decode_marshal(code: str) -> str:
    """Decode marshal.loads() calls."""
    try:
        pattern = r"marshal\.loads\s*KATEX_INLINE_OPEN\s*(.+?)\s*KATEX_INLINE_CLOSE"
        match = re.search(pattern, code, re.DOTALL)
        
        if match:
            data_expr = match.group(1).strip()
            # Try to evaluate the data
            try:
                if data_expr.startswith('b'):
                    data_expr = data_expr[1:].strip("'\"")
                    data = eval(f"b'{data_expr}'")
                else:
                    data = eval(data_expr)
                
                decoded = marshal.loads(data)
                logger.info(f"Decoded marshal: {type(decoded)}")
                
                if hasattr(decoded, 'co_consts'):
                    # It's a code object, try to decompile
                    import dis
                    result = str(decoded.co_consts)
                    return result
                
                return str(decoded)
            except:
                pass
        
        return code
    except Exception as e:
        logger.error(f"Marshal decode error: {e}")
        return code


def decode_zlib(code: str) -> str:
    """Decode zlib compressed data."""
    try:
        pattern = r"zlib\.decompress\s*KATEX_INLINE_OPEN\s*(.+?)\s*KATEX_INLINE_CLOSE"
        match = re.search(pattern, code, re.DOTALL)
        
        if match:
            data_expr = match.group(1).strip()
            try:
                data = eval(data_expr)
                decoded = zlib.decompress(data).decode('utf-8', errors='ignore')
                logger.info(f"Decoded zlib: {len(decoded)} chars")
                return decoded
            except:
                pass
        
        return code
    except Exception as e:
        logger.error(f"Zlib decode error: {e}")
        return code


def decode_hex_strings(code: str) -> str:
    """Decode hex-encoded strings."""
    try:
        # Pattern: bytes.fromhex('...')
        pattern = r"bytes\.fromhex\s*KATEX_INLINE_OPEN\s*['\"]([0-9a-fA-F]+)['\"]\s*KATEX_INLINE_CLOSE"
        
        def replacer(match):
            hex_str = match.group(1)
            try:
                decoded = bytes.fromhex(hex_str).decode('utf-8', errors='ignore')
                return f'"{decoded}"'
            except:
                return match.group(0)
        
        return re.sub(pattern, replacer, code)
    except Exception as e:
        logger.error(f"Hex decode error: {e}")
        return code


def extract_from_zip(code: str) -> str:
    """Extract and decode zip files embedded in code."""
    try:
        # Find base64 encoded data that might be a zip
        pattern = r"['\"]([A-Za-z0-9+/=]{500,})['\"]"
        matches = re.findall(pattern, code)
        
        for match in matches:
            try:
                decoded_data = base64.b64decode(match)
                
                # Check if it's a zip file
                if decoded_data[:2] == b'PK' or decoded_data[:4] == b'PK\x03\x04':
                    logger.info("Found ZIP file in code!")
                    
                    # Extract zip
                    with zipfile.ZipFile(io.BytesIO(decoded_data)) as zf:
                        # Get first file
                        names = zf.namelist()
                        if names:
                            first_file = names[0]
                            content = zf.read(first_file).decode('utf-8', errors='ignore')
                            logger.info(f"Extracted from ZIP: {first_file} ({len(content)} chars)")
                            return content
            except:
                continue
        
        return code
    except Exception as e:
        logger.error(f"ZIP extraction error: {e}")
        return code


def is_still_obfuscated(code: str) -> bool:
    """Check if code still has obfuscation."""
    if not code or len(code) < 10:
        return True
    
    indicators = [
        'exec(',
        'eval(',
        'compile(',
        'base64.b64decode',
        'base64.b32decode',
        'marshal.loads',
        'zlib.decompress',
        '__import__',
        'bytes.fromhex',
    ]
    
    code_lower = code.lower()
    for indicator in indicators:
        if indicator.lower() in code_lower:
            logger.info(f"Still obfuscated: found {indicator}")
            return True
    
    # Check for long base64-like strings
    if re.search(r"['\"][A-Za-z0-9+/=]{200,}['\"]", code):
        logger.info("Still obfuscated: found long base64 string")
        return True
    
    logger.info("✅ Code appears clean")
    return False


def decode_single_layer(code: str) -> str:
    """Decode one layer of obfuscation."""
    original_code = code
    
    # Try different decoding methods
    code = extract_from_zip(code)
    if code != original_code:
        return code
    
    code = decode_base64_strings(code)
    code = extract_from_exec(code)
    code = decode_zlib(code)
    code = decode_marshal(code)
    code = decode_hex_strings(code)
    
    return code


def full_decode(code: str) -> tuple[str, int]:
    """Decode all layers until clean."""
    iteration = 0
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.info(f"🔄 Decode iteration {iteration}")
        
        if not is_still_obfuscated(code):
            logger.info(f"✅ Fully decoded in {iteration} iterations")
            return code, iteration
        
        new_code = decode_single_layer(code)
        
        if new_code == code:
            logger.info("No change detected, stopping")
            break
        
        if len(new_code) < 10:
            logger.warning("Decoded code too short")
            break
        
        code = new_code
    
    return code, iteration


# --- Background Processing ---
async def process_decode_task(chat_id: int, code: str, filename: str, status_msg_id: int):
    """Background decoding task."""
    try:
        logger.info(f"🚀 Starting decode for chat {chat_id}")
        
        # Update status
        await telegram_app.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_msg_id,
            text="🔄 <b>Decoding...</b>\n\nAnalyzing obfuscation layers...",
            parse_mode="HTML"
        )
        
        # Perform decoding
        decoded_code, iterations = await asyncio.get_event_loop().run_in_executor(
            None, full_decode, code
        )
        
        # Check if clean
        still_obfuscated = is_still_obfuscated(decoded_code)
        status = "✅ Fully Decoded" if not still_obfuscated else "⚠️ Partially Decoded"
        
        # Save to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp.write(decoded_code)
            tmp_name = tmp.name
        
        try:
            # Send file
            with open(tmp_name, 'rb') as f:
                await telegram_app.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(f, filename=f"decoded_{filename}"),
                    caption=(
                        f"{status}\n\n"
                        f"📄 {filename}\n"
                        f"📊 {len(decoded_code):,} chars\n"
                        f"🔄 {iterations} layers\n\n"
                        f"{CREDIT}"
                    ),
                    parse_mode="HTML"
                )
            
            # Send preview
            preview = decoded_code[:3900] if len(decoded_code) > 4000 else decoded_code
            if len(decoded_code) > 4000:
                preview += "\n\n..."
            
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text=f"<b>Preview:</b>\n\n<pre>{preview}</pre>",
                parse_mode="HTML"
            )
            
            # Delete status
            await telegram_app.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            
        finally:
            os.remove(tmp_name)
        
        logger.info(f"✅ Completed for chat {chat_id}")
        
    except Exception as e:
        logger.error(f"❌ Decode error: {e}")
        logger.error(traceback.format_exc())
        
        try:
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Decoding error\n\n<code>{str(e)[:300]}</code>",
                parse_mode="HTML"
            )
        except:
            pass


# --- Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome."""
    await update.message.reply_text(
        "🤖 <b>Python Decoder</b>\n\n"
        "Send obfuscated .py file\n"
        "Get fully decoded code\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help."""
    await update.message.reply_text(
        "<b>How to use:</b>\n\n"
        "1. Send .py file or code\n"
        "2. Wait 1-5 minutes\n"
        "3. Get decoded file\n\n"
        "Handles unlimited layers\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


# --- Message Handler ---
async def handle_code_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle code input."""
    code = ""
    filename = "code.py"
    
    try:
        # Extract code
        if update.message.document:
            doc = update.message.document
            
            if not doc.file_name.endswith('.py'):
                await update.message.reply_text("⚠️ Send .py file")
                return
            
            if doc.file_size > MAX_FILE_SIZE:
                await update.message.reply_text("⚠️ File too large")
                return
            
            filename = doc.file_name
            file = await doc.get_file()
            file_bytes = await file.download_as_bytearray()
            code = file_bytes.decode('utf-8', errors='ignore')
            
        elif update.message.text:
            code = update.message.text
        else:
            return

        if not code.strip():
            await update.message.reply_text("⚠️ Empty code")
            return

        # Send status
        status_msg = await update.message.reply_text(
            "🚀 <b>Decoding Started</b>\n\n"
            "⏳ This may take 1-5 minutes\n\n"
            "You can leave this chat.\n"
            "I'll send the file when ready!\n\n"
            "<i>Processing...</i>",
            parse_mode="HTML"
        )
        
        # Start background task
        asyncio.create_task(
            process_decode_task(
                update.effective_chat.id,
                code,
                filename,
                status_msg.message_id
            )
        )
        
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:200]}")


# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bot lifecycle."""
    global telegram_app
    
    logger.info("🚀 Starting bot...")
    
    telegram_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(180)
        .write_timeout(180)
        .build()
    )
    
    telegram_app.add_handler(CommandHandler("start", start_command))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.Document.PY) & ~filters.COMMAND,
            handle_code_input
        )
    )
    
    await telegram_app.initialize()
    await telegram_app.start()
    
    await telegram_app.bot.set_webhook(
        url=f"{WEBHOOK_URL}/webhook",
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )
    
    logger.info("✅ Bot ready!")
    
    yield
    
    await telegram_app.stop()
    await telegram_app.shutdown()


# --- FastAPI App ---
app = FastAPI(
    title="Python Decoder",
    version="5.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {"status": "running", "bot": "Python Decoder", "version": "5.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/webhook")
async def webhook(request: Request):
    try:
        json_data = await request.json()
        update = Update.de_json(json_data, telegram_app.bot)
        await telegram_app.process_update(update)
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return Response(status_code=500)


@app.head("/")
@app.head("/health")
async def head_health():
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
