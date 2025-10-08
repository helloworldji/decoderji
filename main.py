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
import html
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
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set!")
if not WEBHOOK_URL:
    raise ValueError("❌ WEBHOOK_URL not set!")

# --- Initialize Gemini AI ---
genai.configure(api_key=GEMINI_API_KEY)

def get_best_model():
    """Get best Gemini model."""
    try:
        available = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available.append(m.name)
        
        preferred = [
            'models/gemini-2.5-flash-lite',
            'models/gemini-flash-lite-latest',
            'models/gemini-flash-latest',
            'models/gemini-pro',
        ]
        
        for pref in preferred:
            if pref in available:
                logger.info(f"✅ Model: {pref}")
                return pref
        
        return available[0] if available else 'models/gemini-pro'
    except:
        return 'models/gemini-pro'

MODEL_NAME = get_best_model()
model = genai.GenerativeModel(MODEL_NAME)

# --- Constants ---
CREDIT = "Dev: @aadi_io"
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_ITERATIONS = 50

# --- Global Bot Application ---
telegram_app: Optional[Application] = None


# --- MANUAL DECODING FUNCTIONS ---

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
            logger.info(f"Extracted from exec/eval")
            return inner
    
    return code


def decode_base64_strings(code: str) -> str:
    """Decode base64 strings."""
    try:
        # Pattern: base64.b64decode(b'...')
        pattern = r"base64\.b64decode\s*KATEX_INLINE_OPEN\s*[b]?['\"]([A-Za-z0-9+/=]+)['\"]\s*KATEX_INLINE_CLOSE"
        
        def replacer(match):
            encoded = match.group(1)
            try:
                decoded = base64.b64decode(encoded).decode('utf-8', errors='ignore')
                logger.info(f"Decoded base64 chunk")
                return f'"""{decoded}"""'
            except:
                return match.group(0)
        
        code = re.sub(pattern, replacer, code)
        
        # Find standalone base64 strings
        pattern2 = r"[b]?['\"]([A-Za-z0-9+/=]{100,})['\"]"
        matches = re.findall(pattern2, code)
        
        for match in matches:
            try:
                decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                if len(decoded) > 20:
                    code = code.replace(match, decoded)
                    logger.info(f"Decoded standalone base64")
            except:
                pass
        
        return code
    except Exception as e:
        logger.error(f"Base64 error: {e}")
        return code


def extract_from_zip(code: str) -> str:
    """Extract ZIP files from code."""
    try:
        # Find base64 data
        pattern = r"['\"]([A-Za-z0-9+/=]{500,})['\"]"
        matches = re.findall(pattern, code)
        
        for match in matches:
            try:
                decoded_data = base64.b64decode(match)
                
                # Check if ZIP
                if decoded_data[:2] == b'PK':
                    logger.info("Found ZIP file!")
                    
                    with zipfile.ZipFile(io.BytesIO(decoded_data)) as zf:
                        names = zf.namelist()
                        if names:
                            # Try to find main file
                            main_file = None
                            for name in names:
                                if '__main__' in name or name.endswith('.py'):
                                    main_file = name
                                    break
                            
                            if not main_file:
                                main_file = names[0]
                            
                            content = zf.read(main_file).decode('utf-8', errors='ignore')
                            logger.info(f"Extracted {main_file}: {len(content)} chars")
                            return content
            except:
                continue
        
        return code
    except Exception as e:
        logger.error(f"ZIP error: {e}")
        return code


def decode_zlib(code: str) -> str:
    """Decode zlib."""
    try:
        pattern = r"zlib\.decompress\s*KATEX_INLINE_OPEN\s*(.+?)\s*KATEX_INLINE_CLOSE"
        match = re.search(pattern, code, re.DOTALL)
        
        if match:
            data_expr = match.group(1).strip()
            try:
                data = eval(data_expr)
                decoded = zlib.decompress(data).decode('utf-8', errors='ignore')
                logger.info(f"Decoded zlib")
                return decoded
            except:
                pass
        
        return code
    except:
        return code


def is_still_obfuscated(code: str) -> bool:
    """Check if obfuscated."""
    if not code or len(code) < 10:
        return True
    
    indicators = [
        'exec(',
        'eval(',
        'compile(',
        'base64.b64decode',
        'marshal.loads',
        'zlib.decompress',
        '__import__',
    ]
    
    for indicator in indicators:
        if indicator in code:
            return True
    
    # Check for long base64
    if re.search(r"['\"][A-Za-z0-9+/=]{200,}['\"]", code):
        return True
    
    return False


def manual_decode_layer(code: str) -> str:
    """Decode one layer manually."""
    original = code
    
    code = extract_from_zip(code)
    if code != original:
        return code
    
    code = decode_base64_strings(code)
    code = extract_from_exec(code)
    code = decode_zlib(code)
    
    return code


# --- AI VERIFICATION ---

async def ai_verify_and_clean(code: str) -> tuple[bool, str]:
    """Use Gemini to verify and clean up decoded code."""
    try:
        logger.info("🤖 AI verification...")
        
        # Truncate if too long
        code_sample = code[:20000] if len(code) > 20000 else code
        
        prompt = (
            "You are a Python code cleaner. The following code has been partially decoded from obfuscation.\n\n"
            "YOUR TASK:\n"
            "1. Check if ANY obfuscation remains (exec, eval, base64, etc.)\n"
            "2. If obfuscated, decode it completely\n"
            "3. Return ONLY clean, readable Python code\n"
            "4. Remove ALL encoding/obfuscation\n"
            "5. NO explanations, just code\n\n"
            f"CODE:\n{code_sample}\n\n"
            "CLEAN CODE:"
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=16384,
        )
        
        loop = asyncio.get_event_loop()
        
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        )
        
        # Extract response
        if response and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            text_parts.append(part.text)
                    
                    if text_parts:
                        result = ''.join(text_parts).strip()
                        
                        # Clean markdown
                        if result.startswith("```python"):
                            result = result[9:]
                        elif result.startswith("```"):
                            result = result[3:]
                        if result.endswith("```"):
                            result = result[:-3]
                        
                        result = result.strip()
                        
                        if len(result) > 20:
                            logger.info(f"✅ AI cleaned: {len(result)} chars")
                            return True, result
        
        return False, code
        
    except Exception as e:
        logger.error(f"AI verification error: {e}")
        return False, code


# --- FULL DECODING PROCESS ---

async def full_decode(code: str, chat_id: int, status_msg_id: int) -> tuple[str, int]:
    """Full decoding with manual + AI."""
    iteration = 0
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.info(f"🔄 Iteration {iteration}")
        
        # Update user every 3 iterations
        if iteration % 3 == 0:
            try:
                await telegram_app.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg_id,
                    text=f"🔄 <b>Decoding...</b>\n\nLayer {iteration} processing...",
                    parse_mode="HTML"
                )
            except:
                pass
        
        # Check if clean
        if not is_still_obfuscated(code):
            logger.info(f"✅ Clean after {iteration} iterations")
            
            # Final AI verification
            success, verified = await ai_verify_and_clean(code)
            if success and len(verified) > len(code) * 0.8:
                return verified, iteration
            return code, iteration
        
        # Manual decode
        new_code = await asyncio.get_event_loop().run_in_executor(
            None, manual_decode_layer, code
        )
        
        # If manual didn't change, try AI
        if new_code == code:
            logger.info("Manual decode stuck, trying AI...")
            success, ai_code = await ai_verify_and_clean(code)
            if success and ai_code != code:
                new_code = ai_code
            else:
                break
        
        if len(new_code) < 10:
            break
        
        code = new_code
        
        # Small delay
        if iteration % 5 == 0:
            await asyncio.sleep(1)
    
    # Final AI pass
    logger.info("Final AI verification...")
    success, verified = await ai_verify_and_clean(code)
    if success:
        return verified, iteration
    
    return code, iteration


# --- Background Processing ---

async def process_decode_task(chat_id: int, code: str, filename: str, status_msg_id: int):
    """Background decoding."""
    try:
        logger.info(f"🚀 Starting decode for {chat_id}")
        
        # Decode
        decoded_code, iterations = await full_decode(code, chat_id, status_msg_id)
        
        # Verify final state
        still_obfuscated = is_still_obfuscated(decoded_code)
        status_icon = "✅" if not still_obfuscated else "⚠️"
        status_text = "Fully Decoded" if not still_obfuscated else "Partially Decoded"
        
        # Save file
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
                        f"{status_icon} <b>{status_text}</b>\n\n"
                        f"📄 {filename}\n"
                        f"📊 {len(decoded_code):,} chars\n"
                        f"🔄 {iterations} layers\n\n"
                        f"{CREDIT}"
                    ),
                    parse_mode="HTML"
                )
            
            # Send preview - PROPERLY ESCAPED
            preview_len = min(3800, len(decoded_code))
            preview = decoded_code[:preview_len]
            if len(decoded_code) > 3800:
                preview += "\n..."
            
            # Escape HTML entities
            preview_escaped = html.escape(preview)
            
            try:
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text=f"<b>📝 Preview:</b>\n\n<pre>{preview_escaped}</pre>",
                    parse_mode="HTML"
                )
            except Exception as e:
                # If HTML parsing still fails, send as plain text
                logger.error(f"Preview send error: {e}")
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text=f"Preview:\n\n{preview[:4000]}"
                )
            
            # Delete status
            try:
                await telegram_app.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            except:
                pass
            
        finally:
            os.remove(tmp_name)
        
        logger.info(f"✅ Completed for {chat_id}")
        
    except Exception as e:
        logger.error(f"❌ Task error: {e}")
        logger.error(traceback.format_exc())
        
        try:
            error_text = html.escape(str(e)[:300])
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text=f"❌ <b>Decoding Error</b>\n\n<code>{error_text}</code>",
                parse_mode="HTML"
            )
        except:
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Decoding failed"
            )


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
        "• Unlimited layers\n"
        "• AI verification\n"
        "• Background processing\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


# --- Message Handler ---

async def handle_code_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle input."""
    code = ""
    filename = "code.py"
    
    try:
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
            "⏳ Takes 1-5 minutes\n\n"
            "You can leave this chat.\n"
            "File will be sent automatically!\n\n"
            "<i>Processing...</i>",
            parse_mode="HTML"
        )
        
        # Start task
        asyncio.create_task(
            process_decode_task(
                update.effective_chat.id,
                code,
                filename,
                status_msg.message_id
            )
        )
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:200]}")


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle."""
    global telegram_app
    
    logger.info("🚀 Starting...")
    
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
    
    logger.info("✅ Ready!")
    
    yield
    
    await telegram_app.stop()
    await telegram_app.shutdown()


# --- FastAPI ---

app = FastAPI(title="Python Decoder", version="6.0.0", lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "running", "version": "6.0.0"}


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
