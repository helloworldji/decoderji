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
import ast
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

# --- Logging ---
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

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY or not WEBHOOK_URL:
    raise ValueError("❌ Missing environment variables!")

# --- Gemini Init ---
genai.configure(api_key=GEMINI_API_KEY)

def get_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                logger.info(f"Using: {m.name}")
                return genai.GenerativeModel(m.name)
    except:
        pass
    return genai.GenerativeModel('models/gemini-pro')

model = get_model()

# --- Constants ---
CREDIT = "Dev: @aadi_io"
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_ITERATIONS = 100

telegram_app: Optional[Application] = None


# --- AGGRESSIVE DECODING FUNCTIONS ---

def safe_eval_decode(code_str: str) -> str:
    """Safely evaluate and decode expressions."""
    try:
        # Try to evaluate the expression
        result = eval(code_str, {"__builtins__": {}}, {
            "base64": base64,
            "zlib": zlib,
            "marshal": marshal,
            "bytes": bytes,
            "bytearray": bytearray,
        })
        
        if isinstance(result, bytes):
            return result.decode('utf-8', errors='ignore')
        elif isinstance(result, str):
            return result
        else:
            return str(result)
    except:
        return code_str


def extract_all_base64(code: str) -> str:
    """Extract and decode ALL base64 in code."""
    logger.info("🔍 Extracting base64...")
    
    # Pattern 1: base64.b64decode(...)
    pattern1 = r"base64\.b64decode\s*KATEX_INLINE_OPEN\s*([^)]+)\s*KATEX_INLINE_CLOSE"
    matches = re.findall(pattern1, code)
    
    for match in matches:
        try:
            decoded = safe_eval_decode(match)
            if decoded and len(decoded) > 10:
                code = code.replace(f"base64.b64decode({match})", f'"""{decoded}"""')
                logger.info(f"✅ Decoded base64 expression")
        except:
            pass
    
    # Pattern 2: Standalone long base64 strings
    pattern2 = r"[b]?['\"]([A-Za-z0-9+/=]{200,})['\"]"
    matches = re.findall(pattern2, code)
    
    for match in matches:
        try:
            decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
            if decoded and len(decoded) > 10 and decoded.isprintable():
                code = code.replace(match, decoded)
                logger.info(f"✅ Decoded standalone base64")
        except:
            pass
    
    return code


def extract_from_zip_aggressive(code: str) -> str:
    """Aggressively extract ZIP files."""
    logger.info("📦 Checking for ZIP...")
    
    # Find all base64 strings
    pattern = r"['\"]([A-Za-z0-9+/=]{500,})['\"]"
    matches = re.findall(pattern, code)
    
    for match in matches:
        try:
            decoded_bytes = base64.b64decode(match)
            
            # Check if it's a ZIP (multiple signatures)
            if (decoded_bytes[:2] == b'PK' or 
                decoded_bytes[:4] == b'PK\x03\x04' or
                decoded_bytes[:4] == b'PK\x05\x06'):
                
                logger.info("🎯 Found ZIP file!")
                
                try:
                    with zipfile.ZipFile(io.BytesIO(decoded_bytes)) as zf:
                        all_content = []
                        
                        for name in zf.namelist():
                            try:
                                file_content = zf.read(name).decode('utf-8', errors='ignore')
                                logger.info(f"  📄 Extracted: {name} ({len(file_content)} chars)")
                                all_content.append(f"# File: {name}\n{file_content}")
                            except:
                                continue
                        
                        if all_content:
                            return "\n\n".join(all_content)
                except Exception as e:
                    logger.error(f"ZIP extraction error: {e}")
        except:
            continue
    
    return code


def extract_exec_eval(code: str) -> str:
    """Extract from exec/eval calls."""
    logger.info("🔓 Extracting exec/eval...")
    
    # Patterns for exec/eval
    patterns = [
        r'exec\s*KATEX_INLINE_OPEN\s*([^)]+)\s*KATEX_INLINE_CLOSE',
        r'eval\s*KATEX_INLINE_OPEN\s*([^)]+)\s*KATEX_INLINE_CLOSE',
        r'compile\s*KATEX_INLINE_OPEN\s*([^,]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, code, re.DOTALL)
        for match in matches:
            try:
                # Try to decode the expression
                decoded = safe_eval_decode(match.strip())
                if decoded and decoded != match and len(decoded) > 20:
                    logger.info(f"✅ Extracted from exec/eval")
                    return decoded
            except:
                pass
    
    return code


def decode_all_zlib(code: str) -> str:
    """Decode all zlib compressed data."""
    logger.info("🗜️ Checking zlib...")
    
    pattern = r"zlib\.decompress\s*KATEX_INLINE_OPEN\s*([^)]+)\s*KATEX_INLINE_CLOSE"
    matches = re.findall(pattern, code)
    
    for match in matches:
        try:
            decoded = safe_eval_decode(match)
            if decoded and len(decoded) > 10:
                code = code.replace(f"zlib.decompress({match})", f'"""{decoded}"""')
                logger.info(f"✅ Decoded zlib")
        except:
            pass
    
    return code


def decode_marshal(code: str) -> str:
    """Decode marshal data."""
    logger.info("🔐 Checking marshal...")
    
    pattern = r"marshal\.loads\s*KATEX_INLINE_OPEN\s*([^)]+)\s*KATEX_INLINE_CLOSE"
    matches = re.findall(pattern, code)
    
    for match in matches:
        try:
            data = safe_eval_decode(match)
            if data:
                logger.info(f"✅ Found marshal data")
                # Marshal usually contains code objects, try to extract strings
                if hasattr(data, 'co_consts'):
                    consts = str(data.co_consts)
                    return consts
        except:
            pass
    
    return code


def decode_hex(code: str) -> str:
    """Decode hex strings."""
    logger.info("🔢 Checking hex...")
    
    patterns = [
        r"bytes\.fromhex\s*KATEX_INLINE_OPEN\s*['\"]([0-9a-fA-F]+)['\"]\s*KATEX_INLINE_CLOSE",
        r"bytearray\.fromhex\s*KATEX_INLINE_OPEN\s*['\"]([0-9a-fA-F]+)['\"]\s*KATEX_INLINE_CLOSE",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, code)
        for match in matches:
            try:
                decoded = bytes.fromhex(match).decode('utf-8', errors='ignore')
                if decoded:
                    code = code.replace(match, decoded)
                    logger.info(f"✅ Decoded hex")
            except:
                pass
    
    return code


def is_obfuscated(code: str) -> bool:
    """Check if code is still obfuscated."""
    if not code or len(code) < 10:
        return True
    
    # Check for obfuscation indicators
    indicators = [
        (r'exec\s*KATEX_INLINE_OPEN', 'exec()'),
        (r'eval\s*KATEX_INLINE_OPEN', 'eval()'),
        (r'compile\s*KATEX_INLINE_OPEN', 'compile()'),
        (r'base64\.b64decode', 'base64'),
        (r'base64\.b32decode', 'base32'),
        (r'marshal\.loads', 'marshal'),
        (r'zlib\.decompress', 'zlib'),
        (r'__import__.*base64', 'import base64'),
        (r"['\"][A-Za-z0-9+/=]{300,}['\"]", 'long base64 string'),
    ]
    
    for pattern, name in indicators:
        if re.search(pattern, code):
            logger.info(f"⚠️ Still obfuscated: {name}")
            return True
    
    logger.info("✅ Code appears clean")
    return False


def aggressive_decode_layer(code: str) -> str:
    """One aggressive decode pass."""
    original = code
    
    # Try ZIP extraction first (most common)
    code = extract_from_zip_aggressive(code)
    if code != original and len(code) > len(original) * 0.1:
        logger.info("🎯 ZIP extraction successful!")
        return code
    
    # Try other methods
    code = extract_all_base64(code)
    code = decode_all_zlib(code)
    code = decode_marshal(code)
    code = decode_hex(code)
    code = extract_exec_eval(code)
    
    return code


# --- AI DECODER ---

async def ai_decode(code: str) -> tuple[bool, str]:
    """Use AI to decode when manual methods fail."""
    try:
        logger.info("🤖 AI decode attempt...")
        
        # Take sample
        sample = code[:15000] if len(code) > 15000 else code
        
        prompt = (
            "Decode this obfuscated Python code. Extract ALL hidden code.\n"
            "If it contains base64, ZIP files, exec/eval, or any encoding - decode it ALL.\n"
            "Return ONLY the final clean Python code with NO obfuscation.\n"
            "Do not include explanations or markdown.\n\n"
            f"CODE:\n{sample}\n\n"
            "DECODED:"
        )

        safety = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        config = genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=16384,
        )
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(prompt, generation_config=config, safety_settings=safety)
        )
        
        # Extract text
        if response and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            parts.append(part.text)
                    
                    if parts:
                        result = ''.join(parts).strip()
                        
                        # Clean markdown
                        result = result.replace("```python", "").replace("```", "").strip()
                        
                        if len(result) > 50:
                            logger.info(f"✅ AI decoded: {len(result)} chars")
                            return True, result
        
        return False, code
        
    except Exception as e:
        logger.error(f"AI error: {e}")
        return False, code


# --- MAIN DECODE LOOP ---

async def complete_decode(code: str, chat_id: int, msg_id: int) -> tuple[str, int]:
    """Decode until absolutely nothing is left."""
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"🔄 ITERATION {iteration}/{MAX_ITERATIONS}")
        logger.info(f"{'='*50}")
        
        # Update user
        if iteration % 2 == 0:
            try:
                await telegram_app.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=f"🔄 <b>Deep Decoding...</b>\n\nLayer {iteration} of {MAX_ITERATIONS}\n\n<i>Please wait...</i>",
                    parse_mode="HTML"
                )
            except:
                pass
        
        # Check if clean
        if not is_obfuscated(code):
            logger.info(f"✅✅✅ FULLY DECODED in {iteration} iterations!")
            return code, iteration
        
        # Manual decode
        logger.info("🔧 Manual decode...")
        new_code = await asyncio.get_event_loop().run_in_executor(
            None, aggressive_decode_layer, code
        )
        
        # Check if manual worked
        if new_code != code and len(new_code) >= len(code) * 0.5:
            logger.info(f"✅ Manual decode made progress: {len(code)} -> {len(new_code)}")
            code = new_code
            continue
        
        # Try AI every 3 iterations or if stuck
        if iteration % 3 == 0 or new_code == code:
            logger.info("🤖 Trying AI decode...")
            success, ai_code = await ai_decode(code)
            
            if success and ai_code != code and len(ai_code) >= len(code) * 0.5:
                logger.info(f"✅ AI decode made progress: {len(code)} -> {len(ai_code)}")
                code = ai_code
                continue
        
        # If nothing worked
        if new_code == code:
            logger.warning(f"⚠️ Stuck at iteration {iteration}")
            # Try AI one more time
            success, ai_code = await ai_decode(code)
            if success and ai_code != code:
                code = ai_code
            else:
                logger.warning("❌ Cannot decode further")
                break
        
        code = new_code if new_code != code else code
        
        # Prevent infinite loops on small changes
        if iteration > 5 and len(code) < 100:
            logger.warning("Code too small, stopping")
            break
    
    # Final check
    logger.info(f"🏁 Decode complete: {iteration} iterations")
    return code, iteration


# --- Background Task ---

async def decode_task(chat_id: int, code: str, filename: str, msg_id: int):
    """Background decode."""
    try:
        logger.info(f"🚀 Starting decode for {chat_id}")
        
        # Decode
        decoded, iterations = await complete_decode(code, chat_id, msg_id)
        
        # Check status
        clean = not is_obfuscated(decoded)
        status = "✅ Fully Decoded" if clean else "⚠️ Partially Decoded"
        
        # Save
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name
        
        try:
            # Send file
            with open(tmp_path, 'rb') as f:
                await telegram_app.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(f, filename=f"decoded_{filename}"),
                    caption=(
                        f"{status}\n\n"
                        f"📄 {filename}\n"
                        f"📊 {len(decoded):,} chars\n"
                        f"🔄 {iterations} layers\n"
                        f"{'🎯 100% Clean' if clean else '⚠️ Some obfuscation may remain'}\n\n"
                        f"{CREDIT}"
                    ),
                    parse_mode="HTML"
                )
            
            # Send preview
            preview = decoded[:3500] if len(decoded) > 3500 else decoded
            preview_safe = html.escape(preview)
            
            try:
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text=f"<b>Preview:</b>\n\n<pre>{preview_safe}</pre>",
                    parse_mode="HTML"
                )
            except:
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text=f"Preview:\n\n{preview[:4000]}"
                )
            
            # Delete status
            try:
                await telegram_app.bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except:
                pass
        
        finally:
            os.remove(tmp_path)
        
        logger.info(f"✅ Done for {chat_id}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.error(traceback.format_exc())
        
        try:
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Error\n\n{html.escape(str(e)[:300])}"
            )
        except:
            pass


# --- Commands ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 <b>Python Decoder</b>\n\n"
        "Send obfuscated .py file\n"
        "Get 100% decoded code\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "<b>Usage:</b>\n\n"
        "1. Send .py file\n"
        "2. Wait 2-10 minutes\n"
        "3. Get decoded file\n\n"
        "• Up to 100 layers\n"
        "• AI + Manual decoding\n"
        "• Background processing\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


# --- Message Handler ---

async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    code = ""
    filename = "code.py"
    
    try:
        if update.message.document:
            doc = update.message.document
            
            if not doc.file_name.endswith('.py'):
                await update.message.reply_text("⚠️ Send .py file only")
                return
            
            if doc.file_size > MAX_FILE_SIZE:
                await update.message.reply_text(f"⚠️ Max {MAX_FILE_SIZE//1024//1024}MB")
                return
            
            filename = doc.file_name
            file = await doc.get_file()
            code = (await file.download_as_bytearray()).decode('utf-8', errors='ignore')
            
        elif update.message.text:
            code = update.message.text
        else:
            return

        if not code.strip():
            return

        # Status
        msg = await update.message.reply_text(
            "🚀 <b>Decoding Started</b>\n\n"
            "⏳ This may take 2-10 minutes\n"
            "Processing up to 100 layers\n\n"
            "You can close Telegram.\n"
            "File will be sent automatically!\n\n"
            "<i>Deep decoding in progress...</i>",
            parse_mode="HTML"
        )
        
        # Start
        asyncio.create_task(decode_task(update.effective_chat.id, code, filename, msg.message_id))
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        await update.message.reply_text("❌ Error")


# --- App Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global telegram_app
    
    logger.info("🚀 Starting...")
    
    telegram_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(180)
        .write_timeout(180)
        .build()
    )
    
    telegram_app.add_handler(CommandHandler("start", start_cmd))
    telegram_app.add_handler(CommandHandler("help", help_cmd))
    telegram_app.add_handler(MessageHandler((filters.TEXT | filters.Document.PY) & ~filters.COMMAND, handle_input))
    
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.bot.set_webhook(url=f"{WEBHOOK_URL}/webhook", allowed_updates=Update.ALL_TYPES)
    
    logger.info("✅ Ready!")
    
    yield
    
    await telegram_app.stop()
    await telegram_app.shutdown()


# --- FastAPI ---

app = FastAPI(title="Decoder", version="7.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "running", "version": "7.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Webhook: {e}")
        return Response(status_code=500)

@app.head("/")
@app.head("/health")
async def head():
    return Response(status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
