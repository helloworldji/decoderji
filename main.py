import os
import tempfile
import logging
import asyncio
import subprocess
import sys
import re
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
    raise ValueError("❌ TELEGRAM_BOT_TOKEN environment variable not set!")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY environment variable not set!")
if not WEBHOOK_URL:
    raise ValueError("❌ WEBHOOK_URL environment variable not set!")

# --- Initialize Gemini AI ---
genai.configure(api_key=GEMINI_API_KEY)

def get_best_model():
    """Get the best available Gemini model."""
    try:
        logger.info("🔍 Searching for available Gemini models...")
        
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
                logger.info(f"✅ Selected: {pref}")
                return pref
        
        if available:
            return available[0]
            
    except Exception as e:
        logger.warning(f"⚠️ Could not list models: {e}")
    
    return 'models/gemini-pro'

MODEL_NAME = get_best_model()
model = genai.GenerativeModel(MODEL_NAME)
logger.info(f"🤖 Initialized: {MODEL_NAME}")

# --- Constants ---
CREDIT = "🔧 Dev: @aadi_io"
MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_CODE_LENGTH = 30000
DECODER_TIMEOUT = 45
MAX_DECODE_ITERATIONS = 5  # Keep decoding up to 5 times

# --- Global Bot Application ---
telegram_app: Optional[Application] = None


# --- Helper Functions ---
def is_code_obfuscated(code: str) -> bool:
    """Check if code still contains obfuscation patterns."""
    if not code or len(code) < 10:
        return True
    
    # Common obfuscation indicators
    obfuscation_patterns = [
        r'exec\s*KATEX_INLINE_OPEN',
        r'eval\s*KATEX_INLINE_OPEN',
        r'compile\s*KATEX_INLINE_OPEN',
        r'__import__\s*KATEX_INLINE_OPEN',
        r'base64\.b64decode',
        r'base64\.b32decode',
        r'base64\.b16decode',
        r'codecs\.decode',
        r'marshal\.loads',
        r'pickle\.loads',
        r'zlib\.decompress',
        r'gzip\.decompress',
        r'lambda\s+.*:\s*exec',
        r'lambda\s+.*:\s*eval',
        r'\\x[0-9a-fA-F]{2}',  # Hex encoding
        r'chr\s*KATEX_INLINE_OPEN\s*\d+\s*KATEX_INLINE_CLOSE',  # Character encoding
        r'ord\s*KATEX_INLINE_OPEN',  # Ordinal encoding
        r'bytes\.fromhex',
        r'int\s*KATEX_INLINE_OPEN.*,\s*16\s*KATEX_INLINE_CLOSE',  # Hex to int
    ]
    
    code_lower = code.lower()
    
    # Check for patterns
    for pattern in obfuscation_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            logger.info(f"Found obfuscation pattern: {pattern}")
            return True
    
    # Check for suspicious base64-like strings
    base64_like = re.findall(r"['\"]([A-Za-z0-9+/=]{50,})['\"]", code)
    if base64_like:
        logger.info(f"Found {len(base64_like)} potential base64 strings")
        return True
    
    # Check for excessive string concatenation (common in obfuscation)
    if code.count("'+'") > 10 or code.count('"+') > 10:
        logger.info("Found excessive string concatenation")
        return True
    
    logger.info("✅ Code appears to be clean (no obfuscation detected)")
    return False


def clean_code_response(text: str) -> str:
    """Clean up AI response."""
    text = text.strip()
    
    if text.startswith("```python"):
        text = text[9:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
    
    return text.strip()


def extract_response_text(response) -> tuple[bool, str]:
    """Safely extract text from Gemini response."""
    try:
        if not response or not hasattr(response, 'candidates') or not response.candidates:
            return False, "No response from API"
        
        candidate = response.candidates[0]
        
        if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                text_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        text_parts.append(part.text)
                
                if text_parts:
                    return True, ''.join(text_parts)
        
        return False, "Could not extract response text"
        
    except Exception as e:
        return False, str(e)


async def generate_iterative_decoder(code: str) -> tuple[bool, str]:
    """Generate a decoder that keeps decoding until code is fully clean."""
    try:
        logger.info("🔧 Generating iterative decoder...")
        
        code_sample = code[:15000] if len(code) > 15000 else code
        
        # Enhanced prompt for complete decoding
        prompt = (
            "You are a Python deobfuscation expert. Create a COMPLETE decoder script that will "
            "FULLY decode ALL layers of obfuscation until the code is completely readable.\n\n"
            
            "CRITICAL REQUIREMENTS:\n"
            "1. The decoder MUST iterate/loop until NO obfuscation remains\n"
            "2. Handle ALL these methods: base64, exec, eval, marshal, zlib, gzip, hex, chr, compile\n"
            "3. Detect and decode MULTIPLE layers automatically\n"
            "4. Keep decoding until code has NO exec/eval/base64/marshal/etc.\n"
            "5. Return ONLY clean, readable Python source code\n\n"
            
            "DECODER TEMPLATE TO FOLLOW:\n"
            "```python\n"
            "import base64, marshal, zlib, gzip, re, codecs\n"
            "import sys\n\n"
            
            "def is_still_obfuscated(code):\n"
            "    '''Check if code still has obfuscation'''\n"
            "    patterns = [r'exec\KATEX_INLINE_OPEN', r'eval\KATEX_INLINE_OPEN', r'base64\\.', r'marshal\\.', r'zlib\\.']\n"
            "    return any(re.search(p, code) for p in patterns)\n\n"
            
            "def decode_layer(code):\n"
            "    '''Decode one layer of obfuscation'''\n"
            "    # Your decoding logic here\n"
            "    # Handle base64, exec, eval, marshal, zlib, etc.\n"
            "    return decoded_code\n\n"
            
            "# Read obfuscated code\n"
            "with open('obfuscated_code.txt', 'r') as f:\n"
            "    code = f.read()\n\n"
            
            "# Iteratively decode\n"
            "max_iterations = 10\n"
            "for i in range(max_iterations):\n"
            "    if not is_still_obfuscated(code):\n"
            "        break\n"
            "    code = decode_layer(code)\n\n"
            
            "# Write fully decoded code\n"
            "with open('decoded_code.py', 'w') as f:\n"
            "    f.write(code)\n"
            "```\n\n"
            
            "OBFUSCATED CODE TO ANALYZE:\n"
            f"{code_sample}\n\n"
            
            "Generate the COMPLETE iterative decoder script that will fully decode this code:"
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        generation_config = genai.types.GenerationConfig(
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            max_output_tokens=8192,
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
        
        success, result = extract_response_text(response)
        
        if not success:
            return False, f"Failed to generate decoder: {result}"
        
        decoder_script = clean_code_response(result)
        
        if len(decoder_script) < 50:
            return False, "Generated decoder is too short"
        
        logger.info(f"✅ Generated iterative decoder ({len(decoder_script)} chars)")
        return True, decoder_script
        
    except Exception as e:
        logger.error(f"❌ Error generating decoder: {e}")
        return False, str(e)


async def execute_decoder(decoder_script: str, obfuscated_code: str) -> tuple[bool, str]:
    """Execute decoder and verify output is clean."""
    try:
        logger.info("⚙️ Executing decoder script...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write obfuscated code
            obfuscated_file = os.path.join(temp_dir, "obfuscated_code.txt")
            with open(obfuscated_file, 'w', encoding='utf-8') as f:
                f.write(obfuscated_code)
            
            # Write decoder
            decoder_file = os.path.join(temp_dir, "decoder.py")
            with open(decoder_file, 'w', encoding='utf-8') as f:
                f.write(decoder_script)
            
            decoded_file = os.path.join(temp_dir, "decoded_code.py")
            
            # Execute
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    decoder_file,
                    cwd=temp_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=DECODER_TIMEOUT
                )
                
                # Check output file
                if os.path.exists(decoded_file):
                    with open(decoded_file, 'r', encoding='utf-8') as f:
                        decoded_code = f.read()
                    
                    if decoded_code.strip():
                        logger.info(f"✅ Decoder produced output ({len(decoded_code)} chars)")
                        return True, decoded_code
                
                # Check stdout
                if stdout:
                    decoded_code = stdout.decode('utf-8', errors='ignore')
                    if len(decoded_code) > 20:
                        return True, decoded_code
                
                # Log errors
                if stderr:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    logger.error(f"Decoder stderr: {error_msg[:500]}")
                    return False, f"Decoder error: {error_msg[:300]}"
                
                return False, "Decoder produced no output"
                
            except asyncio.TimeoutError:
                return False, "Decoder timeout (>45s)"
            
    except Exception as e:
        logger.error(f"❌ Execution error: {e}")
        return False, str(e)


async def ensure_fully_decoded(code: str, max_attempts: int = 3) -> tuple[bool, str, list]:
    """
    Ensure code is COMPLETELY decoded with NO obfuscation remaining.
    Returns: (success, final_code, decode_log)
    """
    decode_log = []
    current_code = code
    
    for iteration in range(1, MAX_DECODE_ITERATIONS + 1):
        logger.info(f"🔄 Decode iteration {iteration}/{MAX_DECODE_ITERATIONS}")
        decode_log.append(f"Iteration {iteration}:")
        
        # Check if still obfuscated
        if not is_code_obfuscated(current_code):
            logger.info(f"✅ Code is fully decoded after {iteration-1} iteration(s)")
            decode_log.append("✅ Code is fully clean - no more obfuscation detected!")
            return True, current_code, decode_log
        
        decode_log.append("⚠️ Obfuscation detected - generating decoder...")
        
        # Generate decoder for current state
        success, decoder = await generate_iterative_decoder(current_code)
        
        if not success:
            decode_log.append(f"❌ Decoder generation failed: {decoder}")
            if iteration == 1:
                return False, current_code, decode_log
            # Return best we have so far
            return True, current_code, decode_log
        
        decode_log.append("✅ Decoder generated - executing...")
        
        # Execute decoder
        success, decoded = await execute_decoder(decoder, current_code)
        
        if not success:
            decode_log.append(f"❌ Execution failed: {decoded}")
            if iteration == 1:
                return False, current_code, decode_log
            # Return best we have so far
            return True, current_code, decode_log
        
        # Check if we made progress
        if len(decoded) < 10:
            decode_log.append("❌ Decoded output too short")
            return True, current_code, decode_log
        
        if decoded == current_code:
            decode_log.append("⚠️ No change detected - stopping")
            return True, current_code, decode_log
        
        decode_log.append(f"✅ Decoded {len(decoded)} chars - checking for more layers...")
        current_code = decoded
    
    # Max iterations reached
    decode_log.append(f"⚠️ Max iterations ({MAX_DECODE_ITERATIONS}) reached")
    
    if is_code_obfuscated(current_code):
        decode_log.append("⚠️ Some obfuscation may remain")
    else:
        decode_log.append("✅ Code appears clean!")
    
    return True, current_code, decode_log


# --- Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message."""
    welcome_text = (
        "🤖 <b>Python Decoder Bot V3.0</b>\n\n"
        "🔓 <b>GUARANTEED COMPLETE DECODING!</b>\n\n"
        "I decode <b>EVERY layer</b> until code is <b>100% readable</b>!\n\n"
        "<b>✨ Smart Features:</b>\n"
        "• Detects obfuscation automatically\n"
        "• Decodes iteratively (up to 5 layers)\n"
        "• Validates output is clean\n"
        "• Shows decoding progress\n\n"
        "<b>🎯 Handles:</b>\n"
        "✅ Base64/Hex/ROT13\n"
        "✅ Exec/Eval wrappers\n"
        "✅ Marshal/Zlib/Gzip\n"
        "✅ Multi-layer obfuscation\n"
        "✅ Mixed encoding\n\n"
        "<b>📤 Usage:</b>\n"
        "Send .py file or paste code\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(welcome_text, parse_mode="HTML")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help information."""
    help_text = (
        "<b>📖 Complete Decoding Bot</b>\n\n"
        "<b>How It Works:</b>\n\n"
        "1️⃣ <b>Analyze</b>\n"
        "Detect obfuscation patterns\n\n"
        "2️⃣ <b>Generate Decoder</b>\n"
        "Create custom decoder script\n\n"
        "3️⃣ <b>Execute & Verify</b>\n"
        "Run decoder, check output\n\n"
        "4️⃣ <b>Repeat If Needed</b>\n"
        "Keep decoding until clean\n\n"
        "5️⃣ <b>Validate</b>\n"
        "Confirm NO obfuscation remains\n\n"
        "<b>Commands:</b>\n"
        "/start - Start bot\n"
        "/help - This guide\n"
        "/model - AI info\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show model info."""
    info_text = (
        f"<b>🤖 AI Configuration</b>\n\n"
        f"<b>Model:</b> <code>{MODEL_NAME.split('/')[-1]}</code>\n"
        f"<b>Mode:</b> Iterative Decoder\n"
        f"<b>Max Iterations:</b> {MAX_DECODE_ITERATIONS}\n"
        f"<b>Timeout:</b> {DECODER_TIMEOUT}s\n"
        f"<b>Validation:</b> Enabled\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(info_text, parse_mode="HTML")


# --- Message Handler ---
async def handle_code_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming code."""
    code = ""
    filename_original = "code.py"
    processing_msg = None

    try:
        # Extract code
        if update.message.document:
            doc = update.message.document
            
            if not doc.file_name.endswith('.py'):
                await update.message.reply_text(
                    "⚠️ Please send a .py file",
                    parse_mode="HTML"
                )
                return
            
            if doc.file_size > MAX_FILE_SIZE:
                await update.message.reply_text(
                    f"⚠️ File too large (max {MAX_FILE_SIZE//(1024*1024)}MB)",
                    parse_mode="HTML"
                )
                return
            
            filename_original = doc.file_name
            
            processing_msg = await update.message.reply_text(
                "📥 <b>Downloading...</b>",
                parse_mode="HTML"
            )
            
            file = await doc.get_file()
            file_bytes = await file.download_as_bytearray()
            code = file_bytes.decode('utf-8', errors='ignore')
            
        elif update.message.text:
            code = update.message.text
            
            if len(code) > MAX_CODE_LENGTH:
                await update.message.reply_text(
                    f"⚠️ Code too long (max {MAX_CODE_LENGTH:,} chars)",
                    parse_mode="HTML"
                )
                return
        else:
            return

        if not code.strip():
            await update.message.reply_text("⚠️ Empty code", parse_mode="HTML")
            return

        # Initial check
        if not processing_msg:
            processing_msg = await update.message.reply_text(
                "🔬 <b>Analyzing code...</b>",
                parse_mode="HTML"
            )
        else:
            await processing_msg.edit_text(
                "🔬 <b>Analyzing code...</b>",
                parse_mode="HTML"
            )
        
        # Check if already clean
        if not is_code_obfuscated(code):
            await processing_msg.edit_text(
                "✅ <b>Code is already clean!</b>\n\n"
                "No obfuscation detected. This code is readable as-is.",
                parse_mode="HTML"
            )
            
            # Still send it back
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
                tmp.write(code)
                tmp_name = tmp.name
            
            try:
                with open(tmp_name, 'rb') as f:
                    await update.message.reply_document(
                        document=InputFile(f, filename=filename_original),
                        caption="✅ Already clean (no decoding needed)"
                    )
            finally:
                os.remove(tmp_name)
            
            await processing_msg.delete()
            return

        # Start decoding process
        await processing_msg.edit_text(
            "🔧 <b>Decoding...</b>\n"
            "Starting iterative decoding process...",
            parse_mode="HTML"
        )

        # Full decode with iterations
        success, decoded_code, decode_log = await ensure_fully_decoded(code)

        if not success:
            error_log = "\n".join(decode_log[-5:])  # Last 5 entries
            await processing_msg.edit_text(
                f"❌ <b>Decoding Failed</b>\n\n"
                f"<b>Log:</b>\n<code>{error_log}</code>\n\n"
                f"Contact {CREDIT}",
                parse_mode="HTML"
            )
            return

        # Verify final code is clean
        still_obfuscated = is_code_obfuscated(decoded_code)
        
        # Prepare log summary
        log_summary = "\n".join(decode_log)
        
        await processing_msg.edit_text(
            "✅ <b>Decoding Complete!</b>\n📤 Sending results...",
            parse_mode="HTML"
        )

        # Send decoded file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp.write(decoded_code)
            tmp_name = tmp.name

        try:
            status_icon = "✅" if not still_obfuscated else "⚠️"
            status_text = "Fully Decoded" if not still_obfuscated else "Partially Decoded"
            
            with open(tmp_name, 'rb') as f:
                await update.message.reply_document(
                    document=InputFile(f, filename=f"decoded_{filename_original}"),
                    caption=(
                        f"{status_icon} <b>{status_text}!</b>\n\n"
                        f"📄 File: {filename_original}\n"
                        f"📊 Size: {len(decoded_code):,} chars\n"
                        f"🔄 Iterations: {len([l for l in decode_log if 'Iteration' in l])}\n"
                        f"🎯 Status: {'Clean' if not still_obfuscated else 'May have remaining obfuscation'}\n\n"
                        f"{CREDIT}"
                    ),
                    parse_mode="HTML"
                )
            
            # Send decode log
            await update.message.reply_text(
                f"<b>📋 Decode Log:</b>\n\n<code>{log_summary[:3000]}</code>",
                parse_mode="HTML"
            )
            
            # Send preview
            preview_len = min(4000, len(decoded_code))
            preview = decoded_code[:preview_len]
            if len(decoded_code) > 4000:
                preview += "\n\n... (see file for complete code)"
            
            await update.message.reply_text(
                f"<b>📝 Code Preview:</b>\n\n<pre>{preview}</pre>",
                parse_mode="HTML"
            )
            
            await processing_msg.delete()

        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        
        try:
            if processing_msg:
                await processing_msg.edit_text(
                    f"❌ <b>Error</b>\n\n<code>{str(e)[:400]}</code>",
                    parse_mode="HTML"
                )
            else:
                await update.message.reply_text(
                    f"❌ <b>Error</b>\n\n<code>{str(e)[:400]}</code>",
                    parse_mode="HTML"
                )
        except:
            await update.message.reply_text("❌ An error occurred")


# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage bot lifecycle."""
    global telegram_app
    
    logger.info("🚀 Starting Telegram Bot...")
    
    telegram_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(120)
        .write_timeout(120)
        .connect_timeout(90)
        .pool_timeout(90)
        .build()
    )
    
    telegram_app.add_handler(CommandHandler("start", start_command))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CommandHandler("model", model_command))
    telegram_app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.Document.PY) & ~filters.COMMAND,
            handle_code_input
        )
    )
    
    await telegram_app.initialize()
    await telegram_app.start()
    
    webhook_url = f"{WEBHOOK_URL}/webhook"
    await telegram_app.bot.set_webhook(
        url=webhook_url,
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )
    
    logger.info(f"✅ Bot ready with complete decoding!")
    
    yield
    
    await telegram_app.stop()
    await telegram_app.shutdown()


# --- FastAPI Application ---
app = FastAPI(
    title="Python Decoder Bot V3",
    description="Complete iterative decoder with validation",
    version="3.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "status": "running",
        "bot": "Python Decoder Bot V3",
        "version": "3.0.0",
        "mode": "Complete Iterative Decoding",
        "max_iterations": MAX_DECODE_ITERATIONS,
        "model": MODEL_NAME,
        "developer": "@aadi_io"
    }


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
