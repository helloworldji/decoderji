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
        
        if available:
            return available[0]
    except Exception as e:
        logger.warning(f"⚠️ Model listing failed: {e}")
    
    return 'models/gemini-pro'

MODEL_NAME = get_best_model()
model = genai.GenerativeModel(MODEL_NAME)

# --- Constants ---
CREDIT = "Dev: @aadi_io"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CODE_LENGTH = 100000  # 100K chars
DECODER_TIMEOUT = 60
MAX_DECODE_ITERATIONS = 50  # Very high limit for complex files

# --- Global Bot Application ---
telegram_app: Optional[Application] = None


# --- Helper Functions ---
def is_code_obfuscated(code: str) -> bool:
    """Check if code still contains obfuscation."""
    if not code or len(code) < 10:
        return True
    
    obfuscation_patterns = [
        r'exec\s*KATEX_INLINE_OPEN',
        r'eval\s*KATEX_INLINE_OPEN',
        r'compile\s*KATEX_INLINE_OPEN',
        r'__import__\s*KATEX_INLINE_OPEN[\'"]base64',
        r'__import__\s*KATEX_INLINE_OPEN[\'"]marshal',
        r'base64\.b64decode',
        r'base64\.b32decode',
        r'base64\.b16decode',
        r'codecs\.decode',
        r'marshal\.loads',
        r'pickle\.loads',
        r'zlib\.decompress',
        r'gzip\.decompress',
        r'bz2\.decompress',
        r'lambda\s+.*:\s*exec',
        r'lambda\s+.*:\s*eval',
        r'bytes\.fromhex',
        r'bytearray\.fromhex',
        r'intKATEX_INLINE_OPEN.*,\s*16KATEX_INLINE_CLOSE',
    ]
    
    for pattern in obfuscation_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            logger.info(f"Obfuscation detected: {pattern}")
            return True
    
    # Check for base64-like strings
    base64_matches = re.findall(r"['\"]([A-Za-z0-9+/=]{100,})['\"]", code)
    if base64_matches:
        logger.info(f"Found {len(base64_matches)} base64-like strings")
        return True
    
    # Check for excessive string operations
    if code.count("chr(") > 20 or code.count("ord(") > 20:
        logger.info("Found excessive chr/ord usage")
        return True
    
    logger.info("✅ Code appears clean")
    return False


def clean_code_response(text: str) -> str:
    """Clean AI response."""
    text = text.strip()
    if text.startswith("```python"):
        text = text[9:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def extract_response_text(response) -> tuple[bool, str]:
    """Extract text from Gemini response."""
    try:
        if not response or not hasattr(response, 'candidates') or not response.candidates:
            return False, "No response"
        
        candidate = response.candidates[0]
        
        if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                text_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        text_parts.append(part.text)
                if text_parts:
                    return True, ''.join(text_parts)
        
        return False, "No text in response"
    except Exception as e:
        return False, str(e)


async def generate_universal_decoder(code: str) -> tuple[bool, str]:
    """Generate a powerful universal decoder."""
    try:
        logger.info("🔧 Generating universal decoder...")
        
        code_sample = code[:20000] if len(code) > 20000 else code
        
        prompt = (
            "Create a POWERFUL universal Python decoder that handles ALL obfuscation types.\n\n"
            
            "REQUIREMENTS:\n"
            "- Decode UNLIMITED layers automatically\n"
            "- Handle: base64, hex, marshal, zlib, gzip, bz2, exec, eval, compile, pickle\n"
            "- Loop until NO obfuscation remains\n"
            "- Return ONLY clean readable code\n\n"
            
            "DECODER STRUCTURE:\n"
            "```python\n"
            "import base64, marshal, zlib, gzip, bz2, pickle, codecs, re, binascii\n\n"
            
            "def is_obfuscated(s):\n"
            "    patterns = ['exec(', 'eval(', 'base64.', 'marshal.', 'compile(']\n"
            "    return any(p in str(s) for p in patterns)\n\n"
            
            "def decode_all_layers(code):\n"
            "    iteration = 0\n"
            "    max_iter = 100\n"
            "    \n"
            "    while iteration < max_iter:\n"
            "        if not is_obfuscated(code):\n"
            "            break\n"
            "        \n"
            "        # Try all decoding methods\n"
            "        # base64, hex, marshal, zlib, exec extraction, etc.\n"
            "        \n"
            "        iteration += 1\n"
            "    \n"
            "    return code\n\n"
            
            "with open('obfuscated_code.txt', 'r') as f:\n"
            "    original = f.read()\n\n"
            
            "decoded = decode_all_layers(original)\n\n"
            
            "with open('decoded_code.py', 'w') as f:\n"
            "    f.write(decoded)\n"
            "```\n\n"
            
            f"OBFUSCATED CODE:\n{code_sample}\n\n"
            "Generate the complete universal decoder:"
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
        
        success, result = extract_response_text(response)
        
        if not success:
            return False, result
        
        decoder = clean_code_response(result)
        
        if len(decoder) < 50:
            return False, "Decoder too short"
        
        logger.info(f"✅ Generated decoder ({len(decoder)} chars)")
        return True, decoder
        
    except Exception as e:
        logger.error(f"❌ Decoder generation error: {e}")
        return False, str(e)


async def execute_decoder(decoder_script: str, obfuscated_code: str) -> tuple[bool, str]:
    """Execute decoder script."""
    try:
        logger.info("⚙️ Executing decoder...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write files
            with open(os.path.join(temp_dir, "obfuscated_code.txt"), 'w', encoding='utf-8') as f:
                f.write(obfuscated_code)
            
            with open(os.path.join(temp_dir, "decoder.py"), 'w', encoding='utf-8') as f:
                f.write(decoder_script)
            
            decoded_file = os.path.join(temp_dir, "decoded_code.py")
            
            # Execute
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                os.path.join(temp_dir, "decoder.py"),
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=DECODER_TIMEOUT
                )
            except asyncio.TimeoutError:
                process.kill()
                return False, "Timeout"
            
            # Read result
            if os.path.exists(decoded_file):
                with open(decoded_file, 'r', encoding='utf-8') as f:
                    decoded = f.read()
                if decoded.strip():
                    logger.info(f"✅ Decoded: {len(decoded)} chars")
                    return True, decoded
            
            if stdout:
                decoded = stdout.decode('utf-8', errors='ignore')
                if len(decoded) > 20:
                    return True, decoded
            
            if stderr:
                error = stderr.decode('utf-8', errors='ignore')
                logger.error(f"Decoder error: {error[:300]}")
                return False, error[:200]
            
            return False, "No output produced"
            
    except Exception as e:
        logger.error(f"❌ Execution error: {e}")
        return False, str(e)


async def full_decode(code: str, chat_id: int, message_id: int) -> tuple[bool, str, int]:
    """Complete decoding with unlimited iterations."""
    current_code = code
    total_iterations = 0
    
    for iteration in range(1, MAX_DECODE_ITERATIONS + 1):
        logger.info(f"🔄 Iteration {iteration}/{MAX_DECODE_ITERATIONS}")
        
        # Update user every 5 iterations
        if iteration % 5 == 0:
            try:
                await telegram_app.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=f"🔄 <b>Decoding in progress...</b>\n\nLayer {iteration} decoded\nPlease wait...",
                    parse_mode="HTML"
                )
            except:
                pass
        
        # Check if clean
        if not is_code_obfuscated(current_code):
            logger.info(f"✅ Fully decoded after {iteration} iterations")
            return True, current_code, iteration
        
        # Generate decoder
        success, decoder = await generate_universal_decoder(current_code)
        if not success:
            if iteration > 1:
                logger.warning(f"Decoder gen failed at iteration {iteration}, returning current state")
                return True, current_code, iteration
            return False, f"Decoder generation failed: {decoder}", 0
        
        # Execute decoder
        success, decoded = await execute_decoder(decoder, current_code)
        if not success:
            if iteration > 1:
                logger.warning(f"Execution failed at iteration {iteration}, returning current state")
                return True, current_code, iteration
            return False, f"Execution failed: {decoded}", 0
        
        # Check progress
        if len(decoded) < 10:
            return True, current_code, iteration
        
        if decoded == current_code:
            logger.info(f"No change at iteration {iteration}, stopping")
            return True, current_code, iteration
        
        current_code = decoded
        total_iterations = iteration
        
        # Small delay to prevent API rate limits
        if iteration % 3 == 0:
            await asyncio.sleep(2)
    
    logger.info(f"Max iterations ({MAX_DECODE_ITERATIONS}) reached")
    return True, current_code, total_iterations


# --- Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Brief welcome message."""
    await update.message.reply_text(
        "🤖 <b>Python Decoder Bot</b>\n\n"
        "Send obfuscated .py file or code.\n"
        "I'll decode all layers automatically.\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help information."""
    await update.message.reply_text(
        "<b>Python Decoder Bot</b>\n\n"
        "📤 Send: .py file or code text\n"
        "⏳ Wait: 1-10 minutes (complex files)\n"
        "📥 Receive: Fully decoded file\n\n"
        "Features:\n"
        "• Unlimited layers\n"
        "• Auto-detection\n"
        "• Background processing\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Model info."""
    await update.message.reply_text(
        f"<b>AI Model</b>\n\n"
        f"Model: {MODEL_NAME.split('/')[-1]}\n"
        f"Max Layers: {MAX_DECODE_ITERATIONS}\n"
        f"Status: Active\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


# --- Background Processing Task ---
async def process_decoding_task(chat_id: int, code: str, filename: str, status_msg_id: int):
    """Background task for decoding."""
    try:
        logger.info(f"🚀 Starting background decode for chat {chat_id}")
        
        # Check if already clean
        if not is_code_obfuscated(code):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
                tmp.write(code)
                tmp_name = tmp.name
            
            try:
                with open(tmp_name, 'rb') as f:
                    await telegram_app.bot.send_document(
                        chat_id=chat_id,
                        document=InputFile(f, filename=filename),
                        caption="✅ Code is already clean!\n\n" + CREDIT,
                        parse_mode="HTML"
                    )
                await telegram_app.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            finally:
                os.remove(tmp_name)
            return
        
        # Perform full decode
        success, decoded_code, iterations = await full_decode(code, chat_id, status_msg_id)
        
        if not success:
            await telegram_app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg_id,
                text=f"❌ <b>Decoding Failed</b>\n\n{decoded_code[:300]}\n\nContact {CREDIT}",
                parse_mode="HTML"
            )
            return
        
        # Check final status
        still_obfuscated = is_code_obfuscated(decoded_code)
        status = "✅ Fully Decoded" if not still_obfuscated else "⚠️ Partially Decoded"
        
        # Send result
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp.write(decoded_code)
            tmp_name = tmp.name
        
        try:
            with open(tmp_name, 'rb') as f:
                await telegram_app.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(f, filename=f"decoded_{filename}"),
                    caption=(
                        f"{status}\n\n"
                        f"📄 {filename}\n"
                        f"📊 {len(decoded_code):,} chars\n"
                        f"🔄 {iterations} layers decoded\n\n"
                        f"{CREDIT}"
                    ),
                    parse_mode="HTML"
                )
            
            # Send preview
            preview = decoded_code[:3800] if len(decoded_code) <= 4000 else decoded_code[:3800] + "\n\n..."
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text=f"<b>Preview:</b>\n\n<pre>{preview}</pre>",
                parse_mode="HTML"
            )
            
            # Delete status message
            await telegram_app.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            
        finally:
            os.remove(tmp_name)
        
        logger.info(f"✅ Completed decode for chat {chat_id}")
        
    except Exception as e:
        logger.error(f"❌ Background task error: {e}")
        logger.error(traceback.format_exc())
        try:
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Error during decoding\n\n<code>{str(e)[:300]}</code>",
                parse_mode="HTML"
            )
        except:
            pass


# --- Message Handler ---
async def handle_code_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming code - start background processing."""
    code = ""
    filename = "code.py"
    
    try:
        # Extract code
        if update.message.document:
            doc = update.message.document
            
            if not doc.file_name.endswith('.py'):
                await update.message.reply_text("⚠️ Send .py file only")
                return
            
            if doc.file_size > MAX_FILE_SIZE:
                await update.message.reply_text(
                    f"⚠️ File too large\nMax: {MAX_FILE_SIZE//(1024*1024)}MB"
                )
                return
            
            filename = doc.file_name
            file = await doc.get_file()
            file_bytes = await file.download_as_bytearray()
            code = file_bytes.decode('utf-8', errors='ignore')
            
        elif update.message.text:
            code = update.message.text
            
            if len(code) > MAX_CODE_LENGTH:
                await update.message.reply_text(
                    f"⚠️ Code too long\nMax: {MAX_CODE_LENGTH:,} chars"
                )
                return
        else:
            return

        if not code.strip():
            await update.message.reply_text("⚠️ Empty code")
            return

        # Send initial status
        status_msg = await update.message.reply_text(
            "🚀 <b>Decoding Started</b>\n\n"
            "⏳ <b>This may take 1-10 minutes</b>\n\n"
            "You can:\n"
            "• Leave this chat\n"
            "• Use other apps\n"
            "• Close Telegram\n\n"
            "I'll send the decoded file automatically when done!\n\n"
            "<i>Processing in background...</i>",
            parse_mode="HTML"
        )
        
        # Start background task
        chat_id = update.effective_chat.id
        asyncio.create_task(
            process_decoding_task(chat_id, code, filename, status_msg.message_id)
        )
        
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:200]}")


# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage bot lifecycle."""
    global telegram_app
    
    logger.info("🚀 Starting bot...")
    
    telegram_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(180)
        .write_timeout(180)
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
    
    logger.info(f"✅ Bot ready!")
    
    yield
    
    await telegram_app.stop()
    await telegram_app.shutdown()


# --- FastAPI Application ---
app = FastAPI(
    title="Python Decoder Bot",
    description="Unlimited layer decoder with background processing",
    version="4.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "status": "running",
        "bot": "Python Decoder Bot",
        "version": "4.0.0",
        "max_layers": MAX_DECODE_ITERATIONS,
        "model": MODEL_NAME
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
