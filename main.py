import os
import tempfile
import logging
import asyncio
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
                logger.info(f"  ✓ Found: {m.name}")
        
        # Preferred models - fast and reliable
        preferred = [
            'models/gemini-2.5-flash-lite',
            'models/gemini-flash-lite-latest',
            'models/gemini-2.0-flash-lite-preview',
            'models/gemini-flash-latest',
            'models/gemini-1.5-flash',
            'models/gemini-pro',
        ]
        
        for pref in preferred:
            if pref in available:
                logger.info(f"✅ Selected: {pref}")
                return pref
        
        if available:
            logger.info(f"✅ Using: {available[0]}")
            return available[0]
            
    except Exception as e:
        logger.warning(f"⚠️ Could not list models: {e}")
    
    return 'models/gemini-pro'

MODEL_NAME = get_best_model()
model = genai.GenerativeModel(MODEL_NAME)
logger.info(f"🤖 Initialized: {MODEL_NAME}")

# --- Constants ---
CREDIT = "🔧 Dev: @aadi_io"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_CODE_LENGTH = 30000  # 30K characters

# --- Global Bot Application ---
telegram_app: Optional[Application] = None


# --- Helper Functions ---
def clean_code_response(text: str) -> str:
    """Clean up AI response to get pure Python code."""
    text = text.strip()
    
    # Remove markdown code blocks
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
        # Check if response exists
        if not response:
            return False, "No response from API"
        
        # Check for candidates
        if not hasattr(response, 'candidates') or not response.candidates:
            return False, "No response candidates"
        
        # Get first candidate
        candidate = response.candidates[0]
        
        # Check finish reason
        finish_reason = candidate.finish_reason
        
        # Map finish reasons
        finish_reasons = {
            0: "UNSPECIFIED",
            1: "STOP",
            2: "MAX_TOKENS",
            3: "SAFETY",
            4: "RECITATION",
            5: "OTHER"
        }
        
        reason_name = finish_reasons.get(finish_reason, f"UNKNOWN({finish_reason})")
        logger.info(f"Response finish_reason: {reason_name}")
        
        # Try to extract text from parts
        if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                text_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        text_parts.append(part.text)
                
                if text_parts:
                    full_text = ''.join(text_parts)
                    
                    # Check if we got truncated (MAX_TOKENS)
                    if finish_reason == 2:
                        logger.warning("Response was truncated (MAX_TOKENS)")
                        return True, full_text  # Still return what we got
                    
                    # Check safety block
                    if finish_reason == 3:
                        return False, "Response blocked by safety filters"
                    
                    # Normal completion
                    if finish_reason == 1:
                        return True, full_text
                    
                    # Other reasons - try to use text anyway
                    if full_text.strip():
                        return True, full_text
        
        # Check for safety ratings (blocked content)
        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
            blocked = [r for r in candidate.safety_ratings if hasattr(r, 'blocked') and r.blocked]
            if blocked:
                return False, f"Content blocked by safety filters: {reason_name}"
        
        return False, f"Could not extract text (finish_reason: {reason_name})"
        
    except Exception as e:
        logger.error(f"Error extracting response: {e}")
        return False, str(e)


async def decode_with_gemini(code: str, attempt: int = 1) -> tuple[bool, str]:
    """Decode obfuscated code using Gemini AI."""
    try:
        logger.info(f"🔄 Decoding attempt #{attempt}")
        
        # Truncate if needed
        code_to_send = code
        if len(code) > MAX_CODE_LENGTH:
            code_to_send = code[:MAX_CODE_LENGTH]
            logger.warning(f"Code truncated from {len(code)} to {MAX_CODE_LENGTH} chars")
        
        # Simple, clear prompt - fixed f-string syntax
        prompt = (
            "Deobfuscate and decode this Python code. "
            "Return ONLY the clean, readable Python code without any explanations or markdown.\n\n"
            f"Obfuscated code:\n{code_to_send}\n\n"
            "Clean Python code:"
        )

        # Proper safety settings using enums
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Generation config
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.8,
            top_k=40,
            max_output_tokens=16384,
            candidate_count=1,
        )
        
        # Run in executor
        loop = asyncio.get_event_loop()
        
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        )
        
        # Extract text safely
        success, result = extract_response_text(response)
        
        if not success:
            logger.error(f"❌ Failed to extract text: {result}")
            return False, result
        
        # Clean the code
        decoded = clean_code_response(result)
        
        if len(decoded) < 10:
            logger.error(f"❌ Decoded code too short: {len(decoded)} chars")
            return False, "Decoded output is too short or empty"
        
        logger.info(f"✅ Successfully decoded {len(decoded)} characters")
        return True, decoded
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Gemini decode error: {error_msg}")
        logger.error(traceback.format_exc())
        return False, error_msg


# --- Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message."""
    welcome_text = (
        "🤖 <b>Python Decoder Bot</b>\n\n"
        "🔓 Send me obfuscated/encoded Python code and I'll decode it!\n\n"
        "<b>✨ Supported Formats:</b>\n"
        "✅ Base64 encoding\n"
        "✅ Exec/Eval wrappers\n"
        "✅ Marshal serialization\n"
        "✅ Zlib compression\n"
        "✅ Multi-layer obfuscation\n"
        "✅ String encryption\n\n"
        "<b>📤 Usage:</b>\n"
        "• Send .py file as document\n"
        "• Or paste code as text\n\n"
        "<b>⚡ Features:</b>\n"
        "• AI-powered decoding\n"
        "• Auto retry (3 attempts)\n"
        "• Fast processing\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(welcome_text, parse_mode="HTML")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help information."""
    help_text = (
        "<b>📖 Help Guide</b>\n\n"
        "<b>Commands:</b>\n"
        "/start - Start the bot\n"
        "/help - Show help\n"
        "/model - Show AI model info\n\n"
        "<b>How to Use:</b>\n"
        "1️⃣ Send .py file or paste code\n"
        "2️⃣ Wait 15-60 seconds\n"
        "3️⃣ Receive decoded file\n\n"
        "<b>⚠️ Limits:</b>\n"
        "• Max file size: 5 MB\n"
        "• Max code: 30,000 chars\n"
        "• 3 auto retries\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show model info."""
    info_text = (
        f"<b>🤖 AI Model Info</b>\n\n"
        f"<b>Model:</b> <code>{MODEL_NAME.split('/')[-1]}</code>\n"
        f"<b>Status:</b> ✅ Active\n"
        f"<b>Max tokens:</b> 16,384\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(info_text, parse_mode="HTML")


# --- Message Handler ---
async def handle_code_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming code for decoding."""
    code = ""
    filename_original = "code.py"
    processing_msg = None

    try:
        # === Extract Code ===
        if update.message.document:
            doc = update.message.document
            
            if not doc.file_name.endswith('.py'):
                await update.message.reply_text(
                    "⚠️ <b>Invalid File</b>\n\nPlease send a .py file",
                    parse_mode="HTML"
                )
                return
            
            if doc.file_size > MAX_FILE_SIZE:
                await update.message.reply_text(
                    f"⚠️ <b>File Too Large</b>\n\n"
                    f"Max: {MAX_FILE_SIZE // (1024*1024)} MB\n"
                    f"Yours: {doc.file_size / (1024*1024):.2f} MB",
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
                    f"⚠️ <b>Code Too Long</b>\n\n"
                    f"Max: {MAX_CODE_LENGTH:,} chars\n"
                    f"Yours: {len(code):,} chars",
                    parse_mode="HTML"
                )
                return
        else:
            return

        if not code.strip():
            await update.message.reply_text(
                "⚠️ Empty code received",
                parse_mode="HTML"
            )
            return

        # === Process ===
        if not processing_msg:
            processing_msg = await update.message.reply_text(
                "🔄 <b>Decoding...</b>\nAttempt 1/3",
                parse_mode="HTML"
            )
        else:
            await processing_msg.edit_text(
                "🔄 <b>Decoding...</b>\nAttempt 1/3",
                parse_mode="HTML"
            )

        # === Decode with Retry ===
        max_attempts = 3
        success = False
        decoded_code = None
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            try:
                await processing_msg.edit_text(
                    f"🔄 <b>Decoding...</b>\n"
                    f"Attempt {attempt}/{max_attempts}\n"
                    f"Size: {len(code):,} chars",
                    parse_mode="HTML"
                )
                
                success, result = await decode_with_gemini(code, attempt)
                
                if success:
                    decoded_code = result
                    logger.info(f"✅ Success on attempt {attempt}")
                    break
                else:
                    last_error = result
                    logger.warning(f"⚠️ Attempt {attempt} failed: {result}")
                    
                    if attempt < max_attempts:
                        wait_time = 2 ** attempt
                        await processing_msg.edit_text(
                            f"⚠️ <b>Attempt {attempt} failed</b>\n\n"
                            f"<i>{result[:150]}</i>\n\n"
                            f"Retrying in {wait_time}s...",
                            parse_mode="HTML"
                        )
                        await asyncio.sleep(wait_time)
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"❌ Exception on attempt {attempt}: {e}")
                
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** attempt)
                continue

        # === Handle Result ===
        if not success or not decoded_code:
            await processing_msg.edit_text(
                f"❌ <b>Decoding Failed</b>\n\n"
                f"All {max_attempts} attempts failed.\n\n"
                f"<b>Error:</b>\n<code>{last_error[:300]}</code>\n\n"
                f"<b>Try:</b>\n"
                f"• Smaller code snippet\n"
                f"• Valid Python syntax\n"
                f"• Different file\n\n"
                f"Contact {CREDIT}",
                parse_mode="HTML"
            )
            return

        # === Send File ===
        await processing_msg.edit_text(
            "✅ <b>Complete!</b>\n📤 Sending...",
            parse_mode="HTML"
        )

        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False, 
            encoding='utf-8'
        ) as tmp_file:
            tmp_file.write(decoded_code)
            tmp_filename = tmp_file.name

        try:
            with open(tmp_filename, 'rb') as f:
                await update.message.reply_document(
                    document=InputFile(f, filename=f"decoded_{filename_original}"),
                    caption=(
                        f"✅ <b>Decoded Successfully!</b>\n\n"
                        f"📄 Original: {filename_original}\n"
                        f"📊 Size: {len(decoded_code):,} chars\n"
                        f"🤖 Model: {MODEL_NAME.split('/')[-1]}\n\n"
                        f"{CREDIT}"
                    ),
                    parse_mode="HTML"
                )
            
            # Send preview
            preview_len = min(4000, len(decoded_code))
            preview = decoded_code[:preview_len]
            if len(decoded_code) > 4000:
                preview += "\n\n... (see file for complete code)"
            
            await update.message.reply_text(
                f"<b>📝 Preview:</b>\n\n<pre>{preview}</pre>",
                parse_mode="HTML"
            )
            
            await processing_msg.delete()

        finally:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        
        try:
            error_text = str(e)[:400]
            if processing_msg:
                await processing_msg.edit_text(
                    f"❌ <b>Error</b>\n\n<code>{error_text}</code>\n\n{CREDIT}",
                    parse_mode="HTML"
                )
            else:
                await update.message.reply_text(
                    f"❌ <b>Error</b>\n\n<code>{error_text}</code>",
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
        .read_timeout(90)
        .write_timeout(90)
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
    
    logger.info(f"✅ Webhook: {webhook_url}")
    logger.info(f"✅ Model: {MODEL_NAME}")
    logger.info("✅ Bot ready!")
    
    yield
    
    logger.info("🛑 Shutting down...")
    await telegram_app.stop()
    await telegram_app.shutdown()


# --- FastAPI Application ---
app = FastAPI(
    title="Python Decoder Bot",
    description="AI-powered Python deobfuscator",
    version="3.2.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "status": "running",
        "bot": "Python Decoder Bot",
        "version": "3.2.0",
        "model": MODEL_NAME,
        "developer": "@aadi_io"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME}


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
