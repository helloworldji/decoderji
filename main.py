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

# Get available model
def get_best_model():
    """Get the best available Gemini model."""
    try:
        logger.info("🔍 Searching for available Gemini models...")
        
        # Try to list models
        available = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available.append(m.name)
                logger.info(f"  ✓ Found: {m.name}")
        
        # Preferred models in order
        preferred = [
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro', 
            'models/gemini-pro',
        ]
        
        for pref in preferred:
            if pref in available:
                logger.info(f"✅ Selected: {pref}")
                return pref
        
        # Use first available
        if available:
            logger.info(f"✅ Using: {available[0]}")
            return available[0]
            
    except Exception as e:
        logger.warning(f"⚠️ Could not list models: {e}")
    
    # Default fallback
    return 'models/gemini-pro'

MODEL_NAME = get_best_model()
model = genai.GenerativeModel(MODEL_NAME)
logger.info(f"🤖 Initialized: {MODEL_NAME}")

# --- Constants ---
CREDIT = "🔧 Dev: @aadi_io"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CODE_LENGTH = 100000  # Increased to 100K characters

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
    
    # Remove common prefixes
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip explanatory lines
        if line.strip().startswith('#') and any(word in line.lower() for word in ['here', 'decoded', 'result', 'output']):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


async def decode_with_gemini(code: str, attempt: int = 1) -> tuple[bool, str]:
    """
    Decode obfuscated code using Gemini AI.
    Returns: (success: bool, result: str)
    """
    try:
        logger.info(f"🔄 Decoding attempt #{attempt}")
        
        # Enhanced prompt with specific instructions
        prompt = f"""You are an expert Python code deobfuscator and reverse engineer.

Your task: Analyze and decode the following obfuscated/encoded Python code.

INSTRUCTIONS:
1. Identify the obfuscation method (base64, exec, eval, marshal, zlib, etc.)
2. Decode/deobfuscate it completely, layer by layer if needed
3. Return ONLY the clean, readable Python source code
4. Do NOT include any explanations, comments, or markdown
5. Do NOT add your own comments about the decoding process
6. Just output the pure, executable Python code

OBFUSCATED CODE:
{code}

DECODED CODE:"""

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192,
                    candidate_count=1,
                ),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
        )
        
        if not response or not response.text:
            logger.error("❌ Empty response from Gemini")
            return False, "Empty response from AI"
        
        decoded = clean_code_response(response.text)
        
        if len(decoded) < 10:
            logger.error(f"❌ Decoded code too short: {len(decoded)} chars")
            return False, "Decoded code is too short"
        
        logger.info(f"✅ Successfully decoded {len(decoded)} characters")
        return True, decoded
        
    except Exception as e:
        logger.error(f"❌ Gemini decode error: {e}")
        logger.error(traceback.format_exc())
        return False, str(e)


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
        "• Or paste code as text message\n\n"
        "<b>⚡ Features:</b>\n"
        "• AI-powered decoding\n"
        "• Automatic retry (3 attempts)\n"
        "• Clean, formatted output\n"
        "• Fast processing (~15-45 sec)\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(welcome_text, parse_mode="HTML")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help information."""
    help_text = (
        "<b>📖 Help Guide</b>\n\n"
        "<b>Commands:</b>\n"
        "/start - Welcome message\n"
        "/help - This help guide\n"
        "/model - Show current AI model\n\n"
        "<b>How to Use:</b>\n"
        "1️⃣ Send your obfuscated .py file\n"
        "2️⃣ Or paste the code directly\n"
        "3️⃣ Wait 15-45 seconds\n"
        "4️⃣ Receive decoded file\n\n"
        "<b>⚠️ Limits:</b>\n"
        "• Max file size: 10 MB\n"
        "• Max code length: 100,000 chars\n"
        "• Processing time: 15-45 seconds\n"
        "• Auto retry: 3 attempts\n\n"
        "<b>Tips:</b>\n"
        "• Ensure code is actually obfuscated\n"
        "• Larger files take more time\n"
        "• Check file format (.py only)\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show model info."""
    info_text = (
        f"<b>🤖 AI Model Information</b>\n\n"
        f"<b>Current Model:</b>\n"
        f"<code>{MODEL_NAME}</code>\n\n"
        f"<b>Status:</b> ✅ Active\n"
        f"<b>Max Output:</b> 8,192 tokens\n"
        f"<b>Temperature:</b> 0.1 (precise)\n\n"
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
        # === STEP 1: Extract Code ===
        if update.message.document:
            doc = update.message.document
            
            # Validate file type
            if not doc.file_name.endswith('.py'):
                await update.message.reply_text(
                    "⚠️ <b>Invalid File Type</b>\n\n"
                    "Please send a Python file (.py)\n"
                    "or paste code as text.",
                    parse_mode="HTML"
                )
                return
            
            # Validate file size
            if doc.file_size > MAX_FILE_SIZE:
                await update.message.reply_text(
                    f"⚠️ <b>File Too Large</b>\n\n"
                    f"Max size: {MAX_FILE_SIZE // (1024*1024)} MB\n"
                    f"Your file: {doc.file_size / (1024*1024):.2f} MB",
                    parse_mode="HTML"
                )
                return
            
            filename_original = doc.file_name
            
            # Download file
            processing_msg = await update.message.reply_text(
                "📥 <b>Downloading file...</b>",
                parse_mode="HTML"
            )
            
            file = await doc.get_file()
            file_bytes = await file.download_as_bytearray()
            code = file_bytes.decode('utf-8', errors='ignore')
            
        elif update.message.text:
            code = update.message.text
            
            # Validate length
            if len(code) > MAX_CODE_LENGTH:
                await update.message.reply_text(
                    f"⚠️ <b>Code Too Long</b>\n\n"
                    f"Max length: {MAX_CODE_LENGTH:,} characters\n"
                    f"Your code: {len(code):,} characters",
                    parse_mode="HTML"
                )
                return
        else:
            return

        # Check if empty
        if not code.strip():
            await update.message.reply_text(
                "⚠️ <b>Empty Code</b>\n\n"
                "Please send valid Python code.",
                parse_mode="HTML"
            )
            return

        # === STEP 2: Process Code ===
        if not processing_msg:
            processing_msg = await update.message.reply_text(
                "🔄 <b>Processing...</b>\n"
                "⏳ Decoding your code (attempt 1/3)",
                parse_mode="HTML"
            )
        else:
            await processing_msg.edit_text(
                "🔄 <b>Processing...</b>\n"
                "⏳ Decoding your code (attempt 1/3)",
                parse_mode="HTML"
            )

        # === STEP 3: Decode with Retry Logic ===
        max_attempts = 3
        success = False
        decoded_code = None
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            try:
                # Update status
                await processing_msg.edit_text(
                    f"🔄 <b>Processing...</b>\n"
                    f"⏳ Decoding attempt {attempt}/{max_attempts}\n"
                    f"📊 Code size: {len(code):,} chars",
                    parse_mode="HTML"
                )
                
                # Attempt decode
                success, result = await decode_with_gemini(code, attempt)
                
                if success:
                    decoded_code = result
                    logger.info(f"✅ Decode successful on attempt {attempt}")
                    break
                else:
                    last_error = result
                    logger.warning(f"⚠️ Attempt {attempt} failed: {result}")
                    
                    if attempt < max_attempts:
                        # Wait before retry with exponential backoff
                        wait_time = 2 ** attempt
                        await processing_msg.edit_text(
                            f"⚠️ <b>Attempt {attempt} failed</b>\n\n"
                            f"Retrying in {wait_time} seconds...\n"
                            f"(Attempt {attempt + 1}/{max_attempts})",
                            parse_mode="HTML"
                        )
                        await asyncio.sleep(wait_time)
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"❌ Exception on attempt {attempt}: {e}")
                logger.error(traceback.format_exc())
                
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** attempt)
                continue

        # === STEP 4: Handle Result ===
        if not success or not decoded_code:
            await processing_msg.edit_text(
                f"❌ <b>Decoding Failed</b>\n\n"
                f"Attempted {max_attempts} times.\n\n"
                f"<b>Last error:</b>\n<code>{last_error[:300]}</code>\n\n"
                f"<b>Possible reasons:</b>\n"
                f"• Code is not actually obfuscated\n"
                f"• Obfuscation too complex\n"
                f"• Invalid Python syntax\n"
                f"• API quota exceeded\n\n"
                f"Try again or contact {CREDIT}",
                parse_mode="HTML"
            )
            return

        # === STEP 5: Send Decoded File ===
        await processing_msg.edit_text(
            "✅ <b>Decoding Complete!</b>\n"
            "📤 Preparing file...",
            parse_mode="HTML"
        )

        # Create temp file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False, 
            encoding='utf-8'
        ) as tmp_file:
            tmp_file.write(decoded_code)
            tmp_filename = tmp_file.name

        try:
            # Send file
            with open(tmp_filename, 'rb') as f:
                await update.message.reply_document(
                    document=InputFile(f, filename=f"decoded_{filename_original}"),
                    caption=(
                        f"✅ <b>Successfully Decoded!</b>\n\n"
                        f"📄 <b>Original:</b> {filename_original}\n"
                        f"📊 <b>Size:</b> {len(decoded_code):,} chars\n"
                        f"🤖 <b>Model:</b> {MODEL_NAME.split('/')[-1]}\n\n"
                        f"{CREDIT}"
                    ),
                    parse_mode="HTML"
                )
            
            # Also send as text preview (first 4000 chars)
            if len(decoded_code) <= 4000:
                preview = decoded_code
            else:
                preview = decoded_code[:3900] + "\n\n... (truncated, see file)"
            
            await update.message.reply_text(
                f"<b>📝 Code Preview:</b>\n\n"
                f"<pre>{preview}</pre>",
                parse_mode="HTML"
            )
            
            # Delete processing message
            await processing_msg.delete()

        finally:
            # Cleanup
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

    except Exception as e:
        logger.error(f"❌ Fatal error in handle_code_input: {e}")
        logger.error(traceback.format_exc())
        
        try:
            if processing_msg:
                await processing_msg.edit_text(
                    f"❌ <b>Unexpected Error</b>\n\n"
                    f"<code>{str(e)[:500]}</code>\n\n"
                    f"Please try again or contact {CREDIT}",
                    parse_mode="HTML"
                )
            else:
                await update.message.reply_text(
                    f"❌ <b>Unexpected Error</b>\n\n"
                    f"<code>{str(e)[:500]}</code>\n\n"
                    f"Please try again or contact {CREDIT}",
                    parse_mode="HTML"
                )
        except:
            await update.message.reply_text(
                "❌ An unexpected error occurred. Please try again."
            )


# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage bot lifecycle."""
    global telegram_app
    
    # Startup
    logger.info("🚀 Starting Telegram Bot...")
    
    telegram_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(60)
        .write_timeout(60)
        .connect_timeout(60)
        .pool_timeout(60)
        .build()
    )
    
    # Register handlers
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
    
    # Set webhook
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
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    await telegram_app.stop()
    await telegram_app.shutdown()


# --- FastAPI Application ---
app = FastAPI(
    title="Python Decoder Bot",
    description="AI-powered Python code deobfuscator",
    version="3.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "status": "✅ running",
        "bot": "Python Decoder Bot",
        "version": "3.0.0",
        "model": MODEL_NAME,
        "developer": "@aadi_io"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "timestamp": asyncio.get_event_loop().time()
    }


@app.post("/webhook")
async def webhook(request: Request):
    try:
        json_data = await request.json()
        update = Update.de_json(json_data, telegram_app.bot)
        await telegram_app.process_update(update)
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        logger.error(traceback.format_exc())
        return Response(status_code=500)


@app.head("/")
@app.head("/health")
async def head_health():
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
