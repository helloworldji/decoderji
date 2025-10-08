import os
import tempfile
import logging
from contextlib import asynccontextmanager
from typing import Optional

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
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # e.g., https://yourapp.onrender.com
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
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Constants ---
CREDIT = "🔧 Dev: @aadi_io"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CODE_LENGTH = 50000  # characters

# --- Global Bot Application ---
telegram_app: Optional[Application] = None


# --- Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message with bot information."""
    welcome_text = (
        "🤖 <b>Python Decoder Bot</b>\n\n"
        "Send me any obfuscated or encoded Python file/code and I'll decode it for you!\n\n"
        "<b>✨ Supported Formats:</b>\n"
        "• Base64 encoded code\n"
        "• Exec/eval wrapped code\n"
        "• Multi-layer obfuscation\n"
        "• Encrypted strings & payloads\n"
        "• Marshal/zlib compressed code\n\n"
        "<b>📤 How to use:</b>\n"
        "1. Send a .py file (as document)\n"
        "2. Or paste the code directly as text\n\n"
        "<b>⚡ Features:</b>\n"
        "• AI-powered decoding\n"
        "• Clean, readable output\n"
        "• Instant processing\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(welcome_text, parse_mode="HTML")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Provides help information."""
    help_text = (
        "<b>📖 Help Guide</b>\n\n"
        "<b>Commands:</b>\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        "<b>Usage:</b>\n"
        "Simply send me a Python file or paste obfuscated code.\n\n"
        "<b>⚠️ Limitations:</b>\n"
        "• Max file size: 10MB\n"
        "• Max code length: 50,000 characters\n"
        "• Processing time: ~10-30 seconds\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")


# --- Message Handler ---
async def handle_code_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles incoming Python files or code snippets for decoding."""
    code = ""
    filename_original = "code.py"

    try:
        # Handle document (file)
        if update.message.document:
            doc = update.message.document
            
            # Validate file type
            if not doc.file_name.endswith('.py'):
                await update.message.reply_text(
                    "⚠️ Please send a Python file (.py) or paste code as text."
                )
                return
            
            # Validate file size
            if doc.file_size > MAX_FILE_SIZE:
                await update.message.reply_text(
                    f"⚠️ File too large! Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
                )
                return
            
            filename_original = doc.file_name
            
            # Download and read file
            file = await doc.get_file()
            file_bytes = await file.download_as_bytearray()
            code = file_bytes.decode('utf-8', errors='ignore')
            
        # Handle text message
        elif update.message.text:
            code = update.message.text
            
            # Validate code length
            if len(code) > MAX_CODE_LENGTH:
                await update.message.reply_text(
                    f"⚠️ Code too long! Max length: {MAX_CODE_LENGTH} characters"
                )
                return
        else:
            return

        # Check if code is empty
        if not code.strip():
            await update.message.reply_text("⚠️ Empty code received. Please send valid Python code.")
            return

        # Send processing message
        processing_msg = await update.message.reply_text(
            "🔄 Decoding your code...\n⏳ This may take 10-30 seconds."
        )

        # Prepare prompt for Gemini
        prompt = (
            "You are a Python code deobfuscator. Analyze the following obfuscated/encoded Python code "
            "and return ONLY the clean, deobfuscated, readable Python code. "
            "Do not include any explanations, markdown formatting, or comments about the process. "
            "Just output the pure Python code.\n\n"
            "If the code uses base64, exec, eval, marshal, zlib, or any other obfuscation technique, "
            "decode it completely and return the original source code.\n\n"
            f"Code to decode:\n\n{code}"
        )

        # Call Gemini AI
        try:
            response = await model.generate_content_async(prompt)
            decoded_code = response.text.strip()
            
            # Clean up markdown code blocks if present
            if decoded_code.startswith("```python"):
                decoded_code = decoded_code[9:]
            elif decoded_code.startswith("```"):
                decoded_code = decoded_code[3:]
            if decoded_code.endswith("```"):
                decoded_code = decoded_code[:-3]
            
            decoded_code = decoded_code.strip()

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            await processing_msg.edit_text(
                "❌ Failed to decode the code. The AI service encountered an error.\n"
                "Please try again later or contact support."
            )
            return

        # Validate decoded output
        if not decoded_code or len(decoded_code) < 10:
            await processing_msg.edit_text(
                "⚠️ Decoding failed. The code might be too complex or not actually obfuscated."
            )
            return

        # Create temporary file for decoded code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(decoded_code)
            tmp_filename = tmp_file.name

        try:
            # Send decoded file
            with open(tmp_filename, 'rb') as f:
                await update.message.reply_document(
                    document=InputFile(f, filename=f"decoded_{filename_original}"),
                    caption=f"✅ <b>Decoding Complete!</b>\n\n{CREDIT}",
                    parse_mode="HTML"
                )
            
            # Delete processing message
            await processing_msg.delete()

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

    except Exception as e:
        logger.error(f"Error in handle_code_input: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ An unexpected error occurred. Please try again or contact support."
        )


# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage bot lifecycle."""
    global telegram_app
    
    # Startup
    logger.info("🚀 Starting Telegram Bot...")
    
    # Initialize telegram application
    telegram_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(30)
        .write_timeout(30)
        .build()
    )
    
    # Register handlers
    telegram_app.add_handler(CommandHandler("start", start_command))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.Document.PY) & ~filters.COMMAND,
            handle_code_input
        )
    )
    
    # Initialize the application
    await telegram_app.initialize()
    await telegram_app.start()
    
    # Set webhook
    webhook_url = f"{WEBHOOK_URL}/webhook"
    await telegram_app.bot.set_webhook(
        url=webhook_url,
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )
    
    logger.info(f"✅ Webhook set to: {webhook_url}")
    logger.info("✅ Bot is ready!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down bot...")
    await telegram_app.stop()
    await telegram_app.shutdown()


# --- FastAPI Application ---
app = FastAPI(
    title="Python Decoder Bot",
    description="Telegram bot for decoding obfuscated Python code",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "bot": "Python Decoder Bot",
        "version": "2.0.0",
        "developer": "@aadi_io"
    }


@app.get("/health")
async def health():
    """Health check for monitoring."""
    return {"status": "healthy"}


@app.post("/webhook")
async def webhook(request: Request):
    """Webhook endpoint for receiving Telegram updates."""
    try:
        # Parse incoming update
        json_data = await request.json()
        update = Update.de_json(json_data, telegram_app.bot)
        
        # Process update
        await telegram_app.process_update(update)
        
        return Response(status_code=200)
    
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        return Response(status_code=500)


@app.head("/")
@app.head("/health")
async def head_health():
    """HEAD request for health checks."""
    return Response(status_code=200)


# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )
