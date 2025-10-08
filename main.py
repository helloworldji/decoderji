import os
import asyncio
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai

# --- Environment Variables ---
# Ensure these are set in your deployment environment
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")

# --- Initialize Gemini AI ---
# Check if the API key exists before configuring
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set!")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Bot Constants ---
CREDIT = "Dev: @aadi_io"

# --- Command Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a welcome message when the /start command is issued."""
    about = (
        "🤖 Python Decoder Bot\n\n"
        "Send me any encoded, obfuscated, or encrypted Python file/code.\n"
        "I will decode it and return a clean, readable Python file.\n\n"
        "Supports:\n"
        "• Base64\n"
        "• Exec-wrapped code\n"
        "• Multi-layered obfuscation\n"
        "• Encoded strings & payloads\n\n"
        f"{CREDIT}"
    )
    await update.message.reply_text(about)

# --- Message Handler ---

async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles incoming Python files or code snippets for decoding."""
    code = ""
    # Check for a document
    if update.message.document:
        if not update.message.document.file_name.endswith('.py'):
            await update.message.reply_text("Please send a Python file (.py) or a text message with Python code.")
            return
        
        file = await update.message.document.get_file()
        content = await file.download_as_bytearray()
        code = content.decode('utf-8', errors='ignore')
    # Check for text message
    elif update.message.text:
        code = update.message.text
    else:
        return

    await update.message.reply_text("Decoding your code... this might take a moment.")

    prompt = f"Decode this Python obfuscated code and return only the clean, runnable Python code. No explanation. Just the code:\n\n{code}"
    
    decoded_code = ""
    try:
        response = await model.generate_content_async(prompt)
        decoded_code = response.text
    except Exception as e:
        print(f"Error generating content from Gemini: {e}")
        decoded_code = "# Decoding failed due to an API error."
        await update.message.reply_text("Sorry, I encountered an error trying to decode the code.")

    if not decoded_code or decoded_code.startswith("# Decoding failed"):
        return

    filename = "decoded_by_aadi.py"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(decoded_code)
    
    with open(filename, "rb") as f:
        await update.message.reply_document(
            document=InputFile(f, filename=filename),
            caption=CREDIT
        )
    
    os.remove(filename)

# --- ASGI Application Setup ---

async def post_init(application: Application) -> None:
    """This function is called after the application is initialized.
    It sets the webhook for the bot."""
    if not WEBHOOK_URL:
        raise ValueError("WEBHOOK_URL environment variable not set!")
    await application.bot.set_webhook(url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}")

# Build the application
# The `app` object is created at the top level so the ASGI server can find it.
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set!")

application = (
    Application.builder()
    .token(TELEGRAM_BOT_TOKEN)
    .post_init(post_init)
    .build()
)

# Register handlers
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler((filters.TEXT | filters.Document.PY) & ~filters.COMMAND, handle_input))

# Note: There is no `if __name__ == "__main__":` block to run the bot.
# The ASGI server (Uvicorn) will import the `application` object and run it.

