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
        # Ensure it's a Python file
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
        # Ignore other message types
        return

    # Acknowledge receipt and notify the user that processing has started
    await update.message.reply_text("Decoding your code... this might take a moment.")

    prompt = f"Decode this Python obfuscated code and return only the clean, runnable Python code. No explanation. Just the code:\n\n{code}"
    
    decoded_code = ""
    try:
        # Use the asynchronous version of the Gemini API call
        response = await model.generate_content_async(prompt)
        decoded_code = response.text
    except Exception as e:
        print(f"Error generating content from Gemini: {e}")
        decoded_code = "# Decoding failed due to an API error."
        await update.message.reply_text("Sorry, I encountered an error trying to decode the code.")

    # If decoding fails or returns empty, stop here
    if not decoded_code or decoded_code.startswith("# Decoding failed"):
        return

    filename = "decoded_by_aadi.py"
    # Write the decoded code to a file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(decoded_code)
    
    # Send the file back to the user
    with open(filename, "rb") as f:
        await update.message.reply_document(
            document=InputFile(f, filename=filename),
            caption=CREDIT
        )
    
    # Clean up the created file
    os.remove(filename)

# --- Main Application Logic ---

async def main():
    """Sets up and runs the Telegram bot with a webhook."""
    
    # --- IMPORTANT NOTE ON THE ERROR ---
    # The error log shows an "AttributeError" during the build step.
    # This is a known bug in `python-telegram-bot` version 20.7.
    # The best solution is to update the library by setting this in your requirements.txt:
    # python-telegram-bot>=20.8
    #
    # The code below is the corrected way to run an async bot, which fixes
    # another potential issue in your original script.

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler((filters.TEXT | filters.Document.PY) & ~filters.COMMAND, handle_input))
    
    # Get port from environment variables, with a fallback
    port = int(os.environ.get("PORT", 8443))
    
    # The run_webhook method is a coroutine and must be awaited
    await app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=TELEGRAM_BOT_TOKEN,
        webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}"
    )

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
