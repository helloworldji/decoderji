import os
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
WEBHOOK_URL = os.environ["WEBHOOK_URL"]

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

CREDIT = "Dev: @aadi_io"

async def start(update: Update, context):
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

async def handle_input(update: Update, context):
    code = ""
    if update.message.document:
        if not update.message.document.file_name.endswith('.py'):
            return
        file = await update.message.document.get_file()
        content = await file.download_as_bytearray()
        code = content.decode('utf-8', errors='ignore')
    elif update.message.text:
        code = update.message.text
    else:
        return

    prompt = f"Decode this Python obfuscated code and return only the clean, runnable Python code. No explanation. Just the code:\n\n{code}"
    
    response = model.generate_content(prompt)
    decoded_code = response.text if response.text else "# Decoding failed"

    filename = "decoded_by_aadi.py"
    with open(filename, "w") as f:
        f.write(decoded_code)
    
    with open(filename, "rb") as f:
        await update.message.reply_document(
            document=InputFile(f, filename=filename),
            caption=CREDIT
        )
    
    os.remove(filename)

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler((filters.TEXT | filters.Document.PY) & ~filters.COMMAND, handle_input))
    
    port = int(os.environ.get("PORT", 8443))
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=TELEGRAM_BOT_TOKEN,
        webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}"
    )

if __name__ == "__main__":
    main()
