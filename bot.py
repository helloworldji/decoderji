import os
import logging
import tempfile
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Env vars
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_KEY = os.environ["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📤 Send me an encoded Python (.py) file.\n"
        "I'll use AI to decode it and send back `decoded_by_aadi.py`!"
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document

    # Only accept .py files
    if not document.file_name.endswith(".py"):
        await update.message.reply_text("⚠️ Please send a `.py` file only.")
        return

    try:
        # Download file
        file = await context.bot.get_file(document.file_id)
        
        with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        # Read content
        with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        os.unlink(tmp_path)  # Clean up

        if not content.strip():
            await update.message.reply_text("❌ File is empty.")
            return

        # Ask Gemini to decode
        prompt = (
            "You are a Python deobfuscation expert. The following is a Python script that has been encoded "
            "or obfuscated using one or more layers (e.g., base64, eval, exec, rot13, zlib, etc.).\n"
            "Your task:\n"
            "1. Analyze all encoding/obfuscation layers\n"
            "2. Fully decode it to original clean Python source code\n"
            "3. Return ONLY the decoded Python code — no explanations, no markdown, no extra text.\n\n"
            "Encoded content:\n"
            f"```\n{content}\n```"
        )

        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,  # Low temp for accuracy
                max_output_tokens=2048,
                top_p=0.95,
                top_k=40
            )
        )

        decoded_code = response.text.strip() if response.text else ""

        if not decoded_code:
            await update.message.reply_text("❌ Gemini returned empty response.")
            return

        # Remove possible markdown code blocks
        if decoded_code.startswith("```") and decoded_code.endswith("```"):
            decoded_code = "\n".join(decoded_code.split("\n")[1:-1])

        # Save to temp file and send
        output_path = "/tmp/decoded_by_aadi.py"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(decoded_code)

        with open(output_path, "rb") as f:
            await update.message.reply_document(
                document=InputFile(f, filename="decoded_by_aadi.py"),
                caption="✅ Decoded successfully!"
            )

        os.remove(output_path)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ Failed to process the file. Make sure it's a valid encoded .py file."
        )


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    logger.info("Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
