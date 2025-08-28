# main.py

import asyncio
import logging
import base64
import urllib.parse
import html
import codecs
import os

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# --- Configuration ---
# IMPORTANT: It's best practice to set this as an environment variable on your hosting service.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")

# --- Logging Setup ---
# This sets up basic logging to see the bot's activity and any potential errors in your console.
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# --- Decoding Functions ---
# Each function tries a specific decoding method.
# If it fails, it returns the original text, allowing the process to continue.

def decode_base64(text: str) -> str:
    """Tries to decode a Base64 encoded string."""
    try:
        # We need to add padding if it's missing for valid Base64.
        missing_padding = len(text) % 4
        if missing_padding:
            text += '=' * (4 - missing_padding)
        decoded_bytes = base64.b64decode(text)
        return decoded_bytes.decode('utf-8')
    except (UnicodeDecodeError, base64.binascii.Error, ValueError):
        return text

def decode_url(text: str) -> str:
    """Tries to decode a URL-encoded (percent-encoded) string."""
    try:
        decoded_text = urllib.parse.unquote(text)
        # Only return if it actually changed something
        return decoded_text if decoded_text != text else text
    except Exception:
        return text

def decode_html_entities(text: str) -> str:
    """Tries to decode HTML entities (e.g., &amp; -> &)."""
    try:
        decoded_text = html.unescape(text)
        return decoded_text if decoded_text != text else text
    except Exception:
        return text

def decode_rot13(text: str) -> str:
    """Decodes a ROT13 ciphered string."""
    try:
        return codecs.decode(text, 'rot_13')
    except Exception:
        return text

def decode_hex(text: str) -> str:
    """Tries to decode a hex string into text."""
    try:
        # Remove common hex prefixes if they exist
        if text.lower().startswith('0x'):
            text = text[2:]
        
        # Ensure it's a valid hex string (even number of chars, only hex digits)
        if len(text) % 2 == 0 and all(c in '0123456789abcdefABCDEF' for c in text):
             return bytes.fromhex(text).decode('utf-8')
        return text
    except (ValueError, UnicodeDecodeError):
        return text

# --- Core Decoding Logic ---

async def decode_aggressively(original_text: str) -> (str, list):
    """
    Continuously applies all decoding methods until the text stops changing.
    This handles multi-layered encoding.
    """
    current_text = original_text
    decoding_steps = []
    
    # A list of all decoding functions to try, in order.
    decoders = {
        "Base64": decode_base64,
        "URL Encoding": decode_url,
        "Hex": decode_hex,
        "HTML Entities": decode_html_entities,
        "ROT13 Cipher": decode_rot13,
    }

    # Loop until a full pass over the text results in no changes.
    while True:
        text_before_pass = current_text
        for name, decoder_func in decoders.items():
            decoded_text = decoder_func(current_text)
            # If the text was changed by the decoder, we've made progress.
            if decoded_text != current_text:
                decoding_steps.append(f"✅ Decoded with: <b>{name}</b>")
                current_text = decoded_text
                # Break the inner loop and start a new pass from the beginning
                # with the newly decoded text.
                break 
        
        # If the inner loop completed without any changes, we are done.
        if text_before_pass == current_text:
            break
            
    return current_text, decoding_steps


# --- Telegram Bot Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    welcome_message = (
        f"👋 Hello {user.mention_html()}!\n\n"
        "I am the Multi-Decoder Bot. Send me any encoded text, and I will do my best to decode it for you.\n\n"
        "I can handle multiple layers of encoding like Base64, URL, Hex, and more. Just paste the text and send!"
    )
    await update.message.reply_html(welcome_message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The main function to handle incoming text messages."""
    input_text = update.message.text
    
    logger.info(f"Received message from {update.effective_user.username}: {input_text}")
    
    # Show a "typing..." status to the user.
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    final_text, steps = await decode_aggressively(input_text)
    
    if not steps:
        response_message = (
            "🤔 <b>No Encoding Detected</b>\n\n"
            "I tried several methods, but the text doesn't seem to be encoded in a way I recognize, or it's already decoded.\n\n"
            f"<b>Original Text:</b>\n<pre>{html.escape(input_text)}</pre>"
        )
    else:
        steps_text = "\n".join(steps)
        response_message = (
            "🎉 <b>Decoding Complete!</b>\n\n"
            "<b>Decoding Path:</b>\n"
            f"{steps_text}\n\n"
            "<b>Final Decoded Text:</b>\n"
            f"<pre>{html.escape(final_text)}</pre>"
        )
        
    await update.message.reply_html(response_message)


async def main() -> None:
    """Starts the bot using webhooks for deployment."""
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("FATAL: TELEGRAM_BOT_TOKEN is not set. Please set it as an environment variable.")
        return

    # The PORT is usually assigned by the hosting service. Default to 8443 for local testing.
    PORT = int(os.environ.get('PORT', '8443'))
    
    # This is the public URL of your deployed application provided by Render.
    # e.g., https://your-app-name.onrender.com
    WEBHOOK_URL = os.environ.get('WEBHOOK_URL')

    if not WEBHOOK_URL:
        logger.error("FATAL: WEBHOOK_URL environment variable not set. This is required for webhook deployment.")
        return

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Set up the webhook by telling Telegram where to send updates.
    # The URL path should be secret; the bot token is a good choice.
    webhook_full_url = f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}"
    await application.bot.set_webhook(url=webhook_full_url)
    
    logger.info(f"Webhook set to {webhook_full_url}")
    logger.info(f"Bot is starting web server on port {PORT}...")

    # Run the bot as a web server.
    # It will listen on 0.0.0.0 to accept connections from the hosting service.
    await application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=TELEGRAM_BOT_TOKEN
    )


if __name__ == '__main__':
    # Use asyncio.run to execute the async main function.
    asyncio.run(main())
