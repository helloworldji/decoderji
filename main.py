import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Load environment variables from a .env file (for local testing only)
# On Render, this file is ignored, and you will set BOT_TOKEN directly 
# in the environment variables (Secrets).
load_dotenv()

# --- Configuration ---
# Your bot token MUST be set as an environment variable named BOT_TOKEN
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    # Exit if the token is not found (required for deployment)
    raise ValueError("Error: BOT_TOKEN environment variable not found. Please set it in Render secrets.")

# Set up logging for easier debugging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# Disable logging for the underlying Telegram library for cleaner output
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- Handlers (Functions that respond to events) ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a greeting message when the /start command is issued."""
    user = update.effective_user
    logger.info(f"Received /start command from user: {user.username or user.id}")
    await update.message.reply_html(
        f"Hello, {user.mention_html()}! I am your new Python Telegram Bot. I can echo any text you send me. Use /help for more info.",
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a help message when the /help command is issued."""
    logger.info(f"Received /help command from user: {update.effective_user.username or update.effective_user.id}")
    await update.message.reply_text("This bot is running in a polling mode on a continuous server (like your setup on Render/UptimeRobot).\n\nAvailable Commands:\n/start - Say hello\n/help - Show this message\n\nJust send me any message, and I'll echo it back!")

async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echoes the user message back to the chat."""
    # Check if the update contains a message and text
    if update.message and update.message.text:
        text = update.message.text
        logger.info(f"Received message from {update.effective_user.username or update.effective_user.id}: {text}")
        await update.message.reply_text(f"You said: {text}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.warning(f'Update "{update}" caused error "{context.error}"')

# --- Main Bot Logic ---

def main() -> None:
    """Start the bot using Long Polling, suitable for a continuous service like Render."""

    # 1. Create the Application and pass it your bot's token.
    application = Application.builder().token(BOT_TOKEN).build()

    # 2. Register Handlers (Order matters for some handlers, but not these)
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    
    # MessageHandler handles all text messages that are NOT commands
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_message))

    # Error handler
    application.add_error_handler(error_handler)

    # 3. Start Polling
    # This keeps the application running 24/7, constantly checking for new messages.
    # This is the ideal "Start Command" for a continuous worker on Render.
    print("Bot starting... Listening for Telegram updates via long polling.")
    
    # The application.run_polling() is a synchronous blocking call that runs the bot until manually stopped.
    # This is the ideal "Start Command" for a continuous worker on Render.
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Failed to start the bot: {e}")
