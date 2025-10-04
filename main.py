import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Define the async handler for the /start command
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the command /start is issued."""
    # Note: Replace 'Hello! Welcome to the bot.' with your actual welcome message
    await update.message.reply_text('Hello! Welcome to the bot. This is the new v20 structure.')

# Define the main function to run the bot
def main() -> None:
    """Start the bot."""
    # Load environment variables (like BOT_TOKEN) from .env file
    load_dotenv()
    
    # 1. Get the bot token from environment variables
    # Ensure you have BOT_TOKEN set in your environment or a .env file
    token = os.environ.get("BOT_TOKEN")

    if not token:
        logging.error("BOT_TOKEN not found in environment variables. Cannot start the bot.")
        return

    # 2. CREATE THE APPLICATION INSTANCE (replaces Updater)
    application = Application.builder().token(token).build()

    # 3. REGISTER HANDLERS
    # Replaces dispatcher.add_handler()
    application.add_handler(CommandHandler("start", start_command))

    # 4. START THE BOT (using polling for simplicity, replaces updater.start_polling())
    # If you need a webhook for Render, you would use application.run_webhook(...) here instead.
    try:
        logging.info("Starting bot using polling...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        # This will catch the previous error if it was still using the old structure, 
        # but with the new structure, it should handle runtime exceptions.
        logging.error(f"Failed to start the bot: {e}")

if __name__ == "__main__":
    main()
