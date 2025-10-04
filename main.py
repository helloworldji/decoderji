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
    await update.message.reply_text('Hello! Welcome to the bot. Using the Webhook setup now.')

# Define the main function to run the bot
def main() -> None:
    """Start the bot."""
    # Load environment variables (like BOT_TOKEN) from .env file
    load_dotenv()
    
    # 1. Get required environment variables
    token = os.environ.get("BOT_TOKEN")
    
    # Render provides PORT and WEBHOOK_URL
    port = int(os.environ.get("PORT", 8080))
    webhook_url = os.environ.get("WEBHOOK_URL")

    if not token:
        logging.error("BOT_TOKEN not found in environment variables. Cannot start the bot.")
        return

    # 2. CREATE THE APPLICATION INSTANCE (replaces Updater)
    application = Application.builder().token(token).build()

    # 3. REGISTER HANDLERS
    application.add_handler(CommandHandler("start", start_command))

    # 4. START THE BOT (Web service requires Webhook setup)
    if webhook_url:
        logging.info(f"Starting bot using webhook at {webhook_url} on port {port}...")
        
        # The url_path should be unique and secure. Using the token is a common practice.
        url_path = token
        
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=url_path,
            webhook_url=f"{webhook_url}{url_path}",
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        # Fallback to polling for local development if WEBHOOK_URL is not set
        logging.info("WEBHOOK_URL not set. Falling back to local polling...")
        try:
            application.run_polling(allowed_updates=Update.ALL_TYPES)
        except Exception as e:
            logging.error(f"Failed to start local polling: {e}")


if __name__ == "__main__":
    main()
