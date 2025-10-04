import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get environment variables
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Validate environment variables
if not TELEGRAM_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the command /start is issued."""
    await update.message.reply_text(
        "Hello! I'm your AI assistant powered by Gemini. "
        "Send me any complex question or request, and I'll provide a detailed response."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process user messages using Gemini API."""
    user_message = update.message.text
    
    try:
        # Generate response with multi-layer processing
        response = await model.generate_content_async(
            f"""
            You are an expert AI assistant. Analyze the following request in multiple layers:
            1. Identify core components and requirements
            2. Break down into logical sub-tasks
            3. Provide comprehensive, step-by-step reasoning
            4. Deliver a clear final answer with relevant details
            
            User request: {user_message}
            """,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
                top_p=0.8,
                top_k=40
            )
        )
        
        # Handle potential empty responses
        if response.text:
            await update.message.reply_text(response.text[:4096])  # Telegram message limit
        else:
            await update.message.reply_text("I couldn't generate a response. Please try rephrasing your request.")
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await update.message.reply_text(
            "Sorry, I encountered an error processing your request. "
            "Please try again later."
        )

def main():
    """Start the bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the Bot
    logger.info("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
