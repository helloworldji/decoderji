import os
import logging
import asyncio # For sleep/retry logic
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, # New: To handle text messages
    ContextTypes, 
    filters # New: To filter text messages
)
import httpx # New: To make asynchronous API calls

# --- SENSITIVE CONFIGURATION (HARDCODED AS REQUESTED) ---
# ⚠️ WARNING: Hardcoding credentials is a security risk. Use environment variables (os.environ) 
# instead for production environments.

# Hardcoded Gemini API Key (from user's previous input)
GEMINI_API_KEY = "AIzaSyB2Y61nqiU1Yjypw084LSljBypqgxsFz80" 
# Hardcoded Telegram Bot Token (***REPLACE THIS PLACEHOLDER***)
TELEGRAM_BOT_TOKEN = "8243612478:AAHkGrv1tKroi_TqxPoaWoc8iDP33YPZ5mo"

# Model and URL configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

# The user explicitly provided this URL in the query. We will use it for the webhook setup.
WEBHOOK_BASE_URL = "https://decoderji.onrender.com"
# ---------------------

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# --- GEMINI API FUNCTION ---
async def get_gemini_response(prompt: str) -> str:
    """Sends the user prompt to the Gemini API and returns the response text, including sources."""
    
    # 1. Define the API request payload with Google Search grounding
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {} }],
    }
    
    # 2. Add API key to the URL
    api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    
    # 3. Make the API call asynchronously using httpx with exponential backoff
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(3):
                response = await client.post(
                    api_url_with_key, 
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                # Success
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for valid candidate data
                    if not (data.get('candidates') and 
                            data['candidates'][0].get('content') and 
                            data['candidates'][0]['content'].get('parts')):
                        logging.error(f"Gemini response missing content: {data}")
                        return "❌ Error: AI service returned an invalid response structure."

                    # Extract the generated text
                    text = data['candidates'][0]['content']['parts'][0]['text']
                    
                    # Optional: Extract grounding sources (citations)
                    sources_text = ""
                    grounding = data['candidates'][0].get('groundingMetadata', {})
                    if grounding.get('groundingAttributions'):
                        sources = [
                            f"[{attr['web']['title']}]({attr['web']['uri']})" 
                            for attr in grounding['groundingAttributions']
                            if attr.get('web') and attr['web'].get('uri') and attr['web'].get('title')
                        ]
                        if sources:
                            sources_text = "\n\n**Sources:**\n" + "\n".join(sources)
                    
                    # Telegram supports up to 4096 characters per message. Truncate if needed.
                    max_len = 4096
                    full_response = text + sources_text
                    if len(full_response) > max_len:
                        # Truncate text but ensure sources are included if possible
                        text = text[:max_len - len(sources_text) - 5] + "..." 
                        full_response = text + sources_text
                        
                    return full_response
                
                # Handle throttling (429) or other retryable errors
                elif response.status_code == 429 and attempt < 2:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                    logging.warning(f"Rate limited (429). Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time) 
                else:
                    logging.error(f"Gemini API Error {response.status_code}: {response.text}")
                    return f"❌ Error: Failed to get response from AI. Status code: {response.status_code}"

        return "❌ Error: AI service failed after multiple retries."
        
    except httpx.HTTPError as e:
        logging.error(f"HTTP Request failed: {e}")
        return "❌ Error: Could not connect to the AI service."

# --- HANDLERS ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the command /start is issued."""
    welcome_message = (
        "Hello! I am a Telegram bot powered by the **Gemini API** with Google Search grounding. "
        "Send me any text message, and I will generate a grounded response. "
        "I am currently running using a **Webhook** setup on Render."
    )
    await update.message.reply_markdown(welcome_message) # Using markdown for bold text

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Processes user text messages and sends the Gemini response."""
    if not update.message or not update.message.text:
        return
        
    user_prompt = update.message.text
    
    # Show typing indicator while processing
    await update.message.chat.send_action('typing')

    logging.info(f"User prompt: {user_prompt}")
    
    # Get response from Gemini
    gemini_text = await get_gemini_response(user_prompt)
    
    # Send the response back to the user
    await update.message.reply_markdown(gemini_text) # Using markdown to render sources links and bold text

# --- MAIN EXECUTION ---

def main() -> None:
    """Start the bot."""
    # Note: load_dotenv() is still called for other potential environment variables (like PORT)
    load_dotenv()
    
    # 1. Use the hardcoded token
    token = TELEGRAM_BOT_TOKEN
    
    # Check if the user updated the placeholder
    if token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        logging.error("TELEGRAM_BOT_TOKEN has not been updated. Please replace the placeholder in main.py.")
        return

    # Render provides PORT and WEBHOOK_URL
    port = int(os.environ.get("PORT", 8080))
    # We prioritize the environment variable, but default to the URL you provided
    webhook_url = os.environ.get("WEBHOOK_URL", WEBHOOK_BASE_URL) 

    # 2. CREATE THE APPLICATION INSTANCE
    application = Application.builder().token(token).build()

    # 3. REGISTER HANDLERS
    application.add_handler(CommandHandler("start", start_command))
    # Handler for all incoming text messages that are NOT commands
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 4. START THE BOT (Web service requires Webhook setup)
    if webhook_url:
        # Use the bot token as a unique and secure path for the webhook URL
        url_path = token
        
        logging.info(f"Starting bot using webhook at {webhook_url} on port {port}...")
        
        application.run_webhook(
            listen="0.0.0.0", # Listen on all interfaces
            port=port,
            url_path=url_path,
            webhook_url=f"{webhook_url}{url_path}", # Full URL for Telegram to call
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
