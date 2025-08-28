import os
import io
from dotenv import load_dotenv
import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import BufferedInputFile
from aiogram.filters import Command
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Bot and Dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# ---------------------------------------------------------------------

# --- Gemini API interaction function ---
async def decode_with_gemini(code_or_text):
    """Sends code to Gemini for decoding and returns the result."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # The prompt is crucial here. Be specific.
        prompt = (
            "I have an encoded, obfuscated, or encrypted Python script. "
            "Your task is to decode it and return the original, "
            "readable Python code. The encoding method is unknown but may include "
            "base64, marshal, exec, pyarmor, or nested variations. "
            "Do not add any explanations or comments, just the clean, decoded code. "
            "If the code is already clean, return it as is. "
            "If you cannot decode it, return a single phrase: 'Decoding failed.'\n\n"
            "Here is the code to decode:\n\n"
            f"```python\n{code_or_text}\n```"
        )
        
        response = model.generate_content(prompt)
        
        # Check if response is valid and get the text
        if response and response.text:
            cleaned_text = response.text.strip().replace("```python", "").replace("```", "").strip()
            if "Decoding failed." in cleaned_text:
                return None
            return cleaned_text
        return None
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

# ---------------------------------------------------------------------

# --- Telegram Bot Handlers ---

@dp.message(Command('start'))
async def start_command(message: types.Message):
    """Handles the /start command."""
    await message.answer(
        "Welcome! 👋\n\n"
        "I can decode obfuscated, encrypted, or encoded Python code and files. "
        "Just send me a Python file or paste your code directly into the chat.\n\n"
        "Powered by Google's Gemini AI.\n"
        "Created by @aadi.io"
    )

@dp.message(F.document)
async def handle_document(message: types.Message):
    """Handles uploaded files."""
    if message.document.mime_type in ['text/x-python', 'text/plain']:
        # Download the file to an in-memory buffer
        file_id = message.document.file_id
        file_info = await bot.get_file(file_id)
        file_path = file_info.file_path
        
        # Create an in-memory buffer to store the file content
        file_buffer = io.BytesIO()
        await bot.download_file(file_path, file_buffer)
        
        # Decode content and send to Gemini
        code_to_decode = file_buffer.getvalue().decode('utf-8')
        await message.answer("✅ File received. Analyzing and decoding...")
        
        decoded_code = await decode_with_gemini(code_to_decode)
        
        if decoded_code:
            # If the decoded code is large, send as a file
            if len(decoded_code) > 4096: # Telegram message size limit
                decoded_file = BufferedInputFile(decoded_code.encode('utf-8'), filename="decoded_script.py")
                await message.reply_document(
                    document=decoded_file,
                    caption="✅ File decoded successfully.\n\nBot by @aadi.io"
                )
            else:
                # Otherwise, send as a text message
                await message.reply(
                    f"✅ Code decoded successfully.\n\n```python\n{decoded_code}\n```\n\nBot by @aadi.io",
                    parse_mode='Markdown'
                )
        else:
            await message.answer("❌ Could not decode the file. The code might be too complex or already clean.")
    else:
        await message.answer("Please send a valid Python (.py) or text (.txt) file.")

@dp.message()
async def handle_text(message: types.Message):
    """Handles pasted code."""
    code_to_decode = message.text
    if len(code_to_decode) > 20: # Simple check to avoid decoding short phrases
        await message.answer("✅ Code received. Analyzing and decoding...")
        decoded_code = await decode_with_gemini(code_to_decode)
        
        if decoded_code:
            if len(decoded_code) > 4096:
                 decoded_file = BufferedInputFile(decoded_code.encode('utf-8'), filename="decoded_script.py")
                 await message.reply_document(
                    document=decoded_file,
                    caption="✅ Code decoded successfully.\n\nBot by @aadi.io"
                )
            else:
                await message.reply(
                    f"✅ Code decoded successfully.\n\n```python\n{decoded_code}\n```\n\nBot by @aadi.io",
                    parse_mode='Markdown'
                )
        else:
            await message.answer("❌ Could not decode the code. The code might be too complex or already clean.")
    else:
        await message.answer("Please send a valid code snippet or file.")

# ---------------------------------------------------------------------

# --- Main function to run the bot ---
async def main() -> None:
    """Entry point of the bot."""
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
