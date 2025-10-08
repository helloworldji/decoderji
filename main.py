import os
import tempfile
import logging
import asyncio
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import Optional
import traceback

from fastapi import FastAPI, Request, Response
from telegram import Update, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Logging Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Environment Variables ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
PORT = int(os.environ.get("PORT", 10000))

# --- Validation ---
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN environment variable not set!")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY environment variable not set!")
if not WEBHOOK_URL:
    raise ValueError("❌ WEBHOOK_URL environment variable not set!")

# --- Initialize Gemini AI ---
genai.configure(api_key=GEMINI_API_KEY)

def get_best_model():
    """Get the best available Gemini model."""
    try:
        logger.info("🔍 Searching for available Gemini models...")
        
        available = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available.append(m.name)
                logger.info(f"  ✓ Found: {m.name}")
        
        # Preferred models
        preferred = [
            'models/gemini-2.5-flash-lite',
            'models/gemini-flash-lite-latest',
            'models/gemini-flash-latest',
            'models/gemini-pro',
        ]
        
        for pref in preferred:
            if pref in available:
                logger.info(f"✅ Selected: {pref}")
                return pref
        
        if available:
            logger.info(f"✅ Using: {available[0]}")
            return available[0]
            
    except Exception as e:
        logger.warning(f"⚠️ Could not list models: {e}")
    
    return 'models/gemini-pro'

MODEL_NAME = get_best_model()
model = genai.GenerativeModel(MODEL_NAME)
logger.info(f"🤖 Initialized: {MODEL_NAME}")

# --- Constants ---
CREDIT = "🔧 Dev: @aadi_io"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_CODE_LENGTH = 30000  # 30K characters
DECODER_TIMEOUT = 30  # 30 seconds max execution time

# --- Global Bot Application ---
telegram_app: Optional[Application] = None


# --- Helper Functions ---
def clean_code_response(text: str) -> str:
    """Clean up AI response to get pure Python code."""
    text = text.strip()
    
    # Remove markdown code blocks
    if text.startswith("```python"):
        text = text[9:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
    
    return text.strip()


def extract_response_text(response) -> tuple[bool, str]:
    """Safely extract text from Gemini response."""
    try:
        if not response or not hasattr(response, 'candidates') or not response.candidates:
            return False, "No response from API"
        
        candidate = response.candidates[0]
        
        if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                text_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        text_parts.append(part.text)
                
                if text_parts:
                    return True, ''.join(text_parts)
        
        return False, "Could not extract response text"
        
    except Exception as e:
        return False, str(e)


async def generate_decoder_script(obfuscated_code: str, attempt: int = 1) -> tuple[bool, str]:
    """Ask Gemini to create a decoder script for the obfuscated code."""
    try:
        logger.info(f"🔄 Generating decoder script (attempt #{attempt})")
        
        # Truncate if needed
        code_sample = obfuscated_code
        if len(obfuscated_code) > 10000:
            code_sample = obfuscated_code[:10000]
            logger.info(f"Using first 10K chars for analysis")
        
        # Enhanced prompt for decoder generation
        prompt = (
            "You are a Python reverse engineering expert. Analyze this obfuscated Python code "
            "and create a DECODER SCRIPT that will deobfuscate it.\n\n"
            
            "INSTRUCTIONS:\n"
            "1. Identify the obfuscation method (base64, exec, eval, marshal, zlib, etc.)\n"
            "2. Write a complete Python script that will decode the obfuscated code\n"
            "3. The script should read from 'obfuscated_code.txt' and write decoded code to 'decoded_code.py'\n"
            "4. Handle multiple layers of obfuscation if present\n"
            "5. Add error handling in the decoder\n"
            "6. Return ONLY the decoder script code, no explanations\n\n"
            
            "OBFUSCATED CODE SAMPLE:\n"
            f"{code_sample}\n\n"
            
            "DECODER SCRIPT:"
        )

        # Safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Generation config
        generation_config = genai.types.GenerationConfig(
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            max_output_tokens=8192,
        )
        
        # Generate decoder
        loop = asyncio.get_event_loop()
        
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        )
        
        # Extract decoder script
        success, result = extract_response_text(response)
        
        if not success:
            return False, f"Failed to generate decoder: {result}"
        
        decoder_script = clean_code_response(result)
        
        if len(decoder_script) < 20:
            return False, "Generated decoder is too short"
        
        logger.info(f"✅ Generated decoder script ({len(decoder_script)} chars)")
        return True, decoder_script
        
    except Exception as e:
        logger.error(f"❌ Error generating decoder: {e}")
        return False, str(e)


async def execute_decoder(decoder_script: str, obfuscated_code: str) -> tuple[bool, str]:
    """Execute the decoder script safely to decode the obfuscated code."""
    try:
        logger.info("🔧 Executing decoder script...")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write obfuscated code
            obfuscated_file = os.path.join(temp_dir, "obfuscated_code.txt")
            with open(obfuscated_file, 'w', encoding='utf-8') as f:
                f.write(obfuscated_code)
            
            # Write decoder script
            decoder_file = os.path.join(temp_dir, "decoder.py")
            with open(decoder_file, 'w', encoding='utf-8') as f:
                f.write(decoder_script)
            
            # Expected output file
            decoded_file = os.path.join(temp_dir, "decoded_code.py")
            
            # Execute decoder with timeout
            try:
                result = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        sys.executable,
                        decoder_file,
                        cwd=temp_dir,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    ),
                    timeout=DECODER_TIMEOUT
                )
                
                stdout, stderr = await result.communicate()
                
                # Check if decoder created output file
                if os.path.exists(decoded_file):
                    with open(decoded_file, 'r', encoding='utf-8') as f:
                        decoded_code = f.read()
                    
                    if decoded_code.strip():
                        logger.info(f"✅ Decoder executed successfully ({len(decoded_code)} chars)")
                        return True, decoded_code
                
                # If no output file, check stdout
                if stdout:
                    decoded_code = stdout.decode('utf-8', errors='ignore')
                    if len(decoded_code) > 10:
                        logger.info("✅ Got decoded code from stdout")
                        return True, decoded_code
                
                # Log error if any
                if stderr:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    logger.error(f"Decoder stderr: {error_msg}")
                    return False, f"Decoder execution error: {error_msg[:500]}"
                
                return False, "Decoder did not produce output"
                
            except asyncio.TimeoutError:
                return False, "Decoder execution timeout (>30s)"
            
    except Exception as e:
        logger.error(f"❌ Error executing decoder: {e}")
        logger.error(traceback.format_exc())
        return False, str(e)


async def decode_with_ai_decoder(code: str, attempt: int = 1) -> tuple[bool, str, str]:
    """
    Complete decoding process:
    1. Generate decoder script with Gemini
    2. Execute the decoder
    Returns: (success, decoded_code, method_description)
    """
    try:
        # Step 1: Generate decoder script
        success, decoder_script = await generate_decoder_script(code, attempt)
        
        if not success:
            return False, "", f"Decoder generation failed: {decoder_script}"
        
        # Step 2: Execute decoder
        success, decoded_code = await execute_decoder(decoder_script, code)
        
        if not success:
            return False, decoder_script, f"Decoder execution failed: {decoded_code}"
        
        # Step 3: Validate decoded code
        if len(decoded_code) < 10:
            return False, decoder_script, "Decoded code is too short"
        
        # Success!
        method_desc = "AI-generated decoder"
        logger.info(f"✅ Successfully decoded using AI decoder")
        return True, decoded_code, method_desc
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Decoding error: {error_msg}")
        return False, "", error_msg


# --- Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message."""
    welcome_text = (
        "🤖 <b>Python Decoder Bot V2.0</b>\n\n"
        "🔓 <b>NEW: AI-Powered Decoder Generation!</b>\n\n"
        "I don't just decode - I <b>create custom decoders</b> for your code!\n\n"
        "<b>✨ How It Works:</b>\n"
        "1️⃣ AI analyzes your obfuscated code\n"
        "2️⃣ Generates a custom decoder script\n"
        "3️⃣ Executes decoder safely\n"
        "4️⃣ Returns clean code + decoder\n\n"
        "<b>🎯 Supported Methods:</b>\n"
        "✅ Base64, Hex, ROT13\n"
        "✅ Exec/Eval wrappers\n"
        "✅ Marshal/Pickle/Zlib\n"
        "✅ Multi-layer obfuscation\n"
        "✅ Custom encryption\n\n"
        "<b>📤 Usage:</b>\n"
        "Send .py file or paste code\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(welcome_text, parse_mode="HTML")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help information."""
    help_text = (
        "<b>📖 Advanced Decoder Bot</b>\n\n"
        "<b>Commands:</b>\n"
        "/start - Welcome & features\n"
        "/help - This help guide\n"
        "/model - AI model info\n\n"
        "<b>🔬 How It Works:</b>\n\n"
        "<b>Step 1: Analysis</b>\n"
        "AI identifies obfuscation method\n\n"
        "<b>Step 2: Generation</b>\n"
        "Creates custom decoder script\n\n"
        "<b>Step 3: Execution</b>\n"
        "Safely runs decoder (30s max)\n\n"
        "<b>Step 4: Delivery</b>\n"
        "Returns decoded code + decoder\n\n"
        "<b>⚠️ Limits:</b>\n"
        "• Max file: 5 MB\n"
        "• Max code: 30K chars\n"
        "• Execution: 30s timeout\n"
        "• Auto retry: 3 attempts\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show model info."""
    info_text = (
        f"<b>🤖 AI Configuration</b>\n\n"
        f"<b>Model:</b> <code>{MODEL_NAME.split('/')[-1]}</code>\n"
        f"<b>Mode:</b> Decoder Generator\n"
        f"<b>Status:</b> ✅ Active\n"
        f"<b>Max tokens:</b> 8,192\n"
        f"<b>Temperature:</b> 0.2 (precise)\n"
        f"<b>Timeout:</b> 30s per execution\n\n"
        f"<i>{CREDIT}</i>"
    )
    await update.message.reply_text(info_text, parse_mode="HTML")


# --- Message Handler ---
async def handle_code_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming code for decoding."""
    code = ""
    filename_original = "code.py"
    processing_msg = None

    try:
        # === Extract Code ===
        if update.message.document:
            doc = update.message.document
            
            if not doc.file_name.endswith('.py'):
                await update.message.reply_text(
                    "⚠️ <b>Invalid File</b>\n\nPlease send a .py file",
                    parse_mode="HTML"
                )
                return
            
            if doc.file_size > MAX_FILE_SIZE:
                await update.message.reply_text(
                    f"⚠️ <b>File Too Large</b>\n\n"
                    f"Max: {MAX_FILE_SIZE // (1024*1024)} MB\n"
                    f"Yours: {doc.file_size / (1024*1024):.2f} MB",
                    parse_mode="HTML"
                )
                return
            
            filename_original = doc.file_name
            
            processing_msg = await update.message.reply_text(
                "📥 <b>Downloading...</b>",
                parse_mode="HTML"
            )
            
            file = await doc.get_file()
            file_bytes = await file.download_as_bytearray()
            code = file_bytes.decode('utf-8', errors='ignore')
            
        elif update.message.text:
            code = update.message.text
            
            if len(code) > MAX_CODE_LENGTH:
                await update.message.reply_text(
                    f"⚠️ <b>Code Too Long</b>\n\n"
                    f"Max: {MAX_CODE_LENGTH:,} chars\n"
                    f"Yours: {len(code):,} chars",
                    parse_mode="HTML"
                )
                return
        else:
            return

        if not code.strip():
            await update.message.reply_text(
                "⚠️ Empty code received",
                parse_mode="HTML"
            )
            return

        # === Process ===
        if not processing_msg:
            processing_msg = await update.message.reply_text(
                "🔬 <b>Analyzing code...</b>\nStep 1/3: Identifying obfuscation method",
                parse_mode="HTML"
            )
        else:
            await processing_msg.edit_text(
                "🔬 <b>Analyzing code...</b>\nStep 1/3: Identifying obfuscation method",
                parse_mode="HTML"
            )

        # === Decode with AI Decoder ===
        max_attempts = 3
        success = False
        decoded_code = None
        decoder_script = None
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            try:
                await processing_msg.edit_text(
                    f"🔧 <b>Processing...</b>\n"
                    f"Attempt {attempt}/{max_attempts}\n"
                    f"Step 1: Generating decoder...",
                    parse_mode="HTML"
                )
                
                success, result, method = await decode_with_ai_decoder(code, attempt)
                
                if success:
                    decoded_code = result
                    decoder_script = None  # We could save this if needed
                    logger.info(f"✅ Success on attempt {attempt}")
                    break
                else:
                    decoder_script = result  # Contains decoder script if generation succeeded
                    last_error = method
                    logger.warning(f"⚠️ Attempt {attempt} failed: {method}")
                    
                    if attempt < max_attempts:
                        wait_time = 2 ** attempt
                        await processing_msg.edit_text(
                            f"⚠️ <b>Attempt {attempt} failed</b>\n\n"
                            f"<i>{method[:150]}</i>\n\n"
                            f"Retrying in {wait_time}s...",
                            parse_mode="HTML"
                        )
                        await asyncio.sleep(wait_time)
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"❌ Exception on attempt {attempt}: {e}")
                
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** attempt)
                continue

        # === Handle Result ===
        if not success or not decoded_code:
            error_message = (
                f"❌ <b>Decoding Failed</b>\n\n"
                f"All {max_attempts} attempts failed.\n\n"
                f"<b>Error:</b>\n<code>{last_error[:300]}</code>\n\n"
            )
            
            # If we have a decoder script, send it too
            if decoder_script and len(decoder_script) > 20:
                error_message += (
                    f"<b>Generated Decoder:</b>\n"
                    f"Check decoder script below - you can run it manually.\n\n"
                )
            
            error_message += f"Contact {CREDIT}"
            
            await processing_msg.edit_text(error_message, parse_mode="HTML")
            
            # Send decoder script if available
            if decoder_script and len(decoder_script) > 20:
                with tempfile.NamedTemporaryFile(
                    mode='w', 
                    suffix='_decoder.py', 
                    delete=False,
                    encoding='utf-8'
                ) as tmp:
                    tmp.write(decoder_script)
                    tmp_name = tmp.name
                
                try:
                    with open(tmp_name, 'rb') as f:
                        await update.message.reply_document(
                            document=InputFile(f, filename="failed_decoder.py"),
                            caption="⚠️ Generated decoder (execution failed)"
                        )
                finally:
                    os.remove(tmp_name)
            
            return

        # === Send Results ===
        await processing_msg.edit_text(
            "✅ <b>Success!</b>\n📤 Sending results...",
            parse_mode="HTML"
        )

        # Send decoded file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False, 
            encoding='utf-8'
        ) as tmp_file:
            tmp_file.write(decoded_code)
            tmp_filename = tmp_file.name

        try:
            # Send decoded code
            with open(tmp_filename, 'rb') as f:
                await update.message.reply_document(
                    document=InputFile(f, filename=f"decoded_{filename_original}"),
                    caption=(
                        f"✅ <b>Successfully Decoded!</b>\n\n"
                        f"📄 Original: {filename_original}\n"
                        f"📊 Decoded size: {len(decoded_code):,} chars\n"
                        f"🤖 Method: AI Decoder Generation\n"
                        f"🔧 Model: {MODEL_NAME.split('/')[-1]}\n\n"
                        f"{CREDIT}"
                    ),
                    parse_mode="HTML"
                )
            
            # Send preview
            preview_len = min(4000, len(decoded_code))
            preview = decoded_code[:preview_len]
            if len(decoded_code) > 4000:
                preview += "\n\n... (see file for complete code)"
            
            await update.message.reply_text(
                f"<b>📝 Decoded Code Preview:</b>\n\n<pre>{preview}</pre>",
                parse_mode="HTML"
            )
            
            await processing_msg.delete()

        finally:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        
        try:
            error_text = str(e)[:400]
            if processing_msg:
                await processing_msg.edit_text(
                    f"❌ <b>Error</b>\n\n<code>{error_text}</code>\n\n{CREDIT}",
                    parse_mode="HTML"
                )
            else:
                await update.message.reply_text(
                    f"❌ <b>Error</b>\n\n<code>{error_text}</code>",
                    parse_mode="HTML"
                )
        except:
            await update.message.reply_text("❌ An error occurred")


# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage bot lifecycle."""
    global telegram_app
    
    logger.info("🚀 Starting Telegram Bot...")
    
    telegram_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(120)  # Increased for decoder execution
        .write_timeout(120)
        .connect_timeout(90)
        .pool_timeout(90)
        .build()
    )
    
    telegram_app.add_handler(CommandHandler("start", start_command))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CommandHandler("model", model_command))
    telegram_app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.Document.PY) & ~filters.COMMAND,
            handle_code_input
        )
    )
    
    await telegram_app.initialize()
    await telegram_app.start()
    
    webhook_url = f"{WEBHOOK_URL}/webhook"
    await telegram_app.bot.set_webhook(
        url=webhook_url,
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )
    
    logger.info(f"✅ Webhook: {webhook_url}")
    logger.info(f"✅ Model: {MODEL_NAME}")
    logger.info("✅ Bot ready with AI Decoder Generation!")
    
    yield
    
    logger.info("🛑 Shutting down...")
    await telegram_app.stop()
    await telegram_app.shutdown()


# --- FastAPI Application ---
app = FastAPI(
    title="Python Decoder Bot V2",
    description="AI-powered decoder generator",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "status": "running",
        "bot": "Python Decoder Bot V2",
        "version": "2.0.0",
        "mode": "AI Decoder Generation",
        "model": MODEL_NAME,
        "developer": "@aadi_io"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME, "mode": "decoder_generation"}


@app.post("/webhook")
async def webhook(request: Request):
    try:
        json_data = await request.json()
        update = Update.de_json(json_data, telegram_app.bot)
        await telegram_app.process_update(update)
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return Response(status_code=500)


@app.head("/")
@app.head("/health")
async def head_health():
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
