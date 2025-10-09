import os
import sys
import tempfile
import logging
import asyncio
import base64
import zlib
import re
import zipfile
import io
import html
import subprocess
import json
from contextlib import asynccontextmanager
from typing import Optional, Tuple

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
PORT = int(os.environ.get("PORT", 10000))

if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, WEBHOOK_URL]):
    raise ValueError("❌ Missing environment variables!")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-1.5-flash')

CREDIT = "Dev: @aadi_io"
MAX_FILE_SIZE = 50 * 1024 * 1024
telegram_app: Optional[Application] = None


class DeobfuscationEngine:
    """Aggressive deobfuscation engine focused on extracting actual code."""
    
    @staticmethod
    def extract_base64_strings(code: str) -> list:
        """Extract all potential base64 strings."""
        patterns = [
            r'["\']([A-Za-z0-9+/=]{100,})["\']',
            r'=\s*["\']([A-Za-z0-9+/=]{100,})["\']',
            r':\s*["\']([A-Za-z0-9+/=]{100,})["\']',
            r'\(["\']([A-Za-z0-9+/=]{100,})["\']',
            r'\[\s*["\']([A-Za-z0-9+/=]{100,})["\']',
        ]
        
        candidates = []
        for pattern in patterns:
            candidates.extend(re.findall(pattern, code))
        
        return list(set(candidates))
    
    @staticmethod
    def try_decode_base64(encoded: str) -> Optional[str]:
        """Try to decode base64 string."""
        try:
            decoded = base64.b64decode(encoded)
            
            # Try UTF-8
            try:
                text = decoded.decode('utf-8')
                if len(text) > 50:
                    return text
            except:
                pass
            
            # Try as ZIP
            if decoded[:2] == b'PK':
                return DeobfuscationEngine.extract_zip(decoded)
            
            # Try zlib
            try:
                decompressed = zlib.decompress(decoded)
                text = decompressed.decode('utf-8', errors='ignore')
                if len(text) > 50:
                    return text
            except:
                pass
        
        except:
            pass
        
        return None
    
    @staticmethod
    def extract_zip(data: bytes) -> Optional[str]:
        """Extract all files from ZIP."""
        try:
            files_content = []
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in sorted(zf.namelist()):
                    try:
                        content = zf.read(name).decode('utf-8', errors='ignore')
                        files_content.append(content)
                    except:
                        pass
            
            if files_content:
                return "\n\n".join(files_content)
        except:
            pass
        
        return None
    
    @staticmethod
    def find_exec_eval_code(code: str) -> Optional[str]:
        """Find code inside exec() or eval() calls."""
        patterns = [
            r'exec\s*\(\s*["\']([^"\']+)["\']\s*\)',
            r'eval\s*\(\s*["\']([^"\']+)["\']\s*\)',
            r'exec\s*\(\s*b?["\']([^"\']+)["\']\s*\)',
            r'__import__\(["\']([^"\']+)["\']\)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code, re.DOTALL)
            if matches:
                result = "\n".join(matches)
                if len(result) > 50:
                    return result
        
        return None
    
    @staticmethod
    def decode_string_chains(code: str) -> Optional[str]:
        """Decode chained string operations like 'a' + 'b' + 'c'."""
        # Look for string concatenation patterns
        pattern = r'(["\'][^"\']*["\'](?:\s*\+\s*["\'][^"\']*["\'])+)'
        matches = re.findall(pattern, code)
        
        for match in matches:
            try:
                # Evaluate the string concatenation safely
                result = eval(match)
                if isinstance(result, str) and len(result) > 100:
                    if any(x in result for x in ['def ', 'class ', 'import ', 'return ']):
                        return result
            except:
                pass
        
        return None
    
    @staticmethod
    async def aggressive_gemini_decode(code: str) -> Optional[str]:
        """Use Gemini aggressively to decode."""
        try:
            sample = code[:25000] if len(code) > 25000 else code
            
            prompt = f"""You are a code deobfuscator. Your ONLY job is to extract and return the actual source code.

IMPORTANT: Return ONLY the decoded Python source code. Nothing else.

The code you receive may be:
1. Base64 encoded - decode it
2. Zlib/gzip compressed - decompress it
3. Inside ZIP files - extract all files
4. Inside exec/eval calls - extract the string
5. String concatenation - combine strings
6. Nested encoding - decode recursively

Return the CLEAN, READABLE Python source code.

Obfuscated code to decode:
{sample}

---
DECODED SOURCE CODE (ONLY CODE, NO EXPLANATIONS):
"""
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=50000,
                    ),
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                )
            )
            
            if response and response.text:
                result = response.text.strip()
                result = re.sub(r'^```(?:python)?\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
                result = result.strip()
                
                if len(result) > 100 and result != code:
                    return result
        
        except Exception as e:
            logger.error(f"Gemini error: {e}")
        
        return None
    
    @staticmethod
    async def deobfuscate(code: str) -> Tuple[str, str]:
        """Main deobfuscation pipeline."""
        logger.info(f"Starting deobfuscation. Code length: {len(code)}")
        
        # Step 1: Find exec/eval code
        logger.info("Checking for exec/eval patterns...")
        result = DeobfuscationEngine.find_exec_eval_code(code)
        if result:
            logger.info(f"✅ Found exec/eval code: {len(result)} chars")
            return result, "Exec/Eval Extraction"
        
        # Step 2: Try string chain decoding
        logger.info("Checking for string concatenation...")
        result = DeobfuscationEngine.decode_string_chains(code)
        if result:
            logger.info(f"✅ Decoded string chains: {len(result)} chars")
            return result, "String Chain Decoding"
        
        # Step 3: Extract and decode all base64 strings
        logger.info("Extracting base64 strings...")
        b64_strings = DeobfuscationEngine.extract_base64_strings(code)
        logger.info(f"Found {len(b64_strings)} potential base64 strings")
        
        for b64 in sorted(b64_strings, key=len, reverse=True):
            result = DeobfuscationEngine.try_decode_base64(b64)
            if result and len(result) > 100 and result != code:
                logger.info(f"✅ Decoded base64: {len(result)} chars")
                # Recursively deobfuscate the result
                if any(x in result for x in ['base64', 'zlib', 'exec', 'eval']):
                    deeper, method = await DeobfuscationEngine.deobfuscate(result)
                    if deeper != result:
                        return deeper, f"{method} (Recursive)"
                return result, "Base64 Decoding"
        
        # Step 4: Use Gemini for complex cases
        logger.info("Using AI for complex deobfuscation...")
        result = await DeobfuscationEngine.aggressive_gemini_decode(code)
        if result and result != code:
            logger.info(f"✅ AI decoded: {len(result)} chars")
            return result, "AI Deobfuscation"
        
        logger.warning("Could not deobfuscate - returning original")
        return code, "Unable to deobfuscate (original code)"


async def process_decode(chat_id: int, code: str, filename: str, msg_id: int):
    """Process decoding task."""
    try:
        logger.info(f"Processing decode for chat {chat_id}")
        
        # Update status
        try:
            await telegram_app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text="🔍 <b>Analyzing code...</b>\n\nDetecting obfuscation patterns...",
                parse_mode="HTML"
            )
        except:
            pass
        
        # Deobfuscate
        decoded, method = await DeobfuscationEngine.deobfuscate(code)
        
        is_decoded = decoded != code
        status = "✅ <b>Successfully Deobfuscated</b>" if is_decoded else "⚠️ <b>No Obfuscation Detected</b>"
        
        # Save to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name
        
        try:
            # Send decoded file
            with open(tmp_path, 'rb') as f:
                await telegram_app.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(f, filename=f"decoded_{filename}"),
                    caption=(
                        f"{status}\n\n"
                        f"📄 <b>Original:</b> {filename}\n"
                        f"📊 <b>Size:</b> {len(code):,} → {len(decoded):,} chars\n"
                        f"🔧 <b>Method:</b> {method}\n"
                        f"⚡ <b>Reduction:</b> {round((1 - len(decoded)/len(code)) * 100)}%\n\n"
                        f"<i>{CREDIT}</i>"
                    ),
                    parse_mode="HTML"
                )
            
            # Send preview
            preview = decoded[:3000]
            if len(decoded) > 3000:
                preview += "\n\n... [truncated - see file for full code]"
            
            preview_safe = html.escape(preview)
            
            try:
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text=f"<b>📋 Code Preview:</b>\n\n<pre>{preview_safe}</pre>",
                    parse_mode="HTML"
                )
            except:
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text=f"<b>Preview:</b>\n\n{preview[:1500]}"
                )
            
            # Delete processing message
            try:
                await telegram_app.bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except:
                pass
        
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass
        
        logger.info("✅ Decode completed successfully")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        try:
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text=f"❌ <b>Error:</b>\n\n{html.escape(str(e)[:200])}"
            )
        except:
            pass


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 <b>Python Deobfuscator Bot</b>\n\n"
        "Send any obfuscated .py file to decode it!\n\n"
        "<b>Handles:</b>\n"
        "✓ Base64 encoded code\n"
        "✓ ZIP compressed archives\n"
        "✓ Zlib/gzip compression\n"
        "✓ exec/eval patterns\n"
        "✓ String concatenation\n"
        "✓ Complex nested obfuscation\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "<b>How to use:</b>\n\n"
        "1️⃣ Send a .py file (or paste code)\n"
        "2️⃣ Wait for analysis\n"
        "3️⃣ Get decoded file + preview\n\n"
        "<b>Deobfuscation methods:</b>\n"
        "• Pattern recognition\n"
        "• Base64 decoding\n"
        "• Archive extraction\n"
        "• String parsing\n"
        "• AI-powered analysis\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        code = ""
        filename = "code.py"
        
        if update.message.document:
            doc = update.message.document
            if not doc.file_name.endswith('.py'):
                await update.message.reply_text("⚠️ Please send a .py file")
                return
            if doc.file_size > MAX_FILE_SIZE:
                await update.message.reply_text(f"⚠️ File too large (max {MAX_FILE_SIZE // (1024*1024)}MB)")
                return
            
            filename = doc.file_name
            file = await doc.get_file()
            file_data = await file.download_as_bytearray()
            code = file_data.decode('utf-8', errors='ignore')
        
        elif update.message.text:
            code = update.message.text
        else:
            return
        
        if not code.strip():
            return
        
        msg = await update.message.reply_text(
            "⏳ <b>Starting deobfuscation...</b>\n\n"
            "Analyzing your code...",
            parse_mode="HTML"
        )
        
        asyncio.create_task(process_decode(update.effective_chat.id, code, filename, msg.message_id))
    
    except Exception as e:
        logger.error(f"Handler error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global telegram_app
    
    logger.info("🚀 Starting Deobfuscator Bot...")
    
    telegram_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(120)
        .write_timeout(120)
        .build()
    )
    
    telegram_app.add_handler(CommandHandler("start", start_cmd))
    telegram_app.add_handler(CommandHandler("help", help_cmd))
    telegram_app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.Document.PY) & ~filters.COMMAND,
            handle_message
        )
    )
    
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.bot.set_webhook(
        url=f"{WEBHOOK_URL}/webhook",
        allowed_updates=Update.ALL_TYPES
    )
    
    logger.info("✅ Bot ready!")
    
    yield
    
    await telegram_app.stop()
    await telegram_app.shutdown()


app = FastAPI(title="Python Deobfuscator", version="3.0.0", lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "running", "version": "3.0.0", "type": "Deobfuscator"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return Response(status_code=500)


@app.head("/")
@app.head("/health")
async def head():
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
