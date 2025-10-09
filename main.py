import os
import sys
import tempfile
import logging
import asyncio
import base64
import zlib
import ast
import re
import zipfile
import io
import html
import subprocess
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
MAX_FILE_SIZE = 20 * 1024 * 1024
telegram_app: Optional[Application] = None


class CodeDecoder:
    """Advanced code decoder with multiple strategies."""
    
    @staticmethod
    def try_unzip(data: bytes) -> Optional[str]:
        """Extract and combine all files from ZIP archive."""
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                contents = []
                for name in sorted(zf.namelist()):
                    try:
                        content = zf.read(name).decode('utf-8', errors='ignore')
                        contents.append(f"# ===== FILE: {name} =====\n{content}\n")
                    except Exception as e:
                        logger.warning(f"Failed to read {name}: {e}")
                
                if contents:
                    return "\n".join(contents)
        except Exception as e:
            logger.debug(f"ZIP extraction failed: {e}")
        
        return None
    
    @staticmethod
    def try_base64_decode(text: str) -> Optional[str]:
        """Attempt base64 decoding with multiple patterns."""
        patterns = [
            r'["\']([A-Za-z0-9+/=]{200,})["\']',
            r'b?["\']([A-Za-z0-9+/=]{200,})["\']',
            r'=\s*["\']([A-Za-z0-9+/=]{200,})["\']',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    decoded = base64.b64decode(match)
                    
                    # Try as UTF-8
                    try:
                        decoded_str = decoded.decode('utf-8')
                        if len(decoded_str) > 100 and any(c in decoded_str for c in ['def ', 'class ', 'import ', 'return ']):
                            return decoded_str
                    except:
                        pass
                    
                    # Try as ZIP
                    if decoded[:2] == b'PK':
                        result = CodeDecoder.try_unzip(decoded)
                        if result:
                            return result
                    
                    # Try zlib compression
                    try:
                        decompressed = zlib.decompress(decoded)
                        decompressed_str = decompressed.decode('utf-8', errors='ignore')
                        if len(decompressed_str) > 100:
                            return decompressed_str
                    except:
                        pass
                
                except Exception as e:
                    logger.debug(f"Base64 decode attempt failed: {e}")
        
        return None
    
    @staticmethod
    def try_ast_extraction(code: str) -> Optional[str]:
        """Extract string literals from AST for exec/eval patterns."""
        try:
            tree = ast.parse(code)
            strings = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ['exec', 'eval']:
                        if node.args and isinstance(node.args[0], ast.Constant):
                            s = node.args[0].value
                            if isinstance(s, str) and len(s) > 50:
                                strings.append(s)
                
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    s = node.value
                    if len(s) > 200 and any(c in s for c in ['def ', 'class ', 'import ', '\n']):
                        strings.append(s)
            
            if strings:
                # Return longest string (likely the actual code)
                longest = max(strings, key=len)
                if len(longest) > 100:
                    return longest
        
        except Exception as e:
            logger.debug(f"AST extraction failed: {e}")
        
        return None
    
    @staticmethod
    async def try_execution_decode(code: str) -> Optional[str]:
        """Execute code in isolated environment to extract decoded content."""
        script = f'''
import sys
import base64
import zlib
import io
import zipfile

code = """{code}"""

def extract():
    ns = {{'__builtins__': {{}}, 'print': lambda *a, **k: None}}
    try:
        exec(code, ns)
        
        for var in ns.values():
            if isinstance(var, str) and len(var) > 100:
                if any(x in var for x in ['def ', 'class ', 'import ', 'return ']):
                    return var
    except:
        pass
    
    return None

result = extract()
if result:
    print(result)
else:
    print(code)
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                script_path = f.name
            
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, script_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=10 * 1024 * 1024
                )
                
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)
                result = stdout.decode('utf-8', errors='ignore').strip()
                
                if result and len(result) > len(code) * 0.7:
                    return result
            
            finally:
                try:
                    os.remove(script_path)
                except:
                    pass
        
        except Exception as e:
            logger.debug(f"Execution decode failed: {e}")
        
        return None
    
    @staticmethod
    async def try_gemini_decode(code: str) -> Optional[str]:
        """Use Gemini to decode complex obfuscation."""
        try:
            sample = code[:20000] if len(code) > 20000 else code
            
            prompt = f"""You are a Python code deobfuscator. Analyze this obfuscated Python code and return ONLY the clean, decoded source code. Nothing else.

If the code contains:
- Base64 strings: decode them
- Compressed data: decompress it  
- ZIP files: extract and combine all files
- exec/eval calls: extract the executed code
- String concatenation: combine and return the result

Return ONLY valid Python code. No explanations, no markdown, no code blocks.

Obfuscated code:
```python
{sample}
```

Decoded code:"""
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=30000,
                    ),
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                )
            )
            
            if response and response.text:
                result = response.text.strip()
                result = re.sub(r'^```python\n?|```$', '', result).strip()
                
                if len(result) > 100 and any(x in result for x in ['def ', 'class ', 'import ', 'return ', 'if ', 'for ']):
                    return result
        
        except Exception as e:
            logger.error(f"Gemini decode error: {e}")
        
        return None
    
    @staticmethod
    async def decode(code: str) -> Tuple[str, str]:
        """Main decoding orchestrator."""
        logger.info(f"Starting decode process. Code length: {len(code)}")
        
        # Method 1: AST extraction
        logger.info("Method 1: AST Extraction...")
        result = CodeDecoder.try_ast_extraction(code)
        if result and len(result) > len(code) * 0.8:
            logger.info(f"✅ AST extraction successful: {len(result)} chars")
            return result, "AST Extraction"
        
        # Method 2: Base64 decoding
        logger.info("Method 2: Base64 Decoding...")
        result = CodeDecoder.try_base64_decode(code)
        if result and len(result) > len(code) * 0.7:
            logger.info(f"✅ Base64 decode successful: {len(result)} chars")
            return result, "Base64 Decoding"
        
        # Method 3: Execution decode
        logger.info("Method 3: Execution Decode...")
        result = await CodeDecoder.try_execution_decode(code)
        if result and len(result) > len(code) * 0.7:
            logger.info(f"✅ Execution decode successful: {len(result)} chars")
            return result, "Execution Decode"
        
        # Method 4: Gemini AI
        logger.info("Method 4: Gemini AI Decode...")
        result = await CodeDecoder.try_gemini_decode(code)
        if result and len(result) > 100:
            logger.info(f"✅ Gemini decode successful: {len(result)} chars")
            return result, "Gemini AI"
        
        logger.warning("Could not decode fully")
        return code, "Original Code"


async def decode_task(chat_id: int, code: str, filename: str, msg_id: int):
    """Background decoding task."""
    try:
        logger.info(f"🚀 Decode task started for chat {chat_id}")
        
        try:
            await telegram_app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text="🔄 <b>Decoding in progress...</b>\n\nAnalyzing code structure...",
                parse_mode="HTML"
            )
        except Exception as e:
            logger.warning(f"Could not edit message: {e}")
        
        decoded, method = await CodeDecoder.decode(code)
        
        is_different = decoded != code and len(decoded) > 50
        status = "✅ <b>Successfully Decoded</b>" if is_different else "ℹ️ <b>Original Code</b>"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                await telegram_app.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(f, filename=f"decoded_{filename}"),
                    caption=(
                        f"{status}\n\n"
                        f"📄 <b>File:</b> {filename}\n"
                        f"📊 <b>Size:</b> {len(decoded):,} chars\n"
                        f"🔧 <b>Method:</b> {method}\n\n"
                        f"<i>{CREDIT}</i>"
                    ),
                    parse_mode="HTML"
                )
            
            preview = decoded[:2000]
            if len(decoded) > 2000:
                preview += "\n\n... [truncated]"
            
            preview_safe = html.escape(preview)
            
            try:
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text=f"<b>📋 Preview:</b>\n\n<pre>{preview_safe}</pre>",
                    parse_mode="HTML"
                )
            except:
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text=f"Preview:\n\n{preview[:1000]}"
                )
            
            try:
                await telegram_app.bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except:
                pass
        
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass
        
        logger.info("✅ Decode task completed")
    
    except Exception as e:
        logger.error(f"❌ Decode task error: {e}", exc_info=True)
        try:
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text=f"❌ <b>Error:</b>\n\n{html.escape(str(e)[:200])}"
            )
        except:
            pass


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 <b>Python Code Decoder</b>\n\n"
        "Send any obfuscated .py file and get the decoded source!\n\n"
        "Features:\n"
        "• Base64 decoding\n"
        "• ZIP extraction\n"
        "• AST analysis\n"
        "• AI-powered decoding\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "<b>How to use:</b>\n\n"
        "1️⃣ Send a .py file\n"
        "2️⃣ Wait for processing (~30-60s)\n"
        "3️⃣ Get decoded file + preview\n\n"
        "<b>Supported obfuscation:</b>\n"
        "• Base64 encoded strings\n"
        "• ZIP compressed code\n"
        "• Zlib compression\n"
        "• exec/eval patterns\n"
        "• Complex obfuscation\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
            "🚀 <b>Decoder Started</b>\n\n"
            "⏳ Processing your file...\n"
            "<i>This may take up to 60 seconds</i>",
            parse_mode="HTML"
        )
        
        asyncio.create_task(decode_task(update.effective_chat.id, code, filename, msg.message_id))
    
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global telegram_app
    
    logger.info("🚀 Starting bot...")
    
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
            handle_input
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
    
    logger.info("🛑 Shutting down...")
    await telegram_app.stop()
    await telegram_app.shutdown()


app = FastAPI(title="Python Decoder", version="2.0.0", lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "running", "version": "2.0.0"}


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
