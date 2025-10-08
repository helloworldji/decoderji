import os
import tempfile
import logging
import asyncio
import base64
import zlib
import marshal
import codecs
import binascii
import re
import zipfile
import io
import html
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

# --- Logging ---
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

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY or not WEBHOOK_URL:
    raise ValueError("❌ Missing environment variables!")

# --- Gemini Init ---
genai.configure(api_key=GEMINI_API_KEY)

def get_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                logger.info(f"Using: {m.name}")
                return genai.GenerativeModel(m.name)
    except:
        pass
    return genai.GenerativeModel('models/gemini-pro')

model = get_model()

# --- Constants ---
CREDIT = "Dev: @aadi_io"
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_MANUAL_ITERATIONS = 30
MAX_AI_ATTEMPTS = 5

telegram_app: Optional[Application] = None


# --- MANUAL DECODING ---

def safe_eval_decode(expr: str) -> str:
    """Safe eval for decoding."""
    try:
        result = eval(expr, {"__builtins__": {}}, {
            "base64": base64, "zlib": zlib, "marshal": marshal,
            "bytes": bytes, "bytearray": bytearray,
        })
        if isinstance(result, bytes):
            return result.decode('utf-8', errors='ignore')
        return str(result) if result else expr
    except:
        return expr


def extract_all_base64(code: str) -> str:
    """Decode all base64."""
    # Pattern 1: base64.b64decode(...)
    pattern1 = r"base64\.b64decode\s*KATEX_INLINE_OPEN\s*([^)]+)\s*KATEX_INLINE_CLOSE"
    for match in re.findall(pattern1, code):
        try:
            decoded = safe_eval_decode(match)
            if decoded and len(decoded) > 10:
                code = code.replace(f"base64.b64decode({match})", f'"""{decoded}"""')
        except:
            pass
    
    # Pattern 2: Long base64 strings
    pattern2 = r"[b]?['\"]([A-Za-z0-9+/=]{200,})['\"]"
    for match in re.findall(pattern2, code):
        try:
            decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
            if decoded and len(decoded) > 10:
                code = code.replace(match, decoded)
        except:
            pass
    
    return code


def extract_from_zip(code: str) -> str:
    """Extract ZIP files."""
    pattern = r"['\"]([A-Za-z0-9+/=]{500,})['\"]"
    for match in re.findall(pattern, code):
        try:
            data = base64.b64decode(match)
            if data[:2] == b'PK':
                logger.info("🎯 ZIP found!")
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    contents = []
                    for name in zf.namelist():
                        try:
                            content = zf.read(name).decode('utf-8', errors='ignore')
                            contents.append(f"# {name}\n{content}")
                        except:
                            pass
                    if contents:
                        return "\n\n".join(contents)
        except:
            pass
    return code


def extract_exec_eval(code: str) -> str:
    """Extract from exec/eval."""
    patterns = [
        r'exec\s*KATEX_INLINE_OPEN\s*([^)]+)\s*KATEX_INLINE_CLOSE',
        r'eval\s*KATEX_INLINE_OPEN\s*([^)]+)\s*KATEX_INLINE_CLOSE',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, code, re.DOTALL):
            try:
                decoded = safe_eval_decode(match.strip())
                if decoded and decoded != match and len(decoded) > 20:
                    return decoded
            except:
                pass
    return code


def decode_zlib(code: str) -> str:
    """Decode zlib."""
    pattern = r"zlib\.decompress\s*KATEX_INLINE_OPEN\s*([^)]+)\s*KATEX_INLINE_CLOSE"
    for match in re.findall(pattern, code):
        try:
            decoded = safe_eval_decode(match)
            if decoded:
                code = code.replace(f"zlib.decompress({match})", f'"""{decoded}"""')
        except:
            pass
    return code


def is_obfuscated(code: str) -> bool:
    """Check obfuscation."""
    if not code or len(code) < 10:
        return True
    
    indicators = [
        'exec(', 'eval(', 'compile(',
        'base64.b64decode', 'base64.b32decode',
        'marshal.loads', 'zlib.decompress',
        '__import__',
    ]
    
    for ind in indicators:
        if ind in code:
            return True
    
    if re.search(r"['\"][A-Za-z0-9+/=]{300,}['\"]", code):
        return True
    
    return False


def manual_decode_layer(code: str) -> str:
    """One manual decode pass."""
    original = code
    code = extract_from_zip(code)
    if code != original:
        return code
    code = extract_all_base64(code)
    code = decode_zlib(code)
    code = extract_exec_eval(code)
    return code


# --- AI DECODING STRATEGIES ---

async def ai_strategy_direct(code: str) -> tuple[bool, str]:
    """Strategy 1: Direct decode request."""
    try:
        logger.info("🤖 AI Strategy 1: Direct decode")
        
        sample = code[:20000] if len(code) > 20000 else code
        
        prompt = (
            "Decode this obfuscated Python code completely.\n"
            "Extract ALL hidden code from base64, ZIP, exec, eval, marshal, zlib, etc.\n"
            "Return ONLY clean Python source code with NO obfuscation.\n"
            "No explanations, no markdown, just pure Python code.\n\n"
            f"{sample}"
        )
        
        return await call_gemini(prompt)
    except Exception as e:
        logger.error(f"Strategy 1 error: {e}")
        return False, code


async def ai_strategy_step_by_step(code: str) -> tuple[bool, str]:
    """Strategy 2: Step-by-step decode."""
    try:
        logger.info("🤖 AI Strategy 2: Step-by-step")
        
        sample = code[:20000] if len(code) > 20000 else code
        
        prompt = (
            "You are a Python deobfuscator. Follow these steps:\n\n"
            "1. Identify the obfuscation method (base64/exec/eval/marshal/zlib/ZIP)\n"
            "2. Extract any base64 encoded strings and decode them\n"
            "3. If there's a ZIP file, extract all contents\n"
            "4. Remove all exec() and eval() wrappers\n"
            "5. Decode any zlib/marshal data\n"
            "6. Repeat until code is fully clean\n"
            "7. Return the final clean Python code ONLY\n\n"
            f"Code:\n{sample}\n\n"
            "Clean code:"
        )
        
        return await call_gemini(prompt)
    except Exception as e:
        logger.error(f"Strategy 2 error: {e}")
        return False, code


async def ai_strategy_decoder_script(code: str) -> tuple[bool, str]:
    """Strategy 3: Generate decoder script."""
    try:
        logger.info("🤖 AI Strategy 3: Generate decoder")
        
        sample = code[:15000] if len(code) > 15000 else code
        
        prompt = (
            "Create a Python script that will decode this obfuscated code.\n"
            "The script should:\n"
            "1. Import base64, zlib, marshal, zipfile\n"
            "2. Decode the obfuscated code\n"
            "3. Print the decoded result\n\n"
            "Return ONLY the decoder script that I can execute.\n\n"
            f"Obfuscated code:\n{sample}\n\n"
            "Decoder script:"
        )
        
        success, decoder_script = await call_gemini(prompt)
        
        if success and len(decoder_script) > 50:
            # Try to execute the decoder
            try:
                import subprocess
                import sys
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(decoder_script)
                    decoder_file = f.name
                
                result = subprocess.run(
                    [sys.executable, decoder_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                os.remove(decoder_file)
                
                if result.stdout and len(result.stdout) > 20:
                    logger.info("✅ Decoder script executed successfully")
                    return True, result.stdout
            except Exception as e:
                logger.error(f"Decoder execution failed: {e}")
        
        return False, code
    except Exception as e:
        logger.error(f"Strategy 3 error: {e}")
        return False, code


async def ai_strategy_aggressive(code: str) -> tuple[bool, str]:
    """Strategy 4: Aggressive extraction."""
    try:
        logger.info("🤖 AI Strategy 4: Aggressive")
        
        sample = code[:20000] if len(code) > 20000 else code
        
        prompt = (
            "CRITICAL: This code is heavily obfuscated. You MUST decode it completely.\n\n"
            "The code may contain:\n"
            "- Base64 encoded ZIP files\n"
            "- Multiple layers of encoding\n"
            "- Exec/eval wrappers\n"
            "- Marshal/zlib compression\n\n"
            "YOUR TASK:\n"
            "Decode EVERY layer until you reach the actual Python source code.\n"
            "Do NOT stop until the code is 100% readable.\n"
            "Return ONLY the final decoded Python code.\n\n"
            f"{sample}\n\n"
            "FULLY DECODED CODE:"
        )
        
        return await call_gemini(prompt)
    except Exception as e:
        logger.error(f"Strategy 4 error: {e}")
        return False, code


async def ai_strategy_example_based(code: str) -> tuple[bool, str]:
    """Strategy 5: Example-based."""
    try:
        logger.info("🤖 AI Strategy 5: Example-based")
        
        sample = code[:20000] if len(code) > 20000 else code
        
        prompt = (
            "Here's an example of decoding obfuscated Python:\n\n"
            "Obfuscated: exec(base64.b64decode(b'cHJpbnQoImhlbGxvIik='))\n"
            "Decoded: print(\"hello\")\n\n"
            "Now decode this code following the same approach.\n"
            "Extract and decode ALL layers until you get clean Python code.\n"
            "Return ONLY the decoded code:\n\n"
            f"{sample}"
        )
        
        return await call_gemini(prompt)
    except Exception as e:
        logger.error(f"Strategy 5 error: {e}")
        return False, code


async def call_gemini(prompt: str) -> tuple[bool, str]:
    """Call Gemini API."""
    try:
        safety = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        config = genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=16384,
        )
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(prompt, generation_config=config, safety_settings=safety)
        )
        
        if response and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    parts = [p.text for p in candidate.content.parts if hasattr(p, 'text')]
                    if parts:
                        result = ''.join(parts).strip()
                        result = result.replace("```python", "").replace("```", "").strip()
                        if len(result) > 50:
                            return True, result
        
        return False, ""
    except Exception as e:
        logger.error(f"Gemini call error: {e}")
        return False, ""


async def final_ai_verification(code: str) -> tuple[bool, str]:
    """Final verification by AI."""
    try:
        logger.info("🔍 Final AI verification...")
        
        sample = code[:15000] if len(code) > 15000 else code
        
        prompt = (
            "Check if this Python code is fully decoded or still has obfuscation.\n\n"
            "If it still has ANY obfuscation (base64, exec, eval, etc.), decode it completely.\n"
            "If it's already clean, return it as-is.\n\n"
            "Return ONLY the clean Python code:\n\n"
            f"{sample}"
        )
        
        success, result = await call_gemini(prompt)
        
        if success and len(result) > 50:
            # Check if result is better than original
            if not is_obfuscated(result) or len(result) > len(code):
                logger.info("✅ Final verification improved code")
                return True, result
        
        return False, code
    except Exception as e:
        logger.error(f"Final verification error: {e}")
        return False, code


# --- COMPLETE DECODE ---

async def complete_decode(code: str, chat_id: int, msg_id: int) -> tuple[str, int, list]:
    """Complete decode with all strategies."""
    decode_log = []
    
    # Phase 1: Manual decoding
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: MANUAL DECODING")
    logger.info("="*60)
    
    for i in range(1, MAX_MANUAL_ITERATIONS + 1):
        if i % 3 == 0:
            try:
                await telegram_app.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=f"🔄 <b>Phase 1: Manual Decode</b>\n\nLayer {i}/{MAX_MANUAL_ITERATIONS}",
                    parse_mode="HTML"
                )
            except:
                pass
        
        if not is_obfuscated(code):
            logger.info(f"✅ Clean after {i} manual iterations")
            decode_log.append(f"Manual: {i} layers")
            break
        
        new_code = await asyncio.get_event_loop().run_in_executor(None, manual_decode_layer, code)
        
        if new_code == code:
            logger.info(f"Manual decode stuck at iteration {i}")
            break
        
        code = new_code
    
    # Phase 2: AI Strategies (if still obfuscated)
    if is_obfuscated(code):
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: AI DECODING STRATEGIES")
        logger.info("="*60)
        
        strategies = [
            ("Direct", ai_strategy_direct),
            ("Step-by-step", ai_strategy_step_by_step),
            ("Decoder Script", ai_strategy_decoder_script),
            ("Aggressive", ai_strategy_aggressive),
            ("Example-based", ai_strategy_example_based),
        ]
        
        for idx, (name, strategy) in enumerate(strategies, 1):
            try:
                await telegram_app.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=f"🤖 <b>Phase 2: AI Decode</b>\n\nStrategy {idx}/5: {name}",
                    parse_mode="HTML"
                )
            except:
                pass
            
            logger.info(f"\n🤖 Trying AI Strategy: {name}")
            success, result = await strategy(code)
            
            if success and len(result) > 50:
                # Check if this is better
                if not is_obfuscated(result):
                    logger.info(f"✅✅ {name} fully decoded!")
                    code = result
                    decode_log.append(f"AI: {name} SUCCESS")
                    break
                elif len(result) > len(code) * 0.8:
                    logger.info(f"✅ {name} made progress")
                    code = result
                    decode_log.append(f"AI: {name} partial")
                    # Continue to next strategy
            
            await asyncio.sleep(2)  # Rate limiting
    
    # Phase 3: Final Verification
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: FINAL VERIFICATION")
    logger.info("="*60)
    
    try:
        await telegram_app.bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg_id,
            text="🔍 <b>Phase 3: Final Check</b>\n\nVerifying decode quality...",
            parse_mode="HTML"
        )
    except:
        pass
    
    success, verified = await final_ai_verification(code)
    if success:
        code = verified
        decode_log.append("Final: Verified ✓")
    
    # Final status
    is_clean = not is_obfuscated(code)
    total_iterations = len(decode_log)
    
    logger.info("\n" + "="*60)
    logger.info(f"DECODE COMPLETE: {'CLEAN ✅' if is_clean else 'PARTIAL ⚠️'}")
    logger.info(f"Strategies used: {', '.join(decode_log)}")
    logger.info("="*60)
    
    return code, total_iterations, decode_log


# --- Background Task ---

async def decode_task(chat_id: int, code: str, filename: str, msg_id: int):
    """Background decode."""
    try:
        logger.info(f"🚀 Starting decode for {chat_id}")
        
        decoded, iterations, log = await complete_decode(code, chat_id, msg_id)
        
        clean = not is_obfuscated(decoded)
        status = "✅ Fully Decoded" if clean else "⚠️ Best Effort Decode"
        
        # Save
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name
        
        try:
            # Send file
            with open(tmp_path, 'rb') as f:
                await telegram_app.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(f, filename=f"decoded_{filename}"),
                    caption=(
                        f"{status}\n\n"
                        f"📄 {filename}\n"
                        f"📊 {len(decoded):,} chars\n"
                        f"🔄 {iterations} strategies\n"
                        f"📝 {', '.join(log[:3])}\n\n"
                        f"{CREDIT}"
                    ),
                    parse_mode="HTML"
                )
            
            # Preview
            preview = decoded[:3500] if len(decoded) > 3500 else decoded
            preview_safe = html.escape(preview)
            
            try:
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text=f"<b>Preview:</b>\n\n<pre>{preview_safe}</pre>",
                    parse_mode="HTML"
                )
            except:
                await telegram_app.bot.send_message(
                    chat_id=chat_id,
                    text=f"Preview:\n\n{preview[:4000]}"
                )
            
            try:
                await telegram_app.bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except:
                pass
        
        finally:
            os.remove(tmp_path)
        
        logger.info(f"✅ Done for {chat_id}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.error(traceback.format_exc())
        try:
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Error\n\n{html.escape(str(e)[:300])}"
            )
        except:
            pass


# --- Commands ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 <b>Python Decoder Pro</b>\n\n"
        "Send obfuscated .py file\n"
        "Get fully decoded code\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "<b>Multi-Strategy Decoder</b>\n\n"
        "• 30 manual decode layers\n"
        "• 5 AI strategies\n"
        "• Final verification\n\n"
        "Send .py file and wait 2-10 min\n\n"
        f"<i>{CREDIT}</i>",
        parse_mode="HTML"
    )


# --- Handler ---

async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    code = ""
    filename = "code.py"
    
    try:
        if update.message.document:
            doc = update.message.document
            if not doc.file_name.endswith('.py'):
                await update.message.reply_text("⚠️ Send .py file")
                return
            if doc.file_size > MAX_FILE_SIZE:
                await update.message.reply_text("⚠️ Too large")
                return
            filename = doc.file_name
            file = await doc.get_file()
            code = (await file.download_as_bytearray()).decode('utf-8', errors='ignore')
        elif update.message.text:
            code = update.message.text
        else:
            return

        if not code.strip():
            return

        msg = await update.message.reply_text(
            "🚀 <b>Multi-Strategy Decode</b>\n\n"
            "Phase 1: Manual (30 layers)\n"
            "Phase 2: AI (5 strategies)\n"
            "Phase 3: Final verification\n\n"
            "⏳ 2-10 minutes\n\n"
            "<i>Starting...</i>",
            parse_mode="HTML"
        )
        
        asyncio.create_task(decode_task(update.effective_chat.id, code, filename, msg.message_id))
        
    except Exception as e:
        logger.error(f"Handler error: {e}")


# --- Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global telegram_app
    
    telegram_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(180)
        .write_timeout(180)
        .build()
    )
    
    telegram_app.add_handler(CommandHandler("start", start_cmd))
    telegram_app.add_handler(CommandHandler("help", help_cmd))
    telegram_app.add_handler(MessageHandler((filters.TEXT | filters.Document.PY) & ~filters.COMMAND, handle_input))
    
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.bot.set_webhook(url=f"{WEBHOOK_URL}/webhook", allowed_updates=Update.ALL_TYPES)
    
    logger.info("✅ Ready!")
    
    yield
    
    await telegram_app.stop()
    await telegram_app.shutdown()


# --- FastAPI ---

app = FastAPI(title="Decoder", version="8.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "running", "version": "8.0.0", "strategies": 5}

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
        logger.error(f"Webhook: {e}")
        return Response(status_code=500)

@app.head("/")
@app.head("/health")
async def head():
    return Response(status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
