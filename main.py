import os
import tempfile
import logging
import asyncio
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
        
        # Preferred models - fast and reliable
        preferred = [
            'models/gemini-2.5-flash-lite',
            'models/gemini-flash-lite-latest',
            'models/gemini-2.0-flash-lite-preview',
            'models/gemini-flash-latest',
            'models/gemini-1.5-flash',
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
MAX_CODE_LENGTH = 30000  # 30K characters (reduced for better results)

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
        # Check if response exists
        if not response:
            return False, "No response from API"
        
        # Check for candidates
        if not hasattr(response, 'candidates') or not response.candidates:
            return False, "No response candidates"
        
        # Get first candidate
        candidate = response.candidates[0]
        
        # Check finish reason
        finish_reason = candidate.finish_reason
        
        # Map finish reasons
        finish_reasons = {
            0: "UNSPECIFIED",
            1: "STOP",
            2: "MAX_TOKENS",
            3: "SAFETY",
            4: "RECITATION",
            5: "OTHER"
        }
        
        reason_name = finish_reasons.get(finish_reason, f"UNKNOWN({finish_reason})")
        logger.info(f"Response finish_reason: {reason_name}")
        
        # Try to extract text from parts
        if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                text_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        text_parts.append(part.text)
                
                if text_parts:
                    full_text = ''.join(text_parts)
                    
                    # Check if we got truncated (MAX_TOKENS)
                    if finish_reason == 2:
                        logger.warning("Response was truncated (MAX_TOKENS)")
                        return True, full_text  # Still return what we got
                    
                    # Check safety block
                    if finish_reason == 3:
                        return False, "Response blocked by safety filters"
                    
                    # Normal completion
                    if finish_reason == 1:
                        return True, full_text
                    
                    # Other reasons - try to use text anyway
                    if full_text.strip():
                        return True, full_text
        
        # Check for safety ratings (blocked content)
        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
            blocked = [r for r in candidate.safety_ratings if hasattr(r, 'blocked') and r.blocked]
            if blocked:
                return False, f"Content blocked by safety filters: {reason_name}"
        
        return False, f"Could not extract text (finish_reason: {reason_name})"
        
    except Exception as e:
        logger.error(f"Error extracting response: {e}")
        return False, str(e)


async def decode_with_gemini(code: str, attempt: int = 1) -> tuple[bool, str]:
    """Decode obfuscated code using Gemini AI."""
    try:
        logger.info(f"🔄 Decoding attempt #{attempt}")
        
        # Truncate if needed
        code_to_send = code
        if len(code) > MAX_CODE_LENGTH:
            code_to_send = code[:MAX_CODE_LENGTH]
            logger.warning(f"Code truncated from {len(code)} to {MAX_CODE_LENGTH} chars")
        
        # Simple, clear prompt
        prompt = f"""Deobfuscate and decode this Python code. Return ONLY the clean, readable Python code without any explanations or markdown.

Obfuscated code:
