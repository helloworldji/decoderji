import base64
import codecs
import zlib
import ast
import marshal
import pickle
import quopri
import binascii
import lzma
import bz2
import gzip
import re
import chardet
import struct
import string
import itertools
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.backends import default_backend
import hashlib
import requests
import json
import random

try:
    import google.generativeai as genai
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

class UniversalPythonDecoder:
    def __init__(self, gemini_api_key=None):
        self.encoding_layers = []
        self.max_depth = 25
        self.ai_enabled = False
        self.gemini_api_key = gemini_api_key
        
        if AI_AVAILABLE and gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.ai_model = genai.GenerativeModel('gemini-pro')
                self.ai_enabled = True
                print("AI integration enabled with Gemini")
            except Exception as e:
                print(f"AI setup failed: {e}")
        
        # Comprehensive decoder registry
        self.decoders = {
            # Base encodings
            'base64': self.try_base64,
            'base32': self.try_base32,
            'base16': self.try_base16,
            'base85': self.try_base85,
            'base64_url': self.try_base64_url,
            'base45': self.try_base45,
            
            # Text encodings
            'hex': self.try_hex,
            'rot13': self.try_rot13,
            'rot47': self.try_rot47,
            'url': self.try_url,
            'html': self.try_html,
            'quoted_printable': self.try_quoted_printable,
            
            # Compression
            'zlib': self.try_zlib,
            'gzip': self.try_gzip,
            'bz2': self.try_bz2,
            'lzma': self.try_lzma,
            'deflate': self.try_deflate,
            
            # Python serialization
            'marshal': self.try_marshal,
            'pickle': self.try_pickle,
            
            # Binary conversions
            'bytearray': self.try_bytearray,
            'bytes': self.try_bytes,
            'bin': self.try_bin,
            'octal': self.try_octal,
            
            # Unicode
            'unicode_escape': self.try_unicode_escape,
            'string_escape': self.try_string_escape,
            'raw_unicode_escape': self.try_raw_unicode_escape,
            'unicode_raw': self.try_unicode_raw,
            
            # Character encodings
            'utf8': self.try_utf8,
            'utf16': self.try_utf16,
            'utf32': self.try_utf32,
            'ascii': self.try_ascii,
            'latin1': self.try_latin1,
            'cp1252': self.try_cp1252,
            'iso8859_1': self.try_iso8859_1,
            'mbcs': self.try_mbcs,
            
            # Cryptography
            'xor_simple': self.try_xor_simple,
            'xor_advanced': self.try_xor_advanced,
            'caesar': self.try_caesar,
            'atbash': self.try_atbash,
            'reverse': self.try_reverse,
            'char_shift': self.try_char_shift,
            'vigenere': self.try_vigenere,
            
            # Advanced
            'fernett': self.try_fernet,
            'aes_common': self.try_aes_common,
            'rsa_simple': self.try_rsa_simple,
        }
        
        # Add AI-powered decoders if available
        if self.ai_enabled:
            self.decoders['ai_pattern'] = self.try_ai_pattern
            self.decoders['ai_custom'] = self.try_ai_custom

    # ==================== BASE ENCODINGS ====================
    def try_base64(self, data):
        try:
            if isinstance(data, str):
                return base64.b64decode(data.encode()).decode('utf-8', errors='ignore')
            return base64.b64decode(data).decode('utf-8', errors='ignore')
        except: return None

    def try_base32(self, data):
        try:
            if isinstance(data, str):
                return base64.b32decode(data.encode()).decode('utf-8', errors='ignore')
            return base64.b32decode(data).decode('utf-8', errors='ignore')
        except: return None

    def try_base16(self, data):
        try:
            if isinstance(data, str):
                return base64.b16decode(data.encode()).decode('utf-8', errors='ignore')
            return base64.b16decode(data).decode('utf-8', errors='ignore')
        except: return None

    def try_base85(self, data):
        try:
            if isinstance(data, str):
                return base64.b85decode(data.encode()).decode('utf-8', errors='ignore')
            return base64.b85decode(data).decode('utf-8', errors='ignore')
        except: return None

    def try_base64_url(self, data):
        try:
            if isinstance(data, str):
                return base64.urlsafe_b64decode(data.encode()).decode('utf-8', errors='ignore')
            return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        except: return None

    def try_base45(self, data):
        try:
            # Base45 implementation
            BASE45_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
            if isinstance(data, str):
                data = data.upper()
                decoded = []
                for i in range(0, len(data), 3):
                    chunk = data[i:i+3]
                    if len(chunk) < 3:
                        break
                    n = BASE45_CHARSET.index(chunk[0]) + BASE45_CHARSET.index(chunk[1]) * 45
                    n += BASE45_CHARSET.index(chunk[2]) * 45 * 45
                    decoded.extend(divmod(n, 256))
                return bytes(decoded).decode('utf-8', errors='ignore')
        except: return None

    # ==================== TEXT ENCODINGS ====================
    def try_hex(self, data):
        try:
            if isinstance(data, str):
                clean_data = re.sub(r'[^0-9a-fA-F]', '', data)
                return bytes.fromhex(clean_data).decode('utf-8', errors='ignore')
            return data.hex()
        except: return None

    def try_rot13(self, data):
        try:
            if isinstance(data, str):
                return codecs.decode(data, 'rot13')
            return codecs.decode(data.decode('utf-8', errors='ignore'), 'rot13')
        except: return None

    def try_rot47(self, data):
        try:
            if isinstance(data, str):
                result = []
                for char in data:
                    if 33 <= ord(char) <= 126:
                        result.append(chr(33 + ((ord(char) - 33 + 47) % 94)))
                    else:
                        result.append(char)
                return ''.join(result)
        except: return None

    def try_url(self, data):
        try:
            from urllib.parse import unquote
            if isinstance(data, str):
                return unquote(data)
            return unquote(data.decode('utf-8', errors='ignore'))
        except: return None

    def try_html(self, data):
        try:
            import html
            if isinstance(data, str):
                return html.unescape(data)
            return html.unescape(data.decode('utf-8', errors='ignore'))
        except: return None

    def try_quoted_printable(self, data):
        try:
            if isinstance(data, str):
                return quopri.decodestring(data.encode()).decode('utf-8', errors='ignore')
            return quopri.decodestring(data).decode('utf-8', errors='ignore')
        except: return None

    # ==================== COMPRESSION ====================
    def try_zlib(self, data):
        try:
            if isinstance(data, str):
                data = data.encode()
            return zlib.decompress(data).decode('utf-8', errors='ignore')
        except: return None

    def try_gzip(self, data):
        try:
            if isinstance(data, str):
                data = data.encode()
            return gzip.decompress(data).decode('utf-8', errors='ignore')
        except: return None

    def try_bz2(self, data):
        try:
            if isinstance(data, str):
                data = data.encode()
            return bz2.decompress(data).decode('utf-8', errors='ignore')
        except: return None

    def try_lzma(self, data):
        try:
            if isinstance(data, str):
                data = data.encode()
            return lzma.decompress(data).decode('utf-8', errors='ignore')
        except: return None

    def try_deflate(self, data):
        try:
            if isinstance(data, str):
                data = data.encode()
            return zlib.decompress(data, -15).decode('utf-8', errors='ignore')  # raw deflate
        except: return None

    # ==================== PYTHON SERIALIZATION ====================
    def try_marshal(self, data):
        try:
            if isinstance(data, str):
                data = data.encode('latin-1')
            obj = marshal.loads(data)
            return str(obj)
        except: return None

    def try_pickle(self, data):
        try:
            if isinstance(data, str):
                data = data.encode('latin-1')
            obj = pickle.loads(data)
            return str(obj)
        except: return None

    # ==================== BINARY CONVERSIONS ====================
    def try_bytearray(self, data):
        try:
            if isinstance(data, str):
                return bytearray.fromhex(data).decode('utf-8', errors='ignore')
            return bytearray(data).decode('utf-8', errors='ignore')
        except: return None

    def try_bytes(self, data):
        try:
            if isinstance(data, str):
                return bytes.fromhex(data).decode('utf-8', errors='ignore')
            return data.decode('utf-8', errors='ignore')
        except: return None

    def try_bin(self, data):
        try:
            if isinstance(data, str):
                clean_data = re.sub(r'[^01]', '', data)
                # Pad to multiple of 8
                clean_data = clean_data.zfill((len(clean_data) + 7) // 8 * 8)
                byte_array = bytearray()
                for i in range(0, len(clean_data), 8):
                    byte = clean_data[i:i+8]
                    byte_array.append(int(byte, 2))
                return byte_array.decode('utf-8', errors='ignore')
        except: return None

    def try_octal(self, data):
        try:
            if isinstance(data, str):
                clean_data = re.sub(r'[^0-7]', '', data)
                byte_array = bytearray()
                for i in range(0, len(clean_data), 3):
                    byte_str = clean_data[i:i+3]
                    if len(byte_str) == 3:
                        byte_array.append(int(byte_str, 8))
                return byte_array.decode('utf-8', errors='ignore')
        except: return None

    # ==================== UNICODE ====================
    def try_unicode_escape(self, data):
        try:
            if isinstance(data, str):
                return data.encode('utf-8').decode('unicode_escape')
            return data.decode('unicode_escape')
        except: return None

    def try_string_escape(self, data):
        try:
            if isinstance(data, str):
                return data.encode('utf-8').decode('unicode_escape')
            return data.decode('unicode_escape')
        except: return None

    def try_raw_unicode_escape(self, data):
        try:
            if isinstance(data, str):
                return data.encode('utf-8').decode('raw_unicode_escape')
            return data.decode('raw_unicode_escape')
        except: return None

    def try_unicode_raw(self, data):
        try:
            if isinstance(data, str):
                return data.encode('raw_unicode_escape').decode('utf-8')
            return data.decode('utf-8')
        except: return None

    # ==================== CHARACTER ENCODINGS ====================
    def try_utf8(self, data):
        try:
            if isinstance(data, str):
                return data.encode('utf-8').decode('utf-8')
            return data.decode('utf-8')
        except: return None

    def try_utf16(self, data):
        try:
            if isinstance(data, str):
                return data.encode('utf-8').decode('utf-16')
            return data.decode('utf-16')
        except: return None

    def try_utf32(self, data):
        try:
            if isinstance(data, str):
                return data.encode('utf-8').decode('utf-32')
            return data.decode('utf-32')
        except: return None

    def try_ascii(self, data):
        try:
            if isinstance(data, str):
                return data.encode('ascii').decode('ascii')
            return data.decode('ascii')
        except: return None

    def try_latin1(self, data):
        try:
            if isinstance(data, str):
                return data.encode('latin-1').decode('latin-1')
            return data.decode('latin-1')
        except: return None

    def try_cp1252(self, data):
        try:
            if isinstance(data, str):
                return data.encode('cp1252').decode('cp1252')
            return data.decode('cp1252')
        except: return None

    def try_iso8859_1(self, data):
        try:
            if isinstance(data, str):
                return data.encode('iso-8859-1').decode('iso-8859-1')
            return data.decode('iso-8859-1')
        except: return None

    def try_mbcs(self, data):
        try:
            if isinstance(data, str):
                return data.encode('mbcs').decode('mbcs')
            return data.decode('mbcs')
        except: return None

    # ==================== CRYPTOGRAPHY ====================
    def try_xor_simple(self, data):
        """Try XOR with common keys"""
        if isinstance(data, str):
            data = data.encode('latin-1', errors='ignore')
        
        common_keys = [0x00, 0xFF, 0xAA, 0x55, 13, 42, 255, 128, 64, 32, 16, 8, 4, 2, 1]
        
        for key in common_keys:
            try:
                decoded = bytes([b ^ key for b in data])
                text = decoded.decode('utf-8', errors='ignore')
                if self.is_likely_python(text):
                    return text
            except: continue
        return None

    def try_xor_advanced(self, data):
        """Try XOR with multi-byte keys"""
        if isinstance(data, str):
            data = data.encode('latin-1', errors='ignore')
        
        common_multi_keys = [b'key', b'pass', b'secret', b'xor', b'python', b'code']
        
        for key in common_multi_keys:
            try:
                decoded = bytes([data[i] ^ key[i % len(key)] for i in range(len(data))])
                text = decoded.decode('utf-8', errors='ignore')
                if self.is_likely_python(text):
                    return text
            except: continue
        return None

    def try_caesar(self, data):
        """Try Caesar cipher variations"""
        if not isinstance(data, str):
            return None
        
        for shift in range(1, 95):  # Extended ASCII range
            try:
                result = ''
                for char in data:
                    if 32 <= ord(char) <= 126:
                        new_char = chr(32 + (ord(char) - 32 + shift) % 95)
                        result += new_char
                    else:
                        result += char
                
                if self.is_likely_python(result):
                    return result
            except: continue
        return None

    def try_atbash(self, data):
        """Atbash cipher (A-Z -> Z-A, a-z -> z-a)"""
        try:
            if not isinstance(data, str):
                return None
            
            result = []
            for char in data:
                if char.isupper():
                    result.append(chr(155 - ord(char)))  # 65+90=155
                elif char.islower():
                    result.append(chr(219 - ord(char)))  # 97+122=219
                else:
                    result.append(char)
            return ''.join(result)
        except: return None

    def try_reverse(self, data):
        """Try reversing the string"""
        try:
            if isinstance(data, str):
                reversed_data = data[::-1]
                if self.is_likely_python(reversed_data):
                    return reversed_data
        except: pass
        return None

    def try_char_shift(self, data):
        """Try character shifting"""
        if not isinstance(data, str):
            return None
        
        for shift in range(-10, 11):
            if shift == 0:
                continue
            try:
                result = ''.join(chr(ord(c) + shift) for c in data)
                if self.is_likely_python(result):
                    return result
            except: continue
        return None

    def try_vigenere(self, data):
        """Try Vigenere cipher with common keys"""
        if not isinstance(data, str):
            return None
        
        common_keys = ['python', 'code', 'secret', 'key', 'password', 'decode']
        
        for key in common_keys:
            try:
                result = []
                key_index = 0
                for char in data:
                    if char.isalpha():
                        shift = ord(key[key_index % len(key)].lower()) - ord('a')
                        if char.isupper():
                            result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
                        else:
                            result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
                        key_index += 1
                    else:
                        result.append(char)
                decoded = ''.join(result)
                if self.is_likely_python(decoded):
                    return decoded
            except: continue
        return None

    # ==================== ADVANCED CRYPTO ====================
    def try_fernet(self, data):
        """Try Fernet decryption with common keys"""
        try:
            if isinstance(data, str):
                data = data.encode()
            
            # Common Fernet keys derived from passwords
            common_passwords = ['password', 'secret', 'key', 'python', 'code', 'admin']
            for pwd in common_passwords:
                key = hashlib.sha256(pwd.encode()).digest()
                fernet = Fernet(base64.urlsafe_b64encode(key))
                try:
                    result = fernet.decrypt(data).decode('utf-8')
                    if self.is_likely_python(result):
                        return result
                except: continue
        except: pass
        return None

    def try_aes_common(self, data):
        """Try AES decryption with common keys and IVs"""
        try:
            if isinstance(data, str):
                data = data.encode('latin-1')
            
            if len(data) < 16:
                return None
                
            common_keys = [
                b'0123456789abcdef',  # 16 bytes
                b'secretkey1234567',  # 16 bytes  
                b'password12345678',  # 16 bytes
                b'\x00' * 16,         # null key
                b'\xff' * 16,         # full key
            ]
            
            for key in common_keys:
                try:
                    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
                    decryptor = cipher.decryptor()
                    result = decryptor.update(data) + decryptor.finalize()
                    text = result.decode('utf-8', errors='ignore')
                    if self.is_likely_python(text):
                        return text
                except: continue
        except: pass
        return None

    def try_rsa_simple(self, data):
        """Try simple RSA patterns"""
        # This is very basic - real RSA requires proper key management
        try:
            if isinstance(data, str):
                data = data.encode()
            # Simple modular arithmetic attempts
            for n in [65537, 257, 17, 3]:
                try:
                    result = bytes([pow(b, n, 256) for b in data])
                    text = result.decode('utf-8', errors='ignore')
                    if self.is_likely_python(text):
                        return text
                except: continue
        except: pass
        return None

    # ==================== AI-POWERED DECODING ====================
    def try_ai_pattern(self, data):
        """Use AI to detect encoding patterns"""
        if not self.ai_enabled or not isinstance(data, str):
            return None
        
        try:
            prompt = f"""
            Analyze this encoded data and suggest what type of encoding/encryption might be used:
            {data[:500]}...
            
            Common patterns to look for:
            - Base64: ends with =, alphanumeric with +/=
            - Hex: only 0-9, a-f
            - ROT: character substitution
            - XOR: random looking bytes
            - Compression: binary patterns
            
            Respond with JUST the encoding name or 'unknown'.
            """
            
            response = self.ai_model.generate_content(prompt)
            encoding_hint = response.text.strip().lower()
            
            # Map AI suggestions to our decoders
            ai_to_decoder = {
                'base64': 'base64', 'base32': 'base32', 'hex': 'hex',
                'rot13': 'rot13', 'url': 'url', 'zlib': 'zlib',
                'gzip': 'gzip', 'binary': 'bytes', 'unicode': 'unicode_escape'
            }
            
            if encoding_hint in ai_to_decoder:
                decoder_func = self.decoders.get(ai_to_decoder[encoding_hint])
                if decoder_func:
                    return decoder_func(data)
        except Exception as e:
            print(f"AI pattern detection failed: {e}")
        
        return None

    def try_ai_custom(self, data):
        """Use AI to directly attempt decoding"""
        if not self.ai_enabled or not isinstance(data, str):
            return None
        
        try:
            prompt = f"""
            This appears to be encoded Python code. Try to decode it and return ONLY the decoded Python code:
            {data[:1000]}
            
            If you cannot decode it, respond with "DECODE_FAILED".
            """
            
            response = self.ai_model.generate_content(prompt)
            result = response.text.strip()
            
            if result != "DECODE_FAILED" and self.is_likely_python(result):
                return result
        except Exception as e:
            print(f"AI direct decoding failed: {e}")
        
        return None

    # ==================== CORE DECODING LOGIC ====================
    def is_likely_python(self, text):
        """Enhanced Python code detection with multiple heuristics"""
        if not text or not isinstance(text, str):
            return False
        
        python_indicators = [
            r'(import\s+\w+|from\s+\w+\s+import)',
            r'(def\s+\w+\s*KATEX_INLINE_OPEN|class\s+\w+\s*KATEX_INLINE_OPEN?)',
            r'(print\s*KATEX_INLINE_OPEN|if\s+.*:|for\s+.*\s+in|while\s+.*:)',
            r'(__name__\s*==\s*[\'"]__main__[\'"])',
            r'(try:|except|finally:|with\s+)',
            r'(return\s+|yield\s+|await\s+)',
            r'(\#.*?$|""".*?"""|\'\'\'.*?\'\'\')',  # Comments and docstrings
            r'(lambda\s+.*?:)',
            r'(self\.|superKATEX_INLINE_OPENKATEX_INLINE_CLOSE)',
            r'(True|False|None)',
            r'(raise\s+|assert\s+)',
        ]
        
        score = 0
        for pattern in python_indicators:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            score += len(matches)
        
        # Check for valid Python syntax
        try:
            ast.parse(text)
            score += 5  # Big bonus for valid syntax
        except:
            pass
        
        # Check for common Python constructs
        common_python = ['__init__', '__main__', 'sys.argv', 'os.path', 'def main()']
        for construct in common_python:
            if construct in text:
                score += 1
        
        return score >= 3  # Lower threshold to catch more cases

    def decode_recursive(self, data, depth=0, path=None, visited=None):
        """Recursively decode through multiple layers with cycle detection"""
        if path is None:
            path = []
        if visited is None:
            visited = set()
        
        if depth > self.max_depth:
            return None, path
        
        # Create a fingerprint of current data to detect cycles
        data_fingerprint = hashlib.md5(str(data).encode()).hexdigest()
        if data_fingerprint in visited:
            return None, path
        visited.add(data_fingerprint)
        
        # Try all decoders
        for name, decoder in self.decoders.items():
            try:
                result = decoder(data)
                if result and result != data and len(str(result)) > 10:
                    current_path = path + [name]
                    
                    # Check if we found Python code
                    if self.is_likely_python(str(result)):
                        return result, current_path
                    
                    # Recursively decode further
                    final_result, final_path = self.decode_recursive(
                        result, depth + 1, current_path, visited
                    )
                    
                    if final_result:
                        return final_result, final_path
            
            except Exception as e:
                continue
        
        return None, path

    def smart_decode(self, encoded_data):
        """Main decoding function with multiple strategies"""
        print("Starting smart decoding...")
        
        strategies = [
            self.direct_recursive_decode,
            self.brute_force_priority,
            self.pattern_based_decode,
        ]
        
        if self.ai_enabled:
            strategies.insert(0, self.ai_guided_decode)
        
        for i, strategy in enumerate(strategies):
            print(f"Trying strategy {i+1}/{len(strategies)}...")
            result, layers = strategy(encoded_data)
            if result and self.is_likely_python(str(result)):
                print(f"Success with strategy {i+1}! Layers: {layers}")
                return result, layers
        
        return None, []

    def direct_recursive_decode(self, data):
        return self.decode_recursive(data)

    def brute_force_priority(self, data, max_combinations=1000):
        """Try different combinations with priority on common encodings"""
        priority_decoders = ['base64', 'base64_url', 'hex', 'zlib', 'gzip', 'bytes', 'unicode_escape']
        
        # Try priority decoders first in combinations
        for combo_length in range(1, 4):
            for combo in itertools.permutations(priority_decoders, combo_length):
                current = data
                path = []
                for decoder_name in combo:
                    decoder = self.decoders[decoder_name]
                    result = decoder(current)
                    if not result:
                        break
                    current = result
                    path.append(decoder_name)
                
                if current and self.is_likely_python(str(current)):
                    return current, path
        
        return None, []

    def pattern_based_decode(self, data):
        """Use patterns to guess encoding sequence"""
        if isinstance(data, str):
            # Common patterns in obfuscated Python
            patterns = {
                'exec_encoded': r'execKATEX_INLINE_OPEN.*?decodeKATEX_INLINE_OPEN.*?KATEX_INLINE_CLOSEKATEX_INLINE_CLOSE',
                'base64_long': r'[A-Za-z0-9+/]{40,}={0,2}',
                'hex_long': r'[0-9a-fA-F]{40,}',
                'compressed': r'[\\x00-\\xff]{20,}',
            }
            
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, data, re.IGNORECASE):
                    if pattern_name == 'base64_long':
                        result = self.try_base64(data)
                        if result:
                            final, path = self.decode_recursive(result)
                            if final:
                                return final, ['pattern_base64'] + path
                    elif pattern_name == 'hex_long':
                        result = self.try_hex(data)
                        if result:
                            final, path = self.decode_recursive(result)
                            if final:
                                return final, ['pattern_hex'] + path
        
        return None, []

    def ai_guided_decode(self, data):
        """Use AI to guide the decoding process"""
        if not self.ai_enabled:
            return None, []
        
        try:
            # Get AI suggestions
            prompt = f"""
            Analyze this encoded data and suggest a decoding strategy:
            {str(data)[:800]}
            
            Provide a list of decoding methods to try in order, based on patterns you detect.
            Respond with JUST the method names separated by commas.
            """
            
            response = self.ai_model.generate_content(prompt)
            suggestions = [s.strip().lower() for s in response.text.split(',')]
            
            # Try AI-suggested sequence
            current = data
            path = []
            for suggestion in suggestions:
                decoder_func = self.decoders.get(suggestion)
                if decoder_func:
                    result = decoder_func(current)
                    if result:
                        current = result
                        path.append(suggestion)
                        
                        if self.is_likely_python(str(current)):
                            return current, path
            
            # If AI suggestions don't work completely, continue recursively
            if path:
                final, final_path = self.decode_recursive(current, path=path)
                if final:
                    return final, final_path
                    
        except Exception as e:
            print(f"AI guidance failed: {e}")
        
        return None, []

    def decode_file(self, file_path):
        """Decode a Python file with multiple encoding layers"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            
            # Try different initial decodings
            initial_attempts = [
                raw_data.decode('utf-8', errors='ignore'),
                raw_data.decode('latin-1', errors='ignore'),
                raw_data.hex(),
                base64.b64encode(raw_data).decode('utf-8', errors='ignore'),
            ]
            
            for attempt in initial_attempts:
                result, layers = self.smart_decode(attempt)
                if result:
                    return result, layers
            
            return None, []
            
        except Exception as e:
            print(f"File decoding error: {e}")
            return None, []

# ==================== USAGE EXAMPLES ====================
def create_test_encoded_files():
    """Create test files with various encoding layers"""
    
    original_code = '''
import base64
import zlib

def secret_function():
    """A secret function that does something special"""
    message = "Hello from multi-layer encoding!"
    encoded = base64.b64encode(message.encode())
    return encoded.decode()

if __name__ == "__main__":
    result = secret_function()
    print(f"Result: {result}")
'''

    # Test case 1: Multiple base encodings
    layer1 = base64.b64encode(original_code.encode()).decode()
    layer2 = base64.b32encode(layer1.encode()).decode()
    layer3 = layer2.encode().hex()
    
    with open('test_multi_base.py.encoded', 'w') as f:
        f.write(layer3)
    
    # Test case 2: Compression + encoding
    compressed = zlib.compress(original_code.encode())
    layer1 = base64.b64encode(compressed).decode()
    layer2 = layer1.encode().hex()
    
    with open('test_compressed.py.encoded', 'w') as f:
        f.write(layer2)
    
    # Test case 3: XOR obfuscation
    xor_key = 42
    xor_encoded = bytes([b ^ xor_key for b in original_code.encode()])
    layer1 = base64.b85encode(xor_encoded).decode()
    
    with open('test_xor.py.encoded', 'w') as f:
        f.write(layer1)
    
    print("Created test encoded files")

def main():
    """Main demonstration function"""
    
    # Initialize decoder (add your Gemini API key if available)
    decoder = UniversalPythonDecoder(gemini_api_key=None)  # Add your key here
    
    # Create test files
    create_test_encoded_files()
    
    # Test decoding
    test_files = [
        'test_multi_base.py.encoded',
        'test_compressed.py.encoded', 
        'test_xor.py.encoded'
    ]
    
    for test_file in test_files:
        print(f"\n{'='*50}")
        print(f"Decoding: {test_file}")
        print(f"{'='*50}")
        
        result, layers = decoder.decode_file(test_file)
        
        if result:
            print(f"✓ Successfully decoded!")
            print(f"Layers used: {layers}")
            print(f"\nDecoded code preview:")
            print(result[:500] + "..." if len(result) > 500 else result)
        else:
            print("✗ Failed to decode")
    
    # Interactive mode
    print(f"\n{'='*50}")
    print("INTERACTIVE MODE")
    print(f"{'='*50}")
    print("Enter encoded strings to decode (type 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter encoded string: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input:
            result, layers = decoder.smart_decode(user_input)
            if result:
                print(f"✓ Decoded successfully!")
                print(f"Layers: {layers}")
                print(f"Result: {result}")
            else:
                print("✗ Could not decode")

if __name__ == "__main__":
    main()
