import os
import sys
import argparse
import time
import logging
import sqlite3
import zlib
import re
import datetime
from llama_cpp import Llama

# ==============================================================================
# --- LOGGING SETUP ---
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "Qwen2.5-14B-Instruct-Q5_K_M.gguf")

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()

load_env()

default_db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'satya.db')
DB_PATH = os.environ.get('SATYA_DB_PATH', default_db_path)
if DB_PATH:
    DB_PATH = DB_PATH.strip()

def get_db_connection():
    db_url = os.environ.get('SATYA_DB_URL')
    db_token = os.environ.get('SATYA_DB_TOKEN')
    if db_url:
        db_url = db_url.strip()
    if db_token:
        db_token = db_token.strip()
        
    if db_url and (db_url.startswith('libsql://') or db_url.startswith('https://')):
        try:
            import libsql
            normalized_url = db_url.replace("libsql://", "https://")
            return libsql.connect(database=normalized_url, auth_token=db_token)
        except ImportError:
            logging.error("libsql package not installed. Falling back to local sqlite3.")
            
    return sqlite3.connect(DB_PATH)

def validate_formatting(headline):
    cleaned = headline.strip()
    words = cleaned.split()
    if not cleaned:
        return False
    if len(words) < 3 or len(words) > 14:
        return False
    return True

def post_process_headline(headline):
    headline = headline.strip()
    while (headline.startswith('"') and headline.endswith('"')) or \
          (headline.startswith("'") and headline.endswith("'")) or \
          (headline.startswith('*') and headline.endswith('*')):
        headline = headline[1:-1].strip()
    if headline.endswith('.'):
        headline = headline[:-1].strip()
    words = headline.split()
    if words:
        first_word = words[0]
        words[0] = first_word[0].upper() + first_word[1:]
        headline = " ".join(words)
    return headline

# ==============================================================================
# --- AI INFERENCE SETUP ---
# ==============================================================================
def load_llm():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure it is downloaded.")
    logging.info(f"Loading Qwen model for Validation from {MODEL_PATH}...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_batch=512,
        n_threads=4,
        verbose=False 
    )
    logging.info("Validation model loaded.")
    return llm

def main():
    start_time = time.time()
    logging.info("--- Starting Headline Validation & Auto-Fixing Run ---")
    
    # 1. Fetch batch to validate, then close connection immediately
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, content, rephrased_title 
            FROM articles 
            WHERE rephrased_title IS NOT NULL 
              AND rephrased_title != '' 
              AND (headline_verified = 0 OR headline_verified IS NULL)
            ORDER BY id DESC
            LIMIT 100
        """)
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        logging.critical(f"Failed to query verification candidates: {e}")
        sys.exit(1)
        
    if not rows:
        logging.info("No unchecked headlines found. All up to date.")
        sys.exit(0)
        
    logging.info(f"Found {len(rows)} articles to validate.")
    
    # 2. Load Prompts
    prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    headline_safe_prompt_path = os.path.join(prompt_dir, "headline_safe.txt")
    critic_prompt_path = os.path.join(prompt_dir, "critic.txt")
    
    try:
        with open(headline_safe_prompt_path, "r", encoding="utf-8") as f:
            headline_safe_prompt_template = f.read()
        with open(critic_prompt_path, "r", encoding="utf-8") as f:
            critic_prompt_template = f.read()
    except Exception as e:
        logging.critical(f"Failed to load prompts: {e}")
        sys.exit(1)
        
    # 3. Initialize Model
    try:
        llm = load_llm()
    except Exception as e:
        logging.critical(f"Failed to load model: {e}")
        sys.exit(1)
        
    validated_count = 0
    fixed_count = 0
    
    # 4. Loop & Verify
    for r in rows:
        article_id = r[0]
        title = r[1]
        compressed_content = r[2]
        proposed_headline = r[3]
        
        # Decompress content
        try:
            content = zlib.decompress(compressed_content).decode('utf-8') if compressed_content else ""
        except Exception:
            content = ""
            
        body_snippet = content[:1500]
        
        logging.info(f"Fact-checking ID: {article_id} | Headline: '{proposed_headline}'...")
        
        # A. Ask Critic
        critic_valid = False
        try:
            formatted_critic = critic_prompt_template.format(body_snippet=body_snippet, headline=proposed_headline)
            critic_response = llm(
                formatted_critic,
                max_tokens=5,
                stop=["<|im_end|>", "Article:", "<|im_start|>"],
                temperature=0.0, # Greedy
                echo=False
            )
            critic_ans = critic_response['choices'][0].get('text', '').strip().upper()
            if "YES" in critic_ans and "NO" not in critic_ans:
                critic_valid = True
        except Exception as e:
            logging.error(f"Error calling LLM for critic: {e}")
            continue
            
        if critic_valid:
            # Correct headline! Save verified=1
            logging.info(f"  ✓ VERIFIED CORRECT.")
            max_db_retries = 3
            for db_attempt in range(max_db_retries):
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("UPDATE articles SET headline_verified = 1 WHERE id = ?", (article_id,))
                    conn.commit()
                    conn.close()
                    break
                except Exception as db_e:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    time.sleep(2.0)
            validated_count += 1
        else:
            # Incorrect headline! Generate a Safe replacement
            logging.warning(f"  ✗ HALLUCINATION DETECTED. Triggering Auto-Fixer...")
            try:
                formatted_safe = headline_safe_prompt_template.format(title=title, body_snippet=body_snippet)
                safe_response = llm(
                    formatted_safe,
                    max_tokens=50,
                    stop=["<|im_end|>", "Article:", "<|im_start|>"],
                    temperature=0.0, # Cool and strict
                    echo=False
                )
                safe_headline = safe_response['choices'][0].get('text', '').strip()
                clean_fixed = post_process_headline(safe_headline)
                
                # Fallback check if LLM returned empty or too short
                if not validate_formatting(clean_fixed):
                    words = title.split()
                    clean_fixed = " ".join(words[:14]) if len(words) > 14 else title
                    
                logging.info(f"  → Fixed Headline: '{clean_fixed}'")
                
                # Save fixed headline & set verified=1
                max_db_retries = 3
                for db_attempt in range(max_db_retries):
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute(
                            "UPDATE articles SET rephrased_title = ?, headline_verified = 1 WHERE id = ?",
                            (clean_fixed, article_id)
                        )
                        conn.commit()
                        conn.close()
                        break
                    except Exception as db_e:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        time.sleep(2.0)
                fixed_count += 1
            except Exception as e:
                logging.error(f"Failed to auto-fix article {article_id}: {e}")
                
    logging.info(f"--- Verification Completed. Verified: {validated_count} | Auto-fixed: {fixed_count} ---")
    
    # Check if there are more remaining validation candidates
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM articles 
            WHERE rephrased_title IS NOT NULL 
              AND rephrased_title != '' 
              AND (headline_verified = 0 OR headline_verified IS NULL)
            LIMIT 1
        """)
        more = cursor.fetchone()
        conn.close()
        if more:
            print("has_more=true")
        else:
            print("has_more=false")
    except Exception:
        print("has_more=false")

if __name__ == '__main__':
    main()
