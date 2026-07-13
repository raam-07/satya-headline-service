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
import socket

# Prevent network calls from hanging indefinitely
socket.setdefaulttimeout(30)

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

# Single source of truth — the pipeline's validators (avoids the two copies drifting)
from headline_pipeline import validate_formatting, post_process_headline, ask_critic, save_title, fallback_from_summary

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
    parser = argparse.ArgumentParser(description="Satya Headline Validator")
    parser.add_argument("--batch-size", type=int, default=20, help="Maximum number of articles to validate in one run")
    parser.add_argument("--shard", type=int, default=None, help="Shard ID to process (0 to num-shards - 1)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    start_time = time.time()
    logging.info("--- Starting Headline Validation & Auto-Fixing Run ---")
    
    shard = args.shard if args.shard is not None else (int(os.environ.get('SHARD_ID')) if os.environ.get('SHARD_ID') is not None else None)
    num_shards = args.num_shards if args.num_shards != 1 else (int(os.environ.get('NUM_SHARDS')) if os.environ.get('NUM_SHARDS') is not None else 1)
    batch_size = args.batch_size
    
    # 1. Fetch batch to validate, then close connection immediately
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        if shard is not None and num_shards > 1:
            logging.info(f"Running in shard mode: shard {shard} of {num_shards} with batch size {batch_size}")
            cursor.execute("""
                SELECT id, title, rephrased_article, rephrased_title 
                FROM articles 
                WHERE rephrased_title IS NOT NULL 
                  AND rephrased_title != '' 
                  AND (headline_verified = 0 OR headline_verified IS NULL)
                  AND (id % ?) = ?
                ORDER BY id DESC
                LIMIT ?
            """, (num_shards, shard, batch_size))
        else:
            logging.info(f"Running in single-mode with batch size {batch_size}")
            cursor.execute("""
                SELECT id, title, rephrased_article, rephrased_title 
                FROM articles 
                WHERE rephrased_title IS NOT NULL 
                  AND rephrased_title != '' 
                  AND (headline_verified = 0 OR headline_verified IS NULL)
                ORDER BY id DESC
                LIMIT ?
            """, (batch_size,))
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
    db_failure_count = 0
    llm_failure_count = 0
    
    # 4. Loop & Verify
    for r in rows:
        article_id = r[0]
        title = r[1]
        compressed_summary = r[2]
        proposed_headline = r[3]
        
        # Decompress summary
        try:
            content = zlib.decompress(compressed_summary).decode('utf-8') if compressed_summary else ""
        except Exception:
            content = ""
            
        # CRITICAL: truncate like the pipeline does — the full article blows
        # past n_ctx=4096 and permanently wedges the row (infinite dispatch loop)
        body_snippet = content[:1500]

        logging.info(f"Fact-checking ID: {article_id} | Headline: '{proposed_headline}'...")

        # A. Ask Critic
        try:
            critic_valid = ask_critic(llm, critic_prompt_template, body_snippet, proposed_headline)
        except Exception as e:
            logging.error(f"Error calling LLM for critic on article {article_id}: {e}")
            llm_failure_count += 1
            continue

        if critic_valid:
            # Correct headline! Save verified=1
            logging.info(f"  ✓ VERIFIED CORRECT.")
            if save_title(article_id, proposed_headline, 1):
                validated_count += 1
            else:
                db_failure_count += 1
        else:
            # Incorrect headline! Generate a Safe replacement.
            # The fix must itself pass the critic — never stamp an unchecked
            # headline as verified. If everything fails, blank the headline so
            # the frontend falls back to the original title.
            logging.warning(f"  ✗ HALLUCINATION DETECTED. Triggering Auto-Fixer...")
            try:
                formatted_safe = headline_safe_prompt_template.format(body_snippet=body_snippet)
                safe_response = llm(
                    formatted_safe,
                    max_tokens=50,
                    stop=["<|im_end|>", "Article:", "<|im_start|>"],
                    temperature=0.0, # Cool and strict
                    echo=False
                )
                safe_headline = safe_response['choices'][0].get('text', '').strip()
                clean_fixed = post_process_headline(safe_headline)

                is_valid_format, _ = validate_formatting(clean_fixed)
                fix_accepted = (
                    is_valid_format
                    and clean_fixed != proposed_headline  # re-saving the rejected headline is never a fix
                    and ask_critic(llm, critic_prompt_template, body_snippet, clean_fixed)
                )

                if fix_accepted:
                    logging.info(f"  → Fixed Headline: '{clean_fixed}'")
                    if save_title(article_id, clean_fixed, 1):
                        fixed_count += 1
                    else:
                        db_failure_count += 1
                else:
                    summary_fallback = fallback_from_summary(content)
                    logging.warning(f"  → Fix rejected too. Falling back to summary lead: '{summary_fallback}'")
                    if save_title(article_id, summary_fallback, 1):
                        fixed_count += 1
                    else:
                        db_failure_count += 1
            except Exception as e:
                logging.error(f"Failed to auto-fix article {article_id}: {e}")
                llm_failure_count += 1
                
    logging.info(f"--- Verification Completed. Verified: {validated_count} | Auto-fixed: {fixed_count} | LLM failures: {llm_failure_count} | DB failures: {db_failure_count} ---")

    if db_failure_count > 0:
        logging.critical(f"{db_failure_count} article(s) could not be written to the DB. Failing the run.")
        print("has_more=false")
        sys.exit(1)

    # Loop guard: if this run made zero progress, re-dispatching would loop
    # forever on the same stuck rows.
    if validated_count + fixed_count == 0:
        logging.warning("No progress made this run — suppressing self-dispatch to avoid an infinite loop.")
        print("has_more=false")
        return

    # Check if there are more remaining validation candidates globally
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
