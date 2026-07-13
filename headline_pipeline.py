import os
import sys
import argparse
import time
import logging
import sqlite3
import zlib
import re
import string
import datetime
import json
import urllib.request
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

def log_rejection(article_id, stage, generated_headline, reason):
    msg = f"Article ID: {article_id} | Stage: {stage} | Generated: '{generated_headline}' | Reason: {reason}"
    logging.warning(f"[REJECTION] {msg}")
    
    # Append to headline_rejections.log in the same directory
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "headline_rejections.log")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {msg}\n")
    except Exception as e:
        logging.error(f"Failed to write to rejections log file: {e}")

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "Qwen2.5-14B-Instruct-Q5_K_M.gguf")

def load_env():
    # Check parent directory for .env
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
            # Replace libsql:// with https:// to prevent InvalidUriChar error in libsql Rust wrapper
            normalized_url = db_url.replace("libsql://", "https://")
            return libsql.connect(database=normalized_url, auth_token=db_token)
        except ImportError:
            logging.error("libsql package not installed. Falling back to local sqlite3.")
            
    return sqlite3.connect(DB_PATH)

# ==============================================================================
# --- SKIP RULES & MECHANICAL VALIDATOR ---
# ==============================================================================
def should_skip_article(title, content):
    if len(content) > 15000:
        return True
    
    title_lower = title.lower()
    if "live updates" in title_lower or "explained" in title_lower or "from a to z" in title_lower:
        return True
        
    if title.count('|') >= 2:
        return True
        
    if content.startswith("Opinion"):
        return True
        
    if "Opinion |" in title or "| Comment" in title or "Editorial:" in title:
        return True
        
    return False

def validate_formatting(headline):
    cleaned = headline.strip()
    words = cleaned.split()
    if not cleaned:
        return False, "empty headline"
    if len(words) < 3:
        return False, "too short"
    if len(words) > 14:
        return False, f"length {len(words)} exceeds 14 words"
    # Language guard: reject non-Latin script output (e.g. Qwen slipping into
    # Chinese/Devanagari). Latin letters incl. accents sit below U+024F.
    if any(ch.isalpha() and ord(ch) > 0x024F for ch in cleaned):
        return False, "contains non-Latin script characters"
    return True, None

def fallback_from_summary(content):
    """Last-resort headline derived from OUR OWN rephrased summary — never the
    publisher's original title (copyright: original titles must not be shown)."""
    text = ' '.join((content or '').split())
    # Strip markdown markers the rephraser leaves in summaries (**bold**, *em*, `code`)
    text = re.sub(r'[*_`#]+', '', text)
    if not text:
        return ''
    first_sentence = re.split(r'(?<=[.!?])\s+', text)[0]
    words = first_sentence.split()[:12]
    headline = ' '.join(words).rstrip('.,;:')
    return post_process_headline(headline) if headline else ''

def post_process_headline(headline):
    headline = headline.strip()
    # Strip wrapping quotes and asterisks
    while (headline.startswith('"') and headline.endswith('"')) or \
          (headline.startswith("'") and headline.endswith("'")) or \
          (headline.startswith('*') and headline.endswith('*')):
        headline = headline[1:-1].strip()
    # Strip trailing period
    if headline.endswith('.'):
        headline = headline[:-1].strip()
    # Collapse whitespace
    words = headline.split()
    if words:
        first_word = words[0]
        # Capitalize first letter of the first word, keep rest of first word casing
        words[0] = first_word[0].upper() + first_word[1:]
        headline = " ".join(words)
    return headline

# ==============================================================================
# --- SHARED LLM / DB HELPERS ---
# ==============================================================================
def ask_critic(llm, critic_prompt_template, body_snippet, headline):
    """Returns True only if the critic explicitly answers YES."""
    formatted_critic = critic_prompt_template.format(body_snippet=body_snippet, headline=headline)
    critic_response = llm(
        formatted_critic,
        max_tokens=5,
        stop=["<|im_end|>", "Article:", "<|im_start|>"],
        temperature=0.0,
        echo=False
    )
    ans = critic_response['choices'][0].get('text', '').strip().upper()
    verdict = "YES" in ans and "NO" not in ans
    if not verdict:
        logging.info(f"  [critic] answered '{ans or '[empty]'}' for: '{headline}'")
    return verdict

def save_title(article_id, headline, verified, max_db_retries=3):
    """Writes rephrased_title (+ headline_verified). Returns True on success.
    Callers MUST NOT count the row as processed when this returns False."""
    for db_attempt in range(max_db_retries):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE articles SET rephrased_title = ?, headline_verified = ? WHERE id = ?",
                (headline, verified, article_id)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as db_e:
            logging.warning(f"DB save failed for article {article_id} (attempt {db_attempt + 1}/{max_db_retries}): {db_e}")
            try:
                conn.close()
            except Exception:
                pass
            time.sleep(2.0)
    logging.error(f"DB save PERMANENTLY failed for article {article_id} after {max_db_retries} attempts.")
    return False

# ==============================================================================
# --- REJECTION & ERROR HANDLERS ---
# ==============================================================================
def handle_rejection(cursor, conn, article_id, stage, generated_headline, reason, is_test_run):
    log_rejection(article_id, stage, generated_headline, reason)
    if not is_test_run:
        try:
            cursor.execute("UPDATE articles SET rephrased_title = '' WHERE id = ?", (article_id,))
            conn.commit()
        except Exception as e:
            logging.critical(f"Database write failed during rejection update: {e}")
            sys.exit(1)

# ==============================================================================
# --- AI INFERENCE SETUP ---
# ==============================================================================
def load_llm():
    from llama_cpp import Llama
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        logging.info("Downloading Qwen 14B IT model from HuggingFace...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id='bartowski/Qwen2.5-14B-Instruct-GGUF',
            filename='Qwen2.5-14B-Instruct-Q5_K_M.gguf',
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
    logging.info(f"Loading Qwen 14B model from {MODEL_PATH}...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_batch=512,
        n_threads=4,
        verbose=False 
    )
    logging.info("Model loaded successfully.")
    return llm

# ==============================================================================
# --- MAIN PIPELINE ---
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Satya Headline Service")
    parser.add_argument("--test-run", action="store_true", help="Process 50 recent articles without consuming the queue with rejections")
    parser.add_argument("--shard", type=int, default=None, help="Shard ID to process (0 to num-shards - 1)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()
    
    start_time = time.time()
    logging.info("--- Starting News Headline Pipeline ---")
    
    cutoff_timestamp = int(time.time()) - 86400
    shard = args.shard if args.shard is not None else (int(os.environ.get('SHARD_ID')) if os.environ.get('SHARD_ID') is not None else None)
    num_shards = args.num_shards if args.num_shards != 1 else (int(os.environ.get('NUM_SHARDS')) if os.environ.get('NUM_SHARDS') is not None else 1)
    
    # 1. Connect to Database: Fetch batch and close immediately
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        batch_size = 50 if args.test_run else int(os.environ.get("HEADLINE_BATCH_SIZE", 10))
        
        if shard is not None and num_shards > 1:
            logging.info(f"Running in shard mode: shard {shard} of {num_shards}")
            query = """
                SELECT id, title, rephrased_article 
                FROM articles 
                WHERE rephrased_title IS NULL 
                  AND rephrased_article IS NOT NULL
                  AND scraped_at >= ?
                  AND (id % ?) = ?
                ORDER BY id DESC 
                LIMIT ?
            """
            cursor.execute(query, (cutoff_timestamp, num_shards, shard, batch_size))
        else:
            query = """
                SELECT id, title, rephrased_article 
                FROM articles 
                WHERE rephrased_title IS NULL 
                  AND rephrased_article IS NOT NULL
                  AND scraped_at >= ?
                ORDER BY id DESC 
                LIMIT ?
            """
            cursor.execute(query, (cutoff_timestamp, batch_size))
            
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        logging.critical(f"Failed to query database articles: {e}")
        try:
            conn.close()
        except Exception:
            pass
        sys.exit(1)
        
    if not rows:
        logging.info("No articles to process.")
        print("has_more=false")
        sys.exit(0)
        
    logging.info(f"Loaded {len(rows)} articles for headline generation.")
    
    # 2. Load Prompts
    prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    headline_prompt_path = os.path.join(prompt_dir, "headline.txt")
    headline_safe_prompt_path = os.path.join(prompt_dir, "headline_safe.txt")
    critic_prompt_path = os.path.join(prompt_dir, "critic.txt")
    
    try:
        with open(headline_prompt_path, "r", encoding="utf-8") as f:
            headline_prompt_template = f.read()
        with open(headline_safe_prompt_path, "r", encoding="utf-8") as f:
            headline_safe_prompt_template = f.read()
        with open(critic_prompt_path, "r", encoding="utf-8") as f:
            critic_prompt_template = f.read()
    except Exception as e:
        logging.critical(f"Failed to load prompts: {e}")
        sys.exit(1)
        
    # 3. Load Qwen model
    try:
        llm = load_llm()
    except Exception as e:
        logging.critical(f"Failed to initialize model: {e}")
        sys.exit(1)
        
    # 4. Process articles
    processed_count = 0
    db_failure_count = 0

    for idx, r in enumerate(rows):
        article_id = r[0]
        title = r[1]
        compressed_summary = r[2]
        
        # Decompress summary
        try:
            content = zlib.decompress(compressed_summary).decode('utf-8') if compressed_summary else ""
        except Exception as e:
            logging.error(f"Failed to decompress summary for article {article_id}: {e}")
            content = ""
            
        try:
            logging.info(f"Processing ID: {article_id} ({idx + 1} of {len(rows)}) | Title: {title[:50]}...")

            # B. Generate Headline (Masala) — from OUR rephrased summary only.
            # (No skip rules: every article must get a rephrased headline so the
            # publisher's original title is never displayed.)
            body_snippet = content[:1500]
            formatted_prompt = headline_prompt_template.format(body_snippet=body_snippet)
            
            response = llm(
                formatted_prompt,
                max_tokens=50,
                top_p=0.9,
                stop=["<|im_end|>", "Article:", "<|im_start|>"],
                temperature=0.4,
                repeat_penalty=1.1,
                echo=False
            )
            masala_headline = response['choices'][0].get('text', '').strip()
            
            # C. Check masala headline and ask critic
            masala_valid = False
            if masala_headline:
                masala_valid = ask_critic(llm, critic_prompt_template, body_snippet, masala_headline)

            # D. Retry flow (Fallback to Safe — safe must ALSO pass the critic).
            # A headline the critic rejected is never saved. If everything
            # fails, fall back to the first words of OUR OWN summary — the
            # publisher's original title is never used.
            final_headline = None
            if masala_valid:
                final_headline = masala_headline
            else:
                formatted_safe = headline_safe_prompt_template.format(body_snippet=body_snippet)
                safe_response = llm(
                    formatted_safe,
                    max_tokens=50,
                    stop=["<|im_end|>", "Article:", "<|im_start|>"],
                    temperature=0.2,
                    echo=False
                )
                safe_headline = safe_response['choices'][0].get('text', '').strip()

                masala_log = masala_headline if masala_headline else "[empty]"
                logging.info(f"Article ID: {article_id} | Stage: critic-retry | Masala: '{masala_log}' | Safe: '{safe_headline}'")

                if safe_headline and ask_critic(llm, critic_prompt_template, body_snippet, safe_headline):
                    final_headline = safe_headline

            if final_headline is None:
                summary_fallback = fallback_from_summary(content)
                logging.warning(f"No headline passed the critic for article {article_id}. Falling back to summary lead: '{summary_fallback}'")
                if not save_title(article_id, summary_fallback, 1):
                    db_failure_count += 1
                continue

            # E. Clean & Mechanical formatting — a format failure also falls
            # back to the summary lead, never the original title.
            clean_headline = post_process_headline(final_headline)
            is_valid_format, format_reason = validate_formatting(clean_headline)
            if not is_valid_format:
                summary_fallback = fallback_from_summary(content)
                logging.warning(f"Format check failed for '{clean_headline}': {format_reason}. Falling back to summary lead: '{summary_fallback}'")
                if not save_title(article_id, summary_fallback, 1):
                    db_failure_count += 1
                continue

            # F. Save success transactionally (Instant frontend publishing, headline_verified=0)
            if save_title(article_id, clean_headline, 0):
                processed_count += 1
                logging.info(f"Successfully saved rephrased_title for ID {article_id}: '{clean_headline}'")
            else:
                db_failure_count += 1
            
        except Exception as e:
            logging.error(f"Error processing article {article_id}: {e}")
                
    # 5. Complete run
    logging.info(f"--- Pipeline Finished. Processed {processed_count} headlines successfully. ---")

    if db_failure_count > 0:
        logging.critical(f"{db_failure_count} article(s) could not be written to the DB. Failing the run.")
        print("has_more=false")
        sys.exit(1)

    if len(rows) == batch_size:
        print("has_more=true")
    else:
        print("has_more=false")

if __name__ == '__main__':
    main()
