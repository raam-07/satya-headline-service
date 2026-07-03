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
MODEL_PATH = os.path.join(MODEL_DIR, "gemma-2-9b-it-Q4_K_M.gguf")

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
    return True, None

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
        logging.info("Downloading Gemma 9B IT model from HuggingFace...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id='bartowski/gemma-2-9b-it-GGUF',
            filename='gemma-2-9b-it-Q4_K_M.gguf',
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
    logging.info(f"Loading Gemma 9B model from {MODEL_PATH}...")
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
    args = parser.parse_args()
    
    start_time = time.time()
    logging.info("--- Starting News Headline Pipeline ---")
    
    # 1. Connect to Database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
    except Exception as e:
        logging.critical(f"Failed database initialization: {e}")
        sys.exit(1)
        
    # 2. Determine batch size
    batch_size = 50 if args.test_run else int(os.environ.get("HEADLINE_BATCH_SIZE", 10))
    
    # 3. Query articles: Fetch recent unprocessed articles
    try:
        query = """
            SELECT id, title, content 
            FROM articles 
            WHERE rephrased_title IS NULL 
            ORDER BY id DESC 
            LIMIT ?
        """
        cursor.execute(query, (batch_size,))
        rows = cursor.fetchall()
    except Exception as e:
        logging.critical(f"Failed to query database articles: {e}")
        conn.close()
        sys.exit(1)
        
    if not rows:
        logging.info("No articles to process.")
        print("has_more=false")
        conn.close()
        sys.exit(0)
        
    logging.info(f"Loaded {len(rows)} articles for headline generation.")
    
    # 4. Load Prompts
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
        conn.close()
        sys.exit(1)
        
    # 5. Load Gemma 9B model
    try:
        llm = load_llm()
    except Exception as e:
        logging.critical(f"Failed to initialize model: {e}")
        conn.close()
        sys.exit(1)
        
    # 6. Process articles
    processed_count = 0
    
    for idx, r in enumerate(rows):
        article_id = r[0]
        title = r[1]
        compressed_content = r[2]
        
        # Decompress content
        try:
            content = zlib.decompress(compressed_content).decode('utf-8') if compressed_content else ""
        except Exception as e:
            logging.error(f"Failed to decompress content for article {article_id}: {e}")
            content = ""
            
        try:
            # A. Check skip rules
            if should_skip_article(title, content):
                handle_rejection(cursor, conn, article_id, "skip", None, "Skipped by listicle/live-blog title filter or length > 15k chars", args.test_run)
                continue
                
            logging.info(f"Processing ID: {article_id} ({idx + 1} of {len(rows)}) | Title: {title[:50]}...")
            
            # B. Generate Headline (Masala)
            body_snippet = content[:1500]
            formatted_prompt = headline_prompt_template.format(title=title, body_snippet=body_snippet)
            
            response = llm(
                formatted_prompt,
                max_tokens=50,
                top_p=0.9,
                stop=["<end_of_turn>"],
                temperature=0.4,
                repeat_penalty=1.1,
                echo=False
            )
            masala_headline = response['choices'][0].get('text', '').strip()
            
            # C. Check masala headline and ask critic
            critic_ans_masala = ""
            masala_valid = False
            
            if masala_headline:
                formatted_critic = critic_prompt_template.format(body_snippet=body_snippet, headline=masala_headline)
                critic_response = llm(
                    formatted_critic,
                    max_tokens=5,
                    stop=["<end_of_turn>"],
                    temperature=0.0,  # Greedy validation
                    echo=False
                )
                critic_ans_masala = critic_response['choices'][0].get('text', '').strip().upper()
                if "YES" in critic_ans_masala and "NO" not in critic_ans_masala:
                    masala_valid = True

            # D. Retry flow
            if not masala_valid:
                # Trigger retry using safe prompt
                formatted_safe = headline_safe_prompt_template.format(title=title, body_snippet=body_snippet)
                
                safe_response = llm(
                    formatted_safe,
                    max_tokens=50,
                    stop=["<end_of_turn>"],
                    temperature=0.2, # Cooler
                    echo=False
                )
                safe_headline = safe_response['choices'][0].get('text', '').strip()
                
                # Log retry: stage "critic-retry" with both headlines
                masala_log = masala_headline if masala_headline else "[empty]"
                logging.info(f"Article ID: {article_id} | Stage: critic-retry | Masala: '{masala_log}' (Critic: '{critic_ans_masala}') | Safe: '{safe_headline}'")
                
                if not safe_headline:
                    handle_rejection(cursor, conn, article_id, "generation", None, "LLM returned empty safe headline", args.test_run)
                    continue
                    
                # Run Critic validation on safe headline
                formatted_critic_safe = critic_prompt_template.format(body_snippet=body_snippet, headline=safe_headline)
                critic_response_safe = llm(
                    formatted_critic_safe,
                    max_tokens=5,
                    stop=["<end_of_turn>"],
                    temperature=0.0,
                    echo=False
                )
                critic_ans_safe = critic_response_safe['choices'][0].get('text', '').strip().upper()
                
                if "YES" not in critic_ans_safe or "NO" in critic_ans_safe:
                    handle_rejection(cursor, conn, article_id, "critic", safe_headline, f"Critic rejected safe headline: '{critic_ans_safe}'", args.test_run)
                    continue
                    
                final_headline = safe_headline
            else:
                final_headline = masala_headline

            # E. Clean & Mechanical formatting
            clean_headline = post_process_headline(final_headline)
            is_valid_format, format_reason = validate_formatting(clean_headline)
            if not is_valid_format:
                handle_rejection(cursor, conn, article_id, "format", clean_headline, format_reason, args.test_run)
                continue
                
            # F. Save success transactionally
            cursor.execute("UPDATE articles SET rephrased_title = ? WHERE id = ?", (clean_headline, article_id))
            conn.commit()
            
            processed_count += 1
            logging.info(f"Successfully saved rephrased_title for ID {article_id}: '{clean_headline}'")
            
        except Exception as e:
            # Per-article poison pill protection
            logging.error(f"Error processing article {article_id}: {e}")
            try:
                handle_rejection(cursor, conn, article_id, "exception", None, f"Exception occurred: {str(e)}", args.test_run)
            except Exception as db_e:
                logging.critical(f"Database update failed during fallback transaction: {db_e}")
                sys.exit(1)
                
    # 7. Complete run
    conn.close()
    logging.info(f"--- Pipeline Finished. Processed {processed_count} headlines successfully. ---")
    
    # If we processed the full batch size, we indicate there may be more
    if len(rows) == batch_size:
        print("has_more=true")
    else:
        print("has_more=false")

if __name__ == '__main__':
    main()
