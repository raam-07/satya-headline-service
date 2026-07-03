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
MODEL_PATH = os.path.join(MODEL_DIR, "gemma-2-9b-it-Q6_K.gguf")

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

def get_db_connection():
    db_url = os.environ.get('SATYA_DB_URL')
    db_token = os.environ.get('SATYA_DB_TOKEN')
    
    if db_url and (db_url.startswith('libsql://') or db_url.startswith('https://')):
        try:
            import libsql
            return libsql.connect(database=db_url, auth_token=db_token)
        except ImportError:
            logging.error("libsql package not installed. Falling back to local sqlite3.")
            
    return sqlite3.connect(DB_PATH)

# ==============================================================================
# --- DATABASE BOOTSTRAP (Self-healing Checkpoint Table) ---
# ==============================================================================
def bootstrap_database(conn):
    cursor = conn.cursor()
    try:
        # Create cursor checkpoint table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS headline_checkpoint (
                id INTEGER PRIMARY KEY CHECK (id=1),
                last_article_id INTEGER NOT NULL
            )
        """)
        # Insert initial cursor row
        cursor.execute("""
            INSERT OR IGNORE INTO headline_checkpoint (id, last_article_id) 
            VALUES (1, 0)
        """)
        conn.commit()
    except Exception as e:
        logging.error(f"Failed to bootstrap headline checkpoint table: {e}")
        # Re-raise to trigger exit if database connection is broken
        raise e

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

def validate_headline(generated_title, original_title, body):
    gen_title = generated_title.strip()
    
    # 1. Check length
    words = gen_title.split()
    if len(words) < 3 or len(words) > 14:
        return False, f"length {len(words)} out of bounds [3, 14]"
        
    # 2. Check leading/trailing quotes wrapping the whole line
    if (gen_title.startswith('"') and gen_title.endswith('"')) or (gen_title.startswith("'") and gen_title.endswith("'")):
        return False, "contains leading or trailing quotes wrapping the whole line"
        
    # Helper to check if a word is in text
    def clean_word(w):
        return "".join(c for c in w if c.isalnum()).lower()
        
    # Normalize number strings for comparison
    def normalize_numbers(text):
        text_lower = text.lower()
        text_lower = re.sub(r'(\d+)\s+(crore|lakh|million|billion|percent|pct)', r'\1\2', text_lower)
        return text_lower

    norm_gen = normalize_numbers(gen_title)
    norm_orig = normalize_numbers(original_title)
    norm_body = normalize_numbers(body)

    # 3. Extract proper nouns (all capitalized words, including the first word)
    punctuation = string.punctuation
    proper_nouns = []
    for idx, w in enumerate(words):
        cleaned_w = w.strip(punctuation)
        if not cleaned_w:
            continue
        if cleaned_w[0].isupper():
            proper_nouns.append(cleaned_w)
            
    # 4. Extract all numbers (including normalized units)
    numbers = re.findall(r'\b\d+(?:crore|lakh|million|billion|percent|pct)?\b', norm_gen)
    
    # Verify proper nouns are in original title or body
    for pn in proper_nouns:
        pn_clean = clean_word(pn)
        if not pn_clean:
            continue
        if pn_clean not in norm_orig and pn_clean not in norm_body:
            return False, f"proper noun '{pn}' not found in source"

    # Verify numbers are in original title or body
    for num in numbers:
        pattern = r'\b' + re.escape(num) + r'\b'
        if not re.search(pattern, norm_orig) and not re.search(pattern, norm_body):
            return False, f"number '{num}' not found in source"
            
    return True, None

def post_process_headline(headline):
    headline = headline.strip()
    # Strip wrapping quotes
    while (headline.startswith('"') and headline.endswith('"')) or (headline.startswith("'") and headline.endswith("'")):
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
            filename='gemma-2-9b-it-Q6_K.gguf',
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
    parser.add_argument("--test-run", action="store_true", help="Process 50 recent articles for verification without advancing global checkpoint")
    parser.add_argument("--retitle", action="store_true", help="Process articles even if they already have a rephrased_title")
    args = parser.parse_args()
    
    start_time = time.time()
    logging.info("--- Starting News Headline Pipeline ---")
    
    # 1. Connect and bootstrap
    try:
        conn = get_db_connection()
        bootstrap_database(conn)
        cursor = conn.cursor()
    except Exception as e:
        logging.critical(f"Failed database initialization: {e}")
        sys.exit(1)
        
    # 2. Determine batch size
    batch_size = 50 if args.test_run else int(os.environ.get("HEADLINE_BATCH_SIZE", 500))
    
    # 3. Read cursor and query articles
    try:
        if args.test_run:
            # For wet-validation, fetch 50 recent articles
            query = """
                SELECT id, title, content 
                FROM articles 
                WHERE rephrased_title IS NULL 
                ORDER BY id DESC 
                LIMIT ?
            """
            cursor.execute(query, (batch_size,))
        else:
            # Get checkpoint cursor
            cursor.execute("SELECT last_article_id FROM headline_checkpoint WHERE id = 1")
            last_id = cursor.fetchone()[0]
            logging.info(f"Checkpoint cursor last_article_id: {last_id}")
            
            if args.retitle:
                query = """
                    SELECT id, title, content 
                    FROM articles 
                    WHERE id > ? 
                    ORDER BY id ASC 
                    LIMIT ?
                """
            else:
                query = """
                    SELECT id, title, content 
                    FROM articles 
                    WHERE id > ? AND rephrased_title IS NULL 
                    ORDER BY id ASC 
                    LIMIT ?
                """
            cursor.execute(query, (last_id, batch_size))
            
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
    critic_prompt_path = os.path.join(prompt_dir, "critic.txt")
    
    try:
        with open(headline_prompt_path, "r", encoding="utf-8") as f:
            headline_prompt_template = f.read()
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
    
    for r in rows:
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
                log_rejection(article_id, "skip", None, "Skipped by listicle/live-blog title filter or length > 15k chars")
                cursor.execute("UPDATE articles SET rephrased_title = NULL WHERE id = ?", (article_id,))
                if not args.test_run:
                    cursor.execute("UPDATE headline_checkpoint SET last_article_id = ? WHERE id = 1", (article_id,))
                conn.commit()
                continue
                
            logging.info(f"Processing ID: {article_id} | Title: {title[:50]}...")
            
            # B. Generate Headline
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
            generated_headline = response['choices'][0].get('text', '').strip()
            
            if not generated_headline:
                log_rejection(article_id, "generation", None, "LLM returned empty headline")
                cursor.execute("UPDATE articles SET rephrased_title = NULL WHERE id = ?", (article_id,))
                if not args.test_run:
                    cursor.execute("UPDATE headline_checkpoint SET last_article_id = ? WHERE id = 1", (article_id,))
                conn.commit()
                continue
                
            # C. Run Critic validation
            formatted_critic = critic_prompt_template.format(body_snippet=body_snippet, headline=generated_headline)
            
            critic_response = llm(
                formatted_critic,
                max_tokens=5,
                stop=["<end_of_turn>"],
                temperature=0.0,  # Greedy validation
                echo=False
            )
            critic_ans = critic_response['choices'][0].get('text', '').strip().upper()
            
            if "YES" not in critic_ans or "NO" in critic_ans:
                log_rejection(article_id, "critic", generated_headline, f"Critic rejected headline. Response: '{critic_ans}'")
                cursor.execute("UPDATE articles SET rephrased_title = NULL WHERE id = ?", (article_id,))
                if not args.test_run:
                    cursor.execute("UPDATE headline_checkpoint SET last_article_id = ? WHERE id = 1", (article_id,))
                conn.commit()
                continue
                
            # D. Run Mechanical validation
            is_valid, reject_reason = validate_headline(generated_headline, title, content)
            if not is_valid:
                log_rejection(article_id, "validator", generated_headline, reject_reason)
                cursor.execute("UPDATE articles SET rephrased_title = NULL WHERE id = ?", (article_id,))
                if not args.test_run:
                    cursor.execute("UPDATE headline_checkpoint SET last_article_id = ? WHERE id = 1", (article_id,))
                conn.commit()
                continue
                
            # E. Clean & Post-Process
            clean_headline = post_process_headline(generated_headline)
            
            # F. Save transactionally
            cursor.execute("UPDATE articles SET rephrased_title = ? WHERE id = ?", (clean_headline, article_id))
            if not args.test_run:
                cursor.execute("UPDATE headline_checkpoint SET last_article_id = ? WHERE id = 1", (article_id,))
            conn.commit()
            
            processed_count += 1
            logging.info(f"Successfully saved rephrased_title for ID {article_id}: '{clean_headline}'")
            
        except Exception as e:
            # Per-article poison pill protection
            logging.error(f"Error processing article {article_id}: {e}")
            log_rejection(article_id, "exception", None, f"Exception occurred: {str(e)}")
            try:
                cursor.execute("UPDATE articles SET rephrased_title = NULL WHERE id = ?", (article_id,))
                if not args.test_run:
                    cursor.execute("UPDATE headline_checkpoint SET last_article_id = ? WHERE id = 1", (article_id,))
                conn.commit()
            except Exception as db_e:
                # Critical database loss exits 1
                logging.critical(f"Database update failed during fallback transaction: {db_e}")
                sys.exit(1)
                
    # 7. Complete run
    conn.close()
    logging.info(f"--- Pipeline Finished. Processed {processed_count} headlines successfully. ---")
    
    # If not a test run and we processed the full batch size, we indicate there may be more
    if not args.test_run and len(rows) == batch_size:
        print("has_more=true")
    else:
        print("has_more=false")

if __name__ == '__main__':
    main()
