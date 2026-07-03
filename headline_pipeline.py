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

    # Helper to check if a word is a common English word
    def is_common_word(word):
        word_lower = word.lower()
        if len(word_lower) <= 3:
            return True
        common = {
            # Verbs
            "rejects", "reject", "rejections", "rejection", "fines", "fine", "goes", "go", "went", "gone", "aim", "aims", "held", "hold",
            "holds", "withheld", "withhold", "drowns", "drown", "drowned", "abolishes", "abolish", "demands", "demand",
            "face", "faces", "faced", "vanishes", "vanish", "vanished", "sparks", "spark", "sparked", "warned", "warn", "warns",
            "arrested", "arrest", "arrests", "kills", "kill", "killed", "shows", "show", "shown", "showed", "claims", "claim",
            "claimed", "says", "say", "said", "wins", "win", "won", "loses", "lose", "lost", "seeks", "seek", "sought",
            "finds", "find", "found", "takes", "take", "took", "taken", "makes", "make", "made", "comes", "come", "came",
            "gets", "get", "got", "gotten", "gives", "give", "gave", "given", "adds", "add", "added", "helps", "help",
            "helped", "hopes", "hope", "hoped", "calls", "call", "called", "meets", "meet", "met", "stops", "stop",
            "stopped", "drops", "drop", "dropped", "plans", "plan", "planned", "vows", "vow", "vowed", "wants", "want",
            "wanted", "needs", "need", "needed", "seems", "seem", "seemed", "looks", "look", "looked", "tries", "try",
            "tried", "plays", "play", "played", "runs", "run", "ran", "keeps", "keep", "kept", "starts", "start",
            "started", "ends", "end", "ended", "brings", "bring", "brought", "leads", "lead", "led", "breaks", "break",
            "broke", "broken", "buys", "buy", "bought", "sells", "sell", "sold", "cuts", "cut", "sends", "send",
            "sent", "sets", "set", "dies", "die", "died", "kills", "kill", "killed", "warned", "warn", "warns",
            "arrested", "arrest", "arrests", "kills", "kill", "killed", "shows", "show", "shown", "showed", "claims",
            "claim", "claimed", "says", "say", "said", "wins", "win", "won", "loses", "lose", "lost", "seeks", "seek",
            "sought", "finds", "find", "found", "takes", "take", "took", "taken", "makes", "make", "made", "comes",
            "come", "came", "gets", "get", "got", "gotten", "gives", "give", "gave", "given", "adds", "add", "added",
            "helps", "help", "helped", "hopes", "hope", "hoped", "calls", "call", "called", "meets", "meet", "met",
            "stops", "stop", "stopped", "drops", "drop", "dropped", "plans", "plan", "planned", "vows", "vow", "vowed",
            "wants", "want", "wanted", "needs", "need", "needed", "seems", "seem", "seemed", "looks", "look", "looked",
            "tries", "try", "tried", "plays", "play", "played", "runs", "run", "ran", "keeps", "keep", "kept", "starts",
            "start", "started", "ends", "end", "ended", "brings", "bring", "brought", "leads", "lead", "led", "breaks",
            "break", "broke", "broken", "buys", "buy", "bought", "sells", "sell", "sold", "cuts", "cut", "sends",
            "send", "sent", "sets", "set", "dies", "die", "died", "kills", "kill", "killed", "outperforms", "outperform",
            "outperformed", "indicted", "indict", "indicts", "damages", "damage", "damaged", "stuns", "stun", "stunned",
            "wields", "wield", "wielded", "layoffs", "layoff", "pleas", "plea", "reinstate", "reinstates", "reinstated",
            "paralyzes", "paralyze", "paralyzed", "plummets", "plummet", "plummeted", "assaults", "assault", "assaulted",
            "abolishes", "abolish", "abolished", "demands", "demand", "demanded", "looms", "loom", "loomed", "vanishes",
            "vanish", "vanished", "streaming", "stream", "streamed", "wedding", "weddings", "hiding", "hide", "hidden",
            "shuts", "shut", "opens", "open", "opened", "closes", "close", "closed", "accuses", "accuse", "accused",
            "alleges", "allege", "alleged", "reveals", "reveal", "revealed",
            
            # Nouns / Adjectives / Adverbs
            "woman", "women", "people", "police", "officer", "officers", "government", "court", "high", "state", "city",
            "country", "world", "year", "years", "time", "times", "home", "house", "family", "school", "jobs", "work",
            "life", "lives", "death", "deaths", "case", "cases", "news", "report", "reports", "plea", "pleas", "order",
            "orders", "decision", "decisions", "laws", "rules", "bill", "bills", "acts", "plans", "project", "projects",
            "program", "programs", "issue", "issues", "problem", "problems", "group", "groups", "team", "teams",
            "member", "members", "parts", "lines", "point", "points", "number", "numbers", "percent", "crore", "lakh",
            "million", "billion", "rupees", "dollar", "dollars", "deal", "deals", "agreement", "agreements", "talks",
            "meeting", "meetings", "visit", "visits", "clash", "clashes", "protest", "protests", "strike", "strikes",
            "attack", "attacks", "fires", "accident", "accidents", "crash", "crashes", "flood", "floods", "rains",
            "storm", "storms", "quake", "quakes", "earthquake", "earthquakes", "drought", "droughts", "crisis", "crises",
            "scandal", "scandals", "crime", "crimes", "murder", "murders", "theft", "thefts", "robbery", "robberies",
            "fraud", "frauds", "scam", "scams", "arrest", "arrests", "trial", "trials", "verdict", "verdicts",
            "sentence", "sentences", "jail", "jails", "prison", "prisons", "police", "cops", "judge", "judges",
            "lawyer", "lawyers", "court", "courts", "justice", "chief", "minister", "president", "leader", "leaders",
            "party", "parties", "polls", "election", "elections", "votes", "voter", "voters", "campaign", "campaigns",
            "seats", "alliance", "alliances", "coalition", "coalitions", "union", "unions", "board", "boards",
            "council", "councils", "panel", "panels", "committee", "committees", "agency", "agencies", "department",
            "departments", "ministry", "ministries", "security", "defense", "army", "military", "forces", "border",
            "borders", "clash", "clashes", "peace", "treaty", "treaties", "deal", "deals", "trade", "economy",
            "market", "markets", "price", "prices", "hike", "hikes", "rise", "rises", "fall", "falls", "drop",
            "drops", "loss", "losses", "profit", "profits", "revenue", "revenues", "taxes", "budget", "budgets",
            "funds", "money", "cash", "loans", "debt", "debts", "banks", "firms", "company", "companies",
            "business", "businesses", "industry", "industries", "sector", "sectors", "factory", "factories",
            "plants", "worker", "workers", "employee", "employees", "union", "unions", "strike", "strikes",
            "salary", "salaries", "wages", "pension", "pensions", "benefit", "benefits", "welfare", "scheme",
            "schemes", "policy", "policies", "system", "systems", "network", "networks", "service", "services",
            "facility", "facilities", "sites", "station", "stations", "ports", "airport", "airports", "railway",
            "railways", "train", "trains", "buses", "roads", "highway", "highways", "bridge", "bridges", "flyover",
            "flyovers", "tunnel", "tunnels", "metro", "metros", "cars", "vehicle", "vehicles", "truck", "trucks",
            "bikes", "cycle", "cycles", "traffic", "jams", "congestion", "safety", "danger", "hazard", "hazards",
            "risks", "threat", "threats", "warning", "warnings", "alert", "alerts", "notice", "notices", "rules",
            "regulation", "regulations", "order", "orders", "bans", "limits", "restriction", "restrictions",
            "permit", "permits", "license", "licenses", "fees", "fines", "penalty", "penalties", "charge",
            "charges", "allegation", "allegations", "claims", "complaint", "complaints", "suits", "cases",
            "pleas", "petition", "petitions", "appeal", "appeals", "hearing", "hearings", "verdict", "verdicts",
            "ruling", "rulings", "judgment", "judgments", "decree", "decrees", "order", "orders", "stays",
            
            "young", "large", "huge", "great", "good", "worst", "best", "first", "last", "next", "past",
            "future", "daily", "weekly", "monthly", "yearly", "annual", "local", "national", "global",
            "international", "foreign", "public", "private", "official", "unofficial", "alleged", "accused",
            "suspected", "reported", "proposed", "planned", "expected", "likely", "unlikely", "possible",
            "impossible", "certain", "uncertain", "clear", "unclear", "sure", "unsure", "true", "false",
            "real", "fake", "actual", "virtual", "online", "offline", "digital", "analog", "smart", "dumb",
            "free", "busy", "heavy", "light", "strong", "weak", "high", "low", "warm", "cool", "clean",
            "dirty", "fresh", "stale", "safe", "dangerous", "risky", "secure", "insecure", "open", "closed",
            "full", "empty", "half", "double", "triple", "single", "multiple", "several", "many", "some",
            "each", "every", "both", "either", "neither", "only", "even", "just", "very", "than", "more",
            "less", "most", "least", "almost", "nearly", "about", "around", "over", "under", "above",
            "below", "between", "among", "through", "during", "before", "after", "since", "until", "till",
            "while", "when", "where", "why", "how", "what", "whose", "which", "that", "this", "these",
            "those", "here", "there", "once", "twice", "again", "always", "never", "sometimes", "often",
            "seldom", "rarely", "usually", "generally", "specially", "especially", "particularly", "mainly",
            "mostly", "largely", "highly", "extremely", "really", "very", "quite", "rather", "somewhat",
            "slightly", "fairly", "pretty", "well", "badly", "hard", "easy", "soft", "loud", "quiet",
            "fast", "slow", "quick", "rapid", "sudden", "gradual", "smooth", "rough", "sharp", "flat",
            "round", "square", "straight", "crooked", "bent", "tight", "loose", "wide", "narrow", "deep",
            "shallow", "thick", "thin", "heavy", "light", "dark", "bright", "color", "colors", "black",
            "white", "grey", "gray", "brown", "pink", "orange", "purple", "gold", "silver", "bronze",
            "tuition", "refund", "students", "snoring", "divorce", "couple", "thunderous", "portfolio",
            "aviation", "dreams", "tragedy", "option", "corrupt", "commissioner", "bribe", "bribes", "today"
        }
        return word_lower in common

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
        
        # 1. Direct check in source
        if pn_clean in norm_orig or pn_clean in norm_body:
            continue
            
        # 2. Check stripping possessive 's or s
        if pn_clean.endswith('s'):
            pn_base = pn_clean[:-1]
            if pn_base in norm_orig or pn_base in norm_body:
                continue
                
        # 3. Allow if it's a common word rephrased
        if is_common_word(pn_clean):
            continue
            
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
                handle_rejection(cursor, conn, article_id, "generation", None, "LLM returned empty headline", args.test_run)
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
                handle_rejection(cursor, conn, article_id, "critic", generated_headline, f"Critic rejected headline. Response: '{critic_ans}'", args.test_run)
                continue
                
            # D. Run Mechanical validation
            is_valid, reject_reason = validate_headline(generated_headline, title, content)
            if not is_valid:
                handle_rejection(cursor, conn, article_id, "validator", generated_headline, reject_reason, args.test_run)
                continue
                
            # E. Clean & Post-Process
            clean_headline = post_process_headline(generated_headline)
            
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
