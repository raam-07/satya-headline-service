import argparse
import json
import logging
import os
import sys
import time
from llama_cpp import Llama

# Import pure helpers from headline_pipeline
from headline_pipeline import (
    validate_formatting,
    post_process_headline,
    fallback_from_summary,
    _DANGLING
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_chat_prompt(template, model_type):
    """
    Adapts the base prompt template for the model family:
    - Qwen 2.5: ChatML format (<|im_start|>user ... <|im_end|><|im_start|>assistant)
    - Gemma: Turn tokens format (<start_of_turn>user ... <end_of_turn><start_of_turn>model)
    """
    if "qwen" in model_type.lower():
        prompt = template.replace("<start_of_turn>user\n", "<|im_start|>user\n")
        prompt = prompt.replace("\n<end_of_turn>\n<start_of_turn>model", "\n<|im_end|>\n<|im_start|>assistant")
        prompt = prompt.replace("<end_of_turn>\n<start_of_turn>model", "<|im_end|>\n<|im_start|>assistant")
        return prompt
    else:
        # Default is Gemma
        return template

def get_stop_tokens(model_type):
    if "qwen" in model_type.lower():
        return ["<|im_end|>", "<|im_start|>", "\n\n", "Article:"]
    else:
        return ["<end_of_turn>", "<start_of_turn>", "\n\n", "<|im_end|>", "Article:", "<|im_start|>"]

def ask_critic_eval(llm, critic_template, body_snippet, headline, stop_tokens):
    formatted = critic_template.format(body_snippet=body_snippet, headline=headline)
    critic_response = llm(
        formatted,
        max_tokens=5,
        stop=stop_tokens,
        temperature=0.0,
        echo=False
    )
    ans = critic_response['choices'][0].get('text', '').strip().upper()
    verdict = "YES" in ans and "NO" not in ans
    return verdict, ans

def main():
    parser = argparse.ArgumentParser(description="Headline Model Bake-Off Evaluation Runner")
    parser.add_argument("--model-name", required=True, help="Friendly name of model (e.g. qwen-2.5-14b, gemma-4-12b)")
    parser.add_argument("--model-repo", required=True, help="HuggingFace repo ID")
    parser.add_argument("--model-file", required=True, help="GGUF filename")
    parser.add_argument("--sample-file", default="eval/sample_articles.json", help="Path to sample articles JSON")
    parser.add_argument("--output-file", required=True, help="Path to save evaluation results JSON")
    args = parser.parse_args()

    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, args.model_file)

    if not os.path.exists(model_path):
        logging.info(f"Downloading {args.model_file} from {args.model_repo} via HuggingFace...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=args.model_repo,
            filename=args.model_file,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )

    logging.info(f"Loading {args.model_name} from {model_path} with 4 threads...")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=4,
        verbose=False
    )

    # Load prompt templates
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(base_dir, "prompts")
    raw_headline_tmpl = open(os.path.join(prompt_dir, "headline.txt")).read()
    raw_safe_tmpl = open(os.path.join(prompt_dir, "headline_safe.txt")).read()
    raw_critic_tmpl = open(os.path.join(prompt_dir, "critic.txt")).read()

    headline_tmpl = get_chat_prompt(raw_headline_tmpl, args.model_name)
    safe_tmpl = get_chat_prompt(raw_safe_tmpl, args.model_name)
    critic_tmpl = get_chat_prompt(raw_critic_tmpl, args.model_name)
    stop_tokens = get_stop_tokens(args.model_name)

    with open(args.sample_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    logging.info(f"Starting bake-off benchmark on {len(articles)} articles using {args.model_name}...")
    results = []

    for idx, art in enumerate(articles):
        art_id = art.get('id')
        orig_title = art.get('title', '')
        body = art.get('rephrased_article', '')
        body_snippet = body[:1500]

        start_t = time.time()
        
        # 1. Primary headline generation
        formatted_prompt = headline_tmpl.format(body_snippet=body_snippet)
        response = llm(
            formatted_prompt,
            max_tokens=50,
            top_p=0.9,
            stop=stop_tokens,
            temperature=0.4,
            repeat_penalty=1.1,
            echo=False
        )
        raw_output = response['choices'][0].get('text', '').strip()
        headline = post_process_headline(raw_output)

        is_valid, reason = validate_formatting(headline)
        used_safe = False
        used_fallback = False

        # 2. Safe fallback if primary fails validation
        if not is_valid:
            used_safe = True
            formatted_safe = safe_tmpl.format(body_snippet=body_snippet)
            safe_resp = llm(
                formatted_safe,
                max_tokens=50,
                stop=stop_tokens,
                temperature=0.2,
                echo=False
            )
            safe_raw = safe_resp['choices'][0].get('text', '').strip()
            headline = post_process_headline(safe_raw)
            is_valid, reason = validate_formatting(headline)

        # 3. Last-resort summary fallback if still invalid
        if not is_valid:
            used_fallback = True
            headline = fallback_from_summary(body)

        gen_time = round(time.time() - start_t, 2)
        words = headline.split()
        word_count = len(words)
        char_count = len(headline)

        # 4. Critic factual consistency check
        critic_passed, critic_raw = ask_critic_eval(llm, critic_tmpl, body_snippet, headline, stop_tokens)

        logging.info(
            f"[{idx+1}/{len(articles)}] #{art_id} | Words: {word_count} | Critic: {'YES' if critic_passed else 'NO'} "
            f"| Time: {gen_time}s | Headline: '{headline}'"
        )

        results.append({
            "id": art_id,
            "original_title": orig_title,
            "generated_headline": headline,
            "word_count": word_count,
            "char_count": char_count,
            "critic_passed": critic_passed,
            "critic_raw": critic_raw,
            "used_safe_prompt": used_safe,
            "used_fallback": used_fallback,
            "generation_time_sec": gen_time
        })

    # Summary Statistics
    total = len(results)
    critic_yes_count = sum(1 for r in results if r["critic_passed"])
    fallback_count = sum(1 for r in results if r["used_fallback"])
    safe_count = sum(1 for r in results if r["used_safe_prompt"])
    avg_words = round(sum(r["word_count"] for r in results) / total, 1) if total else 0
    in_range_count = sum(1 for r in results if 6 <= r["word_count"] <= 13)
    avg_latency = round(sum(r["generation_time_sec"] for r in results) / total, 2) if total else 0

    summary = {
        "model_name": args.model_name,
        "model_repo": args.model_repo,
        "model_file": args.model_file,
        "total_articles": total,
        "critic_approval_rate_pct": round(critic_yes_count / total * 100, 1) if total else 0,
        "brevity_compliance_rate_pct": round(in_range_count / total * 100, 1) if total else 0,
        "avg_words_per_headline": avg_words,
        "fallback_rate_pct": round(fallback_count / total * 100, 1) if total else 0,
        "safe_prompt_rate_pct": round(safe_count / total * 100, 1) if total else 0,
        "avg_generation_latency_sec": avg_latency,
        "detailed_results": results
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Evaluation complete! Results saved to {args.output_file}")
    print(json.dumps({k: v for k, v in summary.items() if k != "detailed_results"}, indent=2))

if __name__ == "__main__":
    main()
