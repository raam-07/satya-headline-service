import argparse
import json
import logging
import os
import re
import sys
import time
from llama_cpp import Llama

# Mock unused cloud dependencies
from unittest.mock import MagicMock
sys.modules.setdefault('gspread', MagicMock())
sys.modules.setdefault('oauth2client', MagicMock())
sys.modules.setdefault('oauth2client.service_account', MagicMock())
sys.modules.setdefault('libsql', MagicMock())

from headline_pipeline import (
    post_process_headline,
    validate_formatting,
    fallback_from_summary,
    ask_critic
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 12B Sharded Headline Benchmark Runner")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0 to num_shards - 1)")
    parser.add_argument("--num-shards", type=int, default=12, help="Total number of parallel shards")
    parser.add_argument("--model-repo", default="unsloth/gemma-4-12b-it-GGUF", help="HuggingFace model repo")
    parser.add_argument("--model-file", default="gemma-4-12b-it-Q4_K_M.gguf", help="GGUF model filename")
    parser.add_argument("--sample-file", default="eval/sample_60_articles.json", help="Path to sample articles")
    parser.add_argument("--output-file", default=None, help="Output JSON path")
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = f"eval_headline_results_shard_{args.shard}.json"

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

    logging.info(f"Loading Gemma 4 12B from {model_path} with 4 threads...")
    t_load = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_batch=512,
        n_threads=4,
        verbose=False
    )
    logging.info(f"Model loaded in {time.time() - t_load:.2f}s")

    # Load Prompts
    prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    with open(os.path.join(prompt_dir, "headline.txt"), "r", encoding="utf-8") as f:
        headline_tmpl = f.read()
    with open(os.path.join(prompt_dir, "headline_safe.txt"), "r", encoding="utf-8") as f:
        safe_tmpl = f.read()
    with open(os.path.join(prompt_dir, "critic.txt"), "r", encoding="utf-8") as f:
        critic_tmpl = f.read()

    stop_tokens = ["<turn|>", "<|turn>", "<eos>", "\n\n"]

    with open(args.sample_file, "r") as f:
        all_samples = json.load(f)

    # Shard slicing: partition articles evenly across num_shards
    samples = [art for i, art in enumerate(all_samples) if i % args.num_shards == args.shard]
    logging.info(f"Runner Shard {args.shard}/{args.num_shards}: processing {len(samples)} articles (IDs: {[s['id'] for s in samples]})")

    results = []
    critic_passed = 0
    total_latency = 0.0

    for idx, article in enumerate(samples):
        aid = article["id"]
        title = article["title"]
        category = article.get("category", "")
        prev_headline = article.get("prev_headline", "")
        content = article["content"]
        body_snippet = content[:1500]

        t0 = time.time()
        stage = "primary"
        final_headline = ""
        passed = False

        try:
            # 1. Primary Generation (Masala)
            prompt = headline_tmpl.format(body_snippet=body_snippet)
            res = llm(
                prompt,
                max_tokens=60,
                top_p=0.9,
                stop=stop_tokens,
                temperature=0.4,
                repeat_penalty=1.1,
                echo=False
            )
            raw = res['choices'][0].get('text', '').strip()
            masala = post_process_headline(raw)

            if masala and ask_critic(llm, critic_tmpl, body_snippet, masala):
                final_headline = masala
                passed = True
                stage = "primary"
            else:
                # 2. Safe Fallback Generation
                stage = "safe"
                safe_prompt = safe_tmpl.format(body_snippet=body_snippet)
                safe_res = llm(
                    safe_prompt,
                    max_tokens=60,
                    stop=stop_tokens,
                    temperature=0.2,
                    echo=False
                )
                raw_safe = safe_res['choices'][0].get('text', '').strip()
                safe_headline = post_process_headline(raw_safe)

                if safe_headline and ask_critic(llm, critic_tmpl, body_snippet, safe_headline):
                    final_headline = safe_headline
                    passed = True
                else:
                    stage = "summary_lead"
                    final_headline = fallback_from_summary(content)
                    passed = False

            # Final clean
            clean_headline = post_process_headline(final_headline)
            is_valid, _ = validate_formatting(clean_headline)
            if not is_valid:
                clean_headline = fallback_from_summary(content)

            dur = time.time() - t0
            total_latency += dur
            words = len(clean_headline.split())
            if passed:
                critic_passed += 1

            item = {
                "id": aid,
                "category": category,
                "original_title": title,
                "prev_headline": prev_headline,
                "gemma4_headline": clean_headline,
                "stage": stage,
                "critic_approved": passed,
                "word_count": words,
                "latency_seconds": round(dur, 2)
            }
            results.append(item)
            logging.info(f"[Shard {args.shard} | {idx+1}/{len(samples)}] #{aid} [{stage}] | Critic: {'PASS' if passed else 'FAIL'} | G4: '{clean_headline}' | {dur:.2f}s")
        except Exception as e:
            logging.error(f"Error on article #{aid}: {e}")
            results.append({
                "id": aid,
                "category": category,
                "original_title": title,
                "prev_headline": prev_headline,
                "error": str(e)
            })

    pass_pct = (critic_passed / len(samples) * 100) if samples else 0.0
    avg_lat = (total_latency / len(samples)) if samples else 0.0

    summary = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "model_name": "gemma-4-12b",
        "total_articles": len(samples),
        "critic_passed": critic_passed,
        "critic_pass_pct": round(pass_pct, 2),
        "total_latency_seconds": round(total_latency, 2),
        "avg_latency_seconds": round(avg_lat, 2),
        "articles": results
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Shard {args.shard} finished: {critic_passed}/{len(samples)} passed critic ({pass_pct:.1f}%) in {total_latency:.1f}s")

if __name__ == "__main__":
    main()
