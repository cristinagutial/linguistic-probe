"""
linguistic_probe.py
====================
Cross-lingual NPI (Negative Polarity Item) probe using mBERT.

Background
----------
NPIs are words like "any", "ever", "yet" (EN) or "nadie", "nunca", "jamás" (ES)
that are grammatical only in certain *licensing* contexts — negation, questions,
and conditionals. Crucially, the licensor may appear far from the NPI itself
(e.g. "The teacher didn't think that ANY student had cheated"), so this probe
tests long-distance syntactic/semantic sensitivity, not just local patterns.

Method: Minimal Pair Scoring
-----------------------------
For each pair, [MASK] is placed at the NPI position. We compare:

    Δ log p = log p(NPI | licensed context) − log p(NPI | unlicensed context)

If Δ > 0, mBERT assigns higher probability to the NPI in the grammatically
correct (licensed) context — a "pass". Because both sentences use the *same*
target word, this isolates exactly the effect of the licensing context.

Multi-token NPIs (e.g. "en absoluto", "at all") are handled via iterative
masked scoring: each sub-token is scored in sequence, conditioning on
previously predicted sub-tokens.

Usage
-----
  python linguistic_probe.py
  python linguistic_probe.py --data data/sentences.json --output results/
  python linguistic_probe.py --language es
  python linguistic_probe.py --licensor question
"""

import json
import argparse
import os
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "bert-base-multilingual-cased"
MASK_TOKEN = "[MASK]"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str = MODEL_NAME):
    """Load mBERT tokenizer and masked language model."""
    print(f"Loading model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  → Running on: {device}\n")
    return tokenizer, model, device


# ── Scoring ───────────────────────────────────────────────────────────────────

def get_token_log_prob(sentence: str, target_word: str,
                       tokenizer, model, device) -> float:
    """
    Return the summed log-probability of `target_word` appearing at the
    [MASK] position in `sentence`.

    Multi-token targets (e.g. "at all" → ["at", "all"], or Spanish
    "absoluto" with sub-word tokenization) are scored iteratively:
    each sub-token is scored while conditioning on the previous ones
    already filled in. Log-probs are summed across sub-tokens.

    This is important for NPI probing because many strong NPIs are
    multi-word expressions ("at all", "en absoluto", "lift a finger").
    """
    target_tokens = tokenizer.tokenize(target_word)

    # Replace [MASK] placeholder with one [MASK] per sub-token
    multi_mask = " ".join([MASK_TOKEN] * len(target_tokens))
    sentence_filled = sentence.replace(MASK_TOKEN, multi_mask)

    inputs = tokenizer(sentence_filled, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]

    mask_token_id = tokenizer.mask_token_id
    mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0].tolist()

    # Guard: truncate if tokenization produced a mismatch
    n = min(len(mask_positions), len(target_tokens))
    mask_positions = mask_positions[:n]
    target_tokens  = target_tokens[:n]

    total_log_prob = 0.0

    with torch.no_grad():
        for i, (mask_pos, sub_token) in enumerate(zip(mask_positions, target_tokens)):
            current_ids = input_ids.clone()

            # Fill in sub-tokens already scored in previous iterations
            for j in range(i):
                fill_id = tokenizer.convert_tokens_to_ids(target_tokens[j])
                current_ids[mask_positions[j]] = fill_id

            outputs = model(current_ids.unsqueeze(0))
            logits  = outputs.logits[0, mask_pos]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            sub_token_id = tokenizer.convert_tokens_to_ids(sub_token)
            total_log_prob += log_probs[sub_token_id].item()

    return total_log_prob


def score_pair(pair: dict, tokenizer, model, device) -> dict:
    """
    Score a single NPI minimal pair.

    Key NPI insight: both grammatical and ungrammatical sentences use the
    *same* target word (the NPI). The only difference is the licensing
    context (e.g. 'doesn't have any' vs 'does have any'). So Δ log p
    directly measures how much the model is sensitive to that context.
    """
    logp_licensed = get_token_log_prob(
        pair["grammatical"], pair["target_grammatical"],
        tokenizer, model, device
    )
    logp_unlicensed = get_token_log_prob(
        pair["ungrammatical"], pair["target_ungrammatical"],
        tokenizer, model, device
    )

    correct   = logp_licensed > logp_unlicensed
    score_diff = logp_licensed - logp_unlicensed

    return {
        "id":                     pair["id"],
        "language":               pair["language"],
        "phenomenon":             pair["phenomenon"],
        "npi":                    pair.get("npi", ""),
        "licensor":               pair.get("licensor", ""),
        "licensed_sentence":      pair["grammatical"],
        "unlicensed_sentence":    pair["ungrammatical"],
        "target_npi":             pair["target_grammatical"],
        "logp_licensed":          round(logp_licensed, 4),
        "logp_unlicensed":        round(logp_unlicensed, 4),
        "log_prob_diff":          round(score_diff, 4),
        "model_correct":          correct,
        "note":                   pair.get("note", "")
    }


# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_accuracy(results: list[dict]) -> pd.DataFrame:
    """
    Compute accuracy broken down in three ways:
      1. By language (EN vs ES)
      2. By licensor type (negation / question / conditional / long-distance)
      3. By language × licensor
    """
    rows = []
    grouped = defaultdict(list)

    for r in results:
        grouped[("language",  r["language"],            "—")].append(r["model_correct"])
        grouped[("licensor",  "—",          r["licensor"])].append(r["model_correct"])
        grouped[("lang×lic",  r["language"], r["licensor"])].append(r["model_correct"])

    for (grouping, lang, lic), correct_list in grouped.items():
        acc = sum(correct_list) / len(correct_list)
        rows.append({
            "grouping":  grouping,
            "language":  lang,
            "licensor":  lic,
            "n_pairs":   len(correct_list),
            "n_correct": sum(correct_list),
            "accuracy":  round(acc, 3)
        })

    df = pd.DataFrame(rows).sort_values(["grouping", "language", "licensor"])
    return df


def print_summary(results: list[dict], accuracy_df: pd.DataFrame):
    """Print a human-readable summary to the terminal."""
    total   = len(results)
    correct = sum(r["model_correct"] for r in results)

    print("=" * 65)
    print(f"  NPI PROBE RESULTS — {MODEL_NAME}")
    print("=" * 65)
    print(f"\n  Overall accuracy: {correct}/{total} = {correct/total:.1%}\n")

    for grouping in ["language", "licensor", "lang×lic"]:
        subset = accuracy_df[accuracy_df["grouping"] == grouping]
        print(f"  ── By {grouping} ──")
        print(subset[["language", "licensor", "n_pairs", "accuracy"]].to_string(index=False))
        print()

    print("─" * 65)
    print("  Per-pair breakdown:")
    print("─" * 65)
    for r in results:
        status = "✓" if r["model_correct"] else "✗"
        print(f"\n  [{status}] {r['id']}  (NPI: '{r['target_npi']}'  |  licensor: {r['licensor']})")
        print(f"       licensed  : {r['licensed_sentence']}")
        print(f"       unlicensed: {r['unlicensed_sentence']}")
        print(f"       log p (licensed)   = {r['logp_licensed']}")
        print(f"       log p (unlicensed) = {r['logp_unlicensed']}")
        print(f"       Δ = {r['log_prob_diff']:+.4f}  |  {r['note']}")
    print()


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_data(path: str, language: str = None, licensor: str = None) -> list[dict]:
    """Load sentence pairs from JSON, optionally filtered."""
    with open(path) as f:
        data = json.load(f)
    pairs = data["pairs"]
    if language:
        pairs = [p for p in pairs if p["language"] == language]
    if licensor:
        pairs = [p for p in pairs if p.get("licensor") == licensor]
    print(f"Loaded {len(pairs)} sentence pair(s) from '{path}'")
    if language:
        print(f"  → Filtered to language: {language}")
    if licensor:
        print(f"  → Filtered to licensor: {licensor}")
    print()
    return pairs


def save_results(results: list[dict], accuracy_df: pd.DataFrame, output_dir: str):
    """Save raw results and accuracy breakdown to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    results_path  = os.path.join(output_dir, "results_raw.csv")
    accuracy_path = os.path.join(output_dir, "results_accuracy.csv")

    pd.DataFrame(results).to_csv(results_path, index=False)
    accuracy_df.to_csv(accuracy_path, index=False)

    print(f"\nResults saved to:")
    print(f"  {results_path}")
    print(f"  {accuracy_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-lingual NPI probe with mBERT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python linguistic_probe.py
  python linguistic_probe.py --language es
  python linguistic_probe.py --licensor negation_long_distance
  python linguistic_probe.py --language en --licensor question
        """
    )
    parser.add_argument("--data",     default="data/sentences.json", help="Path to sentence pairs JSON")
    parser.add_argument("--output",   default="results/",            help="Directory to save results")
    parser.add_argument("--language", default=None,                  help="Filter by language: en / es")
    parser.add_argument("--licensor", default=None,                  help="Filter by licensor type")
    args = parser.parse_args()

    pairs                  = load_data(args.data, args.language, args.licensor)
    tokenizer, model, device = load_model()

    print("Scoring sentence pairs...")
    results = []
    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {pair['id']}  ({pair.get('licensor', '')})")
        result = score_pair(pair, tokenizer, model, device)
        results.append(result)

    accuracy_df = compute_accuracy(results)
    print_summary(results, accuracy_df)
    save_results(results, accuracy_df, args.output)


if __name__ == "__main__":
    main()
