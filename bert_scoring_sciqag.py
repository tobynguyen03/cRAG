import argparse
import json
import os

import bert_score
import logging
import transformers
from bert_score import BERTScorer

def load_answers(directory):
    """
    Load ground truth and system answers from .json files in the specified directory.
    
    Args:
        directory (str): Path to the directory containing .json files.

    Returns:
        tuple: Two lists - one with ground truth answers and one with system answers.
    """
    ground_truths = []
    system_answers = []

    # Iterate over all .json files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                try:
                    data = json.load(file)
                    # Append the ground truth and system answer to respective lists
                    ground_truths.append(data.get("ground_truth", ""))
                    system_answers.append(data.get("system_answer", ""))
                except json.JSONDecodeError as e:
                    print(f"Error reading {filepath}: {e}")

    return ground_truths, system_answers

def evaluate_answers(ground_truths, system_answers):
    """
    Evaluate ground truth and system answers using BERTScore.

    Args:
        ground_truths (list): List of ground truth answers.
        system_answers (list): List of system answers.

    Returns:
        tuple: Precision, Recall, and F1 scores.
    """
    # Initialize the BERTScorer
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    
    # Calculate BERTScore
    P, R, F1 = scorer.score(system_answers, ground_truths)
    
    return P.mean().item(), R.mean().item(), F1.mean().item()

if __name__ == "__main__":
    # Pass argument to directory containing .json files
    parser = argparse.ArgumentParser(description="Pass directory with .json files to be evaluated")
    parser.add_argument("--dir", type=str, required=True, help="The directory with .json files to be evaluated")
    args = parser.parse_args()

    # Load answers from the JSON files
    ground_truths, system_answers = load_answers(args.dir)

    # Note number of answers
    print(f"Found {len(ground_truths)} ground truths and {len(system_answers)} system answers.")

    # Evaluate the answers using BERTScore
    precision, recall, f1_score = evaluate_answers(ground_truths, system_answers)

    # Output the results
    print(f"BERTScore Results: {args.dir}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1_score:.4f}")

    # Write the results to a .txt file
    results_file = os.path.basename(os.path.normpath(args.dir)) + "_results.txt"
    with open(results_file, "w") as f:
        f.write(f"BERTScore Results: {args.dir}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1_score:.4f}")