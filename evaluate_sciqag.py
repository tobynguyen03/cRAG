import os
import json

from paperqa import Settings, ask
from paperqa.settings import AgentSettings

import litellm

import bert_score
import logging
import transformers
from bert_score import BERTScorer

from sciqag_utils import SciQAGData

class EvaluatorSciQAG:
    def __init__(self, llm_config, question_dataset_path, document_path, dataset_setting):
        """
        Initialize the EvaluatorSciQAG class with necessary configurations and paths.

        Args:
            llm_config (dict): Configuration for the local LLM.
            question_dataset_path (str): Path to the SciQAG dataset being used.
            document_path (str): Path to the directory containing documents for PaperQA.
            dataset_setting (str): The dataset being passed-- "final" (full data), "train", or "test".
        """
        self.llm_config = llm_config

        self.question_dataset_path = question_dataset_path
        self.document_path = document_path
        self.dataset_setting = dataset_setting

        self.qa_pairs_ground_truth = []
        self.generated_answers = []


    def load_questions(self):
        """Load questions and ground truth from the dataset."""
        self.qa_pairs_ground_truth = []

        data = SciQAGData(self.question_dataset_path)

        if self.dataset_setting == 'final':
            data.extract_papers_text_from_final()
            self.qa_pairs_ground_truth = data.qa_pairs
        elif self.dataset_setting == 'train':
            data.extract_qa_pairs_from_train()
            self.qa_pairs_ground_truth = data.qa_pairs
        elif self.dataset_setting == 'test':
            data.extract_qa_pairs_from_test()
            self.qa_pairs_ground_truth = data.qa_pairs
        else:
            print("Invalid dataset setting given. Only 'final', 'train', and 'test' are allowed.")
            return
        

    # def load_ground_truths(self):
    #     """Load ground truth answers from the dataset."""
    #     # Placeholder: Replace with actual file reading logic
    #     return []


    def answer_questions(self):
        """Iterate through each question, use PaperQA to get an answer, and save the result."""
        self.generated_answers = []

        for qa_pair in self.qa_pairs_ground_truth:
            for question, ground_truth in qa_pair.items():
                answer = ask(
                    question,
                    settings=Settings(
                        llm='ollama/llama3.2',
                        llm_config=self.llm_config,
                        summary_llm='ollama/llama3.2',
                        summary_llm_config=self.llm_config,
                        embedding='ollama/mxbai-embed-large',
                        agent=AgentSettings(
                            agent_llm='ollama/llama3.2',
                            agent_llm_config=self.llm_config,
                        ),
                        use_doc_details=False,
                        paper_directory=self.document_path,
                    ),
                )

                self.generated_answers.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "system_answer": answer
                })


    def evaluate_answers(self):
        """Evaluate the answers generated by PaperQA against the ground truths with BERTScore."""
        ground_and_generated_answers = [(answer["system_answer"], answer["ground_truth"]) for answer in self.generated_answers]
        ground_truths, system_answers = zip(*ground_and_generated_answers)

        scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        P, R, F1 = scorer.score(system_answers, ground_truths)

        return P, R, F1


    def save_results_to_json(self, output_path):
        """Save self.generated_answers to a JSON file in a readable format."""
        with open(output_path, 'w') as file:
            # Use indent for pretty-printing
            json.dump(self.generated_answers, file, indent=4)
        print(f"Results saved to {output_path}")


# Example usage: (This part would be in the main file, not in the Evaluator class)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate PaperQA with a question dataset.")
    parser.add_argument("--question_dataset_path", type=str, required=True, help="Path to the question dataset.")
    parser.add_argument("--document_path", type=str, required=True, help="Path to the directory containing documents.")
    parser.add_argument("--dataset_setting", type=str, required=True, help="Pass which SciQAG dataset is being used.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated answers.")

    args = parser.parse_args()

    local_llm_config = {
        "model_list": [
            {
                "model_name": "ollama/llama3.2",
                "litellm_params": {
                    "model": "ollama/llama3.2",
                    "api_base": "http://localhost:11434",
                },
            }
        ]
    }

    evaluator = EvaluatorSciQAG(
        llm_config=local_llm_config,
        question_dataset_path=args.question_dataset_path,
        document_path=args.document_path,
        dataset_setting=args.dataset_setting
    )

    evaluator.answer_questions()
    evaluator.save_results_to_json(args.output_path)
    # Note: Call evaluator.evaluate_answers() once evaluation logic is implemented