import os
import json
from itertools import islice

from paperqa import Settings, ask
from paperqa.settings import AgentSettings

import litellm

import bert_score
import logging
import transformers
from bert_score import BERTScorer

from sciqag_utils import SciQAGData

class EvaluatorSciQAG:
    # def __init__(self, llm_config, question_dataset_path, document_path, dataset_setting):
    #     """
    #     Initialize the EvaluatorSciQAG class with necessary configurations and paths.

    #     Args:
    #         llm_config (dict): Configuration for the local LLM.
    #         question_dataset_path (str): Path to the SciQAG dataset being used.
    #         document_path (str): Path to the directory containing documents for PaperQA.
    #         dataset_setting (str): The dataset being passed--
    #             "final" (full data), "final_short" (full but truncated data), "train", or "test".
    #     """
    #     self.llm_config = llm_config

    #     self.question_dataset_path = question_dataset_path
    #     self.document_path = document_path
    #     self.dataset_setting = dataset_setting

    #     self.qa_pairs_ground_truth = []
    #     self.generated_answers = []


    # def load_questions(self):
    #     """Load questions and ground truth from the dataset."""
    #     if self.dataset_setting == 'final_short': # Manually load instead
    #         return
        
    #     self.qa_pairs_ground_truth = []

    #     data = SciQAGData(self.question_dataset_path)
    #     data.load_data()

    #     if self.dataset_setting == 'final':
    #         data.extract_qa_pairs_from_final()
    #         self.qa_pairs_ground_truth = data.qa_pairs
    #     elif self.dataset_setting == 'train':
    #         data.extract_qa_pairs_from_train()
    #         self.qa_pairs_ground_truth = data.qa_pairs
    #     elif self.dataset_setting == 'test':
    #         data.extract_qa_pairs_from_test()
    #         self.qa_pairs_ground_truth = data.qa_pairs
    #     else:
    #         print("Invalid dataset setting given. Only 'final_short', 'final', 'train', 'test' are allowed.")
    #         return
        

    # def load_ground_truths(self):
    #     """Load ground truth answers from the dataset."""
    #     # Placeholder: Replace with actual file reading logic
    #     return []


    # def answer_questions(self):
    #     """Iterate through each question, use PaperQA to get an answer, and save the result."""
    #     self.generated_answers = []

    #     for qa_pair in self.qa_pairs_ground_truth:
    #         for question, ground_truth in qa_pair.items():
    #             answer = ask(
    #                 question,
    #                 settings=Settings(
    #                     llm='ollama/llama3.2',
    #                     llm_config=self.llm_config,
    #                     summary_llm='ollama/llama3.2',
    #                     summary_llm_config=self.llm_config,
    #                     embedding='ollama/mxbai-embed-large',
    #                     agent=AgentSettings(
    #                         agent_llm='ollama/llama3.2',
    #                         agent_llm_config=self.llm_config,
    #                     ),
    #                     use_doc_details=False,
    #                     paper_directory=self.document_path,
    #                 ),
    #             )

    #             self.generated_answers.append({
    #                 "question": question,
    #                 "ground_truth": ground_truth,
    #                 "system_answer": answer
    #             })

    def __init__(self, llm_config, questions_dir, documents_dir, answers_output_dir):
        """
        Initialize the EvaluatorSciQAG class with necessary configurations and paths.

        Args:
            llm_config (dict): Configuration for the local LLM.
            questions_dir (str): Path to the SciQAG dataset being used.
            documents_dir (str): Path to the directory containing documents for PaperQA.
            answers_output_dir (str):
        """
        self.llm_config = llm_config

        self.questions_dir = questions_dir
        self.documents_dir = documents_dir
        self.answers_output_dir = answers_output_dir


    def answer_all_questions(self):
        """
        Iterate through all question files in the questions_dir, generate answers using
        answer_question, and output the results as JSON files in answers_output_dir.
        Each output file corresponds to the question file, with a modified name.
        """
        os.makedirs(self.answers_output_dir, exist_ok=True)

        for question_file in sorted(os.listdir(self.questions_dir)):
            if question_file.endswith(".json"):
                # Create the corresponding output file name
                output_file_name = question_file.replace("sciqag_question_", "sciqag_paperqa_single_agent_answer_")
                output_path = os.path.join(self.answers_output_dir, output_file_name)

                # Skip processing if the output file already exists
                if os.path.exists(output_path):
                    print(f"Skipping {question_file} as {output_file_name} already exists.")
                    continue

                question_path = os.path.join(self.questions_dir, question_file)

                # Generate the answer using the updated answer_question function
                answer_data = self.answer_question(question_path)

                # Write the answer to the output file
                with open(output_path, 'w') as output_file:
                    json.dump(answer_data, output_file, indent=4)
        
    
    def answer_question(self, question_file):
        """Pass the question to be asked to PaperQA and return the result as a formatted dictionary"""
        # Load the question file
        with open(question_file, 'r') as file:
            question_data = json.load(file)

        # Extract the question and ground truth
        question = list(question_data.keys())[0]
        ground_truth = question_data[question]

        # Pass question to PaperQA to answer
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
                paper_directory=self.documents_dir,
            ),
        )

        # Return formatted answer
        return {
                    "question": question,
                    "ground_truth": ground_truth,
                    "system_answer": answer
                }


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


# def minibatch_list(data, chunk_size):
#     """
#     Create an iterator that yields chunks of the given size.
#     """
#     it = iter(data)
#     while chunk := list(islice(it, chunk_size)):
#         yield chunk


# Example usage: (This part would be in the main file, not in the Evaluator class)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate PaperQA with a question dataset.")
    parser.add_argument("--question_dataset_path", type=str, required=True, help="Path to the question dataset.")
    parser.add_argument("--document_path", type=str, required=True, help="Path to the directory containing documents.")
    parser.add_argument("--dataset_setting", type=str, required=True, help="Pass which SciQAG dataset is being used.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated answers.")
    parser.add_argument("--paper_batch_size", type=str, required=False, help="Number of papers (x10 questions) to evaluate at a time.")

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

    # if args.dataset_setting == 'final_short':
    #     data = SciQAGData(args.question_dataset_path)
    #     data.load_data()
    #     papers_and_qa_pairs = data.extract_papers_and_qa_pairs_from_final()

    #     evaluator = EvaluatorSciQAG(
    #         llm_config=local_llm_config,
    #         question_dataset_path=args.question_dataset_path,
    #         document_path=args.document_path,
    #         dataset_setting=args.dataset_setting
    #     )

    #     os.makedirs(args.document_path, exist_ok=True)
    #     os.makedirs(args.output_path, exist_ok=True)

    #     for batch_number, batch in enumerate(minibatch_list(papers_and_qa_pairs, args.paper_batch_size), start=1):
    #         # Write papers from batch to directory
    #         for paper in batch:
    #             doi_filename = f"{paper['doi'].replace('/', '_').replace('.json', '')}.txt"
    #             filepath = os.path.join(args.document_path, doi_filename)
    #             with open(filepath, 'w') as f:
    #                 f.write(paper['txt'])
            
    #         print(f"Batch {batch_number}: Papers written to {args.document_path}")

    #         # Extract associated questions
    #         evaluator.qa_pairs_ground_truth = [qa_pair for paper in batch for qa_pair in paper['qa_pairs']]

    #         # Answer questions
    #         evaluator.answer_questions()

    #         # Write answers to output file
    #         output_filename = f"sciqag_100_batch_{batch_number}.json"
    #         output_filepath = os.path.join(args.output_path, output_filename)

    #         evaluator.save_results_to_json(output_filepath)

    #         print(f"Batch {batch_number}: Answers written to {output_filepath}")

    #         # Clean up papers in directory
    #         for paper in batch:
    #             doi_filename = f"{paper['doi'].replace('/', '_').replace('.json', '')}.txt"
    #             filepath = os.path.join(args.document_path, doi_filename)
    #             if os.path.exists(filepath):
    #                 os.remove(filepath)
            
    #         print(f"Batch {batch_number}: Papers removed from {args.document_path}")


    

    # evaluator.answer_questions()
    # evaluator.save_results_to_json(args.output_path)
    # Note: Call evaluator.evaluate_answers() once evaluation logic is implemented