import os
import json
import sys
import re 

from datetime import datetime

from itertools import islice

from paperqa import Settings, ask, ask_multiagent
from paperqa.settings import AgentSettings


from paperqa.prompts import EVAL_MULTIAGENT_CONSENSUS_PROMPT


from typing import List

import litellm

import bert_score
import logging
import transformers
from bert_score import BERTScorer

from sciqag_utils import SciQAGData

from paperqa.llms import LiteLLMModel

from paperqa.utils import get_loop


# class EvaluatorSciQAG:
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
    
    
class EvaluatorSciQAG:
    def __init__(self, args, llm_config, questions_dir, documents_dir):
        """
        Initialize the EvaluatorSciQAG class with necessary configurations and paths.

        Args:
            llm_config (dict): Configuration for the local LLM.
            questions_dir (str): Path to the SciQAG dataset being used.
            documents_dir (str): Path to the directory containing documents for PaperQA.
            answers_output_dir (str):
        """
        
        self.args = args
        self.llm_config = llm_config

        self.questions_dir = questions_dir
        self.documents_dir = documents_dir
        
        if self.args.method == "paperqa":
            self.experiment_name = f'{self.args.method}_{self.args.llm_model}'
        elif self.args.method == "paperqa_multiagent":
            self.experiment_name = f'{self.args.method}_{self.args.llm_model}_{self.args.num_agents}agents_{self.args.num_rounds}rounds' 
        
        self.answers_output_dir = f'datasets/SciQAG/answer_results/{self.experiment_name}'
        
        
        # setup query timeout limits for different models and methods 
        if self.args.llm_model == "llama3.2":
            if self.args.method == "paperqa":
                self.args.query_timeout = 180
            elif self.args.method == "paperqa_multiagent":
                self.args.query_timeout = 300
                
            # self.args.query_timeout = 180
            
        elif self.args.llm_model == "llama3.1": # takes more than 180 to load this model             
            if self.args.method == "paperqa":
                self.args.query_timeout = 300
            elif self.args.method == "paperqa_multiagent":
                self.args.query_timeout = 600
            
            

    def answer_all_questions(self):
        """
        Answer all questions in the dataset and save the results to the output directory.
        """
        
        print(f"Running experiment: sciqag_{self.experiment_name}")
        print(f"saving results to: {self.answers_output_dir} \n\n")
        
        os.makedirs(self.answers_output_dir, exist_ok=True)

        for question_file in sorted(os.listdir(self.questions_dir)):
            if question_file.endswith(".json"):
                # Create the corresponding output file name
                output_file_name = question_file.replace("sciqag_question_", "sciqag_answer_")
                output_path = os.path.join(self.answers_output_dir, output_file_name)

                # Skip processing if the output file already exists
                if os.path.exists(output_path):
                    print(f"\n Skipping {question_file} as {output_file_name} already exists. \n")
                    continue

                question_path = os.path.join(self.questions_dir, question_file)

                # Generate the answer using the updated answer_question function
                print(f"\n\n ANSWERING QUESTION: {question_file}\n ----------------------------- \n\n")
                
                if self.args.method == "paperqa":
                    success, answer_data = self.answer_question_paperqa(question_path)

                else: 
                    success, answer_data = self.answer_question_paperqa_multiagent(question_path)

                if success:
                    with open(output_path, 'w') as output_file:
                        json.dump(answer_data, output_file, indent=4)
        
    
    def answer_question_paperqa(self, question_file):
        """Pass the question to be asked to PaperQA and return the result as a formatted dictionary"""
        
        # Load the question file
        with open(question_file, 'r') as file:
            question_data = json.load(file)

        # Extract the question and ground truth
        question = list(question_data.keys())[0]
        ground_truth = question_data[question]

        # Pass question to PaperQA to answer
        success, answer = ask(
            question,
            settings=Settings(
                llm=f'ollama/{self.args.llm_model}',
                llm_config=self.llm_config,
                summary_llm=f'ollama/{self.args.llm_model}',
                summary_llm_config=self.llm_config,
                embedding='ollama/mxbai-embed-large',
                agent=AgentSettings(
                    agent_llm=f'ollama/{self.args.llm_model}',
                    agent_llm_config=self.llm_config,
                ),
                use_doc_details=False,
                paper_directory=self.documents_dir,
            ),
            timeout=self.args.query_timeout,
        )
        
        if not success:
            print(f"Failed to answer question due to timeout: {question}")
            return False, None
        else:
        
            full_answer = answer.session.answer
            match = re.search(r"ANSWER SUMMARY:\s*(.*)", full_answer, re.DOTALL)
            
            if match:
                answer_summary = match.group(1)
                success = True
            else:
                success = False
                answer_summary = "No match found."        
        
            formatted_answer = {"question": question,
                                "ground_truth": ground_truth,
                                "system_answer": answer_summary, 
                                }
            
            return success, formatted_answer

    
    #############################################################################################        
    def answer_question_paperqa_multiagent(self, question_file):
        """Pass the question to be asked to PaperQA and return the result as a formatted dictionary"""
        
        with open(question_file, 'r') as file:
            question_data = json.load(file)

        question = list(question_data.keys())[0]
        ground_truth = question_data[question]
        
        success, agents = ask_multiagent(
            question,
            settings=Settings(
                llm=f'ollama/{self.args.llm_model}',
                llm_config=self.llm_config,
                summary_llm=f'ollama/{self.args.llm_model}',
                summary_llm_config=self.llm_config,
                embedding='ollama/mxbai-embed-large',
                agent=AgentSettings(
                    agent_llm=f'ollama/{self.args.llm_model}',
                    agent_llm_config=self.llm_config,
                ),
                use_doc_details=False,
                paper_directory=self.documents_dir,
            ),
            
            num_agents=self.args.num_agents,
            num_rounds=self.args.num_rounds,
            
            timeout=self.args.query_timeout,
            
        )
            
        # bring in LLM here to summarize the answers from the agents into a final consensus answer
        if success: 
            eval_model_multiagent_consensus = self._init_eval_model_multiagent_consensus()                
            async def get_multiagent_consensus(question: str, multiagent_answers: List[str]) -> str:
                
                multiagent_answers_input = ""
                for agent_idx, agent_answer in enumerate(multiagent_answers):
                    multiagent_answers_input += f"Agent {agent_idx}: {agent_answer}\n"
                    
                output = await eval_model_multiagent_consensus.achat(
                    messages=[
                        {
                            "role": "user",
                            "content": EVAL_MULTIAGENT_CONSENSUS_PROMPT.format(
                                question=question, multiagent_answers_input=multiagent_answers_input,
                            ),
                        }
                    ],
                )
                
                final_answer_match = re.search(r'(?i)final answer:\s*(.*)', output.text, re.DOTALL)

                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
                    return True, final_answer
                else:
                    return False, None

            multiagent_answers = [agent.summarization.answer for agent in agents]
            
            success, final_answer = get_loop().run_until_complete(get_multiagent_consensus(question, multiagent_answers))
            
            if success:
                formatted_answer = {"question": question, "ground_truth": ground_truth, "system_answer": final_answer}
                return success, formatted_answer
            else: 
                print("Failed to extract final answer from multiagent consensus.")  
                return False, None
            
        
        else: 
            return False, None
            
        


        return success, formatted_answer
    #############################################################################################


    def _init_llm_config(self):
        return dict(
            model_list=[
                dict(
                    model_name=f"ollama/{self.args.llm_model}",
                    litellm_params=dict(
                        model=f"ollama/{self.args.llm_model}",
                        api_base="http://localhost:11434", 
                    ),
                )
            ]
        )
        



    def _init_eval_model_multiagent_consensus(self):
        return LiteLLMModel(
            name=f"ollama/{self.args.llm_model}",
            config=self.llm_config,
        )
        
        
        
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
    # parser.add_argument("--dataset_setting", type=str, required=True, help="Pass which SciQAG dataset is being used.")
    # parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated answers.")
    # parser.add_argument("--paper_batch_size", type=str, required=False, help="Number of papers (x10 questions) to evaluate at a time.")
    
    
    parser.add_argument("--method", type=str, default="llama3.2", help="method to use (paperqa or paperqa_multiagent)")
    parser.add_argument("--llm_model", type=str, default="llama3.2", help="Name of the LLM model to use (default: llama).")

    # num_agents, num_rounds 
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents to use in multi-agent setting.")
    parser.add_argument("--num_rounds", type=int, default=2, help="Number of rounds to use in multi-agent setting.")
    
    args = parser.parse_args()
    
    # TODO: add in arguments to specify llm model?
    local_llm_config = {
        "model_list": [
            {
                "model_name": f"ollama/{args.llm_model}",
                "litellm_params": {
                    "model": f"ollama/{args.llm_model}",
                    "api_base": "http://localhost:11434",
                },
            }
        ]
    }

    evaluator = EvaluatorSciQAG(
        args,      
        llm_config=local_llm_config, 
        questions_dir=args.question_dataset_path, 
        documents_dir=args.document_path, 
    )
    
    
    timestamp = datetime.now().strftime("%m-%d-%y-%H-%M")
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/output_sciqag_{evaluator.experiment_name}_{timestamp}.txt"

    # Open the file and redirect stdout to it
    sys.stdout = open(log_filename, "w")


    evaluator.answer_all_questions()




















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
