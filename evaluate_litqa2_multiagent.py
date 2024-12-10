import os
import json
import time
import re
import asyncio
from pathlib import Path
import pandas as pd
import argparse

from datetime import datetime

import sys


from enum import Enum
from typing import cast

from litqa2_utils import process_question_df

from aviary.env import TaskDataset

from paperqa import QueryRequest, Settings, ask, ask_multiagent
from paperqa.settings import AgentSettings
from paperqa.agents.task import TASK_DATASET_NAME
from paperqa.llms import LiteLLMModel
from paperqa.litqa import LitQAEvaluation
from paperqa.utils import get_loop


from paperqa.prompts import EVAL_MULTIAGENT_CONSENSUS_PROMPT

from typing import List




class LitQAEvaluator:
    def __init__(self, args,  batch_size=1):
        """
        """
        
        self.args = args 
        
        self.model = f"ollama/{args.model}"       
        self.document_path = args.paper_directory
        
        if self.args.method == "paperqa":
            self.experiment_name = f'{self.args.method}_{self.args.model}'
        elif self.args.method == "paperqa_multiagent":
            self.experiment_name = f'{self.args.method}_{self.args.model}_{self.args.num_agents}agents_{self.args.num_rounds}rounds' 
            
            
            
            
            
        
        # self.output_path = Path(os.path.join(args.output_directory, f"{self.experiment_name}.txt"))
        # self.checkpoint_path = Path(os.path.join(args.checkpoint_directory, f"{self.experiment_name}.txt"))

        os.makedirs("litqa_single_agent_output", exist_ok=True)
        self.output_dir = os.path.join("litqa_single_agent_output", self.experiment_name)
 



        self.batch_size = batch_size

        self.llm_config = self._init_llm_config()
        self.llm_settings = self._init_llm_settings()
        self.eval_model = self._init_eval_model()
        
        self.question_data = self._load_litqa_questions()
    
    
        
        # self.processed_questions = self._load_checkpoint()
    
    
    
        # setup query timeout limits for different models and methods 
        if self.args.model == "llama3.2":
            if self.args.method == "paperqa":
                self.args.query_timeout = 60
            elif self.args.method == "paperqa_multiagent":
                self.args.query_timeout = 300
                
            # self.args.query_timeout = 180
            
        elif self.args.model == "llama3.1": # takes more than 180 to load this model             
            if self.args.method == "paperqa":
                self.args.query_timeout = 300
            elif self.args.method == "paperqa_multiagent":
                self.args.query_timeout = 600

        elif self.args.model == "llama3.3": # takes more than 180 to load this model             
            if self.args.method == "paperqa":
                self.args.query_timeout = 300
            elif self.args.method == "paperqa_multiagent":
                self.args.query_timeout = 500
            
            
    def _init_llm_config(self):
        return dict(
            model_list=[
            dict(
                model_name=self.model,
                litellm_params=dict(
                model=self.model,
                api_base=f"http://localhost:{self.args.port}", 
                ),
            )
            ]
        )

    def _init_llm_settings(self):




        settings = Settings(
            llm=self.model,
            llm_config=self.llm_config,
            
            summary_llm=self.model,
            summary_llm_config=self.llm_config,
            
            embedding='ollama/mxbai-embed-large',
            embedding_config={'kwargs': {'api_base': f"http://localhost:{self.args.port}"}},

            agent=AgentSettings(
                agent_llm=self.model, 
                agent_llm_config=self.llm_config
            ),
            use_doc_details=False,
            paper_directory=self.document_path
        )
        
        return settings
    
    def _init_eval_model(self):
        return LiteLLMModel(
            name=self.model,
            config=self.llm_config,
        )

    def _init_eval_model_multiagent_consensus(self):
        return LiteLLMModel(
            name=f"ollama/{self.args.model}",
            config=self.llm_config,
        )


    def _load_litqa_questions(self):
        """Load questions and ground truth from the dataset."""
        base_query = QueryRequest(
            settings=self.llm_settings
        )
        dataset = TaskDataset.from_name(TASK_DATASET_NAME, base_query=base_query)
        formatted_dataset = process_question_df(dataset.data)

        return formatted_dataset
    
    
    
    def check_progress(self, question_num):
        """Load previously processed questions from checkpoint if it exists."""
        checkpoint_file = Path(f"{self.output_dir}_question_{question_num}.txt")
        return checkpoint_file.exists()
    
    
    def save_progress(self, data, question_num):
        """Save current progress to output file."""
        results_df = pd.DataFrame.from_dict(data, orient='index')
        output_path = Path(f"{self.output_dir}_question_{question_num}.txt")
        results_df.to_csv(output_path)
    
    
    # def _load_checkpoint(self):
    #     """Load previously processed questions from checkpoint if it exists."""
    #     if self.checkpoint_path.exists():
    #         with open(self.checkpoint_path, 'r') as f:
    #             try:
    #                 return json.load(f)
    #             except Exception as e:
    #                 return {}
    #     return {}
    
    # def _save_checkpoint(self):
    #     """Save current progress to checkpoint file."""
    #     with open(self.checkpoint_path, 'w') as f:
    #         json.dump(self.processed_questions, f)
            
            
            
            
            
            
            
            
            
            
    def _save_results(self):
        """Save current results to the output file."""
        # Convert processed questions to DataFrame
        results_df = pd.DataFrame.from_dict(self.processed_questions, orient='index')
        results_df.to_csv(self.output_path)

    def answer_questions(self):
        """
        Answers questions using the provided LLM function.
        """
        
        print(f"Running experiment: liqa2_{self.experiment_name}")
        print(f"saving results to: {self.output_dir} \n\n")

        questions_processed = 0
        
        try:
            for index, row in self.question_data.iterrows():
                
                
                
                # if str(index) in self.processed_questions:
                #     print(f"Skipping already processed question {index}")
                #     continue
                
                if self.check_progress(index):
                    print(f"Skipping already processed question {index}")
                    continue
                
                print(f"Processing question {index}")
                
                
                # Generate the answer using the updated answer_question function
                print(f"\n\n ANSWERING QUESTION: {index}\n ----------------------------- \n\n")
                
                
                
                try:
                    
                    question, evaluate_answer = LitQAEvaluation.from_question(
                        ideal=cast(str, row.ideal),
                        distractors=cast(list[str], row.distractors),
                        question=cast(str, row.question),
                        eval_model=self.eval_model,
                        seed=42,
                    )

                    
                    if self.args.method == "paperqa":
                        
                        _, answer = ask(
                            question,
                            settings=self.llm_settings
                        )
                        evaluation_result, answer_choice = get_loop().run_until_complete(evaluate_answer(answer))
                        print(question)
                        print("LLM Response: ", answer_choice, evaluation_result)

                        question_base = question.split('Options:')[0].strip()
                        
                        # # Store result
                        # self.processed_questions[str(index)] = {
                        #     'question': question_base,
                        #     'answer': answer_choice,
                        #     'ground_truth': row.ideal,
                        #     'result': evaluation_result,
                        #     'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        # }
                        
                        questions_processed += 1
                        
                        # # Save checkpoint after each batch
                        # if questions_processed % self.batch_size == 0:
                        #     print(f"Saving checkpoint after {questions_processed} questions...")
                        #     self._save_checkpoint()
                        #     self._save_results()
                        full_answer = answer if isinstance(answer, str) else "Insufficient information to answer this question" if answer == None else answer.formatted_answer if hasattr(answer, 'formatted_answer') else answer.session.formatted_answer
                        
                                            
                        data = {
                            'question': question_base,
                            'full_answer': full_answer,
                            'answer_letter': answer_choice,
                            'ground_truth': row.ideal,
                            'result': evaluation_result,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        
                        
                        questions_processed += 1
                        
                        self.save_progress(data, index)
                        
                        
                    
                    ####################################################################################################
                    else: 
                        
                        
                        
                        
                        success, agents = ask_multiagent(
                            question,
                            settings=self.llm_settings, 
                            num_agents=self.args.num_agents,
                            num_rounds=self.args.num_rounds,   
                            timeout=self.args.query_timeout,
                        )
                        
                        
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
                                
                                return output.text      
                            
                            
                            
                            multiagent_answers = [agent.summarization.answer for agent in agents]
                            consensus_answer = get_loop().run_until_complete(get_multiagent_consensus(question, multiagent_answers))
                            
                            
                            print("Consensus Answer: ", consensus_answer) 

                            # extracts the answer choice from the agent consensus response 
                            evaluation_result, answer_choice = get_loop().run_until_complete(evaluate_answer(consensus_answer))
                            print(question)
                            print("LLM Response: ", answer_choice, evaluation_result)

                            question_base = question.split('Options:')[0].strip()
                            
                            # Store result
                            # self.processed_questions[str(index)] = {
                            #     'question': question_base,
                            #     'answer': answer_choice,
                            #     'ground_truth': row.ideal,
                            #     'result': evaluation_result,
                            #     'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                            # }
                            full_answer = answer if isinstance(answer, str) else "Insufficient information to answer this question" if answer == None else answer.formatted_answer if hasattr(answer, 'formatted_answer') else answer.session.formatted_answer
                            
                            
                            full_answer = consensus_answer
                            
                            data = {
                                'question': question_base,
                                'full_answer': full_answer,
                                'answer_letter': answer_choice,
                                'ground_truth': row.ideal,
                                'result': evaluation_result,
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                            
                            
                            questions_processed += 1
                            
                            self.save_progress(data, index)

                            
                            
                            # # Save checkpoint after each batch
                            # if questions_processed % self.batch_size == 0:
                            #     print(f"Saving checkpoint after {questions_processed} questions...")
                            #     self._save_checkpoint()
                            #     self._save_results()                        
                            
                            self.save_progress(data, index)
                            
                            
                            
                            
                            
                        else: 
                            print(f"Error processing question {index}: {str(e)}")
                            # Save progress even if there's an error
                            # self._save_checkpoint()
                            # self._save_results()
                            continue
                
                except Exception as e:
                    print(f"Error processing question {index}: {str(e)}")
                    # Save progress even if there's an error
                    # self._save_checkpoint()
                    # self._save_results()
                    continue
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving progress...")
            # self._save_checkpoint()
            # self._save_results()
            return
        
        # Final save
        # self._save_checkpoint()
        # self._save_results()
        print("All questions processed!")


    def evaluate_answers(self):
        """Evaluate the answers generated by PaperQA against the ground truths."""
        return 0 #TODO


    def save_results_to_json(self, output_path):
        """Save self.generated_answers to a JSON file in a readable format."""
        with open(output_path, 'w') as file:
            # Use indent for pretty-printing
            json.dump(self.generated_answers, file, indent=4)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Evaluation on the L.")

    parser.add_argument("--model", type=str, default="llama3.2", help="The model to use for evaluation.")
    parser.add_argument("--paper_directory", type=str, default="my_papers", help="The directory containing the papers.")
    parser.add_argument("--output_directory", type=str, default="litqa_output", help="The directory to save the output.")
    parser.add_argument("--checkpoint_directory", type=str, default="litqa_checkpoints", help="The directory to save the checkpoints.")
    
    parser.add_argument("--method", type=str, default="llama3.2", help="method to use (paperqa or paperqa_multiagent)")
    
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents to use in multi-agent setting.")
    parser.add_argument("--num_rounds", type=int, default=2, help="Number of rounds to use in multi-agent setting.")
    parser.add_argument("--port", type=str, default="11434", help="method to use (paperqa or paperqa_multiagent)")

    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)

    evaluator = LitQAEvaluator(args)


    timestamp = datetime.now().strftime("%m-%d-%y-%H-%M")
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/output_litqa2_{evaluator.experiment_name}_{timestamp}.txt"

    # Open the file and redirect stdout to it
    sys.stdout = open(log_filename, "w")



    evaluator.answer_questions()
    