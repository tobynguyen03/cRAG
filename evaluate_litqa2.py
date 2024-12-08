import os
import json
import time
import re
import asyncio
from pathlib import Path
import pandas as pd
from enum import Enum
from typing import cast

from litqa2_utils import process_question_df

from aviary.env import TaskDataset

from paperqa import QueryRequest, Settings, ask
from paperqa.settings import AgentSettings
from paperqa.agents.task import TASK_DATASET_NAME
from paperqa.llms import LiteLLMModel
from paperqa.litqa import LitQAEvaluation
from paperqa.utils import get_loop




class LitQAEvaluator:
    def __init__(self, model, document_path, output_dir, checkpoint_dir, batch_size=1):
        """
        Initialize the EvaluatorSciQAG class with necessary configurations and paths.

        Args:
            llm_config (dict): Configuration for the local LLM.5
            document_path (str): Path to the directory containing documents for PaperQA.
            dataset_setting (str): The dataset being passed-- "final" (full data), "train", or "test".
        """
        self.model = f"ollama/{model}"       
        self.document_path = document_path
        self.output_path = Path(os.path.join(output_dir, f"{model}.txt"))
        self.checkpoint_path = Path(os.path.join(checkpoint_dir, f"{model}.txt"))
        self.batch_size = batch_size

        self.llm_config = self._init_llm_config()
        self.llm_settings = self._init_llm_settings()
        self.eval_model = self._init_eval_model()

        self.question_data = self._load_litqa_questions()
        
        self.processed_questions = self._load_checkpoint()
    
    def _init_llm_config(self):
        return dict(
            model_list=[
                dict(
                    model_name=self.model,
                    litellm_params=dict(
                        model=self.model,
                        api_base="http://localhost:11434", 
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

    def _load_litqa_questions(self):
        """Load questions and ground truth from the dataset."""
        base_query = QueryRequest(
            settings=self.llm_settings
        )
        dataset = TaskDataset.from_name(TASK_DATASET_NAME, base_query=base_query)
        formatted_dataset = process_question_df(dataset.data)

        return formatted_dataset
    
    def _load_checkpoint(self):
        """Load previously processed questions from checkpoint if it exists."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                try:
                    return json.load(f)
                except Exception as e:
                    return {}
        return {}
    
    def _save_checkpoint(self):
        """Save current progress to checkpoint file."""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.processed_questions, f)
            
    def _save_results(self):
        """Save current results to the output file."""
        # Convert processed questions to DataFrame
        results_df = pd.DataFrame.from_dict(self.processed_questions, orient='index')
        results_df.to_csv(self.output_path)

    def answer_questions(self):
        """
        Answers questions using the provided LLM function.
        """
        questions_processed = 0
        
        try:
            for index, row in self.question_data.iterrows():
                if str(index) in self.processed_questions:
                    print(f"Skipping already processed question {index}")
                    continue
                
                print(f"Processing question {index}")
                
                try:
                    question, evaluate_answer = LitQAEvaluation.from_question(
                        ideal=cast(str, row.ideal),
                        distractors=cast(list[str], row.distractors),
                        question=cast(str, row.question),
                        eval_model=self.eval_model,
                        seed=42,
                    )

                    answer = ask(
                        question,
                        self.llm_settings
                    )

                    evaluation_result, answer_choice = get_loop().run_until_complete(evaluate_answer(answer))
                    print(question)
                    print("LLM Response: ", answer_choice, evaluation_result)

                    question_base = question.split('Options:')[0].strip()
                    
                    # Store result
                    self.processed_questions[str(index)] = {
                        'question': question_base,
                        'answer': answer_choice,
                        'ground_truth': row.ideal,
                        'result': evaluation_result,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    questions_processed += 1
                    
                    # Save checkpoint after each batch
                    if questions_processed % self.batch_size == 0:
                        print(f"Saving checkpoint after {questions_processed} questions...")
                        self._save_checkpoint()
                        self._save_results()
                        
                except Exception as e:
                    print(f"Error processing question {index}: {str(e)}")
                    # Save progress even if there's an error
                    self._save_checkpoint()
                    self._save_results()
                    continue
                
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving progress...")
            self._save_checkpoint()
            self._save_results()
            return
        
        # Final save
        self._save_checkpoint()
        self._save_results()
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
    model = "llama3.2"
    paper_directory = "my_papers"
    output_directory = "litqa_output"
    checkpoint_directory = "litqa_checkpoints"

    os.makedirs(checkpoint_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)

    evaluator = LitQAEvaluator(model, paper_directory, output_directory, checkpoint_directory)

    evaluator.answer_questions()