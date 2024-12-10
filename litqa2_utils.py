import os
import random
from string import ascii_uppercase
import pandas as pd

from urllib.parse import urlparse
from aviary.env import TaskDataset

from dotenv import load_dotenv

from paperqa import Docs, QueryRequest, Settings
from paperqa.agents.task import TASK_DATASET_NAME
# from paperqa.contrib import ZoteroDB

def extract_paper_urls():
    base_query = QueryRequest(
        settings=Settings(paper_directory="my-papers")
    )
    dataset = TaskDataset.from_name(TASK_DATASET_NAME, base_query=base_query)
    unique_sources = set([source for sources in dataset.data['sources'] for source in sources])

    with open('paper_urls.txt', 'w') as f:
        for source in unique_sources:
            f.write(f"{source}\n")

    return unique_sources

def download_pdfs_from_zotero(paper_dir: str):
    load_dotenv()
    zotero = ZoteroDB(library_type="user", storage=paper_dir)
    total_papers = 0

    print(f"Downloading papers to {paper_dir}")

    for item in zotero.iterate(limit=100):
        total_papers += 1

    print(f"Successfully downloaded {total_papers} papers to {paper_dir}")


def process_question_df(df):
    def process_row(row):
        ideal_answers = [ans.strip() for ans in str(row['ideal']).split(',')]
        distractors = row['distractors']
        all_choices = ideal_answers + distractors
        random.shuffle(all_choices)

        lettered_choices = [f"{ascii_uppercase[i]}: {choice}" 
                          for i, choice in enumerate(all_choices)]
        
        correct_answer = [ascii_uppercase[i] 
                         for i, choice in enumerate(all_choices) 
                         if choice in ideal_answers]
        
        return pd.Series({
            'answer_choices': lettered_choices,
            'correct_answer': correct_answer
        })
    
    result_df = df.join(df.apply(process_row, axis=1)).sort_index()
    with open('paper_urls.txt', 'r') as file:
        urls = set([line.strip() for line in file])
    
    filtered_df = result_df[result_df['sources'].apply(lambda x: x[0] in urls)]

    return filtered_df

if __name__ == "__main__":
    paper_dir = "my_papers"
    download_pdfs_from_zotero(paper_dir)