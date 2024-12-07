from paperqa import Settings
from paperqa.agents.task import (
    GradablePaperQAEnvironment,
    LitQATaskDataset,
    LitQAv2TaskDataset,
    LitQAv2TaskSplit,
)
from paperqa.settings import AgentSettings
import os
import asyncio
from evaluate_litqa2 import evaluate

import litellm
litellm.set_verbose=True

os.environ['OPENAI_API_KEY'] = "ollama"

local_llm_config = dict(
    model_list=[
        dict(
            model_name='ollama/llama3.2',
            litellm_params=dict(
                model='ollama/llama3.2',
                api_base="http://localhost:11434", 
            ),
        )
    ]
)

settings = Settings(
    llm='ollama/llama3.2',
    llm_config=local_llm_config,
    
    summary_llm='ollama/llama3.2',
    summary_llm_config=local_llm_config,
    
    embedding='ollama/mxbai-embed-large',
    
    agent=AgentSettings(
        agent_llm='ollama/llama3.2', 
        agent_llm_config=local_llm_config
    ),
    use_doc_details=False,
    paper_directory="my_papers"
)

def main():
    print('test')
    evaluate(settings)

if __name__ == "__main__":
    print('test')
    main()



