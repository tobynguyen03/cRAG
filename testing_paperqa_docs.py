

from paperqa import Docs, Settings
from paperqa.settings import AgentSettings


import os 



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

settings=Settings(
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
        paper_directory="sciqag_papers_txt_only"
    )



# valid extensions include .pdf, .txt, and .html
doc_paths = ("myfile.pdf", "myotherfile.pdf")


directory_path = 'sciqag_papers_txt_only'

doc_paths = tuple(
    os.path.join(directory_path, filename) for filename in os.listdir(directory_path)
    if os.path.isfile(os.path.join(directory_path, filename))
)

# print(doc_paths)

docs = Docs()

for doc in doc_paths:
    docs.add(doc, settings=settings)

 
answer = docs.query(
    "What manufacturing challenges are unique to bispecific antibodies?",
    settings=Settings(
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
        paper_directory="sciqag_papers_txt_only"
    ),
)



print(answer.formatted_answer)




