from paperqa import Settings, ask
from paperqa.settings import AgentSettings
import os

import litellm
# litellm.set_verbose=True

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

# questions from sciqag
questions_list = [
    "How was the diabetic rat model established in this study?",
    "How selective is the SrCO3 sensor for detecting ethanol among other gases?",
    "How does the SrCO3 material used in the study respond to ethanol vapor?",
    "What is the most commonly used plasticizer in the world?",
    "Why was sodium chloride added to the milk sample during the extraction process?",
    "Why are MIPs considered stable, less costly, and easier to produce than their biological counterparts?",
    "How promising is the flow analysis system based on the MIP based membrane electrode for the detection of melamine in milk?",
    "What is the primary technique used in the analysis of the spatial distribution of biomolecules?",
    "What is the role of 2,5-Dihydroxybenzoic acid (DHB) in the described protocol for matrix deposition?",
    "How was the diabetic rat model established in the study on H NMR-based metabolomics with diabetic rats?"
]


for question in questions_list:
    answer = ask(
        question + " Provide your response strictly as a JSON object with 'summary' and 'relevance_score'.",
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
            paper_directory="sciqag_papers_txt_only",
        ),
    )

    print(answer.session.answer)
    
    
    
q = "How does the SrCO3 material used in the study respond to ethanol vapor?"


answer = ask(
    q,
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

