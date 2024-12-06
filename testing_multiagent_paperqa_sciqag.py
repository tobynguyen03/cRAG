from paperqa import Settings, ask, ask_multiagent


from paperqa.settings import AgentSettings
import os

# import litellm
# litellm.set_verbose=True

local_llm_config = dict(
    model_list=[
        dict(
            model_name='ollama/llama3.1',
            litellm_params=dict(
                model='ollama/llama3.1',
                api_base="http://localhost:11434", 
            ),
        )
    ]
)


q = "How does the SrCO3 material used in the study respond to ethanol vapor?"

answer = ask_multiagent(
    q,
    settings=Settings(
        llm='ollama/llama3.1',
        llm_config=local_llm_config,
        
        summary_llm='ollama/llama3.1',
        summary_llm_config=local_llm_config,
        
        embedding='ollama/mxbai-embed-large',
        
        agent=AgentSettings(
            agent_llm='ollama/llama3.1', 
            agent_llm_config=local_llm_config
        ),
        use_doc_details=False,
        paper_directory="sciqag_papers_txt_only_test_case_relevant"
    ),
)
