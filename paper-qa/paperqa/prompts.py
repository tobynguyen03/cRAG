summary_prompt = (
    "Summarize the excerpt below to help answer a question.\n\nExcerpt from"
    " {citation}\n\n----\n\n{text}\n\n----\n\nQuestion: {question}\n\nDo not directly"
    " answer the question, instead summarize to give evidence to help answer the"
    " question. Stay detailed; report specific numbers, equations, or direct quotes"
    ' (marked with quotation marks). Reply "Not applicable" if the excerpt is'
    " irrelevant. At the end of your response, provide an integer score from 1-10 on a"
    " newline indicating relevance to question. Do not explain your score.\n\nRelevant"
    " Information Summary ({summary_length}):"
)

summary_json_prompt = (
    "Excerpt from {citation}\n\n----\n\n{text}\n\n----\n\nQuestion: {question}\n\n"
)

# qa_prompt = (
#     "Answer the question below with the context.\n\n"
#     "Context (with relevance scores):\n\n{context}\n\n----\n\n"
#     "Question: {question}\n\n"
#     "Write an answer based on the context. "
#     "If the context provides insufficient information reply "
#     '"I cannot answer."'
#     "For each part of your answer, indicate which sources most support "
#     "it via citation keys at the end of sentences, "
#     "like {example_citation}. Only cite from the context "
#     "below and only use the valid keys. Write in the style of a "
#     "Wikipedia article, with concise sentences and coherent paragraphs. "
#     "The context comes from a variety of sources and is only a summary, "
#     "so there may inaccuracies or ambiguities. If quotes are present and "
#     "relevant, use them in the answer. This answer will go directly onto "
#     "Wikipedia, so do not add any extraneous information.\n\n"
#     "Answer ({answer_length}):"
# )

select_paper_prompt = (
    "Select papers that may help answer the question below. "
    "Papers are listed as $KEY: $PAPER_INFO. "
    "Return a list of keys, separated by commas. "
    'Return "None", if no papers are applicable. '
    "Choose papers that are relevant, from reputable sources, and timely "
    "(if the question requires timely information).\n\n"
    "Question: {question}\n\n"
    "Papers: {papers}\n\n"
    "Selected keys:"
)
citation_prompt = (
    "Provide the citation for the following text in MLA Format. "
    "Do not write an introductory sentence. "
    "If reporting date accessed, the current year is 2024\n\n"
    "{text}\n\n"
    "Citation:"
)

structured_citation_prompt = (
    "Extract the title, authors, and doi as a JSON from this MLA citation. "
    "If any field can not be found, return it as null. "
    "Use title, authors, and doi as keys, author's value should be a list of authors. "
    "{citation}\n\n"
    "Citation JSON:"
)

default_system_prompt = (
    "Answer in a direct and concise tone. "
    "Your audience is an expert, so be highly specific. "
    "If there are ambiguous terms or acronyms, first define them."
)

# NOTE: we use double curly braces here so it's not considered an f-string template
summary_json_system_prompt = """\
Provide a summary of the relevant information that could help answer the question based on the excerpt. Respond with the following JSON format:

{{
  "summary": "...",
  "relevance_score": "..."
}}

where `summary` is relevant information from text - {summary_length} words and `relevance_score` is the relevance of `summary` to answer question (out of 10).
"""  # noqa: E501







# Prompt templates for use with LitQA
QA_PROMPT_TEMPLATE = "Q: {question}\n\nOptions:\n{options}"
EVAL_PROMPT_TEMPLATE = (
    "Extract the single letter answer from the given answer. Your answer should only be a single letter, do NOT output anything other than a single letter. Example output: A"
    "\n\n{qa_prompt}"
    "\n\n{qa_answer}"
    "\n\nSingle Letter Answer:"
)

CONTEXT_OUTER_PROMPT = "{context_str}\n\nValid Keys: {valid_keys}"
CONTEXT_INNER_PROMPT = "{name}: {text}\nFrom {citation}"



qa_prompt = (
    "Answer the question within ({answer_length}) below using only the provided context.\n\n"
    "Context (with relevance scores):\n\n{context}\n\n"
    "----\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "1. Provide a detailed answer based on the given context. If the context does not contain enough information, respond with \"I cannot answer.\".\n"
    "2. Discuss your reasoning process and how you derived your answer from the context.\n"
    "3. After completing your reasoning and answer, on a new line, write exactly:\n"
    "ANSWER SUMMARY: followed directly by a succinct summary of your answer and reasoning on the same line.\n"
)







# prompt templates for use with multiagent
multiagent_qa_prompt = (
    "Answer the question below within ({answer_length}) using the provided context and additional summarizations from other agents.\n\n"
    "Context (with relevance scores):\n\n{context}\n\n"
    "----\n\n"
    "Summarizations from other agents:\n\n"
    "{other_agents_summaries}\n\n"
    "Each summarization includes an agent's answer and reasoning. Use these summarizations to:\n"
    "1. Identify points of agreement or disagreement,\n"
    "2. Incorporate complementary insights from other agents' reasoning where relevant,\n"
    "3. Resolve conflicts by relying on evidence and context to generate a coherent, accurate, and consensus-driven answer.\n\n"
    "----\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "1. Write a detailed answer based on the context and other agents' summarizations. If the context or summarizations provide insufficient information, reply \"I cannot answer.\"\n"
    "2. Discuss your reasoning process, referencing the context and other agents' summaries as needed.\n"
    "3. After completing your reasoning and answer, write the following summaries on the same lines:\n"
    "   - ANSWER SUMMARY: Provide a succinct summary of your final answer.\n"
    "   - REASONING SUMMARY: Provide a succinct summary of your reasoning process.\n"
)


EVAL_MULTIAGENT_CONSENSUS_PROMPT = """
You are tasked with synthesizing multiple expert inputs to provide the most accurate and concise answer to the question below.  
The answer should reflect the best understanding of the topic without referencing the individual inputs or their sources.

Focus on brevity and clarity, distilling key insights into 300-500 characters.

Question: {question}

Answers from experts:
{multiagent_answers_input}
"""
