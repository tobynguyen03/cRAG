from __future__ import annotations

import argparse
import logging
import os
import re 
import time
from typing import List
import asyncio
from typing import Any, Tuple, Optional

from pydantic_settings import CliSettingsSource
from rich.logging import RichHandler

from paperqa.settings import Settings, get_settings
from paperqa.utils import get_loop, pqa_directory, setup_default_logs
from paperqa.version import __version__


from .multiagent import MultiAgentAnswerSummarization


from .main import agent_query, index_search
from .models import AnswerResponse, QueryRequest, MultiAgentQueryRequest
from .search import SearchIndex, get_directory_index

logger = logging.getLogger(__name__)

LOG_VERBOSITY_MAP: dict[int, dict[str, int]] = {
    0: {
        "paperqa.agents": logging.INFO,
        "paperqa.agents.helpers": logging.WARNING,
        "paperqa.agents.main": logging.WARNING,
        "paperqa.agents.main.agent_callers": logging.INFO,
        "paperqa.agents.models": logging.WARNING,
        "paperqa.agents.search": logging.INFO,
        "anthropic": logging.WARNING,
        "openai": logging.WARNING,
        "httpcore": logging.WARNING,
        "httpx": logging.WARNING,
        "LiteLLM": logging.WARNING,
        "LiteLLM Router": logging.WARNING,
        "LiteLLM Proxy": logging.WARNING,
    }
}
LOG_VERBOSITY_MAP[1] = LOG_VERBOSITY_MAP[0] | {
    "paperqa.models": logging.INFO,
    "paperqa.agents.main": logging.INFO,
}
LOG_VERBOSITY_MAP[2] = LOG_VERBOSITY_MAP[1] | {
    "paperqa.models": logging.DEBUG,
    "paperqa.agents.helpers": logging.DEBUG,
    "paperqa.agents.main": logging.DEBUG,
    "paperqa.agents.main.agent_callers": logging.DEBUG,
    "paperqa.agents.search": logging.DEBUG,
    "LiteLLM": logging.INFO,
    "LiteLLM Router": logging.INFO,
    "LiteLLM Proxy": logging.INFO,
}
LOG_VERBOSITY_MAP[3] = LOG_VERBOSITY_MAP[2] | {
    "LiteLLM": logging.DEBUG,  # <-- every single LLM call
}
_MAX_PRESET_VERBOSITY: int = max(k for k in LOG_VERBOSITY_MAP)

_PAPERQA_PKG_ROOT_LOGGER = logging.getLogger(__name__.split(".", maxsplit=1)[0])
_INITIATED_FROM_CLI = False


def is_running_under_cli() -> bool:
    """Check if the current Python process comes from the CLI."""
    return _INITIATED_FROM_CLI


def set_up_rich_handler(install: bool = True) -> RichHandler:
    """Add a RichHandler to the paper-qa "root" logger, and return it."""
    rich_handler = RichHandler(
        rich_tracebacks=True, markup=True, show_path=False, show_level=False
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    if install and not any(
        isinstance(h, RichHandler) for h in _PAPERQA_PKG_ROOT_LOGGER.handlers
    ):
        _PAPERQA_PKG_ROOT_LOGGER.addHandler(rich_handler)
    return rich_handler


def configure_log_verbosity(verbosity: int = 0) -> None:
    key = min(verbosity, _MAX_PRESET_VERBOSITY)
    for logger_name, logger_ in logging.Logger.manager.loggerDict.items():
        if isinstance(logger_, logging.Logger) and (
            log_level := LOG_VERBOSITY_MAP.get(key, {}).get(logger_name)
        ):
            logger_.setLevel(log_level)


def configure_cli_logging(verbosity: int | Settings = 0) -> None:
    """Suppress loquacious loggers according to the settings' verbosity level."""
    setup_default_logs()
    set_up_rich_handler()
    if isinstance(verbosity, Settings):
        verbosity = verbosity.verbosity
    configure_log_verbosity(verbosity)
    if verbosity > 0:
        print(f"PaperQA version: {__version__}")


def ask(query: str, settings: Settings, timeout: int = 180) -> Tuple[bool, Optional[AnswerResponse]]:
    """Query PaperQA via an agent."""
    
    configure_cli_logging(settings)

    try:
        answer_response = get_loop().run_until_complete(
            asyncio.wait_for(
                agent_query(
                    QueryRequest(query=query, settings=settings),
                    agent_type=settings.agent.agent_type,
                ),
                timeout=timeout
            )
        )
        return True, answer_response
    
    except asyncio.TimeoutError:
        return False, None








# def ask_multiagent_working_test(query: str, settings: Settings) -> AnswerResponse:
#     """Query multi-agent system with each agent running PaperQA"""
    
    
#     # TODO: add in settings for RAG Agent Communicator class 
#     # shuold have all relevant settings like prompts to generat communications, consensus mechanism, etc.
    
#     configure_cli_logging(settings)



#     other_agents_summarization = []
    
#     other_agents_summarization.append(MultiAgentAnswerSummarization(agent_index=1, answer='this is agent1 answer', reason='agent1 answer reason' ))
#     other_agents_summarization.append(MultiAgentAnswerSummarization(agent_index=2, answer='this is agent2 answer', reason='agent2 answer reason' ))
    
    
#     time_to_answer_start = time.time()
    
#     answer_response = get_loop().run_until_complete(        
#         agent_query(
            
#             MultiAgentQueryRequest(
#                 query=query, 
#                 settings=settings, 
                
#                 other_agents_summarization=other_agents_summarization,
               
#             ),
                    
#             agent_type=settings.agent.agent_type,
#         )
#     )

#     answer_time = time.time() - time_to_answer_start
#     print(f"answer time: {answer_time}")
#     print(f"\n \n LEN contextx: {len(answer_response.session.contexts)}")
    
#     answer = answer_response.session.answer
    
    
#     answer_summary_match = re.search(r"ANSWER SUMMARY:\s+(.*?)\s+REASONING SUMMARY:", answer, re.DOTALL)
#     reasoning_summary_match = re.search(r"REASONING SUMMARY:\s+(.*)", answer, re.DOTALL)

#     answer_summary = answer_summary_match.group(1).strip() if answer_summary_match else None
#     reasoning_summary = reasoning_summary_match.group(1).strip() if reasoning_summary_match else None


#     print("ANSWER SUMMARY:\n", answer_summary)
#     print("\nREASONING SUMMARY:\n", reasoning_summary)
    
    
    
    
    
    
    
#     return answer_response



class Agent:
    def __init__(self, agent_index: int, initial_answer: str = "No answer yet", initial_reason: str = "No reasoning yet"):
        self.agent_index = agent_index
        self.summarization = MultiAgentAnswerSummarization(
            agent_index=agent_index, 
            answer=initial_answer, 
            reason=initial_reason
        )
        self.round_summarizations: List[MultiAgentAnswerSummarization] = []

    def update_summarization(self, answer_summary: str, reasoning_summary: str):
        self.summarization.answer = answer_summary
        self.summarization.reason = reasoning_summary
    
    
    
    
def ask_multiagent(query: str, settings: Settings, num_agents: int = 2, num_rounds: int = 2, timeout=180) -> List[Agent]:
    """
    Query a multi-agent system. Each agent will take a turn calling the ask function.
    On the first round, all agents have a default summarization. In subsequent rounds,
    each agent generates a new answer based on other agents' updated summarizations.
    All responses per round are stored within each agent's `round_responses`.
    """

    configure_cli_logging(settings)
    
    agents = [
        Agent(agent_index=i+1,
              initial_answer=f"Default answer for agent {i+1}",
              initial_reason=f"Default reasoning for agent {i+1}")
        for i in range(num_agents)
    ]

    # timeout = 60
    
    

    for round_idx in range(num_rounds):
        print(f"\n--- ROUND {round_idx+1}/{num_rounds} ---")

        round_success = True 
        
        for agent in agents:
            other_agents_summarization = [a.summarization for a in agents if a.agent_index != agent.agent_index]

            time_to_answer_start = time.time()
            
            max_retries = 3
            retries = 0
            success = False

            while not success and retries < max_retries:
                try:
                    answer_response = get_loop().run_until_complete(
                        asyncio.wait_for(
                            agent_query(
                                MultiAgentQueryRequest(
                                    query=query,
                                    settings=settings,
                                    other_agents_summarization=other_agents_summarization,
                                ),
                                agent_type=settings.agent.agent_type,
                            ),
                            timeout=timeout  # 3 minutes timeout
                        )
                    )
                    
                    
                    full_answer = answer_response.session.answer
                    
                    answer_pattern = re.compile(r'(?i)answer summary:\s*(.*)')
                    reasoning_pattern = re.compile(r'(?i)reasoning summary:\s*(.*)')

                    answer_match = answer_pattern.search(full_answer)
                    reasoning_match = reasoning_pattern.search(full_answer)

                    if answer_match and reasoning_match:
                        answer_summary = answer_match.group(1)
                        reasoning_summary = reasoning_match.group(1)
                        success = True

                
                except asyncio.TimeoutError:
                    retries += 1
                    print(f"Attempt {retries} failed during agent query. Retrying...")

            if not success:
                print(f"Agent {agent.agent_index} failed to get a response after {max_retries} attempts.")
                answer_response = None
            
            if success:
                answer_time = time.time() - time_to_answer_start
                print(f"Agent {agent.agent_index} answer time: {answer_time}")
                agent.update_summarization(answer_summary, reasoning_summary)
                agent.round_summarizations.append(agent.summarization)
            else: 
                # break out of the loop if the agent failed to get a response
                round_success = False
                break
            
        if not round_success:
            print("Round failed. Exiting...")
            success_question_answer = False
            break
        else:
            print("Round successful.")
            success_question_answer = True
            
    if success_question_answer: 
        print("All rounds successful.")
        return success_question_answer, agents
    
    else:
        print("Some rounds failed.")
        return success_question_answer, agents
    
                
                
                # Print out what happened this turn
                # print(f"Agent {agent.agent_index} UPDATED ANSWER SUMMARY:\n{answer_summary}\n")
                # print(f"Agent {agent.agent_index} UPDATED REASONING SUMMARY:\n{reasoning_summary}\n")












def search_query(
    query: str,
    index_name: str,
    settings: Settings,
) -> list[tuple[AnswerResponse, str] | tuple[Any, str]]:
    """Search using a pre-built PaperQA index."""
    configure_cli_logging(settings)
    if index_name == "default":
        index_name = settings.get_index_name()
    return get_loop().run_until_complete(
        index_search(
            query,
            index_name=index_name,
            index_directory=settings.agent.index.index_directory,
        )
    )


def build_index(
    index_name: str | None = None,
    directory: str | os.PathLike | None = None,
    settings: Settings | None = None,
) -> SearchIndex:
    """Build a PaperQA search index, this will also happen automatically upon using `ask`."""
    settings = get_settings(settings)
    if index_name == "default":
        settings.agent.index.name = None
    elif isinstance(index_name, str):
        settings.agent.index.name = index_name
    configure_cli_logging(settings)
    if directory:
        settings.agent.index.paper_directory = directory
    return get_loop().run_until_complete(get_directory_index(settings=settings))


def save_settings(
    settings: Settings,
    settings_path: str | os.PathLike,
) -> None:
    """Save the settings to a file."""
    configure_cli_logging(settings)
    # check if this could be interpreted at an absolute path
    if os.path.isabs(settings_path):
        full_settings_path = os.path.expanduser(settings_path)
    else:
        full_settings_path = os.path.join(pqa_directory("settings"), settings_path)
        if not full_settings_path.endswith(".json"):
            full_settings_path += ".json"

    is_overwrite = os.path.exists(full_settings_path)

    with open(full_settings_path, "w") as f:
        f.write(settings.model_dump_json(indent=2))

    if is_overwrite:
        logger.info(f"Settings overwritten to: {full_settings_path}")
    else:
        logger.info(f"Settings saved to: {full_settings_path}")


def main() -> None:

    parser = argparse.ArgumentParser(description="PaperQA CLI")
    parser.add_argument(
        "--settings",
        "-s",
        default="high_quality",
        help=(
            "Named settings to use. Will search in local, pqa directory, and package"
            " last"
        ),
    )
    parser.add_argument(
        "--index", "-i", default="default", help="Index name to search or create"
    )

    subparsers = parser.add_subparsers(
        title="commands", dest="command", description="Available commands"
    )

    subparsers.add_parser("view", help="View the chosen settings")

    save_parser = subparsers.add_parser("save", help="View the chosen settings")
    save_parser.add_argument(
        "location", help="Location for new settings (name or an absolute path)"
    )

    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question of current index (based on settings)"
    )
    ask_parser.add_argument("query", help="Question to ask")

    search_parser = subparsers.add_parser(
        "search",
        help=(
            "Search the index specified by --index."
            " Pass `--index answers` to search previous answers."
        ),
    )
    search_parser.add_argument("query", help="Keyword search")

    build_parser = subparsers.add_parser(
        "index", help="Build a search index from given directory"
    )
    build_parser.add_argument("directory", help="Directory to build index from")

    # Create CliSettingsSource instance
    cli_settings = CliSettingsSource[argparse.ArgumentParser](
        Settings, root_parser=parser
    )

    # Now use argparse to parse the remaining arguments
    args, remaining_args = parser.parse_known_args()
    # Parse arguments using CliSettingsSource
    settings = Settings.from_name(
        args.settings, cli_source=cli_settings(args=remaining_args)
    )

    match args.command:
        case "ask":
            ask(args.query, settings)
        case "view":
            configure_cli_logging(settings)
            logger.info(f"Viewing: {args.settings}")
            logger.info(settings.model_dump_json(indent=2))
        case "save":
            save_settings(settings, args.location)
        case "search":
            search_query(args.query, args.index, settings)
        case "index":
            build_index(args.index, args.directory, settings)
        case _:
            commands = ", ".join({"view", "ask", "search", "index"})
            brief_help = f"\nRun with commands: {{{commands}}}\n\n"
            brief_help += "For more information, run with --help"
            print(brief_help)


if __name__ == "__main__":
    _INITIATED_FROM_CLI = True
    main()
