





class MultiAgentAnswerSummarization:
    """
    storeand manage the summarized information for each agents answer
    
    """
    def __init__(self, agent_index: int, answer: str, reason: str):
        """

        :param agent_index: index of the agent (track which agent produced this summary)
        :param answer: answer generated by the agent.
        :param reason: reasoning or explanation provided by the agent for the answer
        """
        self.agent_index = agent_index
        self.answer = answer
        self.reason = reason


    def __str__(self):
        """
        Returns a string representation of the AnswerSummarization object.
        """
        return f"Agent {self.agent_index}:\nAnswer: {self.answer}\nReason: {self.reason}"



    def to_dict(self):
        """
        :return: dictionary representation of agent summary
        """
        
        return {
            "agent_index": self.agent_index,
            "answer": self.answer,
            "reason": self.reason
        }


def format_agent_summaries(summaries):
    """
    """
    formatted = []
    for summary in summaries:
        formatted.append(
            f"Agent {summary.agent_index}:\n"
            f"Answer: {summary.answer}\n"
            f"Reasoning: {summary.reason}\n"
        )
    return "\n".join(formatted)