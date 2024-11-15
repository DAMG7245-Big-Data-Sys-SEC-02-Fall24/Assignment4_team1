from langgraph.graph.state import CompiledStateGraph

from backend.agents.bg_task_agent.bg_task_agent import bg_task_agent
from backend.agents.chatbot import chatbot
from backend.agents.research_assistant import research_assistant

DEFAULT_AGENT = "chatbot"


agents: dict[str, CompiledStateGraph] = {
    "chatbot": chatbot,
    "research-assistant": research_assistant,
    "bg-task-agent": bg_task_agent,
}
