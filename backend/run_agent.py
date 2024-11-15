import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

load_dotenv()

from backend.agents.agents import DEFAULT_AGENT, agents  # noqa: E402

agent = agents[DEFAULT_AGENT]


async def main() -> None:
    inputs = {"messages": [("user", "Research on the latest advancements in RAG models")]}
    result = await agent.ainvoke(
        inputs,
        config={"recursion_limit": 50, "thread_id": uuid4()},
    )
    print("\n====== Final Result ======\n\n\n\n\n\n")
    print(result["final_report"])
    # if DEFAULT_AGENT == "research_agent":
    #     print(result["final_report"])
    # else:
    #     result["messages"][-1].pretty_print()

    # Draw the agent graph as png
    # requires:
    # brew install graphviz
    # export CFLAGS="-I $(brew --prefix graphviz)/include"
    # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
    # pip install pygraphviz
    #
    agent.get_graph().draw_png("agent_diagram.png")


asyncio.run(main())
