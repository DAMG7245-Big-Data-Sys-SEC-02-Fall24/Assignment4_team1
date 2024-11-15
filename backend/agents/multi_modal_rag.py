import asyncio
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Import your existing tools
from backend.agents.tools.research_tools import arxiv_tool, web_tool
from backend.agents.models import models


class AgentState(MessagesState, total=False):
    """Simple state tracking"""
    tool_outputs: Dict[str, Any]
    final_report: str | None


async def research_with_tools(state: AgentState, config: RunnableConfig) -> AgentState:
    """Research using both web and arxiv tools"""
    print("\n====== Starting Research ======")
    
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    query = state["messages"][-1].content
    
    print(f"\nQuery: {query}")
    
    # Create research prompt
    research_prompt = """Given the query, use both web_search and fetch_arxiv tools to gather information.
    You must use both tools at least once. Focus on finding:
    1. Recent developments
    2. Technical details
    3. Practical applications

    Structure your findings systematically."""
    
    messages = [
        SystemMessage(content=research_prompt),
        HumanMessage(content=query)
    ]
    
    # Bind tools to model
    model_with_tools = m.bind_tools([web_tool, arxiv_tool])
    
    # Get model's research steps
    print("\n====== Model Planning Research ======")
    response = await model_with_tools.ainvoke(messages, config)
    
    # Extract and execute tool calls
    tool_outputs = {}
    
    if response.tool_calls:
        print("\n====== Executing Tools ======")
        print(response.tool_calls)
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\nExecuting {tool_name} with args: {tool_args}")
            
            if tool_name == "ArXiv":
                result = await arxiv_tool.ainvoke(tool_args)
                tool_outputs["arxiv"] = result
            elif tool_name == "WebSearch":
                result = await web_tool.ainvoke(tool_args)
                tool_outputs["web"] = result
                
            print(f"{tool_name} returned {len(result) if isinstance(result, list) else 0} results")
    
    return {
        **state,
        "tool_outputs": tool_outputs
    }


async def generate_report(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generate final report from tool outputs"""
    print("\n====== Generating Report ======")
    
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    tool_outputs = state.get("tool_outputs", {})
    query = state["messages"][-1].content
    
    # Prepare findings for report
    web_results = tool_outputs.get("web", [])
    arxiv_results = tool_outputs.get("arxiv", [])
    print(f"\nWeb Results: {len(web_results)}")
    print(f"ArXiv Results: {len(arxiv_results)}")
    print(web_results)
    print(arxiv_results)
    web_findings = "\n".join([
        f"- {result.get('title')}: {result.get('snippet', '')}"
        for result in web_results
    ])
    
    arxiv_findings = "\n".join([
        f"- {paper.get('title')}\n  Abstract: {paper.get('abstract', '')[:300]}..."
        for paper in arxiv_results
    ])
    
    report_prompt = """Create a concise research report based on the findings.
    
    Format:
    # Research Summary
    [Brief overview]
    
    ## Key Findings
    [Main points from both web and academic sources]
    
    ## Sources
    [List of sources used]
    """
    
    messages = [
        SystemMessage(content=report_prompt),
        HumanMessage(content=f"""
        Query: {query}
        
        Web Findings:
        {web_findings}
        
        Academic Findings:
        {arxiv_findings}
        """)
    ]
    
    response = await m.ainvoke(messages, config)
    
    return {
        **state,
        "final_report": response.content
    }


# Define the graph
agent = StateGraph(AgentState)

# Add nodes
agent.add_node("research", research_with_tools)
agent.add_node("report", generate_report)

# Set entry point
agent.set_entry_point("research")

# Add edges
agent.add_edge("research", "report")
agent.add_edge("report", END)

# Compile the agent
research_agent = agent.compile(
    checkpointer=MemorySaver(),
)


