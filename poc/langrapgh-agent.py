#!/usr/bin/env python
# coding: utf-8

# In[146]:


from semantic_router.encoders import OpenAIEncoder
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator

from pinecone import Pinecone, ServerlessSpec
encoder = OpenAIEncoder(name='text-embedding-3-small')


# In[147]:


pc = Pinecone(api_key="")
spec = ServerlessSpec(
    cloud="aws",
    region="us-east-1",
)
index = pc.Index("cfa-research")
index.describe_index_stats()


# In[148]:


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


# In[149]:


from langchain_core.tools import tool
import requests
import re

@tool("fetch_arxiv")
def fetch_arxiv(arxiv_id: str):
    """Gets the abstract from an ArXiv paper given the arxiv ID. Useful for
    finding high-level context about a specific paper."""
    # get paper page in html
    res = requests.get(
        f"https://export.arxiv.org/abs/{arxiv_id}"
    )
    # search html for abstract
    abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL
    )
    re_match = abstract_pattern.search(res.text)
    # return abstract text
    if re_match:
        return re_match.group(1)
    else:
        return "Abstract not found."
from langchain_core.tools import tool
from serpapi import GoogleSearch
import os

@tool("search_google")
def search_google(query: str):
    """Searches Google for articles and web pages related to the query.
    Returns a list of relevant search results with titles and snippets."""
    search = GoogleSearch({
        "q": query,
        "api_key": "",
        "num": 5  # Number of results to return
    })
    results = search.get_dict()
    
    formatted_results = []
    for result in results.get("organic_results", []):
        formatted_results.append({
            "title": result.get("title"),
            "snippet": result.get("snippet"),
            "link": result.get("link")
        })
    return formatted_results

@tool("search_scholar")
def search_scholar(query: str):
    """Searches Google Scholar for academic papers related to the query.
    Returns a list of relevant academic papers with titles, authors, and abstracts."""
    search = GoogleSearch({
        "q": query,
        "api_key": "",
        "engine": "google_scholar",
        "num": 5  # Number of results to return
    })
    
    results = search.get_dict()
    print(len(results))
    formatted_results = []
    for result in results.get("organic_results", []):
        formatted_results.append({
            "title": result.get("title"),
            "authors": result.get("authors", ""),
            "publication": result.get("publication_info", {}).get("summary", ""),
            "snippet": result.get("snippet"),
            "link": result.get("link"),
            "citations": result.get("inline_links", {}).get("cited_by", {}).get("total", 0)
        })
    return formatted_results


# In[150]:


fetch_arxiv.invoke("2106.04561")
search_scholar.invoke("transformer architecture improvements")


# In[151]:


search_google.invoke("machine learning advances 2024")


# In[152]:


@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: str,
    main_body: str,
    conclusion: str,
    sources: str
):
    """Returns a natural language response to the user in the form of a research
    report. There are several sections to this report, those are:
    - `introduction`: a short paragraph introducing the user's question and the
    topic we are researching.
    - `research_steps`: a few bullet points explaining the steps that were taken
    to research your report.
    - `main_body`: this is where the bulk of high quality and concise
    information that answers the user's question belongs. It is 3-4 paragraphs
    long in length.
    - `conclusion`: this is a short single paragraph conclusion providing a
    concise but sophisticated view on what was found.
    - `sources`: a bulletpoint list provided detailed sources for all information
    referenced during the research process
    """
    print(f"""
    introduction: {introduction}
    research_steps: {research_steps}
    main_body: {main_body}
    conclusion: {conclusion}
    sources: {sources}
    """)
    return ""


# In[153]:


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = """You are the oracle, the great AI decision maker.
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

If you see that a tool has been used (in the scratchpad) with a particular
query, do NOT use that same tool with the same query again. Also, do NOT use
any tool more than twice (ie, if the tool appears in the scratchpad twice, do
not use it again).

You should aim to collect information from a diverse range of sources before
providing the answer to the user. Once you have collected plenty of information
to answer the user's question (stored in the scratchpad) use the final_answer
tool."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("assistant", "scratchpad: {scratchpad}"),
])


# In[154]:


from langchain_core.messages import ToolCall, ToolMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0
)

tools=[
    fetch_arxiv,
    search_google,
    search_scholar,
    final_answer
]
tool_str_to_func = {
    "fetch_arxiv": fetch_arxiv,
    "search_google": search_google,
    "search_scholar": search_scholar,
    "final_answer": final_answer    
}

# define a function to transform intermediate_steps from list
# of AgentAction to scratchpad string
def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            # this was the ToolExecution
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)

oracle = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(
            intermediate_steps=x["intermediate_steps"]
        ),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)


# In[155]:


inputs = {
    "input": "The Economics of Private Equity: A Critical Review",
    "chat_history": [],
    "intermediate_steps": [],
}
out = oracle.invoke(inputs)


# In[156]:


for i in out.tool_calls:
    print(i["name"])


# In[157]:


def run_oracle(state: list):
    print("run_oracle")
    print(f"intermediate_steps: {state['intermediate_steps']}")
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="TBD"
    )
    return {
        "intermediate_steps": [action_out]
    }

def router(state: list):
    # return the tool name to use
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    else:
        # if we output bad format go to final answer
        print("Router invalid format")
        return "final_answer"
    
def run_tool(state: list):
    # use this as helper function so we repeat less code
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    print(f"{tool_name}.invoke(input={tool_args})")
    # run tool
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )
    return {"intermediate_steps": [action_out]}


# In[158]:


from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
tool_str_to_func = {
    "fetch_arxiv": fetch_arxiv,
    "search_google": search_google,
    "search_scholar": search_scholar,
    "final_answer": final_answer    
}
graph.add_node("oracle", run_oracle)
graph.add_node("fetch_arxiv", run_tool)
graph.add_node("search_google", run_tool)
graph.add_node("search_scholar", run_tool)
graph.add_node("final_answer", run_tool)

graph.set_entry_point("oracle")

graph.add_conditional_edges(
    source="oracle",  # where in graph to start
    path=router,  # function to determine which node is called
)

# create edges from each tool back to the oracle
for tool_obj in tools:
    if tool_obj.name != "final_answer":
        graph.add_edge(tool_obj.name, "oracle")

# if anything goes to final answer, it must then move to END
graph.add_edge("final_answer", END)

runnable = graph.compile()


# In[159]:


from IPython.display import Image

Image(runnable.get_graph().draw_png())


# In[160]:


out = runnable.invoke({
    "input": "The Economics of Private Equity: A Critical Review",
    "chat_history": [],
})


# In[161]:


def build_report(output: dict):
    output = output.get("intermediate_steps")[-1].tool_input
    research_steps = output["research_steps"]
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    sources = output["sources"]
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    return f"""
INTRODUCTION
------------
{output["introduction"]}

RESEARCH STEPS
--------------
{research_steps}

REPORT
------
{output["main_body"]}

CONCLUSION
----------
{output["conclusion"]}

SOURCES
-------
{sources}
"""


# In[162]:


build_report(out)


# In[162]:




