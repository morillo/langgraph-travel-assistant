"""
search.py — DuckDuckGo web search tool for the travel assistant.

Wraps LangChain's DuckDuckGoSearchRun as a factory function so the tool
can be imported, instantiated, and tested independently from the agent.
"""
from langchain_community.tools import DuckDuckGoSearchRun


def get_search_tool() -> DuckDuckGoSearchRun:
    """Return a configured DuckDuckGo web search tool.

    This tool is invoked by the agent whenever it needs current or
    specific travel information: attractions, hotel recommendations,
    visa requirements, flight tips, local customs, etc.
    """
    return DuckDuckGoSearchRun()
