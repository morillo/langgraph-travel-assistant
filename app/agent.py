"""
agent.py — LangGraph travel assistant agent.

Graph topology:
    START → llm_call ⇄ tool_node → END

The agent is given two tools:
  1. duckduckgo_search     — general travel research (attractions, hotels,
                             visa requirements, flight tips, local customs)
  2. get_current_weather   — real-time weather for any city worldwide

The LLM decides autonomously which tool(s) to call, in which order, and
when enough information has been gathered to produce a final response.
"""
import operator
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.tools.search import get_search_tool
from app.tools.weather import get_current_weather

load_dotenv()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
# Both tools are listed here so the LLM receives their JSON schemas and can
# decide which one to call based on the user's query.
search_tool = get_search_tool()
tools = [search_tool, get_current_weather]
tools_by_name = {tool.name: tool for tool in tools}

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)

SYSTEM_PROMPT = """You are a knowledgeable and friendly travel assistant.
You help users plan trips by providing accurate, up-to-date information about:
- Tourist attractions, landmarks, and local experiences
- Hotels, neighbourhoods, and accommodation tips
- Flights, transport, and logistics
- Visa and entry requirements
- Local customs, food, and culture
- Current weather conditions at destinations

Tool usage guidelines:
- Use duckduckgo_search for general travel research and destination information.
- Use get_current_weather whenever the user asks about weather, temperature,
  climate, or what to pack for a trip.
- You may call both tools in a single turn if the query requires it.
- If you cannot complete a task (e.g. booking a flight or hotel), clearly
  explain what you can and cannot do, and suggest where the user can go instead.

Always be concise, helpful, and specific."""


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class MessagesState(TypedDict):
    """Shared state passed between every node in the graph.

    messages uses operator.add as its reducer, meaning each node appends
    new messages rather than replacing the list.
    """
    messages: Annotated[list[AnyMessage], operator.add]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def llm_call(state: MessagesState) -> dict:
    """Invoke the LLM with the full conversation history and available tools.

    The LLM either:
      - Returns tool_calls  → graph routes to tool_node for execution.
      - Returns plain text  → graph routes to END, response sent to user.
    """
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            )
        ]
    }


def tool_node(state: MessagesState) -> dict:
    """Execute every tool call requested by the LLM in the latest message.

    Each call is dispatched by name, invoked with the LLM-supplied arguments,
    and the observation is wrapped in a ToolMessage so the LLM can use it
    in its next turn.
    """
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=observation, tool_call_id=tool_call["id"])
        )
    return {"messages": results}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Route the graph after each LLM call.

    Returns:
        "tool_node"  if the LLM requested one or more tool calls.
        END          if the LLM produced a final answer.
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_agent():
    """Build and compile the LangGraph travel assistant graph."""
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)

    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges(
        "llm_call", should_continue, ["tool_node", END]
    )
    graph_builder.add_edge("tool_node", "llm_call")

    return graph_builder.compile()
