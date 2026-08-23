"""
test_agent.py — Unit tests for the LangGraph travel assistant graph.

Tests cover:
  - Graph construction (all expected nodes are present, graph compiles)
  - Routing logic (should_continue routes correctly for both branches)
  - Tool registry (both tools are registered by name)
"""
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from app.agent import MessagesState, build_agent, should_continue, tools_by_name


# ---------------------------------------------------------------------------
# Tests — should_continue routing
# ---------------------------------------------------------------------------
class TestShouldContinue:
    """Tests for the conditional edge routing function."""

    def test_routes_to_tool_node_when_tool_calls_present(self):
        """should_continue must return 'tool_node' when the AI message has tool calls."""
        ai_message = MagicMock(spec=AIMessage)
        ai_message.tool_calls = [
            {"name": "duckduckgo_search", "args": {"query": "Rome attractions"}, "id": "call_1"}
        ]
        state = MessagesState(messages=[ai_message])
        assert should_continue(state) == "tool_node"

    def test_routes_to_end_when_no_tool_calls(self):
        """should_continue must return '__end__' when the AI message has no tool calls."""
        ai_message = MagicMock(spec=AIMessage)
        ai_message.tool_calls = []
        state = MessagesState(messages=[ai_message])
        assert should_continue(state) == "__end__"

    def test_routes_to_tool_node_for_weather_tool(self):
        """should_continue routes correctly when the weather tool is requested."""
        ai_message = MagicMock(spec=AIMessage)
        ai_message.tool_calls = [
            {"name": "get_current_weather", "args": {"city": "Tokyo"}, "id": "call_2"}
        ]
        state = MessagesState(messages=[ai_message])
        assert should_continue(state) == "tool_node"


# ---------------------------------------------------------------------------
# Tests — graph construction
# ---------------------------------------------------------------------------
class TestBuildAgent:
    """Tests for agent graph construction and node registration."""

    def test_agent_compiles_without_error(self):
        """build_agent() must return a compiled, non-None graph object."""
        agent = build_agent()
        assert agent is not None

    def test_graph_contains_llm_call_node(self):
        """The compiled graph must contain the 'llm_call' node."""
        agent = build_agent()
        node_names = set(agent.get_graph().nodes.keys())
        assert "llm_call" in node_names

    def test_graph_contains_tool_node(self):
        """The compiled graph must contain the 'tool_node' node."""
        agent = build_agent()
        node_names = set(agent.get_graph().nodes.keys())
        assert "tool_node" in node_names


# ---------------------------------------------------------------------------
# Tests — tool registry
# ---------------------------------------------------------------------------
class TestToolRegistry:
    """Tests that both tools are registered and reachable by name."""

    def test_search_tool_is_registered(self):
        """duckduckgo_search must be present in tools_by_name."""
        assert "duckduckgo_search" in tools_by_name

    def test_weather_tool_is_registered(self):
        """get_current_weather must be present in tools_by_name."""
        assert "get_current_weather" in tools_by_name

    def test_exactly_two_tools_are_registered(self):
        """The agent must have exactly 2 tools registered (no accidental extras)."""
        assert len(tools_by_name) == 2
