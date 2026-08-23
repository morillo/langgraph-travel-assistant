"""
run_queries.py — Send 10 travel queries through the assistant to generate Phoenix traces.

Query design rationale:
  The 10 queries are deliberately varied to produce interesting, diverse traces:

  Group A — Happy path / web search tool (queries 1-3)
    General travel research questions the agent can answer well using DuckDuckGo.
    These should produce clean traces with one LLM call + one search tool call.

  Group B — Weather tool (queries 4-5)
    Explicit weather questions that should trigger get_current_weather.
    Verifies the tool is correctly selected and structured output is returned.

  Group C — Multi-tool (queries 6-7)
    Questions that benefit from BOTH tools in a single turn (e.g. "visit Paris
    next week — what should I pack?"). Produces richer traces with multiple spans.

  Group D — Edge cases / potential frustration (queries 8-10)
    Requests the agent cannot fully fulfil (booking, pricing, overly broad scope).
    These are intentionally included to generate "frustrated" labels in the
    evaluation stage, creating a realistic dataset of mixed-quality interactions.

Usage:
    Make sure both services are running first:
      1. docker run -p 6006:6006 arizephoenix/phoenix:latest
      2. poetry run uvicorn app.api:app --reload

    Then run:
      poetry run python scripts/run_queries.py

    All 10 traces will be visible in Phoenix at http://localhost:6006
"""
import json
import time

import requests

API_URL = "http://localhost:8000/chat"
DELAY_BETWEEN_REQUESTS = 3  # seconds — avoids rate-limiting on DuckDuckGo

# ---------------------------------------------------------------------------
# Query definitions
# ---------------------------------------------------------------------------
QUERIES = [
    # ── Group A: Happy path / web search ────────────────────────────────────
    {
        "id": 1,
        "group": "web_search",
        "message": "What are the top 5 tourist attractions in Rome, Italy?",
        "expected_tool": "duckduckgo_search",
        "note": "Classic travel info query — clean happy path",
    },
    {
        "id": 2,
        "group": "web_search",
        "message": "What are the best neighbourhoods to stay in Barcelona for first-time visitors?",
        "expected_tool": "duckduckgo_search",
        "note": "Hotel/area research — should use search tool",
    },
    {
        "id": 3,
        "group": "web_search",
        "message": "What documents do US citizens need to travel to Japan?",
        "expected_tool": "duckduckgo_search",
        "note": "Visa/entry requirements — factual search query",
    },
    # ── Group B: Weather tool ────────────────────────────────────────────────
    {
        "id": 4,
        "group": "weather",
        "message": "What is the current weather in Tokyo?",
        "expected_tool": "get_current_weather",
        "note": "Direct weather query — should trigger weather tool",
    },
    {
        "id": 5,
        "group": "weather",
        "message": "Is it warm enough to visit Reykjavik, Iceland right now?",
        "expected_tool": "get_current_weather",
        "note": "Implicit weather query — agent must infer tool needed",
    },
    # ── Group C: Multi-tool ──────────────────────────────────────────────────
    {
        "id": 6,
        "group": "multi_tool",
        "message": "I want to visit Paris next week. What should I pack based on the current weather there?",
        "expected_tool": "both",
        "note": "Needs weather + general packing tips — multi-tool trace",
    },
    {
        "id": 7,
        "group": "multi_tool",
        "message": "What are the must-see attractions in Sydney, and what is the weather like there right now?",
        "expected_tool": "both",
        "note": "Explicit dual question — should call both tools",
    },
    # ── Group D: Edge cases / frustrating interactions ───────────────────────
    {
        "id": 8,
        "group": "frustrating",
        "message": "Book me a hotel in Dubai for next Friday night under $100.",
        "expected_tool": "none_or_search",
        "note": "Agent cannot book — likely to produce a frustrated interaction",
    },
    {
        "id": 9,
        "group": "frustrating",
        "message": "Find me the cheapest flight from New York to London tomorrow.",
        "expected_tool": "none_or_search",
        "note": "Pricing/booking request agent cannot fulfil directly",
    },
    {
        "id": 10,
        "group": "frustrating",
        "message": "Tell me literally everything about travelling in Southeast Asia.",
        "expected_tool": "duckduckgo_search",
        "note": "Overly broad — response may feel incomplete or overwhelming",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def send_query(query: dict) -> dict:
    """Send a single message to the travel assistant API.

    Args:
        query: Query definition dict from QUERIES list.

    Returns:
        Dict with query metadata + response + status.
    """
    try:
        response = requests.post(
            API_URL,
            json={"message": query["message"]},
            timeout=60,
        )
        response.raise_for_status()
        answer = response.json()["response"]
        status = "ok"
    except requests.exceptions.Timeout:
        answer = "ERROR: request timed out after 60s"
        status = "timeout"
    except requests.exceptions.ConnectionError:
        answer = "ERROR: could not connect — is the API server running?"
        status = "connection_error"
    except Exception as exc:
        answer = f"ERROR: {exc}"
        status = "error"

    return {
        "id": query["id"],
        "group": query["group"],
        "message": query["message"],
        "expected_tool": query["expected_tool"],
        "note": query["note"],
        "response": answer,
        "status": status,
    }


def print_result(result: dict, index: int, total: int) -> None:
    """Print a formatted summary of one query result."""
    status_icon = "✓" if result["status"] == "ok" else "✗"
    print(f"\n[{index}/{total}] {status_icon} Query #{result['id']} ({result['group']})")
    print(f"  Q: {result['message'][:80]}{'...' if len(result['message']) > 80 else ''}")
    print(f"  A: {result['response'][:120]}{'...' if len(result['response']) > 120 else ''}")
    if result["status"] != "ok":
        print(f"  ! Status: {result['status']}")


def main() -> None:
    """Run all queries, print results, and save a summary JSON file."""
    total = len(QUERIES)
    print("=" * 65)
    print("  Travel Assistant — Tracing Query Runner")
    print(f"  Sending {total} queries to {API_URL}")
    print("  Traces will appear at http://localhost:6006")
    print("=" * 65)

    # Quick connectivity check before starting
    try:
        requests.get("http://localhost:8000/health", timeout=5).raise_for_status()
    except Exception:
        print("\nERROR: Cannot reach the API server at http://localhost:8000")
        print("Start it with: poetry run uvicorn app.api:app --reload")
        return

    results = []
    for i, query in enumerate(QUERIES, 1):
        result = send_query(query)
        results.append(result)
        print_result(result, i, total)

        # Pause between requests to be polite to DuckDuckGo rate limits
        if i < total:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Summary
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print("\n" + "=" * 65)
    print(f"  Completed: {ok_count}/{total} queries successful")
    print(f"  Groups: web_search={sum(1 for r in results if r['group']=='web_search' and r['status']=='ok')}/3"
          f"  weather={sum(1 for r in results if r['group']=='weather' and r['status']=='ok')}/2"
          f"  multi_tool={sum(1 for r in results if r['group']=='multi_tool' and r['status']=='ok')}/2"
          f"  frustrating={sum(1 for r in results if r['group']=='frustrating' and r['status']=='ok')}/3")
    print("  View all traces at: http://localhost:6006")
    print("=" * 65)

    # Save results to JSON for reference during evaluation phase
    output_path = "scripts/query_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
