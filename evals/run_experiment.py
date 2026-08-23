"""
run_experiment.py — Phoenix Experiment runner for the travel assistant.

This script uses Phoenix's dataset + experiment API:

    dataset  = px_client.upload_dataset(...)
    experiment = run_experiment(
        dataset,
        task_fn,
        evaluators=[eval1, eval2, ...],
        experiment_name="...",
        experiment_description="...",
    )

Workflow
--------
1. Load the 10 evaluation queries from scripts/query_results.json.
2. Upload them as a Phoenix Dataset (or reuse the existing one).
3. Define a task function that POSTs each question to the live agent API.
4. Define two LLM-as-a-judge evaluators:
     - frustration_eval  → float 0.0 (not frustrated) / 1.0 (frustrated)
     - tool_usage_eval   → float 0.0 (incorrect)     / 1.0 (correct)
5. Call run_experiment() — Phoenix runs each example through the task and
   every evaluator, then stores the results as a named Experiment linked
   to the dataset (visible under Datasets & Experiments in the UI).

Usage
-----
    # Make sure Phoenix + the API server are both running, then:
    poetry run python evals/run_experiment.py --with evals

Prerequisites
-------------
    Docker: docker run -p 6006:6006 arizephoenix/phoenix:latest
    API:    poetry run uvicorn app.api:app --reload --port 8000
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import nest_asyncio
import pandas as pd
import requests as http_requests
from dotenv import load_dotenv
from openai import OpenAI

nest_asyncio.apply()
load_dotenv()

# ---------------------------------------------------------------------------
# Phoenix imports
# ---------------------------------------------------------------------------
import phoenix as px
from phoenix.experiments import run_experiment
from phoenix.experiments.types import Example

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHOENIX_BASE_URL = "http://localhost:6006"
AGENT_URL = "http://localhost:8000/chat"
QUERY_RESULTS_PATH = Path("scripts/query_results.json")
DATASET_NAME = "travel-assistant-eval-v1"
JUDGE_MODEL = "gpt-4o-mini"

os.environ.setdefault("PHOENIX_CLIENT_ENDPOINT", PHOENIX_BASE_URL)

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
px_client = px.Client(endpoint=PHOENIX_BASE_URL)

# ---------------------------------------------------------------------------
# Prompt templates (same as evaluate.py)
# ---------------------------------------------------------------------------
USER_FRUSTRATION_TEMPLATE = """You are evaluating a travel assistant chatbot interaction.

User query: {query}
Assistant response: {response}

Determine whether the user would likely feel frustrated after receiving this response.

A user is FRUSTRATED if:
- The assistant could not fulfil the user's primary request (e.g. booking, pricing)
- The response is overly vague, incomplete, or unhelpful
- The assistant gave a generic answer when a specific one was needed

A user is NOT FRUSTRATED if:
- The assistant directly and accurately answered the question
- The assistant clearly explained what it cannot do AND provided helpful alternatives
- The response is specific, relevant, and useful

Respond with a JSON object:
{{
  "label": "frustrated" or "not frustrated",
  "score": 1.0 if frustrated, 0.0 if not frustrated,
  "explanation": "one sentence explaining your judgment"
}}"""

TOOL_USAGE_TEMPLATE = """You are evaluating whether an AI travel assistant used the correct tools.

Available tools:
- duckduckgo_search: for general travel information (attractions, hotels, visas, etc.)
- get_current_weather: ONLY for current weather or temperature conditions
- (no tool): for booking or flight pricing — agent should decline gracefully

User query: {query}
Expected tool: {expected_tool}
Assistant response: {response}

Respond with a JSON object:
{{
  "label": "correct" or "incorrect",
  "score": 1.0 if correct, 0.0 if incorrect,
  "explanation": "one sentence explaining your judgment"
}}"""


def run_judge(template: str, **kwargs) -> dict:
    """Run a single LLM-as-a-judge call and return parsed JSON result."""
    prompt = template.format(**kwargs)
    try:
        resp = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        return {"label": "error", "score": None, "explanation": str(exc)}


# ---------------------------------------------------------------------------
# Step 1 — Upload dataset to Phoenix
# ---------------------------------------------------------------------------
def upload_dataset() -> px.experiments.types.Dataset:
    """Upload the 10 evaluation queries as a Phoenix Dataset.

    Returns the Dataset object required by run_experiment().
    The dataset is named with a timestamp so each run creates a fresh version.
    """
    with open(QUERY_RESULTS_PATH) as f:
        queries = [q for q in json.load(f) if q["status"] == "ok"]

    df = pd.DataFrame([
        {
            "message": q["message"],
            "expected_tool": q["expected_tool"],
            "group": q["group"],
        }
        for q in queries
    ])

    versioned_name = f"{DATASET_NAME}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"  Uploading dataset '{versioned_name}' ({len(df)} examples)...")

    dataset = px_client.upload_dataset(
        dataframe=df,
        dataset_name=versioned_name,
        input_keys=["message"],
        output_keys=[],
        metadata_keys=["expected_tool", "group"],
    )
    print(f"  Dataset ready: {versioned_name} ✓")
    return dataset


# ---------------------------------------------------------------------------
# Step 2 — Task function
# ---------------------------------------------------------------------------
def run_agent_task(example: Example) -> dict:
    """Task: send one query to the live travel assistant API.

    run_experiment() calls this once per dataset example.
    Returns a dict with the agent's response for the evaluators to consume.
    """
    question = example.input.get("message", "")
    print(f"    → Running agent: {question[:60]}...")
    try:
        resp = http_requests.post(
            AGENT_URL,
            json={"message": question},
            timeout=60,
        )
        resp.raise_for_status()
        response_text = resp.json().get("response", "")
    except Exception as exc:
        response_text = f"[ERROR: {exc}]"

    time.sleep(0.5)  # light rate-limit buffer
    return {
        "message": question,
        "response": response_text,
    }


# ---------------------------------------------------------------------------
# Step 3 — Evaluators
# ---------------------------------------------------------------------------
def frustration_eval(output, input) -> float:  # noqa: A002
    """LLM-as-a-judge: returns 1.0 if user is frustrated, 0.0 otherwise.

    Signature matches the run_experiment evaluator contract:
        evaluator(output, input) -> numeric score
    """
    if output is None:
        return 0.0
    result = run_judge(
        USER_FRUSTRATION_TEMPLATE,
        query=output.get("message", ""),
        response=output.get("response", ""),
    )
    score = result.get("score")
    return float(score) if score is not None else 0.0


def tool_usage_eval(output, input, expected) -> float:  # noqa: A002
    """LLM-as-a-judge: returns 1.0 if correct tool was used, 0.0 otherwise.

    Signature matches the run_experiment evaluator contract:
        evaluator(output, input, expected) -> numeric score
    """
    if output is None:
        return 0.0
    result = run_judge(
        TOOL_USAGE_TEMPLATE,
        query=output.get("message", ""),
        expected_tool=input.get("expected_tool", "any"),
        response=output.get("response", ""),
    )
    score = result.get("score")
    return float(score) if score is not None else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 65)
    print("  Travel Assistant — Phoenix Experiment Runner")
    print(f"  Started     : {now}")
    print(f"  Agent URL   : {AGENT_URL}")
    print(f"  Phoenix URL : {PHOENIX_BASE_URL}")
    print(f"  Judge model : {JUDGE_MODEL}")
    print("=" * 65)

    # ── Dataset ───────────────────────────────────────────────────────────
    print("\n  Step 1 — Uploading dataset to Phoenix...")
    dataset = upload_dataset()

    # ── Experiment ────────────────────────────────────────────────────────
    print("\n  Step 2 — Running experiment (task + evaluators)...")
    print("  Each example will be sent to the live agent, then judged.\n")

    experiment = run_experiment(
        dataset,
        run_agent_task,
        evaluators=[frustration_eval, tool_usage_eval],
        experiment_name=f"Travel Assistant Eval — {now}",
        experiment_description=(
            "LLM-as-a-judge evaluation of the travel assistant agent. "
            "Two evaluators: user_frustration (1=frustrated, 0=satisfied) "
            "and tool_usage_correctness (1=correct tool, 0=wrong tool). "
            "Judge model: gpt-4o-mini."
        ),
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EXPERIMENT COMPLETE")
    print("=" * 65)
    print(f"  View results at {PHOENIX_BASE_URL}")
    print("  Navigate to: Datasets & Experiments → your dataset → Experiments tab")
    print("=" * 65)

    return experiment


if __name__ == "__main__":
    main()
