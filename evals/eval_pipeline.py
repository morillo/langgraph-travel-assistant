"""
eval_pipeline.py — Unified evaluation pipeline for the travel assistant.

Single command that tells one coherent story in Phoenix:

  Step 1 — Send all 10 queries to the live agent
            → Generates traces in the travel-assistant project

  Step 2 — Run LLM-as-a-judge (GPT-4o-mini) on every response
            → user_frustration + tool_usage_correctness

  Step 3 — Get or create the fixed dataset "travel-assistant-eval"
            → Reuses the same dataset across runs so experiments accumulate
               and you can compare scores across code/prompt changes

  Step 4 — Register a Phoenix Experiment via run_experiment()
            → Uses pre-collected results (no second agent call, no extra LLM calls)
            → Evaluators BOTH return scores AND post annotations to the trace spans
            → Annotations appear in Phoenix Traces tab; scores in Experiments tab

  Step 5 — Log SpanEvaluations via px.Client().log_evaluations()
            → Sends scores in Arrow format to /v1/evaluations
            → Creates named evaluation COLUMNS in the Phoenix Traces table
               (user_frustration and tool_usage_correctness visible without
                clicking into any individual trace)

Result in Phoenix:
  Traces tab          — 10 traces, every one annotated with LLM judge scores
  Datasets tab        — one dataset "travel-assistant-eval", N experiments over time
  Experiments tab     — frustration_eval avg and tool_usage_eval avg per run

Usage:
    # Make sure Phoenix and the API server are running, then:
    poetry install --with evals
    poetry run python evals/eval_pipeline.py
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import nest_asyncio
import pandas as pd
import requests as http_requests
from dotenv import load_dotenv
from openai import OpenAI

nest_asyncio.apply()
load_dotenv()

# ---------------------------------------------------------------------------
# Phoenix client imports (evals group — pinned to 7.0.0, see pyproject.toml)
# ---------------------------------------------------------------------------
import phoenix as px
from phoenix.experiments import run_experiment
from phoenix.experiments.types import Dataset, Example
from phoenix.trace import SpanEvaluations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHOENIX_BASE_URL = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/v1/traces").rstrip("/")
AGENT_URL        = "http://localhost:8000/chat"
PHOENIX_PROJECT  = "travel-assistant"
DATASET_NAME     = "travel-assistant-eval"
JUDGE_MODEL      = "gpt-4o-mini"
SPAN_INDEX_WAIT  = 4   # seconds to wait after queries before fetching spans

os.environ.setdefault("PHOENIX_CLIENT_ENDPOINT", PHOENIX_BASE_URL)

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
px_client     = px.Client(endpoint=PHOENIX_BASE_URL)

# ---------------------------------------------------------------------------
# The 10 evaluation queries (same set used across all pipeline runs)
# ---------------------------------------------------------------------------
QUERIES = [
    {"id": 1,  "group": "web_search",   "message": "What are the top 5 tourist attractions in Rome, Italy?",                                              "expected_tool": "duckduckgo_search"},
    {"id": 2,  "group": "web_search",   "message": "What are the best neighbourhoods to stay in Barcelona for first-time visitors?",                      "expected_tool": "duckduckgo_search"},
    {"id": 3,  "group": "web_search",   "message": "What documents do US citizens need to travel to Japan?",                                              "expected_tool": "duckduckgo_search"},
    {"id": 4,  "group": "weather",      "message": "What is the current weather in Tokyo?",                                                               "expected_tool": "get_current_weather"},
    {"id": 5,  "group": "weather",      "message": "Is it warm enough to visit Reykjavik, Iceland right now?",                                            "expected_tool": "get_current_weather"},
    {"id": 6,  "group": "multi_tool",   "message": "I want to visit Paris next week. What should I pack based on the current weather there?",             "expected_tool": "both"},
    {"id": 7,  "group": "multi_tool",   "message": "What are the must-see attractions in Sydney, and what is the weather like there right now?",          "expected_tool": "both"},
    {"id": 8,  "group": "frustrating",  "message": "Book me a hotel in Dubai for next Friday night under $100.",                                          "expected_tool": "none_or_search"},
    {"id": 9,  "group": "frustrating",  "message": "Find me the cheapest flight from New York to London tomorrow.",                                       "expected_tool": "none_or_search"},
    {"id": 10, "group": "frustrating",  "message": "Tell me literally everything about travelling in Southeast Asia.",                                     "expected_tool": "duckduckgo_search"},
]

# ---------------------------------------------------------------------------
# LLM judge prompt templates
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


# ---------------------------------------------------------------------------
# Pre-collected results — populated in Steps 1-2, read in Step 5 evaluators
# ---------------------------------------------------------------------------
_results: dict[str, dict] = {}   # question → full result dict


# ---------------------------------------------------------------------------
# Step 1 helpers — send queries, collect responses + span IDs
# ---------------------------------------------------------------------------
def _send_query(q: dict) -> dict:
    """POST one query to the agent and return the response."""
    try:
        resp = http_requests.post(AGENT_URL, json={"message": q["message"]}, timeout=60)
        resp.raise_for_status()
        return {**q, "response": resp.json().get("response", ""), "status": "ok"}
    except Exception as exc:
        return {**q, "response": f"[ERROR: {exc}]", "status": "error"}


def _extract_query_from_span_input(raw_input) -> str:
    """Parse a LangChain/LangGraph span input value to extract the user query text."""
    if not raw_input:
        return ""
    if isinstance(raw_input, str) and not raw_input.strip().startswith("{"):
        return raw_input.strip()
    try:
        data = json.loads(raw_input) if isinstance(raw_input, str) else raw_input
    except (json.JSONDecodeError, TypeError):
        return str(raw_input)

    messages = data.get("messages", []) if isinstance(data, dict) else []
    ai_roles = {"ai", "assistant", "aimessage", "function", "tool"}

    def _content(msg: dict) -> str:
        c = (msg.get("data") or {}).get("content", "")
        if not c:
            c = msg.get("content", "")
        if not c:
            c = (msg.get("kwargs") or {}).get("content", "")
        return c if isinstance(c, str) else ""

    for item in messages:
        if isinstance(item, list):
            item = item[0] if item else {}
        if not isinstance(item, dict):
            continue
        role = (item.get("type", "") or item.get("role", "")).lower()
        if role in ai_roles:
            continue
        if role == "constructor":
            if not any("human" in str(i).lower() for i in item.get("id", [])):
                continue
        content = _content(item)
        if content.strip():
            return content.strip()
    return str(raw_input)


def _fetch_span_ids(queries: list[dict], pipeline_start: datetime) -> dict[str, str]:
    """Fetch Phoenix spans and return {question → span_id} for root spans.

    Version-agnostic strategy:
      1. Try text-based matching using the input column (works when column
         names are known and input is parseable).
      2. Fall back to time-based matching: take the N most-recently started
         root spans created AFTER pipeline_start and pair them with queries
         in chronological order. This requires no knowledge of column names
         and works across all Phoenix versions.
    """
    try:
        df = px_client.get_spans_dataframe(project_name=PHOENIX_PROJECT)
        if df is None or df.empty:
            print("  [warn] No spans found in Phoenix yet")
            return {}
    except Exception as exc:
        print(f"  [warn] Could not fetch spans: {exc}")
        return {}

    # Normalise index to span_id
    if "context.span_id" in df.columns and df.index.name != "context.span_id":
        df = df.set_index("context.span_id")

    # Root spans only (no parent)
    if "parent_id" in df.columns:
        df = df[df["parent_id"].isna()]

    if df.empty:
        return {}

    # ── Strategy 1: text-based matching ──────────────────────────────────
    # Try every likely input column name across Phoenix versions
    possible_input_cols = [
        "attributes.input.value",
        "input.value",
        "input",
    ]
    input_col = next((c for c in possible_input_cols if c in df.columns), None)

    if input_col:
        mapping: dict[str, str] = {}
        for span_id, row in df.iterrows():
            raw = row.get(input_col, "") or ""
            query_text = _extract_query_from_span_input(raw)
            if query_text:
                mapping[query_text[:100]] = str(span_id)

        matched: dict[str, str] = {}
        for q in queries:
            key = q["message"][:100]
            if key in mapping:
                matched[q["message"]] = mapping[key]
                continue
            prefix = q["message"][:60]
            for k, v in mapping.items():
                if prefix in k:
                    matched[q["message"]] = v
                    break

        if len(matched) == len(queries):
            return matched   # full match — done

    # ── Strategy 2: time-based matching (version-agnostic fallback) ──────
    # Filter to spans created during this pipeline run, sort by start_time,
    # then pair them with queries in order of submission.
    start_col = next(
        (c for c in ["start_time", "attributes.start_time"] if c in df.columns),
        None,
    )
    if start_col:
        # Convert pipeline_start to UTC-aware if needed
        ps = pipeline_start
        if ps.tzinfo is None:
            ps = ps.replace(tzinfo=timezone.utc)

        recent = df.copy()
        try:
            recent[start_col] = pd.to_datetime(recent[start_col], utc=True)
            recent = recent[recent[start_col] >= ps]
            recent = recent.sort_values(start_col, ascending=True)
        except Exception:
            recent = df  # if time filtering fails, use all root spans

        if len(recent) >= len(queries):
            span_ids = list(recent.index[:len(queries)])
            return {q["message"]: str(sid) for q, sid in zip(queries, span_ids)}

    return {}


# ---------------------------------------------------------------------------
# Step 2 helper — LLM judge
# ---------------------------------------------------------------------------
def _run_judge(template: str, **kwargs) -> dict:
    prompt = template.format(**kwargs)
    try:
        resp = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        return result
    except Exception as exc:
        return {"label": "error", "score": None, "explanation": str(exc)}


# ---------------------------------------------------------------------------
# No inline annotation helper needed — evaluations are logged in bulk via
# SpanEvaluations + px_client.log_evaluations() in Step 5, which is the
# correct pattern for Phoenix ≤ v15 (with a REST fallback for newer servers).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 4 helper — get or create the fixed dataset
# ---------------------------------------------------------------------------
def _get_or_create_dataset() -> Dataset:
    """Return the Phoenix Dataset object for DATASET_NAME.

    If the dataset already exists it is reused, so experiments accumulate
    on the same dataset and can be compared across runs.
    If it does not exist yet it is created from the fixed QUERIES list.
    """
    df = pd.DataFrame([
        {"message": q["message"], "expected_tool": q["expected_tool"], "group": q["group"]}
        for q in QUERIES
    ])

    # ── Check whether the dataset already exists ──────────────────────────
    try:
        list_resp = http_requests.get(f"{PHOENIX_BASE_URL}/v1/datasets", timeout=10)
        if list_resp.ok:
            for ds in list_resp.json().get("data", []):
                if ds.get("name") == DATASET_NAME:
                    dataset_id = ds["id"]
                    print(f"  Dataset '{DATASET_NAME}' already exists — reusing it")

                    # Get the latest version
                    ver_resp = http_requests.get(
                        f"{PHOENIX_BASE_URL}/v1/datasets/{dataset_id}/versions",
                        timeout=10,
                    )
                    versions = ver_resp.json().get("data", []) if ver_resp.ok else []
                    version_id = versions[0]["id"] if versions else "unknown"

                    # Get examples for this version
                    ex_resp = http_requests.get(
                        f"{PHOENIX_BASE_URL}/v1/datasets/{dataset_id}/examples",
                        params={"version_id": version_id},
                        timeout=10,
                    )
                    raw_examples = ex_resp.json().get("data", {}).get("examples", []) if ex_resp.ok else []

                    if raw_examples:
                        examples = {}
                        for ex in raw_examples:
                            example = Example(
                                id=ex["id"],
                                updated_at=datetime.fromisoformat(
                                    ex.get("updated_at", datetime.now(timezone.utc).isoformat())
                                    .replace("Z", "+00:00")
                                ),
                                input=ex.get("input", {}),
                                output=ex.get("output", {}),
                                metadata=ex.get("metadata", {}),
                            )
                            examples[example.id] = example
                        return Dataset(id=dataset_id, version_id=version_id, examples=examples)
    except Exception as exc:
        print(f"  [warn] Dataset lookup failed: {exc} — will create new")

    # ── Create new dataset ────────────────────────────────────────────────
    print(f"  Creating dataset '{DATASET_NAME}'...")
    dataset = px_client.upload_dataset(
        dataframe=df,
        dataset_name=DATASET_NAME,
        input_keys=["message"],
        output_keys=[],
        metadata_keys=["expected_tool", "group"],
    )
    print(f"  Dataset '{DATASET_NAME}' created ✓")
    return dataset


# ---------------------------------------------------------------------------
# Step 5 — run_experiment() task and evaluators (use pre-collected results)
# ---------------------------------------------------------------------------
def _task(example: Example) -> dict:
    """Return pre-collected agent response — no second agent call."""
    question = example.input.get("message", "")
    return _results.get(question, {"message": question, "response": "[not collected]"})


def frustration_eval(output, input) -> float:  # noqa: A002
    """Return pre-computed frustration score.

    Scores are registered with Phoenix as SpanEvaluations in Step 5,
    which creates named evaluation columns in the Traces table.
    """
    question = output.get("message", "") if output else ""
    return float(_results.get(question, {}).get("frustration_score") or 0)


def tool_usage_eval(output, input) -> float:  # noqa: A002
    """Return pre-computed tool usage correctness score.

    Scores are registered with Phoenix as SpanEvaluations in Step 5,
    which creates named evaluation columns in the Traces table.
    """
    question = output.get("message", "") if output else ""
    return float(_results.get(question, {}).get("tool_usage_score") or 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 65)
    print("  Travel Assistant — Evaluation Pipeline")
    print(f"  Started     : {now}")
    print(f"  Agent URL   : {AGENT_URL}")
    print(f"  Phoenix URL : {PHOENIX_BASE_URL}")
    print(f"  Judge model : {JUDGE_MODEL}")
    print(f"  Dataset     : {DATASET_NAME}")
    print("=" * 65)

    # ── Connectivity checks ───────────────────────────────────────────────
    try:
        http_requests.get("http://localhost:8000/health", timeout=5).raise_for_status()
    except Exception:
        print("\nERROR: Cannot reach agent at http://localhost:8000")
        print("Start it with:  poetry run uvicorn app.api:app --reload")
        return

    try:
        http_requests.get(PHOENIX_BASE_URL, timeout=5)
    except Exception:
        print(f"\nERROR: Cannot reach Phoenix at {PHOENIX_BASE_URL}")
        print("Start it with:  docker run -p 6006:6006 arizephoenix/phoenix:latest")
        return

    # Record start time for time-based span matching (version-agnostic)
    pipeline_start = datetime.now(timezone.utc)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1 — Send queries to the agent → generate traces
    # ─────────────────────────────────────────────────────────────────────
    print("\n  Step 1 — Sending 10 queries to the agent...")
    query_responses: list[dict] = []
    for i, q in enumerate(QUERIES, 1):
        print(f"    [{i}/10] {q['message'][:65]}...")
        result = _send_query(q)
        query_responses.append(result)
        if i < len(QUERIES):
            time.sleep(2)   # light rate-limit buffer for DuckDuckGo

    ok = sum(1 for r in query_responses if r["status"] == "ok")
    print(f"  {ok}/10 queries completed ✓")

    # Wait for Phoenix to index the spans
    print(f"\n  Waiting {SPAN_INDEX_WAIT}s for Phoenix to index spans...")
    time.sleep(SPAN_INDEX_WAIT)

    # Fetch span IDs for the traces we just created
    print("  Fetching span IDs from Phoenix...")
    span_ids = _fetch_span_ids(query_responses, pipeline_start)
    print(f"  Matched {len(span_ids)}/10 spans ✓")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2 — Run LLM-as-a-judge on every response
    # ─────────────────────────────────────────────────────────────────────
    print("\n  Step 2 — Running LLM-as-a-judge evaluations...")
    results_list: list[dict] = []

    for q in query_responses:
        if q["status"] != "ok":
            continue

        frustration = _run_judge(
            USER_FRUSTRATION_TEMPLATE,
            query=q["message"],
            response=q["response"],
        )
        tool_usage = _run_judge(
            TOOL_USAGE_TEMPLATE,
            query=q["message"],
            expected_tool=q["expected_tool"],
            response=q["response"],
        )

        result = {
            **q,
            "span_id":                span_ids.get(q["message"], ""),
            "frustration_label":      frustration.get("label", "error"),
            "frustration_score":      frustration.get("score"),
            "frustration_explanation": frustration.get("explanation", ""),
            "tool_usage_label":       tool_usage.get("label", "error"),
            "tool_usage_score":       tool_usage.get("score"),
            "tool_usage_explanation": tool_usage.get("explanation", ""),
        }
        results_list.append(result)
        # Populate the global lookup for Step 5 evaluators
        _results[q["message"]] = result

        print(f"    [{q['id']:2d}] frustration={frustration.get('label'):15s}  "
              f"tool_usage={tool_usage.get('label')}")
        time.sleep(0.5)   # light rate-limit buffer for OpenAI

    # Summary stats
    n_frustrated = sum(1 for r in results_list if r["frustration_label"] == "frustrated")
    n_correct    = sum(1 for r in results_list if r["tool_usage_label"] == "correct")
    print(f"\n  Frustrated : {n_frustrated}/10  (avg score {n_frustrated/10:.2f})")
    print(f"  Tool usage : {n_correct}/10 correct  (avg score {n_correct/10:.2f})")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3a — Upload dataset of frustrated interactions to Phoenix
    #           Builds a curated dataset of low-quality interactions,
    #           visible in Datasets & Experiments, for later prompt tuning
    # ─────────────────────────────────────────────────────────────────────
    frustrated = [r for r in results_list if r["frustration_label"] == "frustrated"]
    if frustrated:
        print(f"\n  Step 3a — Creating frustrated interactions dataset ({len(frustrated)} examples)...")
        try:
            frustrated_df = pd.DataFrame([
                {
                    "message":                r["message"],
                    "response":               r["response"],
                    "group":                  r["group"],
                    "frustration_explanation": r["frustration_explanation"],
                    "tool_usage_label":       r["tool_usage_label"],
                }
                for r in frustrated
            ])
            px_client.upload_dataset(
                dataframe=frustrated_df,
                dataset_name="travel-assistant-frustrated-interactions",
                input_keys=["message"],
                output_keys=["response"],
                metadata_keys=["group", "frustration_explanation", "tool_usage_label"],
            )
            print("  Frustrated interactions dataset uploaded to Phoenix ✓")
            print("  → Visible under Datasets & Experiments → travel-assistant-frustrated-interactions")
        except Exception as exc:
            print(f"  [warn] Frustrated dataset upload failed: {exc}")
    else:
        print("\n  Step 3a — No frustrated interactions found in this run (all users satisfied)")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3b — Get or create the main evaluation dataset
    # ─────────────────────────────────────────────────────────────────────
    print("\n  Step 3b — Preparing main evaluation dataset...")
    dataset = _get_or_create_dataset()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4 — Register Phoenix Experiment
    #          Evaluators score each example AND annotate the trace span
    # ─────────────────────────────────────────────────────────────────────
    print("\n  Step 4 — Registering experiment in Phoenix...")
    print("  (Using pre-collected scores — no second agent call)\n")

    experiment = run_experiment(
        dataset,
        _task,
        evaluators=[frustration_eval, tool_usage_eval],
        experiment_name=f"Travel Assistant Eval — {now}",
        experiment_description=(
            "LLM-as-a-judge evaluation. "
            "user_frustration: 1.0=frustrated, 0.0=satisfied. "
            "tool_usage_correctness: 1.0=correct tool, 0.0=wrong tool. "
            f"Judge: {JUDGE_MODEL}."
        ),
        concurrency=1,
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5 — Attach evaluation results to spans in Phoenix Traces table
    #
    # Tries two approaches in order, handling Phoenix version differences:
    #   A) SpanEvaluations + log_evaluations() — Arrow format via
    #      /v1/evaluations. Works on Phoenix ≤ ~v15. Creates named
    #      evaluation columns visible in the Traces table.
    #   B) span_annotations REST API — JSON format via
    #      /v1/span_annotations. Works on Phoenix ≥ v16. Shows
    #      labels and scores in the Annotations column per trace.
    # ─────────────────────────────────────────────────────────────────────
    print("\n  Step 5 — Attaching evaluations to Phoenix traces...")
    annotated = [r for r in results_list if r.get("span_id")]
    if not annotated:
        print("  [warn] No span IDs available — skipping evaluation upload")
    else:
        frustration_df = pd.DataFrame([
            {
                "context.span_id": r["span_id"],
                "label":           r["frustration_label"],
                "score":           float(r.get("frustration_score") or 0),
                "explanation":     r["frustration_explanation"],
            }
            for r in annotated
        ]).set_index("context.span_id")

        tool_df = pd.DataFrame([
            {
                "context.span_id": r["span_id"],
                "label":           r["tool_usage_label"],
                "score":           float(r.get("tool_usage_score") or 0),
                "explanation":     r["tool_usage_explanation"],
            }
            for r in annotated
        ]).set_index("context.span_id")

        # ── Approach A: SpanEvaluations (Phoenix ≤ v15) ──────────────────
        try:
            px_client.log_evaluations(
                SpanEvaluations(eval_name="user_frustration",       dataframe=frustration_df),
                SpanEvaluations(eval_name="tool_usage_correctness", dataframe=tool_df),
            )
            print("  SpanEvaluations logged — named columns visible in Traces table ✓")
        except Exception:
            # ── Approach B: span_annotations REST API (Phoenix ≥ v16) ────
            try:
                annotations = []
                for r in annotated:
                    annotations += [
                        {
                            "span_id": r["span_id"],
                            "name": "user_frustration",
                            "annotator_kind": "LLM",
                            "result": {
                                "label":       r["frustration_label"],
                                "score":       float(r.get("frustration_score") or 0),
                                "explanation": r["frustration_explanation"],
                            },
                        },
                        {
                            "span_id": r["span_id"],
                            "name": "tool_usage_correctness",
                            "annotator_kind": "LLM",
                            "result": {
                                "label":       r["tool_usage_label"],
                                "score":       float(r.get("tool_usage_score") or 0),
                                "explanation": r["tool_usage_explanation"],
                            },
                        },
                    ]
                http_requests.post(
                    f"{PHOENIX_BASE_URL}/v1/span_annotations",
                    json={"data": annotations},
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                ).raise_for_status()
                print("  Span annotations uploaded — visible in Traces → Annotations tab ✓")
            except Exception as exc2:
                print(f"  [warn] Both evaluation upload methods failed: {exc2}")

    # ─────────────────────────────────────────────────────────────────────
    # Done
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Traces + annotations : {PHOENIX_BASE_URL}/projects")
    print(f"  Dataset + experiment : {PHOENIX_BASE_URL}/datasets")
    print("=" * 65)

    return experiment


if __name__ == "__main__":
    main()
