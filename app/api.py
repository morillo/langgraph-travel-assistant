"""
api.py — FastAPI server for the travel assistant agent.

Observability:
  Traces are exported to Phoenix (open-source LLM observability) via standard
  OpenTelemetry OTLP/HTTP. We configure the OTel TracerProvider directly using
  the OTel SDK — no vendor-specific client is imported here, so the app only
  depends on open standards and any OTLP-compatible backend works.

  Instrumentation flow:
    1. OTLPSpanExporter        — ships spans to Phoenix at localhost:6006
    2. BatchSpanProcessor      — buffers and batches spans before export
    3. TracerProvider          — global OTel provider wrapping the exporter
    4. LangChainInstrumentor   — patches LangChain/LangGraph to emit spans
                                 for every LLM call, tool call, and graph node

  Phoenix must be running before this server starts:
    docker run -p 6006:6006 arizephoenix/phoenix:latest

  Phoenix UI: http://localhost:6006

Endpoints:
  POST /chat   — send a message to the travel assistant
  GET  /health — liveness check
"""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel

from app.agent import build_agent

load_dotenv()
logger = logging.getLogger(__name__)

# PHOENIX_COLLECTOR_ENDPOINT lets callers override the Phoenix host.
# - Local dev (Poetry):  defaults to http://localhost:6006/v1/traces
# - Docker Compose:      set to http://phoenix:6006/v1/traces (service name)
_phoenix_base = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
PHOENIX_ENDPOINT = f"{_phoenix_base.rstrip('/')}/v1/traces"
PROJECT_NAME = "travel-assistant"


# ---------------------------------------------------------------------------
# Lifespan — OTel setup on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure OpenTelemetry tracing on startup.

    We set up the OTel TracerProvider manually using the standard SDK so the
    app server has no dependency on any vendor-specific client library.
    The TracerProvider exports spans to Phoenix over OTLP/HTTP.

    LangChainInstrumentor patches LangChain/LangGraph internals so all LLM
    calls, tool invocations, and graph node executions are traced automatically.
    """
    try:
        # 1. Create an OTLP exporter pointing at the Phoenix container
        exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT)

        # 2. Wrap it in a BatchSpanProcessor (async, non-blocking export)
        processor = BatchSpanProcessor(exporter)

        # 3. Build the TracerProvider with service + Phoenix project metadata.
        #    Phoenix reads "openinference.project.name" to assign traces to a project.
        #    It also respects the PHOENIX_PROJECT_NAME environment variable.
        resource = Resource(attributes={
            "service.name": PROJECT_NAME,
            "openinference.project.name": PROJECT_NAME,
        })
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(processor)

        # 4. Register as the global OTel provider
        trace.set_tracer_provider(provider)

        # 5. Auto-instrument LangChain and LangGraph
        LangChainInstrumentor().instrument(tracer_provider=provider)

        logger.info(f"OpenTelemetry tracing enabled — sending spans to {PHOENIX_ENDPOINT}")
        logger.info(f"View traces at http://localhost:6006 (project: {PROJECT_NAME})")

    except Exception as exc:
        # If Phoenix isn't running the app still works — traces are lost but
        # requests are served normally. Useful for local dev without Docker.
        logger.warning(f"OTel tracing could not be configured: {exc}")
        logger.warning("Start Phoenix with: docker run -p 6006:6006 arizephoenix/phoenix:latest")

    yield  # Application runs here


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
agent = build_agent()

app = FastAPI(
    title="Travel Assistant API",
    description=(
        "LangGraph-powered travel assistant with OpenTelemetry observability. "
        "Start Phoenix first: docker run -p 6006:6006 arizephoenix/phoenix:latest"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the travel assistant and receive a response.

    The agent decides autonomously whether to call tools (web search,
    weather lookup) before producing its final answer. Every interaction
    is captured as a trace in Phoenix.
    """
    result = agent.invoke({"messages": [HumanMessage(content=request.message)]})
    return ChatResponse(response=result["messages"][-1].content)


@app.get("/health")
def health() -> dict:
    """Liveness check — returns 200 OK when the service is running."""
    return {"status": "ok"}
