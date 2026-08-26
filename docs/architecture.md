# Production Architecture — Travel Assistant

**Author:** Carlos Morillo  
**Version:** 1.0  
**Date:** May 2026

---

## Overview

This document describes a production-grade deployment of the Travel Assistant: a LangGraph-based AI agent with OpenTelemetry-based observability, using the open-source Phoenix platform as the trace backend. The current implementation is a working prototype (FastAPI + LangGraph + Phoenix running locally via Docker). This document explains how that prototype would be hardened and scaled for production traffic.

The architecture is organized into five concerns: deployment infrastructure, observability and evaluation, scaling and reliability, CI/CD and MLOps, and latency/cost optimization.

---

## System Diagram

```
                         ┌─────────────────────────────────────────────┐
                         │               CLIENT LAYER                   │
                         │   Web / Mobile / Partner API consumers       │
                         └──────────────────┬──────────────────────────┘
                                            │ HTTPS
                         ┌──────────────────▼──────────────────────────┐
                         │           API GATEWAY / CDN                  │
                         │   (AWS API Gateway + CloudFront)             │
                         │   Rate limiting · Auth (JWT/API keys)        │
                         │   Request routing · DDoS protection          │
                         └──────────────────┬──────────────────────────┘
                                            │
                         ┌──────────────────▼──────────────────────────┐
                         │         KUBERNETES CLUSTER (EKS)             │
                         │                                              │
                         │  ┌──────────────────────────────────────┐   │
                         │  │   Travel Assistant Service (pods ×N)  │   │
                         │  │   FastAPI + LangGraph agent           │   │
                         │  │   HPA: scale on CPU / request queue   │   │
                         │  └──────────────┬───────────────────────┘   │
                         │                 │ OTel traces (OTLP/HTTP)    │
                         │  ┌──────────────▼───────────────────────┐   │
                         │  │   Phoenix (observability, OSS)        │   │
                         │  │   Traces · Annotations · Experiments  │   │
                         │  │   Dashboards · Alerts · Eval loops    │   │
                         │  └──────────────────────────────────────┘   │
                         │                                              │
                         └──────────────────────────────────────────────┘
                                            │
               ┌────────────────────────────┼──────────────────────────┐
               │                            │                           │
  ┌────────────▼──────┐      ┌──────────────▼──────┐     ┌─────────────▼──────┐
  │  External APIs     │      │   LLM Provider       │     │  Vector / Cache    │
  │  DuckDuckGo Search │      │   OpenAI GPT-4o      │     │  Redis (sessions)  │
  │  Weather API       │      │   (or Azure OAI)     │     │  Pinecone (RAG)    │
  └───────────────────┘      └─────────────────────┘     └────────────────────┘
```

---

## 1. Deployment & Infrastructure

### Containerization

The application is already containerized (see `Dockerfile` and `docker-compose.yml`). In production, the image is built once per release and promoted through environments (dev → staging → prod) without rebuilding.

```
Dockerfile (multi-stage)
  Stage 1 — builder : install poetry deps, compile requirements
  Stage 2 — runtime : copy only the virtualenv, no build tools
  Result           : lean ~180 MB image, no dev dependencies at runtime
```

### Kubernetes (EKS / GKE / AKS)

Each service runs as a Kubernetes Deployment with resource requests and limits tuned to measured p50/p99 latencies:

| Component | Replicas (baseline) | CPU request | Memory request |
|---|---|---|---|
| Travel Assistant API | 3 | 250m | 512Mi |
| Phoenix | 2 | 500m | 1Gi |
| Redis | 1 (+ replica) | 100m | 256Mi |

**Ingress:** NGINX Ingress Controller with TLS termination. Certificates managed via cert-manager + Let's Encrypt (or ACM on AWS).

**Config & secrets:** Kubernetes Secrets (backed by AWS Secrets Manager or HashiCorp Vault). `OPENAI_API_KEY`, `PHOENIX_API_KEY`, weather API credentials — never baked into images.

### Environments

Three environments share the same Helm chart, differing only in `values.yaml` overrides:

- **dev** — single replica, debug logging, Phoenix dev instance, no rate limiting
- **staging** — mirrors prod sizing, runs the full eval suite before promotion
- **prod** — autoscaling enabled, Phoenix prod instance, PagerDuty alerts active

---

## 2. Observability & Evaluation

Phoenix is the central observability layer. Every agent invocation emits OpenTelemetry spans that Phoenix ingests, stores, and surfaces as traces, dashboards, and evaluation results. Because the app emits standard OTel/OpenInference spans, the backend is swappable for any OTLP-compatible collector.

### Tracing

The application uses `openinference-instrumentation-langchain` to auto-instrument the LangGraph agent. Every trace captures:

- **LLM spans** — model, prompt template, token counts (input/output), latency
- **Tool spans** — which tool was called (`duckduckgo_search` vs `get_current_weather`), arguments, response
- **Chain spans** — the full agent invocation with input message and final response

All spans carry `session.id` and `user.id` attributes for cross-session analysis. In production, `session.id` is derived from the authenticated user's JWT to enable per-user trace filtering.

### Dashboards

Phoenix dashboards track the metrics that matter most for an AI agent in production:

| Metric | Alert threshold |
|---|---|
| P50 / P95 / P99 latency | P95 > 8s → PagerDuty |
| Token usage per request | Spike > 3× rolling avg → Slack |
| Error rate (tool failures) | > 5% over 5 min window |
| Frustration score (rolling avg) | > 0.3 over 1-hour window |
| Tool usage correctness | < 0.9 → immediate alert |

### Continuous Evaluation Loop

The eval pipeline (`evals/evaluate.py`, `evals/run_experiment.py`) runs automatically as part of every deployment:

```
New code merged to main
       │
       ▼
CI pipeline runs staging deploy
       │
       ▼
run_experiment.py fires against staging agent
  ├── frustration_eval    (target: avg ≤ 0.2)
  └── tool_usage_eval     (target: avg = 1.0)
       │
  scores meet thresholds?
  ├── YES → promote to production
  └── NO  → block deploy, open GitHub issue with eval report
```

Phoenix stores each experiment as a versioned record linked to the dataset, so score trends across releases are immediately visible in the Experiments tab.

### Annotation-Driven Quality Loop

Beyond automated evals, Phoenix annotations support a human-in-the-loop quality process:

1. Phoenix flags spans where `frustration_eval` score = 1.0 (predicted frustrated user)
2. A human reviewer opens the trace in Phoenix, reads the full conversation, and adds a corrected label
3. Corrected labels accumulate into a golden dataset for the next fine-tuning run or prompt revision

---

## 3. Scaling & Reliability

### Horizontal Pod Autoscaling

The Travel Assistant Deployment is governed by an HPA that scales on two signals:

```yaml
metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: External
    external:
      metric:
        name: request_queue_depth   # sourced from the API Gateway metrics
      target:
        type: AverageValue
        averageValue: "10"
minReplicas: 3
maxReplicas: 20
```

Scale-up is aggressive (30s stabilization window); scale-down is conservative (5 min) to avoid thrashing under bursty traffic.

### Reliability Patterns

**Retries with backoff:** The agent retries transient OpenAI errors (429, 503) with exponential backoff up to 3 attempts. Tool calls (DuckDuckGo, weather) have a 5-second timeout and a single retry.

**Circuit breaker:** If the weather API fails > 50% of requests in a 60-second window, the circuit opens and the agent falls back to a canned response ("Current weather is unavailable — here is what I know about the destination.") rather than surfacing a raw error to the user.

**Graceful degradation:** If the DuckDuckGo search tool is unavailable, the agent continues to answer from its parametric knowledge with an explicit disclaimer. No tool failure should result in a 500 error reaching the client.

**Health checks:**
- `GET /health` — liveness probe (responds 200 if the process is alive)
- `GET /ready` — readiness probe (responds 200 only after the LangGraph graph is compiled and the OTel exporter has connected to Phoenix)

### Session Persistence

Conversation state is stored in Redis with a 30-minute TTL. This allows any pod in the deployment to continue a conversation started on a different pod, enabling seamless rolling deploys without losing user context.

---

## 4. CI/CD & MLOps

### Pipeline (GitHub Actions)

```
push / PR to main
       │
       ├── lint & type-check (ruff, mypy)
       │
       ├── unit tests (pytest tests/)
       │
       ├── build Docker image
       │
       ├── push to ECR (tagged with git SHA)
       │
       ├── deploy to staging (Helm upgrade)
       │
       ├── run eval suite against staging
       │   └── frustration_eval avg ≤ 0.2
       │   └── tool_usage_eval avg = 1.0
       │
       └── promote to production (Helm upgrade --atomic)
           └── rollback automatically on failed health checks
```

### Prompt Version Control

In production, prompt templates move out of Python and into version-controlled files (`app/prompts/`) — in the current prototype the system prompt lives in `app/agent.py`. Each prompt change triggers a full eval run. Phoenix's Experiments tab provides a side-by-side score comparison between the old and new prompt, making regressions immediately visible before production promotion.

### Model Upgrade Path

When upgrading the underlying LLM (e.g., GPT-4o → GPT-4o-mini for cost, or a new model release):

1. Add the new model as a configuration option
2. Run the eval suite with both models against the same dataset
3. Compare `frustration_eval` and `tool_usage_eval` scores in Phoenix Experiments
4. Promote the new model only if scores are equivalent or better

This gives a reproducible, data-backed justification for any model change.

---

## 5. Latency Optimization & Cost

### Latency Budget

Target end-to-end P95: **< 6 seconds**

| Component | Typical latency | Optimization lever |
|---|---|---|
| API Gateway + network | ~50ms | CloudFront edge caching for static responses |
| LangGraph overhead | ~20ms | Compiled graph cached at startup |
| Tool call (DuckDuckGo) | 300–800ms | Parallel tool dispatch; 5s timeout |
| LLM call (GPT-4o) | 1–4s | Streaming responses; smaller models for simple queries |
| Phoenix trace export | async, ~0ms on hot path | OTLP exporter runs in background thread |

### Streaming Responses

The FastAPI endpoint supports SSE (Server-Sent Events) for streaming LLM tokens to the client. Users see the first token within ~500ms rather than waiting for the full response, dramatically improving perceived latency.

### Semantic Caching

A Redis-backed semantic cache stores embeddings of recent queries alongside their responses. Cache hits (cosine similarity > 0.92) bypass the LLM entirely — saving both latency and token cost. Phoenix traces cache hits as a distinct span kind for hit-rate monitoring.

### Cost Controls

| Lever | Implementation |
|---|---|
| Model routing | Simple queries (weather, single-fact lookups) → GPT-4o-mini; complex multi-hop → GPT-4o |
| Max token limits | `max_tokens=1024` on all completions; tool responses truncated to 2,000 chars |
| Rate limiting | Per-user token budget enforced at API Gateway (10k tokens / min) |
| Batch evaluation | Eval suite runs once per deploy, not on every request |

Monthly cost estimate for 100k requests/month at current tool mix: approximately **$180–240** (LLM tokens) + **$60** (infrastructure) = ~**$240–300/month**.

---

## Summary

The travel assistant is designed to move from prototype to production without architectural rework — the same LangGraph agent, FastAPI server, and Phoenix observability stack scale horizontally behind Kubernetes. The key production additions are: autoscaling, Redis session state, circuit-breaker tool fallbacks, a streaming endpoint, semantic caching, and a CI/CD gate that blocks deploys when eval scores regress. OpenTelemetry tracing with Phoenix is the connective tissue throughout — providing the trace data, evaluation results, and dashboards that make every layer of this system observable and improvable.
