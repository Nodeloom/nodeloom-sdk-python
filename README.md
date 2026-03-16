# NodeLoom Python SDK

Python SDK for instrumenting AI agents and sending telemetry to [NodeLoom](https://nodeloom.io).

## Features

- Fire-and-forget telemetry that never blocks or crashes your application
- Automatic batching and retry with exponential backoff
- Context manager support for traces and spans
- Built-in integrations for LangChain and CrewAI
- Thread-safe client (individual traces/spans are single-threaded)
- Bounded in-memory queue prevents unbounded memory growth
- Configurable via constructor arguments

## Requirements

- Python 3.9+

## Installation

```bash
pip install nodeloom-sdk
```

With LangChain integration:

```bash
pip install nodeloom-sdk[langchain]
```

With CrewAI integration:

```bash
pip install nodeloom-sdk[crewai]
```

## Quick Start

```python
from nodeloom import NodeLoom, SpanType

client = NodeLoom(api_key="sdk_your_api_key")

with client.trace("my-agent", input={"query": "What is NodeLoom?"}) as trace:
    with trace.span("llm-call", type=SpanType.LLM) as span:
        span.set_input({"messages": [{"role": "user", "content": "What is NodeLoom?"}]})
        # ... call your LLM ...
        span.set_output({"text": "NodeLoom is an AI agent operations platform."})
        span.set_token_usage(prompt=15, completion=20, model="gpt-4o")

client.shutdown()
```

## Traces and Spans

A **trace** represents a single end-to-end agent execution. A **span** represents a unit of work within a trace (an LLM call, tool invocation, retrieval step, etc.).

### Span Types

| Type | Description |
|------|-------------|
| `SpanType.LLM` | Language model call |
| `SpanType.TOOL` | Tool or function invocation |
| `SpanType.RETRIEVAL` | Vector search or data retrieval |
| `SpanType.CHAIN` | Pipeline or chain of steps |
| `SpanType.AGENT` | Sub-agent invocation |
| `SpanType.CUSTOM` | User-defined operation |

### Nested Spans

```python
with client.trace("my-agent") as trace:
    with trace.span("agent-step", type=SpanType.AGENT) as parent:
        with trace.span("llm-call", type=SpanType.LLM, parent_span_id=parent.span_id) as child:
            child.set_output({"response": "..."})
            child.set_token_usage(prompt=10, completion=20, model="gpt-4o")
```

### Standalone Events

```python
client.event("guardrail_triggered", level=EventLevel.WARN, data={"rule": "pii_detected"})
```

### Error Handling

Traces and spans used as context managers automatically catch exceptions, mark the span/trace as `ERROR`, and re-raise:

```python
with client.trace("my-agent") as trace:
    with trace.span("risky-call", type=SpanType.TOOL) as span:
        raise ValueError("something went wrong")
        # span is automatically marked as ERROR
    # trace is automatically marked as ERROR
```

You can also set errors manually:

```python
span.set_error("Connection timeout")
trace.end(status=TraceStatus.ERROR, error="Agent failed")
```

## LangChain Integration

```python
from nodeloom import NodeLoom
from nodeloom.integrations.langchain import NodeLoomCallbackHandler

client = NodeLoom(api_key="sdk_your_api_key")
handler = NodeLoomCallbackHandler(client)

# Pass the handler to any LangChain chain, agent, or LLM
result = chain.invoke(input, config={"callbacks": [handler]})

client.shutdown()
```

The callback handler automatically instruments LLM calls, chain runs, tool invocations, and retriever queries with proper parent-child span relationships.

## CrewAI Integration

### Decorator

```python
from nodeloom import NodeLoom
from nodeloom.integrations.crewai import instrument_crew

client = NodeLoom(api_key="sdk_your_api_key")

@instrument_crew(client, agent_name="my-crew", agent_version="1.0.0")
def run_crew():
    crew = Crew(agents=[...], tasks=[...])
    return crew.kickoff()

run_crew()
client.shutdown()
```

### Manual Instrumentation

```python
from nodeloom.integrations.crewai import CrewAIInstrumentation

inst = CrewAIInstrumentation(client)
with inst.trace_crew("my-crew") as ctx:
    with ctx.task("research", agent="researcher") as span:
        result = do_research()
        span.set_output({"result": result})
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | *required* | SDK API key (starts with `sdk_`) |
| `endpoint` | `https://api.nodeloom.io` | NodeLoom API base URL |
| `environment` | `production` | Deployment environment label |
| `batch_size` | `100` | Max events per batch |
| `flush_interval` | `5.0` | Seconds between automatic flushes |
| `max_retries` | `3` | Retry attempts for failed requests |
| `queue_max_size` | `10000` | Max queued events before dropping |
| `timeout` | `10.0` | HTTP request timeout in seconds |
| `enabled` | `True` | Set to `False` to disable telemetry |

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
